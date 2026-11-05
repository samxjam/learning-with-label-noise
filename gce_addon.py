"""Independent GCE experiment; never invokes the original training suite.

Requires the algorithm package beside this file. Zhang & Sabuncu,
NeurIPS 2018, Eq. 6: L_q=(1-p_s**q)/q. Plain GCE, not truncated GCE.
"""
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse
import csv
import json
from pathlib import Path
import time

import numpy as np
import torch
import torch.nn.functional as F

from algorithm.artifacts import atomic_checkpoint, atomic_json, environment, identity, source_identity, statistics
from algorithm.data import DATASETS, load_data, prepare_split, file_hash
from algorithm.engine import seed_everything, cpu_copy, predict, accuracy
from algorithm.main import choose_device, positive_int
from algorithm.models import CNN

ROOT = Path(__file__).resolve().parent
PRIMARY = {"fashion0.3": "fashion0.3_given_trimmed", "fashion0.6": "fashion0.6_given_trimmed",
           "cifar": "cifar_estimated_trimmed"}


def gce_loss(logits, labels, q=.7):
    """Stable plain GCE; q is fixed before observing GCE test results."""
    if not np.isfinite(q) or not 0 < q <= 1:
        raise ValueError("GCE q must lie in (0,1]")
    log_probability = logits.log_softmax(1).gather(1, labels[:, None]).squeeze(1)
    return (-torch.expm1(q * log_probability) / q).mean()


def addon_identity():
    """Hash this file together with the core package source."""
    return {"gce_addon.py": file_hash(__file__), "core": source_identity()}


def load_references(directory):
    """Read completed baseline JSON only; weights are neither copied nor retrained."""
    if directory is None:
        return {}
    directory = Path(directory)
    documents = {}
    for path in sorted(directory.glob("*/results.json")):
        doc = json.loads(path.read_text(encoding="utf-8"))
        manifest = json.loads((path.parent / "manifest.json").read_text(encoding="utf-8"))
        config = doc["config"]
        if not doc["complete"] or not doc["reportable"] or len(doc["runs"]) < 10 or config["smoke"]:
            raise ValueError(f"Baseline reference is incomplete or a smoke run: {path}")
        if doc["signature"] != manifest["signature"] or config != manifest["definition"]["config"]:
            raise ValueError(f"Baseline manifest mismatch: {path}")
        if manifest["definition"]["source"] != source_identity():
            raise ValueError("Baseline core source differs from the supplied v3 package")
        documents[path.parent.name] = {"results": doc, "manifest": manifest, "path": path.parent}
    for name in PRIMARY.values():
        if name not in documents:
            raise ValueError(f"Missing primary baseline reference: {name}")
    return documents


def check_reference(data, split, config, reference):
    """Require matching data, budgets and split so GCE pairs properly with a baseline."""
    baseline = reference["results"]
    old = baseline["config"]
    for key in ("epochs", "batch_size", "lr", "weight_decay", "flip"):
        if config[key] != old[key]:
            raise ValueError(f"GCE/base budget or setup mismatch: {key}")
    if file_hash(data.path) != reference["manifest"]["definition"]["data_sha256"]:
        raise ValueError("GCE dataset differs from completed baseline data")
    saved = json.loads((reference["path"] / f"run_{split['seed']:03d}" / "split.json").read_text())
    if split != saved:
        raise ValueError("GCE split or training-only normalisation differs from the baseline")


def fit_gce(model, tensors, *, epochs, seed, batch_size, lr, weight_decay, flip, q,
            progress_path, signature, stop_after=None):
    """Resume from the last saved epoch, including Adam and all used RNG streams.

    Checkpoint selection uses noisy-validation GCE fixed in advance. Noisy CE
    is logged as a diagnostic. This function accepts no test-label argument
    through its training/validation view (caller supplies only those four keys).
    """
    if epochs < 1 or batch_size < 1 or lr <= 0 or weight_decay < 0:
        raise ValueError("Invalid training configuration")
    if set(tensors) != {"x_train", "s_train", "x_validation", "s_validation"}:
        raise ValueError("fit_gce accepts training and validation tensors only")
    seed_everything(20000 + seed)
    generator = torch.Generator().manual_seed(20000 + seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    history, best, completed, elapsed_before = [], None, 0, 0.
    progress_path = Path(progress_path)
    if progress_path.is_file():
        saved = torch.load(progress_path, map_location="cpu", weights_only=True)
        if saved["signature"] != signature:
            raise ValueError("Epoch checkpoint identity mismatch")
        model.load_state_dict(saved["latest_model"])
        optimizer.load_state_dict(saved["optimizer"])
        generator.set_state(saved["batch_rng"])
        torch.set_rng_state(saved["torch_rng"])
        if saved["cuda_rng"] is not None:
            torch.cuda.set_rng_state_all(saved["cuda_rng"])
        completed, history, best = saved["completed_epoch"], saved["history"], saved["best"]
        elapsed_before = saved["seconds"]
        print(f"  restored epoch {completed}/{epochs}", flush=True)
    start = time.perf_counter()
    x_train, labels = tensors["x_train"], tensors["s_train"]
    for epoch in range(completed + 1, epochs + 1):
        model.train()
        order = torch.randperm(len(x_train), generator=generator).to(x_train.device)
        total = torch.zeros((), device=x_train.device)
        for offset in range(0, len(order), batch_size):
            ids = order[offset:offset + batch_size]
            images, noisy = x_train[ids], labels[ids]
            if flip:
                mask = torch.rand(len(images), device=images.device) < .5
                images[mask] = images[mask].flip(3)
            loss = gce_loss(model(images), noisy, q)
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite GCE training loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float("inf"), error_if_nonfinite=True)
            optimizer.step()
            total += loss.detach() * len(ids)
        with torch.no_grad():
            logits = predict(model, tensors["x_validation"])
            val = float(gce_loss(logits, tensors["s_validation"], q).item())
            noisy_ce = float(F.cross_entropy(logits, tensors["s_validation"]).item())
        if not np.isfinite([val, noisy_ce]).all():
            raise FloatingPointError("Non-finite GCE validation loss")
        history.append({"epoch": epoch, "training_gce": float(total.item() / len(labels)),
                        "validation_gce": val, "validation_ce": noisy_ce})
        if best is None or val < best["validation_loss"]:
            best = {"model_state": cpu_copy(model.state_dict()), "selected_epoch": epoch,
                    "validation_loss": val}
        saved = {"signature": signature, "completed_epoch": epoch, "latest_model": cpu_copy(model.state_dict()),
                 "optimizer": cpu_copy(optimizer.state_dict()), "batch_rng": generator.get_state(),
                 "torch_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state_all()
                 if x_train.device.type == "cuda" else None,
                 "history": history, "best": best, "seconds": elapsed_before + time.perf_counter() - start}
        atomic_checkpoint(progress_path, saved)
        print(f"  gce epoch {epoch:02d}/{epochs}: val={val:.6f} best={best['selected_epoch']}", flush=True)
        if stop_after is not None and epoch == stop_after and epoch < epochs:
            raise InterruptedError("Simulated interruption after committed epoch")
    if best is None:
        raise ValueError("No completed GCE epoch")
    model.load_state_dict(best["model_state"])
    return {**best, "history": history, "seconds": elapsed_before + time.perf_counter() - start,
            "completed_epochs": epochs, "selection": "noisy_validation_gce", "q": q}


def make_config(args, name):
    """Return the recorded GCE configuration for one dataset."""
    return {"dataset": name, "runs": args.runs, "epochs": args.epochs if args.epochs is not None
            else (40 if name == "cifar" else 20), "q": args.q, "batch_size": 128, "lr": .001,
            "weight_decay": .0001, "flip": name == "cifar", "device": args.device,
            "smoke": args.smoke, "sample_limit": args.sample_limit, "test_limit": args.test_limit,
            "initialisation_seed_offset": 10000, "training_seed_offset": 20000,
            "selection": "noisy_validation_gce", "deterministic": True}


def train(args):
    """Train or resume GCE on each requested dataset and split; return result documents."""
    if not args.smoke and (args.runs != 10 or args.epochs is not None or args.sample_limit or args.test_limit):
        raise ValueError("Final GCE uses 10 full-data runs and the original 20/40 epoch budgets")
    if args.sample_limit is not None and args.sample_limit < 5:
        raise ValueError("Sample limit must be at least 5")
    if not np.isfinite(args.q) or not 0 < args.q <= 1:
        raise ValueError("q must lie in (0,1]")
    args.device = choose_device(args.device)
    torch.set_num_threads(args.threads)
    references = load_references(args.reference_dir) if not args.smoke else {}
    names = [args.dataset] if args.dataset else list(DATASETS)
    datasets = {name: load_data(name, args.data_dir) for name in names}
    definition = {"configurations": {n: make_config(args, n) for n in names}, "source": addon_identity(),
                  "environment": environment(args.device), "threads": args.threads,
                  "datasets": {n: file_hash(d.path) for n, d in datasets.items()},
                  "baseline_signatures": {n: r["results"]["signature"] for n, r in references.items()}}
    signature = identity(definition)
    output = args.output.resolve()
    manifest_path = output / "manifest.json"
    if output.exists():
        if not args.resume:
            raise FileExistsError("GCE output exists; use --resume or a new directory")
        if not manifest_path.is_file() or json.loads(manifest_path.read_text())["signature"] != signature:
            raise ValueError("GCE resume refused: source, data, configuration or environment changed")
    else:
        output.mkdir(parents=True)
        atomic_json(manifest_path, {"signature": signature, "definition": definition})
    lock = output / ".running.lock"
    with lock.open("x", encoding="utf-8") as stream:
        stream.write(str(os.getpid()))
    documents = {}
    try:
        for name, data in datasets.items():
            config = definition["configurations"][name]
            records = []
            for seed in range(args.runs):
                directory = output / name / f"run_{seed:03d}"
                directory.mkdir(parents=True, exist_ok=True)
                result_path = directory / "result.json"
                print(f"\n{name} GCE q={args.q} seed={seed}", flush=True)
                tensors, split = prepare_split(data, seed, args.device, args.sample_limit, args.test_limit)
                if references:
                    check_reference(data, split, config, references[PRIMARY[name]])
                if result_path.is_file():
                    record = json.loads(result_path.read_text())
                    verify_run(directory, record, signature, tensors, split, replay=False)
                    print("  verified completed run; no training", flush=True)
                else:
                    atomic_json(directory / "split.json", split)
                    seed_everything(10000 + seed)
                    model = CNN(tensors["x_train"].shape[1]).to(args.device)
                    view = {k: tensors[k] for k in ("x_train", "s_train", "x_validation", "s_validation")}
                    best = fit_gce(model, view, epochs=config["epochs"], seed=seed, batch_size=128,
                                   lr=config["lr"], weight_decay=config["weight_decay"], flip=config["flip"],
                                   q=args.q, progress_path=directory / "progress.pt", signature=signature)
                    best.update({"signature": signature, "seed": seed, "config": config})
                    atomic_checkpoint(directory / "best.pt", best)
                    score = accuracy(model, tensors["x_test"], tensors["y_test"])
                    record = {"signature": signature, "seed": seed, "accuracy": score,
                              "selected_epoch": best["selected_epoch"], "validation_loss": best["validation_loss"],
                              "seconds": best["seconds"], "best_sha256": file_hash(directory / "best.pt"),
                              "split_sha256": file_hash(directory / "split.json")}
                    atomic_json(result_path, record)
                records.append(record)
                documents[name] = {"signature": signature, "config": config, "runs": records,
                                   "summary": statistics([r["accuracy"] for r in records]),
                                   "complete": len(records) == args.runs,
                                   "reportable": not args.smoke and len(records) >= 10}
                atomic_json(output / name / "results.json", documents[name])
                print(f"  completed {len(records)}/{args.runs}: accuracy={record['accuracy']:.4f}", flush=True)
    finally:
        lock.unlink(missing_ok=True)
    return documents


def verify_run(directory, record, signature, tensors, split, replay):
    """Check one run's artefacts and checkpoint selection, optionally replaying accuracy."""
    if record["signature"] != signature or record["seed"] != split["seed"]:
        raise ValueError("GCE run identity mismatch")
    if file_hash(directory / "best.pt") != record["best_sha256"] or file_hash(directory / "split.json") != record["split_sha256"]:
        raise ValueError("GCE artefact checksum mismatch")
    if json.loads((directory / "split.json").read_text()) != split:
        raise ValueError("GCE split/preprocessing mismatch")
    best = torch.load(directory / "best.pt", map_location="cpu", weights_only=True)
    chosen = min(best["history"], key=lambda h: h["validation_gce"])
    if (best["signature"] != signature or best["seed"] != split["seed"]
            or best["selected_epoch"] != chosen["epoch"] or record["selected_epoch"] != chosen["epoch"]
            or best["validation_loss"] != chosen["validation_gce"]
            or record["validation_loss"] != chosen["validation_gce"]):
        raise ValueError("GCE checkpoint selection mismatch")
    if not np.isfinite(record["accuracy"]) or not 0 <= record["accuracy"] <= 100:
        raise ValueError("Invalid GCE accuracy")
    if replay:
        model = CNN(tensors["x_test"].shape[1]).to(tensors["x_test"].device)
        model.load_state_dict(best["model_state"])
        np.testing.assert_allclose(accuracy(model, tensors["x_test"], tensors["y_test"]), record["accuracy"], atol=1e-5, rtol=0)
    return best


def verify(args):
    """Audit every GCE run with test-prediction replay and write audit.json."""
    device = choose_device(args.device)
    torch.set_num_threads(args.threads)
    # CLI verification is a fresh process. Match training's deterministic
    # kernels and disabled TF32 before replaying any saved predictions.
    seed_everything(0)
    manifest = json.loads((args.output / "manifest.json").read_text())
    signature, definition = manifest["signature"], manifest["definition"]
    if identity(definition) != signature:
        raise ValueError("GCE manifest identity mismatch")
    checked = []
    for name, config in definition["configurations"].items():
        data = load_data(name, args.data_dir)
        if file_hash(data.path) != definition["datasets"][name]:
            raise ValueError("GCE data changed")
        doc = json.loads((args.output / name / "results.json").read_text())
        if doc["signature"] != signature or doc["config"] != config or len(doc["runs"]) != config["runs"]:
            raise ValueError("Incomplete GCE configuration")
        if not doc["complete"] or doc["reportable"] != (not config["smoke"] and config["runs"] >= 10):
            raise ValueError("GCE completion/reportable flag mismatch")
        if doc["summary"] != statistics([r["accuracy"] for r in doc["runs"]]):
            raise ValueError("GCE aggregate mismatch")
        if [r["seed"] for r in doc["runs"]] != list(range(config["runs"])):
            raise ValueError("GCE missing or duplicate seed")
        for record in doc["runs"]:
            tensors, split = prepare_split(data, record["seed"], device, config["sample_limit"], config["test_limit"])
            directory = args.output / name / f"run_{record['seed']:03d}"
            if record != json.loads((directory / "result.json").read_text()):
                raise ValueError("GCE run record differs from aggregate")
            best = verify_run(directory, record, signature, tensors, split, replay=True)
            if (best["config"] != config or best["q"] != config["q"]
                    or best["completed_epochs"] != config["epochs"]
                    or [h["epoch"] for h in best["history"]] != list(range(1, config["epochs"] + 1))):
                raise ValueError("GCE checkpoint configuration/history mismatch")
        checked.append({"dataset": name, "runs": len(doc["runs"]), "test_predictions_replayed": True,
                        "reportable": not config["smoke"] and len(doc["runs"]) >= 10})
        print(f"Verified {name}: {len(doc['runs'])} GCE checkpoints", flush=True)
    audit = {"passed": True, "experiments": checked}
    atomic_json(args.output / "audit.json", audit)
    return audit


def export(args):
    """Write combined accuracy and paired-difference tables against baseline results."""
    if args.reference_dir is None:
        raise ValueError("Comparison export requires the completed baseline JSON references")
    references = load_references(args.reference_dir)
    manifest = json.loads((args.output / "manifest.json").read_text())
    destination = args.output / "analysis"
    destination.mkdir(exist_ok=True)
    rows, pairs, differences = [], [], []
    for name, config in manifest["definition"]["configurations"].items():
        doc = json.loads((args.output / name / "results.json").read_text())
        if not doc["complete"] or not doc["reportable"] or doc["summary"] != statistics([r["accuracy"] for r in doc["runs"]]):
            raise ValueError("Final comparison requires complete, verified, non-smoke GCE results")
        audit = json.loads((args.output / "audit.json").read_text())
        if (not audit["passed"] or not any(e["dataset"] == name and e["runs"] == len(doc["runs"])
                and e["reportable"] and e["test_predictions_replayed"] for e in audit["experiments"])):
            raise ValueError("GCE audit must pass before comparison")
        for experiment, reference in references.items():
            baseline = reference["results"]
            if baseline["config"]["dataset"] != name:
                continue
            base_config = baseline["config"]
            expected = {r["seed"]: r for r in baseline["runs"]}
            gce = {r["seed"]: r["accuracy"] for r in doc["runs"]}
            if set(gce) != set(expected):
                raise ValueError("GCE and baseline seeds do not match")
            for seed in gce:
                a = json.loads((args.output / name / f"run_{seed:03d}" / "split.json").read_text())
                b = json.loads((reference["path"] / f"run_{seed:03d}" / "split.json").read_text())
                if a != b:
                    raise ValueError("Unpaired GCE/base splits")
            for method, stat in baseline["summary"]["acc"].items():
                rows.append({"experiment": experiment, "method": method, **stat,
                             "base_epochs": base_config["epochs"],
                             "extra_epochs": base_config["revision_epochs"] if method in ("continued", "continued_full", "trevision") else 0})
                pairs.append({"experiment": experiment, "comparison": "gce_minus_" + method,
                              **statistics([gce[s] - expected[s]["acc"][method] for s in sorted(gce)])})
            rows.append({"experiment": experiment, "method": "gce", **doc["summary"],
                         "base_epochs": config["epochs"], "extra_epochs": 0})
            old_env = reference["manifest"]["definition"]["environment"]
            new_env = manifest["definition"]["environment"]
            differences.append({"experiment": experiment, "changed_environment_fields":
                                [k for k in old_env if old_env[k] != new_env.get(k)]})
    for filename, values in (("combined_accuracy.csv", rows), ("paired_gce.csv", pairs)):
        with (destination / filename).open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(values[0]))
            writer.writeheader()
            writer.writerows(values)
    atomic_json(destination / "comparison_environment.json", differences)
    q = next(iter(manifest["definition"]["configurations"].values()))["q"]
    lines = ["# Completed baseline experiments plus GCE", "", f"GCE uses q={q}, fixed before its test outcomes. No old model was retrained.",
             "Mean ± sample SD across the same ten splits. Refinement methods receive extra epochs, shown separately.", "",
             "| Experiment | Method | Accuracy (%) | Base + extra epochs |", "|---|---|---:|---:|"]
    for row in rows:
        lines.append(f"| {row['experiment']} | {row['method']} | {row['mean']:.2f} ± {row['std']:.2f} | {row['base_epochs']} + {row['extra_epochs']} |")
    (destination / "comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return rows


def main():
    """Command-line entry point for the train, verify and export actions."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("train", "verify", "export"))
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference-dir", type=Path)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--dataset", choices=DATASETS)
    parser.add_argument("--runs", type=positive_int, default=10)
    parser.add_argument("--epochs", type=positive_int)
    parser.add_argument("--q", type=float, default=.7)
    parser.add_argument("--threads", type=positive_int, default=4)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--sample-limit", type=positive_int)
    parser.add_argument("--test-limit", type=positive_int)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.action == "train":
        train(args)
    elif args.action == "verify":
        verify(args)
    else:
        export(args)


if __name__ == "__main__":
    main()
