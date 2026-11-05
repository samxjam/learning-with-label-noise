"""Run a protected, resumable experiment. See --help and ../README.md."""
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import torch

try:
    from .artifacts import atomic_json, atomic_checkpoint, source_identity, environment, identity, summarise
    from .data import DATA_DIR, DATASETS, load_data, prepare_split, file_hash
    from .models import CNN
    from .methods import estimate_transition, matrix_error, validate_transition
    from .engine import seed_everything, fit, predict, accuracy
except ImportError:
    from artifacts import atomic_json, atomic_checkpoint, source_identity, environment, identity, summarise
    from data import DATA_DIR, DATASETS, load_data, prepare_split, file_hash
    from models import CNN
    from methods import estimate_transition, matrix_error, validate_transition
    from engine import seed_everything, fit, predict, accuracy

ROOT = Path(__file__).resolve().parents[1]
METHODS = ("ce", "forward", "reweight", "continued", "continued_full", "trevision")


def choose_device(requested):
    """Probe GPU kernels in a child: a broken native runtime must not kill the run."""
    if requested == "cpu":
        return "cpu"
    if not torch.cuda.is_available():
        if requested == "cuda":
            raise ValueError("CUDA/ROCm requested but unavailable")
        return "cpu"
    code = ("import torch; x=torch.ones(2,1,8,8,device='cuda'); "
            "m=torch.nn.Conv2d(1,2,3).cuda(); m(x).sum().backward(); torch.cuda.synchronize()")
    try:
        result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=45)
        failure = None if result.returncode == 0 else f"GPU kernel probe exited with {result.returncode}"
    except subprocess.TimeoutExpired:
        failure = "GPU kernel probe timed out after 45 seconds"
    if failure:
        if requested == "cuda":
            raise ValueError(f"{failure}; use --device cpu or repair the GPU environment")
        print(f"{failure}; automatic selection is using CPU", flush=True)
        return "cpu"
    return "cuda"


def positive_int(value):
    """Argparse type accepting integers of at least one."""
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def parser():
    """Build the command-line interface for a single experiment."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", required=True, choices=DATASETS)
    p.add_argument("--data-dir", type=Path, default=DATA_DIR)
    p.add_argument("--output", type=Path, required=True, help="new experiment directory; never silently overwritten")
    p.add_argument("--runs", type=positive_int, default=10)
    p.add_argument("--epochs", type=positive_int, help="default: Fashion 20, CIFAR 40")
    p.add_argument("--revision-epochs", type=positive_int, help="default: max(1, epochs//2)")
    p.add_argument("--estimated", "--use_estimated_T", action="store_true")
    p.add_argument("--estimator", choices=("trimmed", "argmax"), default="trimmed")
    p.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    p.add_argument("--batch-size", type=positive_int, default=128)
    p.add_argument("--seed", type=int, default=0, help="first split seed, nonnegative")
    p.add_argument("--threads", type=positive_int, default=4)
    p.add_argument("--resume", action="store_true", help="continue only an identical manifest; verify completed artefacts")
    p.add_argument("--smoke", action="store_true", help="marks output as non-reportable; permits fewer than ten runs")
    p.add_argument("--sample-limit", type=positive_int, help="smoke-only total train/validation subset")
    p.add_argument("--test-limit", type=positive_int, help="smoke-only test subset")
    p.add_argument("--allow-nondeterministic", action="store_true", help="explicitly relax deterministic kernels")
    return p


def build_config(args):
    """Validate arguments and return the full configuration recorded with the results."""
    if args.seed < 0 or args.seed + args.runs + 30000 >= 2 ** 32:
        raise ValueError("Seed range must be nonnegative and fit in 32 bits after stream offsets")
    if not args.smoke and (args.runs < 10 or args.sample_limit is not None or args.test_limit is not None):
        raise ValueError("Final experiments need >=10 runs and all examples; use --smoke for limited checks")
    if args.sample_limit is not None and args.sample_limit < 5:
        raise ValueError("Sample limit must be at least 5")
    epochs = args.epochs if args.epochs is not None else (40 if args.dataset == "cifar" else 20)
    device = choose_device(args.device)
    return {"dataset": args.dataset, "runs": args.runs, "epochs": epochs,
            "revision_epochs": args.revision_epochs if args.revision_epochs is not None else max(1, epochs // 2),
            "transition_mode": "estimated" if args.estimated or args.dataset == "cifar" else "given",
            "estimator": args.estimator, "device": device, "batch_size": args.batch_size,
            "seed": args.seed, "threads": args.threads, "lr": .001, "revision_lr": .0002,
            "transition_lr": .0005, "weight_decay": .0001,
            "deterministic": not args.allow_nondeterministic, "smoke": args.smoke,
            "sample_limit": args.sample_limit, "test_limit": args.test_limit,
            "methods": list(METHODS), "flip": args.dataset == "cifar",
            "revision_variant": "full_gradient_projected_simplex_noisy_forward_validation",
            "revision_epoch_zero": True, "normalisation": "training_split_scalar_mean_std",
            "sd_ddof": 1}


def checked_checkpoint(path, signature, stage):
    """Load a completed stage if present, rejecting a wrong identity or checksum."""
    meta_path = path.with_suffix(".json")
    if not path.is_file() or not meta_path.is_file():
        return None
    metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    if metadata.get("signature") != signature or metadata.get("sha256") != file_hash(path):
        raise ValueError(f"Checkpoint identity/integrity mismatch: {path}")
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint["method"] != stage or checkpoint["signature"] != signature:
        raise ValueError(f"Checkpoint stage mismatch: {path}")
    return checkpoint


def run_one(data, config, output, signature, seed):
    """Run one 80/20 split end to end and save its result record.

    Order: CE, transition estimates from CE posteriors, forward, reweight, then three
    refinements from the selected reweight model. Clean test labels are used only after
    every stage has selected its checkpoint.
    """
    directory = output / f"run_{seed:03d}"
    directory.mkdir(parents=True, exist_ok=True)
    tensors, split = prepare_split(data, seed, config["device"], config["sample_limit"], config["test_limit"])
    atomic_json(directory / "split.json", split)
    seed_everything(10000 + seed, config["deterministic"])
    initial = CNN(tensors["x_train"].shape[1]).to(config["device"])
    initial_state = {k: v.detach().clone() for k, v in initial.state_dict().items()}
    del initial
    checkpoint_hashes = {}

    def train_stage(stage, matrix=None, start=None):
        """Train a stage, or reuse its verified checkpoint when resuming."""
        path = directory / f"{stage}.pt"
        checkpoint = checked_checkpoint(path, signature, stage)
        model = CNN(tensors["x_train"].shape[1]).to(config["device"])
        if checkpoint is None:
            refining = stage in ("trevision", "continued", "continued_full")
            model.load_state_dict(start if start is not None else initial_state)
            checkpoint = fit(model, stage, tensors["x_train"], tensors["s_train"],
                             tensors["x_validation"], tensors["s_validation"], transition=matrix,
                             epochs=config["revision_epochs"] if refining else config["epochs"],
                             lr=config["revision_lr"] if refining else config["lr"],
                             batch_size=config["batch_size"], flip=config["flip"],
                             seed=(30000 if refining else 20000) + seed,
                             deterministic=config["deterministic"], transition_lr=config["transition_lr"],
                             include_initial=refining,
                             epoch_callback=lambda history: atomic_json(directory / f"{stage}_history.json", history))
            checkpoint.update({"signature": signature, "split_seed": seed, "initialisation_seed": 10000 + seed,
                               "normalisation": {k: split[k] for k in ("mean", "std")}, "config": config})
            atomic_checkpoint(path, checkpoint)
            atomic_json(path.with_suffix(".json"), {"signature": signature, "sha256": file_hash(path)})
        else:
            print(f"  verified cached stage: {stage}", flush=True)
            model.load_state_dict(checkpoint["model_state"])
        checkpoint_hashes[stage] = file_hash(path)
        return model, checkpoint

    # Estimate both candidate matrices on training predictions only.
    baseline, ce = train_stage("ce")
    probabilities = predict(baseline, tensors["x_train"]).softmax(1).cpu().numpy()
    estimates = {}
    for estimator in ("trimmed", "argmax"):
        matrix, diagnostics = estimate_transition(probabilities, estimator)
        estimates[estimator] = {"matrix": matrix.tolist(), "anchors": diagnostics}
        if data.true_transition is not None:
            estimates[estimator]["relative_l1_error"] = matrix_error(matrix, data.true_transition)
    del probabilities, baseline
    # Known-T runs start from the true matrix; estimated-T runs from the chosen estimate.
    selected = estimates[config["estimator"]]["matrix"]
    starting = data.true_transition if config["transition_mode"] == "given" else selected
    transition = torch.tensor(starting, dtype=torch.float32, device=config["device"])
    validate_transition(transition)
    model, forward = train_stage("forward", transition)
    del model
    model, reweight = train_stage("reweight", transition)
    del model
    # The three refinements share a starting model, budget and random stream, so their
    # paired differences isolate the gradient rule and matrix learning.
    model, continued = train_stage("continued", transition, reweight["model_state"])
    del model
    model, continued_full = train_stage("continued_full", transition, reweight["model_state"])
    del model
    model, revision = train_stage("trevision", transition, reweight["model_state"])
    del model
    checkpoints = dict(zip(METHODS, (ce, forward, reweight, continued, continued_full, revision)))
    scores = {}
    # Only now evaluate clean test labels: no checkpoint/estimator selection uses them.
    for method, checkpoint in checkpoints.items():
        model = CNN(tensors["x_train"].shape[1]).to(config["device"])
        model.load_state_dict(checkpoint["model_state"])
        scores[method] = accuracy(model, tensors["x_test"], tensors["y_test"])
        del model
    revised = revision["transition"].numpy()
    validate_transition(revised)
    record = {"signature": signature, "seed": seed, "acc": scores,
              "T_hat": selected, "T_used": transition.cpu().tolist(), "T_revised": revised.tolist(),
              "T_delta": (revised - transition.cpu().numpy()).tolist(),
              "transition_mode": config["transition_mode"], "estimators": estimates,
              "selected_epochs": {m: c["selected_epoch"] for m, c in checkpoints.items()},
              "validation_losses": {m: c["validation_loss"] for m, c in checkpoints.items()},
              "training_seconds": {m: c["seconds"] for m, c in checkpoints.items()},
              "checkpoint_hashes": checkpoint_hashes, "split_sha256": file_hash(directory / "split.json")}
    if data.true_transition is not None:
        record.update({"T_true": data.true_transition.tolist(),
                       "T_error": matrix_error(selected, data.true_transition),
                       "T_used_error": matrix_error(transition.cpu().numpy(), data.true_transition),
                       "T_revised_error": matrix_error(revised, data.true_transition)})
    atomic_json(directory / "result.json", record)
    return record


def verify_record(record, directory, signature, seed):
    """Check a saved run: identity, accuracy range, matrices, split and checkpoint hashes."""
    if record["signature"] != signature or record["seed"] != seed or set(record["acc"]) != set(METHODS):
        raise ValueError(f"Run identity/completeness mismatch: {directory}")
    if not all(np.isfinite(v) and 0 <= v <= 100 for v in record["acc"].values()):
        raise ValueError(f"Invalid accuracy in {directory}")
    for key in ("T_hat", "T_used", "T_revised"):
        validate_transition(record[key])
    if file_hash(directory / "split.json") != record["split_sha256"]:
        raise ValueError(f"Split integrity mismatch: {directory}")
    for method in METHODS:
        path = directory / f"{method}.pt"
        if file_hash(path) != record["checkpoint_hashes"][method]:
            raise ValueError(f"Checkpoint integrity mismatch: {path}")
        if checked_checkpoint(path, signature, method) is None:
            raise ValueError(f"Incomplete checkpoint: {path}")


def run_experiment(args):
    """Run or resume all splits of one experiment under a lock; return the aggregate."""
    config = build_config(args)
    torch.set_num_threads(config["threads"])
    data = load_data(args.dataset, args.data_dir)
    definition = {"schema_version": 1, "config": config, "source": source_identity(),
                  "data_sha256": file_hash(data.path), "environment": environment(config["device"])}
    signature = identity(definition)
    output = args.output.resolve()
    manifest_path = output / "manifest.json"
    if output.exists():
        if not args.resume:
            raise FileExistsError(f"Output already exists: {output}. Choose a new path or use --resume.")
        if not manifest_path.is_file():
            raise ValueError("Cannot resume a directory without a valid manifest")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("signature") != signature or manifest.get("definition") != definition:
            raise ValueError("Resume refused: source, config, dataset or environment changed; choose a new output")
    else:
        output.mkdir(parents=True, exist_ok=False)
        atomic_json(manifest_path, {"signature": signature, "definition": definition,
                    "created_utc": datetime.now(timezone.utc).isoformat(), "data_path": str(data.path)})
    lock = output / ".running.lock"
    try:
        handle = lock.open("x", encoding="utf-8")
    except FileExistsError as exc:
        raise RuntimeError(f"Experiment is locked: {lock}. Confirm no runner is alive before removing a stale lock.") from exc
    records = []
    try:
        with handle:
            handle.write(str(os.getpid()))
            handle.flush()
            for seed in range(config["seed"], config["seed"] + config["runs"]):
                directory = output / f"run_{seed:03d}"
                result_path = directory / "result.json"
                print(f"\n{config['dataset']} {config['transition_mode']}/{config['estimator']} seed={seed}", flush=True)
                if result_path.is_file():
                    record = json.loads(result_path.read_text(encoding="utf-8"))
                    verify_record(record, directory, signature, seed)
                    print("  verified complete run", flush=True)
                else:
                    record = run_one(data, config, output, signature, seed)
                records.append(record)
                payload = {"signature": signature, "config": config, "runs": records,
                           "summary": summarise(records), "complete": len(records) == config["runs"],
                           "reportable": not config["smoke"] and len(records) >= 10}
                atomic_json(output / "results.json", payload)
                print(f"  completed {len(records)}/{config['runs']}: {record['acc']}", flush=True)
    finally:
        lock.unlink(missing_ok=True)
    return payload


def main():
    """Command-line entry point; reports expected failures without a traceback."""
    p = parser()
    args = p.parse_args()
    try:
        run_experiment(args)
    except (ValueError, FileNotFoundError, FileExistsError, RuntimeError) as exc:
        p.exit(2, f"error: {exc}\n")


if __name__ == "__main__":
    main()
