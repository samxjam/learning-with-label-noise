"""Post-hoc diagnostics for the report; they never influence any reported model.

gradient:   replays RW and its three refinements through the unchanged fit(), logging
            observed-label probabilities, the learned matrix and clean-test accuracy
            per epoch. Clean-test values are logged only; selection stays on noisy
            validation inside fit().
estimators: compares transition-estimator variants using CE posteriors, with no
            clean test labels. Fashion errors use the given matrices.
per_class:  replays saved checkpoints to give per-class clean-test accuracy.
"""
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from algorithm import engine
from algorithm.data import DATA_DIR, load_data, prepare_split
from algorithm.engine import accuracy, fit, predict, seed_everything
from algorithm.methods import estimate_transition, matrix_error, validate_transition
from algorithm.models import CNN

GRADIENT_RUNS = {"fashion0.6_given_trimmed": "fashion0.6", "cifar_estimated_trimmed": "cifar"}
PER_CLASS_RUNS = {"fashion0.3": "fashion0.3_given_trimmed", "fashion0.6": "fashion0.6_given_trimmed",
                  "cifar": "cifar_estimated_trimmed"}
BANDS = {"band_90_95": (90, 95), "band_94_96": (94, 96), "band_98_100": (98, 100)}
THRESHOLD = .33
REFINEMENTS = ("continued", "continued_full", "trevision")


def epochs_for(name, override):
    """Base epochs: 40 for CIFAR and 20 for Fashion, unless a smoke run overrides them."""
    return override or (40 if name == "cifar" else 20)


def base_fit(method, tensors, initial_state, matrix, epochs, flip, seed, callback=None, lr=.001, include_initial=False):
    """Train one stage with exactly the arguments used by main.run_one."""
    model = CNN(tensors["x_train"].shape[1]).to(tensors["x_train"].device)
    model.load_state_dict(initial_state)
    checkpoint = fit(model, method, tensors["x_train"], tensors["s_train"], tensors["x_validation"], tensors["s_validation"],
                     transition=matrix, epochs=epochs, lr=lr, batch_size=128, flip=flip, seed=seed, deterministic=True,
                     transition_lr=.0005, include_initial=include_initial, epoch_callback=callback(model) if callback else None)
    return model, checkpoint


def initial_weights(tensors, seed):
    """Create the shared initial weights for a split, as main.run_one does."""
    seed_everything(10000 + seed, True)
    model = CNN(tensors["x_train"].shape[1]).to(tensors["x_train"].device)
    return {k: v.detach().clone() for k, v in model.state_dict().items()}


def observed_probability(model, tensors):
    """Return each training example's probability for its observed noisy label."""
    logits = predict(model, tensors["x_train"])
    return logits.softmax(1).gather(1, tensors["s_train"][:, None]).squeeze(1)


def reference_scores(root, name):
    """Load reported per-split accuracies, when the original results are available."""
    path = None if root is None else Path(root) / name / "results.json"
    if path is None or not path.is_file():
        return {}
    return {r["seed"]: r for r in json.loads(path.read_text(encoding="utf-8"))["runs"]}


def gradient_study(args):
    """Replay refinements on known-T Fashion 0.6 and trimmed CIFAR with per-epoch logging."""
    latest = {}
    original_projection = engine.project_transition

    def recording_projection(matrix):
        result = original_projection(matrix)
        latest["R"] = result.detach().cpu().clone()
        return result

    engine.project_transition = recording_projection
    for name, dataset in GRADIENT_RUNS.items():
        data = load_data(dataset, args.data_dir)
        reported = reference_scores(args.reference_root, name)
        epochs = epochs_for(dataset, args.epochs)
        extra = max(1, epochs // 2)
        flip = dataset == "cifar"
        for seed in args.splits:
            path = args.output / f"gradient__{name}__{seed}.json"
            if path.is_file():
                print("skip", path.name, flush=True)
                continue
            tensors, _ = prepare_split(data, seed, args.device, args.sample_limit, args.test_limit)
            initial = initial_weights(tensors, seed)
            ce_model, ce = base_fit("ce", tensors, initial, None, epochs, flip, 20000 + seed)
            probabilities = predict(ce_model, tensors["x_train"]).softmax(1).cpu().numpy()
            estimated = estimate_transition(probabilities, "trimmed")[0]
            start = data.true_transition if data.true_transition is not None else estimated
            matrix = torch.tensor(start, dtype=torch.float32, device=args.device)
            validate_transition(matrix)
            rw_model, reweight = base_fit("reweight", tensors, initial, matrix, epochs, flip, 20000 + seed)
            record = {"experiment": name, "seed": seed, "start_matrix": np.asarray(start).tolist(), "threshold": THRESHOLD,
                      "ce_accuracy": accuracy(ce_model, tensors["x_test"], tensors["y_test"]),
                      "reweight_accuracy": accuracy(rw_model, tensors["x_test"], tensors["y_test"]), "branches": {}}
            p0 = observed_probability(rw_model, tensors)
            below = p0 < THRESHOLD
            record["share_below_at_start"] = float(below.float().mean().item())
            del rw_model, ce_model
            for stage in REFINEMENTS:
                latest.pop("R", None)
                rows = []

                def snapshot(model, epoch):
                    p = observed_probability(model, tensors)
                    return {"epoch": epoch, "test_accuracy": accuracy(model, tensors["x_test"], tensors["y_test"]),
                            "share_below": float((p < THRESHOLD).float().mean().item()),
                            "mean_p_start_below": float(p[below].mean().item()) if below.any() else None,
                            "mean_p_start_above": float(p[~below].mean().item()) if (~below).any() else None,
                            "R": latest["R"].tolist() if "R" in latest else None}

                def callback(model):
                    def on_epoch(history):
                        row = snapshot(model, history[-1]["epoch"])
                        row["validation_loss"] = history[-1]["validation_loss"]
                        rows.append(row)
                    return on_epoch

                probe = CNN(tensors["x_train"].shape[1]).to(args.device)
                probe.load_state_dict(reweight["model_state"])
                rows.append(snapshot(probe, 0))
                del probe
                model, checkpoint = base_fit(stage, tensors, reweight["model_state"], matrix, extra, flip, 30000 + seed,
                                             callback=callback, lr=.0002, include_initial=True)
                rows[0]["validation_loss"] = checkpoint["history"][0]["validation_loss"]
                score = accuracy(model, tensors["x_test"], tensors["y_test"])
                record["branches"][stage] = {"selected_epoch": checkpoint["selected_epoch"], "selected_accuracy": score, "epochs": rows}
                del model
            if seed in reported:
                old = reported[seed]
                record["reproduction"] = {
                    "reweight_difference": record["reweight_accuracy"] - old["acc"]["reweight"],
                    "ce_difference": record["ce_accuracy"] - old["acc"]["ce"],
                    **{stage + "_difference": record["branches"][stage]["selected_accuracy"] - old["acc"][stage] for stage in REFINEMENTS},
                    "selected_epochs_match": all(record["branches"][s]["selected_epoch"] == old["selected_epochs"][s] for s in REFINEMENTS)}
            path.write_text(json.dumps(record), encoding="utf-8")
            print(name, "seed", seed, "start share below", round(record["share_below_at_start"], 3),
                  "reproduction", record.get("reproduction", "not available"), flush=True)
    engine.project_transition = original_projection


def summarise_gradient(output, splits):
    """Average per-epoch logs across splits and write one compact JSON summary."""
    summary = {}
    for name in GRADIENT_RUNS:
        records = [json.loads(p.read_text(encoding="utf-8")) for s in splits
                   if (p := output / f"gradient__{name}__{s}.json").is_file()]
        if not records:
            continue
        entry = {"splits": len(records), "share_below_at_start": float(np.mean([r["share_below_at_start"] for r in records]))}
        reproductions = [r["reproduction"] for r in records if "reproduction" in r]
        if reproductions:
            differences = [abs(v) for r in reproductions for k, v in r.items() if k.endswith("_difference")]
            entry["reproduction"] = {"max_absolute_accuracy_difference": max(differences),
                                     "selected_epochs_all_match": all(r["selected_epochs_match"] for r in reproductions)}
        entry["branches"] = {}
        for stage in REFINEMENTS:
            runs = [r["branches"][stage] for r in records]
            columns = {key: np.array([[row[key] for row in run["epochs"]] for run in runs], dtype=float)
                       for key in ("test_accuracy", "share_below", "mean_p_start_below", "mean_p_start_above", "validation_loss")}
            selected = np.array([run["selected_accuracy"] for run in runs])
            branch = {"mean_by_epoch": {key: np.round(values.mean(0), 5).tolist() for key, values in columns.items()},
                      "selected_epochs": [run["selected_epoch"] for run in runs],
                      "selected_accuracy": {"mean": float(selected.mean()), "per_split": np.round(selected, 2).tolist()},
                      "final_epoch_accuracy": {"mean": float(columns["test_accuracy"][:, -1].mean()),
                                               "per_split": np.round(columns["test_accuracy"][:, -1], 2).tolist()},
                      "best_test_epoch_accuracy_diagnostic_only": float(columns["test_accuracy"].max(1).mean()),
                      "p_below_change_epoch1_per_split": np.round(columns["mean_p_start_below"][:, 1] - columns["mean_p_start_below"][:, 0], 5).tolist(),
                      "p_above_change_epoch1_per_split": np.round(columns["mean_p_start_above"][:, 1] - columns["mean_p_start_above"][:, 0], 5).tolist(),
                      "p_below_change_final_per_split": np.round(columns["mean_p_start_below"][:, -1] - columns["mean_p_start_below"][:, 0], 5).tolist(),
                      "p_above_change_final_per_split": np.round(columns["mean_p_start_above"][:, -1] - columns["mean_p_start_above"][:, 0], 5).tolist()}
            if runs[0]["epochs"][-1]["R"] is not None:
                matrices = np.array([[row["R"] for row in run["epochs"][1:]] for run in runs])
                starts = np.array([r["start_matrix"] for r in records])
                branch["R_mean_by_epoch"] = np.round(matrices.mean(0), 4).tolist()
                branch["R_mean_absolute_change_from_start_by_epoch"] = np.round(
                    np.abs(matrices - starts[:, None]).sum(axis=(2, 3)).mean(0) / 3, 5).tolist()
            entry["branches"][stage] = branch
        summary[name] = entry
    (output / "gradient_summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")


def trimmed_row_matrix(probabilities, low, high):
    """Average noisy posteriors inside a percentile band of each class's confidence."""
    rows = []
    for klass in range(3):
        lower, upper = np.percentile(probabilities[:, klass], [low, high])
        rows.append(probabilities[(probabilities[:, klass] >= lower) & (probabilities[:, klass] <= upper)].mean(axis=0))
    matrix = np.stack(rows)
    return matrix / matrix.sum(axis=1, keepdims=True)


def fit_temperature(logits, labels):
    """Choose the temperature minimising noisy-validation cross-entropy over a fixed grid."""
    grid = np.geomspace(.25, 8, 241)
    losses = [float(F.cross_entropy(logits / t, labels).item()) for t in grid]
    return float(grid[int(np.argmin(losses))])


def estimator_study(args):
    """Evaluate estimator variants on every dataset; no clean test label is used."""
    summary = {}
    for dataset in ("fashion0.3", "fashion0.6", "cifar"):
        data = load_data(dataset, args.data_dir)
        epochs = epochs_for(dataset, args.epochs)
        flip = dataset == "cifar"
        rows = []
        for seed in args.splits:
            path = args.output / f"estimators__{dataset}__{seed}.json"
            if path.is_file():
                rows.append(json.loads(path.read_text(encoding="utf-8")))
                print("skip", path.name, flush=True)
                continue
            tensors, _ = prepare_split(data, seed, args.device, args.sample_limit, args.test_limit)
            del tensors["x_test"], tensors["y_test"]
            initial = initial_weights(tensors, seed)
            model, _ = base_fit("ce", tensors, initial, None, epochs, flip, 20000 + seed)
            train_logits = predict(model, tensors["x_train"])
            validation_logits = predict(model, tensors["x_validation"])
            probabilities = train_logits.softmax(1).cpu().numpy().astype(np.float64)
            temperature = fit_temperature(validation_logits, tensors["s_validation"])
            matrices = {"trimmed_96_98": estimate_transition(probabilities, "trimmed")[0],
                        "argmax": estimate_transition(probabilities, "argmax")[0],
                        "temperature_96_98": trimmed_row_matrix((train_logits / temperature).softmax(1).cpu().numpy().astype(np.float64), 96, 98),
                        "validation_96_98": trimmed_row_matrix(validation_logits.softmax(1).cpu().numpy().astype(np.float64), 96, 98)}
            for label, (low, high) in BANDS.items():
                matrices[label] = trimmed_row_matrix(probabilities, low, high)
            ensemble = [probabilities]
            for member in (1, 2):
                other_initial = initial_weights(tensors, 40000 + 10 * member + seed)
                other, _ = base_fit("ce", tensors, other_initial, None, epochs, flip, 50000 + 10 * member + seed)
                ensemble.append(predict(other, tensors["x_train"]).softmax(1).cpu().numpy().astype(np.float64))
            matrices["ensemble3_96_98"] = trimmed_row_matrix(np.mean(ensemble, axis=0), 96, 98)
            record = {"dataset": dataset, "seed": seed, "temperature": temperature,
                      "matrices": {k: v.tolist() for k, v in matrices.items()}}
            if data.true_transition is not None:
                record["errors"] = {k: matrix_error(v, data.true_transition) for k, v in matrices.items()}
            path.write_text(json.dumps(record), encoding="utf-8")
            rows.append(record)
            print(dataset, "seed", seed, "T", round(temperature, 2),
                  {k: round(v, 3) for k, v in record.get("errors", {}).items()}, flush=True)
        names = rows[0]["matrices"]
        summary[dataset] = {"temperature": float(np.mean([r["temperature"] for r in rows])),
                            "mean_matrix": {k: np.mean([r["matrices"][k] for r in rows], axis=0).tolist() for k in names}}
        if "errors" in rows[0]:
            summary[dataset]["errors"] = {k: {"mean": float(np.mean([r["errors"][k] for r in rows])),
                                              "sd": float(np.std([r["errors"][k] for r in rows], ddof=1)) if len(rows) > 1 else None}
                                          for k in names}
    (args.output / "estimators_summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")


def per_class_study(args):
    """Replay saved checkpoints; each overall score must match its saved value."""
    if args.reference_root is None:
        print("per_class skipped: no --reference-root", flush=True)
        return
    summary = {}
    seed_everything(0, True)
    for dataset, name in PER_CLASS_RUNS.items():
        data = load_data(dataset, args.data_dir)
        directory = Path(args.reference_root) / name
        document = json.loads((directory / "results.json").read_text(encoding="utf-8"))
        config = document["config"]
        collected = {}
        for run in document["runs"]:
            seed = run["seed"]
            tensors, _ = prepare_split(data, seed, args.device, config["sample_limit"], config["test_limit"])
            sources = {m: (directory / f"run_{seed:03d}" / f"{m}.pt", run["acc"][m]) for m in ("ce", "forward", "reweight", "trevision")}
            if args.gce_root is not None:
                gce_result = Path(args.gce_root) / dataset / f"run_{seed:03d}"
                if (gce_result / "best.pt").is_file():
                    sources["gce"] = (gce_result / "best.pt", json.loads((gce_result / "result.json").read_text())["accuracy"])
            for method, (path, expected) in sources.items():
                checkpoint = torch.load(path, map_location="cpu", weights_only=True)
                model = CNN(tensors["x_test"].shape[1]).to(args.device)
                model.load_state_dict(checkpoint["model_state"])
                predictions = predict(model, tensors["x_test"]).argmax(1)
                overall = float((predictions == tensors["y_test"]).float().mean().item() * 100)
                np.testing.assert_allclose(overall, expected, atol=1e-5, rtol=0)
                confusion = np.zeros((3, 3))
                for truth, guess in zip(tensors["y_test"].cpu().numpy(), predictions.cpu().numpy()):
                    confusion[truth, guess] += 1
                collected.setdefault(method, []).append(confusion / confusion.sum(axis=1, keepdims=True))
        summary[dataset] = {m: {"mean_confusion": np.mean(v, axis=0).tolist(), "sd_confusion": np.std(v, axis=0, ddof=1).tolist() if len(v) > 1 else None,
                                "runs": len(v)} for m, v in collected.items()}
        print("per_class", dataset, {m: np.round(np.diag(np.mean(v, axis=0)) * 100, 1).tolist() for m, v in collected.items()}, flush=True)
    (args.output / "per_class.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")


def main():
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=("gradient", "estimators", "per_class", "all"), default="all")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--splits", type=int, nargs="+", default=list(range(10)))
    parser.add_argument("--reference-root", type=Path, help="directory holding the reported experiment folders")
    parser.add_argument("--gce-root", type=Path, help="directory holding the reported GCE dataset folders")
    parser.add_argument("--epochs", type=int, help="smoke checks only")
    parser.add_argument("--sample-limit", type=int, help="smoke checks only")
    parser.add_argument("--test-limit", type=int, help="smoke checks only")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    if args.task in ("per_class", "all"):
        per_class_study(args)
    if args.task in ("gradient", "all"):
        gradient_study(args)
        summarise_gradient(args.output, args.splits)
    if args.task in ("estimators", "all"):
        estimator_study(args)


if __name__ == "__main__":
    main()
