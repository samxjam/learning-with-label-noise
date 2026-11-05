"""Export reproducible tables; optional Matplotlib figures from saved runs only."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

try:
    from .artifacts import summarise, atomic_json, statistics
except ImportError:
    from artifacts import summarise, atomic_json, statistics


def experiment_files(root):
    """Find results.json for one experiment directory or for every experiment in a suite."""
    direct = Path(root) / "results.json"
    return [direct] if direct.is_file() else sorted(Path(root).glob("*/results.json"))


def export(root, output, plots=False):
    """Write CSV, JSON and Markdown summaries, plus optional figures; return accuracy rows.

    Stored summaries are recomputed from run records first, so a stale table raises an error.
    """
    files = experiment_files(root)
    if not files:
        raise FileNotFoundError(f"No experiment results in {root}")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    rows, matrices, pairs, estimator_rows, documents = [], {}, [], [], []
    for path in files:
        result = json.loads(path.read_text(encoding="utf-8"))
        computed = summarise(result["runs"])
        if computed != result["summary"]:
            raise ValueError(f"Stored summary does not match run records: {path}")
        config = result["config"]
        label = path.parent.name
        for method, values in computed["acc"].items():
            rows.append({"experiment": label, "dataset": config["dataset"],
                         "transition": config["transition_mode"], "estimator": config["estimator"],
                         "method": method, "n": values["n"], "mean_accuracy": values["mean"],
                         "sample_sd": values["std"], "complete": result["complete"],
                         "reportable": result["reportable"]})
        matrices[label] = {key: computed[key] for key in
                           ("T_hat", "T_used", "T_revised", "T_error", "T_used_error", "T_revised_error") if key in computed}
        for comparison in ("paired_revision_minus_continued_accuracy", "paired_revision_minus_full_gradient_fixed_T",
                           "paired_full_gradient_minus_detached_fixed_T"):
            pairs.append({"experiment": label, "comparison": comparison, **computed[comparison]})
        for strategy in ("trimmed", "argmax"):
            errors = [run["estimators"][strategy].get("relative_l1_error") for run in result["runs"]]
            if all(value is not None for value in errors):
                estimator_rows.append({"experiment": label, "estimator": strategy, **statistics(errors)})
        documents.append((path, result))
    for filename, values in (("accuracy.csv", rows), ("paired_revision.csv", pairs), ("estimator_error.csv", estimator_rows)):
        if values:
            with (output / filename).open("w", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(values[0]))
                writer.writeheader()
                writer.writerows(values)
    atomic_json(output / "transition_matrices.json", matrices)
    lines = ["# Experiment results", "", "SD is the sample standard deviation across splits (ddof=1).",
             "Smoke/partial results are verification evidence, not final assignment results.", "",
             "| Experiment | Method | Runs | Accuracy (%) | Final eligible |", "|---|---|---:|---:|---|"]
    for row in rows:
        sd = f"{row['sample_sd']:.2f}" if row["sample_sd"] is not None else "undefined (n=1)"
        lines.append(f"| {row['experiment']} | {row['method']} | {row['n']} | {row['mean_accuracy']:.2f} ± {sd} | {row['reportable'] and row['complete']} |")
    lines += ["", "Paired accuracy differences (percentage points):"]
    for row in pairs:
        lines.append(f"- {row['experiment']} / {row['comparison']}: mean {row['mean']:.4f}, sample SD {row['std']}, n={row['n']}.")
    (output / "tables.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    if plots:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        for path, result in documents:
            summary = result["summary"]
            names = list(summary["acc"])
            fig, ax = plt.subplots(figsize=(7, 4))
            ax.bar(names, [summary["acc"][m]["mean"] for m in names],
                   yerr=[summary["acc"][m]["std"] or 0 for m in names], capsize=4)
            ax.set(ylabel="Clean-test top-1 accuracy (%)", ylim=(0, 100), title=path.parent.name)
            fig.tight_layout()
            fig.savefig(output / f"{path.parent.name}_accuracy.pdf")
            plt.close(fig)
            fig, axes = plt.subplots(1, 3, figsize=(10, 3))
            for ax, key in zip(axes, ("T_hat", "T_used", "T_revised")):
                values = np.asarray(summary[key]["mean"])
                ax.imshow(values, vmin=0, vmax=1, cmap="Blues")
                for i in range(3):
                    for j in range(3):
                        ax.text(j, i, f"{values[i,j]:.3f}", ha="center", va="center", color="white" if values[i,j] > .6 else "black")
                ax.set(title=key, xlabel="Noisy class", ylabel="Clean class", xticks=range(3), yticks=range(3))
            fig.tight_layout()
            fig.savefig(output / f"{path.parent.name}_matrices.pdf")
            plt.close(fig)
            seed = result["runs"][0]["seed"]
            fig, axes = plt.subplots(2, 3, figsize=(11, 6))
            for ax, method in zip(axes.flat, names):
                history_path = path.parent / f"run_{seed:03d}" / f"{method}_history.json"
                history = json.loads(history_path.read_text(encoding="utf-8"))
                ax.plot([h["epoch"] for h in history], [h["validation_loss"] for h in history], label="Noisy validation")
                chosen = result["runs"][0]["selected_epochs"][method]
                ax.axvline(chosen, linestyle="--", color="grey", label="Selected epoch")
                ax.set(title=method, xlabel="Epoch", ylabel="Validation loss")
                ax.legend(fontsize=7)
            fig.suptitle(f"{path.parent.name}: diagnostic curves, seed {seed}")
            fig.tight_layout()
            fig.savefig(output / f"{path.parent.name}_validation.pdf")
            plt.close(fig)
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--plots", action="store_true", help="requires Matplotlib")
    args = parser.parse_args()
    export(args.root, args.output, args.plots)
