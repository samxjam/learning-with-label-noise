# COMP4328 Assignment 2: learning with noisy labels

Group 4: Abdullah Shabib and Samuel James.

This package trains and compares classifiers that are robust to class-conditional label noise, on three supplied datasets. It implements cross-entropy (CE), forward correction, posterior-ratio reweighting, a constrained T-revision variant and Generalised Cross Entropy (GCE), with two transition-matrix estimators. `METHODS.md` defines each method and the protocol. `README_GCE.md` covers GCE.

## Layout

| Path | Purpose |
|---|---|
| `algorithm/data.py` | Load datasets; make random 80/20 splits; fit normalisation on the training split |
| `algorithm/models.py` | Shared CNN |
| `algorithm/methods.py` | Losses, simplex projection, transition estimators, matrix error |
| `algorithm/engine.py` | Training loop and noisy-validation checkpoint selection |
| `algorithm/main.py` | Run one experiment: ten splits, six logged stages |
| `algorithm/suite.py` | Run the required or extended list of experiments |
| `algorithm/verify.py` | Audit saved results and replay clean-test predictions |
| `algorithm/analyse.py` | Export tables and optional figures |
| `algorithm/artifacts.py` | Atomic saves, identities and statistics |
| `algorithm/diagnostics.py` | Per-class accuracy, replay of refinements with per-epoch logging, estimator comparison |
| `algorithm/pilot.py` | Blind comparison of training settings on noisy validation only |
| `gce_addon.py` | Train, verify and export GCE; imports `algorithm/` and needs no main-suite results |
| `tests/`, `test_gce_addon.py` | Unit and regression tests |
| `analysis.ipynb` | Optional notebook view over the same modules |
| `colab/` | Reference copies of the notebooks that launched the reported runs |
| `data/` | Empty: place the datasets here |

## Setup

Use Python 3.11 or newer with NumPy and a PyTorch build suited to your CPU or GPU (`requirements.txt`). Figures need Matplotlib (`requirements-analysis.txt`).

Place `FashionMNIST0.3.npz`, `FashionMNIST0.6.npz` and `CIFAR.npz` in `data/`. The alternate spelling `FashionMINIST` is also accepted, and `--data-dir` can point elsewhere. Run every command below from the package root.

## Quick check (a few minutes on CPU)

```text
python -m unittest discover -s tests -v
python -m unittest test_gce_addon -v
python algorithm/suite.py --output results/smoke --suite required --smoke --runs 2 --epochs 2 --sample-limit 600 --test-limit 300 --device cpu
python algorithm/verify.py --root results/smoke --replay --output results/smoke_audit.json
python algorithm/analyse.py --root results/smoke --output results/smoke_tables
```

Smoke results are marked non-reportable. Their scores are not meaningful.

## Reproduce the report

```text
python algorithm/suite.py --output results/main --device auto
python algorithm/verify.py --root results/main --replay --device auto --output results/main_audit.json
python algorithm/analyse.py --root results/main --output results/analysis --plots
python gce_addon.py train --output results/gce --device auto
python gce_addon.py verify --output results/gce --device auto
python gce_addon.py export --output results/gce --reference-dir results/main
```

The default extended suite has eight settings: known-T Fashion 0.3 and 0.6; trimmed estimates on all three datasets; and argmax estimates on all three. Each setting uses ten random 80/20 splits, with 20 base epochs on Fashion or 40 on CIFAR, plus 10 or 20 extra epochs for each refinement. The trimmed estimator is the predefined primary choice.

Add `--suite required` to `suite.py` for only known-T Fashion and trimmed-estimate CIFAR.

The report's diagnostics and pilot need the outputs above for the replay check and per-class accuracy:

```text
python algorithm/diagnostics.py --output results/diagnostics --reference-root results/main --gce-root results/gce
python algorithm/pilot.py --data-dir data --output results/pilot
```

Together they took about 1.5 hours on the L4.

| Run | Fit time on an NVIDIA L4 |
|---|---:|
| Extended suite | 94 minutes |
| Required suite | 38 minutes |
| GCE | 9 minutes |

Expect hours on a CPU. `--device auto` tests the GPU in a child process and falls back to the CPU if it fails.

## Colab notebooks (reference only)

The reported runs were launched on an NVIDIA L4 from the notebooks in `colab/`, which keep their saved outputs. They are a record of how the runs were launched and will not run elsewhere as they are, because they mount the authors' Google Drive and read code archives and results stored there. To reproduce the work, use the commands above.

The GCE notebook's last two code cells record a debugging step. A separate verification process did not use training's deterministic settings, so a replayed score differed slightly (79.2 against 79.17). Restoring those settings fixed this without changing any checkpoint or relaxing the check, and `gce_addon.py` now does it automatically.

| Notebook | What it launched | Code it used |
|---|---|---|
| `COMP4328_A2_Colab_v2.ipynb` | Main suite | Original core modules, before docstrings were added |
| `COMP4328_A2_GCE_Addon_v1.ipynb` | GCE fits | Same core modules, plus `gce_addon.py` before a later verifier fix that does not affect training |
| `COMP4328_A2_Tier23_Colab_v1.ipynb` | Pilot and diagnostics | Same core modules, plus `pilot.py` and `diagnostics.py` |

## Outputs

Each experiment directory contains:

- `manifest.json`: configuration, source and data hashes, and environment.
- `run_NNN/split.json`: split indices and training-only normalisation values.
- `run_NNN/METHOD.pt`: selected model, optimiser state, matrix and history.
- `run_NNN/result.json`: accuracies, matrices, anchors, selected epochs and timings.
- `results.json`: aggregate, updated after every completed run.

## Resuming and repeatability

Existing output directories are refused unless `--resume` is given, and a resume requires identical source, configuration, data and environment. Completed stages are reused; an interrupted stage restarts from its beginning. A lock file prevents simultaneous runners. After a hard crash, confirm the process has ended before removing a stale `.running.lock`.

Deterministic kernels are requested by default, which supports repeated runs in the same environment rather than identical results across hardware. `verify.py --replay` uses the CPU by default, and small CPU/GPU differences can occasionally change a prediction, so replay on the training device with `--device auto` or `--device cuda`.

## Source and the saved results

The reported runs used this code before docstrings, comments and documentation were tidied. Since then, `diagnostics.py` and `pilot.py` have been added, and no other executable statement has changed, as confirmed by comparing syntax trees with docstrings removed. Run on the same environment, the diagnostics reproduced every reported replayed score and selected epoch exactly. Saved manifests hash the source text, so `--resume` and `--reference-dir` will refuse those original outputs. Fresh runs are unaffected.
