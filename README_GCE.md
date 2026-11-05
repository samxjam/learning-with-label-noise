# Generalised Cross Entropy (GCE)

`gce_addon.py` trains GCE on the three datasets, with ten splits each. GCE uses no transition matrix, so each dataset needs only these 30 fits, and one dataset's GCE result is compared against all of its matrix settings.

## Method

Plain GCE: `L_q(z,s) = (1 - softmax(z)[s]**q) / q`, with `q=0.7` fixed from the original paper's experimental setting and not tuned on test accuracy. It is computed as `-expm1(q*log_softmax(z)[s])/q` for numerical stability. This is not the paper's truncated variant.

Source: Zhang & Sabuncu, *Generalized Cross Entropy Loss for Training Deep Neural Networks with Noisy Labels*, NeurIPS 2018, Eq. (6) and Section 4.1: https://papers.nips.cc/paper/2018/file/f2925f97bc13ad2852a7a551802feea0-Paper.pdf

The network, 80/20 splits, training-only normalisation, initialisation seed `10000+split`, training seed `20000+split`, batch size 128, Adam (learning rate 0.001, weight decay 0.0001) and CIFAR flips all match the base methods. Budgets are 20 Fashion epochs and 40 CIFAR epochs.

Checkpoint selection minimises GCE on noisy validation labels, which differs from the CE and forward-CE criteria of the other methods. Noisy CE is logged as a diagnostic. Clean test labels are used only after selection.

GCE can reduce the influence of low-confidence noisy targets, but it can also suppress hard clean examples.

## Commands

Run from the package root with the datasets in `data/`:

```text
python -m unittest test_gce_addon -v
python gce_addon.py train --output results/gce --device auto
python gce_addon.py verify --output results/gce --device auto
```

Training does not need the main-suite results. To write paired comparisons, complete the main suite first, then run:

```text
python gce_addon.py export --output results/gce --reference-dir results/main
```

Splits, preprocessing and base budgets must match. Environment differences are recorded in `comparison_environment.json`.

For a quick check, add `--smoke --runs 2 --epochs 2 --sample-limit 600 --test-limit 300` and use a separate output path. Smoke results are not reportable.

## Recovery

After every epoch, `progress.pt` records the latest model, Adam state, random generator states, history and best model. Add `--resume` to continue from the last saved epoch using the same code, configuration, environment and device type. Completed runs are verified and skipped.

## Outputs

Each run saves its selected and latest checkpoints, history and score. Each dataset gets a summary with sample SD, and `audit.json` records the replay of all GCE test predictions. Export writes `combined_accuracy.csv`, `paired_gce.csv`, `comparison.md` and `comparison_environment.json`.

Verification sets the same deterministic kernels as training, because a prediction near a class boundary can otherwise change on NVIDIA hardware. The saved accuracy must still match.
