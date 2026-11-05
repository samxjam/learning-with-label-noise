# Definitions and experimental protocol

## Notation and orientation

There are K=3 classes. X is an image, Y is its unobserved clean training label, and S is the supplied noisy training label. The CNN produces logits z_theta(X) and row-vector probabilities p_theta(X)=softmax(z_theta(X)). We use T_ij=P(S=j | Y=i): each row is nonnegative and sums to one. The noisy posterior is q_theta(X)=p_theta(X) T. This follows the matrices displayed in the assignment, which are the transpose of the lecture notes' column convention, so they are used here without transposing.

T_hat is estimated from a CE model's noisy posteriors. T_start is either the given matrix or T_hat, according to the experiment. R denotes the learned revision matrix; Delta=R−T_start is saved for interpretation. The Euclidean row-simplex projection enforces R_ij≥0 and sum_j R_ij=1 after each optimiser step.

## Implemented objectives

For each noisy training example (x,s), CE is −log p_s. Forward correction is −log (pT)_s. Fixed-T reweighting uses the numerical loss [p_s/(pT)_s] CE but stops the gradient through p in the ratio. This is an explicit surrogate update; it is not described as the full derivative of that scalar expression.

Revision minimises [p_s/(pR)_s] CE, differentiating both factors through the network and the denominator through R. R is updated by Adam followed by Euclidean projection of each row onto the probability simplex. Optimising R directly is equivalent to an additive Delta parameterisation with R=T_start+Delta and the same constraint. Both prediction losses use a denominator/log floor of 1e−8 for numerical protection.

The revision stage is inspired by Xia et al., *Are Anchor Points Really Indispensable in Label-Noise Learning?* (NeurIPS 2019), Eq. (3) and Algorithm 1. It learns a classifier and a transition correction after initial estimation and weighted pretraining. Our explicit adaptations are simplex projection, noisy forward-CE checkpoint selection, epoch-zero eligibility, and the optionally trimmed anchor initializer. This is a documented implementation variant, not a claim of exact reference-code reproduction. [Paper](https://proceedings.neurips.cc/paper_files/paper/2019/file/9308b0d6e5898366a4a986bc33f3d3e7-Paper.pdf).

Forward correction follows the noisy-posterior likelihood construction described by Patrini et al., *Making Deep Neural Networks Robust to Label Noise: A Loss Correction Approach* (CVPR 2017). It is also covered in the supplied course materials. [Paper](https://openaccess.thecvf.com/content_cvpr_2017/html/Patrini_Making_Deep_Neural_CVPR_2017_paper.html).

Consistency arguments require assumptions about the noise process and the posterior/matrix estimates; they do not guarantee that this finite-data CNN with a jointly estimated matrix reaches the best clean accuracy or recovers the true matrix.

## Direct comparison

| Method | How it uses noise information | Strength | Limitation |
|---|---|---|---|
| CE | Treats supplied labels as targets | Simple baseline; provides noisy posteriors for estimation | Can fit label corruption |
| Forward | Uses a fixed T in noisy-label likelihood | No matrix inversion; directly models the corruption process | Depends on T and can be weakly informative at high noise |
| Reweight | Fixed T; detached posterior-ratio weights | Clean-risk motivation; simple stop-gradient update | Estimated weights introduce error; surrogate gradient must be disclosed |
| T-revision variant | Jointly optimises the classifier and constrained R | Can adapt an inaccurate initial matrix | Joint estimation can drift; benefit and identifiability are not guaranteed |
| GCE | Uses no matrix; down-weights low-confidence targets | No transition estimate needed | May also suppress hard clean examples; see `README_GCE.md` |

## Transition estimators

First train a CE classifier using noisy training/validation labels. Obtain its softmax outputs on training images only. Under the anchor assumption, an example with clean class probability near one provides a noisy posterior approximating the corresponding T row. This also needs adequate noisy-posterior estimation and a meaningful association between high class confidence and clean class identity.

Two predefined strategies are implemented:

- `trimmed`: average posterior vectors whose class-i confidence lies between the empirical 96th and 98th percentiles. This fixed heuristic was chosen before the final experiments and not tuned. It reduces reliance on a single extreme prediction but may introduce bias; it is not the exact argmax anchor method.
- `argmax`: use the posterior vector with maximal class-i probability. This directly tests the high-confidence-anchor alternative, but may be sensitive to an overconfident outlier.

Anchor counts, thresholds, local training indices and rare tiny-sample empty-band fallback are saved. Both estimators are evaluated for matrix recovery on Fashion, whose truth is given. The extended suite also trains classifiers with both estimators. CIFAR's true matrix is unknown: no CIFAR matrix-error score is invented. The primary estimator is fixed in advance; clean-test accuracy is not used to select one.

## Splits, preprocessing and optimisation

For each seed, randomly permute all supplied noisy examples and take floor(0.8N) for training and the remainder for validation. Fit one scalar mean and population standard deviation on the training images after dividing intensities by 255. Apply those same statistics to validation and test. No test or validation feature is used to fit them.

All methods share one three-convolution CNN: 32/64/128 channels, batch normalisation/ReLU, two max pools, global average pooling, dropout 0.3, and three logits. CIFAR uses training-only horizontal flips. Inputs remain unmodified outside their batch copies; labels are never additionally corrupted.

Base training uses Adam, learning rate 0.001, weight decay 1e−4, batch size 128, 20 epochs for Fashion and 40 for CIFAR. Refinements use a fresh Adam optimiser, network learning rate 0.0002, matrix learning rate 0.0005 with no matrix weight decay, and half as many additional epochs (at least one). The network/matrix pair from the best noisy-validation epoch is restored. CE uses noisy CE validation; robust methods use forward CE with their current matrix. Refinements also consider their starting epoch-zero checkpoint.

Per split seed r: initialisation seed is 10000+r, base training seed is 20000+r and refinement seed is 30000+r. All base models share initial weights and random streams; the refinement branches share their starting checkpoint and stream. Deterministic kernels are requested by default, and software/device identities are saved. This supports repeated execution in the same environment, not a universal cross-hardware bitwise guarantee.

## Controlled refinements

All three refinements start from the exact same selected reweight model and run the same extra epochs with the same network learning rate:

1. `continued`: fixed T, original detached-weight update.
2. `continued_full`: fixed T, fully differentiated weighted objective.
3. `trevision`: learned R, fully differentiated weighted objective.

Revision minus `continued_full` isolates allowing the matrix to learn under matched objectives and budgets. `continued_full` minus `continued` measures the gradient-rule change. Revision minus `continued` measures the combined change. Paired differences use the same split seeds. The report tests key differences with an exact two-sided sign-flip test over the ten splits; the p-values are approximate because splits overlap, and are uncorrected for multiple comparisons.

## Evaluation and interpretation

Clean test labels enter only final accuracy evaluation, after all stages in a run have selected their checkpoints. Results are top-1 percentages with sample SD across ten independently generated splits. Repeated random holdouts can overlap; their SD is not a confidence interval or a guarantee about population performance.

Estimated and revised matrices are saved as means with per-entry sample SD. Fashion matrix accuracy is the relative entrywise L1 error sum_ij|T_est−T_true| / sum_ij|T_true|. Given-T runs start with zero matrix error, so only estimated-T Fashion runs give a meaningful before/after estimation comparison.

A lower noisy-validation loss does not guarantee improved clean-test accuracy, and a lower matrix error does not guarantee that every classifier improves. The provided high-noise Fashion matrix is more ill-conditioned; this suggests sensitivity but does not determine the empirical ranking.

## Blind training-settings pilot

After the main study, `algorithm/pilot.py` compared Forward under a cosine learning-rate schedule, random crops, weight averaging and ResNet-20 with the baseline, on three splits. It deletes clean test tensors immediately after each split, so it can only use noisy validation.

The adoption rule was fixed before any pilot result existed. A candidate passes only if, against the baseline: on CIFAR and Fashion 0.6 its mean selected noisy-validation forward CE is lower, its selected loss is lower in at least two of three splits, and its mean noisy-validation accuracy is not lower; and on Fashion 0.3 its mean selected loss is not worse by more than 0.002. Among passing candidates, take the simplest in the order cosine, crops, weight averaging, ResNet-20, unless a more complex one lowers mean CIFAR validation loss by at least 0.002 more. No candidate passed, so the reported settings were kept.

## Diagnostics

`algorithm/diagnostics.py` replays RW and its refinements through the unchanged `fit()`, logging observed-label probabilities, the learned matrix and clean-test accuracy each epoch. Clean-test values are logged after the fact and never affect selection. In the reported environment, the replay reproduced every reported score and selected epoch exactly. The same script compares estimator variants without clean labels and computes per-class accuracy from saved checkpoints.
