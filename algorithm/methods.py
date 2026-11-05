"""Losses and estimators under T[i,j] = P(noisy=j | clean=i).

Rows of T sum to one; row-vector posteriors satisfy q = p @ T. See
METHODS.md for sources, assumptions and the disclosed T-revision adaptations.
"""
import numpy as np
import torch
import torch.nn.functional as F


def validate_transition(matrix):
    """Raise unless the matrix is finite, nonnegative, 3x3 and row-stochastic."""
    values = torch.as_tensor(matrix)
    if (values.shape != (3, 3) or not torch.isfinite(values).all()
            or (values < 0).any()
            or not torch.allclose(values.sum(1), torch.ones(3, dtype=values.dtype,
                                                          device=values.device), atol=1e-6, rtol=1e-6)):
        raise ValueError("Transition matrix must be finite, nonnegative, 3x3 and row-stochastic")


def forward_loss(logits, labels, transition):
    """Noisy-label negative log likelihood with forward-corrected probabilities."""
    noisy = logits.softmax(dim=1) @ transition
    return F.nll_loss(noisy.clamp_min(1e-8).log(), labels)


def reweight_loss(logits, labels, transition, *, detach_weights=True):
    """Posterior-ratio weighted CE.

    Fixed-T reweighting uses a stop-gradient surrogate. Revision explicitly
    sets detach_weights=False so both the ratio and CE differentiate through
    the network, and the denominator differentiates through T.
    """
    probabilities = logits.softmax(dim=1)
    if detach_weights:
        probabilities = probabilities.detach()
    clean = probabilities.gather(1, labels[:, None]).squeeze(1)
    noisy = (probabilities @ transition).gather(1, labels[:, None]).squeeze(1)
    return (clean / noisy.clamp_min(1e-8) * F.cross_entropy(logits, labels, reduction="none")).mean()


@torch.no_grad()
def project_transition(matrix):
    """Euclidean projection of each row onto the probability simplex.

    Sorting identifies the active coordinates; unlike clamp/row-normalise,
    this defines a valid distribution even for a zero or all-negative row.
    Projection is an optimiser constraint, not differentiated through.
    """
    if matrix.ndim != 2 or not torch.isfinite(matrix).all():
        raise ValueError("Cannot project a non-finite or non-matrix value")
    sorted_values = matrix.sort(dim=1, descending=True).values
    # Floating-point CUDA cumsum can reject strict deterministic mode. There
    # are only three classes: explicit prefix reductions avoid that kernel.
    cumulative = torch.stack([sorted_values[:, :end].sum(dim=1)
                              for end in range(1, matrix.shape[1] + 1)], dim=1) - 1
    counts = torch.arange(1, matrix.shape[1] + 1, device=matrix.device, dtype=matrix.dtype)
    active = sorted_values - cumulative / counts > 0
    rho = active.sum(dim=1, keepdim=True).clamp_min(1)
    theta = cumulative.gather(1, rho - 1) / rho
    result = (matrix - theta).clamp_min(0)
    return result / result.sum(dim=1, keepdim=True)


def estimate_transition(probabilities, strategy="trimmed"):
    """Estimate rows using CE noisy posteriors on training examples only.

    trimmed: mean of the 96th--98th percentile band (original heuristic).
    argmax: posterior of the most confident example for each class.
    Also returns the selected local row indices and confidence thresholds.
    """
    p = np.asarray(probabilities, dtype=np.float64)
    if (p.ndim != 2 or p.shape[1] != 3 or len(p) == 0 or not np.isfinite(p).all()
            or (p < 0).any() or not np.allclose(p.sum(1), 1, atol=1e-6)):
        raise ValueError("Expected a nonempty array of three-class probability rows")
    rows, diagnostics = [], []
    for klass in range(3):
        fallback = False
        if strategy == "argmax":
            anchors = np.array([int(np.argmax(p[:, klass]))])
            low = high = float(p[anchors[0], klass])
        elif strategy == "trimmed":
            low, high = np.percentile(p[:, klass], [96, 98])
            anchors = np.flatnonzero((p[:, klass] >= low) & (p[:, klass] <= high))
            if len(anchors) == 0:
                # Interpolated percentiles can select no sample in tiny smoke inputs.
                anchors = np.array([int(np.argmin(np.abs(p[:, klass] - (low + high) / 2)))])
                fallback = True
        else:
            raise ValueError(f"Unknown estimator strategy: {strategy}")
        rows.append(p[anchors].mean(axis=0))
        diagnostics.append({"class": klass, "count": len(anchors), "low": float(low),
                            "high": float(high), "indices_in_training_split": anchors.tolist(),
                            "empty_band_fallback": fallback})
    matrix = np.stack(rows)
    matrix /= matrix.sum(axis=1, keepdims=True)
    validate_transition(matrix)
    return matrix, diagnostics


def matrix_error(estimate, truth):
    """Relative entrywise L1 error; unavailable for CIFAR's unknown truth."""
    return float(np.abs(np.asarray(estimate) - truth).sum() / np.abs(truth).sum())
