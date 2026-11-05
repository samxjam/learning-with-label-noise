"""Load assignment data and fit preprocessing on each training split only."""
from dataclasses import dataclass
from pathlib import Path
import hashlib

import numpy as np
import torch

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
T_03 = np.array([[.7, .3, 0.], [0., .7, .3], [.3, 0., .7]])
T_06 = np.array([[.4, .3, .3], [.3, .4, .3], [.3, .3, .4]])
DATASETS = {
    "fashion0.3": (["FashionMNIST0.3.npz", "FashionMINIST0.3.npz"], T_03),
    "fashion0.6": (["FashionMNIST0.6.npz", "FashionMINIST0.6.npz"], T_06),
    "cifar": (["CIFAR.npz"], None),
}


def file_hash(path):
    """Return the SHA-256 hex digest of a file."""
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


@dataclass
class Dataset:
    """Images in [0, 1], noisy training labels, clean test labels and the true T if known."""

    images: np.ndarray
    noisy_labels: np.ndarray
    test_images: np.ndarray
    test_labels: np.ndarray
    true_transition: np.ndarray | None
    path: Path


def load_data(name, directory=DATA_DIR):
    """Images become float32 NCHW in [0,1]; labels are never modified."""
    if name not in DATASETS:
        raise ValueError(f"Unknown dataset: {name}")
    names, transition = DATASETS[name]
    path = next((Path(directory) / n for n in names if (Path(directory) / n).is_file()), None)
    if path is None:
        raise FileNotFoundError(f"Expected one of {names} in {Path(directory).resolve()}")
    with np.load(path, allow_pickle=False) as archive:
        x, xt = (archive[k].astype(np.float32) for k in ("Xtr", "Xts"))
        labels = []
        for key in ("Str", "Yts"):
            raw = archive[key]
            if raw.ndim != 1 or not np.isin(raw, [0, 1, 2]).all():
                raise ValueError(f"{key} must be a 1-D vector of labels 0, 1, 2")
            labels.append(raw.astype(np.int64))
    if len(x) != len(labels[0]) or len(xt) != len(labels[1]) or min(len(x), len(xt)) < 2:
        raise ValueError("Image and label lengths disagree or dataset is too small")
    for values in (x, xt):
        if not np.isfinite(values).all() or values.min() < 0 or values.max() > 255:
            raise ValueError("Expected finite image intensities between 0 and 255")
    if name == "cifar":
        if x.shape[1:] != (32, 32, 3) or xt.shape[1:] != (32, 32, 3):
            raise ValueError("Expected CIFAR images shaped (N,32,32,3)")
        x, xt = x.transpose(0, 3, 1, 2), xt.transpose(0, 3, 1, 2)
    else:
        x, xt = x.reshape(-1, 1, 28, 28), xt.reshape(-1, 1, 28, 28)
    return Dataset(np.ascontiguousarray(x / 255), labels[0],
                   np.ascontiguousarray(xt / 255), labels[1], transition, path.resolve())


def prepare_split(data, seed, device="cpu", sample_limit=None, test_limit=None):
    """Random 80/20 split; optional limits are exclusively for labelled smoke runs.

    A separate RNG selects a smoke subset, so its selection does not consume the
    split RNG. Normalisation is a scalar mean/std fitted on training images.
    """
    indices = np.arange(len(data.images))
    if sample_limit is not None:
        indices = np.random.default_rng(9182).permutation(indices)[:sample_limit]
    permutation = np.random.default_rng(seed).permutation(indices)
    boundary = int(.8 * len(permutation))
    train_ids, validation_ids = permutation[:boundary], permutation[boundary:]
    if min(len(train_ids), len(validation_ids)) < 1:
        raise ValueError("Both train and validation splits must be nonempty")
    training = data.images[train_ids]
    mean = float(training.mean(dtype=np.float64))
    std = float(training.std(dtype=np.float64))
    if not np.isfinite(std) or std <= 1e-12:
        raise ValueError("Training images have zero or invalid variance")
    test_ids = np.arange(len(data.test_images))
    if test_limit is not None:
        test_ids = np.random.default_rng(9183).permutation(test_ids)[:test_limit]

    def images(a):
        """Apply the training-split normalisation and move images to the device."""
        return torch.as_tensor(np.ascontiguousarray((a - mean) / std), device=device)

    tensors = {
        "x_train": images(training),
        "s_train": torch.as_tensor(data.noisy_labels[train_ids], device=device),
        "x_validation": images(data.images[validation_ids]),
        "s_validation": torch.as_tensor(data.noisy_labels[validation_ids], device=device),
        "x_test": images(data.test_images[test_ids]),
        "y_test": torch.as_tensor(data.test_labels[test_ids], device=device),
    }
    metadata = {"seed": seed, "mean": mean, "std": std,
                "train_indices": train_ids.tolist(), "validation_indices": validation_ids.tolist(),
                "test_indices": test_ids.tolist(), "normalisation_fit": "training_split_only"}
    return tensors, metadata
