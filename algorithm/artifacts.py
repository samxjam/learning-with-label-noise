"""Atomic experiment artefacts, strict JSON and content-based identities."""
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import uuid

import numpy as np
import torch


def atomic_json(path, value):
    """Write strict JSON through a temporary file so readers never see a partial file."""
    path = Path(path)
    payload = json.dumps(value, indent=2, allow_nan=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_checkpoint(path, value):
    """Save a torch checkpoint through a temporary file, replacing the target atomically."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        torch.save(value, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def source_identity():
    """Hash every module in this package; recorded so results can be tied to their source."""
    root = Path(__file__).resolve().parent
    digest = hashlib.sha256()
    files = {}
    for path in sorted(root.glob("*.py")):
        contents = path.read_bytes()
        files[path.name] = hashlib.sha256(contents).hexdigest()
        digest.update(path.name.encode() + b"\0" + contents)
    return {"sha256": digest.hexdigest(), "files": files}


def environment(device):
    """Describe the software and hardware used by a run."""
    return {"python": sys.version, "platform": platform.platform(), "numpy": np.__version__,
            "torch": str(torch.__version__), "device": str(device), "hip": torch.version.hip,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(device) if str(device).startswith("cuda") else None}


def identity(value):
    """Return a stable SHA-256 of a JSON-serialisable experiment definition."""
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def statistics(values):
    """Return mean, sample SD (ddof=1; None for one value) and count, elementwise."""
    array = np.asarray(values, dtype=np.float64)
    if len(array) == 0 or not np.isfinite(array).all():
        raise ValueError("Cannot summarise empty or non-finite values")
    return {"mean": array.mean(axis=0).tolist(),
            "std": array.std(axis=0, ddof=1).tolist() if len(array) > 1 else None,
            "n": len(array)}


def summarise(records):
    """Aggregate run records: accuracies, matrices, matrix errors and paired differences."""
    if not records:
        return {"completed_runs": 0, "acc": {}}
    methods = records[0]["acc"]
    output = {"completed_runs": len(records),
              "acc": {m: statistics([r["acc"][m] for r in records]) for m in methods}}
    for key in ("T_hat", "T_used", "T_revised", "T_error", "T_used_error", "T_revised_error"):
        if key in records[0]:
            output[key] = statistics([r[key] for r in records])
    output["paired_revision_minus_continued_accuracy"] = statistics(
        [r["acc"]["trevision"] - r["acc"]["continued"] for r in records])
    output["paired_revision_minus_full_gradient_fixed_T"] = statistics(
        [r["acc"]["trevision"] - r["acc"]["continued_full"] for r in records])
    output["paired_full_gradient_minus_detached_fixed_T"] = statistics(
        [r["acc"]["continued_full"] - r["acc"]["continued"] for r in records])
    return output
