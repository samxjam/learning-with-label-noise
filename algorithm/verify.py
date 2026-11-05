"""Audit saved experiment integrity and optionally replay every test prediction."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

try:
    from .main import verify_record, METHODS, choose_device
    from .artifacts import summarise, atomic_json
    from .analyse import experiment_files
    from .data import load_data, prepare_split, file_hash, DATA_DIR
    from .engine import accuracy, seed_everything
    from .models import CNN
    from .methods import validate_transition, matrix_error
except ImportError:
    from main import verify_record, METHODS, choose_device
    from artifacts import summarise, atomic_json
    from analyse import experiment_files
    from data import load_data, prepare_split, file_hash, DATA_DIR
    from engine import accuracy, seed_everything
    from models import CNN
    from methods import validate_transition, matrix_error


def verify(root, data_dir=DATA_DIR, replay=False, device="cpu"):
    """Audit saved experiments, optionally replaying every clean-test prediction.

    Checks identities, summaries, splits, matrices and checkpoint selection; raises on
    any mismatch and otherwise returns an audit record.
    """
    files = experiment_files(root)
    if not files:
        raise ValueError("No results found")
    torch.set_num_threads(4)
    audited = []
    for path in files:
        document = json.loads(path.read_text(encoding="utf-8"))
        manifest = json.loads((path.parent / "manifest.json").read_text(encoding="utf-8"))
        config = document["config"]
        assert config == manifest["definition"]["config"], "Configuration mismatch"
        assert document["signature"] == manifest["signature"], "Signature mismatch"
        assert document["summary"] == summarise(document["runs"]), "Summary mismatch"
        assert document["complete"] == (len(document["runs"]) == config["runs"]), "Completion flag mismatch"
        assert document["reportable"] == (not config["smoke"] and len(document["runs"]) >= 10), "Reportable flag mismatch"
        data = load_data(config["dataset"], data_dir)
        assert file_hash(data.path) == manifest["definition"]["data_sha256"], "Dataset changed"
        seen = set()
        for row in document["runs"]:
            seed = row["seed"]
            assert seed not in seen, "Duplicate split seed"
            seen.add(seed)
            directory = path.parent / f"run_{seed:03d}"
            verify_record(row, directory, document["signature"], seed)
            assert row == json.loads((directory / "result.json").read_text()), "Run record mismatch"
            tensors, split = prepare_split(data, seed, device, config["sample_limit"], config["test_limit"])
            assert split == json.loads((directory / "split.json").read_text()), "Split/preprocessing mismatch"
            for strategy in ("trimmed", "argmax"):
                estimate = row["estimators"][strategy]
                validate_transition(estimate["matrix"])
                if data.true_transition is not None:
                    np.testing.assert_allclose(estimate["relative_l1_error"], matrix_error(estimate["matrix"], data.true_transition))
            for method in METHODS:
                checkpoint = torch.load(directory / f"{method}.pt", map_location="cpu", weights_only=True)
                history = checkpoint["history"]
                selected = min(history, key=lambda item: item["validation_loss"])
                assert selected["epoch"] == checkpoint["selected_epoch"], "Best epoch not restored"
                assert selected["validation_loss"] == checkpoint["validation_loss"], "Best validation loss mismatch"
                assert checkpoint["selected_epoch"] == row["selected_epochs"][method]
                if method in ("continued", "continued_full", "trevision"):
                    assert history[0]["epoch"] == 0, "Missing initial refinement checkpoint"
                    assert checkpoint["validation_loss"] <= history[0]["validation_loss"], "Refinement worsened selection criterion"
                if replay:
                    seed_everything(0, config["deterministic"])
                    model = CNN(tensors["x_test"].shape[1]).to(device)
                    model.load_state_dict(checkpoint["model_state"])
                    actual = accuracy(model, tensors["x_test"], tensors["y_test"])
                    np.testing.assert_allclose(actual, row["acc"][method], atol=1e-5, rtol=0)
            print(f"Verified {path.parent.name} seed {seed}", flush=True)
        audited.append({"experiment": path.parent.name, "runs": len(seen), "methods": list(METHODS),
                        "test_predictions_replayed": replay, "reportable": document["reportable"], "complete": document["complete"]})
    return {"passed": True, "experiments": audited}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--replay", action="store_true")
    parser.add_argument("--device", choices=("cpu", "auto", "cuda"), default="cpu")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    atomic_json(args.output, verify(args.root, args.data_dir, args.replay, choose_device(args.device)))
