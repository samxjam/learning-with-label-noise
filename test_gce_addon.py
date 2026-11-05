"""Scientific, exact-resume and persistence regressions for the GCE add-on."""
import argparse
import contextlib
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
import torch.nn.functional as F

# In the distributed package, algorithm/ is next to this file. The second
# path permits development against the frozen local solution without copying it.
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.append(str(Path(__file__).resolve().parents[1] / "solution"))
from gce_addon import gce_loss, fit_gce, train, verify, export, check_reference, load_references, PRIMARY
from algorithm.engine import seed_everything
from algorithm.artifacts import atomic_json, source_identity, statistics
from algorithm.data import Dataset, prepare_split, file_hash
from algorithm.engine import accuracy as actual_accuracy


class GCETests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_formula_endpoints_and_extreme_logits(self):
        x = torch.tensor([[.2, -.3, .7], [-.8, .4, .5]], dtype=torch.double)
        y = torch.tensor([2, 0])
        torch.testing.assert_close(gce_loss(x, y, 1), (1 - x.softmax(1).gather(1, y[:, None])).mean())
        torch.testing.assert_close(gce_loss(x, y, 1e-8), F.cross_entropy(x, y), atol=1e-7, rtol=1e-7)
        extreme = torch.tensor([[1000., -1000., 0.]], requires_grad=True)
        loss = gce_loss(extreme, torch.tensor([1]), .7)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(torch.isfinite(extreme.grad).all())
        for bad in (0, -1, 2, float("nan")):
            with self.assertRaises(ValueError):
                gce_loss(x, y, bad)

    def test_gradient_and_confidence_weighting(self):
        x = torch.tensor([[.2, -.3, .7], [-.8, .4, .5]], dtype=torch.double, requires_grad=True)
        y = torch.tensor([2, 0])
        self.assertTrue(torch.autograd.gradcheck(lambda z: gce_loss(z, y, .7), (x,)))
        a = torch.autograd.grad(gce_loss(x, y, .7), x, retain_graph=True)[0]
        b = torch.autograd.grad(F.cross_entropy(x, y), x)[0]
        confidence = x.softmax(1).gather(1, y[:, None]).detach() ** .7
        torch.testing.assert_close(a, b * confidence)

    def test_exact_epoch_recovery_with_dropout_batchnorm_and_flips(self):
        device = os.environ.get("GCE_TEST_DEVICE", "cpu")
        seed_everything(5)
        x = torch.rand(12, 1, 4, 4, device=device)
        y = torch.arange(12, device=device) % 3
        tensors = {"x_train": x, "s_train": y, "x_validation": x, "s_validation": y}

        def model():
            seed_everything(10000)
            return torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(16, 8), torch.nn.BatchNorm1d(8),
                                       torch.nn.ReLU(), torch.nn.Dropout(.35), torch.nn.Linear(8, 3)).to(device)

        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as temporary:
            root = Path(temporary)
            common = dict(epochs=3, seed=0, batch_size=4, lr=.001, weight_decay=.0001, flip=True,
                          q=.7, signature="fixed-identity")
            with contextlib.redirect_stdout(io.StringIO()):
                uninterrupted = fit_gce(model(), tensors, progress_path=root / "full.pt", **common)
                with self.assertRaises(InterruptedError):
                    fit_gce(model(), tensors, progress_path=root / "resume.pt", stop_after=1, **common)
                resumed = fit_gce(model(), tensors, progress_path=root / "resume.pt", **common)
            self.assertEqual(uninterrupted["history"], resumed["history"])
            self.assertEqual(uninterrupted["selected_epoch"], resumed["selected_epoch"])
            for key in uninterrupted["model_state"]:
                torch.testing.assert_close(uninterrupted["model_state"][key], resumed["model_state"][key], atol=0, rtol=0)
            full, resumed = [torch.load(root / name, weights_only=True) for name in ("full.pt", "resume.pt")]
            for key in full["latest_model"]:
                torch.testing.assert_close(full["latest_model"][key], resumed["latest_model"][key], atol=0, rtol=0)
            for key in ("batch_rng", "torch_rng"):
                self.assertTrue(torch.equal(full[key], resumed[key]))

    def test_training_interface_rejects_test_labels(self):
        with self.assertRaisesRegex(ValueError, "training and validation"):
            fit_gce(torch.nn.Linear(2, 3), {"y_test": torch.tensor([0])}, epochs=1, seed=0,
                    batch_size=1, lr=.001, weight_decay=0, flip=False, q=.7,
                    progress_path=Path("unused"), signature="unused")

    def test_runner_roundtrip_skip_resume_and_identity(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as temporary:
            root = Path(temporary)
            rng = np.random.default_rng(19)
            np.savez(root / "FashionMNIST0.3.npz", Xtr=rng.integers(0, 256, (30, 784), dtype=np.uint8),
                     Str=np.arange(30) % 3, Xts=rng.integers(0, 256, (9, 784), dtype=np.uint8), Yts=np.arange(9) % 3)
            args = argparse.Namespace(smoke=True, runs=1, epochs=2, sample_limit=None, test_limit=None,
                                      reference_dir=None, device="cpu", threads=2, dataset="fashion0.3",
                                      q=.7, output=root / "experiment", data_dir=root, resume=False)
            with contextlib.redirect_stdout(io.StringIO()):
                first = train(args)
                audit = verify(args)
            self.assertTrue(audit["passed"])
            self.assertFalse(audit["experiments"][0]["reportable"])
            with self.assertRaises(FileExistsError):
                train(args)
            args.resume = True
            with contextlib.redirect_stdout(io.StringIO()), patch("gce_addon.fit_gce", side_effect=AssertionError("must skip")):
                self.assertEqual(first, train(args))
            # A crash after the final epoch but before committing the run does
            # not require repeating that epoch or any training step.
            (args.output / "fashion0.3/run_000/result.json").unlink()
            with contextlib.redirect_stdout(io.StringIO()), patch("gce_addon.gce_loss", side_effect=AssertionError("must not train")):
                recovered = train(args)
            self.assertEqual(first["fashion0.3"]["runs"][0]["accuracy"], recovered["fashion0.3"]["runs"][0]["accuracy"])
            args.q = .8
            with self.assertRaisesRegex(ValueError, "resume refused"):
                train(args)

    def test_verify_restores_training_backend_settings_in_fresh_process(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as temporary:
            root = Path(temporary)
            rng = np.random.default_rng(22)
            np.savez(root / "FashionMNIST0.3.npz", Xtr=rng.integers(0, 256, (30, 784), dtype=np.uint8),
                     Str=np.arange(30) % 3, Xts=rng.integers(0, 256, (9, 784), dtype=np.uint8), Yts=np.arange(9) % 3)
            args = argparse.Namespace(smoke=True, runs=1, epochs=1, sample_limit=None, test_limit=None,
                                      reference_dir=None, device="cpu", threads=2, dataset="fashion0.3",
                                      q=.7, output=root / "experiment", data_dir=root, resume=False)
            with contextlib.redirect_stdout(io.StringIO()):
                train(args)
            # Reproduce a new verifier process's defaults rather than inheriting
            # settings left behind by training in the same unit-test process.
            torch.use_deterministic_algorithms(False)
            torch.backends.cudnn.deterministic = False
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cuda.matmul.allow_tf32 = True

            def check_settings(*values):
                self.assertTrue(torch.are_deterministic_algorithms_enabled())
                self.assertTrue(torch.backends.cudnn.deterministic)
                self.assertFalse(torch.backends.cudnn.allow_tf32)
                self.assertFalse(torch.backends.cuda.matmul.allow_tf32)
                return actual_accuracy(*values)

            try:
                with contextlib.redirect_stdout(io.StringIO()), patch("gce_addon.accuracy", side_effect=check_settings):
                    self.assertTrue(verify(args)["passed"])
            finally:
                seed_everything(0)

    def test_reference_alignment_and_paired_export_without_old_training(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as temporary:
            root = Path(temporary)
            data_path = root / "data.bin"
            data_path.write_bytes(b"fixed-dataset")
            images = np.arange(30 * 4, dtype=np.float32).reshape(30, 1, 2, 2)
            data = Dataset(images, np.arange(30) % 3, images[:9], np.arange(9) % 3, None, data_path)
            reference_dir, output = root / "references", root / "gce"
            configurations = {}
            for name, primary in PRIMARY.items():
                cfg = {"dataset": name, "epochs": 40 if name == "cifar" else 20, "batch_size": 128,
                       "lr": .001, "weight_decay": .0001, "flip": name == "cifar", "smoke": False,
                       "q": .7, "revision_epochs": 20 if name == "cifar" else 10}
                configurations[name] = cfg
                baseline_runs = [{"seed": seed, "acc": {"ce": 75. + seed, "forward": 78. + seed}} for seed in range(10)]
                baseline = {"signature": name, "complete": True, "reportable": True, "config": cfg,
                            "runs": baseline_runs, "summary": {"acc": {method: statistics([r["acc"][method] for r in baseline_runs])
                                                                      for method in ("ce", "forward")}}}
                atomic_json(reference_dir / primary / "results.json", baseline)
                atomic_json(reference_dir / primary / "manifest.json", {"signature": name, "definition": {
                    "config": cfg, "source": source_identity(), "data_sha256": file_hash(data_path),
                    "environment": {"torch": "previous-environment"}}})
                records = [{"seed": seed, "accuracy": 80. + seed} for seed in range(10)]
                atomic_json(output / name / "results.json", {"complete": True, "reportable": True,
                            "config": cfg, "runs": records, "summary": statistics([r["accuracy"] for r in records])})
                for seed in range(10):
                    _, split = prepare_split(data, seed)
                    atomic_json(reference_dir / primary / f"run_{seed:03d}" / "split.json", split)
                    atomic_json(output / name / f"run_{seed:03d}" / "split.json", split)
            atomic_json(output / "manifest.json", {"definition": {"configurations": configurations,
                        "environment": {"torch": "current-environment"}}})
            atomic_json(output / "audit.json", {"passed": True, "experiments": [
                {"dataset": n, "runs": 10, "reportable": True, "test_predictions_replayed": True} for n in PRIMARY]})
            references = load_references(reference_dir)
            _, split = prepare_split(data, 0)
            reference = references[PRIMARY["fashion0.3"]]
            check_reference(data, split, configurations["fashion0.3"], reference)
            with self.assertRaisesRegex(ValueError, "normalisation differs"):
                check_reference(data, {**split, "mean": split["mean"] + 1}, configurations["fashion0.3"], reference)
            rows = export(argparse.Namespace(output=output, reference_dir=reference_dir))
            self.assertEqual(len(rows), 9)
            import csv
            with (output / "analysis/paired_gce.csv").open() as stream:
                pairs = list(csv.DictReader(stream))
            self.assertTrue(all(float(r["mean"]) == (5 if r["comparison"] == "gce_minus_ce" else 2) for r in pairs))
            changed = json.loads((output / "analysis/comparison_environment.json").read_text())
            self.assertTrue(all(r["changed_environment_fields"] == ["torch"] for r in changed))


if __name__ == "__main__":
    unittest.main()
