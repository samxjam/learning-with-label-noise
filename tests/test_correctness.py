"""Meaningful regression tests for the scientific and persistence defects found."""
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from algorithm.data import Dataset, prepare_split, T_03, T_06
from algorithm.methods import (forward_loss, reweight_loss, project_transition,
                               validate_transition, estimate_transition)
from algorithm.artifacts import atomic_json, statistics, summarise
from algorithm.engine import fit, seed_everything
from algorithm.main import parser, build_config, run_experiment, checked_checkpoint, choose_device


class ScientificTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_asymmetric_transition_direction(self):
        matrix = torch.tensor(T_03)
        self.assertTrue(torch.equal(torch.tensor([[1., 0., 0.]], dtype=torch.double) @ matrix,
                                    torch.tensor([[.7, .3, 0.]], dtype=torch.double)))
        logits = torch.tensor([[1., 2., -1.]], dtype=torch.double)
        expected = -(logits.softmax(1) @ matrix)[0, 1].log()
        torch.testing.assert_close(forward_loss(logits, torch.tensor([1]), matrix), expected)

    def test_identity_matrix_losses_reduce_to_ce(self):
        logits = torch.tensor([[.4, -.2, .6], [-.7, .3, .9]], dtype=torch.double)
        y = torch.tensor([2, 0])
        for detach in (False, True):
            torch.testing.assert_close(reweight_loss(logits, y, torch.eye(3, dtype=torch.double),
                                                     detach_weights=detach), F.cross_entropy(logits, y))
        torch.testing.assert_close(forward_loss(logits, y, torch.eye(3, dtype=torch.double)),
                                   F.cross_entropy(logits, y))

    def test_full_revision_gradient_network_and_matrix(self):
        logits = torch.tensor([[.3, -.2, .8], [-.1, .6, .2]], dtype=torch.double, requires_grad=True)
        matrix = torch.tensor(T_06, dtype=torch.double, requires_grad=True)
        labels = torch.tensor([0, 2])
        self.assertTrue(torch.autograd.gradcheck(
            lambda x, t: reweight_loss(x, labels, t, detach_weights=False), (logits, matrix),
            eps=1e-6, atol=1e-5, rtol=1e-4))
        full = torch.autograd.grad(reweight_loss(logits, labels, matrix, detach_weights=False),
                                   (logits, matrix))
        detached = torch.autograd.grad(reweight_loss(logits, labels, matrix), logits)[0]
        self.assertGreater(float(full[1].abs().sum()), 0)
        self.assertFalse(torch.allclose(full[0], detached))

    def test_simplex_projection_zero_negative_and_known_solution(self):
        raw = torch.tensor([[0., 0., 0.], [-10., -10., -10.], [1.2, -.1, .4]])
        result = project_transition(raw)
        validate_transition(result)
        torch.testing.assert_close(result[:2], torch.full((2, 3), 1 / 3))
        torch.testing.assert_close(result[2], torch.tensor([.9, 0., .1]))
        for matrix in (T_03, T_06):
            original = torch.tensor(matrix)
            torch.testing.assert_close(project_transition(original), original)
        with self.assertRaises(ValueError):
            project_transition(torch.full((3, 3), float("nan")))

    def test_estimator_known_anchor_recovery_and_tiny_band(self):
        for strategy in ("argmax", "trimmed"):
            probabilities = np.repeat(T_03, 100, axis=0)
            result, info = estimate_transition(probabilities, strategy)
            np.testing.assert_allclose(result, T_03)
            self.assertTrue(all(row["count"] > 0 for row in info))
            tiny, info = estimate_transition(T_03[:2], strategy)
            validate_transition(tiny)
        with self.assertRaises(ValueError):
            estimate_transition(np.array([[np.nan, 0, 1]]))

    def test_split_isolation_and_training_only_statistics(self):
        images = np.arange(20 * 4, dtype=np.float32).reshape(20, 1, 2, 2)
        data = Dataset(images, np.arange(20) % 3, images[:5].copy(), np.arange(5) % 3, T_03, Path("fake"))
        first, meta = prepare_split(data, 7)
        self.assertEqual(len(meta["train_indices"]), 16)
        self.assertFalse(set(meta["train_indices"]) & set(meta["validation_indices"]))
        changed = images.copy()
        changed[meta["validation_indices"]] += 10000
        second, meta2 = prepare_split(Dataset(changed, data.noisy_labels, data.test_images + 9999,
                                             data.test_labels, T_03, Path("fake")), 7)
        self.assertEqual(meta["mean"], meta2["mean"])
        self.assertEqual(meta["std"], meta2["std"])
        torch.testing.assert_close(first["x_train"], second["x_train"])
        self.assertAlmostEqual(float(first["x_train"].mean()), 0, places=6)
        _, different = prepare_split(data, 8)
        self.assertNotEqual(meta["train_indices"], different["train_indices"])

    def test_refinement_can_restore_epoch_zero(self):
        seed_everything(1)
        model = torch.nn.Linear(2, 3)
        before = {k: v.detach().clone() for k, v in model.state_dict().items()}
        x = torch.rand(6, 2)
        y = torch.arange(6) % 3
        # Force validation to worsen; check model/matrix selection independently of training.
        losses = [torch.tensor(.1), torch.tensor(.2)]
        with patch("algorithm.engine.forward_loss", side_effect=losses), contextlib.redirect_stdout(io.StringIO()):
            result = fit(model, "trevision", x, y, x, y, transition=torch.tensor(T_06, dtype=torch.float32),
                         epochs=1, lr=.001, include_initial=True)
        self.assertEqual(result["selected_epoch"], 0)
        for key in before:
            torch.testing.assert_close(before[key], model.state_dict()[key])
        torch.testing.assert_close(result["transition"], torch.tensor(T_06, dtype=torch.float32))

    def test_sample_sd_and_single_run_not_fake_zero_uncertainty(self):
        self.assertEqual(statistics([1., 3.])["std"], np.sqrt(2))
        self.assertIsNone(statistics([1.])["std"])
        with self.assertRaises(ValueError):
            statistics([float("nan")])


class PersistenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def args(self, output, data_dir):
        return parser().parse_args(["--dataset", "fashion0.3", "--output", str(output),
                                   "--data-dir", str(data_dir), "--device", "cpu", "--smoke",
                                   "--runs", "1", "--epochs", "1", "--threads", "2"])

    def test_argument_validation(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory:
            args = self.args(directory, directory)
            self.assertEqual(build_config(args)["revision_epochs"], 1)
            args.smoke = False
            with self.assertRaisesRegex(ValueError, ">=10"):
                build_config(args)
            for option in ("--epochs", "--revision-epochs", "--runs", "--batch-size"):
                with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                    parser().parse_args(["--dataset", "cifar", "--output", directory, option, "0"])

    def test_atomic_json_does_not_replace_valid_file_with_nan(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory:
            path = Path(directory) / "result.json"
            atomic_json(path, {"value": 5})
            with self.assertRaises(ValueError):
                atomic_json(path, {"value": float("nan")})
            self.assertEqual(json.loads(path.read_text())["value"], 5)

    def test_broken_gpu_auto_fallback_and_explicit_error(self):
        with patch("algorithm.main.torch.cuda.is_available", return_value=True), \
                patch("algorithm.main.subprocess.run") as run:
            run.return_value.returncode = 3221225477
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(choose_device("auto"), "cpu")
            with self.assertRaisesRegex(ValueError, "GPU kernel probe"):
                choose_device("cuda")

    def test_end_to_end_resume_overwrite_identity_and_corruption(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory:
            root = Path(directory)
            rng = np.random.default_rng(19)
            np.savez(root / "FashionMNIST0.3.npz", Xtr=rng.integers(0, 256, (30, 784), dtype=np.uint8),
                     Str=np.arange(30) % 3, Xts=rng.integers(0, 256, (9, 784), dtype=np.uint8), Yts=np.arange(9) % 3)
            args = self.args(root / "experiment", root)
            with contextlib.redirect_stdout(io.StringIO()):
                first = run_experiment(args)
            self.assertTrue(first["complete"])
            self.assertFalse(first["reportable"])
            result = root / "experiment/results.json"
            contents = result.read_bytes()
            with self.assertRaises(FileExistsError):
                run_experiment(args)
            self.assertEqual(result.read_bytes(), contents)
            args.resume = True
            with contextlib.redirect_stdout(io.StringIO()), patch("algorithm.main.fit", side_effect=AssertionError("must not retrain")):
                second = run_experiment(args)
            self.assertEqual(first, second)
            # Simulate interruption after method checkpoints but before the run commit.
            run_result = root / "experiment/run_000/result.json"
            run_result.unlink()
            with contextlib.redirect_stdout(io.StringIO()), patch("algorithm.main.fit", side_effect=AssertionError("must reuse completed stages")):
                recovered = run_experiment(args)
            self.assertEqual(first["runs"][0]["acc"], recovered["runs"][0]["acc"])
            args.epochs = 2
            with self.assertRaisesRegex(ValueError, "Resume refused"):
                run_experiment(args)
            args.epochs = 1
            checkpoint = root / "experiment/run_000/ce.pt"
            checkpoint.write_bytes(checkpoint.read_bytes() + b"corruption")
            with contextlib.redirect_stdout(io.StringIO()), self.assertRaisesRegex(ValueError, "integrity mismatch"):
                run_experiment(args)

    def test_clean_test_labels_do_not_influence_any_fitted_checkpoint(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory:
            root = Path(directory)
            rng = np.random.default_rng(20)
            images = rng.integers(0, 256, (30, 784), dtype=np.uint8)
            test = rng.integers(0, 256, (9, 784), dtype=np.uint8)
            for label in (0, 1):
                np.savez(root / "FashionMNIST0.3.npz", Xtr=images, Str=np.arange(30) % 3,
                         Xts=test, Yts=np.full(9, label))
                args = self.args(root / f"experiment_{label}", root)
                with contextlib.redirect_stdout(io.StringIO()):
                    run_experiment(args)
            # Changing every clean test label must not change fitting or selection.
            for path in (root / "experiment_0/run_000").glob("*.pt"):
                a = torch.load(path, weights_only=True)
                b = torch.load(root / "experiment_1/run_000" / path.name, weights_only=True)
                self.assertEqual(a["selected_epoch"], b["selected_epoch"])
                self.assertEqual(a["validation_loss"], b["validation_loss"])
                for key in a["model_state"]:
                    torch.testing.assert_close(a["model_state"][key], b["model_state"][key], rtol=0, atol=0)
                if a["transition"] is not None:
                    torch.testing.assert_close(a["transition"], b["transition"], rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
