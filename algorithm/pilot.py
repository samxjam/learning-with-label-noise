"""Blind tier 3 pilot: compare training settings on noisy validation only. The adoption rule is in METHODS.md.

Clean test tensors are deleted straight after each split, so no candidate can be
scored or chosen on test accuracy. Results are one JSON file per fit, so a stopped
run resumes by skipping completed fits.
"""
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse
import copy
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from algorithm.data import load_data, prepare_split
from algorithm.engine import fit, predict, seed_everything
from algorithm.methods import estimate_transition, forward_loss
from algorithm.models import CNN

CANDIDATES = {
    "A_baseline": {"network": "cnn", "cosine": False, "crop": False, "ema": False},
    "B_cosine": {"network": "cnn", "cosine": True, "crop": False, "ema": False},
    "C_cosine_crop": {"network": "cnn", "cosine": True, "crop": True, "ema": False},
    "D_cosine_crop_ema": {"network": "cnn", "cosine": True, "crop": True, "ema": True},
    "E_resnet": {"network": "resnet", "cosine": True, "crop": True, "ema": False},
}
DATASETS = {"cifar": 40, "fashion0.6": 20, "fashion0.3": 20}
EMA_DECAY = .999


class Block(nn.Module):
    """Standard two-convolution residual block."""

    def __init__(self, channels_in, channels_out, stride):
        super().__init__()
        self.conv1 = nn.Conv2d(channels_in, channels_out, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels_out)
        self.conv2 = nn.Conv2d(channels_out, channels_out, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels_out)
        self.skip = nn.Identity()
        if stride != 1 or channels_in != channels_out:
            self.skip = nn.Sequential(nn.Conv2d(channels_in, channels_out, 1, stride, bias=False),
                                      nn.BatchNorm2d(channels_out))

    def forward(self, x):
        out = self.bn2(self.conv2(F.relu(self.bn1(self.conv1(x)))))
        return F.relu(out + self.skip(x))


class ResNet20(nn.Module):
    """CIFAR-style ResNet-20 with three logits, for greyscale or colour input."""

    def __init__(self, in_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, 16, 3, 1, 1, bias=False)
        self.bn = nn.BatchNorm2d(16)
        layers = []
        for channels_in, channels_out, stride in ((16, 16, 1), (16, 32, 2), (32, 64, 2)):
            layers.append(Block(channels_in, channels_out, stride))
            layers += [Block(channels_out, channels_out, 1) for _ in range(2)]
        self.layers = nn.Sequential(*layers)
        self.fc = nn.Linear(64, 3)

    def forward(self, x):
        return self.fc(self.layers(F.relu(self.bn(self.conv(x)))).mean(dim=(2, 3)))


def random_crop(x, padding=4):
    """Pad by reflection and take one random crop per image."""
    height, width = x.shape[2:]
    padded = F.pad(x, (padding,) * 4, mode="reflect")
    y0 = torch.randint(0, 2 * padding + 1, (len(x),), device=x.device)
    x0 = torch.randint(0, 2 * padding + 1, (len(x),), device=x.device)
    rows = y0[:, None] + torch.arange(height, device=x.device)
    cols = x0[:, None] + torch.arange(width, device=x.device)
    batch = torch.arange(len(x), device=x.device)[:, None, None]
    return padded[batch, :, rows[:, :, None], cols[:, None, :]].permute(0, 3, 1, 2)


def train_forward(candidate, tensors, transition, epochs, split, flip):
    """Train Forward under one candidate; return noisy-validation history and selection."""
    settings = CANDIDATES[candidate]
    seed_everything(10000 + split)
    channels = tensors["x_train"].shape[1]
    model = (ResNet20(channels) if settings["network"] == "resnet" else CNN(channels)).to(tensors["x_train"].device)
    seed_everything(20000 + split)
    generator = torch.Generator().manual_seed(20000 + split)
    optimizer = torch.optim.Adam(model.parameters(), lr=.001, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs) if settings["cosine"] else None
    shadow = copy.deepcopy(model) if settings["ema"] else None
    x_train, s_train = tensors["x_train"], tensors["s_train"]
    history, steps, start = [], 0, time.perf_counter()
    for epoch in range(1, epochs + 1):
        model.train()
        order = torch.randperm(len(x_train), generator=generator).to(x_train.device)
        for offset in range(0, len(order), 128):
            indices = order[offset:offset + 128]
            x, labels = x_train[indices], s_train[indices]
            if flip:
                mask = torch.rand(len(x), device=x.device) < .5
                x[mask] = x[mask].flip(3)
                if settings["crop"]:
                    x = random_crop(x)
            loss = forward_loss(model(x), labels, transition)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            steps += 1
            if shadow is not None:
                decay = min(EMA_DECAY, (1 + steps) / (10 + steps))
                with torch.no_grad():
                    for target, source in zip(shadow.state_dict().values(), model.state_dict().values()):
                        if target.is_floating_point():
                            target.mul_(decay).add_(source, alpha=1 - decay)
                        else:
                            target.copy_(source)
        if scheduler is not None:
            scheduler.step()
        logits = predict(shadow if shadow is not None else model, tensors["x_validation"])
        noisy = logits.softmax(1) @ transition
        history.append({"epoch": epoch, "validation_loss": float(forward_loss(logits, tensors["s_validation"], transition).item()),
                        "validation_noisy_accuracy": float((noisy.argmax(1) == tensors["s_validation"]).float().mean().item() * 100)})
    best = min(history, key=lambda row: row["validation_loss"])
    return {"candidate": candidate, "split": split, "epochs": epochs, "selected_epoch": best["epoch"],
            "selected_loss": best["validation_loss"], "selected_noisy_accuracy": best["validation_noisy_accuracy"],
            "seconds": time.perf_counter() - start, "parameters": sum(p.numel() for p in model.parameters()),
            "history": history}


def run(args):
    """Run every candidate, dataset and split that is not already saved."""
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    for name, default_epochs in DATASETS.items():
        epochs = args.epochs or default_epochs
        data = load_data(name, args.data_dir)
        for split in args.splits:
            tensors, _ = prepare_split(data, split, args.device, args.sample_limit, args.sample_limit)
            del tensors["x_test"], tensors["y_test"]
            flip = name == "cifar"
            if data.true_transition is None:
                seed_everything(10000 + split)
                model = CNN(tensors["x_train"].shape[1]).to(args.device)
                fit(model, "ce", tensors["x_train"], tensors["s_train"], tensors["x_validation"], tensors["s_validation"],
                    epochs=epochs, lr=.001, flip=flip, seed=20000 + split)
                probabilities = predict(model, tensors["x_train"]).softmax(1).cpu().numpy()
                matrix = estimate_transition(probabilities, "trimmed")[0]
            else:
                matrix = data.true_transition
            transition = torch.tensor(matrix, dtype=torch.float32, device=args.device)
            for candidate, settings in CANDIDATES.items():
                if not flip and settings["crop"] and not settings["ema"] and settings["network"] == "cnn":
                    continue  # Without crops this candidate is identical to B_cosine.
                path = args.output / f"{name}__{candidate}__{split}.json"
                if path.is_file():
                    print("skip", path.name, flush=True)
                    continue
                result = train_forward(candidate, tensors, transition, epochs, split, flip)
                result["dataset"] = name
                result["matrix"] = np.asarray(matrix).tolist()
                path.write_text(json.dumps(result, indent=1), encoding="utf-8")
                print(f"{name} {candidate} split={split} loss={result['selected_loss']:.5f} "
                      f"noisy_acc={result['selected_noisy_accuracy']:.2f} epoch={result['selected_epoch']} "
                      f"{result['seconds']:.0f}s", flush=True)


def load_results(output):
    """Index saved fits by dataset, candidate and split."""
    results = {}
    for path in Path(output).glob("*__*__*.json"):
        row = json.loads(path.read_text(encoding="utf-8"))
        results[row["dataset"], row["candidate"], row["split"]] = row
    return results


def decide(output, splits):
    """Apply the METHODS.md adoption rule and write decision.json."""
    results = load_results(output)

    def rows(name, candidate):
        lookup = "B_cosine" if name != "cifar" and candidate == "C_cosine_crop" else candidate
        return [results[name, lookup, split] for split in splits]

    def mean(name, candidate, key):
        return float(np.mean([row[key] for row in rows(name, candidate)]))

    table, passing = {}, []
    for candidate in CANDIDATES:
        entry = {}
        for name in DATASETS:
            entry[name] = {"loss": mean(name, candidate, "selected_loss"),
                           "noisy_accuracy": mean(name, candidate, "selected_noisy_accuracy"),
                           "seconds": mean(name, candidate, "seconds"),
                           "splits_better": sum(a["selected_loss"] < b["selected_loss"] for a, b in
                                                zip(rows(name, candidate), rows(name, "A_baseline")))}
        table[candidate] = entry
        if candidate == "A_baseline":
            continue
        base = {name: {k: mean(name, "A_baseline", k) for k in ("selected_loss", "selected_noisy_accuracy")} for name in DATASETS}
        checks = {
            "lower_mean_loss": all(entry[n]["loss"] < base[n]["selected_loss"] for n in ("cifar", "fashion0.6")),
            "lower_in_most_splits": all(entry[n]["splits_better"] >= 2 for n in ("cifar", "fashion0.6")),
            "noisy_accuracy_not_lower": all(entry[n]["noisy_accuracy"] >= base[n]["selected_noisy_accuracy"] for n in ("cifar", "fashion0.6")),
            "fashion03_not_worse": entry["fashion0.3"]["loss"] <= base["fashion0.3"]["selected_loss"] + .002,
        }
        entry["checks"] = checks
        if all(checks.values()):
            passing.append(candidate)
    choice = None
    for candidate in passing:
        if choice is None or table[candidate]["cifar"]["loss"] <= table[choice]["cifar"]["loss"] - .002:
            choice = candidate
    decision = {"passing": passing, "recommended": choice, "splits": list(splits), "table": table,
                "note": "Noisy validation only; no clean test label was evaluated."}
    (Path(output) / "decision.json").write_text(json.dumps(decision, indent=1), encoding="utf-8")
    for candidate, entry in table.items():
        print(candidate, {n: (round(entry[n]["loss"], 5), round(entry[n]["noisy_accuracy"], 2), entry[n]["splits_better"], round(entry[n]["seconds"]))
                          for n in DATASETS}, entry.get("checks", ""), flush=True)
    print("Passing:", passing, "| Recommended:", choice, flush=True)
    return decision


def main():
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--splits", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--epochs", type=int, help="smoke checks only")
    parser.add_argument("--sample-limit", type=int, help="smoke checks only")
    args = parser.parse_args()
    run(args)
    decide(args.output, args.splits)


if __name__ == "__main__":
    main()
