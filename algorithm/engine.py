"""Training uses noisy labels only; held-out clean labels are evaluation-only."""
import copy
import random
import time

import numpy as np
import torch
import torch.nn.functional as F

try:
    from .methods import forward_loss, reweight_loss, project_transition, validate_transition
except ImportError:
    from methods import forward_loss, reweight_loss, project_transition, validate_transition


def seed_everything(seed, deterministic=True):
    """Seed all random generators and request deterministic kernels where available."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(deterministic)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False


def cpu_copy(value):
    """Deep-copy nested tensors to the CPU, so a saved best state cannot change later."""
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {k: cpu_copy(v) for k, v in value.items()}
    if isinstance(value, list):
        return [cpu_copy(v) for v in value]
    if isinstance(value, tuple):
        return tuple(cpu_copy(v) for v in value)
    return copy.deepcopy(value)


@torch.no_grad()
def predict(model, images, batch_size=512):
    """Return logits for all images in evaluation mode."""
    model.eval()
    return torch.cat([model(images[i:i + batch_size]) for i in range(0, len(images), batch_size)])


@torch.no_grad()
def accuracy(model, images, labels):
    """Return top-1 accuracy as a percentage."""
    return float((predict(model, images).argmax(1) == labels).float().mean().item() * 100)


def fit(model, method, x_train, s_train, x_validation, s_validation, *,
        transition=None, epochs, lr, batch_size=128, flip=False, seed=0,
        deterministic=True, transition_lr=5e-4, include_initial=False,
        epoch_callback=None):
    """Train and restore the model and matrix selected by noisy validation loss.

    Refinements use fresh Adam states and the same batch/augmentation/dropout
    seed, starting from exactly the same selected reweight checkpoint. The
    initial checkpoint is eligible for both refinements. No test input is
    accepted here. Returned optimiser state corresponds to the selected epoch.
    """
    if epochs < 1 or batch_size < 1 or lr <= 0 or transition_lr <= 0:
        raise ValueError("Epochs, batch size and learning rates must be positive")
    if method not in {"ce", "forward", "reweight", "trevision", "continued", "continued_full"}:
        raise ValueError(f"Unknown method {method}")
    seed_everything(seed, deterministic)
    if method != "ce":
        if transition is None:
            raise ValueError("Robust methods require a transition matrix")
        validate_transition(transition)
    current = (torch.nn.Parameter(transition.detach().clone()) if method == "trevision"
               else transition)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    if method == "trevision":
        optimizer.add_param_group({"params": [current], "lr": transition_lr, "weight_decay": 0})
    parameters = list(model.parameters()) + ([current] if method == "trevision" else [])
    generator = torch.Generator().manual_seed(seed)
    best = None
    history = []
    start = time.perf_counter()

    @torch.no_grad()
    def validation():
        """Noisy-validation loss: CE for the baseline, forward CE for matrix methods."""
        logits = predict(model, x_validation)
        value = (F.cross_entropy(logits, s_validation) if method == "ce"
                 else forward_loss(logits, s_validation, current))
        if not torch.isfinite(value):
            raise FloatingPointError(f"Non-finite {method} validation loss")
        return float(value.item())

    def consider(epoch, value):
        """Keep this epoch's model and matrix if they beat the best validation loss so far."""
        nonlocal best
        if best is None or value < best["validation_loss"]:
            best = {"model_state": cpu_copy(model.state_dict()),
                    "optimizer_state": cpu_copy(optimizer.state_dict()),
                    "transition": cpu_copy(current), "selected_epoch": epoch,
                    "validation_loss": value}

    if include_initial:
        value = validation()
        consider(0, value)
        history.append({"epoch": 0, "training_loss": None, "validation_loss": value})
    for epoch in range(1, epochs + 1):
        model.train()
        order = torch.randperm(len(x_train), generator=generator).to(x_train.device)
        total_loss = torch.zeros((), device=x_train.device)
        for offset in range(0, len(order), batch_size):
            indices = order[offset:offset + batch_size]
            x, labels = x_train[indices], s_train[indices]
            if flip:
                # Indexing above copied the batch, so flips never alter the stored training images.
                mask = torch.rand(len(x), device=x.device) < .5
                x[mask] = x[mask].flip(3)
            logits = model(x)
            if method == "ce":
                loss = F.cross_entropy(logits, labels)
            elif method == "forward":
                loss = forward_loss(logits, labels, current)
            else:
                # Stop-gradient for reweight/continued; full gradient for continued_full/trevision.
                loss = reweight_loss(logits, labels, current,
                                     detach_weights=method not in ("trevision", "continued_full"))
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite {method} training loss at epoch {epoch}")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, float("inf"), error_if_nonfinite=True)
            optimizer.step()
            if method == "trevision":
                # Keep each learned matrix row a valid probability distribution.
                with torch.no_grad():
                    current.copy_(project_transition(current))
            total_loss += loss.detach() * len(indices)
        value = validation()
        consider(epoch, value)
        row = {"epoch": epoch, "training_loss": float(total_loss.item() / len(x_train)),
               "validation_loss": value}
        history.append(row)
        print(f"  {method:10s} epoch {epoch:02d}/{epochs}: val={value:.6f} best={best['selected_epoch']}", flush=True)
        if epoch_callback is not None:
            epoch_callback(history)
    model.load_state_dict(best["model_state"])
    best.update({"method": method, "history": history, "trained_epochs": epochs,
                 "training_seed": seed, "seconds": time.perf_counter() - start,
                 "selection": "noisy_ce" if method == "ce" else "noisy_forward_ce",
                 "initial_checkpoint_eligible": include_initial})
    return best
