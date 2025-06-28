"""
model/trainer.py
----------------

Defines the neural network **and** all training / evaluation
utilities.  Keeping them here (instead of in `main.py`) means
you can later do hyper-parameter sweeps without touching the
entry-point script.
"""

from pathlib import Path
from typing import Dict, Optional
import numpy as np
from torch.optim import Optimizer

import torch
from torch import nn
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    roc_curve,
)

# ────────────────────────────────────────────────────────────────
# ✦ 1. Model definition
# ────────────────────────────────────────────────────────────────
class HeartNet(nn.Module):
    """
    A *very* small fully-connected network.
    For tabular data, depth usually matters less than good
    preprocessing, so we keep it simple:
        Input  → 64 → 32 → 1
    The final Sigmoid squeezes outputs into [0,1] so we can
    interpret them as probabilities of heart disease.
    """

    def __init__(self, num_features: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(num_features, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),  # Sigmoid instead of Softmax for binary task
        )

    def forward(self, x):
        return self.net(x).view(-1)  # flatten to shape (batch,)


# ────────────────────────────────────────────────────────────────
# ✦ 2. Training & evaluation helpers
# ────────────────────────────────────────────────────────────────
def _step(
    model: nn.Module,
    loader: DataLoader,
    criterion,
    device: torch.device,
    train: bool = False,
    optimizer: Optimizer | None = None,
) -> Dict[str, float]:
    """
    One epoch *or* one validation pass.

    Args:
        train=True  → back-prop & update
        train=False → just forward pass
    """
    if train:
        model.train()
    else:
        model.eval()

    losses: list[float] = []
    y_true: list[float] = []
    y_pred: list[float] = []

    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        # Forward pass
        logits = model(X_batch)
        loss = criterion(logits, y_batch)

        # Backward pass (only if training)
        if train:
            optimizer.zero_grad() #type: ignore
            loss.backward()
            optimizer.step() #type: ignore

        losses.append(loss.item())
        y_true.extend(y_batch.detach().cpu().numpy())
        y_pred.extend(logits.detach().cpu().numpy())

    # Compute metrics on CPU
    y_pred_bin = (torch.tensor(y_pred) >= 0.5).float().numpy()
    acc = accuracy_score(y_true, y_pred_bin)
    auc = roc_auc_score(y_true, y_pred)

    return {
        "loss": sum(losses) / len(losses),
        "accuracy": acc,
        "auc": auc,
    } #type: ignore


def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = 100,
    lr: float = 1e-3,
    device: str | torch.device = "cpu",
    patience: int = 10,
    ckpt_path: Path | str = "model_best.pth",
):
    """
    Standard training loop with *early stopping*.

    Early stopping rationale:
      • ML models usually start to over-fit after some epochs.
      • We stop when val loss hasn't improved for `patience` epochs.
    """

    device = torch.device(device)
    model = model.to(device)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_val_loss = float("inf")
    epochs_no_improve = 0

    for epoch in range(1, epochs + 1):
        train_metrics = _step(
            model, train_loader, criterion, device, train=True, optimizer=optimizer
        )
        val_metrics = _step(model, val_loader, criterion, device, train=False)

        print(
            f"Epoch {epoch:03d}: "
            f"train_loss={train_metrics['loss']:.4f}, val_loss={val_metrics['loss']:.4f}, "
            f"val_acc={val_metrics['accuracy']:.4f}, val_auc={val_metrics['auc']:.4f}"
        )

        # Check early-stop condition
        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            epochs_no_improve = 0
            torch.save(model.state_dict(), ckpt_path)
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping after {epoch} epochs.")
                break

    # Load best weights before returning
    model.load_state_dict(torch.load(ckpt_path))
    return model


def evaluate(
    model: nn.Module,
    test_loader: DataLoader,
    *,
    device: str | torch.device = "cpu",
    threshold: float | None = 0.5,
) -> Dict[str, float]:
    """
    Final unbiased evaluation on the *test* split.

    Parameters
    ----------
    model : trained network
    test_loader : DataLoader
        Holds only the test-set batches.
    device : "cpu" | "cuda"
    threshold : float | None
        • If a float (e.g. 0.5) → use that fixed cut-off  
        • If None          → find the *EER* threshold first, then
                              report FMR & FNMR *at that point*.
    """
    device = torch.device(device)
    model = model.to(device)
    criterion = nn.BCELoss()

    # ─────────────────────────────────────────────────────────────
    # Pass once through the test set  → collect scores & labels
    # ─────────────────────────────────────────────────────────────
    model.eval()
    losses, y_true, y_scores = [], [], []

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            scores = model(X_batch)          # sigmoid outputs ∈ (0,1)
            loss = criterion(scores, y_batch)

            losses.append(loss.item())
            y_true.extend(y_batch.cpu().numpy())
            y_scores.extend(scores.cpu().numpy())

    y_true = np.asarray(y_true, dtype=np.float32)
    y_scores = np.asarray(y_scores, dtype=np.float32)

    # Basic global metrics -------------------------------------------------
    mean_loss = float(np.mean(losses))
    auc = roc_auc_score(y_true, y_scores)

    # ──────────────────────────────────────────────────────────────────────
    # 1. Choose a threshold
    #    • given explicitly, OR
    #    • find the Equal-Error-Rate (EER) threshold
    # ──────────────────────────────────────────────────────────────────────
    if threshold is None:
        fpr, tpr, thresh = roc_curve(y_true, y_scores)
        fnr = 1 - tpr
        i_eer = np.argmin(np.abs(fpr - fnr))        # where FPR ≈ FNR
        threshold = thresh[i_eer]
        eer = (fpr[i_eer] + fnr[i_eer]) / 2
    else:
        # still compute EER just for reference
        fpr, tpr, thresh = roc_curve(y_true, y_scores)
        fnr = 1 - tpr
        i_eer = np.argmin(np.abs(fpr - fnr))
        eer = (fpr[i_eer] + fnr[i_eer]) / 2

    # ──────────────────────────────────────────────────────────────────────
    # 2. Binarise predictions at the chosen threshold
    # ──────────────────────────────────────────────────────────────────────
    y_pred_bin = (y_scores >= threshold).astype(np.float32) # type: ignore

    # Confusion-matrix elements -------------------------------------------
    TP = float(((y_pred_bin == 1) & (y_true == 1)).sum())
    TN = float(((y_pred_bin == 0) & (y_true == 0)).sum())
    FP = float(((y_pred_bin == 1) & (y_true == 0)).sum())
    FN = float(((y_pred_bin == 0) & (y_true == 1)).sum())

    # FMR = FP / (FP + TN)         (impostors accepted)
    # FNMR = FN / (TP + FN)        (genuine rejected)
    fmr = FP / (FP + TN + 1e-9)    # add ε to avoid /0
    fnmr = FN / (TP + FN + 1e-9)

    # Accuracy at this threshold
    acc = (TP + TN) / (TP + TN + FP + FN + 1e-9)

    return {
        "loss": mean_loss,
        "accuracy": acc,
        "auc": auc,
        "eer": eer,
        "threshold": threshold,
        "fmr": fmr,
        "fnmr": fnmr,
    } # type: ignore