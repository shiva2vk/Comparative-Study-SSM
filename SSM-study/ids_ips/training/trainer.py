"""
Generic training loop for any IDS model.

Implements:
  • AdamW optimiser with cosine LR scheduling
  • Mixed-precision training (AMP, FP16/BF16)
  • Gradient clipping
  • Per-epoch metric logging
  • Checkpoint saving (best val-F1)
  • Optional 5-fold stratified cross-validation
"""

import logging
import time
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Subset

from sklearn.metrics import f1_score, accuracy_score

from ids_ips.data.dataset import NetworkFlowDataset, get_kfold_splits
from ids_ips.training.losses import get_loss_fn
from ids_ips.utils.gcs import gcs

logger = logging.getLogger(__name__)

def _train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimiser: torch.optim.Optimizer,
    loss_fn: nn.Module,
    scaler: GradScaler,
    device: torch.device,
    grad_clip: float,
    use_amp: bool,
) -> Dict[str, float]:
    model.train()
    total_loss = 0.0
    all_preds, all_targets = [], []

    for X, y in loader:
        X, y = X.to(device), y.to(device)
        optimiser.zero_grad()

        with autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_amp):
            logits = model(X)
            loss   = loss_fn(logits, y)

        scaler.scale(loss).backward()
        scaler.unscale_(optimiser)
        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimiser)
        scaler.update()

        total_loss += loss.item() * len(y)
        preds = logits.argmax(dim=-1).cpu().numpy()
        all_preds.extend(preds)
        all_targets.extend(y.cpu().numpy())

    n = len(all_targets)
    return {
        "loss": total_loss / n,
        "acc":  accuracy_score(all_targets, all_preds),
        "f1":   f1_score(all_targets, all_preds, average="macro", zero_division=0),
    }

@torch.no_grad()
def _eval_epoch(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
    use_amp: bool,
) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    all_preds, all_targets = [], []

    for X, y in loader:
        X, y = X.to(device), y.to(device)
        with autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_amp):
            logits = model(X)
            loss   = loss_fn(logits, y)

        total_loss += loss.item() * len(y)
        preds = logits.argmax(dim=-1).cpu().numpy()
        all_preds.extend(preds)
        all_targets.extend(y.cpu().numpy())

    n = len(all_targets)
    return {
        "loss": total_loss / n,
        "acc":  accuracy_score(all_targets, all_preds),
        "f1":   f1_score(all_targets, all_preds, average="macro", zero_division=0),
    }

class Trainer:
    """
    Wraps model + optimiser + scheduler + loss into a reusable training object.

    Usage
    -----
    trainer = Trainer(model, config, y_train_np, device)
    history = trainer.fit(train_ds, val_ds)
    cv_results = trainer.fit_kfold(train_ds)
    """

    def __init__(
        self,
        model:        nn.Module,
        epochs:       int,
        batch_size:   int,
        lr:           float,
        weight_decay: float,
        focal_gamma:  float,
        grad_clip:    float,
        use_amp:      bool,
        n_folds:      int,
        save_dir:     Path,
        y_train:      np.ndarray,
        num_classes:  int,
        device:       torch.device,
        model_name:   str = "model",
    ):
        self.model        = model.to(device)
        self.epochs       = epochs
        self.batch_size   = batch_size
        self.lr           = lr
        self.weight_decay = weight_decay
        self.focal_gamma  = focal_gamma
        self.grad_clip    = grad_clip
        self.use_amp      = use_amp and device.type == "cuda"
        self.n_folds      = n_folds
        self.save_dir     = save_dir
        self.y_train      = y_train
        self.num_classes  = num_classes
        self.device       = device
        self.model_name   = model_name
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.loss_fn = get_loss_fn(
            y_train, num_classes, focal_gamma=focal_gamma, use_focal=True
        )

    def _build_optimiser(self):
        opt = AdamW(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
        sched = CosineAnnealingLR(opt, T_max=self.epochs, eta_min=self.lr * 0.01)
        return opt, sched

    def fit(
        self,
        train_ds:   NetworkFlowDataset,
        val_ds:     NetworkFlowDataset,
        fold_idx:   int = 0,
    ) -> Dict[str, list]:
        """Train on train_ds, validate on val_ds. Returns history dict."""
        train_loader = DataLoader(
            train_ds, batch_size=self.batch_size,
            shuffle=True, num_workers=4, pin_memory=self.device.type == "cuda"
        )
        val_loader = DataLoader(
            val_ds, batch_size=self.batch_size * 2,
            shuffle=False, num_workers=4
        )

        optimiser, scheduler = self._build_optimiser()
        scaler = GradScaler(device=self.device.type, enabled=self.use_amp)

        best_val_f1   = -1.0
        best_ckpt     = self.save_dir / f"{self.model_name}_fold{fold_idx}_best.pt"
        history: Dict[str, list] = {k: [] for k in
                                    ["train_loss","train_acc","train_f1",
                                     "val_loss","val_acc","val_f1"]}

        for epoch in range(1, self.epochs + 1):
            t0 = time.time()
            train_metrics = _train_epoch(
                self.model, train_loader, optimiser, self.loss_fn,
                scaler, self.device, self.grad_clip, self.use_amp
            )
            val_metrics = _eval_epoch(
                self.model, val_loader, self.loss_fn, self.device, self.use_amp
            )
            scheduler.step()

            for k in ["loss","acc","f1"]:
                history[f"train_{k}"].append(train_metrics[k])
                history[f"val_{k}"].append(val_metrics[k])

            if val_metrics["f1"] > best_val_f1:
                best_val_f1 = val_metrics["f1"]
                torch.save(self.model.state_dict(), best_ckpt)
                gcs.upload(best_ckpt, f"checkpoints/{best_ckpt.name}")

            elapsed = time.time() - t0
            logger.info(
                "[Fold %d | Epoch %d/%d | %.1fs] "
                "train loss=%.4f acc=%.4f f1=%.4f | "
                "val loss=%.4f acc=%.4f f1=%.4f",
                fold_idx, epoch, self.epochs, elapsed,
                train_metrics["loss"], train_metrics["acc"], train_metrics["f1"],
                val_metrics["loss"],   val_metrics["acc"],   val_metrics["f1"],
            )

        self.model.load_state_dict(torch.load(best_ckpt, map_location=self.device,
                                              weights_only=True))
        logger.info("Best val F1 = %.4f (fold %d)", best_val_f1, fold_idx)
        return history

    def fit_kfold(self, train_ds: NetworkFlowDataset) -> Dict:
        """
        Run n_folds stratified CV on train_ds.
        Returns dict with per-fold histories and aggregate stats.
        """
        y_full = train_ds.y.numpy()
        fold_histories = []

        for fold, (tr_idx, vl_idx) in enumerate(
            get_kfold_splits(train_ds.X.numpy(), y_full, self.n_folds)
        ):
            logger.info(" Fold %d / %d ", fold + 1, self.n_folds)
            self.model.apply(self._reset_weights)

            tr_sub = Subset(train_ds, tr_idx)
            vl_sub = Subset(train_ds, vl_idx)

            h = self.fit(tr_sub, vl_sub, fold_idx=fold)
            fold_histories.append(h)

        mean_val_f1 = np.mean([max(h["val_f1"]) for h in fold_histories])
        std_val_f1  = np.std( [max(h["val_f1"]) for h in fold_histories])
        logger.info("CV macro-F1: %.4f ± %.4f", mean_val_f1, std_val_f1)

        return {
            "fold_histories":  fold_histories,
            "mean_val_f1":     float(mean_val_f1),
            "std_val_f1":      float(std_val_f1),
        }

    @staticmethod
    def _reset_weights(m: nn.Module):
        if hasattr(m, "reset_parameters"):
            m.reset_parameters()
