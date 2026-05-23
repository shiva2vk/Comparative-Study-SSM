"""
Loss functions for imbalanced multi-class classification.

• FocalLoss          — down-weights easy examples ( 5.4)
• ClassWeightedCE    — standard CE weighted by inverse class frequency
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional

class FocalLoss(nn.Module):
    """
    Multi-class focal loss (Lin et al. 2017).

        FL(p_t) = -α_t * (1 - p_t)^γ * log(p_t)

    Parameters
    ----------
    gamma : float
        Focusing parameter. 0  standard CE. 2.0 recommended.
    alpha : Optional[Tensor]
        Per-class weights (same shape as class dimension).
    reduction : str
        'mean' | 'sum' | 'none'
    """

    def __init__(
        self,
        gamma: float = 2.0,
        alpha: Optional[torch.Tensor] = None,
        reduction: str = "mean",
    ):
        super().__init__()
        self.gamma     = gamma
        self.alpha     = alpha
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        logits  : (B, C) unnormalised scores
        targets : (B,)   integer class indices
        """
        log_prob = F.log_softmax(logits, dim=-1)
        prob     = log_prob.exp()

        log_pt = log_prob.gather(1, targets.unsqueeze(1)).squeeze(1)
        pt     = prob.gather(1, targets.unsqueeze(1)).squeeze(1)

        focal_weight = (1 - pt) ** self.gamma

        if self.alpha is not None:
            alpha_t = self.alpha.to(logits.device)[targets]
            focal_weight = focal_weight * alpha_t

        loss = -focal_weight * log_pt

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss

def compute_class_weights(y: np.ndarray, num_classes: int) -> torch.Tensor:
    """
    Inverse-frequency class weights for weighted CE or focal loss alpha.

    Returns
    -------
    weights : (num_classes,) float32 tensor, sum-normalised
    """
    counts = np.bincount(y, minlength=num_classes).astype(np.float32)
    counts = np.where(counts == 0, 1e-6, counts)
    weights = 1.0 / counts
    weights = weights / weights.sum() * num_classes
    return torch.from_numpy(weights).float()

def get_loss_fn(
    y_train: np.ndarray,
    num_classes: int,
    focal_gamma: float = 2.0,
    use_focal: bool = True,
) -> nn.Module:
    """
    Build the loss function.

    If use_focal=True : FocalLoss with class-frequency alpha.
    Otherwise         : weighted CrossEntropyLoss.
    """
    class_weights = compute_class_weights(y_train, num_classes)

    if use_focal:
        return FocalLoss(gamma=focal_gamma, alpha=class_weights)
    else:
        return nn.CrossEntropyLoss(weight=class_weights)
