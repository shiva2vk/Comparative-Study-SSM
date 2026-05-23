"""
 evaluation and Metrics and Detection quality, Model efficiency job .

"""

import time
import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    average_precision_score,
)
from sklearn.preprocessing import label_binarize

logger = logging.getLogger(__name__)

ATTACK_CATEGORIES = [
    "Normal", "Generic", "Exploits", "Fuzzers", "DoS",
    "Reconnaissance", "Analysis", "Backdoor", "Shellcode", "Worms",
]

@torch.no_grad()
def predict(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
  
    model.eval()
    model.to(device)

    all_preds, all_probs, all_targets = [], [], []
    for X, y in loader:
        X = X.to(device)
        logits = model(X)
        probs  = torch.softmax(logits, dim=-1).cpu().numpy()
        preds  = np.argmax(probs, axis=-1)
        all_preds.extend(preds)
        all_probs.append(probs)
        all_targets.extend(y.numpy())

    return (
        np.array(all_preds),
        np.vstack(all_probs),
        np.array(all_targets),
    )

def compute_detection_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    class_names: List[str] = ATTACK_CATEGORIES,
) -> Dict:
    num_classes = len(class_names)
    acc  = accuracy_score(y_true, y_pred)
    f1   = f1_score(y_true, y_pred, average="macro", zero_division=0)
    prec = precision_score(y_true, y_pred, average="macro", zero_division=0)
    rec  = recall_score(y_true, y_pred, average="macro", zero_division=0)

    y_bin = label_binarize(y_true, classes=list(range(num_classes)))
    try:
        roc_auc = roc_auc_score(y_bin, y_prob, average="macro",
                                multi_class="ovr")
    except ValueError:
        roc_auc = float("nan")

    pr_aucs = []
    for c in range(num_classes):
        if y_bin[:, c].sum() > 0:
            pr_aucs.append(average_precision_score(y_bin[:, c], y_prob[:, c]))
    pr_auc = float(np.mean(pr_aucs)) if pr_aucs else float("nan")

    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
    fpr_per_class = []
    for c in range(num_classes):
        tp = cm[c, c]
        fn = cm[c].sum() - tp
        fp = cm[:, c].sum() - tp
        tn = cm.sum() - tp - fn - fp
        fpr_per_class.append(fp / (fp + tn + 1e-9))
    fpr = float(np.mean(fpr_per_class))

    per_class_f1 = f1_score(
        y_true, y_pred, average=None, zero_division=0, labels=list(range(num_classes))
    )

    return {
        "accuracy":      float(acc),
        "macro_f1":      float(f1),
        "macro_precision": float(prec),
        "macro_recall":  float(rec),
        "roc_auc":       float(roc_auc),
        "pr_auc":        float(pr_auc),
        "fpr":           float(fpr),
        "per_class_f1":  {class_names[i]: float(per_class_f1[i])
                          for i in range(min(num_classes, len(per_class_f1)))},
        "confusion_matrix": cm.tolist(),
    }

def benchmark_latency(
    model: nn.Module,
    n_features: int,
    batch_sizes: List[int] = [1, 32, 256, 1024],
    n_warmup: int = 50,
    n_runs: int = 500,
    device: Optional[torch.device] = None,
) -> Dict:

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval().to(device)

    results = {}
    for bs in batch_sizes:
        x = torch.randn(bs, n_features, device=device)

        with torch.no_grad():
            for _ in range(n_warmup):
                _ = model(x)

        if device.type == "cuda":
            torch.cuda.synchronize()

        times = []
        with torch.no_grad():
            for _ in range(n_runs):
                t0 = time.perf_counter()
                _ = model(x)
                if device.type == "cuda":
                    torch.cuda.synchronize()
                times.append(time.perf_counter() - t0)

        avg_s      = np.mean(times)
        p50_s      = np.percentile(times, 50)
        p99_s      = np.percentile(times, 99)
        throughput = bs / avg_s

        results[bs] = {
            "batch_size":           bs,
            "avg_latency_ms":       avg_s * 1000,
            "per_sample_latency_ms": avg_s * 1000 / bs,
            "p50_latency_ms":       p50_s * 1000,
            "p99_latency_ms":       p99_s * 1000,
            "throughput_pps":       throughput,
        }
        logger.info(
            "BS=%4d | avg=%.3f ms | per-sample=%.4f ms | %.0f samples/s",
            bs, avg_s * 1000, avg_s * 1000 / bs, throughput
        )

    return results

def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def model_memory_mb(model: nn.Module) -> float:
    
    total_bytes = sum(
        p.numel() * p.element_size() for p in model.parameters()
    )
    return total_bytes / (1024 ** 2)

def print_report(
    model_name: str,
    det_metrics: Dict,
    latency_results: Optional[Dict] = None,
    n_params: Optional[int] = None,
    mem_mb: Optional[float] = None,
):
    sep = "─" * 60
    print(f"\n{sep}")
    print(f"  Model : {model_name}")
    print(sep)
    print(f"  Accuracy        : {det_metrics['accuracy']:.4f}")
    print(f"  Macro F1        : {det_metrics['macro_f1']:.4f}")
    print(f"  Macro Precision : {det_metrics['macro_precision']:.4f}")
    print(f"  Macro Recall    : {det_metrics['macro_recall']:.4f}")
    print(f"  ROC-AUC (macro) : {det_metrics['roc_auc']:.4f}")
    print(f"  PR-AUC  (macro) : {det_metrics['pr_auc']:.4f}")
    print(f"  False Pos. Rate : {det_metrics['fpr']:.4f}")
    print()
    print("  Per-class F1:")
    for cls, v in det_metrics["per_class_f1"].items():
        print(f"    {cls:<20s}: {v:.4f}")

    if n_params is not None:
        print(f"\n  Parameters      : {n_params:,}")
    if mem_mb is not None:
        print(f"  Model memory    : {mem_mb:.2f} MB")

    if latency_results:
        print("\n  Latency / Throughput:")
        for bs, r in latency_results.items():
            print(
                f"    BS={bs:>5d} | "
                f"avg={r['avg_latency_ms']:.3f} ms | "
                f"per-sample={r['per_sample_latency_ms']:.4f} ms | "
                f"{r['throughput_pps']:>10,.0f} pps"
            )
    print(sep)
