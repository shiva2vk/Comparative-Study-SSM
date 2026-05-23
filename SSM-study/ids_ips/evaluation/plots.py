"""
Visualisation utilities — confusion matrices, ROC curves, training curves,
model comparison bar charts.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
from pathlib import Path
from typing import Dict, List, Optional
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from ids_ips.utils.gcs import gcs

ATTACK_CATEGORIES = [
    "Normal", "Generic", "Exploits", "Fuzzers", "DoS",
    "Reconnaissance", "Analysis", "Backdoor", "Shellcode", "Worms",
]

def save_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str],
    title: str,
    save_path: Path,
):
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.colorbar(im, ax=ax)
    tick_marks = np.arange(len(class_names))
    ax.set_xticks(tick_marks)
    ax.set_yticks(tick_marks)
    ax.set_xticklabels(class_names, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(class_names, fontsize=9)

    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, f"{cm[i,j]:,}",
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black",
                    fontsize=7)

    ax.set_ylabel("True label")
    ax.set_xlabel("Predicted label")
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    gcs.upload(save_path, f"results/{Path(save_path).name}")

def save_roc_curves(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    class_names: List[str],
    title: str,
    save_path: Path,
):
    num_classes = len(class_names)
    y_bin = label_binarize(y_true, classes=list(range(num_classes)))

    fig, ax = plt.subplots(figsize=(10, 8))
    colors = plt.cm.tab10(np.linspace(0, 1, num_classes))

    for i, (name, color) in enumerate(zip(class_names, colors)):
        if y_bin[:, i].sum() == 0:
            continue
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_prob[:, i])
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=color, lw=1.5,
                label=f"{name} (AUC={roc_auc:.3f})")

    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title)
    ax.legend(loc="lower right", fontsize=8)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    gcs.upload(save_path, f"results/{Path(save_path).name}")

def save_training_curves(
    histories: Dict[str, list],
    model_name: str,
    save_path: Path,
):
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    for ax, metric, ylabel in zip(
        axes,
        [("train_loss","val_loss"), ("train_f1","val_f1")],
        ["Loss", "Macro F1"],
    ):
        for key, label, ls in zip(metric, ["Train","Val"], ["-","--"]):
            if key in histories:
                ax.plot(histories[key], linestyle=ls, label=label)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        ax.set_title(f"{model_name} — {ylabel}")
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    gcs.upload(save_path, f"results/{Path(save_path).name}")

def save_model_comparison(
    results: Dict[str, Dict],
    save_path: Path,
):

    model_names = list(results.keys())
    metrics     = ["accuracy", "macro_f1", "roc_auc"]
    labels      = ["Accuracy", "Macro F1", "ROC-AUC"]
    x           = np.arange(len(metrics))
    width       = 0.8 / len(model_names)

    fig, axes = plt.subplots(1, 3, figsize=(14, 6))
    colors = plt.cm.Set2(np.linspace(0, 0.8, len(model_names)))

    for ax, metric, label in zip(axes, metrics, labels):
        all_vals = [res.get(metric, 0.0) for res in results.values()]
        y_min = max(0, min(all_vals) - 0.05)
        y_max = min(1.0, max(all_vals) + 0.08)

        for i, (name, res) in enumerate(results.items()):
            val = res.get(metric, 0.0)
            offset = (i - len(model_names) / 2 + 0.5) * width
            bar = ax.bar(offset, val, width, label=name,
                         color=colors[i], edgecolor="black", linewidth=0.5)
            ax.text(offset, val + (y_max - y_min) * 0.01,
                    f"{val:.4f}", ha="center", va="bottom", fontsize=8, rotation=45)

        ax.set_xlim(-0.5, 0.5)
        ax.set_ylim([y_min, y_max])
        ax.set_xticks([])
        ax.set_ylabel("Score")
        ax.set_title(label, fontsize=12, fontweight='bold')
        ax.grid(True, axis="y", alpha=0.3)
        ax.legend(loc="lower center", fontsize=7,
                  handles=[plt.Rectangle((0,0),1,1, color=colors[i])
                            for i in range(len(model_names))],
                  labels=model_names)

    plt.suptitle("Model Comparison — UNSW-NB15 Test Set", fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    gcs.upload(save_path, f"results/{Path(save_path).name}")

def save_throughput_comparison(
    throughput_data: Dict[str, float],
    save_path: Path,
):
   
    names  = list(throughput_data.keys())
    values = list(throughput_data.values())

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = plt.cm.Set1(np.linspace(0, 0.8, len(names)))
    bars = ax.bar(names, values, color=colors, edgecolor="black")
    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() * 1.01,
                f"{v:,.0f}", ha="center", va="bottom", fontsize=9)

    ax.axhline(100_000, linestyle="--", color="red", label="Target 100K pps")
    ax.set_ylabel("Throughput (samples/sec)")
    ax.set_title("Inference Throughput Comparison")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    gcs.upload(save_path, f"results/{Path(save_path).name}")
