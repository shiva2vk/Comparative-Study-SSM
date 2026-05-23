import json
import logging
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ids_ips.utils.config import RESULTS_DIR
from ids_ips.utils.gcs import gcs

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("extra_plots")

ATTACK_CATEGORIES = [
    "Normal", "Generic", "Exploits", "Fuzzers", "DoS",
    "Reconnaissance", "Analysis", "Backdoor", "Shellcode", "Worms",
]

COLORS  = ["#2ecc71", "#e74c3c", "#9b59b6", "#f39c12", "#3498db"]
MARKERS = ["o", "s", "D", "^", "P"]


def load_results():
    ev_path = RESULTS_DIR / "evaluation_results.json"
    bm_path = RESULTS_DIR / "benchmark_results.json"
    ip_paths = {
        m: RESULTS_DIR / f"ips_demo_{m}.json"
        for m in ["mamba", "rwkv", "hybrid", "transformer", "lstm"]
    }
    with open(ev_path) as f: ev = json.load(f)
    with open(bm_path) as f: bm = json.load(f)
    ips = {}
    for m, p in ip_paths.items():
        if p.exists():
            with open(p) as f: ips[m] = json.load(f)
    return ev, bm, ips


def save_class_difficulty(ev, save_path):
    models = list(ev.keys())
    avg_f1, min_f1, max_f1 = [], [], []

    for cls in ATTACK_CATEGORIES:
        vals = [ev[m]["detection"]["per_class_f1"].get(cls, 0.0) for m in models]
        avg_f1.append(np.mean(vals))
        min_f1.append(np.min(vals))
        max_f1.append(np.max(vals))

    x = np.arange(len(ATTACK_CATEGORIES))
    bar_colors = [
        "#e74c3c" if v < 0.2 else "#f39c12" if v < 0.5 else "#2ecc71"
        for v in avg_f1
    ]

    fig, ax = plt.subplots(figsize=(13, 6))
    bars = ax.bar(x, avg_f1, color=bar_colors, edgecolor="black",
                  linewidth=0.5, alpha=0.85)
    ax.errorbar(
        x, avg_f1,
        yerr=[np.array(avg_f1) - np.array(min_f1),
              np.array(max_f1) - np.array(avg_f1)],
        fmt="none", color="black", capsize=5, linewidth=1.5,
    )
    for bar, v in zip(bars, avg_f1):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{v:.3f}", ha="center", va="bottom",
                fontsize=9, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(ATTACK_CATEGORIES, rotation=30, ha="right", fontsize=10)
    ax.set_ylabel("Average F1 Score (across all 5 models)", fontsize=11)
    ax.set_title(
        "Per-Class Detection Difficulty - Average F1 and Range Across All Models",
        fontsize=12, fontweight="bold",
    )
    ax.axhline(0.5, color="orange", linestyle="--", alpha=0.6, label="F1=0.5")
    ax.axhline(0.8, color="green",  linestyle="--", alpha=0.6, label="F1=0.8")
    ax.legend(fontsize=9)
    ax.grid(True, axis="y", alpha=0.3)
    ax.set_ylim(0, 1.05)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    logger.info("Saved: %s", save_path)
    gcs.upload(save_path, f"results/{save_path.name}")


def save_detection_vs_speed(ev, bm, save_path):
    models   = list(ev.keys())
    accuracy = [ev[m]["detection"]["accuracy"] for m in models]
    f1       = [ev[m]["detection"]["macro_f1"] for m in models]
    tput     = [
        bm[m]["latency"].get("256", bm[m]["latency"].get(256, {}))
        .get("throughput_pps", 0)
        for m in models
    ]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for ax, y_vals, y_label, title in zip(
        axes,
        [accuracy, f1],
        ["Accuracy", "Macro F1"],
        ["Accuracy vs Throughput (pps)", "Macro F1 vs Throughput (pps)"],
    ):
        for i, (m, x, y) in enumerate(zip(models, tput, y_vals)):
            ax.scatter(x, y, s=200, color=COLORS[i], marker=MARKERS[i],
                       zorder=5, label=m.upper(),
                       edgecolors="black", linewidth=1)
            ax.annotate(m.upper(), (x, y),
                        textcoords="offset points", xytext=(8, 5),
                        fontsize=9, fontweight="bold", color=COLORS[i])

        ax.axvline(100_000, color="red", linestyle="--",
                   alpha=0.7, label="100K pps target")
        ax.set_xlabel("Throughput @BS256 (samples/sec)", fontsize=11)
        ax.set_ylabel(y_label, fontsize=11)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.legend(fontsize=8, loc="lower right")
        ax.grid(True, alpha=0.3)
        ax.set_xscale("log")

    plt.suptitle(
        "Detection Quality vs Inference Speed - All Models (UNSW-NB15)",
        fontsize=13, fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    logger.info("Saved: %s", save_path)
    gcs.upload(save_path, f"results/{save_path.name}")


def save_latency_scaling(bm, save_path):
    bs_vals = [1, 32, 256, 1024]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for i, (model, data) in enumerate(bm.items()):
        lat  = [data["latency"].get(str(bs), data["latency"].get(bs, {}))
                .get("per_sample_latency_ms", None) for bs in bs_vals]
        tput = [data["latency"].get(str(bs), data["latency"].get(bs, {}))
                .get("throughput_pps", None) for bs in bs_vals]

        axes[0].plot(bs_vals, lat,  marker=MARKERS[i], color=COLORS[i],
                     label=model.upper(), linewidth=2, markersize=8)
        axes[1].plot(bs_vals, tput, marker=MARKERS[i], color=COLORS[i],
                     label=model.upper(), linewidth=2, markersize=8)

    axes[0].axhline(1.0, color="red", linestyle="--", alpha=0.7, label="1ms target")
    axes[0].set_xlabel("Batch Size", fontsize=11)
    axes[0].set_ylabel("Per-Sample Latency (ms)", fontsize=11)
    axes[0].set_title("Per-Sample Latency vs Batch Size", fontsize=12, fontweight="bold")
    axes[0].set_yscale("log")
    axes[0].set_xscale("log")
    axes[0].set_xticks(bs_vals)
    axes[0].set_xticklabels(bs_vals)
    axes[0].legend(fontsize=9)
    axes[0].grid(True, alpha=0.3)

    axes[1].axhline(100_000, color="red", linestyle="--",
                    alpha=0.7, label="100K pps target")
    axes[1].set_xlabel("Batch Size", fontsize=11)
    axes[1].set_ylabel("Throughput (samples/sec)", fontsize=11)
    axes[1].set_title("Throughput vs Batch Size", fontsize=12, fontweight="bold")
    axes[1].set_xscale("log")
    axes[1].set_xticks(bs_vals)
    axes[1].set_xticklabels(bs_vals)
    axes[1].legend(fontsize=9)
    axes[1].grid(True, alpha=0.3)

    plt.suptitle(
        "Inference Latency and Throughput Scaling - All Batch Sizes",
        fontsize=13, fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    logger.info("Saved: %s", save_path)
    gcs.upload(save_path, f"results/{save_path.name}")


def save_ips_action_comparison(ips, save_path):
    models  = list(ips.keys())
    actions = ["ALLOW", "ALERT", "RATE_LIMIT", "BLOCK"]
    colors  = ["#2ecc71", "#f39c12", "#e67e22", "#e74c3c"]

    data = {
        m: {a: ips[m]["summary"]["action_breakdown"].get(a, 0) * 100
            for a in actions}
        for m in models
    }

    x     = np.arange(len(models))
    width = 0.18
    fig, ax = plt.subplots(figsize=(13, 6))

    for i, (action, color) in enumerate(zip(actions, colors)):
        vals   = [data[m][action] for m in models]
        offset = (i - 1.5) * width
        bars   = ax.bar(x + offset, vals, width, label=action,
                        color=color, edgecolor="black",
                        linewidth=0.5, alpha=0.85)
        for bar, v in zip(bars, vals):
            if v > 3:
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.5,
                        f"{v:.1f}%", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels([m.upper() for m in models], fontsize=11)
    ax.set_ylabel("Percentage of Flows (%)", fontsize=11)
    ax.set_title(
        "IPS Action Distribution - All 5 Models (2,000 test flows each)",
        fontsize=12, fontweight="bold",
    )
    ax.legend(title="IPS Action", fontsize=10)
    ax.grid(True, axis="y", alpha=0.3)
    ax.set_ylim(0, 50)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    logger.info("Saved: %s", save_path)
    gcs.upload(save_path, f"results/{save_path.name}")


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    logger.info("Loading results from %s", RESULTS_DIR)
    ev, bm, ips = load_results()

    logger.info("Generating class difficulty plot...")
    save_class_difficulty(ev, RESULTS_DIR / "class_difficulty.png")

    logger.info("Generating detection vs speed scatter plot...")
    save_detection_vs_speed(ev, bm, RESULTS_DIR / "detection_vs_speed.png")

    logger.info("Generating latency scaling plot...")
    save_latency_scaling(bm, RESULTS_DIR / "latency_scaling.png")

    logger.info("Generating IPS action comparison plot...")
    if ips:
        save_ips_action_comparison(ips, RESULTS_DIR / "ips_action_comparison.png")
    else:
        logger.warning("No IPS demo results found - skipping ips_action_comparison.png")

    logger.info("All extra plots saved to %s", RESULTS_DIR)


if __name__ == "__main__":
    main()
