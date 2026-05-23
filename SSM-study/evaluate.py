
"""
TBD evaluate.py — Vks complete checkpoint and run full evaluation on the test .

"""

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("evaluate")

from ids_ips.data       import load_unsw_nb15, ATTACK_CATEGORIES
from ids_ips.models     import MODEL_REGISTRY
from ids_ips.evaluation import (
    predict, compute_detection_metrics, print_report,
    save_confusion_matrix, save_roc_curves, save_model_comparison,
)
from ids_ips.utils.config import TRAIN_CSV, TEST_CSV, NUM_CLASSES, CKPT_DIR, RESULTS_DIR
from ids_ips.utils.gcs import gcs

def load_model(model_name: str, n_features: int, ckpt_path: Path, device: torch.device):
    model_cls = MODEL_REGISTRY[model_name]
    if model_name == "mamba":
        model = model_cls(n_features=n_features, num_classes=NUM_CLASSES,
                          d_model=256, n_layers=4, d_state=16, d_conv=4)
    elif model_name == "rwkv":
        model = model_cls(n_features=n_features, num_classes=NUM_CLASSES,
                          d_model=256, n_layers=4)
    elif model_name == "hybrid":
        model = model_cls(n_features=n_features, num_classes=NUM_CLASSES,
                          d_model=256, n_layers=4, d_state=16, d_conv=4)
    elif model_name == "transformer":
        model = model_cls(n_features=n_features, num_classes=NUM_CLASSES,
                          d_model=256, n_heads=4, n_layers=4)
    elif model_name == "lstm":
        model = model_cls(n_features=n_features, num_classes=NUM_CLASSES,
                          hidden_size=128, n_layers=2, bidirectional=True)
    else:
        raise ValueError(f"Unknown model: {model_name}")

    state = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.to(device)
    return model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",      default="mamba",
                        choices=["mamba","rwkv","hybrid","transformer","lstm","all"])
    parser.add_argument("--checkpoint", default=None,
                        help="Path to .pt checkpoint (auto-detected if omitted)")
    parser.add_argument("--batch",      type=int, default=1024)
    parser.add_argument("--device",     default=None)
    args = parser.parse_args()

    device = torch.device(args.device or (
        "cuda" if torch.cuda.is_available() else "cpu"
    ))
    logger.info("Device: %s", device)

    preprocessor, _, test_ds = load_unsw_nb15(
        TRAIN_CSV, TEST_CSV, use_smote=False
    )
    n_features  = preprocessor.n_features
    test_loader = DataLoader(test_ds, batch_size=args.batch, shuffle=False)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    model_names = (
        ["mamba", "rwkv", "hybrid", "transformer", "lstm"]
        if args.model == "all" else [args.model]
    )

    all_results = {}
    for mname in model_names:
        ckpt = Path(args.checkpoint) if args.checkpoint else \
               CKPT_DIR / f"{mname}_final.pt"
        if not ckpt.exists():
            logger.warning("Checkpoint not found for %s at %s — skipping", mname, ckpt)
            continue

        logger.info("Evaluating %s from %s", mname.upper(), ckpt)
        model  = load_model(mname, n_features, ckpt, device)
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        y_pred, y_prob, y_true = predict(model, test_loader, device)
        det = compute_detection_metrics(y_true, y_pred, y_prob, ATTACK_CATEGORIES)

        print_report(mname.upper(), det, n_params=n_params)
        all_results[mname] = {"detection": det, "n_params": n_params}

        save_confusion_matrix(
            np.array(det["confusion_matrix"]), ATTACK_CATEGORIES,
            f"{mname.upper()} Confusion Matrix",
            RESULTS_DIR / f"{mname}_confusion_matrix.png",
        )
        save_roc_curves(
            y_true, y_prob, ATTACK_CATEGORIES,
            f"{mname.upper()} ROC Curves",
            RESULTS_DIR / f"{mname}_roc_curves.png",
        )

    if len(all_results) > 1:
        save_model_comparison(
            {k.upper(): v["detection"] for k, v in all_results.items()},
            RESULTS_DIR / "model_comparison.png",
        )

    out = RESULTS_DIR / "evaluation_results.json"
    with open(out, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    logger.info("Saved  %s", out)
    gcs.upload(out, "results/evaluation_results.json")

if __name__ == "__main__":
    main()
