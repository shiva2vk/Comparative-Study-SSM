
"""
Vks use train.py formate for training — Main training script for IDS/IPS models.
"""

import argparse
import json
import logging
import random
import sys
from pathlib import Path

import numpy as np
import torch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("train")

from ids_ips.data   import load_unsw_nb15, NetworkFlowDataset, ATTACK_CATEGORIES
from ids_ips.models import MODEL_REGISTRY
from ids_ips.training import Trainer
from ids_ips.evaluation import (
    predict, compute_detection_metrics, print_report,
    save_training_curves, save_confusion_matrix, save_roc_curves,
)
from ids_ips.utils.config import TRAIN_CSV, TEST_CSV, NUM_CLASSES, TrainConfig, CKPT_DIR, RESULTS_DIR
from ids_ips.utils.gcs import gcs

SAVE_DIR = CKPT_DIR

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def get_device(requested=None) -> torch.device:
    if requested:
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

def build_model(model_name: str, n_features: int):

    if model_name == "mamba":
        return MODEL_REGISTRY["mamba"](
            n_features=n_features, num_classes=NUM_CLASSES,
            d_model=256, n_layers=4, d_state=16, d_conv=4, expand=2, dropout=0.1,
        )
    elif model_name == "rwkv":

        return MODEL_REGISTRY["rwkv"](
            n_features=n_features, num_classes=NUM_CLASSES,
            d_model=256, n_layers=4, dropout=0.1,
        )
    elif model_name == "hybrid":

        return MODEL_REGISTRY["hybrid"](
            n_features=n_features, num_classes=NUM_CLASSES,
            d_model=256, n_layers=4, d_state=16, d_conv=4, expand=2, dropout=0.1,
        )
    elif model_name == "transformer":
        return MODEL_REGISTRY["transformer"](
            n_features=n_features, num_classes=NUM_CLASSES,
            d_model=256, n_heads=4, n_layers=4, ff_dim=512, dropout=0.1,
        )
    elif model_name == "lstm":

        return MODEL_REGISTRY["lstm"](
            n_features=n_features, num_classes=NUM_CLASSES,
            hidden_size=128, n_layers=2, dropout=0.1, bidirectional=True,
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")

def train_one(
    model_name: str,
    train_ds: NetworkFlowDataset,
    test_ds:  NetworkFlowDataset,
    y_train_np: np.ndarray,
    n_features: int,
    cfg: TrainConfig,
    device: torch.device,
    run_kfold: bool,
) -> dict:
    
    logger.info(" Training %s ", model_name.upper())

    model = build_model(model_name, n_features)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("Parameters: %s", f"{n_params:,}")

    trainer = Trainer(
        model        = model,
        epochs       = cfg.epochs,
        batch_size   = cfg.batch_size,
        lr           = cfg.lr,
        weight_decay = cfg.weight_decay,
        focal_gamma  = cfg.focal_gamma,
        grad_clip    = cfg.grad_clip,
        use_amp      = cfg.use_amp,
        n_folds      = cfg.n_folds,
        save_dir     = SAVE_DIR,
        y_train      = y_train_np,
        num_classes  = NUM_CLASSES,
        device       = device,
        model_name   = model_name,
    )

    if run_kfold:
        cv_results = trainer.fit_kfold(train_ds)
        history    = cv_results["fold_histories"][0]
    else:
        from torch.utils.data import random_split
        val_size  = int(0.1 * len(train_ds))
        tr_size   = len(train_ds) - val_size
        tr_sub, vl_sub = random_split(
            train_ds, [tr_size, val_size],
            generator=torch.Generator().manual_seed(cfg.seed),
        )
        history = trainer.fit(tr_sub, vl_sub, fold_idx=0)

    from torch.utils.data import DataLoader
    test_loader = DataLoader(test_ds, batch_size=cfg.batch_size * 2, shuffle=False)
    y_pred, y_prob, y_true = predict(model, test_loader, device)
    det = compute_detection_metrics(y_true, y_pred, y_prob, ATTACK_CATEGORIES)

    print_report(model_name.upper(), det, n_params=n_params)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    save_training_curves(
        history, model_name.upper(),
        RESULTS_DIR / f"{model_name}_training_curves.png"
    )
    save_confusion_matrix(
        np.array(det["confusion_matrix"]),
        ATTACK_CATEGORIES,
        f"{model_name.upper()} Confusion Matrix",
        RESULTS_DIR / f"{model_name}_confusion_matrix.png",
    )
    save_roc_curves(
        y_true, y_prob, ATTACK_CATEGORIES,
        f"{model_name.upper()} ROC Curves",
        RESULTS_DIR / f"{model_name}_roc_curves.png",
    )

    ckpt = SAVE_DIR / f"{model_name}_final.pt"
    torch.save(model.state_dict(), ckpt)
    logger.info("Model saved  %s", ckpt)
    gcs.upload(ckpt, f"checkpoints/{ckpt.name}")

    return {"detection": det, "n_params": n_params}

def main():
    parser = argparse.ArgumentParser(description="IDS/IPS Model Trainer")
    parser.add_argument("--model",  default="mamba",
                        choices=["mamba","rwkv","hybrid","transformer","lstm","all"])
    parser.add_argument("--epochs", type=int,   default=30)
    parser.add_argument("--batch",  type=int,   default=512)
    parser.add_argument("--lr",     type=float, default=3e-4)
    parser.add_argument("--kfold",  action="store_true")
    parser.add_argument("--smote",  action="store_true")
    parser.add_argument("--seed",   type=int,   default=42)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    set_seed(args.seed)
    device = get_device(args.device)
    logger.info("Device: %s", device)

    preprocessor, train_ds, test_ds = load_unsw_nb15(
        TRAIN_CSV, TEST_CSV, use_smote=args.smote, seed=args.seed
    )
    n_features  = preprocessor.n_features
    y_train_np  = train_ds.y.numpy()
    logger.info("Features per flow: %d", n_features)

    cfg = TrainConfig(
        epochs=args.epochs,
        batch_size=args.batch,
        lr=args.lr,
    )

    model_names = (
        ["mamba", "rwkv", "hybrid", "transformer", "lstm"]
        if args.model == "all" else [args.model]
    )

    all_results = {}
    for mname in model_names:
        res = train_one(
            mname, train_ds, test_ds, y_train_np,
            n_features, cfg, device, args.kfold
        )
        all_results[mname] = res

    if len(all_results) > 1:
        from ids_ips.evaluation.plots import save_model_comparison
        comparison_data = {
            k.upper(): v["detection"] for k, v in all_results.items()
        }
        save_model_comparison(
            comparison_data,
            RESULTS_DIR / "model_comparison.png"
        )

    out_json = RESULTS_DIR / "results.json"
    with open(out_json, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    logger.info("Results saved  %s", out_json)
    gcs.upload(out_json, "results/results.json")

if __name__ == "__main__":
    main()
