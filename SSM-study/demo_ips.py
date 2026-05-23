
"""
demo_ips.py — Vks fix the end-to-end IPS response pipeline after OOM issue.

"""

import argparse
import json
import logging
from pathlib import Path

import torch
from torch.utils.data import DataLoader

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("demo_ips")

from ids_ips.data         import load_unsw_nb15
from ids_ips.models       import MODEL_REGISTRY
from ids_ips.ips          import IPSEngine, IPSAction
from ids_ips.utils.config import TRAIN_CSV, TEST_CSV, NUM_CLASSES, CKPT_DIR, RESULTS_DIR
from ids_ips.utils.gcs    import gcs

def load_model(model_name, n_features, ckpt, device):
    cls = MODEL_REGISTRY[model_name]
    if model_name == "mamba":
        m = cls(n_features=n_features, num_classes=NUM_CLASSES,
                d_model=256, n_layers=4, d_state=16, d_conv=4)
    elif model_name == "rwkv":
        m = cls(n_features=n_features, num_classes=NUM_CLASSES,
                d_model=256, n_layers=4)
    elif model_name == "hybrid":
        m = cls(n_features=n_features, num_classes=NUM_CLASSES,
                d_model=256, n_layers=4, d_state=16, d_conv=4)
    elif model_name == "transformer":
        m = cls(n_features=n_features, num_classes=NUM_CLASSES,
                d_model=256, n_heads=4, n_layers=4)
    else:
        m = cls(n_features=n_features, num_classes=NUM_CLASSES,
                hidden_size=128, n_layers=2, bidirectional=True)
    m.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
    return m

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",      default="mamba",
                        choices=["mamba","rwkv","hybrid","transformer","lstm",
                                 "baseline","all"])
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--n_samples",  type=int, default=2000)
    parser.add_argument("--batch_size", type=int, default=256)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    preprocessor, _, test_ds = load_unsw_nb15(TRAIN_CSV, TEST_CSV)
    n_features = preprocessor.n_features

    MODEL_GROUPS = {
        "baseline": ["transformer", "lstm"],
        "all":      ["mamba", "rwkv", "hybrid", "transformer", "lstm"],
    }
    model_names = MODEL_GROUPS.get(args.model, [args.model])

    for model_name in model_names:
        _run_ips_demo(
            model_name=model_name,
            n_features=n_features,
            test_ds=test_ds,
            n_samples=args.n_samples,
            batch_size=args.batch_size,
            checkpoint=args.checkpoint,
            device=device,
        )

def _run_ips_demo(model_name, n_features, test_ds, n_samples,
                  batch_size, checkpoint, device):
    """Run IPS demo for a single model and save results."""
    ckpt = Path(checkpoint) if checkpoint else \
           CKPT_DIR / f"{model_name}_final.pt"

    if not ckpt.exists():
        logger.error("Checkpoint not found: %s  (run train.py first)", ckpt)
        return

    model = load_model(model_name, n_features, ckpt, device)
    logger.info("Loaded %s from %s", model_name.upper(), ckpt)

    from torch.utils.data import Subset
    idx    = torch.randperm(len(test_ds))[:n_samples].tolist()
    subset = Subset(test_ds, idx)
    loader = DataLoader(subset, batch_size=batch_size, shuffle=False)

    ips = IPSEngine(
        model=model,
        confidence_threshold=0.85,
        rate_limit_threshold=0.70,
        device=device,
    )

    print(f"\nProcessing {n_samples} flow records through IPS engine [{model_name.upper()}] …\n")
    batch_no = 0
    for X, _ in loader:
        batch_no += 1
        src_ids = [f"10.0.{batch_no}.{i+1}" for i in range(len(X))]
        ips.process_batch(X, source_ids=src_ids)

    block_events = [e for e in ips._event_log if e.action == IPSAction.BLOCK]
    print(f"\nSample firewall rules (showing up to 5 BLOCK events):")
    for ev in block_events[:5]:
        print(" ", ips.simulate_firewall_rule(ev))

    ips.print_summary()

    summary = ips.get_summary()

    sample_events = []
    for ev in ips._event_log[:20]:
        sample_events.append({
            "timestamp":       ev.timestamp,
            "predicted_class": ev.predicted_class,
            "confidence":      round(ev.confidence, 4),
            "action":          ev.action.name,
            "source_id":       ev.source_id,
            "firewall_rule":   ips.simulate_firewall_rule(ev),
        })

    firewall_rules = [
        ips.simulate_firewall_rule(ev)
        for ev in ips._event_log
        if ev.action == IPSAction.BLOCK
    ]

    output = {
        "model":          model_name.upper(),
        "n_samples":      n_samples,
        "device":         str(device),
        "summary":        summary,
        "sample_events":  sample_events,
        "firewall_rules": firewall_rules,
    }

    out_path = RESULTS_DIR / f"ips_demo_{model_name}.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)

    logger.info("IPS demo results saved  %s", out_path)
    gcs.upload(out_path, f"results/ips_demo_{model_name}.json")

if __name__ == "__main__":
    main()
