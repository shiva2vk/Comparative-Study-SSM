
"""
benchmark.py — Latency and throughput benchmark for all  models.

Vks try to Measures
--------
• Per-sample latency (ms) at batch sizes 1, 32, 256, 1024
• Sustained throughput (samples/sec)
• Parameter count and memory footprint (MB)

 try to use pytorch profiller , if time permits
"""

import argparse
import json
import logging
from pathlib import Path

import torch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("benchmark")

from ids_ips.models     import MODEL_REGISTRY
from ids_ips.evaluation import benchmark_latency, count_parameters, model_memory_mb
from ids_ips.evaluation.plots import save_throughput_comparison
from ids_ips.utils.config import NUM_CLASSES, RESULTS_DIR
from ids_ips.utils.gcs import gcs

N_FEATURES  = 42

def build_model(name: str, n_features: int):
    if name == "mamba":
        return MODEL_REGISTRY["mamba"](
            n_features=n_features, num_classes=NUM_CLASSES,
            d_model=256, n_layers=4, d_state=16, d_conv=4,
        )
    elif name == "rwkv":
        return MODEL_REGISTRY["rwkv"](
            n_features=n_features, num_classes=NUM_CLASSES,
            d_model=256, n_layers=4,
        )
    elif name == "hybrid":
        return MODEL_REGISTRY["hybrid"](
            n_features=n_features, num_classes=NUM_CLASSES,
            d_model=256, n_layers=4, d_state=16, d_conv=4,
        )
    elif name == "transformer":
        return MODEL_REGISTRY["transformer"](
            n_features=n_features, num_classes=NUM_CLASSES,
            d_model=256, n_heads=4, n_layers=4,
        )
    elif name == "lstm":
        return MODEL_REGISTRY["lstm"](
            n_features=n_features, num_classes=NUM_CLASSES,
            hidden_size=128, n_layers=2, bidirectional=True,
        )

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",   default="all",
                        choices=["mamba","rwkv","hybrid","transformer","lstm","all"])
    parser.add_argument("--n_runs",  type=int, default=500)
    parser.add_argument("--n_warmup",type=int, default=50)
    parser.add_argument("--device",  default=None)
    args = parser.parse_args()

    device = torch.device(args.device or (
        "cuda" if torch.cuda.is_available() else "cpu"
    ))
    logger.info("Benchmark device: %s", device)

    model_names = (
        ["mamba", "rwkv", "hybrid", "transformer", "lstm"]
        if args.model == "all" else [args.model]
    )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    all_bench = {}
    throughput_bs256 = {}

    for mname in model_names:
        logger.info("Benchmarking %s …", mname.upper())
        model  = build_model(mname, N_FEATURES)
        n_par  = count_parameters(model)
        mem_mb = model_memory_mb(model)

        results = benchmark_latency(
            model, N_FEATURES,
            batch_sizes=[1, 32, 256, 1024],
            n_warmup=args.n_warmup,
            n_runs=args.n_runs,
            device=device,
        )

        print(f"\n  {mname.upper()}")
        print(f"  Parameters : {n_par:,}")
        print(f"  Memory     : {mem_mb:.2f} MB")
        print(f"  {'BS':>6}  {'avg ms':>10}  {'per-sample ms':>15}  {'pps':>12}")
        for bs, r in results.items():
            target_ok = "" if r["per_sample_latency_ms"] < 1.0 else ""
            print(
                f"  {bs:>6}  {r['avg_latency_ms']:>10.3f}  "
                f"{r['per_sample_latency_ms']:>15.4f}  "
                f"{r['throughput_pps']:>12,.0f}  {target_ok}"
            )

        all_bench[mname] = {
            "n_params": n_par, "memory_mb": mem_mb, "latency": results
        }
        if 256 in results:
            throughput_bs256[mname.upper()] = results[256]["throughput_pps"]

    if len(throughput_bs256) > 1:
        save_throughput_comparison(
            throughput_bs256,
            RESULTS_DIR / "throughput_comparison.png",
        )

    out = RESULTS_DIR / "benchmark_results.json"
    with open(out, "w") as f:
        json.dump(all_bench, f, indent=2, default=str)
    logger.info("Benchmark saved  %s", out)
    gcs.upload(out, "results/benchmark_results.json")

    print("\n" + "" * 70)
    print(f"  {'Model':<15} {'Params':>10} {'Mem(MB)':>10} "
          f"{'pps@BS256':>14} {'<1ms/sample':>12}")
    print("─" * 70)
    for mname, d in all_bench.items():
        pps = d["latency"].get(256, {}).get("throughput_pps", 0)
        lat = d["latency"].get(256, {}).get("per_sample_latency_ms", 99)
        ok  = "YES" if lat < 1.0 else "NO"
        print(
            f"  {mname.upper():<15} {d['n_params']:>10,} {d['memory_mb']:>10.2f} "
            f"{pps:>14,.0f} {ok:>12}"
        )
    print("" * 70)
    print(f"  Target: >100,000 pps and <1 ms/sample  ( 5.5)")

if __name__ == "__main__":
    main()
