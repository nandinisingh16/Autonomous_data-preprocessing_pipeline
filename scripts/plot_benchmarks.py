"""Plot benchmark comparison for execution time and PTMA

Usage:
    python scripts/plot_benchmarks.py --json scripts/benchmark_selected_datasets_results.json --out scripts/benchmark_comparison.png --show

Outputs:
    - saves `benchmark_comparison.png` (or provided --out) with two subplots:
        1) avg execution time (seconds) per dataset grouped by configuration
        2) avg PTMA per dataset grouped by configuration

Requirements: pandas, matplotlib, seaborn (seaborn optional but used if available)
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import pandas as pd

try:
    import seaborn as sns
    _HAS_SNS = True
except Exception:
    _HAS_SNS = False


def load_aggregated(json_path: Path) -> pd.DataFrame:
    with open(json_path, "r", encoding="utf-8") as fh:
        data = json.load(fh)

    rows = []
    aggregated = data.get("aggregated", {})
    for dataset, configs in aggregated.items():
        for config_name, cfg in configs.items():
            # skip if cfg is not a dict
            if not isinstance(cfg, dict):
                continue
            avg_time = cfg.get("avg_time_s")
            ptma = None
            avg_metrics = cfg.get("avg_autonomy_metrics") or {}
            ptma = avg_metrics.get("PTMA")
            eda_success = cfg.get("eda_success_rate")
            rows.append({
                "dataset": dataset,
                "configuration": config_name,
                "avg_time_s": float(avg_time) if avg_time is not None else None,
                "PTMA": float(ptma) if ptma is not None else None,
                "eda_success_rate": float(eda_success) if eda_success is not None else None,
            })

    df = pd.DataFrame(rows)
    # Normalize configuration names for nicer plot labels
    df["configuration"] = df["configuration"].str.replace("_", " ").str.title()
    return df


def plot(df: pd.DataFrame, out_path: Path, show: bool = False) -> None:
    # Set style
    if _HAS_SNS:
        sns.set(style="whitegrid")
    else:
        plt.style.use("seaborn-v0_8")

    configs = sorted(df["configuration"].unique())
    datasets = sorted(df["dataset"].unique())

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(14, 6), constrained_layout=True)

    # Execution time plot
    ax = axes[0]
    width = 0.15
    x = range(len(datasets))
    offsets = [(i - (len(configs)-1)/2) * width for i in range(len(configs))]

    for off, cfg in zip(offsets, configs):
        subset = df[df["configuration"] == cfg]
        times = [subset[subset["dataset"] == ds]["avg_time_s"].values[0] if (subset[subset["dataset"] == ds].shape[0] > 0) else 0 for ds in datasets]
        ax.bar([xi + off for xi in x], times, width=width, label=cfg)

    ax.set_title("Avg Execution Time (s) by Dataset and Configuration")
    ax.set_xticks(list(x))
    ax.set_xticklabels(datasets, rotation=45, ha="right")
    ax.set_ylabel("Seconds")
    ax.legend(title="Configuration")

    # PTMA plot
    ax2 = axes[1]
    for off, cfg in zip(offsets, configs):
        subset = df[df["configuration"] == cfg]
        ptmas = [subset[subset["dataset"] == ds]["PTMA"].values[0] if (subset[subset["dataset"] == ds].shape[0] > 0) else 0 for ds in datasets]
        ax2.bar([xi + off for xi in x], ptmas, width=width, label=cfg)

    ax2.set_title("Avg PTMA by Dataset and Configuration")
    ax2.set_xticks(list(x))
    ax2.set_xticklabels(datasets, rotation=45, ha="right")
    ax2.set_ylabel("PTMA")
    ax2.set_ylim(-0.05, 1.05)

    # Save
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    print(f"Saved plot to {out_path}")

    if show:
        plt.show()


def parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--json", "-j", type=Path, default=(Path(__file__).parent / "benchmark_selected_datasets_results.json"))
    p.add_argument("--out", "-o", type=Path, default=(Path(__file__).parent / "benchmark_comparison.png"))
    p.add_argument("--show", action="store_true", help="Show the plot interactively")
    p.add_argument("--chart", choices=["bar", "line"], default="bar", help="Type of chart to generate (bar or line)")
    p.add_argument("--datasets", "-d", type=str, default="titanic.csv,B_cancer.csv,text_heavy.csv", help="Comma-separated list of datasets to include (file names from JSON)")
    return p.parse_args(argv)


def plot_line_comparison(df: pd.DataFrame, out_path: Path, show: bool = False, selected_datasets: list[str] | None = None) -> None:
    """Plot two charts for selected datasets:
       - left: line chart of average execution time (configurations on x-axis)
       - right: grouped bar chart of PTMA (annotated values)
    """
    if _HAS_SNS:
        sns.set(style="whitegrid")
    else:
        plt.style.use("seaborn-v0_8")

    # canonical configuration order and readable labels
    cfg_order = ["With Llm", "Without Llm", "Without Agents", "Manual Pipeline"]
    label_map = {
        "With Llm": "Agentic w/ LLM",
        "Without Llm": "Agentic w/o LLM",
        "Without Agents": "Without Agents",
        "Manual Pipeline": "Manual",
    }

    # filter to requested datasets if provided
    if selected_datasets is not None:
        df = df[df["dataset"].isin(selected_datasets)]

    # pivot data so rows=config, columns=dataset
    time_pivot = df.pivot(index="configuration", columns="dataset", values="avg_time_s").reindex(cfg_order)
    ptma_pivot = df.pivot(index="configuration", columns="dataset", values="PTMA").reindex(cfg_order)

    datasets = list(time_pivot.columns)
    x = range(len(cfg_order))

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(14, 6), constrained_layout=True)

    # Execution time line chart
    ax = axes[0]
    for ds in datasets:
        y = time_pivot[ds].fillna(0).values
        ax.plot(x, y, marker="o", label=ds)

    ax.set_xticks(list(x))
    ax.set_xticklabels([label_map.get(c, c) for c in cfg_order], rotation=30)
    ax.set_title("Avg Execution Time (s) by Configuration and Dataset")
    ax.set_ylabel("Seconds")
    ax.legend(title="Dataset")

    # PTMA grouped bar chart
    ax2 = axes[1]
    n = len(datasets)
    width = 0.18
    offsets = [(-1.5 + i) * width for i in range(n)]

    for off, ds in zip(offsets, datasets):
        vals = ptma_pivot[ds].fillna(0).values
        bars = ax2.bar([xi + off for xi in x], vals, width=width, label=ds)
        # annotate
        for b in bars:
            h = b.get_height()
            ax2.annotate(f"{h:.3f}", (b.get_x() + b.get_width() / 2, h), ha='center', va='bottom', fontsize=8)

    ax2.set_xticks(list(x))
    ax2.set_xticklabels([label_map.get(c, c) for c in cfg_order], rotation=30)
    ax2.set_title("Avg PTMA by Configuration and Dataset")
    ax2.set_ylabel("PTMA")
    ax2.set_ylim(-0.05, 1.05)
    ax2.legend(title="Dataset")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    print(f"Saved line+bar plot to {out_path}")

    if show:
        plt.show()


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    df = load_aggregated(args.json)
    if df.empty:
        print("No aggregated data found in JSON.")
        return 1

    if args.chart == "bar":
        plot(df, args.out, show=args.show)
    else:
        # change default output filename to indicate line chart
        out = args.out.with_name(args.out.stem + "_line" + args.out.suffix)
        selected = [d.strip() for d in args.datasets.split(",") if d.strip()]
        plot_line_comparison(df, out, show=args.show, selected_datasets=selected)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
