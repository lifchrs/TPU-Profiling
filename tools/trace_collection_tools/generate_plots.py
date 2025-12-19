#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


PARQUET_PATH = Path("/Users/adhityapolavaram/Desktop/experiments/communication_results.parquet")
OUT_DIR_FULL = PARQUET_PATH.parent / "cs5470_plots_full"
OUT_DIR_CURATED = PARQUET_PATH.parent / "cs5470_plots_curated"


def ensure_sorted_numeric(series: pd.Series) -> list:
    """Sort batch/tp values numerically when possible, otherwise lexicographically."""
    def key(x):
        try:
            return (0, int(x))
        except (TypeError, ValueError):
            return (1, str(x))

    return sorted(series.dropna().unique(), key=key)


def bar_total_by_comm(df: pd.DataFrame, title: str, fname: Path) -> None:
    agg = df.groupby("comm_type", as_index=False)["total_us"].sum()
    plt.figure(figsize=(8, 4))
    sns.barplot(data=agg, x="comm_type", y="total_us", estimator=sum, hue=None)
    plt.title(title)
    plt.xticks(rotation=45, ha="right")
    plt.xlabel("Communication Type")
    plt.ylabel("Total Communication Time (μs)")
    plt.tight_layout()
    fname.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(fname)
    plt.close()


def line_total_vs_batch(df: pd.DataFrame, title: str, fname: Path) -> None:
    # Total time per batch aggregated over tp for each comm_type.
    agg = df.groupby(["batch", "comm_type"], as_index=False)["total_us"].sum()
    plt.figure(figsize=(8, 4))
    sns.lineplot(data=agg, x="batch", y="total_us", hue="comm_type", marker="o")
    plt.title(title)
    plt.ylabel("Total Communication Time (μs)")
    plt.xlabel("Batch Size")
    plt.legend(title="Communication Type")
    plt.tight_layout()
    fname.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(fname)
    plt.close()


def line_total_vs_tp(df: pd.DataFrame, title: str, fname: Path) -> None:
    agg = df.groupby(["tp", "comm_type"], as_index=False)["total_us"].sum()
    plt.figure(figsize=(8, 4))
    sns.lineplot(data=agg, x="tp", y="total_us", hue="comm_type", marker="o")
    plt.title(title)
    plt.ylabel("Total Communication Time (μs)")
    plt.xlabel("Tensor Parallelism Degree (TP)")
    plt.legend(title="Communication Type")
    plt.tight_layout()
    fname.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(fname)
    plt.close()


def line_total_vs_batch_by_tp_comm(df: pd.DataFrame, title: str, fname: Path) -> None:
    """Line plot of total_us vs batch, colored by comm_type and styled by tp."""
    agg = df.groupby(["batch", "tp", "comm_type"], as_index=False)["total_us"].sum()
    agg["batch"] = pd.Categorical(agg["batch"], categories=ensure_sorted_numeric(agg["batch"]), ordered=True)
    agg["tp"] = pd.Categorical(agg["tp"], categories=ensure_sorted_numeric(agg["tp"]), ordered=True)
    plt.figure(figsize=(10, 5))
    sns.lineplot(data=agg, x="batch", y="total_us", hue="comm_type", style="tp", markers=True)
    plt.title(title)
    plt.ylabel("Total Communication Time (μs)")
    plt.xlabel("Batch Size")
    plt.legend(title="Communication Type", title_fontsize=10, fontsize=9)
    plt.tight_layout()
    fname.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(fname)
    plt.close()


def line_total_vs_tp_by_batch_comm(df: pd.DataFrame, title: str, fname: Path) -> None:
    """Line plot of total_us vs tp, colored by comm_type and styled by batch."""
    agg = df.groupby(["tp", "batch", "comm_type"], as_index=False)["total_us"].sum()
    agg["tp"] = pd.Categorical(agg["tp"], categories=ensure_sorted_numeric(agg["tp"]), ordered=True)
    agg["batch"] = pd.Categorical(agg["batch"], categories=ensure_sorted_numeric(agg["batch"]), ordered=True)
    plt.figure(figsize=(10, 5))
    sns.lineplot(data=agg, x="tp", y="total_us", hue="comm_type", style="batch", markers=True)
    plt.title(title)
    plt.ylabel("Total Communication Time (μs)")
    plt.xlabel("Tensor Parallelism Degree (TP)")
    plt.legend(title="Communication Type", title_fontsize=10, fontsize=9)
    plt.tight_layout()
    fname.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(fname)
    plt.close()


def facet_batch_scaling(df: pd.DataFrame, title: str, fname: Path) -> None:
    """Small multiples: total_us vs batch, faceted by comm_type, hue=tp."""
    agg = df.groupby(["batch", "tp", "comm_type"], as_index=False)["total_us"].sum()
    agg["batch"] = pd.Categorical(agg["batch"], categories=ensure_sorted_numeric(agg["batch"]), ordered=True)
    g = sns.relplot(
        data=agg,
        x="batch",
        y="total_us",
        hue="tp",
        col="comm_type",
        kind="line",
        marker="o",
        col_wrap=3,
        facet_kws={"sharey": False},
    )
    g.fig.suptitle(title, y=1.02)
    g.set_axis_labels("Batch Size", "Total Communication Time (μs)")
    if g._legend is not None:
        g._legend.set_title("Tensor Parallelism (TP)")
    fname.parent.mkdir(parents=True, exist_ok=True)
    g.savefig(fname, bbox_inches="tight")
    plt.close()


def facet_tp_scaling(df: pd.DataFrame, title: str, fname: Path) -> None:
    """Small multiples: total_us vs tp, faceted by comm_type, hue=batch."""
    agg = df.groupby(["tp", "batch", "comm_type"], as_index=False)["total_us"].sum()
    agg["tp"] = pd.Categorical(agg["tp"], categories=ensure_sorted_numeric(agg["tp"]), ordered=True)
    g = sns.relplot(
        data=agg,
        x="tp",
        y="total_us",
        hue="batch",
        col="comm_type",
        kind="line",
        marker="o",
        col_wrap=3,
        facet_kws={"sharey": False},
    )
    g.fig.suptitle(title, y=1.02)
    g.set_axis_labels("Tensor Parallelism Degree (TP)", "Total Communication Time (μs)")
    if g._legend is not None:
        g._legend.set_title("Batch Size")
    fname.parent.mkdir(parents=True, exist_ok=True)
    g.savefig(fname, bbox_inches="tight")
    plt.close()


def stacked_area_over_batch(df: pd.DataFrame, title: str, fname: Path) -> None:
    """Stacked area showing comm breakdown across batches (aggregated over tp)."""
    pivot = df.pivot_table(index="batch", columns="comm_type", values="total_us", aggfunc="sum").fillna(0)
    pivot = pivot.loc[ensure_sorted_numeric(pivot.index)]
    plt.figure(figsize=(10, 5))
    plt.stackplot(pivot.index, pivot.T, labels=pivot.columns)
    plt.legend(loc="upper left", bbox_to_anchor=(1.01, 1), borderaxespad=0, title="Communication Type")
    plt.title(title)
    plt.ylabel("Total Communication Time (μs)")
    plt.xlabel("Batch Size")
    plt.tight_layout()
    fname.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(fname, bbox_inches="tight")
    plt.close()


def stacked_area_over_tp(df: pd.DataFrame, title: str, fname: Path) -> None:
    """Stacked area showing comm breakdown across tp (aggregated over batch)."""
    pivot = df.pivot_table(index="tp", columns="comm_type", values="total_us", aggfunc="sum").fillna(0)
    pivot = pivot.loc[ensure_sorted_numeric(pivot.index)]
    plt.figure(figsize=(10, 5))
    plt.stackplot(pivot.index, pivot.T, labels=pivot.columns)
    plt.legend(loc="upper left", bbox_to_anchor=(1.01, 1), borderaxespad=0, title="Communication Type")
    plt.title(title)
    plt.ylabel("Total Communication Time (μs)")
    plt.xlabel("Tensor Parallelism Degree (TP)")
    plt.tight_layout()
    fname.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(fname, bbox_inches="tight")
    plt.close()


def stacked_by_tp(df: pd.DataFrame, title: str, fname: Path) -> None:
    pivot = df.pivot_table(index="tp", columns="comm_type", values="total_us", aggfunc="sum").fillna(0)
    pivot = pivot.loc[ensure_sorted_numeric(pivot.index)]
    pivot.plot(kind="bar", stacked=True, figsize=(10, 5))
    plt.title(title)
    plt.ylabel("Total Communication Time (μs)")
    plt.xlabel("Tensor Parallelism Degree (TP)")
    plt.legend(title="Communication Type", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    fname.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(fname)
    plt.close()


def box_avg_us_per_call(df: pd.DataFrame, title: str, fname: Path) -> None:
    df = df.assign(avg_us_per_call=df["total_us"] / df["count"].clip(lower=1))
    plt.figure(figsize=(8, 4))
    sns.boxplot(data=df, x="comm_type", y="avg_us_per_call")
    plt.title(title)
    plt.xticks(rotation=45, ha="right")
    plt.xlabel("Communication Type")
    plt.ylabel("Average Time per Call (μs)")
    plt.tight_layout()
    fname.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(fname)
    plt.close()


def violin_avg_us_per_call(df: pd.DataFrame, title: str, fname: Path, log_scale: bool = True) -> None:
    df = df.assign(avg_us_per_call=df["total_us"] / df["count"].clip(lower=1))
    plt.figure(figsize=(8, 4))
    sns.violinplot(
        data=df,
        x="comm_type",
        y="avg_us_per_call",
        cut=0,
        density_norm="width",
        inner="quartile",
    )
    if log_scale:
        plt.yscale("log")
    plt.title(title)
    plt.xticks(rotation=45, ha="right")
    plt.xlabel("Communication Type")
    plt.ylabel("Average Time per Call (μs)")
    plt.tight_layout()
    fname.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(fname)
    plt.close()


def topn_comm_types(df: pd.DataFrame, title: str, fname: Path, metric: str, n: int = 10) -> None:
    if metric == "avg_us_per_call":
        df = df.assign(avg_us_per_call=df["total_us"] / df["count"].clip(lower=1))
    agg = df.groupby("comm_type", as_index=False)[metric].sum().sort_values(metric, ascending=False).head(n)
    plt.figure(figsize=(8, 4))
    sns.barplot(data=agg, x="comm_type", y=metric)
    plt.title(title)
    plt.xticks(rotation=45, ha="right")
    plt.xlabel("Communication Type")
    if metric == "total_us":
        plt.ylabel("Total Communication Time (μs)")
    elif metric == "avg_us_per_call":
        plt.ylabel("Average Time per Call (μs)")
    else:
        plt.ylabel(metric)
    plt.tight_layout()
    fname.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(fname)
    plt.close()


def speedup_vs_tp(df: pd.DataFrame, title: str, fname: Path) -> None:
    """Compute speedup relative to tp==1 baseline (if available) for total_us."""
    has_tp1 = (df["tp"] == "1").any()
    if not has_tp1:
        return
    base = (
        df[df["tp"] == "1"]
        .groupby(["batch", "comm_type"], as_index=False)["total_us"]
        .sum()
        .rename(columns={"total_us": "base_us"})
    )
    merged = (
        df.groupby(["tp", "batch", "comm_type"], as_index=False)["total_us"]
        .sum()
        .merge(base, on=["batch", "comm_type"], how="left")
    )
    merged = merged[merged["base_us"] > 0]
    merged["speedup"] = merged["base_us"] / merged["total_us"]
    merged["tp"] = pd.Categorical(merged["tp"], categories=ensure_sorted_numeric(merged["tp"]), ordered=True)
    plt.figure(figsize=(10, 5))
    sns.lineplot(data=merged, x="tp", y="speedup", hue="comm_type", style="batch", marker="o")
    plt.title(title)
    plt.ylabel("Speedup vs TP=1 Baseline")
    plt.xlabel("Tensor Parallelism Degree (TP)")
    plt.legend(title="Communication Type", title_fontsize=10, fontsize=9)
    plt.tight_layout()
    fname.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(fname)
    plt.close()


def write_story_readme(model: str, dest_dir: Path) -> None:
    """Write a short README describing the intent of curated plots."""
    lines = [
        f"# {model} – Communication Plots",
        "",
        "- `total_us_by_comm_type`: Which collectives dominate total comm time.",
        "- `total_us_vs_batch`: How comm cost scales with batch at fixed/agg TP.",
        "- `total_us_vs_tp`: How comm cost scales with TP (parallelism).",
        "- `total_us_vs_batch_by_tp_comm`: Batch scaling by TP and comm type.",
        "- `total_us_vs_tp_by_batch_comm`: TP scaling by batch and comm type.",
        "- `stacked_total_us_by_tp`: Breakdown across TP sizes.",
        "- `stacked_area_over_batch`: Breakdown across batch sizes.",
        "- `stacked_area_over_tp`: Breakdown across TP sizes (area).",
        "- `avg_us_per_call_box` / `violin`: Efficiency distribution; long tails.",
        "- `top_comm_types_*`: Which collectives are hottest by total time/avg call.",
        "- `speedup_vs_tp`: Speedup vs TP=1 baseline (if TP=1 data exists).",
    ]
    dest_dir.mkdir(parents=True, exist_ok=True)
    (dest_dir / "README.md").write_text("\n".join(lines))


def generate_full_for_model(df: pd.DataFrame, model: str, out_dir: Path) -> None:
    model_token = model.replace(" ", "_").lower()
    subset = df[df["model"] == model]

    # Aggregate plots across comm types.
    bar_total_by_comm(
        subset,
        title=f"{model} | Communication Time by Type",
        fname=out_dir / model_token / "total_us_by_comm_type.png",
    )
    line_total_vs_batch(
        subset,
        title=f"{model} | Communication Time vs Batch Size",
        fname=out_dir / model_token / "total_us_vs_batch.png",
    )
    line_total_vs_tp(
        subset,
        title=f"{model} | Communication Time vs Tensor Parallelism",
        fname=out_dir / model_token / "total_us_vs_tp.png",
    )
    line_total_vs_batch_by_tp_comm(
        subset,
        title=f"{model} | Communication Time vs Batch Size by TP and Communication Type",
        fname=out_dir / model_token / "total_us_vs_batch_by_tp_comm.png",
    )
    line_total_vs_tp_by_batch_comm(
        subset,
        title=f"{model} | Communication Time vs Tensor Parallelism by Batch and Communication Type",
        fname=out_dir / model_token / "total_us_vs_tp_by_batch_comm.png",
    )
    facet_batch_scaling(
        subset,
        title=f"{model} | Batch Scaling by Communication Type",
        fname=out_dir / model_token / "facet_batch_scaling.png",
    )
    facet_tp_scaling(
        subset,
        title=f"{model} | Tensor Parallelism Scaling by Communication Type",
        fname=out_dir / model_token / "facet_tp_scaling.png",
    )
    stacked_by_tp(
        subset,
        title=f"{model} | Communication Breakdown by Tensor Parallelism Degree",
        fname=out_dir / model_token / "stacked_total_us_by_tp.png",
    )
    stacked_area_over_batch(
        subset,
        title=f"{model} | Communication Breakdown vs Batch Size",
        fname=out_dir / model_token / "stacked_area_over_batch.png",
    )
    stacked_area_over_tp(
        subset,
        title=f"{model} | Communication Breakdown vs Tensor Parallelism",
        fname=out_dir / model_token / "stacked_area_over_tp.png",
    )
    box_avg_us_per_call(
        subset,
        title=f"{model} | Average Latency per Communication Call",
        fname=out_dir / model_token / "avg_us_per_call_box.png",
    )
    violin_avg_us_per_call(
        subset,
        title=f"{model} | Average Latency per Communication Call Distribution",
        fname=out_dir / model_token / "avg_us_per_call_violin_log.png",
        log_scale=True,
    )
    topn_comm_types(
        subset,
        title=f"{model} | Dominant Communication Types by Total Time",
        fname=out_dir / model_token / "top_comm_types_total_us.png",
        metric="total_us",
    )
    topn_comm_types(
        subset,
        title=f"{model} | Dominant Communication Types by Average Latency",
        fname=out_dir / model_token / "top_comm_types_avg_us_per_call.png",
        metric="avg_us_per_call",
    )
    speedup_vs_tp(
        subset,
        title=f"{model} | Communication Speedup vs TP=1 Baseline",
        fname=out_dir / model_token / "speedup_vs_tp.png",
    )


def generate_curated_for_model(df: pd.DataFrame, model: str, out_dir: Path) -> None:
    model_token = model.replace(" ", "_").lower()
    subset = df[df["model"] == model]

    bar_total_by_comm(
        subset,
        title=f"{model} | Communication Time by Type",
        fname=out_dir / model_token / "total_us_by_comm_type.png",
    )
    stacked_by_tp(
        subset,
        title=f"{model} | Communication Breakdown by Tensor Parallelism Degree",
        fname=out_dir / model_token / "stacked_total_us_by_tp.png",
    )
    stacked_area_over_batch(
        subset,
        title=f"{model} | Communication Breakdown vs Batch Size",
        fname=out_dir / model_token / "stacked_area_over_batch.png",
    )
    line_total_vs_tp_by_batch_comm(
        subset,
        title=f"{model} | Tensor Parallelism Scaling by Batch and Communication Type",
        fname=out_dir / model_token / "total_us_vs_tp_by_batch_comm.png",
    )
    line_total_vs_batch_by_tp_comm(
        subset,
        title=f"{model} | Batch Scaling by Tensor Parallelism and Communication Type",
        fname=out_dir / model_token / "total_us_vs_batch_by_tp_comm.png",
    )
    violin_avg_us_per_call(
        subset,
        title=f"{model} | Average Latency per Communication Call Distribution",
        fname=out_dir / model_token / "avg_us_per_call_violin_log.png",
        log_scale=True,
    )
    topn_comm_types(
        subset,
        title=f"{model} | Dominant Communication Types by Total Time",
        fname=out_dir / model_token / "top_comm_types_total_us.png",
        metric="total_us",
    )
    speedup_vs_tp(
        subset,
        title=f"{model} | Communication Speedup vs TP=1 Baseline",
        fname=out_dir / model_token / "speedup_vs_tp.png",
    )
    write_story_readme(model, out_dir / model_token)


# Model sizes in billions of parameters for cross-model analysis
MODEL_SIZES = {
    "Qwen_Qwen3-4B": 4,
    "meta-llama_Llama-3.1-8B": 8,
    "Qwen_Qwen3-32B": 32,
    "meta-llama_Llama-3.3-70B-Instruct": 70,
}


def cross_model_total_time(df: pd.DataFrame, out_dir: Path) -> None:
    """Bar chart comparing total communication time across models, ordered by size."""
    agg = df.groupby("model", as_index=False)["total_us"].sum()
    agg["params_b"] = agg["model"].map(MODEL_SIZES)
    agg = agg.sort_values("params_b")
    agg["model_label"] = agg.apply(lambda r: f"{r['model'].split('_')[-1]} ({r['params_b']}B)", axis=1)

    plt.figure(figsize=(10, 5))
    sns.barplot(data=agg, x="model_label", y="total_us", hue="model_label", palette="viridis", legend=False)
    plt.title("Total Communication Time by Model Size")
    plt.ylabel("Total Communication Time (μs)")
    plt.xlabel("Model (Parameters)")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_dir / "total_time_by_model_size.png")
    plt.close()


def cross_model_efficiency(df: pd.DataFrame, out_dir: Path) -> None:
    """Communication time per billion parameters - efficiency metric."""
    agg = df.groupby("model", as_index=False)["total_us"].sum()
    agg["params_b"] = agg["model"].map(MODEL_SIZES)
    agg["us_per_billion"] = agg["total_us"] / agg["params_b"]
    agg = agg.sort_values("params_b")
    agg["model_label"] = agg.apply(lambda r: f"{r['model'].split('_')[-1]} ({r['params_b']}B)", axis=1)

    plt.figure(figsize=(10, 5))
    sns.barplot(data=agg, x="model_label", y="us_per_billion", hue="model_label", palette="magma", legend=False)
    plt.title("Communication Efficiency: Time per Billion Parameters")
    plt.ylabel("Communication Time per Billion Parameters (μs/B)")
    plt.xlabel("Model (Parameters)")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_dir / "comm_efficiency_by_model_size.png")
    plt.close()


def cross_model_allreduce_pct(df: pd.DataFrame, out_dir: Path) -> None:
    """All-reduce percentage of total communication by model size."""
    total_by_model = df.groupby("model")["total_us"].sum()
    allreduce_by_model = df[df["comm_type"] == "all-reduce"].groupby("model")["total_us"].sum()
    pct = (allreduce_by_model / total_by_model * 100).reset_index()
    pct.columns = ["model", "allreduce_pct"]
    pct["params_b"] = pct["model"].map(MODEL_SIZES)
    pct = pct.sort_values("params_b")
    pct["model_label"] = pct.apply(lambda r: f"{r['model'].split('_')[-1]} ({r['params_b']}B)", axis=1)

    plt.figure(figsize=(10, 5))
    sns.barplot(data=pct, x="model_label", y="allreduce_pct", hue="model_label", palette="coolwarm", legend=False)
    plt.title("All-Reduce Dominance by Model Size")
    plt.ylabel("All-Reduce Percentage of Total Communication (%)")
    plt.xlabel("Model (Parameters)")
    plt.ylim(80, 100)
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_dir / "allreduce_pct_by_model_size.png")
    plt.close()


def cross_model_tp_scaling(df: pd.DataFrame, out_dir: Path) -> None:
    """Line plot showing TP scaling across all models."""
    agg = df.groupby(["model", "tp"], as_index=False)["total_us"].sum()
    agg["params_b"] = agg["model"].map(MODEL_SIZES)
    agg["model_label"] = agg.apply(lambda r: f"{r['model'].split('_')[-1]} ({r['params_b']}B)", axis=1)
    agg["tp"] = pd.Categorical(agg["tp"], categories=ensure_sorted_numeric(agg["tp"]), ordered=True)

    plt.figure(figsize=(10, 6))
    sns.lineplot(data=agg, x="tp", y="total_us", hue="model_label", marker="o", linewidth=2)
    plt.title("Tensor Parallelism Scaling Comparison Across Models")
    plt.ylabel("Total Communication Time (μs)")
    plt.xlabel("Tensor Parallelism Degree (TP)")
    plt.legend(title="Model", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_dir / "tp_scaling_by_model.png", bbox_inches="tight")
    plt.close()


def cross_model_batch_scaling(df: pd.DataFrame, out_dir: Path) -> None:
    """Line plot showing batch scaling across all models."""
    agg = df.groupby(["model", "batch"], as_index=False)["total_us"].sum()
    agg["params_b"] = agg["model"].map(MODEL_SIZES)
    agg["model_label"] = agg.apply(lambda r: f"{r['model'].split('_')[-1]} ({r['params_b']}B)", axis=1)
    agg["batch"] = pd.Categorical(agg["batch"], categories=ensure_sorted_numeric(agg["batch"]), ordered=True)

    plt.figure(figsize=(10, 6))
    sns.lineplot(data=agg, x="batch", y="total_us", hue="model_label", marker="o", linewidth=2)
    plt.title("Batch Size Scaling Comparison Across Models")
    plt.ylabel("Total Communication Time (μs)")
    plt.xlabel("Batch Size")
    plt.legend(title="Model", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_dir / "batch_scaling_by_model.png", bbox_inches="tight")
    plt.close()


def cross_model_comm_breakdown(df: pd.DataFrame, out_dir: Path) -> None:
    """Stacked bar showing communication breakdown by model."""
    pivot = df.pivot_table(index="model", columns="comm_type", values="total_us", aggfunc="sum").fillna(0)
    pivot["params_b"] = pivot.index.map(MODEL_SIZES)
    pivot = pivot.sort_values("params_b")
    pivot = pivot.drop(columns=["params_b"])
    
    # Create model labels
    labels = [f"{m.split('_')[-1]} ({MODEL_SIZES[m]}B)" for m in pivot.index]

    pivot.index = labels
    pivot.plot(kind="bar", stacked=True, figsize=(12, 6), colormap="Set2")
    plt.title("Communication Breakdown by Model")
    plt.ylabel("Total Communication Time (μs)")
    plt.xlabel("Model (Parameters)")
    plt.xticks(rotation=30, ha="right")
    plt.legend(title="Communication Type", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_dir / "comm_breakdown_by_model.png", bbox_inches="tight")
    plt.close()


def generate_cross_model_plots(df: pd.DataFrame, out_dir: Path) -> None:
    """Generate all cross-model comparison plots."""
    cross_model_dir = out_dir / "cross_model"
    cross_model_total_time(df, cross_model_dir)
    cross_model_efficiency(df, cross_model_dir)
    cross_model_allreduce_pct(df, cross_model_dir)
    cross_model_tp_scaling(df, cross_model_dir)
    cross_model_batch_scaling(df, cross_model_dir)
    cross_model_comm_breakdown(df, cross_model_dir)
    print(f"Wrote cross-model comparison plots to {cross_model_dir}")


def main(argv: Iterable[str] | None = None) -> None:
    if not PARQUET_PATH.exists():
        raise SystemExit(f"Parquet not found at {PARQUET_PATH}")
    df = pd.read_parquet(PARQUET_PATH)
    if df.empty:
        raise SystemExit("No data found in parquet.")

    OUT_DIR_FULL.mkdir(parents=True, exist_ok=True)
    OUT_DIR_CURATED.mkdir(parents=True, exist_ok=True)
    for model in sorted(df["model"].unique()):
        generate_full_for_model(df, model, OUT_DIR_FULL)
        generate_curated_for_model(df, model, OUT_DIR_CURATED)
        print(f"Wrote plots for model: {model} (full + curated)")

    # Generate cross-model comparison plots
    generate_cross_model_plots(df, OUT_DIR_CURATED)


if __name__ == "__main__":
    main()

