"""
Plot target label distributions for dissertation slides.

Generates two figures:
  results/plots/labels_severity.png      — severity class distribution (bar + pie)
  results/plots/labels_blast_radius.png  — blast_radius_final distribution (histogram + box)

Usage:
  python scripts/plot_labels.py
  python scripts/plot_labels.py --data dataset/processed/training_dataset.csv
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).parent.parent

SEVERITY_ORDER  = ["Low", "Medium", "High"]
SEVERITY_COLORS = {
    "Low":    "#55A868",   # green
    "Medium": "#DD8452",   # orange
    "High":   "#C00000",   # red
}


def _save(fig: plt.Figure, out_dir: Path, name: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {path}")


def plot_severity(df_exp: pd.DataFrame, out_dir: Path) -> None:
    """Bar chart + pie chart of severity class distribution."""
    counts = (
        df_exp["severity"]
        .value_counts()
        .reindex(SEVERITY_ORDER)
        .fillna(0)
        .astype(int)
    )
    total = counts.sum()
    colors = [SEVERITY_COLORS[s] for s in SEVERITY_ORDER]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    # ── Bar chart ─────────────────────────────────────────────────────────────
    ax = axes[0]
    bars = ax.bar(SEVERITY_ORDER, counts.values, color=colors, edgecolor="white",
                  linewidth=1.5, width=0.55)
    for bar, count in zip(bars, counts.values):
        pct = count / total * 100
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.8,
                f"{count}\n({pct:.1f}%)",
                ha="center", va="bottom", fontsize=12, fontweight="bold")
    ax.set_ylabel("Number of Experiments", fontsize=12)
    ax.set_title("Severity Class Distribution\n(520 rows — 130 experiments × 4 snapshots)", fontsize=13, fontweight="bold")
    ax.set_ylim(0, counts.max() * 1.25)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(axis="x", labelsize=13)

    # ── Pie chart ─────────────────────────────────────────────────────────────
    ax = axes[1]
    wedge_props = {"linewidth": 2, "edgecolor": "white"}
    _, _, autotexts = ax.pie(
        counts.values,
        labels=None,
        colors=colors,
        autopct="%1.1f%%",
        startangle=90,
        wedgeprops=wedge_props,
        pctdistance=0.72,
    )
    for at in autotexts:
        at.set_fontsize(12)
        at.set_fontweight("bold")
        at.set_color("white")

    legend_patches = [
        mpatches.Patch(color=SEVERITY_COLORS[s], label=f"{s}  (n={counts[s]})")
        for s in SEVERITY_ORDER
    ]
    ax.legend(handles=legend_patches, loc="lower center",
              bbox_to_anchor=(0.5, -0.12), ncol=3, fontsize=11, frameon=False)
    ax.set_title("Severity Distribution — Proportion", fontsize=13, fontweight="bold")

    fig.suptitle(
        "Target Label 1: Severity  ·  3 classes derived from Prometheus + Jaeger signals"
        "  ·  proportions identical at experiment and row level",
        fontsize=11, color="#444444", y=1.01
    )
    plt.tight_layout()
    _save(fig, out_dir, "labels_severity")


def plot_severity_calculation(df_exp60: pd.DataFrame, out_dir: Path) -> None:
    """
    Show how severity labels are derived from target_latency_delta.
    Left:  scatter of latency delta per experiment at T=60s, coloured by severity,
           with threshold lines at 0.2 (Medium) and 1.0 (High).
    Right: stacked bar showing how many experiments have latency data vs no latency data
           per severity class (no-data experiments are labeled by replica/restart rule).
    """
    rng = np.random.default_rng(42)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # ── Left: scatter strip with threshold lines ───────────────────────────────
    ax = axes[0]
    x_pos = {"Low": 0, "Medium": 1, "High": 2}
    for s in SEVERITY_ORDER:
        sub = df_exp60.loc[df_exp60["severity"] == s, "target_latency_delta"].dropna()
        if len(sub) == 0:
            continue
        jitter = rng.uniform(-0.18, 0.18, size=len(sub))
        ax.scatter(x_pos[s] + jitter, sub.values,
                   color=SEVERITY_COLORS[s], alpha=0.65, s=45, zorder=3,
                   edgecolors="white", linewidths=0.4)

    ax.axhline(0.20, color="#DD8452", linestyle="--", linewidth=1.8,
               label="Medium threshold  (0.20)", zorder=4)
    ax.axhline(1.00, color="#C00000", linestyle="--", linewidth=1.8,
               label="High threshold  (1.00)", zorder=4)

    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(SEVERITY_ORDER, fontsize=13)
    ax.set_ylabel("target_latency_delta at T=60 s", fontsize=12)
    ax.set_title("Latency Delta by Severity Class\n(T=60 s, symlog scale)", fontsize=13, fontweight="bold")
    ax.set_yscale("symlog", linthresh=0.05)
    ax.legend(fontsize=10, loc="upper left")
    ax.spines[["top", "right"]].set_visible(False)
    ax.text(0.98, 0.97,
            "Experiments with no latency\ndata are excluded here →",
            transform=ax.transAxes, ha="right", va="top",
            fontsize=9, color="#888888", style="italic")

    # ── Right: data availability per severity class ────────────────────────────
    ax = axes[1]
    has_data, no_data = [], []
    for s in SEVERITY_ORDER:
        sub = df_exp60[df_exp60["severity"] == s]["target_latency_delta"]
        has_data.append(sub.notna().sum())
        no_data.append(sub.isna().sum())

    x = np.arange(len(SEVERITY_ORDER))
    ax.bar(x, has_data,
                      color=[SEVERITY_COLORS[s] for s in SEVERITY_ORDER],
                      edgecolor="white", linewidth=1.2,
                      label="Has latency data\n(labeled by threshold rules)")
    ax.bar(x, no_data, bottom=has_data,
           color="#AAAAAA", edgecolor="white", linewidth=1.2,
           label="No latency data\n(labeled by replica / restart rule)")

    # Annotate counts
    for i, (h, n) in enumerate(zip(has_data, no_data)):
        if h > 0:
            ax.text(i, h / 2, str(h), ha="center", va="center",
                    fontsize=11, fontweight="bold", color="white")
        if n > 0:
            ax.text(i, h + n / 2, str(n), ha="center", va="center",
                    fontsize=11, fontweight="bold", color="white")

    ax.set_xticks(x)
    ax.set_xticklabels(SEVERITY_ORDER, fontsize=13)
    ax.set_ylabel("Number of Experiments", fontsize=12)
    ax.set_title("Labeling Rule Used per Severity Class\n(130 experiments)", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10, loc="upper right")
    ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle(
        "Severity Calculation  ·  threshold rules on target_latency_delta  +  replica/restart override",
        fontsize=11, color="#444444", y=1.01
    )
    plt.tight_layout()
    _save(fig, out_dir, "labels_severity_calculation")


def plot_blast_radius(df: pd.DataFrame, out_dir: Path) -> None:
    """Histogram of blast_radius_final."""
    values = df["blast_radius_final"].dropna().astype(int)
    unique_vals = sorted(values.unique())
    counts_by_val = values.value_counts().reindex(unique_vals, fill_value=0)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(unique_vals, counts_by_val.values,
           color="#4C72B0", edgecolor="white", linewidth=1.2, width=0.65)
    ax.set_xlabel("blast_radius_final  (number of services degraded)", fontsize=12)
    ax.set_ylabel("Number of Rows", fontsize=12)
    ax.set_title(
        "Blast Radius Distribution\n(520 rows — 130 experiments × 4 snapshots)",
        fontsize=13, fontweight="bold"
    )
    ax.set_xticks(unique_vals)
    for x, cnt in zip(unique_vals, counts_by_val.values):
        if cnt > 0:
            ax.text(x, cnt + 0.3, str(cnt), ha="center", va="bottom", fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle(
        "Target Label 2: blast_radius_final  ·  Count of services degraded in final 60 s",
        fontsize=11, color="#444444", y=1.02
    )
    plt.tight_layout()
    _save(fig, out_dir, "labels_blast_radius")


def main(data_path: Path, out_dir: Path) -> None:
    df = pd.read_csv(data_path)

    n_exp = df["experiment_id"].nunique()
    print(f"Dataset: {len(df)} rows  |  {n_exp} experiments  ({n_exp} × 4 snapshots)")

    # Verify that labels are constant per experiment (they should be — both are derived
    # from the final 60 s state of the experiment, independent of the T-snapshot)
    inconsistent_sev = df.groupby("experiment_id")["severity"].nunique().gt(1).sum()
    inconsistent_br  = df.groupby("experiment_id")["blast_radius_final"].nunique().gt(1).sum()
    if inconsistent_sev > 0 or inconsistent_br > 0:
        print(f"WARNING: {inconsistent_sev} experiments have >1 severity value across snapshots")
        print(f"WARNING: {inconsistent_br} experiments have >1 blast_radius_final across snapshots")
    else:
        print("Verified: severity and blast_radius_final are constant across all 4 snapshots per experiment")

    # Plot at row level — the model trains and infers on rows (one per T-snapshot),
    # not on experiments. 520 rows is the actual ML dataset size.
    print(f"Plotting with {len(df)} rows (130 experiments × 4 time snapshots)")
    print(f"Severity counts (rows):\n{df['severity'].value_counts().reindex(SEVERITY_ORDER)}")
    print(f"Blast radius range: {df['blast_radius_final'].min()} – {df['blast_radius_final'].max()}")

    df_exp60 = df[df["time_into_fault_s"] == 60].drop_duplicates("experiment_id")
    plot_severity(df, out_dir)
    plot_severity_calculation(df_exp60, out_dir)
    plot_blast_radius(df, out_dir)
    print("\nDone.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot target label distributions")
    parser.add_argument(
        "--data", type=Path,
        default=REPO_ROOT / "dataset" / "processed" / "training_dataset.csv",
        help="Path to processed dataset CSV (default: training_dataset.csv)",
    )
    parser.add_argument(
        "--out", type=Path,
        default=REPO_ROOT / "results" / "plots",
        help="Output directory for plots (default: results/plots/)",
    )
    args = parser.parse_args()
    main(args.data, args.out)
