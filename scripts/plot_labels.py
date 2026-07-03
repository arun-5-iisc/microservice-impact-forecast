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
    ax.set_title("Severity Class Distribution\n(130 experiments)", fontsize=13, fontweight="bold")
    ax.set_ylim(0, counts.max() * 1.25)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(axis="x", labelsize=13)

    # ── Pie chart ─────────────────────────────────────────────────────────────
    ax = axes[1]
    wedge_props = {"linewidth": 2, "edgecolor": "white"}
    wedges, texts, autotexts = ax.pie(
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
        "Target Label 1: Severity  ·  3 classes derived from Prometheus + Jaeger signals",
        fontsize=11, color="#444444", y=1.01
    )
    plt.tight_layout()
    _save(fig, out_dir, "labels_severity")


def plot_blast_radius(df_exp: pd.DataFrame, out_dir: Path) -> None:
    """Histogram + box-by-severity of blast_radius_final."""
    values = df_exp["blast_radius_final"].dropna().astype(int)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    # ── Histogram ─────────────────────────────────────────────────────────────
    ax = axes[0]
    unique_vals = sorted(values.unique())
    counts_by_val = values.value_counts().reindex(unique_vals, fill_value=0)
    ax.bar(unique_vals, counts_by_val.values,
           color="#4C72B0", edgecolor="white", linewidth=1.2, width=0.65)
    ax.set_xlabel("blast_radius_final  (number of services degraded)", fontsize=12)
    ax.set_ylabel("Number of Experiments", fontsize=12)
    ax.set_title("Blast Radius Distribution\n(130 experiments)", fontsize=13, fontweight="bold")
    ax.set_xticks(unique_vals)
    for x, cnt in zip(unique_vals, counts_by_val.values):
        if cnt > 0:
            ax.text(x, cnt + 0.3, str(cnt), ha="center", va="bottom", fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)

    # ── Box plot by severity ───────────────────────────────────────────────────
    ax = axes[1]
    data_by_sev = [
        df_exp.loc[df_exp["severity"] == s, "blast_radius_final"].dropna().astype(int).values
        for s in SEVERITY_ORDER
    ]
    bp = ax.boxplot(
        data_by_sev,
        patch_artist=True,
        medianprops={"color": "white", "linewidth": 2.5},
        whiskerprops={"linewidth": 1.5},
        capprops={"linewidth": 1.5},
        flierprops={"marker": "o", "markersize": 5, "alpha": 0.5},
    )
    for patch, s in zip(bp["boxes"], SEVERITY_ORDER):
        patch.set_facecolor(SEVERITY_COLORS[s])
        patch.set_alpha(0.85)
    ax.set_xticklabels(SEVERITY_ORDER, fontsize=13)
    ax.set_ylabel("blast_radius_final", fontsize=12)
    ax.set_title("Blast Radius by Severity Class", fontsize=13, fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)

    # annotate medians
    for i, (data, s) in enumerate(zip(data_by_sev, SEVERITY_ORDER), start=1):
        if len(data):
            med = np.median(data)
            ax.text(i, med + 0.15, f"med={med:.0f}", ha="center", va="bottom",
                    fontsize=10, fontweight="bold", color="white")

    fig.suptitle(
        "Target Label 2: blast_radius_final  ·  Count of services degraded in final 60 s",
        fontsize=11, color="#444444", y=1.01
    )
    plt.tight_layout()
    _save(fig, out_dir, "labels_blast_radius")


def main(data_path: Path, out_dir: Path) -> None:
    df = pd.read_csv(data_path)

    print(f"Dataset: {len(df)} rows  |  {df['experiment_id'].nunique()} experiments")

    # Verify that severity and blast_radius_final are truly constant per experiment
    # (they should be — both are derived from the final 60 s state, not the T-snapshot)
    inconsistent_sev = (
        df.groupby("experiment_id")["severity"]
        .nunique()
        .gt(1)
        .sum()
    )
    inconsistent_br = (
        df.groupby("experiment_id")["blast_radius_final"]
        .nunique()
        .gt(1)
        .sum()
    )
    if inconsistent_sev > 0 or inconsistent_br > 0:
        print(f"WARNING: {inconsistent_sev} experiments have >1 severity value across snapshots")
        print(f"WARNING: {inconsistent_br} experiments have >1 blast_radius_final across snapshots")
    else:
        print("Verified: severity and blast_radius_final are constant across all 4 snapshots per experiment")

    # Reduce to one row per experiment for plotting
    df_exp = df.drop_duplicates(subset="experiment_id").copy()
    print(f"Plotting with {len(df_exp)} experiments (one per experiment_id)")
    print(f"Severity counts:\n{df_exp['severity'].value_counts().reindex(SEVERITY_ORDER)}")
    print(f"Blast radius range: {df_exp['blast_radius_final'].min()} – {df_exp['blast_radius_final'].max()}")

    plot_severity(df_exp, out_dir)
    plot_blast_radius(df_exp, out_dir)
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
