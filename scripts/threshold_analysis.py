"""
Threshold analysis script — run on VM where processed dataset exists.

Analyses the distribution of latency delta and error rate across all experiments
to provide empirical evidence for the severity thresholds defined in preprocess.py:
  LATENCY_DELTA_HIGH   = 1.0  (>100% increase → High)
  LATENCY_DELTA_MEDIUM = 0.20 (>20%  increase → Medium)
  ERROR_RATE_HIGH      = 0.05 (>5%   error rate → High)
  ERROR_RATE_MEDIUM    = 0.01 (>1%   error rate → Medium)

Outputs (saved to results/threshold_analysis/ and uploaded to GCS):
  threshold_summary.csv       — percentile statistics per severity class
  latency_delta_dist.png      — latency delta distribution by severity class
  error_rate_dist.png         — error rate distribution by severity class
  latency_delta_kde.png       — KDE plot showing threshold positions
  feature_boxplots.png        — box plots of key features by severity class
  class_distribution.png      — severity class counts and proportions
  correlation_heatmap.png     — feature correlation heatmap

Usage (on VM):
  python scripts/threshold_analysis.py
  python scripts/threshold_analysis.py --data dataset/processed/training_dataset_fe.csv
  python scripts/threshold_analysis.py --upload gs://ff-dataset-bucket/processed/
"""
import argparse
import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import seaborn as sns

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger(__name__)

REPO_ROOT   = Path(__file__).parent.parent
DEFAULT_DATA = REPO_ROOT / "dataset" / "processed" / "training_dataset.csv"
OUT_DIR      = REPO_ROOT / "results" / "threshold_analysis"

SEVERITY_ORDER  = ["Low", "Medium", "High"]
SEVERITY_COLORS = {"Low": "#55A868", "Medium": "#DD8452", "High": "#C43235"}

# Thresholds defined in preprocess.py
LATENCY_DELTA_HIGH   = 1.0
LATENCY_DELTA_MEDIUM = 0.20
ERROR_RATE_HIGH      = 0.05
ERROR_RATE_MEDIUM    = 0.01

plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 300,
    "font.size": 11, "axes.titlesize": 13,
    "axes.labelsize": 12, "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True,
    "grid.alpha": 0.3, "grid.linestyle": "--",
})


# ── 1. Threshold summary table ────────────────────────────────────────────────

def threshold_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Percentile statistics per severity class for:
      - final_latency_delta : value used for labeling (last 60s of fault phase)
      - target_latency_delta: early-window feature (T=15-60s into fault)
      - target_error_rate   : early-window error rate feature

    final_latency_delta is the primary validation column — thresholds were
    applied to this value, so class separation here is the empirical justification.
    target_latency_delta shows how early the signal appears (proactive value).
    """
    rows = []
    for sev in SEVERITY_ORDER:
        sdf = df[df["severity"] == sev]
        for col, thresholds in [
            ("final_latency_delta",  [LATENCY_DELTA_MEDIUM, LATENCY_DELTA_HIGH]),
            ("target_latency_delta", [LATENCY_DELTA_MEDIUM, LATENCY_DELTA_HIGH]),
            ("target_error_rate",    [ERROR_RATE_MEDIUM,    ERROR_RATE_HIGH]),
        ]:
            if col not in df.columns:
                continue
            vals = sdf[col].dropna()
            if vals.empty:
                continue
            rows.append({
                "severity":  sev,
                "feature":   col,
                "count":     len(vals),
                "mean":      round(vals.mean(), 4),
                "median":    round(vals.median(), 4),
                "p25":       round(vals.quantile(0.25), 4),
                "p75":       round(vals.quantile(0.75), 4),
                "p90":       round(vals.quantile(0.90), 4),
                "p95":       round(vals.quantile(0.95), 4),
                "max":       round(vals.max(), 4),
                "pct_above_low_threshold":  round(100 * (vals > thresholds[0]).mean(), 1),
                "pct_above_high_threshold": round(100 * (vals > thresholds[1]).mean(), 1),
            })
    return pd.DataFrame(rows)


# ── 2. Latency delta distribution ─────────────────────────────────────────────

def plot_latency_delta_dist(df: pd.DataFrame, out_dir: Path):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), sharey=False)

    for ax, sev in zip(axes, SEVERITY_ORDER):
        vals = df[df["severity"] == sev]["target_latency_delta"].dropna()
        color = SEVERITY_COLORS[sev]
        ax.hist(vals, bins=30, color=color, alpha=0.8, edgecolor="white")
        ax.axvline(LATENCY_DELTA_MEDIUM, color="orange", linewidth=1.8,
                   linestyle="--", label=f"Medium threshold ({LATENCY_DELTA_MEDIUM*100:.0f}%)")
        ax.axvline(LATENCY_DELTA_HIGH, color="red", linewidth=1.8,
                   linestyle="--", label=f"High threshold ({LATENCY_DELTA_HIGH*100:.0f}%)")
        ax.set_title(f"{sev} Severity (n={len(vals)})")
        ax.set_xlabel("Latency Delta (fraction of baseline)")
        ax.set_ylabel("Count")
        ax.legend(fontsize=9)

        # Annotate % above each threshold
        pct_med  = 100 * (vals > LATENCY_DELTA_MEDIUM).mean()
        pct_high = 100 * (vals > LATENCY_DELTA_HIGH).mean()
        ax.text(0.97, 0.95, f">{LATENCY_DELTA_MEDIUM*100:.0f}%: {pct_med:.1f}% of rows\n"
                             f">{LATENCY_DELTA_HIGH*100:.0f}%: {pct_high:.1f}% of rows",
                transform=ax.transAxes, ha="right", va="top",
                fontsize=9, color="black",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7))

    fig.suptitle("Latency Delta Distribution by Severity Class\n"
                 "(dashed lines = severity thresholds defined in preprocess.py)",
                 fontsize=13, y=1.02)
    plt.tight_layout()
    path = out_dir / "latency_delta_dist.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved → %s", path)


# ── 3. Error rate distribution ────────────────────────────────────────────────

def plot_error_rate_dist(df: pd.DataFrame, out_dir: Path):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), sharey=False)

    for ax, sev in zip(axes, SEVERITY_ORDER):
        vals = df[df["severity"] == sev]["target_error_rate"].dropna()
        color = SEVERITY_COLORS[sev]
        ax.hist(vals, bins=30, color=color, alpha=0.8, edgecolor="white")
        ax.axvline(ERROR_RATE_MEDIUM, color="orange", linewidth=1.8,
                   linestyle="--", label=f"Medium threshold ({ERROR_RATE_MEDIUM*100:.0f}%)")
        ax.axvline(ERROR_RATE_HIGH, color="red", linewidth=1.8,
                   linestyle="--", label=f"High threshold ({ERROR_RATE_HIGH*100:.0f}%)")
        ax.set_title(f"{sev} Severity (n={len(vals)})")
        ax.set_xlabel("Error Rate (fraction of requests)")
        ax.set_ylabel("Count")
        ax.legend(fontsize=9)

        pct_med  = 100 * (vals > ERROR_RATE_MEDIUM).mean()
        pct_high = 100 * (vals > ERROR_RATE_HIGH).mean()
        ax.text(0.97, 0.95, f">{ERROR_RATE_MEDIUM*100:.0f}%: {pct_med:.1f}% of rows\n"
                             f">{ERROR_RATE_HIGH*100:.0f}%: {pct_high:.1f}% of rows",
                transform=ax.transAxes, ha="right", va="top",
                fontsize=9, color="black",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7))

    fig.suptitle("Error Rate Distribution by Severity Class\n"
                 "(dashed lines = severity thresholds defined in preprocess.py)",
                 fontsize=13, y=1.02)
    plt.tight_layout()
    path = out_dir / "error_rate_dist.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved → %s", path)


# ── 4. KDE overlay — all three classes on one plot ────────────────────────────

def plot_latency_kde(df: pd.DataFrame, out_dir: Path):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

    for ax, col, med_t, high_t, xlabel in [
        (axes[0], "target_latency_delta", LATENCY_DELTA_MEDIUM, LATENCY_DELTA_HIGH,
         "Latency Delta (fraction of baseline)"),
        (axes[1], "target_error_rate", ERROR_RATE_MEDIUM, ERROR_RATE_HIGH,
         "Error Rate"),
    ]:
        if col not in df.columns:
            continue
        for sev in SEVERITY_ORDER:
            vals = df[df["severity"] == sev][col].dropna()
            vals_clipped = vals.clip(upper=vals.quantile(0.99))
            vals_clipped.plot.kde(ax=ax, label=sev, color=SEVERITY_COLORS[sev],
                                  linewidth=2)

        ax.axvline(med_t,  color="orange", linestyle="--", linewidth=1.5,
                   label=f"Medium threshold ({med_t})")
        ax.axvline(high_t, color="red",    linestyle="--", linewidth=1.5,
                   label=f"High threshold ({high_t})")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Density")
        ax.legend()
        ax.set_xlim(left=0)

    axes[0].set_title("Latency Delta — KDE by Severity Class")
    axes[1].set_title("Error Rate — KDE by Severity Class")
    fig.suptitle("Kernel Density — Threshold Positions in Data Distribution",
                 fontsize=13, y=1.02)
    plt.tight_layout()
    path = out_dir / "latency_delta_kde.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved → %s", path)


# ── 5. Box plots — key features by severity ───────────────────────────────────

def plot_feature_boxplots(df: pd.DataFrame, out_dir: Path):
    features = [
        ("target_latency_delta",   "Latency Delta"),
        ("target_error_rate",      "Error Rate"),
        ("target_cpu",             "CPU Usage"),
        ("target_memory_bytes",    "Memory Bytes"),
        ("blast_radius_early",     "Blast Radius Early"),
        ("target_request_rate",    "Request Rate"),
    ]
    features = [(f, l) for f, l in features if f in df.columns]

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    axes = axes.flatten()

    for ax, (feat, label) in zip(axes, features):
        data = [df[df["severity"] == sev][feat].dropna().values
                for sev in SEVERITY_ORDER]
        bp = ax.boxplot(data, patch_artist=True, tick_labels=SEVERITY_ORDER)
        for patch, sev in zip(bp["boxes"], SEVERITY_ORDER):
            patch.set_facecolor(SEVERITY_COLORS[sev])
            patch.set_alpha(0.7)
        ax.set_title(label)
        ax.set_xlabel("Severity Class")
        ax.set_ylabel(label)

    for ax in axes[len(features):]:
        ax.set_visible(False)

    fig.suptitle("Key Feature Distributions by Severity Class\n"
                 "(shows how well each feature separates classes)",
                 fontsize=13, y=1.01)
    plt.tight_layout()
    path = out_dir / "feature_boxplots.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved → %s", path)


# ── 6. Class distribution ─────────────────────────────────────────────────────

def plot_class_distribution(df: pd.DataFrame, out_dir: Path):
    counts = df["severity"].value_counts().reindex(SEVERITY_ORDER)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    colors = [SEVERITY_COLORS[s] for s in SEVERITY_ORDER]
    axes[0].bar(SEVERITY_ORDER, counts.values, color=colors, edgecolor="white")
    for i, (sev, cnt) in enumerate(zip(SEVERITY_ORDER, counts.values)):
        axes[0].text(i, cnt + 2, str(cnt), ha="center", fontsize=12, fontweight="bold")
    axes[0].set_ylabel("Count")
    axes[0].set_title("Severity Class Distribution — Counts")

    axes[1].pie(counts.values, labels=SEVERITY_ORDER, colors=colors,
                autopct="%1.1f%%", startangle=90,
                wedgeprops=dict(edgecolor="white", linewidth=1.5))
    axes[1].set_title("Severity Class Distribution — Proportions")

    fig.suptitle(f"Label Distribution (total rows = {len(df)})", fontsize=13)
    plt.tight_layout()
    path = out_dir / "class_distribution.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved → %s", path)


# ── 7. Correlation heatmap ────────────────────────────────────────────────────

def plot_correlation_heatmap(df: pd.DataFrame, out_dir: Path):
    severity_order = {"Low": 0, "Medium": 1, "High": 2}
    plot_df = df.copy()
    plot_df["severity_encoded"] = plot_df["severity"].map(severity_order)

    numeric_cols = plot_df.select_dtypes(include=[np.number]).columns.tolist()
    drop_cols    = ["final_latency_delta"]   # label-basis column, not a feature
    numeric_cols = [c for c in numeric_cols if c not in drop_cols]

    # Put target labels last so they appear on the right/bottom of the heatmap
    label_cols   = [c for c in ["severity_encoded", "blast_radius_final"] if c in numeric_cols]
    feature_cols = [c for c in numeric_cols if c not in label_cols]
    numeric_cols = feature_cols + label_cols

    corr = plot_df[numeric_cols].corr(method="spearman")

    fig, ax = plt.subplots(figsize=(max(10, len(numeric_cols) * 0.7),
                                    max(8,  len(numeric_cols) * 0.7)))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="RdBu_r",
                vmin=-1, vmax=1, center=0, linewidths=0.3,
                ax=ax, cbar_kws={"shrink": 0.7, "label": "Pearson r"},
                annot_kws={"size": 7})
    ax.set_title("Feature Correlation Matrix (Spearman)\n"
                 "(lower triangle; severity_encoded: Low=0 Medium=1 High=2; "
                 "dark red = strong positive, dark blue = strong negative)",
                 fontsize=11)
    plt.xticks(rotation=45, ha="right", fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()
    path = out_dir / "correlation_heatmap.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved → %s", path)


# ── 8. Coverage table ─────────────────────────────────────────────────────────

def coverage_table(df: pd.DataFrame, out_dir: Path):
    """
    Per-severity count of fault types, services, and unique fault×service
    combinations. Demonstrates all three classes are operationally diverse
    and not artifacts of a single fault type or service.
    """
    rows = []
    for sev in SEVERITY_ORDER:
        sdf = df[df["severity"] == sev].drop_duplicates(subset="experiment_id")
        rows.append({
            "severity":               sev,
            "experiments":            len(sdf),
            "fault_types":            sdf["fault_type"].nunique(),
            "services":               sdf["service"].nunique(),
            "unique_fault_x_service": sdf.groupby(["fault_type", "service"]).ngroups,
        })
    result = pd.DataFrame(rows)
    csv_path = out_dir / "coverage_table.csv"
    result.to_csv(csv_path, index=False)
    logger.info("Coverage table:\n%s", result.to_string(index=False))
    return result


# ── 9. Sensitivity analysis ───────────────────────────────────────────────────

def sensitivity_analysis(df: pd.DataFrame, out_dir: Path):
    """
    Varies LATENCY_DELTA_HIGH and LATENCY_DELTA_MEDIUM independently and shows
    how experiment-level class counts change. Flat lines = threshold choice is
    stable; steep lines = boundary is sensitive and warrants justification.

    Uses final_latency_delta (the value used for labeling, not the early feature).
    Operates at experiment level (one row per experiment) to avoid 4× inflation.
    """
    col = "final_latency_delta"
    if col not in df.columns:
        logger.warning("sensitivity_analysis: '%s' not in dataset — re-run preprocess.py first", col)
        return

    exp_df = df.drop_duplicates(subset="experiment_id")[["experiment_id", col]].copy()
    vals = exp_df[col].dropna()
    logger.info("Sensitivity analysis: %d experiments with %s values", len(vals), col)

    high_variants = [0.60, 0.70, 0.80, 0.90, 1.00, 1.10, 1.20, 1.30, 1.40]
    med_variants  = [0.10, 0.12, 0.15, 0.18, 0.20, 0.22, 0.25, 0.28, 0.30]

    rows = []
    for high_t in high_variants:
        rows.append({
            "vary":     "high_threshold",
            "medium_t": LATENCY_DELTA_MEDIUM,
            "high_t":   high_t,
            "Low":      int((vals <= LATENCY_DELTA_MEDIUM).sum()),
            "Medium":   int(((vals > LATENCY_DELTA_MEDIUM) & (vals <= high_t)).sum()),
            "High":     int((vals > high_t).sum()),
            "current":  high_t == LATENCY_DELTA_HIGH,
        })
    for med_t in med_variants:
        rows.append({
            "vary":     "medium_threshold",
            "medium_t": med_t,
            "high_t":   LATENCY_DELTA_HIGH,
            "Low":      int((vals <= med_t).sum()),
            "Medium":   int(((vals > med_t) & (vals <= LATENCY_DELTA_HIGH)).sum()),
            "High":     int((vals > LATENCY_DELTA_HIGH).sum()),
            "current":  med_t == LATENCY_DELTA_MEDIUM,
        })

    result = pd.DataFrame(rows)
    csv_path = out_dir / "sensitivity_analysis.csv"
    result.to_csv(csv_path, index=False)
    logger.info("Saved → %s", csv_path)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, vary, x_col, current_val, title in [
        (axes[0], "high_threshold",   "high_t",   LATENCY_DELTA_HIGH,
         f"Varying High threshold\n(Medium fixed at {LATENCY_DELTA_MEDIUM})"),
        (axes[1], "medium_threshold", "medium_t", LATENCY_DELTA_MEDIUM,
         f"Varying Medium threshold\n(High fixed at {LATENCY_DELTA_HIGH})"),
    ]:
        sub = result[result["vary"] == vary]
        ax.plot(sub[x_col], sub["High"],   "o-", color=SEVERITY_COLORS["High"],   label="High",   linewidth=2)
        ax.plot(sub[x_col], sub["Medium"], "s-", color=SEVERITY_COLORS["Medium"], label="Medium", linewidth=2)
        ax.plot(sub[x_col], sub["Low"],    "^-", color=SEVERITY_COLORS["Low"],    label="Low",    linewidth=2)
        ax.axvline(current_val, color="black", linestyle=":", linewidth=1.5, label="Current")
        ax.set_xlabel("Threshold value")
        ax.set_ylabel("Experiment count")
        ax.set_title(title)
        ax.legend(fontsize=9)

    fig.suptitle(
        "Sensitivity Analysis — Latency Delta Threshold Variation\n"
        "(flat lines = stable threshold; steep = boundary-sensitive)",
        fontsize=13, y=1.02,
    )
    plt.tight_layout()
    png_path = out_dir / "sensitivity_analysis.png"
    fig.savefig(png_path, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved → %s", png_path)

    for vary, x_col, current_val in [
        ("high_threshold",   "high_t",   LATENCY_DELTA_HIGH),
        ("medium_threshold", "medium_t", LATENCY_DELTA_MEDIUM),
    ]:
        sub = result[(result["vary"] == vary) & result["current"]]
        logger.info(
            "Current %s=%.2f → Low=%d  Medium=%d  High=%d",
            x_col, current_val,
            sub["Low"].values[0], sub["Medium"].values[0], sub["High"].values[0],
        )


# ── Main ───────────────────────────────────────────────────────────────────────

def main(data_path: Path, upload_prefix: str | None):
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("Loading dataset from %s", data_path)
    df = pd.read_csv(data_path)
    logger.info("Dataset: %d rows, %d columns, %d experiments",
                len(df), len(df.columns), df["experiment_id"].nunique())

    logger.info("\nSeverity distribution:\n%s",
                df["severity"].value_counts().reindex(SEVERITY_ORDER).to_string())

    # ── Threshold summary ─────────────────────────────────────────────────────
    logger.info("Computing threshold summary…")
    summary = threshold_summary(df)
    summary.to_csv(OUT_DIR / "threshold_summary.csv", index=False)
    logger.info("\nThreshold summary:\n%s", summary.to_string(index=False))

    # ── Plots ─────────────────────────────────────────────────────────────────
    logger.info("Generating plots…")
    plot_class_distribution(df, OUT_DIR)
    plot_latency_delta_dist(df, OUT_DIR)
    plot_error_rate_dist(df, OUT_DIR)
    plot_latency_kde(df, OUT_DIR)
    plot_feature_boxplots(df, OUT_DIR)
    plot_correlation_heatmap(df, OUT_DIR)
    coverage_table(df, OUT_DIR)
    sensitivity_analysis(df, OUT_DIR)

    # ── Print key findings ────────────────────────────────────────────────────
    logger.info("\n%s\nKEY THRESHOLD FINDINGS\n%s", "="*55, "="*55)
    for col, med_t, high_t in [
        ("target_latency_delta", LATENCY_DELTA_MEDIUM, LATENCY_DELTA_HIGH),
        ("target_error_rate",    ERROR_RATE_MEDIUM,    ERROR_RATE_HIGH),
    ]:
        if col not in df.columns:
            continue
        logger.info("\n%s", col)
        for sev in SEVERITY_ORDER:
            vals = df[df["severity"] == sev][col].dropna()
            logger.info(
                "  %s: mean=%.3f  median=%.3f  p90=%.3f  "
                "pct_above_medium=%.1f%%  pct_above_high=%.1f%%",
                sev, vals.mean(), vals.median(), vals.quantile(0.90),
                100*(vals > med_t).mean(), 100*(vals > high_t).mean()
            )

    # ── Upload to GCS ─────────────────────────────────────────────────────────
    if upload_prefix:
        import subprocess
        logger.info("Uploading results to %s", upload_prefix)
        subprocess.run([
            "gcloud", "storage", "cp", "-r",
            str(OUT_DIR) + "/",
            upload_prefix.rstrip("/") + "/threshold_analysis/",
        ], check=True)
        # Also upload processed CSVs
        for csv_name in ["training_dataset.csv", "training_dataset_fe.csv"]:
            csv_path = data_path.parent / csv_name
            if csv_path.exists():
                subprocess.run([
                    "gcloud", "storage", "cp",
                    str(csv_path),
                    upload_prefix.rstrip("/") + "/" + csv_name,
                ], check=True)
                logger.info("Uploaded %s → %s", csv_name, upload_prefix)

    logger.info("\nAll outputs saved to %s/", OUT_DIR)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Threshold analysis and EDA")
    parser.add_argument("--data",   type=Path, default=DEFAULT_DATA)
    parser.add_argument("--upload", type=str, default=None,
                        help="GCS prefix to upload results and processed CSVs "
                             "(e.g. gs://ff-dataset-bucket/processed)")
    args = parser.parse_args()
    main(args.data, args.upload)
