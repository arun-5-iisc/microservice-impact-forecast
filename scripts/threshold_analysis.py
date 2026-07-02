"""
Threshold analysis and feature EDA — two distinct modes, run in order:

  --mode threshold   (BEFORE holdout split, full dataset)
      Validates the severity label design:
        class_distribution.png, blast_radius_dist.png, threshold_summary.csv,
        latency_delta_dist.png, error_rate_dist.png, latency_delta_kde.png,
        threshold_confusion_table.csv, sensitivity_analysis.csv/.png,
        coverage_table.csv, dataset_overview.csv, missing_values.csv/.png
      Output dir: results/threshold_analysis/threshold/

  --mode features    (AFTER holdout split + feature_engineer.py, training set only)
      Justifies which features to include for classification and regression:
        correlation_heatmap.png, feature_target_correlations.png (clf vs reg
        ranked bar charts), feature_boxplots.png, feature_justification.csv,
        time_evolution.png/.csv, missing_values.csv/.png
      Run twice — once on base features, once on engineered features:
        Output dir: results/threshold_analysis/features_base/   (training_dataset.csv)
        Output dir: results/threshold_analysis/features_fe/     (training_dataset_fe.csv)

Pipeline order:
  preprocess.py
  threshold_analysis.py --mode threshold                         (full dataset)
  create_holdout.py
  threshold_analysis.py --mode features --data training_dataset.csv
  feature_engineer.py
  threshold_analysis.py --mode features --data training_dataset_fe.csv
  train_model.py → tune_model.py → visualize.py

Thresholds defined in preprocess.py:
  LATENCY_DELTA_HIGH   = 1.0   (>100% increase → High)
  LATENCY_DELTA_MEDIUM = 0.20  (>20%  increase → Medium)
  ERROR_RATE_HIGH      = 0.05  (>5%   error rate → High)
  ERROR_RATE_MEDIUM    = 0.01  (>1%   error rate → Medium)
"""
import argparse
import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger(__name__)

REPO_ROOT    = Path(__file__).parent.parent
DEFAULT_DATA = REPO_ROOT / "dataset" / "processed" / "training_dataset.csv"
DEFAULT_OUT  = REPO_ROOT / "results" / "threshold_analysis"

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
    for i, (_, cnt) in enumerate(zip(SEVERITY_ORDER, counts.values)):
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


# ── 8. Threshold confusion table ──────────────────────────────────────────────

def threshold_confusion_table(df: pd.DataFrame, out_dir: Path):
    """
    Cross-tabulates actual severity label against the latency band that
    final_latency_delta falls into. Shows how well latency thresholds alone
    explain the assigned labels, and explicitly accounts for experiments where
    High was triggered by non-latency signals (replicas=0, restarts, error_rate).

    Operates at experiment level to avoid 4x row inflation.
    Only includes experiments where final_latency_delta is measurable.
    """
    col = "final_latency_delta"
    if col not in df.columns:
        logger.warning("threshold_confusion_table: '%s' not in dataset", col)
        return

    exp_df = df.drop_duplicates(subset="experiment_id").copy()

    def latency_band(v):
        if pd.isna(v):
            return "no_data"
        if v > LATENCY_DELTA_HIGH:
            return f"> {LATENCY_DELTA_HIGH} (High band)"
        if v > LATENCY_DELTA_MEDIUM:
            return f"{LATENCY_DELTA_MEDIUM}–{LATENCY_DELTA_HIGH} (Medium band)"
        return f"< {LATENCY_DELTA_MEDIUM} (Low band)"

    exp_df["latency_band"] = exp_df[col].apply(latency_band)

    band_order = [
        f"< {LATENCY_DELTA_MEDIUM} (Low band)",
        f"{LATENCY_DELTA_MEDIUM}–{LATENCY_DELTA_HIGH} (Medium band)",
        f"> {LATENCY_DELTA_HIGH} (High band)",
        "no_data",
    ]

    table = pd.crosstab(
        exp_df["severity"],
        exp_df["latency_band"],
    ).reindex(index=SEVERITY_ORDER, columns=band_order, fill_value=0)

    csv_path = out_dir / "threshold_confusion_table.csv"
    table.to_csv(csv_path)
    logger.info("Threshold confusion table (experiments):\n%s", table.to_string())
    return table


# ── 9. Coverage table ─────────────────────────────────────────────────────────

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


# ── A. Dataset overview ───────────────────────────────────────────────────────

def dataset_overview(df: pd.DataFrame, out_dir: Path) -> dict:
    """EDA §1: experiment and feature inventory."""
    exp_df = df.drop_duplicates(subset="experiment_id")
    label_cols = {"severity", "blast_radius_final", "final_latency_delta",
                  "experiment_id", "latency_observable"}
    summary = {
        "total_rows":        len(df),
        "total_experiments": exp_df["experiment_id"].nunique(),
        "total_features":    len([c for c in df.columns if c not in label_cols]),
        "services":          exp_df["service"].nunique(),
        "fault_types":       exp_df["fault_type"].nunique(),
        "time_snapshots":    df["time_into_fault_s"].nunique(),
    }
    for sev in SEVERITY_ORDER:
        sev_exp = exp_df[exp_df["severity"] == sev]
        summary[f"experiments_{sev}"] = len(sev_exp)
        summary[f"rows_{sev}"]        = len(df[df["severity"] == sev])

    pd.DataFrame([summary]).to_csv(out_dir / "dataset_overview.csv", index=False)
    logger.info("Dataset overview:\n%s",
                "\n".join(f"  {k}: {v}" for k, v in summary.items()))
    return summary


# ── B. Missing value analysis ─────────────────────────────────────────────────

def missing_value_analysis(df: pd.DataFrame, out_dir: Path):
    """EDA §2: per-feature missingness — bar chart + CSV."""
    miss_pct = (df.isnull().sum() / len(df) * 100).sort_values(ascending=False)
    miss_pct = miss_pct[miss_pct > 0]
    if miss_pct.empty:
        logger.info("No missing values found.")
        return

    miss_df = pd.DataFrame({
        "feature":       miss_pct.index,
        "missing_pct":   miss_pct.values.round(2),
        "missing_count": df[miss_pct.index].isnull().sum().values,
    })
    miss_df.to_csv(out_dir / "missing_values.csv", index=False)
    logger.info("Missing value summary:\n%s", miss_df.to_string(index=False))

    colors = ["#C43235" if p > 30 else "#DD8452" if p > 5 else "#4C72B0"
              for p in miss_df["missing_pct"]]
    fig, ax = plt.subplots(figsize=(8, max(4, len(miss_df) * 0.4)))
    bars = ax.barh(miss_df["feature"], miss_df["missing_pct"], color=colors)
    for bar, pct, cnt in zip(bars, miss_df["missing_pct"], miss_df["missing_count"]):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                f"{pct:.1f}% ({cnt})", va="center", fontsize=9)
    ax.set_xlabel("Missing (%)")
    ax.set_xlim(0, miss_df["missing_pct"].max() * 1.3)
    ax.set_title("Missing Values per Feature\n"
                 "(red >30%: structural — uninstrumented services; "
                 "orange >5%: incidental)")
    plt.tight_layout()
    path = out_dir / "missing_values.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved → %s", path)


# ── C. Blast radius distribution ──────────────────────────────────────────────

def plot_blast_radius_dist(df: pd.DataFrame, out_dir: Path):
    """EDA §3: regression target distribution — overall histogram + by severity."""
    col = "blast_radius_final"
    if col not in df.columns:
        logger.warning("plot_blast_radius_dist: '%s' not in dataset", col)
        return

    exp_df = df.drop_duplicates(subset="experiment_id")
    vals   = exp_df[col].dropna()

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    bins = range(0, int(vals.max()) + 2)
    axes[0].hist(vals, bins=bins, color="#4C72B0", alpha=0.8, edgecolor="white", align="left")
    axes[0].axvline(vals.mean(),   color="red",    linestyle="--", linewidth=1.8,
                    label=f"Mean = {vals.mean():.2f}")
    axes[0].axvline(vals.median(), color="orange", linestyle="--", linewidth=1.8,
                    label=f"Median = {vals.median():.1f}")
    axes[0].set_xlabel("Blast Radius (# services affected)")
    axes[0].set_ylabel("Experiments")
    axes[0].set_title("Blast Radius Final — Overall Distribution")
    axes[0].legend()

    data = [exp_df[exp_df["severity"] == sev][col].dropna().values for sev in SEVERITY_ORDER]
    bp   = axes[1].boxplot(data, patch_artist=True, tick_labels=SEVERITY_ORDER)
    for patch, sev in zip(bp["boxes"], SEVERITY_ORDER):
        patch.set_facecolor(SEVERITY_COLORS[sev])
        patch.set_alpha(0.7)
    axes[1].set_xlabel("Severity Class")
    axes[1].set_ylabel("Blast Radius Final")
    axes[1].set_title("Blast Radius Final by Severity Class")

    fig.suptitle("Regression Target: Blast Radius Final\n"
                 "(services whose latency rose >10% during fault vs baseline)",
                 fontsize=12, y=1.02)
    plt.tight_layout()
    path = out_dir / "blast_radius_dist.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved → %s", path)


# ── D. Time evolution — proactive signal build-up ─────────────────────────────

def plot_time_evolution(df: pd.DataFrame, out_dir: Path):
    """
    EDA §4: the central proactive argument.

    Plots mean ± 1 std of target_latency_delta and target_error_rate per
    severity class across T ∈ {15, 30, 45, 60} seconds.  Clear class
    separation at T=15s is the empirical case for early detectability.
    """
    time_col = "time_into_fault_s"
    if time_col not in df.columns:
        logger.warning("plot_time_evolution: '%s' not in dataset", time_col)
        return

    T_vals = sorted(df[time_col].unique())

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, feat, ylabel, title in [
        (axes[0], "target_latency_delta", "Median Latency Delta (fraction of baseline)",
         "Latency Signal Separation Across Observation Windows"),
        (axes[1], "target_restart_count", "Median Restart Count",
         "Restart Signal Separation Across Observation Windows"),
    ]:
        if feat not in df.columns:
            continue
        for sev in SEVERITY_ORDER:
            medians, q25s, q75s = [], [], []
            for T in T_vals:
                vals = df[(df["severity"] == sev) & (df[time_col] == T)][feat].dropna()
                if len(vals):
                    medians.append(vals.median())
                    q25s.append(vals.quantile(0.25))
                    q75s.append(vals.quantile(0.75))
                else:
                    medians.append(np.nan)
                    q25s.append(np.nan)
                    q75s.append(np.nan)
            medians = np.array(medians, dtype=float)
            q25s    = np.array(q25s,    dtype=float)
            q75s    = np.array(q75s,    dtype=float)
            ax.plot(T_vals, medians, "o-", color=SEVERITY_COLORS[sev],
                    label=sev, linewidth=2.2, markersize=7)
            ax.fill_between(T_vals, q25s, q75s,
                            color=SEVERITY_COLORS[sev], alpha=0.18)
        ax.set_xlabel("Time into Fault Phase (seconds)")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.set_xticks(T_vals)
        ax.legend(title="Severity")

    fig.suptitle(
        "Evolution of median observability signals during the first 60 s of fault injection\n"
        "(shaded region = IQR)",
        fontsize=12, y=1.02,
    )
    plt.tight_layout()
    path = out_dir / "time_evolution.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved → %s", path)

    rows = []
    for feat in ["target_latency_delta", "target_error_rate"]:
        if feat not in df.columns:
            continue
        for sev in SEVERITY_ORDER:
            for T in T_vals:
                vals = df[(df["severity"] == sev) & (df[time_col] == T)][feat].dropna()
                rows.append({
                    "feature":           feat,
                    "severity":          sev,
                    "time_into_fault_s": T,
                    "median":            round(vals.median(),        4) if len(vals) else None,
                    "q25":               round(vals.quantile(0.25),  4) if len(vals) else None,
                    "q75":               round(vals.quantile(0.75),  4) if len(vals) else None,
                    "mean":              round(vals.mean(),          4) if len(vals) else None,
                    "n":                 len(vals),
                })
    pd.DataFrame(rows).to_csv(out_dir / "time_evolution.csv", index=False)
    logger.info("Saved → %s", out_dir / "time_evolution.csv")


# ── E. Feature justification table ───────────────────────────────────────────

def feature_justification_table(df: pd.DataFrame, out_dir: Path):
    """
    EDA §6: per-feature rationale with Spearman correlations vs severity and
    blast_radius_final.  Saved as CSV for direct use in the dissertation.
    """
    plot_df = df.copy()
    plot_df["severity_encoded"] = plot_df["severity"].map({"Low": 0, "Medium": 1, "High": 2})

    FEATURE_META = {
        # Base features
        "target_latency_delta":     ("trace",      "Relative latency change — primary degradation signal"),
        "target_error_rate":        ("trace",      "Error span fraction — direct user-visible impact"),
        "target_request_rate":      ("trace",      "Request throughput; drops signal cascading failure"),
        "target_cpu":               ("prometheus", "Pod CPU; spikes precede OOMKill and latency faults"),
        "target_cpu_delta":         ("prometheus", "Normalised CPU change from baseline"),
        "target_memory_bytes":      ("prometheus", "Pod memory; pressure leads to OOM restarts"),
        "target_memory_delta":      ("prometheus", "Normalised memory change from baseline"),
        "target_replicas":          ("prometheus", "0 = service unavailable → guaranteed High"),
        "target_restart_count":     ("prometheus", "Container restarts under fault → guaranteed High"),
        "system_cpu_mean":          ("prometheus", "Cluster-wide CPU; captures coordinated degradation"),
        "system_memory_mean":       ("prometheus", "Cluster-wide memory; baseline load context"),
        "blast_radius_early":       ("trace",      "Upstream propagation count; quantifies cascade scope"),
        "max_trace_depth":          ("trace",      "Trace depth; long chains amplify tail latency"),
        "time_into_fault_s":        ("context",    "Observation window — enables proactive formulation"),
        "fault_value_numeric":      ("context",    "Fault magnitude; distinguishes mild vs severe injection"),
        # Engineered features (names match feature_engineer.py output columns exactly)
        "fault_severity_ratio":     ("fe",         "Normalised fault magnitude vs service resource limit"),
        "service_depth":            ("fe_topo",    "Call-graph depth — latency amplification along chain"),
        "service_downstream_count": ("fe_topo",    "Fan-out: services this service calls — cascade amplifier"),
        "service_upstream_count":   ("fe_topo",    "Fan-in: services that call this service — criticality"),
        "service_is_gateway":       ("fe_topo",    "1 if service is the user-facing entry point"),
        "system_error_rate":        ("fe_sys",     "System-wide error rate across all services"),
        "system_latency_ratio":     ("fe_sys",     "System-wide latency ratio vs baseline — cluster health"),
    }

    rows = []
    for feat, (category, rationale) in FEATURE_META.items():
        if feat not in plot_df.columns:
            continue
        miss_pct = round(plot_df[feat].isnull().mean() * 100, 1)
        aligned  = plot_df[[feat, "severity_encoded", "blast_radius_final"]].dropna()
        sev_corr = br_corr = None
        if len(aligned) > 10:
            sev_corr = round(
                aligned[[feat, "severity_encoded"]].corr(method="spearman").iloc[0, 1], 3)
            br_corr  = round(
                aligned[[feat, "blast_radius_final"]].corr(method="spearman").iloc[0, 1], 3)
        rows.append({
            "feature":              feat,
            "category":             category,
            "spearman_vs_severity": sev_corr,
            "spearman_vs_blast_r":  br_corr,
            "missing_pct":          miss_pct,
            "rationale":            rationale,
        })

    result = pd.DataFrame(rows)
    path   = out_dir / "feature_justification.csv"
    result.to_csv(path, index=False)
    logger.info("Feature justification table:\n%s",
                result[["feature", "spearman_vs_severity", "missing_pct"]].to_string(index=False))
    return result


# ── Feature–target correlation bar charts (clf vs reg) ───────────────────────

def plot_feature_target_correlations(df: pd.DataFrame, out_dir: Path):
    """
    Two ranked bar charts showing Spearman |correlation| of every numeric feature
    against each target:
      Left  — ranked by |corr vs severity_encoded|  → guides classifier feature selection
      Right — ranked by |corr vs blast_radius_final| → guides regressor feature selection

    Allows direct comparison of which features matter most for each task.
    """
    plot_df = df.copy()
    plot_df["severity_encoded"] = plot_df["severity"].map({"Low": 0, "Medium": 1, "High": 2})

    exclude = {"severity_encoded", "blast_radius_final", "final_latency_delta",
               "experiment_id", "time_into_fault_s", "latency_observable"}
    feature_cols = [
        c for c in plot_df.select_dtypes(include=[np.number]).columns
        if c not in exclude
    ]

    records = []
    for feat in feature_cols:
        aligned = plot_df[[feat, "severity_encoded", "blast_radius_final"]].dropna()
        if len(aligned) < 10:
            continue
        clf_corr = aligned[[feat, "severity_encoded"]].corr(method="spearman").iloc[0, 1]
        reg_corr = aligned[[feat, "blast_radius_final"]].corr(method="spearman").iloc[0, 1]
        records.append({"feature": feat,
                        "clf_spearman": round(clf_corr, 3),
                        "reg_spearman": round(reg_corr, 3)})

    if not records:
        logger.warning("plot_feature_target_correlations: no numeric features found")
        return

    corr_df = pd.DataFrame(records)
    corr_df.to_csv(out_dir / "feature_target_correlations.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(16, max(6, len(records) * 0.35 + 1.5)))

    for ax, col, title, color in [
        (axes[0], "clf_spearman",
         "Classification — |Spearman vs severity|\n(higher = more useful for clf)",
         "#4C72B0"),
        (axes[1], "reg_spearman",
         "Regression — |Spearman vs blast_radius_final|\n(higher = more useful for reg)",
         "#DD8452"),
    ]:
        ranked = corr_df.reindex(
            corr_df[col].abs().sort_values(ascending=True).index
        )
        bars = ax.barh(ranked["feature"], ranked[col].abs(),
                       color=color, alpha=0.8, edgecolor="white")
        # Colour bars negative correlations slightly differently
        for bar, val in zip(bars, ranked[col]):
            if val < 0:
                bar.set_alpha(0.5)
                bar.set_hatch("//")
        ax.set_xlabel("|Spearman correlation|")
        ax.set_title(title, fontsize=11)
        ax.set_xlim(0, 1)
        ax.axvline(0.3, color="grey", linestyle=":", linewidth=1, label="weak (0.3)")
        ax.axvline(0.5, color="grey", linestyle="--", linewidth=1, label="moderate (0.5)")
        ax.legend(fontsize=8)

    fig.suptitle("Feature–Target Correlations: Classification vs Regression",
                 fontsize=13, y=1.01)
    plt.tight_layout()
    path = out_dir / "feature_target_correlations.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved → %s", path)


# ── Main ───────────────────────────────────────────────────────────────────────

def main(data_path: Path, out_dir: Path, upload_prefix: str | None, mode: str):
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading dataset from %s  [mode=%s]", data_path, mode)
    df = pd.read_csv(data_path)

    if mode == "threshold":
        # Use full dataset — threshold validation is about the labeling methodology,
        # decided before any train/test split existed.
        logger.info(
            "Threshold mode: using full dataset (%d experiments)",
            df["experiment_id"].nunique(),
        )
    else:
        # Feature EDA informs model decisions → must exclude holdout.
        holdout_path = REPO_ROOT / "dataset" / "processed" / "holdout_ids.txt"
        if holdout_path.exists():
            holdout_ids = set(holdout_path.read_text().strip().splitlines())
            before = df["experiment_id"].nunique()
            df = df[~df["experiment_id"].isin(holdout_ids)].reset_index(drop=True)
            logger.info(
                "Holdout excluded: %d → %d training experiments (%d withheld)",
                before, df["experiment_id"].nunique(),
                before - df["experiment_id"].nunique(),
            )
        else:
            logger.warning(
                "holdout_ids.txt not found — feature EDA includes ALL experiments. "
                "Run create_holdout.py first."
            )

    logger.info("Dataset: %d rows, %d columns, %d experiments",
                len(df), len(df.columns), df["experiment_id"].nunique())
    logger.info("Severity distribution:\n%s",
                df["severity"].value_counts().reindex(SEVERITY_ORDER).to_string())

    if mode == "threshold":
        # ── Threshold validation (full dataset, before split) ─────────────────
        logger.info("Computing threshold summary…")
        summary = threshold_summary(df)
        summary.to_csv(out_dir / "threshold_summary.csv", index=False)
        logger.info("\nThreshold summary:\n%s", summary.to_string(index=False))

        logger.info("Dataset overview…")
        dataset_overview(df, out_dir)

        logger.info("Missing value analysis…")
        missing_value_analysis(df, out_dir)

        logger.info("Target distributions…")
        plot_class_distribution(df, out_dir)
        plot_blast_radius_dist(df, out_dir)
        plot_latency_delta_dist(df, out_dir)
        plot_error_rate_dist(df, out_dir)
        plot_latency_kde(df, out_dir)

        logger.info("Threshold validation…")
        threshold_confusion_table(df, out_dir)
        coverage_table(df, out_dir)
        sensitivity_analysis(df, out_dir)

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

    else:
        # ── Feature EDA (training set only, run before base features and after FE) ──
        is_fe = any(c in df.columns for c in
                    ["fault_severity_ratio", "topology_fan_out", "sys_trace_error_rate"])
        label = "FE features" if is_fe else "base features"
        logger.info("Feature EDA on %s", label)

        logger.info("Missing value analysis…")
        missing_value_analysis(df, out_dir)

        logger.info("Time evolution (proactive signal build-up)…")
        plot_time_evolution(df, out_dir)

        logger.info("Feature boxplots by severity class (classification)…")
        plot_feature_boxplots(df, out_dir)

        logger.info("Correlation heatmap…")
        plot_correlation_heatmap(df, out_dir)

        logger.info("Feature–target correlations: clf vs reg…")
        plot_feature_target_correlations(df, out_dir)

        logger.info("Feature justification table…")
        feature_justification_table(df, out_dir)

    # ── Upload to GCS ─────────────────────────────────────────────────────────
    if upload_prefix:
        import subprocess
        remote = upload_prefix.rstrip("/") + f"/threshold_analysis/{mode}/"
        logger.info("Uploading %s → %s", out_dir, remote)
        subprocess.run(
            ["gcloud", "storage", "cp", "-r", str(out_dir) + "/", remote],
            check=True,
        )

    logger.info("\nAll outputs → %s/", out_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Threshold analysis and feature EDA.\n\n"
            "Run order:\n"
            "  1. --mode threshold  (before create_holdout.py, full dataset)\n"
            "  2. --mode features --data training_dataset.csv     (after holdout split)\n"
            "  3. --mode features --data training_dataset_fe.csv  (after feature_engineer.py)"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--mode",   choices=["threshold", "features"], required=True,
                        help="'threshold' = label validation on full data; "
                             "'features' = feature EDA on training set only")
    parser.add_argument("--data",   type=Path, default=DEFAULT_DATA,
                        help="Input dataset CSV (default: training_dataset.csv)")
    parser.add_argument("--out",    type=Path, default=None,
                        help="Output directory (auto-derived from --mode and --data if omitted)")
    parser.add_argument("--upload", type=str,  default=None,
                        help="GCS prefix to upload outputs (e.g. gs://bucket/processed)")
    args = parser.parse_args()

    if args.out is None:
        if args.mode == "threshold":
            out_dir = REPO_ROOT / "results" / "threshold_analysis" / "threshold"
        else:
            is_fe = "fe" in args.data.stem
            out_dir = REPO_ROOT / "results" / "threshold_analysis" / (
                "features_fe" if is_fe else "features_base"
            )
    else:
        out_dir = args.out

    main(args.data, out_dir, args.upload, args.mode)
