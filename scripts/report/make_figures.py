#!/usr/bin/env python
"""
Draw the report's data figures from the project's result files, in one plain style for print.

  report/figures/*.pdf     one figure per file, \\includegraphics them in the chapters

Re-run after any change to the results:   [REPORT_DIR=<LaTeX project>] venv/bin/python scripts/report/make_figures.py [--preview DIR]
--preview DIR also saves PNG copies to DIR, to look at them. The rollout figure needs the span cache
(dataset/spans/, built by ./scripts/run_genai_eval.sh prepare) and is skipped without it.
"""
from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
RES, PROC = ROOT / "results", ROOT / "dataset" / "processed"
# the LaTeX project: REPORT_DIR if set (e.g. a clone of the Overleaf project), otherwise report/ (not in git)
OUT = Path(os.environ.get("REPORT_DIR", ROOT / "report")).expanduser() / "figures"
PREVIEW: Path | None = None

# colours: the first three categorical slots of the dataviz reference palette (distinct for every pairing and
# for colour-blind readers), grey for the no-ML baselines, one blue ramp for counts
BLUE, ORANGE, AQUA, GREY = "#2a78d6", "#eb6834", "#1baf7a", "#8a8985"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e6e5e1"
RAMP = LinearSegmentedColormap.from_list("blues", ["#fcfcfb", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])
WIDTH = 6.0   # inches, about the text width
SEV = ["Low", "Medium", "High"]
ORDER = ["frontend", "checkoutservice", "recommendationservice", "productcatalogservice", "cartservice",
         "currencyservice", "shippingservice", "paymentservice", "emailservice", "adservice"]

plt.rcParams.update({
    "font.size": 9, "axes.labelsize": 9, "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
    "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "axes.grid.axis": "y",
    "grid.color": GRID, "grid.linewidth": 0.6, "lines.linewidth": 1.8, "lines.markersize": 5.5,
    "legend.frameon": False, "pdf.fonttype": 42, "savefig.bbox": "tight", "savefig.pad_inches": 0.03,
})


def short(s: str) -> str:
    return s.replace("service", "")


def save(fig, name: str) -> None:
    fig.savefig(OUT / f"{name}.pdf", metadata={"CreationDate": None})   # no date: same figure, same file
    if PREVIEW:
        fig.savefig(PREVIEW / f"{name}.png", dpi=200)
    plt.close(fig)


def end_label(ax, x, y, text, dy=0.0) -> None:
    """Direct label at the end of a line, in text ink (the line's colour carries the identity)."""
    ax.annotate(text, (x, y), xytext=(7, dy), textcoords="offset points", va="center", color=INK2, fontsize=8,
                annotation_clip=False)


def accuracy_vs_time() -> None:
    a = pd.read_csv(RES / "fe_tuned" / "accuracy_vs_time_holdout.csv")
    clf = next(m for m in a.model.unique() if not m.endswith("Baseline"))
    fig, ax = plt.subplots(figsize=(WIDTH * 0.7, 2.4))
    for m, label, color, ls in ((clf, f"Tuned {clf}", BLUE, "-"), ("FullRuleBaseline", "Reactive alerting", ORANGE, "--")):
        d = a[a.model == m].sort_values("time_into_fault_s")
        ax.plot(d.time_into_fault_s, d.accuracy, ls, color=color, marker="o", label=label)
        for x, y in zip(d.time_into_fault_s, d.accuracy):      # the value of each point, above the model's line, below the rule's
            ax.annotate(f"{y:.2f}", (x, y), xytext=(0, 6 if m == clf else -11), textcoords="offset points",
                        ha="center", fontsize=7, color=INK2)
        end_label(ax, d.time_into_fault_s.iloc[-1], d.accuracy.iloc[-1], label)
    ax.set(xlabel="Seconds since the change", ylabel="Holdout accuracy", xticks=[15, 30, 45, 60], ylim=(0.5, 1.0),
           xlim=(12, 63))
    ax.legend(loc="lower left")
    save(fig, "accuracy_vs_time")


def mae_vs_time() -> None:
    m = pd.read_csv(RES / "fe_tuned" / "mae_vs_time_holdout.csv")
    reg = next(x for x in m.model.unique() if not x.endswith("Baseline"))
    fig, ax = plt.subplots(figsize=(WIDTH * 0.7, 2.4))
    for name, label, color, ls in ((reg, f"Tuned {reg}", BLUE, "-"), ("EarlyBlastBaseline", "Reactive (spread so far)", ORANGE, "--"),
                                   ("NoSpreadBaseline", "Always zero", GREY, ":")):
        d = m[m.model == name].sort_values("time_into_fault_s")
        ax.plot(d.time_into_fault_s, d.mae, ls, color=color, marker="o", label=label)
        end_label(ax, d.time_into_fault_s.iloc[-1], d.mae.iloc[-1], label, dy=-6 if name == reg else 6 if "Early" in name else 0)
    ax.set(xlabel="Seconds since the change", ylabel="Holdout MAE (services)", xticks=[15, 30, 45, 60], ylim=(0, 0.8),
           xlim=(12, 63))
    ax.legend(loc="lower left")
    save(fig, "mae_vs_time")


def confusion() -> None:
    p = pd.read_csv(RES / "fe_tuned" / "holdout_predictions.csv")
    clf = p.model.iloc[0]
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH * 0.8, 2.3))
    mats = [pd.crosstab(p.true_label, p[c]).reindex(index=SEV, columns=SEV, fill_value=0).to_numpy()
            for c in ("pred_label", "baseline_pred")]
    vmax = max(x.max() for x in mats)
    for ax, mat, title in zip(axes, mats, (f"Tuned {clf}", "Reactive alerting")):
        ax.imshow(mat, cmap=RAMP, vmin=0, vmax=vmax)
        for i in range(3):
            for j in range(3):
                ax.text(j, i, mat[i, j], ha="center", va="center", fontsize=8,
                        color="white" if mat[i, j] > 0.55 * vmax else INK)
        ax.set_xticks(range(3), SEV); ax.set_yticks(range(3), SEV)
        ax.set_xlabel("Predicted"); ax.set_title(title, fontsize=9)
        ax.grid(False)
        for s in ax.spines.values():
            s.set_visible(False)
        ax.tick_params(length=0)
    axes[0].set_ylabel("Actual")
    fig.tight_layout(w_pad=2)
    save(fig, "confusion")


def confusion_untuned() -> None:
    """Untuned classifiers, cross-validation on the training rows (scripts/analysis/screening_confusion.py)."""
    c = pd.read_csv(RES / "reliability" / "screening_confusion" / "confusion.csv")
    models = (("RandomForest", "Random forest"), ("XGBoost", "XGBoost"), ("LightGBM", "LightGBM"))
    sets = (("Base", "baseline", "base features"), ("Engineered", "fe", "engineered features"))
    vmax = c.filter(like="pred_").to_numpy().max()
    fig, axes = plt.subplots(2, 3, figsize=(WIDTH, 4.4))
    for row, (feat, stage, feat_label) in zip(axes, sets):
        rep = pd.read_csv(RES / stage / "classification_report.csv").set_index("model")
        for ax, (m, label) in zip(row, models):
            g = c[(c.features == feat) & (c.model == m)].set_index("true").loc[SEV]
            mat = g[[f"pred_{s}" for s in SEV]].to_numpy()
            ax.imshow(mat, cmap=RAMP, vmin=0, vmax=vmax)
            for i in range(3):
                for j in range(3):
                    ax.text(j, i, mat[i, j], ha="center", va="center", fontsize=8,
                            color="white" if mat[i, j] > 0.55 * vmax else INK)
            ax.set_xticks(range(3), SEV, fontsize=7); ax.set_yticks(range(3), SEV, fontsize=7)
            ax.set_xlabel("Predicted", fontsize=8); ax.set_ylabel("Actual", fontsize=8)
            ax.set_title(f"{label}, {feat_label}\naccuracy {rep.loc[m].accuracy:.3f}, macro-F1 {rep.loc[m].macro_f1:.3f}", fontsize=8)
            ax.grid(False); ax.tick_params(length=0)
            for sp in ax.spines.values():
                sp.set_visible(False)
    fig.tight_layout(h_pad=1.5, w_pad=1.2)
    save(fig, "confusion_untuned")


NAMES = {"fault_type": "type of change", "target_cpu_delta": "change in CPU use", "target_cpu": "CPU use",
         "fault_value_numeric": "size of the change", "target_error_rate": "error rate", "traffic_level": "traffic level",
         "target_memory_bytes": "memory use", "target_latency_delta": "change in latency", "target_latency_mean_ms": "latency",
         "target_request_rate": "request rate", "system_memory_mean": "memory use, all services",
         "system_cpu_mean": "CPU use, all services", "target_memory_delta": "change in memory use",
         "target_replicas": "replicas", "target_restart_count": "restarts", "service": "changed service",
         "time_into_fault_s": "time since the change", "max_trace_depth": "trace depth",
         "blast_radius_early": "early blast radius", "latency_observable": "service exports spans",
         "fault_severity_ratio": "size relative to original", "service_upstream_count": "services that call it",
         "service_downstream_count": "services it calls", "service_depth": "depth in call graph",
         "system_error_rate": "error rate, all services", "system_latency_ratio": "latency, all services"}


def _shap_bar(csv: Path, name: str) -> None:
    s = pd.read_csv(csv).abs().mean().sort_values().tail(12)
    fig, ax = plt.subplots(figsize=(WIDTH * 0.7, 2.8))
    ax.set_axisbelow(True)
    ax.barh([NAMES.get(c, c) for c in s.index], s.values, color=BLUE, height=0.62)
    ax.grid(axis="y", visible=False); ax.grid(axis="x", visible=True)
    ax.set_xlabel("Mean |SHAP value|")
    ax.tick_params(axis="y", length=0)
    save(fig, name)


def shap() -> None:
    _shap_bar(RES / "baseline_clf_xgb" / "plots" / "shap_values.csv", "shap")
    _shap_bar(RES / "fe_reg_xgb" / "plots" / "reg_shap_values.csv", "shap_blast")


def threshold_sensitivity() -> None:
    t = pd.read_csv(RES / "threshold_analysis" / "threshold" / "sensitivity_analysis.csv")
    colors = dict(zip(SEV, (BLUE, ORANGE, AQUA)))
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH, 2.3), sharey=True)
    for ax, vary, col, xlabel in ((axes[0], "high_threshold", "high_t", "High threshold (relative latency rise)"),
                                  (axes[1], "medium_threshold", "medium_t", "Medium threshold (relative latency rise)")):
        d = t[t.vary == vary].sort_values(col)
        for s in SEV:
            ax.plot(d[col], d[s], color=colors[s], marker="o", label=s)
        ax.axvline(d.loc[d.current, col].iloc[0], color=INK2, ls="--", lw=0.9, label="threshold used")
        ax.set_xlabel(xlabel)
    for s in SEV:
        d = t[t.vary == "medium_threshold"].sort_values("medium_t")
        end_label(axes[1], d.medium_t.iloc[-1], d[s].iloc[-1], s)
    axes[0].set_ylabel("Runs")
    axes[0].set_ylim(0, 160)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.08))
    fig.tight_layout(w_pad=1.5)
    save(fig, "threshold_sensitivity")


def spread_matrix() -> None:
    lm = pd.read_csv(PROC / "label_measurements.csv")
    mat = pd.DataFrame(0, index=ORDER, columns=ORDER)
    for r in lm.itertuples():
        for s in (r.blast_affected.split(";") if isinstance(r.blast_affected, str) and r.blast_affected else []):
            mat.loc[s, r.service] += 1
    runs = lm.service.value_counts()
    rows = [s for s in ORDER if mat.loc[s].sum() > 0]          # services never affected are left out
    sub = mat.loc[rows]
    shown = sub.to_numpy().astype(float)
    for i, s in enumerate(rows):
        shown[i, ORDER.index(s)] = np.nan                        # a service is not counted against itself
    fig, ax = plt.subplots(figsize=(WIDTH * 0.75, 0.42 * len(rows) + 1.3))
    ax.imshow(shown, cmap=RAMP, vmin=0, vmax=np.nanmax(shown))
    for i in range(len(rows)):
        for j in range(len(ORDER)):
            if not np.isnan(shown[i, j]) and sub.iat[i, j]:
                ax.text(j, i, sub.iat[i, j], ha="center", va="center", fontsize=7.5,
                        color="white" if sub.iat[i, j] > 0.55 * np.nanmax(shown) else INK)
    ax.set_xticks(range(len(ORDER)), [f"{short(s)} ({runs[s]})" for s in ORDER], rotation=45, ha="right")
    ax.set_yticks(range(len(rows)), [short(s) for s in rows])
    ax.set_xlabel("Changed service (runs)"); ax.set_ylabel("Affected service")
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0)
    save(fig, "spread_matrix")


def rollout(run: str = "EXP_0162", service: str = "currencyservice") -> None:
    cache = ROOT / "dataset" / "spans" / f"{run}.npz"
    meta_path = ROOT / "dataset" / "raw" / run / "metadata.json"
    if not cache.exists() or not meta_path.exists():
        print(f"{cache} or {meta_path} missing: rollout figure skipped")
        return
    ph = json.loads(meta_path.read_text())["phases"]
    t0, t_end = ph["fault"]["start"], ph["fault"]["end"]
    with np.load(cache) as z:
        calls = z[f"{service}__calls"]
    t, dur, err = calls[:, 0] - t0, calls[:, 1], calls[:, 2] > 0
    keep = (t >= ph["baseline"]["start"] - t0) & (t <= t_end - t0)
    t, dur, err = t[keep], dur[keep], err[keep]
    # pod replacement, from the Kubernetes events recorded with the run
    ev = json.loads((ROOT / "dataset" / "raw" / run / "events.json").read_text())["events"]
    marks = {}
    for e in ev:
        name = e.get("involvedObject", {}).get("name", "")
        ts = e.get("lastTimestamp") or e.get("eventTime")
        if not name.startswith(service) or not ts:
            continue
        dt = datetime.fromisoformat(ts.replace("Z", "+00:00")).timestamp() - t0
        if 0 <= dt <= 60 and e.get("reason") in ("Started", "Killing"):
            marks.setdefault(e["reason"], dt)
    fig, ax = plt.subplots(figsize=(WIDTH, 2.4))
    label_start = t_end - t0 - 60
    ax.axvspan(label_start, t_end - t0, color=GRID, alpha=0.7, lw=0)
    ax.text((label_start + t_end - t0) / 2, 4e4, "label window", ha="center", va="top", fontsize=8, color=INK2)
    ax.scatter(t[~err], dur[~err], s=4, color=BLUE, alpha=0.5, lw=0, label="call", rasterized=True)
    ax.scatter(t[err], dur[err], s=14, color=ORANGE, marker="x", lw=1, label="failed call")
    ax.axvline(0, color=INK, lw=0.9)
    ax.text(-1.5, 300, "change", fontsize=8, color=INK2, rotation=90, ha="right", va="center")
    for reason, text in (("Started", "new pod started"), ("Killing", "old pod stopped")):
        if reason in marks:
            ax.axvline(marks[reason], color=INK2, lw=0.8, ls="--")
            ax.text(marks[reason] - 1.5, 300, text, fontsize=8, color=INK2, rotation=90, ha="right", va="center")
    ax.set_yscale("log")
    ax.set(xlabel="Seconds since the change", ylabel="Call duration (ms)", ylim=(0.08, 6e4),
           xlim=(ph["baseline"]["start"] - t0, t_end - t0))
    ax.legend(loc="upper left", markerscale=2)
    save(fig, "rollout")
    print(f"rollout: {run} {service}, pod events at {marks}")


def main() -> None:
    global PREVIEW
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--preview", type=Path, help="also save PNG copies here")
    a = ap.parse_args()
    if a.preview:
        PREVIEW = a.preview
        PREVIEW.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    figures = (accuracy_vs_time, mae_vs_time, confusion, confusion_untuned, shap, threshold_sensitivity, spread_matrix, rollout)
    for f in figures:
        f()
    print(f"{len(figures)} figures -> {OUT}")


if __name__ == "__main__":
    main()
