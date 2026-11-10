#!/usr/bin/env python
"""
Generate the report's tables, key numbers and figures from the project's result files, so every
number in the report comes from the data and nothing is typed by hand.

  report/tables/*.tex      one table per file (booktabs), \\input{} them in the chapters
  report/tables/numbers.tex  \\newcommand macros for numbers used in the text, e.g. \\nRuns

 [REPORT_DIR=<LaTeX project>] venv/bin/python scripts/report/make_tables.py
"""
from __future__ import annotations

import csv
import json
import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(1, str(ROOT))
RES, PROC, RAW = ROOT / "results", ROOT / "dataset" / "processed", ROOT / "dataset"
# the LaTeX project: REPORT_DIR if set (e.g. a clone of the Overleaf project), otherwise report/ (not in git)
REPORT_DIR = Path(os.environ.get("REPORT_DIR", ROOT / "report")).expanduser()
OUT_T = REPORT_DIR / "tables"
SEV = ["Low", "Medium", "High"]
CAUSES = {"cpu": "CPU limit", "memory": "Memory limit", "latency": "Network latency", "replicas": "Replicas to 0"}
PHASES = [("phase1_initial_matrix", "1. Initial matrix"), ("phase2_coverage", "2. Coverage"),
          ("phase3_rerun", "3. Re-runs on consistent hardware"), ("phase3_targeted", "3. Targeted settings and repeats")]
numbers: dict[str, str] = {}


def num(name: str, value, fmt: str = "{}") -> None:
    numbers[name] = fmt.format(value)


def f3(x) -> str:
    return "--" if pd.isna(x) else f"{x:.3f}"


def f2(x) -> str:
    return "--" if pd.isna(x) else f"{x:.2f}"


def bootstrap_diff(runs: pd.DataFrame, a: str, b: str, draws: int = 5000, seed: int = 0) -> tuple[float, float, float]:
    """Mean of runs[a] - runs[b] with a 95% interval from resampling the runs (rows of `runs`)."""
    d = (runs[a] - runs[b]).to_numpy()
    boot = d[np.random.default_rng(seed).integers(0, len(d), size=(draws, len(d)))].mean(axis=1)
    return float(np.percentile(boot, 2.5)), float(d.mean()), float(np.percentile(boot, 97.5))


def pct(x, d=0) -> str:
    return "--" if pd.isna(x) else f"{100 * x:.{d}f}\\%"


# short titles for the list of tables (the full caption stays under the table)
SHORT = {
    "tab:software": "Software deployed for the experiments",
    "tab:changes": "The four types of change",
    "tab:changes_detail": "Change types and values per service",
    "tab:validation": "Validation of the blast-radius measurement",
    "tab:features": "Features used by the models",
    "tab:model_screening": "Untuned regressors on the engineered features",
    "tab:tuning": "Hyperparameter search for the XGBoost models",
    "tab:llm_models": "The three LLMs evaluated",
    "tab:composition": "Runs by type of change and final severity",
    "tab:exposure": "Structural exposure of each service",
    "tab:severity_results": "Severity forecasting against reactive alerting",
    "tab:forecasting_gain": "Where the forecast adds value",
    "tab:blast_results": "Blast-radius forecasting against baselines",
    "tab:retrieval": "Retrieval methods on the training runs and the holdout",
    "tab:user_impact": "Severity of the changed service and effect on users",
    "tab:binary": "Severity forecast as a yes-or-no decision",
    "tab:dev_rounds": "Development rounds of the agent",
    "tab:llm_main": "Holdout results of the LLM systems",
    "tab:llm_paired": "Paired differences in severity accuracy",
    "tab:llm_affected": "Affected services named by the agents",
    "tab:llm_sensitivity": "Prompt sensitivity of the agents",
    "tab:threshold_sensitivity": "Sensitivity of the labels to the latency thresholds",
    "tab:correlations": "Correlation of each feature with the two targets",
    "tab:screening_errors": "Untuned classifiers on the base features",
}


def write_table(name: str, caption: str, label: str, header: list[str], rows: list[list], align: str,
                note: str | None = None, size: str = r"\small", colsep: str | None = None, groups: list | None = None) -> None:
    align = align.replace("p{", r">{\raggedright\arraybackslash}p{")      # wrapped columns: ragged, not justified
    lines = [r"\begin{table}[htbp]", r"\centering", size] + ([rf"\setlength{{\tabcolsep}}{{{colsep}}}"] if colsep else []) + \
            [rf"\begin{{tabular}}{{{align}}}", r"\toprule",
             " & ".join(rf"\textbf{{{h}}}" for h in header) + r" \\", r"\midrule"]
    if groups:
        pos, cm, gl = 1, [], []
        for t, n in groups:
            gl.append(rf"\multicolumn{{{n}}}{{c}}{{\textbf{{{t}}}}}" if t else rf"\multicolumn{{{n}}}{{c}}{{}}")
            if t:
                cm.append(rf"\cmidrule(lr){{{pos}-{pos+n-1}}}")
            pos += n
        lines[-2:-2] = [" & ".join(gl) + r" \\", "".join(cm)]
    for r in rows:
        lines.append(r"\midrule" if r == "MID" else " & ".join(str(c) for c in r) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    if note:
        lines.append(rf"\par\smallskip\parbox{{0.92\linewidth}}{{\footnotesize {note}}}")
    short = SHORT.get(label)
    lines += [rf"\caption[{short}]{{{caption}}}" if short else rf"\caption{{{caption}}}", rf"\label{{{label}}}",
              r"\end{table}"]
    (OUT_T / f"{name}.tex").write_text("% generated by scripts/report/make_tables.py, do not edit by hand\n" + "\n".join(lines) + "\n")


# dataset

def dataset_tables():
    ds = pd.read_csv(PROC / "training_dataset.csv").drop_duplicates("experiment_id")
    ho = set((PROC / "holdout_ids.txt").read_text().split())
    ds["setting"] = ds.service + "|" + ds.fault_type + "|" + ds.fault_value.astype(str) + "|" + ds.traffic_level
    num("nRuns", len(ds)); num("nSettings", ds.setting.nunique()); num("nRepeated", int((ds.setting.value_counts() == 2).sum()))
    num("nServices", ds.service.nunique()); num("nRows", len(ds) * 4)
    num("nHoldoutRuns", ds.experiment_id.isin(ho).sum()); num("nTrainRuns", (~ds.experiment_id.isin(ho)).sum())
    num("nHoldoutSettings", ds[ds.experiment_id.isin(ho)].setting.nunique())
    ct = pd.crosstab(ds.fault_type, ds.severity).reindex(columns=SEV, fill_value=0)
    bt = pd.crosstab(ds.fault_type, ds.blast_radius_final).reindex(columns=[0, 1, 2, 3], fill_value=0)
    rows = [[CAUSES[f], *[int(ct.loc[f, s]) for s in SEV], *[int(bt.loc[f, b]) for b in range(4)], int(ct.loc[f].sum())] for f in ct.index]
    rows += ["MID", ["Total", *[int(ct[s].sum()) for s in SEV], *[int(bt[b].sum()) for b in range(4)], int(ct.values.sum())]]
    for s in SEV:
        num(f"nSev{s}", int(ct[s].sum()))
    for b, w in enumerate(["Zero", "One", "Two", "Three"]):
        num(f"nBlast{w}", int(bt[b].sum()))
    write_table("dataset_composition", "Runs in the final dataset by type of change, counted by final severity and the blast radius",
                "tab:composition", ["Type of change", "Low", "Medium", "High", "0", "1", "2", "3", "Runs"], rows, "lcccccccc",
                groups=[("", 1), ("Final severity", 3), ("Blast radius", 4), ("", 1)])

    # the values used per service and type of change, with the service's original limit in brackets
    from feature_engineer import ORIGINAL_LIMITS
    def _num(v):
        m = re.match(r"[0-9.]+", str(v))
        return float(m.group()) if m else 0.0
    rows = []
    for svc, g in ds.groupby("service"):
        cells = []
        for f in ("cpu", "memory", "latency", "replicas"):
            vals = sorted(g[g.fault_type == f].fault_value.astype(str).unique(), key=_num)
            text = ", ".join(re.sub(r"(m|Mi)$", "", v) for v in vals) or "--"
            if f == "replicas":                     # the only value is 0, so show how often it was used
                text = str(int((g.fault_type == f).sum()))
            orig = ORIGINAL_LIMITS.get(svc, {}).get(f) if f in ("cpu", "memory") else None
            cells.append(f"{text} ({orig:g})" if orig else text)
        rows.append([svc.replace("service", ""), *cells, len(g)])
    write_table("changes", "Change types and the values used, per service. CPU limits are in millicores and memory "
                "limits in MiB, with the service's original limit in brackets; delays are in milliseconds. The column "
                "``Scaled to 0'' counts the runs in which the service was scaled from one replica to zero.", "tab:changes_detail",
                ["Service", "CPU limit", "Memory limit", "Delay (ms)", "Scaled to 0", "Runs"], rows,
                "lp{0.17\\linewidth}p{0.17\\linewidth}p{0.22\\linewidth}cc", size=r"\footnotesize",
                note=f"Runs were recorded under normal load (10 simulated users) or high load (25 users); "
                     f"{int((ds.traffic_level == 'high').sum())} of the {len(ds)} runs are under high load.")

    # the simple overview used in the methodology: one row per type of change
    def _range(f):
        v = [_num(x) for x in ds[ds.fault_type == f].fault_value.astype(str)]
        return f"{min(v):g}--{max(v):g}"
    orig = lambda f: sorted({lim[f] for lim in ORIGINAL_LIMITS.values()})
    what = {"cpu": f"Lowered from the original limit ({' or '.join(f'{o:g}' for o in orig('cpu'))} millicores) towards the service's request",
            "memory": f"Lowered from the original limit ({min(orig('memory')):g}--{max(orig('memory')):g} MiB), never below the request",
            "latency": "Added to all traffic of the service's pod",
            "replicas": "Scaled to zero, so the service is unavailable"}
    unit = {"cpu": " millicores", "memory": " MiB", "latency": " ms", "replicas": ""}
    rows = [[CAUSES[f].replace("Replicas to 0", "Replicas"), what[f], ("0" if f == "replicas" else _range(f) + unit[f]),
             int((ds.fault_type == f).sum())] for f in ("cpu", "memory", "latency", "replicas")]
    rows += ["MID", [r"\textbf{Total}", "", "", rf"\textbf{{{len(ds)}}}"]]
    n_high = int((ds.traffic_level == "high").sum())
    write_table("changes_summary", "The four types of change. Each run changes one thing on one service, while everything "
                "else keeps its original setting. All ten services were changed in every way, under normal load "
                "(10 simulated users) or high load (25 users); "
                "Appendix~B lists the values used for each service.", "tab:changes",
                ["Type of change", "What is changed", "Values used", "Runs"], rows,
                "lp{0.42\\linewidth}lc")

    # collection phases, from the plan state files (dataset/<plan>_state.json), which record the run each plan row
    # produced, plus any earlier attempts that were set aside. A run is kept if it is in the final dataset,
    # superseded if phase3_rerun.csv re-ran it, and excluded otherwise (reasons in dataset/excluded/README.md).
    kept = set(ds.experiment_id)
    superseded = set()
    for r in csv.DictReader(open(ROOT / "experiments" / "plans" / "phase3_rerun.csv")):
        superseded |= set(re.findall(r"Re-run of (EXP_\d{4})", r["purpose"]))
    rows, seen = [], []
    for plan, label in PHASES:
        entries = json.loads((RAW / f"{plan}_state.json").read_text())["experiments"]
        runs = [e["exp_id"] for e in entries if e.get("exp_id")] + \
               [x["exp_id"] for e in entries for x in e.get("excluded_runs", [])]
        seen += runs
        k, s = sum(r in kept for r in runs), sum(r in superseded for r in runs)
        rows.append([label, len(entries), len(runs), k, s, len(runs) - k - s])
    if set(seen) & kept != kept or len(seen) != len(set(seen)):
        raise SystemExit("plan state files and the final dataset disagree: check dataset/*_state.json")
    num("nRecorded", sum(r[2] for r in rows)); num("nSuperseded", sum(r[4] for r in rows))
    num("nExcluded", sum(r[5] for r in rows))


# ML forecasting vs no-ML baselines

LABEL = {"XGBoost": "XGBoost", "RandomForest": "random forest", "LightGBM": "LightGBM"}


def ml_tables():
    global CLF, REG
    c = pd.read_csv(RES / "fe_tuned" / "baseline_comparison.csv")
    # the chosen models are the rows that are not baselines (run_pipeline.sh gathers them into results/fe_tuned)
    CLF = next(m for m in c[c.task == "severity"].model if not m.endswith("Baseline"))
    REG = next(m for m in c[c.task == "blast_radius"].model if not m.endswith("Baseline"))
    sev = c[c.task == "severity"].set_index(["split", "model"])
    rows = []
    for split in ("cv", "holdout"):
        for label, r in ((f"Tuned {LABEL[CLF]}", sev.loc[(split, CLF)]),
                         ("Reactive alerting", sev.loc[(split, "FullRuleBaseline")])):
            head = split == "holdout" and label.startswith("Tuned")      # the headline numbers are set in bold
            bold = (lambda v: rf"\textbf{{{v}}}") if head else (lambda v: v)
            rows.append(["CV" if split == "cv" else "Holdout", label, bold(f3(r.accuracy)), bold(f3(r.macro_f1)),
                         f3(r.Low_f1), f3(r.Medium_f1), f3(r.High_f1)])
        if split == "cv":
            rows.append("MID")
    write_table("severity_results", "Severity forecasting: the tuned classifier against the severity rule applied to the "
                "current snapshot (reactive alerting), in grouped cross-validation on the training runs and on the holdout.", "tab:severity_results",
                ["Split", "System", "Accuracy", "Macro-F1", "Low F1", "Medium F1", "High F1"], rows, "llccccc", colsep="4.5pt")
    br = c[c.task == "blast_radius"].set_index(["split", "model"])
    graph = pd.read_csv(RES / "reliability" / "structural_exposure" / "graph_only_baseline.csv").set_index("split")
    # R^2 is given for the holdout only: the models were chosen on the cross-validation MAE
    r2 = lambda x: "--" if pd.isna(x) else (f"$-${-x:.2f}" if x < 0 else f"{x:.2f}")
    rows = []
    for s, gsplit in (("cv", "training"), ("holdout", "holdout")):
        for m, label in ((REG, f"Tuned {LABEL[REG]}"), ("EarlyBlastBaseline", "Reactive (spread so far)"),
                         ("NoSpreadBaseline", "Always zero")):
            mae = f3(br.loc[(s, m)].mae)
            if s == "holdout" and m == REG:                                  # the headline number is set in bold
                mae = rf"\textbf{{{mae}}}"
            rows.append(["CV" if s == "cv" else "Holdout", label, mae, f3(br.loc[(s, m)].rmse),
                         r2(br.loc[(s, m)].r2) if s == "holdout" else "--"])
        rows.append(["CV" if s == "cv" else "Holdout", "Call graph (all exposed services)",
                     f3(graph.loc[gsplit, "graph_only_mae"]), f3(graph.loc[gsplit, "graph_only_rmse"]),
                     r2(graph.loc[gsplit, "graph_only_r2"]) if s == "holdout" else "--"])
        if s == "cv":
            rows.append("MID")
    write_table("blast_results", "Blast-radius forecasting: the tuned regressor against three baselines without machine "
                "learning, in grouped cross-validation on the training runs and on the holdout.", "tab:blast_results",
                ["Split", "System", "MAE", "RMSE", "$R^2$"], rows, "llccc",
                note="The call-graph baseline has nothing to fit, so its cross-validation row is computed over all "
                     "training runs. $R^2$ is given for the holdout only; the models were chosen on the MAE in "
                     "cross-validation.")
    h, rcv = sev.loc[("holdout", CLF)], sev.loc[("cv", CLF)]
    hr, rr = sev.loc[("holdout", "FullRuleBaseline")], sev.loc[("cv", "FullRuleBaseline")]
    num("mlHoldoutAcc", f3(h.accuracy)); num("ruleHoldoutAcc", f3(hr.accuracy))
    num("mlHoldoutFone", f3(h.macro_f1)); num("ruleHoldoutFone", f3(hr.macro_f1))
    num("mlCvAcc", f3(rcv.accuracy)); num("ruleCvAcc", f3(rr.accuracy))
    num("mlCvFone", f3(rcv.macro_f1)); num("ruleCvFone", f3(rr.macro_f1))
    num("regHoldoutMAE", f3(br.loc[("holdout", REG)].mae)); num("earlyHoldoutMAE", f3(br.loc[("holdout", "EarlyBlastBaseline")].mae))
    num("regCvMAE", f3(br.loc[("cv", REG)].mae)); num("earlyCvMAE", f3(br.loc[("cv", "EarlyBlastBaseline")].mae))
    num("zeroCvMAE", f3(br.loc[("cv", "NoSpreadBaseline")].mae)); num("graphCvMAE", f3(graph.loc["training", "graph_only_mae"]))
    num("zeroHoldoutMAE", f3(br.loc[("holdout", "NoSpreadBaseline")].mae))
    num("graphHoldoutMAE", f3(graph.loc["holdout", "graph_only_mae"]))
    num("earlyHoldoutRtwo", r2(br.loc[("holdout", "EarlyBlastBaseline")].r2))
    num("zeroHoldoutRtwo", r2(br.loc[("holdout", "NoSpreadBaseline")].r2))
    num("graphHoldoutRtwo", r2(graph.loc["holdout", "graph_only_r2"]))
    num("regHoldoutRtwoPct", f"{100 * br.loc[('holdout', REG)].r2:.0f}")

    # untuned models (train_model.py) and the two tuned families (tune_model.py), cross-validation on the training runs
    scr_c = {st: pd.read_csv(RES / st / "classification_report.csv").set_index("model") for st in ("baseline", "fe")}
    scr_r = {st: pd.read_csv(RES / st / "regression_report.csv").set_index("model") for st in ("baseline", "fe")}
    fam = {"RandomForest": ("Random forest", "rf"), "XGBoost": ("XGBoost", "xgb"), "LightGBM": ("LightGBM", "lgbm")}
    rows = []
    for st, feat, key in (("baseline", "Base", "Base"), ("fe", "Engineered", "Fe")):
        for m, (label, ab) in fam.items():
            c_, r_ = scr_c[st].loc[m], scr_r[st].loc[m]
            if st == "fe":
                rows.append([label, f3(r_.mae_mean), f3(r_.rmse_mean)])
            num(f"{ab}{key}Fone", f3(c_.macro_f1)); num(f"{ab}{key}MAE", f3(r_.mae_mean))
    write_table("model_screening", "Untuned blast-radius regressors on the engineered features, in grouped cross-validation "
                "on the training runs.", "tab:model_screening", ["Model", "MAE", "RMSE"], rows, "lcc")
    screening_confusion_tables(scr_c, scr_r, fam)
    num("rfFeMAEsd", f3(scr_r["fe"].loc["RandomForest"].mae_std))
    for m, ab in (("XGBoost", "xgb"), ("RandomForest", "rf")):
        cc = pd.read_csv(RES / f"baseline_clf_{ab}" / "baseline_comparison.csv").set_index(["split", "model"]).loc[("cv", m)]
        rc = pd.read_csv(RES / f"fe_reg_{ab}" / "baseline_comparison.csv").set_index(["split", "model"]).loc[("cv", m)]
        num(f"{ab}TunedCvAcc", f3(cc.accuracy)); num(f"{ab}TunedCvFone", f3(cc.macro_f1))
        num(f"{ab}TunedCvMAE", f3(rc.mae)); num(f"{ab}TunedCvRMSE", f3(rc.rmse))
        num(f"{ab}ClfTuneGain", f"{pd.read_csv(RES / f'baseline_clf_{ab}' / 'tuning_report.csv').improvement_pct.iloc[0]:.1f}")
        num(f"{ab}RegTuneGain", f"{pd.read_csv(RES / f'fe_reg_{ab}' / 'tuning_report.csv').improvement_pct.iloc[0]:.1f}")

    # per snapshot (holdout)
    a = pd.read_csv(RES / "fe_tuned" / "accuracy_vs_time_holdout.csv").set_index(["model", "time_into_fault_s"])
    m = pd.read_csv(RES / "fe_tuned" / "mae_vs_time_holdout.csv").set_index(["model", "time_into_fault_s"])
    rows = [[f"{T} s", f3(a.loc[(CLF, T)].accuracy), f3(a.loc[("FullRuleBaseline", T)].accuracy),
             f3(m.loc[(REG, T)].mae), f3(m.loc[("EarlyBlastBaseline", T)].mae)] for T in (15, 30, 45, 60)]
    write_table("per_snapshot", "Holdout results by time since the change: forecasting helps most at the earliest snapshot.",
                "tab:per_snapshot", ["Snapshot", "Model acc.", "Rule acc.", "Model MAE", "Reactive MAE"], rows, "lcccc")
    num("mlTfifteenAcc", f3(a.loc[(CLF, 15)].accuracy)); num("ruleTfifteenAcc", f3(a.loc[("FullRuleBaseline", 15)].accuracy))
    num("regTfifteenMAE", f3(m.loc[(REG, 15)].mae)); num("earlyTfifteenMAE", f3(m.loc[("EarlyBlastBaseline", 15)].mae))
    num("regTsixtyMAE", f3(m.loc[(REG, 60)].mae)); num("earlyTsixtyMAE", f3(m.loc[("EarlyBlastBaseline", 60)].mae))
    num("regHoldoutRMSE", f3(br.loc[("holdout", REG)].rmse)); num("regHoldoutRtwo", f2(br.loc[("holdout", REG)].r2))

    # holdout differences with 95% bootstrap intervals: resample the 50 runs (each run's 4 rows averaged), 5,000 draws
    sp = pd.read_csv(RES / "fe_tuned" / "holdout_predictions.csv")
    sp = sp.assign(model=(sp.pred_label == sp.true_label).astype(float), rule=(sp.baseline_pred == sp.true_label).astype(float))
    lo, mid, hi = bootstrap_diff(sp.groupby("experiment_id")[["model", "rule"]].mean(), "model", "rule")
    num("sevGain", f"{100 * mid:.1f}"); num("sevGainLo", f"{100 * lo:.1f}"); num("sevGainHi", f"{100 * hi:.1f}")
    rp = pd.read_csv(RES / "fe_tuned" / "holdout_regression_predictions.csv")
    rp = rp.assign(model=(rp.pred_blast_radius - rp.true_blast_radius).abs(),
                   early=(rp.baseline_pred - rp.true_blast_radius).abs(), zero=rp.true_blast_radius.abs())
    runs = rp.groupby("experiment_id")[["model", "early", "zero"]].mean()
    for base, key in (("early", "Early"), ("zero", "Zero")):
        lo, mid, hi = bootstrap_diff(runs, base, "model")       # positive = the model's error is lower
        num(f"blastGain{key}", f3(mid)); num(f"blastGain{key}Lo", f3(lo)); num(f"blastGain{key}Hi", f3(hi))

    g = pd.read_csv(RES / "fe_tuned" / "forecasting_gain.csv").set_index(["split", "subset"])
    label = {"all": "All rows", "changed_after_snapshot": "Severity changes later", "unchanged": "Severity unchanged"}
    rows = [["CV" if s == "cv" else "Holdout", label[sub], int(g.loc[(s, sub)].rows), pct(g.loc[(s, sub)].model_accuracy),
             pct(g.loc[(s, sub)].baseline_accuracy)] for s in ("cv", "holdout") for sub in ("all", "changed_after_snapshot", "unchanged")]
    write_table("forecasting_gain", "Where the forecast adds value: accuracy on all rows, on rows whose severity changes after "
                "the snapshot, and on rows whose severity does not.", "tab:forecasting_gain",
                ["Split", "Rows", "Number of rows", f"Tuned {LABEL[CLF]}", "Reactive alerting"], rows, "llccc")
    num("changedHoldoutModelAcc", pct(g.loc[("holdout", "changed_after_snapshot")].model_accuracy))


# features: group and meaning, checked against preprocess.py and feature_engineer.py
FEATURE_INFO = {
    "fault_type": ("Change", "type of change: CPU limit, memory limit, network delay or replica count"),
    "fault_value_numeric": ("Change", "size of the change: the new limit, the delay or the replica count"),
    "service": ("Change", "the changed service"),
    "fault_severity_ratio": ("Change", "size of the change relative to the service's original setting"),
    "traffic_level": ("Context", "load during the run: normal or high"),
    "time_into_fault_s": ("Context", "seconds since the change: 15, 30, 45 or 60"),
    "latency_observable": ("Context", "whether the changed service exports spans of its own"),
    "target_cpu": ("Resources", "CPU use of the changed service at the snapshot"),
    "target_cpu_delta": ("Resources", "its relative change from the baseline"),
    "target_memory_bytes": ("Resources", "memory use of the changed service at the snapshot"),
    "target_memory_delta": ("Resources", "its relative change from the baseline"),
    "target_replicas": ("Resources", "running replicas of the changed service"),
    "target_restart_count": ("Resources", "container restarts of the changed service"),
    "system_cpu_mean": ("Resources", "mean CPU use over all services"),
    "system_memory_mean": ("Resources", "mean memory use over all services"),
    "target_request_rate": ("Requests", "requests per second to the changed service since the change"),
    "target_error_rate": ("Requests", "share of those requests that failed"),
    "target_latency_mean_ms": ("Requests", "their mean latency, as the callers saw it"),
    "target_latency_delta": ("Requests", "relative change in that latency from the baseline"),
    "blast_radius_early": ("Requests", "other services already significantly slower or failing"),
    "max_trace_depth": ("Requests", "depth of the deepest trace through the changed service since the change"),
    "system_error_rate": ("Requests", "mean error rate over all services since the change"),
    "system_latency_ratio": ("Requests", "mean latency over all services relative to the baseline"),
    "service_depth": ("Call graph", "depth of the changed service in the call graph"),
    "service_downstream_count": ("Call graph", "number of services it calls"),
    "service_upstream_count": ("Call graph", "number of services that call it"),
    "service_is_gateway": ("Call graph", "whether it is the frontend, the entry point"),
}


def features_table():
    import train_model as T
    base = set(pd.read_csv(PROC / "training_dataset.csv", nrows=1).columns)
    fe = set(pd.read_csv(PROC / "training_dataset_fe.csv", nrows=1).columns)
    used_clf = [c for c in T.BASE_NUMERIC_FEATURES if c in base] + T.CATEGORICAL_FEATURES
    used_reg = [c for c in T.BASE_NUMERIC_FEATURES + T.FE_NUMERIC_FEATURES if c in fe] + T.CATEGORICAL_FEATURES
    missing = [c for c in used_reg if c not in FEATURE_INFO]
    if missing:
        raise SystemExit(f"features without a description in make_tables.FEATURE_INFO: {missing}")
    order = ["Change", "Context", "Resources", "Requests", "Call graph"]
    feats = sorted(used_reg, key=lambda c: (order.index(FEATURE_INFO[c][0]), list(FEATURE_INFO).index(c)))
    rows, last = [], None
    for c in feats:
        group, text = FEATURE_INFO[c]
        if last and group != last:
            rows.append("MID")
        rows.append([group if group != last else "", f"\\texttt{{{c.replace('_', chr(92) + '_')}}}", text,
                     "both models" if c in used_clf else "regression only"])
        last = group
    num("nClfFeatures", len(used_clf)); num("nRegFeatures", len(used_reg))
    write_table("features", f"Features used by the models: {len(used_clf)} for the severity classifier and {len(used_reg)} for the "
                "blast-radius regressor", "tab:features", ["Group", "Feature", "Meaning", "Used by"], rows,
                "llp{0.45\\linewidth}l", size=r"\footnotesize", colsep="4pt")


# hyperparameter search of the two final models, read from tune_model.py (search space) and best_params.json
PARAM_NAMES = {"n_estimators": "number of trees", "max_depth": "maximum tree depth", "learning_rate": "learning rate",
               "subsample": "share of rows per tree", "colsample_bytree": "share of features per tree",
               "min_child_weight": "minimum child weight", "min_samples_leaf": "minimum samples per leaf",
               "min_samples_split": "minimum samples to split", "max_features": "features per split"}


def tuning_table():
    import ast
    src = (ROOT / "scripts" / "tune_model.py").read_text()
    tree = ast.parse(src)
    consts = {t.id: ast.literal_eval(n.value) for n in tree.body if isinstance(n, ast.Assign)
              for t in n.targets if isinstance(t, ast.Name) and t.id in ("N_TRIALS", "N_FOLDS")}
    seed = re.search(r"TPESampler\(seed=(\d+)\)", src).group(1)

    def space(fn: str) -> dict:
        f = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == fn)
        out = {}
        for c in ast.walk(f):
            if isinstance(c, ast.Call) and isinstance(c.func, ast.Attribute) and c.func.attr.startswith("suggest_"):
                args = [ast.literal_eval(a) for a in c.args]
                kw = {k.arg: ast.literal_eval(k.value) for k in c.keywords}
                if c.func.attr == "suggest_categorical":
                    out[args[0]] = ", ".join(str(v) for v in args[1])
                else:
                    s = f"{args[1]:g} to {args[2]:g}"
                    s += f", step {kw['step']:g}" if kw.get("step") else ""
                    s += ", log scale" if kw.get("log") else ""
                    out[args[0]] = s
        return out

    clf_space, reg_space = space("_xgb_clf_objective"), space("_xgb_reg_objective")
    if clf_space != reg_space:
        raise SystemExit("the two XGBoost search spaces differ: tuning_table needs a column per space")
    clf = json.loads((RES / "baseline_clf_xgb" / "best_params.json").read_text())["xgb_clf"]
    reg = json.loads((RES / "fe_reg_xgb" / "best_params.json").read_text())["xgb_reg"]
    fmt = lambda v: f"{v:.3g}" if isinstance(v, float) else str(v)     # noqa: E731
    rows = [[f"\\texttt{{{k.replace('_', chr(92) + '_')}}}", PARAM_NAMES.get(k, k), clf_space[k], fmt(clf[k]), fmt(reg[k])]
            for k in clf_space]
    num("nTrials", consts["N_TRIALS"]); num("nFolds", consts["N_FOLDS"])
    num("clfTrees", clf["n_estimators"]); num("clfDepth", clf["max_depth"])
    num("regTrees", reg["n_estimators"]); num("regDepth", reg["max_depth"])
    write_table("tuning", "Hyperparameter search for the two XGBoost models: the search space and the values chosen.",
                "tab:tuning", ["Hyperparameter", "Meaning", "Search space", "Severity", "Blast radius"], rows,
                "llp{0.27\\linewidth}cc", size=r"\footnotesize",
                note=f"Optuna, {consts['N_TRIALS']} trials per model with the TPE sampler (seed {seed}); each trial is "
                     f"scored by the mean over the grouped {consts['N_FOLDS']}-fold cross-validation, macro-F1 for "
                     "severity and MAE for the blast radius. Class weights are fixed, not tuned.")


# label reliability

def reliability_tables():
    b = pd.read_csv(RES / "reliability" / "reliability_full" / "bootstrap_per_run.csv")
    b["stable"] = b.sev_stability >= 0.9
    rows = [[s, int((b.severity == s).sum()), pct(b[b.severity == s].stable.mean())] for s in SEV]
    rows += ["MID", ["All", len(b), pct(b.stable.mean())]]
    write_table("label_stability", "Severity-label stability: share of runs whose label is unchanged in at least 90\\% of "
                "200 bootstrap resamples of the run's own traces.", "tab:label_stability",
                ["Final severity", "Runs", "Stable"], rows, "lcc")
    num("stableAll", pct(b.stable.mean())); num("stableMedium", pct(b[b.severity == "Medium"].stable.mean()))
    num("stableHigh", pct(b[b.severity == "High"].stable.mean()))
    # the three blast-radius definitions: no change vs fault (30 s windows), and agreement of the repeated settings
    v = pd.read_csv(RES / "reliability" / "blast_v2" / "per_run.csv").set_index("run")
    ds = pd.read_csv(PROC / "training_dataset.csv").drop_duplicates("experiment_id")
    ds["setting"] = ds.service + "|" + ds.fault_type + "|" + ds.fault_value.astype(str) + "|" + ds.traffic_level
    pairs = [tuple(sorted(g.experiment_id)) for _, g in ds.groupby("setting") if len(g) == 2]
    rows = []
    for label, k in (("Trace roots, plain averages, rise $>$10\\% (first)", "D0"),
                     ("Request spans, plain averages, rise $>$10\\%", "D1")):
        same = sum(v.loc[a, f"{k}_final60"] == v.loc[b, f"{k}_final60"] for a, b in pairs)
        rows.append([label, f"{v[f'{k}_null'].mean():.2f}", f"{v[f'{k}_signal'].mean():.2f}", f"{same} of {len(pairs)}"])
    # the final measurement itself (scripts/analysis/null_check_final.py), not blast_v2's candidate rule D2
    nf = pd.read_csv(RES / "reliability" / "null_final" / "per_run.csv").set_index("run")
    same = sum(nf.loc[a, "final60"] == nf.loc[b, "final60"] for a, b in pairs)
    rows.append(["Request spans with statistical tests (final)", f"{nf.null.mean():.2f}", f"{nf.signal.mean():.2f}",
                 f"{same} of {len(pairs)}"])
    num("nullFinalFlagged", int(nf.null.sum())); num("nullFinalMean", f"{nf.null.mean():.2f}")
    num("nullFinalSignal", f"{nf.signal.mean():.2f}")
    write_table("validation", "Checking the blast-radius measurement: services counted as affected per run with no change "
                "and during the change, and how many of the 20 settings run twice gave the same blast radius in both runs",
                "tab:validation",
                ["Measurement", "No change", "Change", "Repeats agreeing"], rows, "lccc")

    # structural exposure: who calls each service, and how much of that set was actually affected
    se = RES / "reliability" / "structural_exposure"
    ex = pd.read_csv(se / "exposure_per_service.csv")
    labels = pd.read_csv(PROC / "label_measurements.csv")
    short = lambda x: x.replace("service", "")
    names = lambda x: [] if pd.isna(x) or x == "" else str(x).split(";")
    total = inside = 0
    rows = []
    for r in ex.itertuples():
        exposed = {s.strip() for s in r.exposed_services.split(",")} if r.exposure else set()
        cells = []
        for sev in reversed(SEV):      # High, Medium, Low
            g = labels[(labels.service == r.service) & (labels.severity == sev)]
            if g.empty:
                cells.append("no runs")
                continue
            counts = pd.Series([s for x in g.blast_affected for s in names(x)], dtype=object).value_counts()
            total += int(counts.sum())
            inside += int(counts[counts.index.isin(exposed)].sum())
            items = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
            found = ", ".join((short(k) if k in exposed else rf"\emph{{{short(k)}}}") + f" {v}" for k, v in items)
            cells.append(f"{len(g)} run{'s' if len(g) != 1 else ''}: {found or 'none'}")
        rows.append([short(r.service), ", ".join(sorted(short(s) for s in exposed)) or "none", *cells])
    num("nAffectedServices", total); num("nAffectedExposed", inside)
    write_table("exposure", "Services exposed to a change in each service, and the services actually affected in the runs. "
                "High, Medium and Low are the final severity of the changed service: runs are grouped by it, and each number "
                "is how many of those runs had that other service measured as affected.", "tab:exposure",
                ["Changed service", "Exposed (services that call it)", "High", "Medium", "Low"], rows,
                "lp{0.19\\linewidth}p{0.27\\linewidth}p{0.13\\linewidth}p{0.12\\linewidth}", size=r"\footnotesize", colsep="4pt",
                note=f"Services in italics do not call the changed service. This happened {total - inside} times out of {total}.")


# GenAI

def genai_tables():
    # the retrieval table (training and holdout side by side) is written by retrieval_table()

    dev_rows = []
    # only the round<N> folders are reported
    for d in sorted((RES / "genai" / "dev").glob("round[0-9]*/results.csv")):
        if not d.parent.name.removeprefix("round").isdigit():
            continue
        df = pd.read_csv(d)
        ok = df[df.error.isna()] if "error" in df else df
        dev_rows.append([d.parent.name.replace("round", "Round "), pct(ok.severity_correct.mean()),
                         pct(ok.cause_correct.dropna().astype(bool).mean()), int(ok.false_alert.sum()), int(ok.missed_alert.sum()),
                         f"{ok.cost_usd.sum():.2f}"])
        n_dev = len(ok)
    if dev_rows:
        write_table("dev_rounds", f"Development rounds (LLM only, Claude Opus 5.5, {n_dev} development incidents from the "
                    "training runs).", "tab:dev_rounds",
                    ["Round", "Severity acc.", "Cause acc.", "False alerts", "Missed", "Cost (USD)"], dev_rows, "lccccc")

    # holdout evaluation (results/genai/holdout_summary, written by genai/analyze_holdout.py)
    hs = RES / "genai" / "holdout_summary"
    if not (hs / "main.csv").exists():
        print("results/genai/holdout_summary is missing: LLM holdout tables skipped")
        return
    short = {"claude-opus-5-5": "Opus", "gpt-6-sol": "gpt-6-sol", "claude-haiku-5-5": "Haiku"}
    sysname = {"llm_only": "LLM only", "ml_llm": "ML + LLM", "gated": "Gated"}
    m = pd.read_csv(hs / "main.csv")
    ml, rule = m[m.system == "ml_only"].iloc[0], m[m.system == "rule_at_30s"].iloc[0]
    g2 = lambda x: "--" if pd.isna(x) else f"{x:.2f}"      # noqa: E731
    ci = lambda r: f"{r.ci95_low:.2f}--{r.ci95_high:.2f}"   # noqa: E731
    rows = [["Rule at 30\\,s", "--", f3(rule.severity_acc), ci(rule), "--", "--", "--",
             f"{rule.false_alerts_per_run:.1f}", f"{rule.missed_alerts_per_run:.1f}", "--", "0"],
            ["ML only", "--", f3(ml.severity_acc), ci(ml), "--", "--", "--", f"{ml.false_alerts_per_run:.1f}",
             f"{ml.missed_alerts_per_run:.1f}", g2(ml.action_ok), "0"], "MID"]
    for s in ("llm_only", "ml_llm", "gated"):
        for mod in short:
            r = m[(m.system == s) & (m.model == mod)].iloc[0]
            rows.append([sysname[s] if mod == "claude-opus-5-5" else "", short[mod], f3(r.severity_acc), ci(r),
                         g2(r.run_to_run_sd), g2(r["pass^k"]), g2(r.cause_acc_MH), f"{r.false_alerts_per_run:.1f}",
                         f"{r.missed_alerts_per_run:.1f}", g2(r.action_ok), f"{r.cost_usd_per_incident_run:.3f}"])
        if s != "gated":
            rows.append("MID")
    write_table("llm_main", "Holdout results per system and model on the 50 incidents, means over three repeats.",
                "tab:llm_main",
                ["System", "Model", "Acc.", "95\\% CI", "SD", "pass\\textasciicircum{}3", "Cause", "False", "Missed",
                 "Action", "Cost"], rows, "llccccccccc", size=r"\footnotesize", colsep="3.5pt",
                note="Acc.: severity accuracy, with its 95\\% bootstrap interval over incidents. SD: standard deviation "
                     "between the three repeats. pass\\textasciicircum{}3: share of incidents right in all three repeats. "
                     "Cause: root-cause accuracy on the Medium and High incidents. False, Missed: alerts per run. Action: "
                     "share of acceptable actions. Cost: mean cost per incident in USD. In the gated system, incidents the "
                     "ML model does not flag are not investigated; they count as Low with no cause.")

    # paired differences: the comparisons of analyze_holdout.py plus the three models within the gated system
    from genai import analyze_holdout as H
    _, by_key = H.main_table(H.load(RES / "genai"))
    p = pd.read_csv(hs / "paired.csv")
    def name(cmp: str) -> str:
        head, rest = cmp.split(": ", 1)
        a, b = rest.split(" vs ")
        if head in short:
            return f"{short[head]}: {sysname.get(a, 'ML only')} vs {sysname.get(b, 'ML only')}"
        return f"{sysname[head]}: {short[a]} vs {short[b]}"
    pm = lambda x: f"{x:+.1f}".replace("-", "$-$")          # noqa: E731  (a minus sign, not a hyphen)
    rows = [[name(r.comparison), pm(100 * r.difference), f"{pm(100 * r.ci95_low)} to {pm(100 * r.ci95_high)}",
             f"{r.a_better_on} / {r.b_better_on}"] for r in p.itertuples()]
    rows.append("MID")
    for a, b in (("gpt-6-sol", "claude-opus-5-5"), ("gpt-6-sol", "claude-haiku-5-5"), ("claude-opus-5-5", "claude-haiku-5-5")):
        r = H.paired(by_key[(a, "gated")], by_key[(b, "gated")])
        rows.append([f"Gated: {short[a]} vs {short[b]}", pm(100 * r['difference']),
                     f"{pm(100 * r['ci95_low'])} to {pm(100 * r['ci95_high'])}", f"{r['a_better_on']} / {r['b_better_on']}"])
        key = f"gated{short[a].split('-')[0].capitalize()}{short[b].split('-')[0].capitalize()}"
        num(key, f"{100 * r['difference']:+.1f}"); num(key + "Lo", f"{100 * r['ci95_low']:+.1f}")
        num(key + "Hi", f"{100 * r['ci95_high']:+.1f}")
    write_table("llm_paired", "Paired differences in severity accuracy on the same 50 incidents (percentage points, the "
                "first system minus the second), with 95\\% bootstrap intervals over incidents and the number of incidents on "
                "which each side was better.", "tab:llm_paired",
                ["Comparison", "Difference", "95\\% CI", "Better on"], rows, "lccc", size=r"\footnotesize")

    s = pd.read_csv(hs / "sensitivity.csv")
    rows = []
    for mod in short:
        r = s[s.model == mod].iloc[0]
        lo = m[(m.system == "llm_only") & (m.model == mod)].iloc[0]
        rows.append([short[mod], f3(r["frozen_prompt_acc (mean of repeats)"]), f3(r["alternative_prompt_acc (1 run)"]),
                     pm(100 * r.difference), f"{pm(100 * r.ci95_low)} to {pm(100 * r.ci95_high)}",
                     f"{lo.false_alerts_per_run:.1f} / {r.false_alerts_alt:.0f}",
                     f"{lo.missed_alerts_per_run:.1f} / {r.missed_alerts_alt:.0f}"])
    write_table("llm_sensitivity", "Prompt sensitivity, LLM only on the 50 holdout incidents: the frozen prompt (version 1, "
                "mean of three repeats) against the unrefined version 0 (one run).", "tab:llm_sensitivity",
                ["Model", "v1", "v0", "Difference", "95\\% CI", "False v1 / v0", "Missed v1 / v0"], rows, "lcccccc")

    # the three models: provider, effort and list price (genai/agent.py, used for every cost figure)
    from genai.agent import PRICES
    full = {"claude-opus-5-5": "Claude Opus 5.5", "gpt-6-sol": "gpt-6-sol", "claude-haiku-5-5": "Claude Haiku 5.5"}
    rows = [[full[k], f"\\texttt{{{k}}}", "OpenAI" if k.startswith("gpt") else "Anthropic", "medium",
             f"{PRICES[k][0]:.2f} / {PRICES[k][1]:.2f}"] for k in full]
    write_table("llm_models", "The three models evaluated. All use the same prompt, tools and reasoning effort. "
                "Prices are in USD per million tokens.",
                "tab:llm_models", ["Model", "API name", "Provider", "Reasoning effort", "Input / Output price"], rows, "lllcc")


# experimental environment: deployed versions and tracing coverage

# language of each Online Boutique service (upstream README); call-graph order, traced services first
SERVICE_LANG = {"frontend": "Go", "checkoutservice": "Go", "productcatalogservice": "Go",
                "currencyservice": "Node.js", "paymentservice": "Node.js", "emailservice": "Python",
                "recommendationservice": "Python", "adservice": "Java", "cartservice": "C\\#",
                "shippingservice": "Go"}
MACHINE_TYPE = "e2-standard-2"   # final dataset hardware, see dataset/superseded/README.md


def _grab(path: Path, pattern: str) -> str:
    import re
    m = re.search(pattern, path.read_text(), re.M)
    if not m:
        raise ValueError(f"version pattern {pattern!r} not found in {path}")
    return m.group(1)


def environment_tables():
    import numpy as np
    import yaml

    # versions, read from the repository's install scripts and Helm chart (not typed here)
    chart = ROOT / "deployment" / "online-boutique" / "helm-chart"
    ob = _grab(chart / "Chart.yaml", r'^appVersion:\s*"?([^"\s]+)"?')
    overrides = yaml.safe_load((ROOT / "deployment" / "online-boutique" / "values.yaml").read_text()) or {}
    if (overrides.get("images") or {}).get("tag"):
        raise ValueError("values.yaml overrides the image tag; the version table would be wrong")
    otel = _grab(chart / "templates" / "opentelemetry-collector.yaml", r"opentelemetry-collector-contrib:([0-9.]+)")
    jaeger_sh = ROOT / "monitoring" / "jaeger" / "install.sh"
    jaeger_app, jaeger_chart = _grab(jaeger_sh, r"Jaeger (v[0-9.]+)"), _grab(jaeger_sh, r'^VERSION="([^"]+)"')
    kps = _grab(ROOT / "monitoring" / "kube-prometheus-stack" / "install.sh", r'^VERSION="([^"]+)"')
    chaos = _grab(ROOT / "deployment" / "chaos-mesh" / "install.sh", r'^VERSION="([^"]+)"')
    cfg = yaml.safe_load((ROOT / "config" / "project.yaml").read_text())
    num("obVersion", ob)
    rows = [["Online Boutique (application under test)", ob, "all services and the load generator; images use the chart version"],
            ["OpenTelemetry Collector (contrib)", otel, "deployed by the Online Boutique chart; forwards spans to Jaeger"],
            ["Jaeger (all-in-one)", jaeger_app, f"distributed tracing; Helm chart {jaeger_chart}"],
            ["kube-prometheus-stack", kps, "Prometheus metrics (Helm chart version)"],
            ["Chaos Mesh", chaos, "network-latency injection"],
            ["Google Kubernetes Engine", "Standard", f"{cfg['node_count']} $\\times$ {MACHINE_TYPE} nodes, {cfg['zone']}"]]
    write_table("software_versions", "Software deployed for the experiments and their versions.", "tab:software",
                ["Component", "Version", "Role"], rows, "llp{0.42\\linewidth}", colsep="4.5pt",
                note="Versions are read from the repository's install scripts and Helm chart by "
                     "\\texttt{scripts/report/make\\_tables.py}.")

    # tracing coverage, counted from the per-run span data (dataset/spans/, built from the raw data and not in git)
    files = sorted((ROOT / "dataset" / "spans").glob("EXP_*.npz"))
    if not files:
        # keep the table and its numbers from the last run that had the span data
        old = (OUT_T / "numbers.tex").read_text() if (OUT_T / "numbers.tex").exists() else ""
        for name in ("nTracedServices", "nUntracedServices", "nSpanRuns"):
            m = re.search(rf"\\newcommand{{\\{name}}}{{([^}}]*)}}", old)
            if m:
                num(name, m.group(1))
        print("dataset/spans/ is missing (./scripts/run_genai_eval.sh prepare): tracing_coverage.tex left as it is")
        return
    ds_runs = set(pd.read_csv(PROC / "training_dataset.csv", usecols=["experiment_id"]).experiment_id)
    files = [f for f in files if f.stem in ds_runs]
    own_runs = dict.fromkeys(SERVICE_LANG, 0)
    caller_spans = dict.fromkeys(SERVICE_LANG, 0)
    missing: dict[str, list[str]] = {s: [] for s in SERVICE_LANG}
    for f in files:
        with np.load(f) as z:
            for s in SERVICE_LANG:
                has = len(z[f"{s}__own"]) > 0
                own_runs[s] += has
                caller_spans[s] += len(z[f"{s}__callers"])
                if not has:
                    missing[s].append(f.stem)
    traced = [s for s in SERVICE_LANG if own_runs[s] > 0]
    untraced = [s for s in SERVICE_LANG if own_runs[s] == 0]
    num("nTracedServices", len(traced)); num("nUntracedServices", len(untraced)); num("nSpanRuns", len(files))
    rows = []
    for group in (traced, untraced):
        for s in group:
            rows.append([f"\\texttt{{{s}}}", SERVICE_LANG[s], "yes" if own_runs[s] else "no",
                         f"{own_runs[s]} / {len(files)}", f"{caller_spans[s]:,}" if caller_spans[s] else "--"])
        if group is traced:
            rows.append("MID")
    partial = sorted({r for s in traced for r in missing[s]})
    gaps = ""
    if partial:
        where = {s: sum(r in missing[s] for r in partial) for s in traced if missing[s]}
        gaps = (f" Run{'s' if len(partial) > 1 else ''} {', '.join(p.replace('_', chr(92) + '_') for p in partial)}: "
                f"{' and '.join(f'{chr(92)}texttt{{{s}}}' for s in where)} recorded only health checks, so "
                f"{'they have' if len(where) > 1 else 'it has'} no request spans there.")
    write_table("tracing_coverage",
                f"Tracing coverage of Online Boutique {ob}: {len(traced)} of {len(SERVICE_LANG)} services export "
                f"spans; the other {len(untraced)} are seen only through their callers.",
                "tab:tracing_coverage",
                ["Service", "Language", "Exports spans", "Runs with request spans", "Calls recorded by callers"],
                rows, "llccr",
                note="Counted over the final dataset's runs; health-check and telemetry-export spans are excluded. "
                     "\\texttt{adservice} and \\texttt{cartservice} have no tracing settings in the deployed Helm chart; "
                     f"in \\texttt{{shippingservice}} {ob} tracing is an unimplemented placeholder "
                     "(\\texttt{initTracing()} is empty)~\\cite{onlineboutique,obissue422}. "
                     "\\texttt{frontend} is called only by the load generator, which is not traced." + gaps)


OBSOLETE = ["severity_vs_baseline", "blast_vs_baseline", "blast_null", "collection_phases", "retrieval_train"]   # older names of tables written here


def correlation_table():
    """Spearman correlation of each numeric model feature with severity and blast radius, training rows only."""
    import train_model as T
    d = pd.read_csv(PROC / "training_dataset_fe.csv")
    ho = set((PROC / "holdout_ids.txt").read_text().split())
    d = d[~d.experiment_id.isin(ho)].copy()
    d["severity_encoded"] = d.severity.map({"Low": 0, "Medium": 1, "High": 2})
    base = set(pd.read_csv(PROC / "training_dataset.csv", nrows=1).columns)
    order = ["Change", "Context", "Resources", "Requests", "Call graph"]
    feats = sorted(T.BASE_NUMERIC_FEATURES + T.FE_NUMERIC_FEATURES,
                   key=lambda c: (order.index(FEATURE_INFO[c][0]), list(FEATURE_INFO).index(c)))
    rows, last = [], None
    for c in feats:
        group = FEATURE_INFO[c][0]
        if last and group != last:
            rows.append("MID")
        x = d[c].astype(float)
        sev = x.corr(d.severity_encoded, method="spearman")
        br = x.corr(d.blast_radius_final, method="spearman")
        fmt = lambda v: "--" if pd.isna(v) else f"{v:.2f}".replace("-", "$-$")
        rows.append([group if group != last else "", f"\\texttt{{{c.replace('_', chr(92) + '_')}}}", fmt(sev), fmt(br),
                     "both models" if c in base else "regression only"])
        last = group
    n_runs = d.experiment_id.nunique()
    write_table("correlations", f"Spearman correlation of each numeric feature with the final severity (Low, Medium, High "
                f"as 0, 1, 2) and with the blast radius, over the rows of the {n_runs} training runs. The categorical "
                "features (type of change, service and traffic level) have no single correlation and are left out.",
                "tab:correlations", ["Group", "Feature", "Severity", "Blast radius", "Used by"], rows, "llccl",
                size=r"\footnotesize")


def screening_confusion_tables(scr_c, scr_r, fam):
    """Untuned classifiers on the base features: scores and the errors from their confusion matrices."""
    c = pd.read_csv(RES / "reliability" / "screening_confusion" / "confusion.csv")
    rows = []
    for m, (label, _) in fam.items():
        g = c[(c.features == "Base") & (c.model == m)].set_index("true")
        s = scr_c["baseline"].loc[m]
        missed = int(g.loc["Medium", "pred_Low"] + g.loc["High", "pred_Low"])
        alarms = int(g.loc["Low", "pred_Medium"] + g.loc["Low", "pred_High"])
        rows.append([label, f3(s.accuracy), f3(s.macro_f1), f3(s.Medium_f1), int(g.loc["Medium", "pred_Medium"]), missed, alarms])
    n_med = int(c[(c.features == "Base") & (c.model == "XGBoost") & (c.true == "Medium")].filter(like="pred_").sum(axis=1).iloc[0])
    write_table("screening_errors", "Untuned severity classifiers on the base features, in grouped cross-validation on the "
                f"training runs, with balanced class weights. Correct Medium: Medium rows forecast as Medium (of {n_med}). "
                "Missed: Medium or High rows forecast as Low. False alarms: Low rows forecast as Medium or High.",
                "tab:screening_errors", ["Model", "Accuracy", "Macro-F1", "Medium F1", "Correct Medium", "Missed", "False alarms"],
                rows, "lcccccc", size=r"\footnotesize", colsep="4.5pt")


def affected_table():
    """Affected services named by the agents against the measured blast radius (scripts/analysis/score_affected.py)."""
    a = RES / "genai" / "affected"
    if not (a / "summary.csv").exists():
        print("results/genai/affected is missing: affected-services table skipped")
        return
    s = pd.read_csv(a / "summary.csv").set_index(["system", "model"])
    short = {"claude-opus-5-5": "Opus", "gpt-6-sol": "gpt-6-sol", "claude-haiku-5-5": "Haiku"}
    sysname = {"llm_only": "LLM only", "ml_llm": "ML + LLM", "gated": "Gated"}
    p0 = lambda x: f"{100 * x:.0f}\\%"      # noqa: E731
    ml = s.ml_mae_30s.iloc[0]                 # the regressor at 30 s, negative forecasts counted as 0
    rows = [["ML only", "regressor", f"{ml:.2f}", "--", "--", "--"], "MID"]
    for sy in sysname:
        for mo in short:
            r = s.loc[(sy, mo)]
            rows.append([sysname[sy] if mo == "claude-opus-5-5" else "", short[mo], f"{r.count_mae:.2f}", p0(r.exact),
                         p0(r.named_affected), p0(r.affected_named)])
        if sy != "gated":
            rows.append("MID")
    write_table("llm_affected", "The services the agents named as affected, compared with the measured blast radius on the "
                "50 holdout incidents (averages over three repeats).", "tab:llm_affected",
                ["System", "Model", "Count error", "Exact list", "Named, affected", "Affected, named"], rows, "llcccc",
                note="Count error: mean absolute difference between the number of services named and the measured blast "
                     "radius. Named, affected: share of the named services that were affected. Affected, named: share of "
                     "the affected services that were named. The ML regressor gives only a count, at 30 seconds, with "
                     "negative forecasts counted as 0, as shown to the agents.")
    num("affMlMAE", f"{ml:.2f}")
    mlllm = s.xs("ml_llm").count_mae
    num("affMlLlmLo", f"{mlllm.min():.2f}"); num("affMlLlmHi", f"{mlllm.max():.2f}")
    num("affLlmGptMAE", f"{s.loc[('llm_only', 'gpt-6-sol')].count_mae:.2f}")
    g = s.loc[("gated", "gpt-6-sol")]
    num("affGatedGptNamedAffected", f"{100 * g.named_affected:.0f}"); num("affGatedGptAffectedNamed", f"{100 * g.affected_named:.0f}")
    one = pd.read_csv(a / "llm_only_by_true_blast_radius.csv").set_index("true_n")
    num("affOneExact", f"{100 * one.loc[1, 'exact']:.0f}")


def retrieval_table():
    """Retrieval methods at 30 s on the training runs (scripts/analysis/retrieval_train_check.py, each run's own setting
    hidden) and on the holdout incidents (genai/eval_retrieval.py), side by side."""
    s = pd.read_csv(RES / "genai" / "retrieval_train" / "summary.csv").set_index(["T", "mode"])
    h = pd.read_csv(RES / "genai" / "retrieval_eval.csv")
    h = h[(h["T"] == 30) & (~h.service_filter)].set_index("mode")
    names = (("hybrid", "Hybrid (RRF)"), ("keyword", "Keyword (TF-IDF)"), ("vector", "Vector (Chroma, MiniLM)"),
             ("random", "Random"))
    t2 = lambda v: f"{v:.2f}"
    rows = [[label, "--" if m == "random" else t2(s.loc[(30, m), "hit@1"]),
             "--" if m == "random" else t2(s.loc[(30, m), "precision@5"]),
             t2(h.loc[m, "hit@1"]), t2(h.loc[m, "precision@5"])] for m, label in names]
    n = pd.read_csv(RES / "genai" / "retrieval_train" / "per_query.csv").run.nunique()
    write_table("retrieval", f"Retrieval of past incidents with the same cause at 30 seconds, on the {n} training runs (each "
                "with its own setting hidden from the search) and on the 50 holdout incidents.", "tab:retrieval",
                ["Search", "Top result", "Top five", "Top result", "Top five"], rows, "lcccc",
                groups=[("", 1), ("Training runs", 2), ("Holdout", 2)],
                note="Top result: share of queries whose first result has the same cause. Top five: share of the first "
                     "five results with the same cause.")
    num("rhHybridHit", t2(h.loc["hybrid", "hit@1"])); num("rhKeywordHit", t2(h.loc["keyword", "hit@1"]))
    num("rhRandomHit", t2(h.loc["random", "hit@1"]))
    p = pd.read_csv(RES / "genai" / "retrieval_train" / "paired.csv").set_index(["T", "a", "b"])
    num("rtHybridHit", f"{s.loc[(30, 'hybrid'), 'hit@1']:.2f}"); num("rtKeywordHit", f"{s.loc[(30, 'keyword'), 'hit@1']:.2f}")
    num("rtVectorHit", f"{s.loc[(30, 'vector'), 'hit@1']:.2f}")
    r = p.loc[(30, "hybrid", "keyword")]
    num("rtHybridKeywordLo", f"{r.lo:+.2f}".replace("-", "$-$")); num("rtHybridKeywordHi", f"{r.hi:+.2f}")


def user_impact_table():
    """Severity of the changed service against the effect on users (scripts/analysis/user_impact.py)."""
    u = pd.read_csv(RES / "reliability" / "user_impact" / "user_impact.csv")
    ct = pd.crosstab(u.severity, u.user_impact).reindex(index=SEV, columns=SEV, fill_value=0)
    rows = [[s, *[int(ct.loc[s, c]) for c in SEV], int(ct.loc[s].sum())] for s in SEV]
    rows += ["MID", ["Total", *[int(ct[c].sum()) for c in SEV], int(ct.values.sum())]]
    write_table("user_impact", "Final severity of the changed service (rows) against the effect on users (columns), measured "
                "by applying the same severity rule to the frontend's requests, over all runs.", "tab:user_impact",
                ["Changed service", "Users: Low", "Users: Medium", "Users: High", "Runs"], rows, "lcccc",
                groups=None)
    num("nUserDiffer", int(ct.values.sum() - np.trace(ct.values)))
    num("nHighNoUserEffect", int(ct.loc["High", "Low"]))


def binary_table():
    """The severity forecast as two yes-or-no decisions on the holdout rows, and the confusion counts quoted in RQ3."""
    from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
    h = pd.read_csv(RES / "fe_tuned" / "holdout_predictions.csv")
    clf = h.model.iloc[0]
    rows = []
    for question, positive in (("Will it become High?", ["High"]), ("Is an alert needed (Medium or High)?", ["Medium", "High"])):
        if rows:
            rows.append("MID")
        t = h.true_label.isin(positive)
        for i, (label, col) in enumerate(((f"Tuned {clf}", "pred_label"), ("Reactive alerting", "baseline_pred"))):
            p = h[col].isin(positive)
            rows.append([question if i == 0 else "", label, f"{precision_score(t, p):.2f}", f"{recall_score(t, p):.2f}",
                         f"{f1_score(t, p):.2f}"])
    write_table("binary", "The severity forecast as a yes-or-no decision, on the holdout rows.", "tab:binary",
                ["Question", "System", "Precision", "Recall", "F1"], rows, "llccc")
    for name, col in (("ml", "pred_label"), ("rule", "baseline_pred")):
        cm = confusion_matrix(h.true_label, h[col], labels=SEV)
        num(f"{name}Mistakes", int(cm.sum() - np.trace(cm)))
    cm = confusion_matrix(h.true_label, h.baseline_pred, labels=SEV)
    num("ruleLowAsHigh", int(cm[SEV.index("Low"), SEV.index("High")]))


def repeat_tables():
    """How often the two runs of a repeated setting gave the same severity, for settings where either run was Medium."""
    d = pd.read_csv(PROC / "training_dataset.csv").drop_duplicates("experiment_id")
    d["setting"] = d.service + "|" + d.fault_type + "|" + d.fault_value.astype(str) + "|" + d.traffic_level
    pairs = d.groupby("setting").filter(lambda g: len(g) == 2).groupby("setting").severity.apply(list)
    medium = [p for p in pairs if "Medium" in p]
    num("nMediumRepeat", len(medium)); num("nMediumRepeatBoth", sum(p.count("Medium") == 2 for p in medium))

    # why the two runs disagree (scripts/analysis/repeat_disagreements.py)
    s = pd.read_csv(RES / "reliability" / "repeat_disagreements" / "summary.csv", index_col=0)["value"]
    if s["settings"] != len(pairs):
        raise SystemExit("repeat_disagreements results do not match the dataset: re-run the script")
    num("nRepeatSevSame", s["same_severity"]); num("nRepeatSevDiffer", s["severity_differs"])
    num("nRepeatCauseRestart", s["cause_restart"]); num("nRepeatCauseFew", s["cause_few"])
    num("nRepeatCauseBusy", s["cause_busy"])
    num("nRepeatFewMin", s["few_min_requests"]); num("nRepeatFewMax", s["few_max_requests"])

    # training with or without the second run of each repeated setting (scripts/analysis/repeat_robustness.py)
    rob = ROOT / "results" / "reliability" / "robustness"
    b = pd.read_csv(rob / "B_holdout_cross.csv").query("scored_on == 'all 50 runs'").set_index("trained_on")
    c = pd.read_csv(rob / "C_repeated_cv_summary.csv").query("scored_on == 'single-run settings'").set_index("version")
    holdout = set((PROC / "holdout_ids.txt").read_text().split())
    train = d[~d.experiment_id.isin(holdout)]
    n_all, n_one = len(train), train.setting.nunique()
    rows = [[label, f3(b.loc[v].acc), f3(b.loc[v].macro_f1), f3(b.loc[v].mae), f3(c.loc[v].acc_w_mean)]
            for v, label in (("full", f"All training runs ({n_all})"),
                             ("keep_first", f"Repeats removed, earlier run kept ({n_one})"),
                             ("keep_second", f"Repeats removed, later run kept ({n_one})"))]
    write_table("repeat_check", "The tuned models trained with and without the second run of each repeated setting.",
                "tab:repeat_check", ["Training data", "Accuracy", "Macro-F1", "MAE", "Accuracy"], rows, "lcccc",
                groups=[("", 1), ("Holdout", 3), ("CV", 1)],
                note="All three are scored on the same rows: the 50 holdout runs, and in cross-validation (ten "
                     "reshuffles) the settings that were run once.")
    for v, key in (("keep_first", "Earlier"), ("keep_second", "Later")):
        num(f"repeatHoldoutAcc{key}", f3(b.loc[v].acc))
    num("repeatHoldoutAccAll", f3(b.loc["full"].acc))


def threshold_table():
    """Full-rule sensitivity of the severity labels to the latency thresholds (scripts/analysis/threshold_sensitivity_full.py)."""
    t = pd.read_csv(RES / "reliability" / "threshold_full" / "summary.csv")
    used = t[t.threshold == "used"].iloc[0]
    rows = [["(as used)", "--", "--", f"{used.Low} / {used.Medium} / {used.High}", "--"]]
    for name, label in (("latency_medium", "Medium"), ("latency_high", "High")):
        g = t[t.threshold == name]
        rows.append("MID")
        for i, (_, r) in enumerate(g.iterrows()):
            rows.append([label if i == 0 else "", f"{r.used:.0%}".replace("%", r"\%") if i == 0 else "",
                         f"{r.moved_to:.0%}".replace("%", r"\%"), f"{r.Low} / {r.Medium} / {r.High}", int(r.changed)])
    write_table("threshold_sensitivity", "Sensitivity of the severity labels to the latency thresholds. Each row moves one "
                "threshold, given as a rise in latency, keeps the other thresholds as used and applies the full severity "
                f"rule to all {used.Low + used.Medium + used.High} runs.", "tab:threshold_sensitivity",
                ["Latency threshold", "Used", "Moved to", "Low / Medium / High", "Runs that change class"], rows, "lcccc")


def main():
    OUT_T.mkdir(parents=True, exist_ok=True)
    dataset_tables(); ml_tables(); features_table(); tuning_table(); reliability_tables(); genai_tables(); environment_tables()
    threshold_table(); correlation_table(); retrieval_table(); affected_table(); user_impact_table(); binary_table(); repeat_tables()
    for name in OBSOLETE:
        (OUT_T / f"{name}.tex").unlink(missing_ok=True)
    (OUT_T / "numbers.tex").write_text("% generated by scripts/report/make_tables.py, do not edit by hand\n" +
                                       "".join(rf"\newcommand{{\{k}}}{{{v}}}" + "\n" for k, v in sorted(numbers.items())))
    print(f"{len(list(OUT_T.glob('*.tex')))} table files, {len(numbers)} numbers → {OUT_T}")


if __name__ == "__main__":
    main()
