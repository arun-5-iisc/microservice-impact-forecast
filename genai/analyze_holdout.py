"""
Summarise the holdout evaluation

Reads
  results/genai/holdout/<model>/rep<K>/results.csv         main comparison, frozen prompt, K repeats
  results/genai/sensitivity/<model>/prompt_<v>/results.csv  prompt-sensitivity check, 1 run
Writes (to --out, default results/genai/holdout_summary/)
  main.csv         per model x system: severity accuracy (mean over repeats, run-to-run SD, 95% CI over incidents),
                   pass^k / pass@k, cause accuracy, false and missed alerts, action acceptability, cost, errors
  paired.csv       paired comparisons on the same incidents (difference in severity accuracy, 95% CI)
  sensitivity.csv  frozen prompt (mean of repeats) vs the alternative prompt (1 run), per model
  errors.csv       investigations that still failed (counted as wrong)
  summary.md       the same tables in Markdown

Confidence intervals resample the 50 incidents (bootstrap, 5,000 draws, seed 0); each incident's score is first
averaged over its repeats, so the intervals reflect which incidents were tested, not run-to-run noise. Run-to-run
noise is reported separately as the SD of the per-repeat accuracies.

Usage:  python -m genai.analyze_holdout [--root results/genai] [--out results/genai/holdout_summary]
"""
from __future__ import annotations

import argparse
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
SYSTEMS = ["ml_only", "llm_only", "ml_llm", "gated"]
N_BOOT, SEED = 5000, 0


def as_float(s: pd.Series) -> pd.Series:
    return s.map({True: 1.0, False: 0.0, "True": 1.0, "False": 0.0}).astype(float)


def load(root: Path) -> pd.DataFrame:
    parts = []
    for f in sorted((root / "holdout").glob("*/rep*/results.csv")):
        d = pd.read_csv(f).assign(model_dir=f.parent.parent.name, rep=int(f.parent.name[3:]), prompt_set="main")
        parts.append(d)
    for f in sorted((root / "sensitivity").glob("*/prompt_*/results.csv")):
        d = pd.read_csv(f).assign(model_dir=f.parent.parent.name, rep=1, prompt_set=f.parent.name.removeprefix("prompt_"))
        parts.append(d)
    if not parts:
        sys.exit(f"no holdout results under {root}")
    df = pd.concat(parts, ignore_index=True)
    for c in ("severity_correct", "cause_correct", "action_ok", "false_alert", "missed_alert"):
        df[c] = as_float(df[c])
    df["severity_correct"] = df["severity_correct"].fillna(0.0)     # a failed investigation counts as wrong
    return df


def boot_ci(per_incident: pd.Series) -> tuple[float, float]:
    v = per_incident.to_numpy(float)
    rng = np.random.default_rng(SEED)
    means = v[rng.integers(0, len(v), size=(N_BOOT, len(v)))].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def per_incident(g: pd.DataFrame) -> pd.Series:
    """Severity correctness per incident, averaged over that incident's repeats."""
    return g.groupby("run")["severity_correct"].mean()


def main_table(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    rows, by_key = [], {}
    main = df[df.prompt_set == "main"]
    models = sorted(main.model_dir.unique())
    for system in SYSTEMS:
        for model in (models[:1] if system == "ml_only" else models):     # ml_only is the same model for all
            g = main[(main.model_dir == model) & (main.system == system)]
            if system == "ml_only":
                g = g[g.rep == g.rep.min()]                                    # deterministic: one run
            if g.empty:
                continue
            reps = sorted(g.rep.unique())
            acc_rep = g.groupby("rep")["severity_correct"].mean()
            inc = per_incident(g)
            lo, hi = boot_ci(inc)
            k = g.groupby("run")["rep"].nunique()
            full = inc[k == len(reps)]
            n_inv = int(g.cost_usd.notna().sum()) if "cost_usd" in g else 0
            rows.append({
                "model": "— (same ML model)" if system == "ml_only" else model, "system": system,
                "repeats": len(reps), "incidents": int(inc.size),
                "severity_acc": acc_rep.mean(), "run_to_run_sd": acc_rep.std(ddof=1) if len(reps) > 1 else np.nan,
                "ci95_low": lo, "ci95_high": hi,
                # consistency over the k repeats: pass^k = right in every repeat, pass@k = right in at least one
                "pass^k": (g[g.run.isin(full.index)].groupby("run")["severity_correct"].min() == 1).mean()
                if len(reps) > 1 else np.nan,
                "pass@k": (g.groupby("run")["severity_correct"].max() == 1).mean() if len(reps) > 1 else np.nan,
                "cause_acc_MH": g["cause_correct"].mean(),
                "false_alerts_per_run": g.groupby("rep")["false_alert"].sum().mean(),
                "missed_alerts_per_run": g.groupby("rep")["missed_alert"].sum().mean(),
                "action_ok": g["action_ok"].mean(),
                "cost_usd_total": g["cost_usd"].sum() if "cost_usd" in g else np.nan,
                "cost_usd_per_incident_run": g["cost_usd"].sum() / (inc.size * len(reps)) if "cost_usd" in g else np.nan,
                "llm_investigations": n_inv,
                "errors": int(g["error"].notna().sum()) if "error" in g else 0,
            })
            by_key[(model, system)] = inc
    return pd.DataFrame(rows), by_key


def rule_row(df: pd.DataFrame) -> dict | None:
    """Context: the labelling rule applied to the state at 30 s (reactive alerting, no ML, no LLM)."""
    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    from preprocess import _compute_severity    # noqa: E402
    data = REPO_ROOT / "dataset" / "processed" / "training_dataset.csv"
    if not data.exists():
        return None
    runs = set(df.run)
    t = pd.read_csv(data)
    t = t[(t.time_into_fault_s == 30) & t.experiment_id.isin(runs)]
    nz = lambda v: None if pd.isna(v) else v    # noqa: E731
    pred = [_compute_severity(nz(r.target_latency_delta), nz(r.target_error_rate), nz(r.target_restart_count),
                              nz(r.target_replicas)) for r in t.itertuples()]
    ok = pd.Series([p == s for p, s in zip(pred, t.severity)], index=t.experiment_id, dtype=float)
    lo, hi = boot_ci(ok)
    alert = pd.Series([p in ("Medium", "High") for p in pred], index=t.experiment_id)
    should = t.set_index("experiment_id").severity.isin(["Medium", "High"])
    return {"model": "— (no ML, no LLM)", "system": "rule_at_30s", "repeats": 1, "incidents": int(ok.size),
            "severity_acc": ok.mean(), "ci95_low": lo, "ci95_high": hi,
            "false_alerts_per_run": int((alert & ~should).sum()), "missed_alerts_per_run": int((~alert & should).sum())}


def paired(a: pd.Series, b: pd.Series) -> dict:
    j = pd.concat([a.rename("a"), b.rename("b")], axis=1, join="inner")
    d = j.a - j.b
    lo, hi = boot_ci(d)
    return {"incidents": int(d.size), "acc_a": j.a.mean(), "acc_b": j.b.mean(), "difference": d.mean(),
            "ci95_low": lo, "ci95_high": hi, "a_better_on": int((d > 0).sum()), "b_better_on": int((d < 0).sum())}


def paired_table(by_key: dict) -> pd.DataFrame:
    rows = []
    models = sorted({m for m, s in by_key if s != "ml_only"})
    ml = next((v for (m, s), v in by_key.items() if s == "ml_only"), None)
    for m in models:
        get = lambda s: by_key.get((m, s))    # noqa: E731
        pairs = [("llm_only", ml, "ml_only"), ("ml_llm", ml, "ml_only"), ("gated", ml, "ml_only"),
                 ("ml_llm", get("llm_only"), "llm_only"), ("gated", get("llm_only"), "llm_only")]
        for sa, b, sb in pairs:
            a = get(sa)
            if a is not None and b is not None:
                rows.append({"comparison": f"{m}: {sa} vs {sb}", **paired(a, b)})
    for m1, m2 in combinations(models, 2):
        a, b = by_key.get((m1, "llm_only")), by_key.get((m2, "llm_only"))
        if a is not None and b is not None:
            rows.append({"comparison": f"llm_only: {m1} vs {m2}", **paired(a, b)})
    return pd.DataFrame(rows)


def sensitivity_table(df: pd.DataFrame, by_key: dict, run_sd: dict) -> pd.DataFrame:
    rows = []
    for (model, prompt), g in df[(df.prompt_set != "main") & (df.system == "llm_only")].groupby(["model_dir", "prompt_set"]):
        main = by_key.get((model, "llm_only"))
        if main is None:
            continue
        r = paired(main, per_incident(g))
        rows.append({"model": model, "alternative_prompt": prompt, "frozen_prompt_acc (mean of repeats)": r["acc_a"],
                     "alternative_prompt_acc (1 run)": r["acc_b"], "difference": r["difference"],
                     "ci95_low": r["ci95_low"], "ci95_high": r["ci95_high"],
                     "run_to_run_sd (frozen prompt)": run_sd.get(model, np.nan),
                     "false_alerts_alt": g["false_alert"].sum(), "missed_alerts_alt": g["missed_alert"].sum(),
                     "cost_usd_alt": g["cost_usd"].sum()})
    return pd.DataFrame(rows)


def to_md(df: pd.DataFrame) -> str:
    fmt = lambda v: "" if pd.isna(v) else (f"{v:.3f}" if isinstance(v, float) else str(v))    # noqa: E731
    lines = ["| " + " | ".join(df.columns) + " |", "|" + "---|" * len(df.columns)]
    lines += ["| " + " | ".join(fmt(v) for v in r) + " |" for r in df.itertuples(index=False)]
    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", type=Path, default=REPO_ROOT / "results" / "genai")
    p.add_argument("--out", type=Path, default=REPO_ROOT / "results" / "genai" / "holdout_summary")
    a = p.parse_args()
    df = load(a.root)
    main_df, by_key = main_table(df)
    rr = rule_row(df)
    if rr:
        main_df = pd.concat([main_df, pd.DataFrame([rr])], ignore_index=True)
    run_sd = {r.model: r.run_to_run_sd for r in main_df[main_df.system == "llm_only"].itertuples()}
    pair_df = paired_table(by_key)
    sens_df = sensitivity_table(df, by_key, run_sd)
    err_df = df[df["error"].notna()][["model_dir", "prompt_set", "rep", "system", "run", "error"]] if "error" in df else pd.DataFrame()
    a.out.mkdir(parents=True, exist_ok=True)
    for name, d in (("main", main_df), ("paired", pair_df), ("sensitivity", sens_df), ("errors", err_df)):
        d.to_csv(a.out / f"{name}.csv", index=False)
    md = ["# Holdout evaluation summary", "", "Generated by `genai/analyze_holdout.py`; the analysis plan was fixed before any holdout run.",
          "Severity accuracy: mean over repeats; 95% CI from resampling incidents; run-to-run SD across repeats.", "",
          "## Main comparison", "", to_md(main_df), "", "## Paired comparisons (same incidents)", "", to_md(pair_df), "",
          "## Prompt sensitivity", "", to_md(sens_df) if len(sens_df) else "(no sensitivity runs yet)", "",
          f"## Failed investigations: {len(err_df)}", ""]
    (a.out / "summary.md").write_text("\n".join(md))
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    print(main_df.round(3).to_string(index=False))
    print("\n" + pair_df.round(3).to_string(index=False))
    if len(sens_df):
        print("\n" + sens_df.round(3).to_string(index=False))
    print(f"\nfailed investigations: {len(err_df)}   → {a.out}")


if __name__ == "__main__":
    main()
