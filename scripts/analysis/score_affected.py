"""
do the LLM agents name the right affected services?

For every holdout investigation, the services the agent lists as affected are compared with the services the
blast-radius measurement found affected in the label window (dataset/processed/label_measurements.csv).
  - the changed service is removed from the agent's list (the blast radius counts only other services);
  - services the measurement could not judge (too few requests) count neither for nor against the agent;
  - a gated incident that the ML screen cleared names no services.
Reported per system and model, averaged over incidents and repeats: count error (MAE against the measured blast
radius), exact list right, share of named services that were affected, share of affected services that were named.
The ML regressor's MAE at 30 s on the same runs is given for reference, with a paired bootstrap of the difference.

Reads only saved results; makes no model calls.
Usage:  venv/bin/python scripts/analysis/score_affected.py   -> results/genai/affected/
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
LABELS = ROOT / "dataset" / "processed" / "label_measurements.csv"
HOLDOUT = ROOT / "results" / "genai" / "holdout"
REG_PRED = ROOT / "results" / "fe_tuned" / "holdout_regression_predictions.csv"
OUT = ROOT / "results" / "genai" / "affected"
DECISION_T = 30
N_BOOT, SEED = 5000, 0


def services(x) -> set[str]:
    if pd.isna(x) or str(x).strip() == "":
        return set()
    return {s.strip() for s in str(x).replace(";", ",").split(",") if s.strip()}


def per_incident() -> pd.DataFrame:
    lab = pd.read_csv(LABELS).set_index("experiment_id")
    rows = []
    for f in sorted(HOLDOUT.glob("*/rep*/results.csv")):
        model, rep = f.parent.parent.name, f.parent.name
        for _, r in pd.read_csv(f).iterrows():
            if r.system == "ml_only":
                continue
            lr = lab.loc[r.run]
            truth, unjudged = services(lr.blast_affected), services(lr.blast_unjudged)
            named = services(r.affected)
            pred = named - {lr.service}
            judged = pred - unjudged
            rows.append(dict(model=model, rep=rep, system=r.system, run=r.run, true_n=len(truth), pred_n=len(pred),
                             abs_err=abs(len(pred) - len(truth)), exact=judged == truth,
                             tp=len(judged & truth), fp=len(judged - truth), fn=len(truth - pred)))
    return pd.DataFrame(rows)


def summary(df: pd.DataFrame) -> pd.DataFrame:
    reg = pd.read_csv(REG_PRED).query("time_into_fault_s == @DECISION_T")
    reg = reg.assign(ml_err=(reg.pred_blast_radius.clip(lower=0) - reg.true_blast_radius).abs()).set_index("experiment_id")
    rng = np.random.default_rng(SEED)
    out = []
    for (system, model), g in df.groupby(["system", "model"]):
        tp, fp, fn = g.tp.sum(), g.fp.sum(), g.fn.sum()
        inc = g.groupby("run").abs_err.mean()                      # each incident averaged over its repeats
        j = pd.concat([inc, reg.ml_err], axis=1, join="inner")
        diff = (j.abs_err - j.ml_err).to_numpy()
        boot = [rng.choice(diff, len(diff)).mean() for _ in range(N_BOOT)]
        out.append(dict(system=system, model=model, count_mae=g.abs_err.mean(), exact=g.exact.mean(),
                        named_affected=tp / (tp + fp) if tp + fp else np.nan,
                        affected_named=tp / (tp + fn) if tp + fn else np.nan,
                        ml_mae_30s=j.ml_err.mean(), diff_vs_ml=diff.mean(),
                        diff_lo=np.percentile(boot, 2.5), diff_hi=np.percentile(boot, 97.5), incidents=len(j)))
    return pd.DataFrame(out)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    df = per_incident()
    s = summary(df)
    df.to_csv(OUT / "per_incident.csv", index=False)
    s.to_csv(OUT / "summary.csv", index=False)
    by_n = (df[df.system == "llm_only"].groupby("true_n")
            .agg(n=("run", "size"), mean_named=("pred_n", "mean"), count_mae=("abs_err", "mean"), exact=("exact", "mean")))
    by_n.to_csv(OUT / "llm_only_by_true_blast_radius.csv")
    print(s.round(3).to_string(index=False))
    print("\nLLM only, by measured blast radius:\n" + by_n.round(2).to_string())


if __name__ == "__main__":
    main()
