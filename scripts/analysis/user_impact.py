#!/usr/bin/env python
"""
User impact at the frontend vs the changed service's severity, per run.

User impact is measured on the frontend's own request spans, last 60 s of the fault vs the baseline, and graded
with the severity rule's thresholds. When frontend is the changed service, it equals the severity. Severity comes
from dataset/processed/label_measurements.csv, so run scripts/run_pipeline.sh first.

Limitation: frontend requests are pooled over all pages, so a failure on one page (e.g. checkout, about 5 of 165
requests a minute) barely shows.
Usage: python scripts/analysis/user_impact.py [output folder]   (default results/reliability/user_impact/)
"""
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
OUT = (Path(sys.argv[1]) if len(sys.argv) > 1 else REPO / "results" / "reliability" / "user_impact").resolve()
OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(REPO))
from genai.replay import RecordedRun  # noqa: E402
import span_measures as sm  # noqa: E402   (on sys.path via genai.replay)

ORDER = ["Low", "Medium", "High"]


def grade(lat_delta: float, err: float) -> str:
    if lat_delta > 1.0 or err > 0.05:
        return "High"
    if lat_delta > 0.2 or err > 0.01:
        return "Medium"
    return "Low"


def main():
    lab = pd.read_csv(REPO / "dataset" / "processed" / "label_measurements.csv")
    rows = []
    for r in lab.itertuples():
        run = RecordedRun(r.experiment_id)
        fe = run.spans.own["frontend"]
        base = sm.window(fe, run.baseline_start, run.baseline_end)
        fin = sm.window(fe, run.fault_end - 60, run.fault_end)
        rec = {"run": r.experiment_id, "service": r.service, "fault": r.fault_type, "value": r.fault_value,
               "severity": r.severity, "blast_radius": r.blast_radius_final, "n_base": len(base), "n_final": len(fin)}
        if r.service == "frontend":
            rec["user_impact"] = r.severity           # the user-facing service itself
        elif len(base) >= 10 and len(fin) >= 10 and base[:, 1].mean() > 0:
            d = fin[:, 1].mean() / base[:, 1].mean() - 1
            e = fin[:, 2].mean()
            rec.update(fe_mean_delta=d, fe_error_rate=e, user_impact=grade(d, e))
        rows.append(rec)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "user_impact.csv", index=False)

    lines = [f"runs: {len(df)}; user impact measurable: {df.user_impact.notna().sum()}",
             f"severity and user impact differ in {(df.user_impact.notna() & (df.severity != df.user_impact)).sum()} runs", "",
             "severity of the changed service (rows) vs user impact at the frontend (columns):",
             pd.crosstab(df.severity, df.user_impact, margins=True)
               .reindex(index=ORDER + ["All"], columns=ORDER + ["All"], fill_value=0).to_string(), ""]
    h = df[(df.severity == "High") & df.user_impact.notna() & (df.service != "frontend")]
    per = h.groupby("service").agg(high_runs=("run", "size"), users_high=("user_impact", lambda s: int((s == "High").sum())),
                                   users_low=("user_impact", lambda s: int((s == "Low").sum())))
    lines += ["changed service High: user impact per changed service:", per.to_string(), ""]
    (OUT / "report.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"→ {OUT}")


if __name__ == "__main__":
    main()
