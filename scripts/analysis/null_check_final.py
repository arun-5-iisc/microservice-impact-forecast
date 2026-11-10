"""
check of the final blast-radius measurement (scripts/span_measures.py).

For every run, the blast radius is computed with the final code in three ways, as in blast_v2.py's comparison:
  null    first half of the baseline vs its second half (nothing has changed, so ideally 0);
  signal  first half of the baseline vs the last 30 s of the change;
  final60 the baseline vs the last 60 s of the change (the label window; must equal the stored label).
blast_v2.py ran the same comparison for the candidate rule D2 that the final rule grew out of; this script measures
the final rule itself, which also has the 1 ms minimum, the error test and its own minimum-evidence check.

Uses the span cache (dataset/spans/, written by ./scripts/run_genai_eval.sh prepare), which was checked to be
identical to the spans parsed from every run's traces.json.
Usage:  venv/bin/python scripts/analysis/null_check_final.py   -> results/reliability/null_final/
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
import span_measures as sm  # noqa: E402
from genai.replay import RecordedRun  # noqa: E402

LABELS = ROOT / "dataset" / "processed" / "label_measurements.csv"
OUT = ROOT / "results" / "reliability" / "null_final"


def main() -> None:
    lab = pd.read_csv(LABELS)
    rows = []
    for _, r in lab.iterrows():
        run = RecordedRun(r.experiment_id)
        ph = json.loads((run.dir / "metadata.json").read_text())["phases"]
        bs, be = ph["baseline"]["start"], ph["baseline"]["end"]
        fs, fe = ph["fault"]["start"], ph["fault"]["end"]
        mid = (bs + be) / 2
        rs = run.spans
        rows.append(dict(run=r.experiment_id, fault_type=r.fault_type,
                         null=sm.blast_radius(rs, r.service, (bs, mid), (mid, be))["count"],
                         signal=sm.blast_radius(rs, r.service, (bs, mid), (fe - 30, fe))["count"],
                         final60=sm.blast_radius(rs, r.service, (bs, be), (max(fe - 60, fs), fe))["count"],
                         stored=r.blast_radius_final))
    d = pd.DataFrame(rows)
    if not (d.final60 == d.stored).all():
        raise SystemExit("the label-window blast radius does not reproduce the stored labels; check the span cache")
    OUT.mkdir(parents=True, exist_ok=True)
    d.to_csv(OUT / "per_run.csv", index=False)
    by_type = d.groupby("fault_type")[["null", "signal"]].mean()
    by_type.loc["ALL"] = d[["null", "signal"]].mean()
    report = [f"Reproduction check: the label-window blast radius equals the stored label for all {len(d)} runs. OK",
              "", "Mean services counted as affected, 30 s vs 30 s (final measurement):",
              by_type.round(2).to_string(), "",
              f"With no change: {int(d.null.sum())} services flagged in {int((d.null > 0).sum())} of {len(d)} runs."]
    (OUT / "report.txt").write_text("\n".join(report) + "\n")
    print("\n".join(report))


if __name__ == "__main__":
    main()
