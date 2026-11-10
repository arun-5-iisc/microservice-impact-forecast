"""
Why the two runs of a repeated setting can get different severity labels.

For each of the settings that were run twice, this compares the two runs' label measurements (the label window is
the last 60 s of the change) and, where the severity differs, gives one of three causes:
  restart     one run had a restart or no running replica, the other did not;
  few         the changed service had fewer than FEW requests in the label window in both runs, so a few slow or
              failed requests can move the label;
  busy        the service was busy in both runs, and the same change slowed it by different amounts.
It also gives each run's latency change with its slowest request in the label window left out, which shows where a
single request decides the label.

Uses the span cache (dataset/spans/, written by ./scripts/run_genai_eval.sh prepare).
Usage:  venv/bin/python scripts/analysis/repeat_disagreements.py   -> results/reliability/repeat_disagreements/
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
import span_measures as sm  # noqa: E402
from genai.replay import RecordedRun  # noqa: E402

LABELS = ROOT / "dataset" / "processed" / "label_measurements.csv"
OUT = ROOT / "results" / "reliability" / "repeat_disagreements"
FEW = 20            # requests in the label window; the low-traffic services had 4 to 10, the busy ones hundreds


def without_slowest(run_id: str, service: str) -> float | None:
    """Latency change (as a fraction) in the label window with the slowest request left out."""
    run = RecordedRun(run_id)
    ph = json.loads((run.dir / "metadata.json").read_text())["phases"]
    reqs, _ = sm.target_view(run.spans, service)
    base = sm.window(reqs, ph["baseline"]["start"], ph["baseline"]["end"])
    fe = ph["fault"]["end"]
    lat = np.sort(sm.window(reqs, max(fe - 60, ph["fault"]["start"]), fe)[:, 1])
    if len(base) == 0 or len(lat) < 2:
        return None
    return float(lat[:-1].mean() / base[:, 1].mean() - 1)


def cause(a: pd.Series, b: pd.Series) -> str:
    if a.severity == b.severity:
        return ""
    broke = [r.final_restart_count > 0 or r.final_replicas == 0 for r in (a, b)]
    if broke[0] != broke[1]:
        return "restart"
    if a.n_final < FEW and b.n_final < FEW:
        return "few"
    return "busy"


def main() -> None:
    lab = pd.read_csv(LABELS)
    lab["setting"] = lab.service + "|" + lab.fault_type + "|" + lab.fault_value.astype(str) + "|" + lab.traffic_level
    rows = []
    for setting, g in lab.groupby("setting"):
        if len(g) != 2:
            continue
        a, b = g.sort_values("experiment_id").itertuples(index=False)
        a, b = pd.Series(a._asdict()), pd.Series(b._asdict())
        row = dict(setting=setting, cause=cause(a, b), same_severity=a.severity == b.severity,
                   same_blast=a.blast_radius_final == b.blast_radius_final)
        for tag, r in (("earlier", a), ("later", b)):
            row.update({f"{tag}_run": r.experiment_id, f"{tag}_severity": r.severity, f"{tag}_blast": r.blast_radius_final,
                        f"{tag}_requests": r.n_final, f"{tag}_latency_change": r.final_latency_delta,
                        f"{tag}_latency_change_without_slowest": without_slowest(r.experiment_id, r.service),
                        f"{tag}_error_rate": r.final_error_rate, f"{tag}_restarts": r.final_restart_count,
                        f"{tag}_replicas": r.final_replicas})
        rows.append(row)
    d = pd.DataFrame(rows)
    differ = d[~d.same_severity]
    few = differ[differ.cause == "few"]
    summary = {"settings": len(d), "same_severity": int(d.same_severity.sum()), "same_blast": int(d.same_blast.sum()),
               "severity_differs": len(differ),
               "cpu_or_memory": int(differ.setting.str.contains(r"\|(?:cpu|memory)\|").sum()),
               **{f"cause_{c}": int((differ.cause == c).sum()) for c in ("restart", "few", "busy")},
               "few_min_requests": int(few[["earlier_requests", "later_requests"]].min().min()),
               "few_max_requests": int(few[["earlier_requests", "later_requests"]].max().max())}
    OUT.mkdir(parents=True, exist_ok=True)
    d.to_csv(OUT / "per_setting.csv", index=False)
    pd.Series(summary).to_csv(OUT / "summary.csv", header=["value"])
    pd.set_option("display.width", 250)
    print(differ[["setting", "cause", "earlier_severity", "later_severity", "earlier_requests", "later_requests",
                  "earlier_latency_change", "later_latency_change", "earlier_latency_change_without_slowest",
                  "later_latency_change_without_slowest", "earlier_error_rate", "later_error_rate"]].round(2).to_string(index=False))
    print(pd.Series(summary).to_string())


if __name__ == "__main__":
    main()
