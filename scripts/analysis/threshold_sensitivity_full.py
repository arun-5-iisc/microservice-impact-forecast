"""
Sensitivity of the final severity labels to each threshold of the severity rule (full rule).

The pipeline's threshold analysis (threshold_analysis.py --mode threshold) applies the latency rule alone. This
script applies the full rule of scripts/preprocess.py (replicas, restarts, latency, error rate) to the stored label
measurements and moves one threshold at a time, keeping the others as used. It reports the class sizes and how
many runs change class compared with the labels in the dataset.

Reads dataset/processed/label_measurements.csv only.
Usage:  venv/bin/python scripts/analysis/threshold_sensitivity_full.py   -> results/reliability/threshold_full/
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
import preprocess  # noqa: E402  (thresholds as used for the labels)

LABELS = ROOT / "dataset" / "processed" / "label_measurements.csv"
OUT = ROOT / "results" / "reliability" / "threshold_full"

USED = {"latency_medium": preprocess.LATENCY_DELTA_MEDIUM, "latency_high": preprocess.LATENCY_DELTA_HIGH,
        "error_medium": preprocess.ERROR_RATE_MEDIUM, "error_high": preprocess.ERROR_RATE_HIGH}
MOVES = {"latency_medium": (0.15, 0.25), "latency_high": (0.8, 1.2),
         "error_medium": (0.005, 0.02), "error_high": (0.03, 0.10)}


def severity(r, latency_medium, latency_high, error_medium, error_high) -> str:
    """The rule of preprocess._compute_severity with the thresholds as arguments."""
    lat = r.final_latency_delta if (r.latency_measurable and pd.notna(r.final_latency_delta)) else None
    err = r.final_error_rate if pd.notna(r.final_error_rate) else None
    if (r.final_replicas == 0 or (pd.notna(r.final_restart_count) and r.final_restart_count > 0)
            or (lat is not None and lat > latency_high) or (err is not None and err > error_high)):
        return "High"
    if (lat is not None and lat > latency_medium) or (err is not None and err > error_medium):
        return "Medium"
    return "Low"


def main() -> None:
    lab = pd.read_csv(LABELS)
    label = lambda th: lab.apply(lambda r: severity(r, **th), axis=1)
    used = label(USED)
    if not (used == lab.severity).all():
        raise SystemExit("the rule here does not reproduce the stored labels; check against preprocess.py")
    rows = [dict(threshold="used", used=None, moved_to=None, **used.value_counts().reindex(["Low", "Medium", "High"]).to_dict(),
                 changed=0)]
    for name, values in MOVES.items():
        for v in values:
            s = label({**USED, name: v})
            rows.append(dict(threshold=name, used=USED[name], moved_to=v,
                             **s.value_counts().reindex(["Low", "Medium", "High"], fill_value=0).to_dict(),
                             changed=int((s != used).sum())))
    out = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT / "summary.csv", index=False)
    print(f"Reproduction check: the rule gives the stored labels for all {len(lab)} runs. OK")
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
