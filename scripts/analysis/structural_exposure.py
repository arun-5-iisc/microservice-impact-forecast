#!/usr/bin/env python
"""
Structural exposure: which services can a change reach, and how well does the call graph alone predict the spread?

A service is structurally exposed to a change if it calls the changed service, directly or indirectly (failures
travel up the call graph to the callers). For every run this compares the exposed services with the services that
were actually affected (the blast radius label), and scores a graph-only baseline that predicts every exposed
service to be affected, on the training runs and on the holdout.

Usage: python scripts/analysis/structural_exposure.py [output folder]   (default results/reliability/structural_exposure/)
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
OUT = (Path(sys.argv[1]) if len(sys.argv) > 1 else REPO / "results" / "reliability" / "structural_exposure").resolve()
OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(REPO / "scripts"))
from feature_engineer import CALL_GRAPH  # noqa: E402

CALLERS = {s: {c for c, callees in CALL_GRAPH.items() if s in callees} for s in CALL_GRAPH}


def exposed(service: str) -> set:
    """Services that call `service`, directly or indirectly."""
    seen, todo = set(), list(CALLERS[service])
    while todo:
        c = todo.pop()
        if c not in seen:
            seen.add(c)
            todo += list(CALLERS[c])
    return seen


def main():
    runs = pd.read_csv(REPO / "dataset" / "processed" / "training_dataset.csv").drop_duplicates("experiment_id")
    holdout = set((REPO / "dataset" / "processed" / "holdout_ids.txt").read_text().split())
    runs["split"] = np.where(runs.experiment_id.isin(holdout), "holdout", "training")
    runs["exposure"] = runs.service.map(lambda s: len(exposed(s)))
    runs["within_exposure"] = runs.blast_radius_final <= runs.exposure

    report = []

    def out(s=""):
        print(s)
        report.append(s)

    per_service = pd.DataFrame([{"service": s, "exposure": len(exposed(s)),
                                 "exposed_services": ", ".join(sorted(exposed(s))) or "-"} for s in sorted(CALL_GRAPH)])
    per_service.to_csv(OUT / "exposure_per_service.csv", index=False)
    out("Structural exposure per changed service (services that call it, directly or indirectly):")
    out(per_service.to_string(index=False))

    rows = []
    for split in ("training", "holdout"):
        g = runs[runs.split == split]
        y, e = g.blast_radius_final, g.exposure
        rows.append({"split": split, "runs": len(g),
                     "graph_only_mae": round(float(np.mean(np.abs(e - y))), 4),
                     "graph_only_rmse": round(float(np.sqrt(np.mean((e - y) ** 2))), 4),
                     # R^2 = 1 - (squared error / squared spread of the labels around their mean)
                     "graph_only_r2": round(float(1 - np.sum((e - y) ** 2) / np.sum((y - y.mean()) ** 2)), 4),
                     "always_zero_mae": round(float(np.mean(np.abs(y))), 4),
                     "spread_within_exposure": round(float(g.within_exposure.mean()), 4)})
    scores = pd.DataFrame(rows)
    scores.to_csv(OUT / "graph_only_baseline.csv", index=False)
    out("\nGraph-only baseline (predict every exposed service to be affected):")
    out(scores.to_string(index=False))

    sev = runs.groupby("severity").agg(runs=("experiment_id", "size"), mean_exposure=("exposure", "mean"),
                                       mean_spread=("blast_radius_final", "mean"),
                                       exposed=("exposure", "sum"), affected=("blast_radius_final", "sum"))
    sev["share_of_exposed_affected"] = (sev.affected / sev.exposed).round(3)
    sev = sev.reindex(["Low", "Medium", "High"])
    sev.to_csv(OUT / "exposure_by_severity.csv")
    out("\nExposed services that were actually affected, by the changed service's severity:")
    out(sev.round(3).to_string())
    (OUT / "report.txt").write_text("\n".join(report) + "\n")


if __name__ == "__main__":
    main()
