"""
Evaluate past-incident retrieval (RAG) on the holdout incidents.

For every holdout run, the query is what is observable T seconds after the change
(knowledge.incident_query: no cause, no outcome). A retrieved past incident is relevant when
its cause (fault type) matches the holdout run's true cause. Metrics over the top 5:
  hit@1, hit@5, precision@5, MRR, and vote@5 (majority cause of the top 5 = true cause).
Compared for keyword / vector / hybrid search, with and without filtering to the changed
service, against random retrieval (expected values computed exactly).

Usage:  venv/bin/python -m genai.eval_retrieval      -> results/genai/retrieval_eval.csv
"""
from __future__ import annotations

import json
import sys
import time
from collections import Counter
from pathlib import Path

import pandas as pd
from scipy.stats import hypergeom

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from genai.knowledge import DOCS_PATH, SEARCH_MODES, RagIndex, incident_query, setting_of  # noqa: E402
from genai.replay import RecordedRun, holdout_ids  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = REPO_ROOT / "results" / "genai"
SNAPSHOTS = (15, 30)
K = 5


def build_queries() -> list[dict]:
    """The query for every holdout run at each snapshot, built from the replayed telemetry with the current code
    every time (a saved copy could be out of date), and written to retrieval_queries.jsonl as a record."""
    rows = []
    for i, e in enumerate(holdout_ids(), 1):
        run = RecordedRun(e)
        m = run._meta
        for T in SNAPSHOTS:
            rows.append({"run": e, "T": T, "service": run.changed_service, "fault_type": m["fault_type"],
                         "setting": setting_of(run.changed_service, m["fault_type"], m["fault_value"], run.traffic_level),
                         "query": incident_query(run.at(T))})
        print(f"  queries [{i}/50] {e}", file=sys.stderr, flush=True)
    (OUT_DIR / "retrieval_queries.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    return rows


def random_baseline(kb: pd.DataFrame, q: dict, service_filter: bool) -> dict:
    pool = kb[kb["service"] == q["service"]] if service_filter else kb
    n, m = len(pool), int((pool["fault_type"] == q["fault_type"]).sum())
    k = min(K, n)
    return {"hit@1": m / n if n else 0, "hit@5": 1 - hypergeom.pmf(0, n, m, k) if n else 0,
            "precision@5": m / n if n else 0}


def score(retrieved: list[str], truth: str) -> dict:
    hits = [c == truth for c in retrieved]
    first = next((i for i, h in enumerate(hits) if h), None)
    vote = Counter(retrieved).most_common()
    top_votes = [c for c, n in vote if n == vote[0][1]] if vote else []
    winner = next((c for c in retrieved if c in top_votes), None)   # ties -> highest-ranked
    return {"hit@1": float(bool(hits) and hits[0]), "hit@5": float(any(hits)),
            "precision@5": sum(hits) / len(hits) if hits else 0.0,
            "mrr": 1 / (first + 1) if first is not None else 0.0, "vote@5": float(winner == truth)}


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    queries = build_queries()
    kb = pd.DataFrame([json.loads(line) for line in DOCS_PATH.read_text().splitlines()])
    rows, timing = [], {m: [] for m in SEARCH_MODES}
    for q in queries:
        index = RagIndex(exclude_setting=q["setting"])
        for service_filter in (False, True):
            svc = q["service"] if service_filter else None
            for mode in SEARCH_MODES:
                t = time.time()
                res = index.search(q["query"], service=svc, k=K, mode=mode)
                timing[mode].append(time.time() - t)
                causes = [kb.set_index("incident_id").loc[r["incident_id"], "fault_type"] for r in res]
                rows.append({"run": q["run"], "T": q["T"], "fault_type": q["fault_type"], "mode": mode,
                             "service_filter": service_filter, **score(causes, q["fault_type"]),
                             "retrieved": ",".join(r["incident_id"] for r in res)})
            rows.append({"run": q["run"], "T": q["T"], "fault_type": q["fault_type"], "mode": "random",
                         "service_filter": service_filter, **random_baseline(kb, q, service_filter)})
    R = pd.DataFrame(rows)
    R.to_csv(OUT_DIR / "retrieval_eval_per_query.csv", index=False)
    metrics = ["hit@1", "hit@5", "precision@5", "mrr", "vote@5"]
    summ = R.groupby(["T", "service_filter", "mode"], sort=False)[metrics].mean().round(3).reset_index()
    summ.to_csv(OUT_DIR / "retrieval_eval.csv", index=False)
    print(f"\nRetrieval of past incidents with the same cause — {len(set(R.run))} holdout incidents, "
          f"{len(kb)} past incidents indexed\n")
    print(summ.to_string(index=False))
    best = summ[summ["mode"] != "random"].sort_values(["precision@5", "hit@1"], ascending=False).iloc[0]
    print(f"\nBest by precision@5: mode={best['mode']}, service_filter={best['service_filter']}, T={best['T']}")
    per_type = (R[(R["mode"] == best["mode"]) & (R["service_filter"] == best["service_filter"]) & (R["T"] == best["T"])]
                .groupby("fault_type")[["hit@1", "precision@5", "vote@5"]].mean().round(2))
    print("\nBy true cause (best setting):\n" + per_type.to_string())
    print("\nMean search time (ms): " + ", ".join(f"{m} {1000 * sum(v) / len(v):.1f}" for m, v in timing.items()))


if __name__ == "__main__":
    main()
