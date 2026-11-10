"""
Retrieval methods compared on the training runs, without using the holdout.

genai/eval_retrieval.py scores keyword, vector and hybrid search on the holdout incidents. This script runs the same
comparison on the training runs: each of them is a query, built fresh from its recording at 15 and 30 s, and every run
of its own setting is hidden from the search, as for the agent. A retrieved incident counts if it has the same cause.
The difference between methods in the top-result rate gets a 95% bootstrap interval over the runs.

Works on a temporary copy of the search index, so the index in models/rag_index is only read.
Needs the span cache (./scripts/run_genai_eval.sh prepare) and the index (python -m genai.knowledge build).
Usage:  venv/bin/python scripts/analysis/retrieval_train_check.py   -> results/genai/retrieval_train/
"""
from __future__ import annotations

import json
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from genai import eval_retrieval as E  # noqa: E402
from genai import knowledge as K  # noqa: E402
from genai.replay import RecordedRun  # noqa: E402

OUT = ROOT / "results" / "genai" / "retrieval_train"
SNAPSHOTS = (15, 30)
PAIRS = (("hybrid", "keyword"), ("hybrid", "vector"), ("keyword", "vector"))
N_BOOT, SEED = 5000, 0


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        index = Path(tmp) / "index"
        shutil.copytree(K.INDEX_DIR, index)
        K.CHROMA_DIR, K.DOCS_PATH = index / "chroma", index / K.DOCS_PATH.name
        kb = pd.DataFrame([json.loads(line) for line in K.DOCS_PATH.read_text().splitlines()]).set_index("incident_id")
        rows = []
        for n, run_id in enumerate(kb.index, 1):
            run = RecordedRun(run_id)
            meta = run._meta
            search = K.RagIndex(exclude_setting=K.setting_of(run.changed_service, meta["fault_type"], meta["fault_value"],
                                                             run.traffic_level))
            if run_id in search.docs:
                raise SystemExit(f"{run_id} can find itself: its setting was not hidden")
            for t in SNAPSHOTS:
                query = K.incident_query(run.at(t))
                for mode in K.SEARCH_MODES:
                    found = search.search(query, service=None, k=E.K, mode=mode)
                    causes = [kb.loc[r["incident_id"], "fault_type"] for r in found]
                    rows.append({"run": run_id, "T": t, "mode": mode, "fault_type": meta["fault_type"],
                                 **E.score(causes, meta["fault_type"])})
            if n % 40 == 0:
                print(f"  {n}/{len(kb)}", file=sys.stderr, flush=True)
    d = pd.DataFrame(rows)
    summary = d.groupby(["T", "mode"], sort=False)[["hit@1", "hit@5", "precision@5", "mrr", "vote@5"]].mean().reset_index()
    rng = np.random.default_rng(SEED)
    paired = []
    for t in SNAPSHOTS:
        wide = d[d["T"] == t].pivot(index="run", columns="mode", values="hit@1")
        for a, b in PAIRS:
            diff = (wide[a] - wide[b]).to_numpy()
            boot = [rng.choice(diff, len(diff)).mean() for _ in range(N_BOOT)]
            paired.append(dict(T=t, a=a, b=b, diff=diff.mean(), lo=np.percentile(boot, 2.5), hi=np.percentile(boot, 97.5)))
    OUT.mkdir(parents=True, exist_ok=True)
    d.to_csv(OUT / "per_query.csv", index=False)
    summary.to_csv(OUT / "summary.csv", index=False)
    pd.DataFrame(paired).to_csv(OUT / "paired.csv", index=False)
    print(f"{len(kb)} training runs as queries")
    print(summary.round(3).to_string(index=False))
    print(pd.DataFrame(paired).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
