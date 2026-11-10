"""
Choose the development set: training-run incidents used for prompt and tool refinement (never the holdout).

Stratified by cause (fault type) x final severity: proportional allocation with at least 2 per non-empty
stratum (or all of it, if smaller), one run per setting, seed 42. Holdout runs are never included.

Usage:  venv/bin/python -m genai.dev_set [--size 40]   -> results/genai/dev_set.txt
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from genai.replay import PROCESSED_DIR, holdout_ids  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT = REPO_ROOT / "results" / "genai" / "dev_set.txt"
SEED, MIN_PER_STRATUM = 42, 2


def choose(size: int) -> pd.DataFrame:
    df = pd.read_csv(PROCESSED_DIR / "training_dataset.csv").drop_duplicates("experiment_id")
    df = df[~df["experiment_id"].isin(set(holdout_ids()))].copy()
    df["setting"] = df.service + "|" + df.fault_type + "|" + df.fault_value.astype(str) + "|" + df.traffic_level
    rng = np.random.default_rng(SEED)
    df = df.sample(frac=1, random_state=SEED).drop_duplicates("setting")          # one run per setting
    strata = df.groupby(["fault_type", "severity"])
    sizes = strata.size()
    alloc = (sizes / sizes.sum() * size).clip(lower=MIN_PER_STRATUM).clip(upper=sizes)
    alloc = alloc.round().astype(int)
    while alloc.sum() != size:                                                      # fix rounding to hit the target
        room = (sizes - alloc) if alloc.sum() < size else (alloc - np.minimum(MIN_PER_STRATUM, sizes))
        key = room[room > 0].index[rng.integers(len(room[room > 0]))]
        alloc[key] += 1 if alloc.sum() < size else -1
    picked = [g.sort_values("experiment_id").sample(int(alloc[k]), random_state=SEED) for k, g in strata]
    return pd.concat(picked).sort_values("experiment_id")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--size", type=int, default=40)
    a = p.parse_args()
    dev = choose(a.size)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(dev["experiment_id"]) + "\n")
    print(pd.crosstab(dev.fault_type, dev.severity, margins=True).to_string())
    print(f"\n{len(dev)} development incidents → {OUT}")


if __name__ == "__main__":
    main()
