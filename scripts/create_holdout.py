"""
Creates a reproducible 110/20 train/holdout split at the experiment level.

Stratifies by fault_type × severity (proportional ~15%) so both fault type
and severity class are represented in both splits. Strata with fewer than
3 experiments contribute 0 to holdout — those rare examples stay in training
where the model needs them most.

Saves experiment IDs to:
  dataset/processed/holdout_ids.txt  — 20 holdout IDs (one per line)

Run once; commit the output file so the split is fixed for the whole
dissertation. The script refuses to overwrite an existing file.

Usage:
  python scripts/create_holdout.py
  python scripts/create_holdout.py --data dataset/processed/training_dataset.csv
  python scripts/create_holdout.py --target 20     # default
"""
import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger(__name__)

REPO_ROOT    = Path(__file__).parent.parent
DEFAULT_DATA = REPO_ROOT / "dataset" / "processed" / "training_dataset.csv"
HOLDOUT_OUT  = REPO_ROOT / "dataset" / "processed" / "holdout_ids.txt"

RANDOM_SEED  = 42
MIN_STRATUM  = 3    # strata smaller than this contribute 0 to holdout


def create_holdout(data_path: Path, out_path: Path, n_holdout_target: int = 20) -> None:
    df = pd.read_csv(data_path)
    # One row per experiment (4 T-snapshots per experiment → deduplicate)
    experiments = df.drop_duplicates("experiment_id")[
        ["experiment_id", "fault_type", "severity"]
    ].copy()
    n_total = len(experiments)
    logger.info("Total experiments: %d  (target holdout: %d)", n_total, n_holdout_target)

    fraction = n_holdout_target / n_total

    # ── Proportional allocation per fault_type × severity stratum ─────────────
    experiments["stratum"] = experiments["fault_type"] + "_" + experiments["severity"]
    strata = experiments.groupby("stratum")

    allocation: dict[str, int] = {}
    for name, grp in strata:
        n = len(grp)
        if n < MIN_STRATUM:
            # Too few to safely split; keep all in training
            allocation[name] = 0
        else:
            allocation[name] = max(1, round(n * fraction))

    raw_total = sum(allocation.values())

    # ── Adjust allocation to hit exactly n_holdout_target ────────────────────
    # Add/remove 1 from the largest eligible strata until total matches
    diff = n_holdout_target - raw_total
    eligible = sorted(
        [k for k, v in allocation.items() if v > 0],
        key=lambda k: len(strata.get_group(k)),
        reverse=(diff > 0),   # add to large strata, remove from large strata
    )
    for k in eligible:
        if diff == 0:
            break
        grp_size = len(strata.get_group(k))
        if diff > 0 and allocation[k] < grp_size - 1:
            allocation[k] += 1
            diff -= 1
        elif diff < 0 and allocation[k] > 1:
            allocation[k] -= 1
            diff += 1

    # ── Sample from each stratum ──────────────────────────────────────────────
    rng = np.random.default_rng(RANDOM_SEED)
    holdout_ids: list[str] = []

    logger.info("\nPer-stratum allocation (fault_type × severity):")
    logger.info("  %-25s  %5s  %7s  %8s", "stratum", "total", "holdout", "training")
    logger.info("  " + "-" * 55)
    for name, grp in sorted(strata, key=lambda x: x[0]):
        n      = len(grp)
        n_pick = allocation[name]
        if n_pick > 0:
            picked = rng.choice(grp["experiment_id"].values, size=n_pick, replace=False)
            holdout_ids.extend(picked.tolist())
        logger.info("  %-25s  %5d  %7d  %8d", name, n, n_pick, n - n_pick)

    actual_holdout = len(holdout_ids)
    actual_train   = n_total - actual_holdout
    logger.info("  " + "-" * 55)
    logger.info("  %-25s  %5d  %7d  %8d", "TOTAL", n_total, actual_holdout, actual_train)

    # ── Severity check ────────────────────────────────────────────────────────
    holdout_mask = experiments["experiment_id"].isin(holdout_ids)
    logger.info("\nHoldout severity:  %s",
                experiments[holdout_mask]["severity"].value_counts().to_dict())
    logger.info("Training severity: %s",
                experiments[~holdout_mask]["severity"].value_counts().to_dict())

    # ── Save ──────────────────────────────────────────────────────────────────
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(sorted(holdout_ids)) + "\n")
    logger.info("\nHoldout IDs (%d) → %s", actual_holdout, out_path)


if __name__ == "__main__":
    if HOLDOUT_OUT.exists():
        logger.warning(
            "%s already exists — delete it first if you want a new split.\n"
            "Keeping the existing split to ensure reproducibility.", HOLDOUT_OUT
        )
    else:
        parser = argparse.ArgumentParser()
        parser.add_argument("--data",   type=Path, default=DEFAULT_DATA)
        parser.add_argument("--target", type=int,  default=20,
                            help="Number of holdout experiments (default: 20)")
        args = parser.parse_args()
        create_holdout(args.data, HOLDOUT_OUT, n_holdout_target=args.target)
