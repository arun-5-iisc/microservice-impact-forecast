"""
Draws the train/holdout split by experiment setting (service x fault x value x traffic): all runs of a
held-out setting go to the holdout.

About 20% of the settings are held out, stratified by fault type x the severity of each setting's first
run; strata with fewer than 3 settings stay in training.

dataset/processed/holdout_ids.txt was drawn once, with seed 42, before any model was trained, and is never
redrawn. It was drawn on the midterm labels (scripts/analysis/midterm_labels.csv), so --check below can
confirm it; drawing from the final labels would give a different split.

Usage:
  python scripts/create_holdout.py      # draws only if holdout_ids.txt doesn't exist yet
  python scripts/create_holdout.py --data scripts/analysis/midterm_labels.csv --check
"""
import argparse
import logging
import sys
import tempfile
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
MIN_STRATUM  = 3    # strata with fewer settings than this contribute 0 to holdout


def create_holdout(data_path: Path, out_path: Path, fraction: float = 0.2) -> None:
    df = pd.read_csv(data_path)
    # One row per experiment (4 T-snapshots per experiment -> deduplicate)
    runs = df.drop_duplicates("experiment_id")[
        ["experiment_id", "service", "fault_type", "fault_value", "traffic_level", "severity"]
    ].sort_values("experiment_id").copy()
    runs["setting"] = (runs["service"] + "|" + runs["fault_type"] + "|"
                       + runs["fault_value"].astype(str) + "|" + runs["traffic_level"])

    # One row per setting; a repeated setting is stratified by its first run's severity
    settings = runs.groupby("setting", sort=True).agg(
        fault_type=("fault_type", "first"), severity=("severity", "first"),
        n_runs=("experiment_id", "size"),
    ).reset_index()
    n_total = len(settings)
    n_holdout_target = round(n_total * fraction)
    logger.info("Experiments: %d  settings: %d (%d repeated)  target holdout: %d settings (%.0f%%)",
                len(runs), n_total, (settings["n_runs"] > 1).sum(), n_holdout_target, 100 * fraction)

    # Proportional allocation per fault_type x severity stratum
    settings["stratum"] = settings["fault_type"] + "_" + settings["severity"]
    strata = settings.groupby("stratum")

    allocation: dict[str, int] = {}
    for name, grp in strata:
        n = len(grp)
        if n < MIN_STRATUM:
            # Too few to safely split; keep all in training
            allocation[name] = 0
        else:
            allocation[name] = max(1, round(n * fraction))

    raw_total = sum(allocation.values())

    # rounding can miss the target by a few settings: add to the largest strata first,
    # or take away from the smallest first, one at a time
    diff = n_holdout_target - raw_total
    eligible = sorted(
        [k for k, v in allocation.items() if v > 0],
        key=lambda k: len(strata.get_group(k)),
        reverse=(diff > 0),
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

    # Sample settings from each stratum
    rng = np.random.default_rng(RANDOM_SEED)
    holdout_settings: list[str] = []

    logger.info("\nPer-stratum allocation of settings (fault_type x severity):")
    logger.info("  %-25s  %5s  %7s  %8s", "stratum", "total", "holdout", "training")
    logger.info("  " + "-" * 55)
    for name, grp in sorted(strata, key=lambda x: x[0]):
        n      = len(grp)
        n_pick = allocation[name]
        if n_pick > 0:
            picked = rng.choice(grp["setting"].values, size=n_pick, replace=False)
            holdout_settings.extend(picked.tolist())
        logger.info("  %-25s  %5d  %7d  %8d", name, n, n_pick, n - n_pick)

    logger.info("  " + "-" * 55)
    logger.info("  %-25s  %5d  %7d  %8d", "TOTAL", n_total, len(holdout_settings),
                n_total - len(holdout_settings))

    # Every run of a held-out setting goes to the holdout
    holdout_mask = runs["setting"].isin(holdout_settings)
    holdout_ids  = runs.loc[holdout_mask, "experiment_id"].tolist()
    logger.info("\nHoldout runs: %d   training runs: %d", holdout_mask.sum(), (~holdout_mask).sum())
    logger.info("Holdout severity:  %s", runs[holdout_mask]["severity"].value_counts().to_dict())
    logger.info("Training severity: %s", runs[~holdout_mask]["severity"].value_counts().to_dict())

    # Save
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(sorted(holdout_ids)) + "\n")
    logger.info("\nHoldout IDs (%d) -> %s", len(holdout_ids), out_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Draw the train/holdout split (once).")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--fraction", type=float, default=0.2,
                        help="share of experiment settings to hold out (default: 0.2)")
    parser.add_argument("--check", action="store_true",
                        help="draw again into a temporary file and compare it with the saved holdout_ids.txt")
    args = parser.parse_args()

    if args.check:
        with tempfile.TemporaryDirectory() as tmp:
            redrawn = Path(tmp) / "holdout_ids.txt"
            create_holdout(args.data, redrawn, fraction=args.fraction)
            same = redrawn.read_text() == HOLDOUT_OUT.read_text()
        print(f"Holdout redrawn from {args.data}: {'identical to' if same else 'DIFFERENT from'} {HOLDOUT_OUT}")
        sys.exit(0 if same else 1)
    if HOLDOUT_OUT.exists():
        logger.warning("%s already exists, keeping it (delete it first if you really want a new split).",
                       HOLDOUT_OUT)
    else:
        create_holdout(args.data, HOLDOUT_OUT, fraction=args.fraction)
