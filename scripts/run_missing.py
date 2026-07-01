"""
Runs the 8 missing experiments that fill service-fault coverage gaps.

Gaps identified from coverage analysis:
  adservice       — replicas
  emailservice    — replicas
  paymentservice  — latency (500ms, 1000ms)
  shippingservice — latency (500ms, 1000ms)
  currencyservice — latency (200ms, 500ms)

State is persisted to dataset/missing_state.json after every experiment so the
script can be safely interrupted and resumed.

Usage:
  python scripts/run_missing.py               # run all pending
  python scripts/run_missing.py --dry-run     # preview without executing
  python scripts/run_missing.py --status      # show counts and exit
  python scripts/run_missing.py --reset-failed  # re-queue failed, then run
"""
import argparse
import json
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from run_experiment import run_experiment

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger(__name__)

REPO_ROOT   = Path(__file__).parent.parent
STATE_FILE  = REPO_ROOT / "dataset" / "missing_state.json"
DATASET_RAW = REPO_ROOT / "dataset" / "raw"
GCS_BUCKET  = "ff-dataset-bucket"
COOLDOWN_SECONDS = 30

# ─────────────────────────────────────────────────────────────────────────────
# The 8 missing experiments
# ─────────────────────────────────────────────────────────────────────────────

MISSING_EXPERIMENTS = [
    {"service": "adservice",       "fault": "replicas", "value": "0",    "traffic": "normal"},
    {"service": "emailservice",    "fault": "replicas", "value": "0",    "traffic": "normal"},
    {"service": "paymentservice",  "fault": "latency",  "value": "500",  "traffic": "normal"},
    {"service": "paymentservice",  "fault": "latency",  "value": "1000", "traffic": "normal"},
    {"service": "shippingservice", "fault": "latency",  "value": "500",  "traffic": "normal"},
    {"service": "shippingservice", "fault": "latency",  "value": "1000", "traffic": "normal"},
    {"service": "currencyservice", "fault": "latency",  "value": "200",  "traffic": "normal"},
    {"service": "currencyservice", "fault": "latency",  "value": "500",  "traffic": "normal"},
]

for e in MISSING_EXPERIMENTS:
    e["key"] = f"{e['service']}|{e['fault']}|{e['value']}|{e['traffic']}"


# ─────────────────────────────────────────────────────────────────────────────
# State helpers
# ─────────────────────────────────────────────────────────────────────────────

def _load_state():
    if STATE_FILE.exists():
        with open(STATE_FILE) as f:
            state = json.load(f)
        existing_keys = {e["key"] for e in state["experiments"]}
        for exp in MISSING_EXPERIMENTS:
            if exp["key"] not in existing_keys:
                state["experiments"].append({"key": exp["key"], "status": "pending"})
        return state
    return {
        "version": 1,
        "experiments": [{"key": e["key"], "status": "pending"} for e in MISSING_EXPERIMENTS],
    }


def _save_state(state):
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    tmp = STATE_FILE.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(state, f, indent=2)
    tmp.replace(STATE_FILE)


def _lookup(key):
    return next(e for e in MISSING_EXPERIMENTS if e["key"] == key)


# ─────────────────────────────────────────────────────────────────────────────
# GCS upload
# ─────────────────────────────────────────────────────────────────────────────

def _upload_to_gcs(exp_dir: Path) -> None:
    import subprocess
    logger.info("Uploading %s → gs://%s/raw/", exp_dir.name, GCS_BUCKET)
    subprocess.run(
        ["gcloud", "storage", "cp", "-r", str(exp_dir), f"gs://{GCS_BUCKET}/raw/"],
        check=True,
    )
    logger.info("Upload complete: gs://%s/raw/%s", GCS_BUCKET, exp_dir.name)


# ─────────────────────────────────────────────────────────────────────────────
# Batch logic
# ─────────────────────────────────────────────────────────────────────────────

def run_batch(dry_run=False):
    state = _load_state()
    _save_state(state)

    entries = state["experiments"]
    pending = [e for e in entries if e["status"] == "pending"]
    done    = [e for e in entries if e["status"] == "done"]
    failed  = [e for e in entries if e["status"] == "failed"]

    logger.info(
        "Missing-experiments state: total=%d  done=%d  pending=%d  failed=%d",
        len(entries), len(done), len(pending), len(failed),
    )

    if not pending:
        logger.info("All missing experiments already complete.")
        return

    for i, entry in enumerate(pending, 1):
        key = entry["key"]
        exp = _lookup(key)
        logger.info("─" * 60 + "\n[%d/%d] %s", i, len(pending), key)

        if dry_run:
            logger.info(
                "DRY RUN — service=%s fault=%s value=%s traffic=%s",
                exp["service"], exp["fault"], exp["value"], exp["traffic"],
            )
            continue

        try:
            exp_id = run_experiment(
                exp["service"], exp["fault"], exp["value"], exp["traffic"]
            )
            entry["status"] = "done"
            entry["exp_id"] = exp_id
            logger.info("Completed  key=%s  exp_id=%s", key, exp_id)
            _upload_to_gcs(DATASET_RAW / exp_id)
        except Exception as exc:
            entry["status"] = "failed"
            entry["error"]  = str(exc)
            logger.error("FAILED  key=%s  error=%s", key, exc)
        finally:
            _save_state(state)

        if i < len(pending):
            logger.info("Cooldown %ds…", COOLDOWN_SECONDS)
            time.sleep(COOLDOWN_SECONDS)

    done_now   = sum(1 for e in state["experiments"] if e["status"] == "done")
    failed_now = sum(1 for e in state["experiments"] if e["status"] == "failed")
    logger.info("Done.  completed=%d  failed=%d", done_now, failed_now)
    if failed_now:
        logger.info("Re-run with --reset-failed to retry failed experiments.")


def print_status():
    state = _load_state()
    entries = state["experiments"]
    done    = [e for e in entries if e["status"] == "done"]
    pending = [e for e in entries if e["status"] == "pending"]
    failed  = [e for e in entries if e["status"] == "failed"]
    print(f"Total:   {len(entries)}")
    print(f"Done:    {len(done)}")
    print(f"Pending: {len(pending)}")
    print(f"Failed:  {len(failed)}")
    if failed:
        print("\nFailed experiments:")
        for e in failed:
            print(f"  {e['key']}  →  {e.get('error', '?')}")


def reset_failed(then_run=True):
    state = _load_state()
    count = 0
    for e in state["experiments"]:
        if e["status"] == "failed":
            e["status"] = "pending"
            count += 1
    _save_state(state)
    logger.info("Reset %d failed experiment(s) → pending", count)
    if then_run:
        run_batch()


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run the 8 missing coverage-gap experiments"
    )
    parser.add_argument("--dry-run",      action="store_true", help="Print plan without executing")
    parser.add_argument("--status",       action="store_true", help="Show counts and exit")
    parser.add_argument("--reset-failed", action="store_true", help="Re-queue failed experiments, then run")
    args = parser.parse_args()

    if args.status:
        print_status()
    elif args.reset_failed:
        reset_failed(then_run=True)
    else:
        run_batch(dry_run=args.dry_run)
