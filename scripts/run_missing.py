"""
Runs the 40 missing experiments that complete the service-fault coverage matrix.

Original 90-experiment batch covered CPU for all 10 services but left gaps in
memory, replicas and latency for 4-5 services.  This script fills every gap so
every service has at least one experiment per fault type.

Gaps filled:
  Memory    — currencyservice, emailservice, paymentservice, shippingservice (12)
  Replicas  — adservice, emailservice                                         (2)
  Latency   — adservice, currencyservice, emailservice, paymentservice,
               shippingservice × 4 values normal                             (20)
  Latency   — paymentservice, shippingservice, currencyservice × high traffic  (6)

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
    # ── Memory gaps (4 services not in original _MEMORY_SERVICES) ────────────
    # Values calibrated the same way as run_batch.py: limit ≥ request, 3 severity levels
    # currencyservice: req=128Mi lim=256Mi  →  same pattern as cartservice
    {"service": "currencyservice", "fault": "memory", "value": "192Mi", "traffic": "normal"},
    {"service": "currencyservice", "fault": "memory", "value": "160Mi", "traffic": "normal"},
    {"service": "currencyservice", "fault": "memory", "value": "128Mi", "traffic": "normal"},
    # emailservice: req=64Mi lim=128Mi  →  same pattern as checkoutservice
    {"service": "emailservice",    "fault": "memory", "value": "100Mi", "traffic": "normal"},
    {"service": "emailservice",    "fault": "memory", "value": "80Mi",  "traffic": "normal"},
    {"service": "emailservice",    "fault": "memory", "value": "64Mi",  "traffic": "normal"},
    # paymentservice: req=128Mi lim=256Mi  →  same pattern as cartservice
    {"service": "paymentservice",  "fault": "memory", "value": "192Mi", "traffic": "normal"},
    {"service": "paymentservice",  "fault": "memory", "value": "160Mi", "traffic": "normal"},
    {"service": "paymentservice",  "fault": "memory", "value": "128Mi", "traffic": "normal"},
    # shippingservice: req=64Mi lim=128Mi  →  same pattern as checkoutservice
    {"service": "shippingservice", "fault": "memory", "value": "100Mi", "traffic": "normal"},
    {"service": "shippingservice", "fault": "memory", "value": "80Mi",  "traffic": "normal"},
    {"service": "shippingservice", "fault": "memory", "value": "64Mi",  "traffic": "normal"},

    # ── Replica gaps (2 services not in original _REPLICA_SERVICES) ──────────
    {"service": "adservice",       "fault": "replicas", "value": "0", "traffic": "normal"},
    {"service": "emailservice",    "fault": "replicas", "value": "0", "traffic": "normal"},

    # ── Latency gaps — normal traffic (5 services not in original _LATENCY_SERVICES)
    {"service": "adservice",       "fault": "latency", "value": "100",  "traffic": "normal"},
    {"service": "adservice",       "fault": "latency", "value": "200",  "traffic": "normal"},
    {"service": "adservice",       "fault": "latency", "value": "500",  "traffic": "normal"},
    {"service": "adservice",       "fault": "latency", "value": "1000", "traffic": "normal"},

    {"service": "currencyservice", "fault": "latency", "value": "100",  "traffic": "normal"},
    {"service": "currencyservice", "fault": "latency", "value": "200",  "traffic": "normal"},
    {"service": "currencyservice", "fault": "latency", "value": "500",  "traffic": "normal"},
    {"service": "currencyservice", "fault": "latency", "value": "1000", "traffic": "normal"},

    {"service": "emailservice",    "fault": "latency", "value": "100",  "traffic": "normal"},
    {"service": "emailservice",    "fault": "latency", "value": "200",  "traffic": "normal"},
    {"service": "emailservice",    "fault": "latency", "value": "500",  "traffic": "normal"},
    {"service": "emailservice",    "fault": "latency", "value": "1000", "traffic": "normal"},

    {"service": "paymentservice",  "fault": "latency", "value": "100",  "traffic": "normal"},
    {"service": "paymentservice",  "fault": "latency", "value": "200",  "traffic": "normal"},
    {"service": "paymentservice",  "fault": "latency", "value": "500",  "traffic": "normal"},
    {"service": "paymentservice",  "fault": "latency", "value": "1000", "traffic": "normal"},

    {"service": "shippingservice", "fault": "latency", "value": "100",  "traffic": "normal"},
    {"service": "shippingservice", "fault": "latency", "value": "200",  "traffic": "normal"},
    {"service": "shippingservice", "fault": "latency", "value": "500",  "traffic": "normal"},
    {"service": "shippingservice", "fault": "latency", "value": "1000", "traffic": "normal"},

    # ── Latency gaps — high traffic (checkout-path services, mirrors original pattern)
    {"service": "paymentservice",  "fault": "latency", "value": "500",  "traffic": "high"},
    {"service": "paymentservice",  "fault": "latency", "value": "1000", "traffic": "high"},
    {"service": "shippingservice", "fault": "latency", "value": "500",  "traffic": "high"},
    {"service": "shippingservice", "fault": "latency", "value": "1000", "traffic": "high"},
    {"service": "currencyservice", "fault": "latency", "value": "500",  "traffic": "high"},
    {"service": "currencyservice", "fault": "latency", "value": "1000", "traffic": "high"},
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
