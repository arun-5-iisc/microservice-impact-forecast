"""
Automated batch runner for all 90 experiments.

Runs each experiment sequentially and persists state after every run so the batch
can be safely interrupted and resumed.  Failed experiments are logged and skipped;
run with --reset-failed to re-queue them.

Usage:
  python scripts/run_batch.py                # run all pending experiments
  python scripts/run_batch.py --dry-run      # print plan without executing
  python scripts/run_batch.py --reset-failed # re-queue failed → pending, then run
  python scripts/run_batch.py --status       # print counts and exit
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

REPO_ROOT = Path(__file__).parent.parent
STATE_FILE = REPO_ROOT / "dataset" / "batch_state.json"
DATASET_RAW = REPO_ROOT / "dataset" / "raw"

COOLDOWN_SECONDS = 30   # wait between experiments for the cluster to stabilise

# ─────────────────────────────────────────────────────────────────────────────
# Full experiment matrix — 90 experiments × 4 time snapshots = ~360 training rows
# ─────────────────────────────────────────────────────────────────────────────

# CPU faults: reduce limit only (request unchanged)
# Services with request=100m limit=200m → Low=150m, Med=125m, High=100m
_CPU_100M_SERVICES = [
    "recommendationservice", "productcatalogservice", "checkoutservice",
    "currencyservice", "emailservice", "frontend", "paymentservice", "shippingservice",
]
# Services with request=200m limit=300m → Low=275m, Med=250m, High=200m
_CPU_200M_SERVICES = ["adservice", "cartservice"]

# Memory faults: reduce limit only (request unchanged)
# Values chosen so limit stays ≥ original request for each service
_MEMORY_SERVICES = {
    "recommendationservice": ["337Mi", "225Mi", "220Mi"],  # req=220Mi lim=450Mi
    "productcatalogservice": ["100Mi",  "80Mi",  "64Mi"],  # req=64Mi  lim=128Mi
    "checkoutservice":       ["100Mi",  "80Mi",  "64Mi"],  # req=64Mi  lim=128Mi
    "frontend":              ["100Mi",  "80Mi",  "64Mi"],  # req=64Mi  lim=128Mi
    "cartservice":           ["192Mi", "160Mi", "128Mi"],  # req=128Mi lim=256Mi
    "adservice":             ["225Mi", "200Mi", "180Mi"],  # req=180Mi lim=300Mi
}

# Replica faults: scale to 0 (complete unavailability)
_REPLICA_SERVICES = [
    "recommendationservice", "productcatalogservice", "checkoutservice",
    "frontend", "cartservice", "paymentservice", "shippingservice", "currencyservice",
]

# Latency faults: Chaos Mesh NetworkChaos (all services, any latency value)
_LATENCY_SERVICES = [
    "checkoutservice", "frontend", "recommendationservice",
    "productcatalogservice", "cartservice",
]
_LATENCY_VALUES = ["100", "200", "500", "1000"]  # milliseconds

# High traffic variants for latency (selected services + higher values only)
_LATENCY_HIGH_TRAFFIC_SERVICES = ["checkoutservice", "frontend", "productcatalogservice"]
_LATENCY_HIGH_TRAFFIC_VALUES = ["500", "1000"]

# High traffic variants for CPU / replica faults
_CPU_HIGH_TRAFFIC = [
    ("recommendationservice", "125m"),
    ("recommendationservice", "100m"),
    ("productcatalogservice", "125m"),
    ("checkoutservice",       "100m"),
    ("frontend",              "100m"),
]
_REPLICA_HIGH_TRAFFIC = [
    "recommendationservice", "productcatalogservice", "checkoutservice",
]


def _build_experiment_list():
    experiments = []

    def add(service, fault, value, traffic="normal"):
        experiments.append({
            "key":     f"{service}|{fault}|{value}|{traffic}",
            "service": service,
            "fault":   fault,
            "value":   value,
            "traffic": traffic,
        })

    # CPU — normal traffic
    for svc in _CPU_100M_SERVICES:
        for val in ["150m", "125m", "100m"]:
            add(svc, "cpu", val)
    for svc in _CPU_200M_SERVICES:
        for val in ["275m", "250m", "200m"]:
            add(svc, "cpu", val)

    # Memory — normal traffic
    for svc, values in _MEMORY_SERVICES.items():
        for val in values:
            add(svc, "memory", val)

    # Replicas — normal traffic
    for svc in _REPLICA_SERVICES:
        add(svc, "replicas", "0")

    # Latency — normal traffic
    for svc in _LATENCY_SERVICES:
        for val in _LATENCY_VALUES:
            add(svc, "latency", val)

    # Latency — high traffic
    for svc in _LATENCY_HIGH_TRAFFIC_SERVICES:
        for val in _LATENCY_HIGH_TRAFFIC_VALUES:
            add(svc, "latency", val, traffic="high")

    # CPU — high traffic
    for svc, val in _CPU_HIGH_TRAFFIC:
        add(svc, "cpu", val, traffic="high")

    # Replicas — high traffic
    for svc in _REPLICA_HIGH_TRAFFIC:
        add(svc, "replicas", "0", traffic="high")

    return experiments


EXPERIMENTS = _build_experiment_list()


# ─────────────────────────────────────────────────────────────────────────────
# State management
# ─────────────────────────────────────────────────────────────────────────────

def _load_state():
    if STATE_FILE.exists():
        with open(STATE_FILE) as f:
            return json.load(f)
    planned = [{"key": e["key"], "status": "pending"} for e in EXPERIMENTS]
    return {"version": 1, "experiments": planned}


def _save_state(state):
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    tmp = STATE_FILE.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(state, f, indent=2)
    tmp.replace(STATE_FILE)   # atomic rename — safe if process is killed mid-write


def _merge_state(state):
    """Add any new experiments from EXPERIMENTS that aren't in the state file yet."""
    existing_keys = {e["key"] for e in state["experiments"]}
    for exp in EXPERIMENTS:
        if exp["key"] not in existing_keys:
            state["experiments"].append({"key": exp["key"], "status": "pending"})
    return state


def _lookup(key):
    """Return the experiment definition for a given key."""
    return next(e for e in EXPERIMENTS if e["key"] == key)


# ─────────────────────────────────────────────────────────────────────────────
# GCS upload placeholder
# ─────────────────────────────────────────────────────────────────────────────

def _upload_to_gcs(exp_dir: Path) -> None:
    pass
    # TODO: set GCS_BUCKET env var and uncomment:
    # import os, subprocess
    # bucket = os.environ.get("GCS_BUCKET")
    # if bucket:
    #     subprocess.run(
    #         ["gsutil", "-m", "cp", "-r", str(exp_dir), f"gs://{bucket}/raw/"],
    #         check=True,
    #     )


# ─────────────────────────────────────────────────────────────────────────────
# Main batch logic
# ─────────────────────────────────────────────────────────────────────────────

def run_batch(dry_run=False):
    state = _load_state()
    state = _merge_state(state)
    _save_state(state)

    entries = state["experiments"]
    pending  = [e for e in entries if e["status"] == "pending"]
    done     = [e for e in entries if e["status"] == "done"]
    failed   = [e for e in entries if e["status"] == "failed"]

    logger.info(
        "Batch state: total=%d  done=%d  pending=%d  failed=%d",
        len(entries), len(done), len(pending), len(failed),
    )

    if not pending:
        logger.info("Nothing to run.")
        return

    for i, entry in enumerate(pending, 1):
        key = entry["key"]
        exp = _lookup(key)
        logger.info(
            "─" * 60 + "\n[%d/%d] %s", i, len(pending), key
        )

        if dry_run:
            logger.info("DRY RUN — would run: service=%s fault=%s value=%s traffic=%s",
                        exp["service"], exp["fault"], exp["value"], exp["traffic"])
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
            entry["error"] = str(exc)
            logger.error("FAILED  key=%s  error=%s", key, exc)
        finally:
            _save_state(state)

        if i < len(pending):
            logger.info("Cooldown %ds before next experiment…", COOLDOWN_SECONDS)
            time.sleep(COOLDOWN_SECONDS)

    done_now  = sum(1 for e in state["experiments"] if e["status"] == "done")
    failed_now = sum(1 for e in state["experiments"] if e["status"] == "failed")
    logger.info("Batch complete.  done=%d  failed=%d", done_now, failed_now)
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


def reset_failed():
    state = _load_state()
    count = 0
    for e in state["experiments"]:
        if e["status"] == "failed":
            e["status"] = "pending"
            e.pop("error", None)
            count += 1
    _save_state(state)
    logger.info("Reset %d failed experiment(s) to pending.", count)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run all experiments in batch")
    parser.add_argument("--dry-run",       action="store_true", help="Print plan without running")
    parser.add_argument("--status",        action="store_true", help="Print counts and exit")
    parser.add_argument("--reset-failed",  action="store_true", help="Re-queue failed experiments")
    args = parser.parse_args()

    if args.status:
        print_status()
    elif args.reset_failed:
        reset_failed()
        run_batch()
    else:
        run_batch(dry_run=args.dry_run)
