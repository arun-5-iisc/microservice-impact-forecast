"""
Run the experiments in a plan file, one after another.

A plan is a CSV in experiments/plans/ with columns service, fault, value, traffic, run, group,
purpose. `run` tells repeated runs of the same setting apart. Progress is saved to
dataset/<plan name>_state.json after every experiment, so a batch can be stopped and resumed
by running the same command again. Failed experiments are skipped; --reset-failed queues them
again.

Before each experiment the script checks that the cluster is back at its baseline (Helm values
equal deployment/online-boutique/values.yaml, no NetworkChaos left, every service at 1
replica), and stops if not. It also stops if gcloud or kubectl credentials have expired, so a
fault is never applied without a way to undo it. A failed GCS upload doesn't fail the
experiment; it is retried on the next run.

Usage:
  python scripts/run_batch.py --plan experiments/plans/phase3_targeted.csv --dry-run
  python scripts/run_batch.py --plan experiments/plans/phase3_targeted.csv
  python scripts/run_batch.py --plan experiments/plans/phase3_targeted.csv --limit 3
  python scripts/run_batch.py --plan experiments/plans/phase3_targeted.csv --status
  python scripts/run_batch.py --plan experiments/plans/phase3_targeted.csv --reset-failed
"""
import argparse
import csv
import json
import logging
import signal
import subprocess
import sys
import time
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).parent))

from deployment_manager import HELM_RELEASE, NAMESPACE
from run_experiment import run_experiment

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).parent.parent
DATASET_RAW = REPO_ROOT / "dataset" / "raw"
VALUES_FILE = REPO_ROOT / "deployment" / "online-boutique" / "values.yaml"
GCS_BUCKET = "ff-dataset-bucket"
COOLDOWN_SECONDS = 30   # pause between experiments so the cluster settles

SERVICES = {
    "adservice", "cartservice", "checkoutservice", "currencyservice", "emailservice",
    "frontend", "paymentservice", "productcatalogservice", "recommendationservice",
    "shippingservice",
}
FAULTS = {"cpu", "memory", "replicas", "latency"}
TRAFFIC = {"normal", "high"}

# set from --plan in main
EXPERIMENTS = []
STATE_FILE = None


def load_plan(path: Path) -> list:
    """Read a plan CSV and check every row before anything runs."""
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))

    experiments = []
    for line, row in enumerate(rows, start=2):   # line 1 is the header
        if row["service"] not in SERVICES or row["fault"] not in FAULTS or row["traffic"] not in TRAFFIC:
            raise ValueError(f"{path}:{line}: unknown service/fault/traffic in {row}")
        run = int(row.get("run") or 1)
        experiments.append({
            "key":     f"{row['service']}|{row['fault']}|{row['value']}|{row['traffic']}|r{run}",
            "service": row["service"],
            "fault":   row["fault"],
            "value":   row["value"],
            "traffic": row["traffic"],
        })

    keys = [e["key"] for e in experiments]
    duplicates = sorted({k for k in keys if keys.count(k) > 1})
    if duplicates:
        raise ValueError(f"{path}: duplicate rows {duplicates}; give repeats different run numbers")
    return experiments


class AuthError(RuntimeError):
    """gcloud or kubectl credentials are missing or expired."""


# kubectl also fails for network reasons, so only these messages count as a credentials problem
_KUBECTL_AUTH_MARKERS = (
    "unauthorized", "provide credentials", "getting credentials",
    "gke-gcloud-auth-plugin", "reauthentication", "auth tokens", "gcloud auth login",
)


def check_auth() -> None:
    gcloud = subprocess.run(["gcloud", "auth", "print-access-token", "--quiet"],
                            capture_output=True, text=True)
    if gcloud.returncode != 0:
        raise AuthError(f"gcloud: {gcloud.stderr.strip()[:500]}")

    kubectl = subprocess.run(["kubectl", "get", "namespace", NAMESPACE, "-o", "name",
                              "--request-timeout=30s"], capture_output=True, text=True)
    if kubectl.returncode != 0 and any(m in kubectl.stderr.lower() for m in _KUBECTL_AUTH_MARKERS):
        raise AuthError(f"kubectl: {kubectl.stderr.strip()[:500]}")


class ClusterNotCleanError(RuntimeError):
    """A fault from an earlier run is still in place."""


def _get_json(cmd) -> dict:
    return json.loads(subprocess.run(cmd, capture_output=True, text=True, check=True).stdout) or {}


def check_cluster_clean() -> None:
    try:
        applied = _get_json(["helm", "get", "values", HELM_RELEASE, "-n", NAMESPACE, "-o", "json"])
        chaos = _get_json(["kubectl", "get", "networkchaos", "-n", NAMESPACE, "-o", "json"])["items"]
        deployments = _get_json(["kubectl", "get", "deployments", "-n", NAMESPACE, "-o", "json"])["items"]
    except subprocess.CalledProcessError as exc:
        raise ClusterNotCleanError(f"could not check the cluster: {exc.stderr.strip()[:300]}")
    except json.JSONDecodeError as exc:
        raise ClusterNotCleanError(f"could not check the cluster: unexpected output ({exc})")

    problems = []
    expected = yaml.safe_load(VALUES_FILE.read_text()) or {}
    if applied != expected:
        problems.append(f"Helm values differ from {VALUES_FILE.name}: {json.dumps(applied)}")
    if chaos:
        problems.append("latency faults still active: " + ", ".join(c["metadata"]["name"] for c in chaos))
    scaled = [f"{d['metadata']['name']}={d['spec'].get('replicas')}"
              for d in deployments if d["spec"].get("replicas") != 1]
    if scaled:
        problems.append("services not at 1 replica: " + ", ".join(scaled))
    if problems:
        raise ClusterNotCleanError("; ".join(problems))


def load_state():
    if STATE_FILE.exists():
        with open(STATE_FILE) as f:
            state = json.load(f)
    else:
        state = {"version": 1, "experiments": []}
    # rows added to the plan since the last run are queued at the end
    known = {e["key"] for e in state["experiments"]}
    state["experiments"] += [{"key": e["key"], "status": "pending"} for e in EXPERIMENTS if e["key"] not in known]
    return state


def save_state(state):
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    tmp = STATE_FILE.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(state, f, indent=2)
    tmp.replace(STATE_FILE)   # rename, so a kill mid-write can't corrupt the state file


def upload(entry) -> None:
    """Copy a finished run to GCS. If that fails the run still counts as done and is retried later."""
    exp_dir = DATASET_RAW / entry["exp_id"]
    try:
        logger.info("Uploading %s to gs://%s/raw/", exp_dir.name, GCS_BUCKET)
        subprocess.run(["gcloud", "storage", "cp", "-r", str(exp_dir), f"gs://{GCS_BUCKET}/raw/"], check=True)
    except Exception as exc:
        entry["uploaded"] = False
        entry["upload_error"] = str(exc)
        logger.warning("Upload failed for %s (kept as done, retried next run): %s", entry["exp_id"], exc)
        check_auth()   # stop here if the upload failed because the login expired
    else:
        entry["uploaded"] = True
        entry.pop("upload_error", None)


def run_batch(dry_run=False, limit=None):
    state = load_state()
    save_state(state)

    entries = state["experiments"]
    order = {e["key"]: i for i, e in enumerate(EXPERIMENTS)}
    pending = sorted((e for e in entries if e["status"] == "pending"),
                     key=lambda e: order.get(e["key"], len(order)))
    logger.info("Batch state (%s): total=%d  done=%d  pending=%d  failed=%d", STATE_FILE.name, len(entries),
                sum(e["status"] == "done" for e in entries), len(pending),
                sum(e["status"] == "failed" for e in entries))

    if not dry_run:
        check_auth()
        for entry in entries:   # uploads that failed last time
            if entry["status"] == "done" and entry.get("uploaded") is False:
                try:
                    upload(entry)
                finally:
                    save_state(state)

    if not pending:
        logger.info("Nothing to run.")
        return

    to_run = pending[:limit] if limit is not None else pending
    by_key = {e["key"]: e for e in EXPERIMENTS}
    for i, entry in enumerate(to_run, 1):
        exp = by_key[entry["key"]]
        logger.info("-" * 60 + "\n[%d/%d] %s", i, len(to_run), entry["key"])

        if dry_run:
            logger.info("DRY RUN, would run: service=%s fault=%s value=%s traffic=%s",
                        exp["service"], exp["fault"], exp["value"], exp["traffic"])
            continue

        check_auth()
        check_cluster_clean()

        try:
            exp_id = run_experiment(exp["service"], exp["fault"], exp["value"], exp["traffic"])
        except Exception as exc:
            entry["error"] = str(exc)
            save_state(state)
            check_auth()   # if the login expired, stop and leave the entry pending
            entry["status"] = "failed"
            logger.error("FAILED  key=%s  error=%s", entry["key"], exc)
            save_state(state)
        else:
            entry["status"] = "done"
            entry["exp_id"] = exp_id
            entry.pop("error", None)
            logger.info("Completed  key=%s  exp_id=%s", entry["key"], exp_id)
            save_state(state)
            try:
                upload(entry)
            finally:
                save_state(state)

        if i < len(to_run):
            logger.info("Cooldown %ds before the next experiment", COOLDOWN_SECONDS)
            time.sleep(COOLDOWN_SECONDS)

    counts = {s: sum(e["status"] == s for e in state["experiments"]) for s in ("done", "failed", "pending")}
    logger.info("Batch complete.  done=%d  failed=%d  pending=%d", counts["done"], counts["failed"], counts["pending"])
    if counts["failed"]:
        logger.info("Re-run with --reset-failed to retry the failed experiments.")
    if counts["pending"] and not dry_run:
        logger.info("%d experiment(s) still pending; run the same command again to continue.", counts["pending"])


def print_status():
    entries = load_state()["experiments"]
    done = [e for e in entries if e["status"] == "done"]
    failed = [e for e in entries if e["status"] == "failed"]
    print(f"Total:   {len(entries)}")
    print(f"Done:    {len(done)}")
    print(f"Pending: {sum(e['status'] == 'pending' for e in entries)}")
    print(f"Failed:  {len(failed)}")
    not_uploaded = [e for e in done if e.get("uploaded") is False]
    if not_uploaded:
        print(f"Done but not yet uploaded to GCS: {len(not_uploaded)} (retried on the next run)")
    for e in failed:
        print(f"  failed: {e['key']}: {e.get('error', '?')}")


def reset_failed():
    state = load_state()
    count = 0
    for e in state["experiments"]:
        if e["status"] == "failed":
            e["status"] = "pending"
            e.pop("error", None)
            count += 1
    save_state(state)
    logger.info("Reset %d failed experiment(s) to pending.", count)


def _raise_keyboard_interrupt(signum, frame):
    raise KeyboardInterrupt(f"received signal {signum}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the experiments in a plan file")
    parser.add_argument("--plan", type=Path, required=True, help="plan CSV in experiments/plans/")
    parser.add_argument("--dry-run", action="store_true", help="list what would run")
    parser.add_argument("--status", action="store_true", help="print counts and exit")
    parser.add_argument("--reset-failed", action="store_true", help="queue failed experiments again")
    parser.add_argument("--limit", type=int, help="run at most this many pending experiments")
    args = parser.parse_args()

    EXPERIMENTS = load_plan(args.plan)
    STATE_FILE = REPO_ROOT / "dataset" / f"{args.plan.stem}_state.json"

    # treat a kill or a closed tmux session like Ctrl+C, so the fault in progress gets restored
    signal.signal(signal.SIGTERM, _raise_keyboard_interrupt)
    signal.signal(signal.SIGHUP, _raise_keyboard_interrupt)

    try:
        if args.status:
            print_status()
        elif args.reset_failed:
            reset_failed()
            run_batch(limit=args.limit)
        else:
            run_batch(dry_run=args.dry_run, limit=args.limit)
    except AuthError as exc:
        logger.error("Batch stopped, authentication failed: %s", exc)
        logger.error("Run `gcloud auth login`, then the same command again; it checks the cluster "
                     "is clean before continuing.")
        sys.exit(2)
    except ClusterNotCleanError as exc:
        logger.error("Batch stopped, the cluster is not at its baseline: %s", exc)
        logger.error("Fix it (Helm values: `bash deployment/online-boutique/install.sh`; latency faults: "
                     "`kubectl delete networkchaos --all -n %s`; replicas: "
                     "`kubectl scale deploy/<name> --replicas=1 -n %s`), then run the same command again.",
                     NAMESPACE, NAMESPACE)
        sys.exit(3)
    except KeyboardInterrupt:
        logger.error("Batch interrupted; run the same command again to continue. It checks the cluster "
                     "is clean before starting.")
        sys.exit(130)
