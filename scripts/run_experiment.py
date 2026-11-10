"""
Runs one experiment:
  1. 60 s baseline
  2. apply the change (fault) and hold it for 180 s
  3. undo it, then 60 s of recovery
  4. collect Prometheus metrics, Jaeger traces and Kubernetes events for the whole run
  5. save them, with metadata.json, to dataset/raw/EXP_<n>/

Applying and undoing a change take a few seconds each, so the recorded fault phase is about
186 s and the recovery about 78 s; metadata.json has the exact phase times.

Usage:
  python run_experiment.py --service recommendationservice --fault cpu --value 50m
  python run_experiment.py --service productcatalogservice --fault latency --value 200
  python run_experiment.py --service cartservice --fault replicas --value 0
  python run_experiment.py --service emailservice --fault memory --value 64Mi
"""
import argparse
import json
import logging
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from deployment_manager import apply_fault, restore_deployment
from jaeger_collector import JaegerCollector
from kubernetes_collector import KubernetesCollector
from prometheus_collector import PrometheusCollector
from traffic_manager import restore_traffic, set_traffic

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).parent.parent
DATASET_RAW = REPO_ROOT / "dataset" / "raw"
DATASET_EXCLUDED = REPO_ROOT / "dataset" / "excluded"   # runs set aside; their IDs stay taken

BASELINE_DURATION = 60    # seconds
FAULT_DURATION = 180      # seconds of observation after fault is applied
RECOVERY_DURATION = 60    # seconds of observation after restoration


def next_experiment_id() -> str:
    DATASET_RAW.mkdir(parents=True, exist_ok=True)
    existing = sorted(
        d.name for root in (DATASET_RAW, DATASET_EXCLUDED) if root.exists()
        for d in root.iterdir()
        if d.is_dir() and d.name.startswith("EXP_")
    )
    if not existing:
        return "EXP_0001"
    last_num = int(existing[-1].split("_")[1])
    return f"EXP_{last_num + 1:04d}"


def _cluster_info() -> dict:
    """Cluster the experiment runs on - the dataset spans clusters with different node counts."""
    context = subprocess.run(
        ["kubectl", "config", "current-context"],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    nodes = json.loads(subprocess.run(
        ["kubectl", "get", "nodes", "-o", "json"],
        capture_output=True, text=True, check=True,
    ).stdout)["items"]

    parts = context.split("_")   # GKE contexts look like gke_<project>_<zone>_<cluster>
    is_gke = context.startswith("gke_") and len(parts) == 4
    return {
        "name": parts[3] if is_gke else context,
        "zone": parts[2] if is_gke else None,
        "node_count": len(nodes),
        "machine_types": sorted({
            n["metadata"]["labels"].get("node.kubernetes.io/instance-type", "unknown")
            for n in nodes
        }),
    }


def run_experiment(service: str, fault_type: str, fault_value: str,
                   traffic_level: str = "normal") -> str:
    exp_id = next_experiment_id()
    exp_dir = DATASET_RAW / exp_id
    exp_dir.mkdir(parents=True)

    logger.info("=" * 60)
    logger.info("Experiment  id=%s  service=%s  fault=%s  value=%s  traffic=%s",
                exp_id, service, fault_type, fault_value, traffic_level)
    logger.info("=" * 60)

    cluster = _cluster_info()

    phases = {}

    set_traffic(traffic_level)
    experiment_start = datetime.now(timezone.utc)

    try:
        # Phase 1: Baseline
        logger.info("[1/5] Baseline observation (%ds)", BASELINE_DURATION)
        phases["baseline"] = {"start": _now()}
        time.sleep(BASELINE_DURATION)
        phases["baseline"]["end"] = _now()

        # Phase 2: Inject fault
        logger.info("[2/5] Injecting fault: %s=%s on %s", fault_type, fault_value, service)
        phases["fault"] = {"start": _now()}
        original_state = apply_fault(service, fault_type, fault_value)
        try:
            time.sleep(FAULT_DURATION)
        except BaseException:
            # Ctrl+C or a stop signal mid-fault: undo the fault first, otherwise the
            # next experiment would start on a cluster that still has it.
            logger.warning("Interrupted during the fault phase — restoring %s first", service)
            restore_deployment(service, original_state)
            raise
        phases["fault"]["end"] = _now()

        # Phase 3: Restore
        logger.info("[3/5] Restoring deployment")
        phases["recovery"] = {"start": _now()}
        restore_deployment(service, original_state)
        time.sleep(RECOVERY_DURATION)
        phases["recovery"]["end"] = _now()

        experiment_end = datetime.now(timezone.utc)

        # Phase 4: Collect Prometheus metrics
        logger.info("[4/5] Collecting infrastructure metrics from Prometheus")
        prom = PrometheusCollector()
        df = prom.collect(experiment_start, experiment_end, step=15)
        prom.save(df, exp_dir / "prometheus.csv")

        # Phase 5: Collect Jaeger traces
        logger.info("[5/5] Collecting traces from Jaeger")
        jaeger = JaegerCollector()
        traces = jaeger.collect(experiment_start, experiment_end)
        jaeger.save(traces, exp_dir / "traces.json")

        # Phase 6: Collect Kubernetes events
        logger.info("[6/6] Collecting Kubernetes events")
        k8s = KubernetesCollector()
        events = k8s.collect("online-boutique", service, experiment_start, experiment_end)
        k8s.save(events, exp_dir / "events.json")

        # Metadata
        metadata = {
            "experiment_id": exp_id,
            "service": service,
            "fault_type": fault_type,
            "fault_value": fault_value,
            "traffic_level": traffic_level,
            "cluster": cluster,
            "baseline_duration": BASELINE_DURATION,
            "fault_duration": FAULT_DURATION,
            "recovery_duration": RECOVERY_DURATION,
            "start_time": experiment_start.isoformat(),
            "end_time": experiment_end.isoformat(),
            "phases": phases,
        }
        with open(exp_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        logger.info("=" * 60)
        logger.info("Experiment complete  id=%s  path=%s", exp_id, exp_dir)
        logger.info("=" * 60)
        return exp_id

    finally:
        restore_traffic()


def _now() -> float:
    return datetime.now(timezone.utc).timestamp()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run a controlled deployment impact experiment"
    )
    parser.add_argument("--service", required=True,
                        help="Target microservice (e.g. recommendationservice)")
    parser.add_argument("--fault", required=True,
                        choices=["cpu", "memory", "replicas", "latency"],
                        help="Fault type to inject")
    parser.add_argument("--value", required=True,
                        help="Fault magnitude (e.g. 150m, 64Mi, 0, 200)")
    parser.add_argument("--traffic", default="normal",
                        choices=["normal", "high"],
                        help="Traffic level during experiment (default: normal)")
    args = parser.parse_args()

    run_experiment(args.service, args.fault, args.value, args.traffic)
