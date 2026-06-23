"""
Collects Kubernetes-level events and pod state for an experiment window.

Captures:
  - Namespace events (OOMKilled, scheduling failures, pod restarts, rollouts)
  - Pod status for the target service (phase, conditions, container states)
  - Restart counts per pod
"""
import json
import logging
import subprocess
from datetime import datetime

logger = logging.getLogger(__name__)


def _run(cmd):
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return result.stdout


class KubernetesCollector:
    def collect(
        self,
        namespace: str,
        service: str,
        start: datetime,
        end: datetime,
    ) -> dict:
        """
        Collect events and pod state for the given namespace/service.
        start and end define the experiment window (stored in metadata only —
        event filtering happens during preprocessing since kubectl returns all events).
        """
        return {
            "namespace": namespace,
            "service": service,
            "start_time": start.isoformat(),
            "end_time": end.isoformat(),
            "events": self._get_events(namespace),
            "pods": self._get_pods(namespace, service),
            "restart_counts": self._get_restart_counts(namespace, service),
        }

    def _get_events(self, namespace):
        try:
            raw = _run(["kubectl", "get", "events", "-n", namespace, "-o", "json"])
            return json.loads(raw).get("items", [])
        except Exception as exc:
            logger.warning("Failed to collect events: %s", exc)
            return []

    def _get_pods(self, namespace, service):
        try:
            raw = _run([
                "kubectl", "get", "pods", "-n", namespace,
                "-l", f"app={service}", "-o", "json",
            ])
            pods = []
            for item in json.loads(raw).get("items", []):
                pods.append({
                    "name": item["metadata"]["name"],
                    "phase": item["status"].get("phase"),
                    "conditions": item["status"].get("conditions", []),
                    "container_statuses": item["status"].get("containerStatuses", []),
                    "start_time": item["status"].get("startTime"),
                })
            return pods
        except Exception as exc:
            logger.warning("Failed to collect pod info: %s", exc)
            return []

    def _get_restart_counts(self, namespace, service):
        try:
            raw = _run([
                "kubectl", "get", "pods", "-n", namespace,
                "-l", f"app={service}", "-o", "json",
            ])
            counts = {}
            for item in json.loads(raw).get("items", []):
                pod_name = item["metadata"]["name"]
                restarts = sum(
                    cs.get("restartCount", 0)
                    for cs in item["status"].get("containerStatuses", [])
                )
                counts[pod_name] = restarts
            return counts
        except Exception as exc:
            logger.warning("Failed to collect restart counts: %s", exc)
            return {}

    @staticmethod
    def save(data: dict, path):
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        logger.info("Saved K8s events  path=%s", path)
