"""
Applies and restores deployment faults for experiments.

Fault types:
  cpu      - Helm upgrade to reduce CPU limit only; restored via helm rollback
  memory   - Helm upgrade to reduce memory limit only; restored via helm rollback
  replicas - kubectl scale (chart has no replicaCount value); restored via kubectl scale
  latency  - Chaos Mesh NetworkChaos resource; restored via kubectl delete
             Requires Chaos Mesh installed in the cluster (chaos-mesh namespace).
"""
import json
import logging
import subprocess
import tempfile
from pathlib import Path

logger = logging.getLogger(__name__)

NAMESPACE = "online-boutique"
HELM_RELEASE = "online-boutique"
CHART_PATH = str(Path(__file__).parent.parent / "deployment" / "online-boutique" / "helm-chart")

SERVICE_HELM_KEY = {
    "adservice": "adService",
    "cartservice": "cartService",
    "checkoutservice": "checkoutService",
    "currencyservice": "currencyService",
    "emailservice": "emailService",
    "frontend": "frontend",
    "paymentservice": "paymentService",
    "productcatalogservice": "productCatalogService",
    "recommendationservice": "recommendationService",
    "shippingservice": "shippingService",
}


def _run(cmd, check=True):
    logger.debug("Running: %s", " ".join(str(c) for c in cmd))
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        # With check=False the caller expects failures (e.g. 2-second rollout polls)
        level = logging.ERROR if check else logging.DEBUG
        if result.stdout.strip():
            logger.log(level, "stdout: %s", result.stdout.strip())
        if result.stderr.strip():
            logger.log(level, "stderr: %s", result.stderr.strip())
        if check:
            raise subprocess.CalledProcessError(
                result.returncode, cmd, result.stdout, result.stderr
            )
    return result


def _helm_revision():
    result = _run(
        ["helm", "history", HELM_RELEASE, "-n", NAMESPACE, "--max", "1", "-o", "json"]
    )
    history = json.loads(result.stdout)
    return history[0]["revision"]


def _get_replicas(service):
    result = _run(
        ["kubectl", "get", "deployment", service, "-n", NAMESPACE,
         "-o", "jsonpath={.spec.replicas}"]
    )
    text = result.stdout.strip()
    return int(text) if text else 1


def wait_for_rollout(service, timeout=120):
    logger.info("Waiting for rollout: %s", service)
    _run(
        ["kubectl", "rollout", "status", f"deployment/{service}",
         "-n", NAMESPACE, f"--timeout={timeout}s"]
    )


def _wait_for_rollout_start(service, timeout=30):
    """
    Wait until Kubernetes has acknowledged the new ReplicaSet (i.e. a rollout
    has started), but do NOT wait for pods to become ready.  This is intentional
    for fault-injection: the degraded / crash-looping state IS the experiment.
    """
    import time
    deadline = time.time() + timeout
    while time.time() < deadline:
        result = _run(
            ["kubectl", "rollout", "status", f"deployment/{service}",
             "-n", NAMESPACE, "--timeout=2s"],
            check=False,
        )
        # Either it finished cleanly or it started but isn't ready yet
        # Kubernetes accepted the change and a rollout is underway.
        if result.returncode == 0 or "Waiting for" in result.stdout or "Waiting for" in result.stderr:
            logger.info("Rollout started for %s", service)
            return
        time.sleep(2)
    logger.warning("Could not confirm rollout started for %s within %ds", service, timeout)


def apply_fault(service, fault_type, fault_value):
    """
    Apply a fault to the named service.
    Returns an original_state dict that must be passed to restore_deployment().
    """
    logger.info("Applying fault  service=%s  type=%s  value=%s", service, fault_type, fault_value)

    if fault_type == "cpu":
        helm_key = _require_helm_key(service)
        revision = _helm_revision()
        # No --wait: the pod may never become fully ready under resource pressure,
        # and that degraded state is exactly what the experiment observes.
        _run([
            "helm", "upgrade", "--reuse-values", HELM_RELEASE, CHART_PATH,
            "-n", NAMESPACE,
            "--set", f"{helm_key}.resources.limits.cpu={fault_value}",
        ])
        _wait_for_rollout_start(service)
        return {"fault_type": "cpu", "helm_revision": revision}

    elif fault_type == "memory":
        helm_key = _require_helm_key(service)
        revision = _helm_revision()
        # Only reduce the limit - request stays unchanged so scheduling is unaffected.
        # OOMKill triggers when actual usage exceeds the new (lower) limit.
        _run([
            "helm", "upgrade", "--reuse-values", HELM_RELEASE, CHART_PATH,
            "-n", NAMESPACE,
            "--set", f"{helm_key}.resources.limits.memory={fault_value}",
        ])
        _wait_for_rollout_start(service)
        return {"fault_type": "memory", "helm_revision": revision}

    elif fault_type == "replicas":
        original = _get_replicas(service)
        _run([
            "kubectl", "scale", "deployment", service,
            "-n", NAMESPACE, f"--replicas={fault_value}",
        ])
        wait_for_rollout(service)
        return {"fault_type": "replicas", "original_replicas": original}

    elif fault_type == "latency":
        chaos_name = f"latency-{service}-{fault_value}ms"
        manifest = {
            "apiVersion": "chaos-mesh.org/v1alpha1",
            "kind": "NetworkChaos",
            "metadata": {"name": chaos_name, "namespace": NAMESPACE},
            "spec": {
                "action": "delay",
                "mode": "all",
                "selector": {
                    "namespaces": [NAMESPACE],
                    "labelSelectors": {"app": service},
                },
                "delay": {
                    "latency": f"{fault_value}ms",
                    "jitter": "0ms",
                    "correlation": "0",
                },
                # Duration longer than FAULT_DURATION; we delete the resource on restore.
                "duration": "600s",
            },
        }
        import json as _json
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            _json.dump(manifest, f)
            tmp_path = f.name
        _run(["kubectl", "apply", "-f", tmp_path])
        return {"fault_type": "latency", "chaos_name": chaos_name}

    else:
        raise ValueError(f"Unknown fault type: {fault_type!r}")


def restore_deployment(service, original_state):
    """Restore the deployment to its pre-fault state."""
    fault_type = original_state["fault_type"]
    logger.info("Restoring  service=%s  fault_type=%s", service, fault_type)

    if fault_type in ("cpu", "memory"):
        revision = original_state["helm_revision"]
        _run(["helm", "rollback", HELM_RELEASE, str(revision), "-n", NAMESPACE, "--wait"])

    elif fault_type == "latency":
        _run([
            "kubectl", "delete", "networkchaos", original_state["chaos_name"],
            "-n", NAMESPACE, "--ignore-not-found",
        ])

    elif fault_type == "replicas":
        _run([
            "kubectl", "scale", "deployment", service,
            "-n", NAMESPACE, f"--replicas={original_state['original_replicas']}",
        ])
        wait_for_rollout(service)


def _require_helm_key(service):
    key = SERVICE_HELM_KEY.get(service)
    if not key:
        raise ValueError(f"Unknown service: {service!r}")
    return key
