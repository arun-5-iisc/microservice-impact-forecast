"""
Sets the load generator's traffic level for an experiment.

Patches the loadgenerator deployment with kubectl set env (no Helm change); the pod restarts
with the new values, and the script waits STABILISE_SECONDS before the baseline starts.

Levels (Locust users; RATE is how many users start per second, not requests per second):
  normal  USERS=10  RATE=1   the loadgenerator default
  high    USERS=25  RATE=1   2.5 times the users
"""
import logging
import subprocess
import time

logger = logging.getLogger(__name__)

NAMESPACE = "online-boutique"
LOADGENERATOR = "loadgenerator"
STABILISE_SECONDS = 20

TRAFFIC_LEVELS = {
    "normal": {"USERS": "10", "RATE": "1"},
    "high":   {"USERS": "25", "RATE": "1"},
}

_DEFAULT_LEVEL = "normal"


def set_traffic(level: str) -> None:
    """
    Set the traffic level and wait for it to settle. Setting the values the deployment already
    has changes nothing on the cluster (no restart).
    """
    if level not in TRAFFIC_LEVELS:
        raise ValueError(f"Unknown traffic level {level!r}. Choose from: {list(TRAFFIC_LEVELS)}")

    env = TRAFFIC_LEVELS[level]
    logger.info("Setting traffic level: %s  (USERS=%s RATE=%s)", level, env["USERS"], env["RATE"])

    _patch_env(env)

    logger.info("Waiting %ds for loadgenerator to restart and traffic to stabilise", STABILISE_SECONDS)
    time.sleep(STABILISE_SECONDS)


def restore_traffic() -> None:
    """Restore the loadgenerator to its default traffic level (normal)."""
    env = TRAFFIC_LEVELS[_DEFAULT_LEVEL]
    logger.info("Restoring loadgenerator to default traffic (USERS=%s RATE=%s)",
                env["USERS"], env["RATE"])
    _patch_env(env)


def _patch_env(env: dict) -> None:
    env_args = [f"{k}={v}" for k, v in env.items()]
    cmd = [
        "kubectl", "set", "env",
        f"deployment/{LOADGENERATOR}",
        "-n", NAMESPACE,
    ] + env_args

    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        logger.error("kubectl set env failed: %s", result.stderr.strip())
        raise subprocess.CalledProcessError(result.returncode, cmd, result.stdout, result.stderr)
    logger.debug("kubectl set env output: %s", result.stdout.strip())
