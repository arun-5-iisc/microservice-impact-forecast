"""
Collects Tier 3 infrastructure metrics from Prometheus for all Online Boutique services.

Metrics collected (per service, per 15-second step):
  cpu_usage     - CPU cores consumed (rate over 1m window)
  memory_bytes  - Working set memory in bytes
  replicas      - Available replica count
  restart_count - Total container restart count

Application-level metrics (latency, error rate, request rate) are derived from
Jaeger traces during preprocessing — not collected here.

Note: Online Boutique containers are named "server" (not the service name), so
queries filter by pod name prefix rather than container name.
"""
import contextlib
import logging
import socket
import subprocess
import time
from datetime import datetime

import pandas as pd
import requests

logger = logging.getLogger(__name__)

PROMETHEUS_SERVICE = "prometheus-kube-prometheus-prometheus"
PROMETHEUS_NAMESPACE = "monitoring"
PROMETHEUS_PORT = 9090
APP_NAMESPACE = "online-boutique"

ALL_SERVICES = [
    "adservice",
    "cartservice",
    "checkoutservice",
    "currencyservice",
    "emailservice",
    "frontend",
    "paymentservice",
    "productcatalogservice",
    "recommendationservice",
    "shippingservice",
]


@contextlib.contextmanager
def _port_forward(namespace, service, remote_port):
    local_port = _free_port()
    proc = subprocess.Popen(
        ["kubectl", "port-forward", f"svc/{service}",
         f"{local_port}:{remote_port}", "-n", namespace],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )
    try:
        _wait_for_port(local_port, proc)
        yield f"http://localhost:{local_port}"
    finally:
        proc.terminate()
        proc.wait(timeout=5)


def _free_port():
    with socket.socket() as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def _wait_for_port(port, proc=None, timeout=30):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if proc is not None and proc.poll() is not None:
            err = proc.stderr.read().decode(errors="replace").strip()
            raise RuntimeError(f"kubectl port-forward exited early: {err}")
        try:
            with socket.create_connection(("localhost", port), timeout=1):
                return
        except OSError:
            time.sleep(0.5)
    raise TimeoutError(f"Port {port} not ready after {timeout}s")


class PrometheusCollector:
    def __init__(
        self,
        service=PROMETHEUS_SERVICE,
        namespace=PROMETHEUS_NAMESPACE,
        port=PROMETHEUS_PORT,
    ):
        self.service = service
        self.namespace = namespace
        self.port = port

    def collect(self, start: datetime, end: datetime, step: int = 15) -> pd.DataFrame:
        """
        Query Prometheus for the [start, end] window with the given step (seconds).
        Returns a DataFrame with one row per (service, timestamp).
        """
        start_ts = start.timestamp()
        end_ts = end.timestamp()
        rows = []

        with _port_forward(self.namespace, self.service, self.port) as base_url:
            for svc in ALL_SERVICES:
                svc_rows = self._collect_service(base_url, svc, start_ts, end_ts, step)
                rows.extend(svc_rows)

        df = pd.DataFrame(rows)
        if not df.empty:
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s", utc=True)
        return df

    def _collect_service(self, base_url, service, start, end, step):
        ns = APP_NAMESPACE
        # Filter by pod prefix because the container name is "server" for all services.
        pod_re = f"{service}-.*"
        queries = {
            "cpu_usage": (
                f'sum(rate(container_cpu_usage_seconds_total{{'
                f'namespace="{ns}",pod=~"{pod_re}",'
                f'container!="POD",container!=""}}[1m]))'
            ),
            "memory_bytes": (
                f'sum(container_memory_working_set_bytes{{'
                f'namespace="{ns}",pod=~"{pod_re}",'
                f'container!="POD",container!=""}})'
            ),
            "replicas": (
                f'kube_deployment_status_replicas_available{{'
                f'namespace="{ns}",deployment="{service}"}}'
            ),
            "restart_count": (
                f'sum(kube_pod_container_status_restarts_total{{'
                f'namespace="{ns}",pod=~"{pod_re}"}})'
            ),
        }

        timestamps = None
        series = {}

        for metric, query in queries.items():
            try:
                result = self._query_range(base_url, query, start, end, step)
                if result:
                    values = result[0]["values"]
                    if timestamps is None:
                        timestamps = [v[0] for v in values]
                    series[metric] = [
                        (float(v[1]) if v[1] not in ("NaN", "Inf", "-Inf") else None)
                        for v in values
                    ]
                else:
                    logger.debug("No data for %s.%s", service, metric)
                    series[metric] = None
            except Exception as exc:
                logger.warning("Query failed  service=%s  metric=%s  error=%s", service, metric, exc)
                series[metric] = None

        if timestamps is None:
            return []

        n = len(timestamps)
        return [
            {
                "timestamp": timestamps[i],
                "service": service,
                "cpu_usage": series["cpu_usage"][i] if series.get("cpu_usage") else None,
                "memory_bytes": series["memory_bytes"][i] if series.get("memory_bytes") else None,
                "replicas": series["replicas"][i] if series.get("replicas") else None,
                "restart_count": series["restart_count"][i] if series.get("restart_count") else None,
            }
            for i in range(n)
        ]

    def _query_range(self, base_url, query, start, end, step):
        resp = requests.get(
            f"{base_url}/api/v1/query_range",
            params={"query": query, "start": start, "end": end, "step": step},
            timeout=30,
        )
        resp.raise_for_status()
        result = resp.json()
        if result["status"] != "success":
            raise RuntimeError(f"Prometheus error: {result.get('error')}")
        return result["data"]["result"]

    @staticmethod
    def save(df: pd.DataFrame, path):
        df.to_csv(path, index=False)
        logger.info("Saved Prometheus data  path=%s  rows=%d", path, len(df))
