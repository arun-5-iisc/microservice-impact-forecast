"""
Collects the Jaeger traces of all Online Boutique services for an experiment window.

The traces are saved as they are. Latency, error rate and blast radius are measured from them
later, by span_measures.py.
"""
import contextlib
import json
import logging
import socket
import subprocess
import time
from datetime import datetime

import requests

logger = logging.getLogger(__name__)

JAEGER_SERVICE = "jaeger"
JAEGER_NAMESPACE = "monitoring"
JAEGER_PORT = 16686

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


class JaegerCollector:
    def __init__(
        self,
        service=JAEGER_SERVICE,
        namespace=JAEGER_NAMESPACE,
        port=JAEGER_PORT,
    ):
        self.service = service
        self.namespace = namespace
        self.port = port

    def collect(self, start: datetime, end: datetime, limit: int = 10000) -> dict:
        """
        Fetch all traces in [start, end] for every Online Boutique service.
        Returns a dict: { service_name: [trace, ...] }
        Jaeger timestamps are in microseconds.
        """
        start_us = int(start.timestamp() * 1_000_000)
        end_us = int(end.timestamp() * 1_000_000)
        all_traces = {}

        with _port_forward(self.namespace, self.service, self.port) as base_url:
            for svc in ALL_SERVICES:
                traces = self._fetch_traces(base_url, svc, start_us, end_us, limit)
                if traces:
                    all_traces[svc] = traces
                    logger.info("Collected traces  service=%s  count=%d", svc, len(traces))
                else:
                    logger.debug("No traces found for service: %s", svc)

        total = sum(len(v) for v in all_traces.values())
        logger.info("Total traces collected: %d across %d services", total, len(all_traces))
        return all_traces

    def _fetch_traces(self, base_url, service, start_us, end_us, limit):
        try:
            resp = requests.get(
                f"{base_url}/api/traces",
                params={
                    "service": service,
                    "start": start_us,
                    "end": end_us,
                    "limit": limit,
                },
                timeout=30,
            )
            resp.raise_for_status()
            return resp.json().get("data", [])
        except Exception as exc:
            logger.warning("Failed to fetch traces  service=%s  error=%s", service, exc)
            return []

    @staticmethod
    def save(traces: dict, path):
        with open(path, "w") as f:
            json.dump(traces, f)
        total = sum(len(v) for v in traces.values())
        logger.info(
            "Saved Jaeger traces  path=%s  traces=%d  services=%d",
            path, total, len(traces),
        )
