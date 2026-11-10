"""
Replay recorded fault-injection runs as if they were happening live.

A ReplayView shows what an operator could see at a moment `now` during a recorded run:
Prometheus metrics, Kubernetes events and per-service latency from Jaeger traces, all cut
off at `now`. It is the data source for the agent's tools during offline evaluation.

What is hidden from the agent:
  - the injected fault (type and value). Only RecordedRun.ground_truth() returns it, for scoring.
  - Chaos Mesh objects in the event stream: a real faulty deployment would not create them.
  - anything recorded after `now`, including the end-of-run pod/restart snapshot in events.json.

The agent is told only that a change was applied to one service at a known time
(deployment-triggered framing); it has to work out what the change did from the telemetry.

Usage:
    run = RecordedRun("EXP_0307")
    view = run.at(30)                      # 30 s after the change
    view.change_notice(); view.metrics("checkoutservice"); view.latency_overview()
"""
from __future__ import annotations

import json
from datetime import datetime
from functools import cached_property
from pathlib import Path

import sys

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

REPO_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = REPO_ROOT / "dataset" / "raw"
PROCESSED_DIR = REPO_ROOT / "dataset" / "processed"
# Compact per-run span files (start, duration, error per service, and who made each call): built once from
# traces.json with `python -m genai.replay build-cache`; a few MB instead of up to ~500 MB of JSON
SPAN_CACHE_DIR = REPO_ROOT / "dataset" / "spans"
SPAN_CACHE_FORMAT = 2   # 2: spans parsed by scripts/span_measures.py (format 1 had no caller names)

# The same measurement as the ML labels: scripts/span_measures.py (scripts/ is not a package)
sys.path.insert(0, str(REPO_ROOT / "scripts"))
import span_measures as sm  # noqa: E402

# Service call graph, verified from parent/child spans in the recorded trace;
# feature_engineer.CALL_GRAPH is the same graph.
CALL_GRAPH: dict[str, list[str]] = {
    "frontend": ["adservice", "cartservice", "checkoutservice", "currencyservice",
                 "productcatalogservice", "recommendationservice", "shippingservice"],
    "checkoutservice": ["cartservice", "currencyservice", "emailservice", "paymentservice",
                        "productcatalogservice", "shippingservice"],
    "recommendationservice": ["productcatalogservice"],
    "adservice": [], "cartservice": [], "currencyservice": [], "emailservice": [],
    "paymentservice": [], "productcatalogservice": [], "shippingservice": [],
}
SERVICES = sorted(CALL_GRAPH)
# These services export no traces of their own (Online Boutique v0.10.5); they are seen only through their callers.
UNTRACED = {"adservice", "cartservice", "shippingservice"}


def _ts(iso: str | None) -> float | None:
    return datetime.fromisoformat(iso.replace("Z", "+00:00")).timestamp() if iso else None


class RecordedRun:
    """One recorded experiment from dataset/raw/<experiment_id>/."""

    def __init__(self, experiment_id: str, raw_dir: Path = RAW_DIR):
        self.experiment_id = experiment_id
        self.dir = Path(raw_dir) / experiment_id
        meta = json.loads((self.dir / "metadata.json").read_text())
        self._meta = meta
        ph = meta["phases"]
        self.baseline_start, self.baseline_end = ph["baseline"]["start"], ph["baseline"]["end"]
        self.change_time = ph["fault"]["start"]
        self.fault_end = ph["fault"]["end"]
        self.run_end = ph["recovery"]["end"]
        self.changed_service: str = meta["service"]
        self.traffic_level: str = meta.get("traffic_level", "normal")

    # ground truth: for scoring only, never shown to the agent

    def ground_truth(self) -> dict:
        labels = {}
        path = PROCESSED_DIR / "training_dataset.csv"
        if path.exists():
            df = pd.read_csv(path, usecols=["experiment_id", "severity", "blast_radius_final"])
            row = df[df["experiment_id"] == self.experiment_id]
            if len(row):
                labels = {"severity": row["severity"].iat[0],
                          "blast_radius_final": int(row["blast_radius_final"].iat[0])}
        return {"experiment_id": self.experiment_id, "service": self.changed_service,
                "fault_type": self._meta["fault_type"], "fault_value": self._meta["fault_value"],
                "traffic_level": self.traffic_level, **labels}

    # raw data, loaded once per run

    @cached_property
    def metrics_df(self) -> pd.DataFrame:
        df = pd.read_csv(self.dir / "prometheus.csv")
        df["ts"] = pd.to_datetime(df["timestamp"], utc=True).map(lambda t: t.timestamp())
        return df.drop(columns="timestamp").sort_values(["service", "ts"]).reset_index(drop=True)

    @cached_property
    def events_list(self) -> list[dict]:
        raw = json.loads((self.dir / "events.json").read_text())
        out = []
        for e in raw.get("events", []):
            obj = e.get("involvedObject", {})
            text = f"{obj.get('kind', '')} {obj.get('name', '')} {e.get('message', '')}".lower()
            if obj.get("kind", "").endswith("Chaos") or "chaos" in text:
                continue    # Chaos Mesh objects would reveal the injected fault
            first = _ts(e.get("firstTimestamp")) or _ts(e.get("eventTime"))
            last = _ts(e.get("lastTimestamp")) or first
            if first is None or last < self.baseline_start or first > self.run_end:
                continue    # history from earlier experiments, or after this run
            count = e.get("count") or 1
            if first < self.baseline_start:
                # Kubernetes merged this with identical events from earlier experiments: keep only the
                # latest occurrence, which falls in this run, and drop the cross-experiment count
                first, count = last, None
            out.append({"first": first, "last": last, "count": count,
                        "kind": obj.get("kind"), "object": obj.get("name"), "reason": e.get("reason"),
                        "type": e.get("type"), "message": e.get("message", "")})
        return sorted(out, key=lambda x: x["first"])

    @cached_property
    def spans(self) -> "sm.RunSpans":
        """
        Request spans per service (health checks and telemetry export left out), as in scripts/span_measures.py:
        own = its server spans; calls = its callers' client spans to it; callers = who made each call.
        Read from the compact cache when it is current, otherwise parsed from traces.json.
        """
        cache = SPAN_CACHE_DIR / f"{self.experiment_id}.npz"
        if cache.exists():
            with np.load(cache) as z:
                if "format" in z.files and int(z["format"]) == SPAN_CACHE_FORMAT:
                    return sm.RunSpans(own={s: z[f"{s}__own"] for s in sm.SERVICES},
                                       calls={s: z[f"{s}__calls"] for s in sm.SERVICES},
                                       callers={s: z[f"{s}__callers"].astype(object) for s in sm.SERVICES})
        return self.parse_spans()

    def write_span_cache(self) -> Path:
        SPAN_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        path = SPAN_CACHE_DIR / f"{self.experiment_id}.npz"
        rs = self.parse_spans()
        arrays = {"format": np.array(SPAN_CACHE_FORMAT)}
        for svc in sm.SERVICES:
            arrays[f"{svc}__own"] = rs.own[svc]
            arrays[f"{svc}__calls"] = rs.calls[svc]
            arrays[f"{svc}__callers"] = np.array(list(rs.callers[svc]), dtype="U32")
        np.savez_compressed(path, **arrays)
        return path

    def parse_spans(self) -> "sm.RunSpans":
        return sm.load_run_spans(self.dir)

    def at(self, seconds_after_change: float) -> "ReplayView":
        now = self.change_time + seconds_after_change
        if now > self.fault_end:
            raise ValueError(f"{seconds_after_change}s is after the end of the recorded fault phase")
        return ReplayView(self, now)


class ReplayView:
    """What an operator could see at time `now` during a recorded run. Times are seconds relative to the change."""

    def __init__(self, run: RecordedRun, now: float):
        self.run = run
        self.now = now

    def _rel(self, ts: float) -> float:
        return round(ts - self.run.change_time, 1)

    # context

    def change_notice(self) -> dict:
        return {"changed_service": self.run.changed_service,
                "seconds_since_change": self._rel(self.now),
                "traffic_level": self.run.traffic_level,
                "note": "A change was applied to this service. Its content is not known; "
                        "work out its effect from the telemetry."}

    def service_map(self) -> dict:
        callers = {s: sorted(c for c, callees in CALL_GRAPH.items() if s in callees) for s in SERVICES}
        return {s: {"calls": CALL_GRAPH[s], "called_by": callers[s], "exports_traces": s not in UNTRACED}
                for s in SERVICES}

    # telemetry, cut off at `now`

    def metrics(self, service: str, last_s: float | None = None) -> list[dict]:
        """Prometheus samples (every 15 s) for one service from the baseline up to now."""
        self._check(service)
        df = self.run.metrics_df
        start = self.run.baseline_start if last_s is None else max(self.run.baseline_start, self.now - last_s)
        rows = df[(df["service"] == service) & (df["ts"] >= start) & (df["ts"] <= self.now)]
        return [{"t": self._rel(r.ts),
                 "cpu_millicores": round(r.cpu_usage * 1000, 1) if pd.notna(r.cpu_usage) else None,
                 "memory_mib": round(r.memory_bytes / 2**20, 1) if pd.notna(r.memory_bytes) else None,
                 "replicas": None if pd.isna(r.replicas) else int(r.replicas),
                 "restarts": None if pd.isna(r.restart_count) else int(r.restart_count)}
                for r in rows.itertuples()]

    def events(self, service: str | None = None) -> list[dict]:
        """Kubernetes events in the namespace since the baseline began, as known at `now`."""
        out = []
        for e in self.run.events_list:
            if e["first"] > self.now or (service and not (e["object"] or "").startswith(service)):
                continue
            known = e["last"] <= self.now    # repeats after `now` are future information
            out.append({"t_first": self._rel(e["first"]),
                        "t_last": self._rel(e["last"]) if known else None,
                        "count": e["count"] if known else None,
                        "kind": e["kind"], "object": e["object"], "reason": e["reason"],
                        "type": e["type"], "message": e["message"][:300]})
        return out

    def latency(self, service: str, window_s: float = 30) -> dict:
        """
        Request latency and errors of one service: the baseline vs the last `window_s` seconds, measured as the
        ML labels are (scripts/span_measures.py). The changed service is measured from its callers (time they
        waited, including the network; frontend: its own spans); other services from their own spans, or their
        callers' if they export none. Where a second view exists it is added under "also", so a slowdown in the
        network (callers slower, own spans unchanged) can be told apart from slow processing (both slower).
        """
        self._check(service)
        changed = self.run.changed_service
        rs = self.run.spans
        arr, src = sm.target_view(rs, service) if service == changed else sm.other_view(rs, service, changed)
        out = self._latency_view(arr, src, window_s)
        out = {"service": service, **out}
        if service != "frontend" and len(rs.own[service]):
            other = (rs.own[service], "own") if src == "callers" else (rs.calls[service], "callers")
            if len(other[0]):
                out["also"] = self._latency_view(*other, window_s)
        return out

    def _latency_view(self, arr: np.ndarray, src: str, window_s: float) -> dict:
        source = {"own": "own server spans", "callers": "callers' client spans"}[src]
        if not len(arr):
            return {"source": None, "note": "no trace data for this service"}
        base = sm.window(arr, self.run.baseline_start, self.run.baseline_end)
        recent = sm.window(arr, self.now - window_s, self.now)

        def summ(a):
            if not len(a):
                return {"requests": 0}
            d = a[:, 1]
            return {"requests": int(len(a)), "p50_ms": round(float(np.median(d)), 2),
                    "p95_ms": round(float(np.percentile(d, 95)), 2), "error_rate": round(float(a[:, 2].mean()), 4)}

        out = {"source": source, "baseline": summ(base), f"last_{int(window_s)}s": summ(recent)}
        c = sm.compare(base, recent)
        if c["judged"] and np.median(base[:, 1]) > 0:
            out["p50_change_pct"] = round(float(100 * (np.median(recent[:, 1]) / np.median(base[:, 1]) - 1)), 1)
            out["slower_p_value"] = float(f"{mannwhitneyu(recent[:, 1], base[:, 1], alternative='greater').pvalue:.2g}")
            out["significantly_slower"] = c["slower"]
            out["significantly_more_errors"] = c["erroring"]
        else:
            out["note"] = "too few requests to compare reliably"
        return out

    def latency_overview(self, window_s: float = 30) -> list[dict]:
        return [self.latency(s, window_s) for s in SERVICES]

    def _check(self, service: str):
        if service not in CALL_GRAPH:
            raise ValueError(f"unknown service '{service}'; known: {', '.join(SERVICES)}")


def holdout_ids() -> list[str]:
    """Runs never used for training or tuning - the evaluation set for Phase 3."""
    return sorted((PROCESSED_DIR / "holdout_ids.txt").read_text().split())


def build_span_cache() -> None:
    """Write dataset/spans/<run>.npz for every complete run (one traces.json in memory at a time)."""
    import gc
    import sys
    import time
    runs = sorted(p.parent.name for p in RAW_DIR.glob("EXP_*/metadata.json") if (p.parent / "traces.json").exists())
    t0 = time.time()
    for i, e in enumerate(runs, 1):
        cache = SPAN_CACHE_DIR / f"{e}.npz"
        if cache.exists():
            with np.load(cache) as z:
                if "format" in z.files and int(z["format"]) == SPAN_CACHE_FORMAT:
                    continue    # current; older formats are rebuilt
        RecordedRun(e).write_span_cache()
        gc.collect()
        print(f"  [{i}/{len(runs)}] {e} ({time.time() - t0:.0f}s)", file=sys.stderr, flush=True)


if __name__ == "__main__":
    import sys
    if sys.argv[1:] == ["build-cache"]:
        build_span_cache()
    else:
        sys.exit("usage: python -m genai.replay build-cache")
