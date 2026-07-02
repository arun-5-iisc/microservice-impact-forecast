"""
Builds the proactive ML training dataset from raw experiment data.

Proactive formulation:
  Features = system state at T seconds into the fault phase (T ∈ {15, 30, 45, 60})
  Labels   = final severity + blast radius from the LAST 60s of the fault phase

Each experiment produces 4 rows (one per T), so 90 experiments → ~360 training rows.

Usage:
  python scripts/preprocess.py
  python scripts/preprocess.py --raw-dir dataset/raw --out dataset/processed/training_dataset.csv
"""
import argparse
import json
import logging
from pathlib import Path

import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).parent.parent
DEFAULT_RAW  = REPO_ROOT / "dataset" / "raw"
DEFAULT_OUT  = REPO_ROOT / "dataset" / "processed" / "training_dataset.csv"

SNAPSHOT_OFFSETS = [15, 30, 45, 60]   # seconds into fault phase
LABEL_WINDOW     = 60                  # seconds before fault end used for labels

LATENCY_DELTA_HIGH   = 1.0    # >100% increase → High
LATENCY_DELTA_MEDIUM = 0.20   # >20%  increase → Medium
ERROR_RATE_HIGH      = 0.05
ERROR_RATE_MEDIUM    = 0.01


# ─────────────────────────────────────────────────────────────────────────────
# Prometheus helpers
# ─────────────────────────────────────────────────────────────────────────────

def _load_prometheus(exp_dir: Path) -> pd.DataFrame:
    path = exp_dir / "prometheus.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path, parse_dates=["timestamp"])
    df["ts"] = df["timestamp"].apply(lambda t: t.timestamp())
    return df


def _prom_snapshot(df: pd.DataFrame, service: str, ts: float) -> dict:
    """Nearest Prometheus row for (service, ts)."""
    sdf = df[df["service"] == service].copy()
    if sdf.empty:
        return {}
    idx = (sdf["ts"] - ts).abs().idxmin()
    row = sdf.loc[idx]
    return {
        "cpu_usage":     row.get("cpu_usage"),
        "memory_bytes":  row.get("memory_bytes"),
        "replicas":      row.get("replicas"),
        "restart_count": row.get("restart_count"),
    }


def _prom_baseline(df: pd.DataFrame, service: str, t_start: float, t_end: float) -> dict:
    mask = (df["service"] == service) & (df["ts"] >= t_start) & (df["ts"] <= t_end)
    sdf = df[mask]
    if sdf.empty:
        return {"baseline_cpu": None, "baseline_memory": None}
    return {
        "baseline_cpu":    sdf["cpu_usage"].mean(),
        "baseline_memory": sdf["memory_bytes"].mean(),
    }


def _system_snapshot(df: pd.DataFrame, ts: float) -> dict:
    """Mean CPU and memory across ALL services at the nearest timestamp."""
    if df.empty:
        return {"system_cpu_mean": None, "system_memory_mean": None}
    ts_vals = df["ts"].unique()
    nearest_ts = ts_vals[(pd.Series(ts_vals) - ts).abs().idxmin()]
    snap = df[df["ts"] == nearest_ts]
    return {
        "system_cpu_mean":    snap["cpu_usage"].mean(),
        "system_memory_mean": snap["memory_bytes"].mean(),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Trace helpers
# ─────────────────────────────────────────────────────────────────────────────

def _load_traces(exp_dir: Path) -> dict:
    path = exp_dir / "traces.json"
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)


def _is_error_span(span: dict) -> bool:
    for tag in span.get("tags", []):
        k, v = tag.get("key"), tag.get("value")
        if k == "otel.status_code" and str(v).upper() == "ERROR":
            return True
        if k == "error" and v is True:
            return True
    return False


def _span_start_s(span: dict) -> float:
    return span["startTime"] / 1_000_000   # microseconds → seconds


def _span_end_s(span: dict) -> float:
    return (_span_start_s(span) + span["duration"] / 1_000_000)


def _root_duration_ms(trace: dict) -> float | None:
    """Duration of the root span (no incoming references) in milliseconds."""
    span_ids = {s["spanID"] for s in trace["spans"]}
    referenced = {
        ref["spanID"]
        for s in trace["spans"]
        for ref in s.get("references", [])
        if ref.get("spanID") in span_ids
    }
    roots = [s for s in trace["spans"] if s["spanID"] not in referenced]
    if not roots:
        return None
    return roots[0]["duration"] / 1000   # microseconds → milliseconds


def _max_depth(trace: dict) -> int:
    """Longest reference chain (proxy for trace depth)."""
    children: dict[str, list[str]] = {}
    for span in trace["spans"]:
        for ref in span.get("references", []):
            parent = ref.get("spanID")
            if parent:
                children.setdefault(parent, []).append(span["spanID"])
    if not children:
        return 1
    all_ids = {s["spanID"] for s in trace["spans"]}
    roots = all_ids - {sid for kids in children.values() for sid in kids}

    def depth(sid):
        kids = children.get(sid, [])
        return 1 + max((depth(k) for k in kids), default=0)

    return max((depth(r) for r in roots), default=1)


def _trace_features(traces: dict, service: str, t_start: float, t_end: float) -> dict:
    """
    Compute trace-derived features for a service over [t_start, t_end].
    Window duration used to compute request_rate.
    """
    window_s = max(t_end - t_start, 1)
    svc_traces = traces.get(service, [])

    # Filter traces whose root span falls in the window
    in_window = []
    for trace in svc_traces:
        for span in trace.get("spans", []):
            ss = _span_start_s(span)
            if t_start <= ss <= t_end:
                in_window.append(trace)
                break

    if not in_window:
        return {
            "request_rate":   0.0,
            "error_rate":     0.0,
            "latency_mean_ms": None,
            "max_trace_depth": 0,
        }

    total_spans = sum(len(t["spans"]) for t in in_window)
    error_spans = sum(
        1 for t in in_window for s in t["spans"] if _is_error_span(s)
    )
    durations = [d for t in in_window if (d := _root_duration_ms(t)) is not None]
    depths    = [_max_depth(t) for t in in_window]

    return {
        "request_rate":    len(in_window) / window_s,
        "error_rate":      error_spans / max(total_spans, 1),
        "latency_mean_ms": sum(durations) / len(durations) if durations else None,
        "max_trace_depth": max(depths, default=0),
    }


def _blast_radius(traces: dict, target: str, t_start: float, t_end: float,
                  baseline_latencies: dict, threshold: float = 0.10) -> int:
    """
    Count services (excluding target) whose mean latency increased by > threshold
    compared to their baseline latency.
    """
    affected = 0
    for svc, base_lat in baseline_latencies.items():
        if svc == target or base_lat is None or base_lat == 0:
            continue
        feats = _trace_features(traces, svc, t_start, t_end)
        lat = feats.get("latency_mean_ms")
        if lat is not None and (lat - base_lat) / base_lat > threshold:
            affected += 1
    return affected


# ─────────────────────────────────────────────────────────────────────────────
# Severity label
# ─────────────────────────────────────────────────────────────────────────────

def _compute_severity(latency_delta: float | None, error_rate: float,
                      restart_count: float | None, replicas: float | None) -> str:
    """
    Observation-based severity label derived entirely from runtime signals,
    not from fault configuration inputs.

    Priority (highest to lowest):
      1. replicas == 0        → service unavailable          → High
      2. restart_count > 0    → pod crashed under this fault → High
      3. error_rate thresholds → direct user-visible impact
      4. latency_delta thresholds → relative degradation from baseline
    """
    if replicas is not None and replicas == 0:
        return "High"

    if restart_count is not None and restart_count > 0:
        return "High"

    if error_rate > ERROR_RATE_HIGH:
        return "High"
    if error_rate > ERROR_RATE_MEDIUM:
        return "Medium"

    if latency_delta is not None:
        if latency_delta > LATENCY_DELTA_HIGH:
            return "High"
        if latency_delta > LATENCY_DELTA_MEDIUM:
            return "Medium"

    return "Low"


# ─────────────────────────────────────────────────────────────────────────────
# Per-experiment processing
# ─────────────────────────────────────────────────────────────────────────────

def process_experiment(exp_dir: Path) -> list[dict]:
    meta_path = exp_dir / "metadata.json"
    if not meta_path.exists():
        logger.warning("Skipping %s — no metadata.json", exp_dir.name)
        return []

    with open(meta_path) as f:
        meta = json.load(f)

    phases       = meta["phases"]
    fault_start  = phases["fault"]["start"]
    fault_end    = phases["fault"]["end"]
    base_start   = phases["baseline"]["start"]
    base_end     = phases["baseline"]["end"]
    service      = meta["service"]
    fault_type   = meta["fault_type"]
    fault_value  = meta["fault_value"]
    traffic      = meta.get("traffic_level", "normal")

    # Numeric fault value for ML (strip units like 'm', 'Mi')
    try:
        fault_value_numeric = float("".join(c for c in fault_value if c.isdigit() or c == "."))
    except ValueError:
        fault_value_numeric = None

    prom   = _load_prometheus(exp_dir)
    traces = _load_traces(exp_dir)

    # ── Baseline stats ────────────────────────────────────────────────────────
    base_prom  = _prom_baseline(prom, service, base_start, base_end)
    base_feats = _trace_features(traces, service, base_start, base_end)
    base_lat   = base_feats.get("latency_mean_ms")

    # Baseline latency for ALL services (used for blast radius)
    all_services = list(traces.keys())
    base_latencies = {}
    for svc in all_services:
        bf = _trace_features(traces, svc, base_start, base_end)
        base_latencies[svc] = bf.get("latency_mean_ms")

    # ── Label window (last 60s of fault phase) ────────────────────────────────
    label_start  = max(fault_end - LABEL_WINDOW, fault_start)
    final_feats  = _trace_features(traces, service, label_start, fault_end)
    final_lat    = final_feats.get("latency_mean_ms")
    final_lat_delta = (
        (final_lat - base_lat) / base_lat
        if (final_lat is not None and base_lat is not None and base_lat > 0)
        else None
    )
    final_error_rate   = final_feats.get("error_rate", 0.0)
    final_prom         = _prom_snapshot(prom, service, fault_end)
    final_restart      = final_prom.get("restart_count")
    final_replicas     = final_prom.get("replicas")
    blast_radius_final = _blast_radius(traces, service, label_start, fault_end, base_latencies)
    severity           = _compute_severity(
        final_lat_delta, final_error_rate, final_restart, final_replicas
    )

    # ── One row per T ─────────────────────────────────────────────────────────
    rows = []
    for T in SNAPSHOT_OFFSETS:
        snap_time = fault_start + T
        if snap_time > fault_end:
            continue

        p_snap    = _prom_snapshot(prom, service, snap_time)
        sys_snap  = _system_snapshot(prom, snap_time)
        t_feats   = _trace_features(traces, service, fault_start, snap_time)

        cpu      = p_snap.get("cpu_usage")
        base_cpu = base_prom.get("baseline_cpu")
        mem      = p_snap.get("memory_bytes")
        base_mem = base_prom.get("baseline_memory")
        lat      = t_feats.get("latency_mean_ms")

        rows.append({
            # Context
            "experiment_id":        meta["experiment_id"],
            "service":              service,
            "fault_type":           fault_type,
            "fault_value":          fault_value,
            "fault_value_numeric":  fault_value_numeric,
            "traffic_level":        traffic,
            "time_into_fault_s":    T,

            # Prometheus features
            "target_cpu":            cpu,
            "target_cpu_delta":      (cpu - base_cpu) / base_cpu if (cpu and base_cpu) else None,
            "target_memory_bytes":   mem,
            "target_memory_delta":   (mem - base_mem) / base_mem if (mem and base_mem) else None,
            "target_replicas":       p_snap.get("replicas"),
            "target_restart_count":  p_snap.get("restart_count"),
            "system_cpu_mean":       sys_snap.get("system_cpu_mean"),
            "system_memory_mean":    sys_snap.get("system_memory_mean"),

            # Trace features (early window only)
            "target_request_rate":   t_feats.get("request_rate"),
            "target_error_rate":     t_feats.get("error_rate"),
            "target_latency_mean_ms": lat,
            "target_latency_delta":  (
                (lat - base_lat) / base_lat
                if (lat is not None and base_lat is not None and base_lat > 0)
                else None
            ),
            "blast_radius_early":    _blast_radius(
                traces, service, fault_start, snap_time, base_latencies
            ),
            "max_trace_depth":       t_feats.get("max_trace_depth"),

            # Labels (from final window — same for all 4 rows of this experiment)
            "severity":              severity,
            "blast_radius_final":    blast_radius_final,
        })

    return rows


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main(raw_dir: Path, out_path: Path):
    exp_dirs = sorted(d for d in raw_dir.iterdir() if d.is_dir() and d.name.startswith("EXP_"))
    if not exp_dirs:
        logger.error("No EXP_* directories found in %s", raw_dir)
        return

    all_rows = []
    for exp_dir in exp_dirs:
        logger.info("Processing %s", exp_dir.name)
        try:
            rows = process_experiment(exp_dir)
            all_rows.extend(rows)
            logger.info("  → %d rows", len(rows))
        except Exception as exc:
            logger.error("  FAILED: %s", exc)

    if not all_rows:
        logger.error("No rows generated.")
        return

    df = pd.DataFrame(all_rows)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    logger.info(
        "Saved %d rows (%d experiments) → %s",
        len(df), df["experiment_id"].nunique(), out_path,
    )
    logger.info("Severity distribution:\n%s", df["severity"].value_counts().to_string())
    logger.info("Fault type counts:\n%s", df.groupby("fault_type").size().to_string())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build proactive ML training dataset")
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW)
    parser.add_argument("--out",     type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    main(args.raw_dir, args.out)
