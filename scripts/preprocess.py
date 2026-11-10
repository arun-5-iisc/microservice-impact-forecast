"""
Builds the training dataset from the raw experiment runs.

  Features: the state of the system T seconds into the fault phase, T = 15, 30, 45, 60
  Labels:   final severity and blast radius, measured over the last 60 s of the fault phase

So each run gives 4 rows, one per T.

The measurements come from span_measures.py:
  severity     latency and error rate of the changed service as its callers saw them (frontend: its
               own spans), last 60 s vs baseline. The latency rule is only applied when there are
               enough requests to tell a change from chance at the 5% level; otherwise
               final_latency_delta is left empty.
  blast radius other services that got significantly slower or started erroring.
Every measurement behind each run's labels is also written to label_measurements.csv.

Usage:
  python scripts/preprocess.py
  python scripts/preprocess.py --raw-dir dataset/raw --out dataset/processed/training_dataset.csv
"""
import argparse
import json
import logging
from pathlib import Path

import pandas as pd

from span_measures import (SEVERITY_ALPHA, blast_radius, enough_requests, parse_spans, summary, target_view,
                           window)

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

LATENCY_DELTA_HIGH   = 1.0    # >100% increase -> High
LATENCY_DELTA_MEDIUM = 0.20   # >20%  increase -> Medium
ERROR_RATE_HIGH      = 0.05
ERROR_RATE_MEDIUM    = 0.01

# These export no spans in Online Boutique v0.10.5: adservice and cartservice have no tracing
# settings in their Helm templates. SO span from them are measured from
# their callers' client spans. latency_observable records this as a feature.
NO_TRACE_SERVICES = {"adservice", "cartservice", "shippingservice"}


# Prometheus helpers

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


# Trace helpers

def _load_traces(exp_dir: Path) -> dict:
    path = exp_dir / "traces.json"
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)


def _span_start_s(span: dict) -> float:
    return span["startTime"] / 1_000_000   # microseconds -> seconds


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


def _max_trace_depth(traces: dict, service: str, t_start: float, t_end: float) -> int:
    """Deepest trace stored under `service` with a span in [t_start, t_end] (trace structure feature)."""
    depths = [_max_depth(t) for t in traces.get(service, [])
              if any(t_start <= _span_start_s(s) <= t_end for s in t.get("spans", []))]
    return max(depths, default=0)


def _latency_delta(lat: float | None, base_lat: float | None) -> float | None:
    return (lat - base_lat) / base_lat if (lat is not None and base_lat is not None and base_lat > 0) else None


# Severity label

def _compute_severity(latency_delta: float | None, error_rate: float,
                      restart_count: float | None, replicas: float | None) -> str:
    """
    Observation-based severity label derived entirely from runtime signals,
    not from fault configuration inputs.

    Any High condition gives High, otherwise any Medium condition gives Medium, otherwise Low:
      High:   replicas == 0, restart_count > 0, latency_delta > 1.0, error_rate > 5%
      Medium: latency_delta > 0.2, error_rate > 1%
    latency_delta is None when there were too few requests to judge it (see span_measures.py).
    """
    if (
        (replicas is not None and replicas == 0) or
        (restart_count is not None and restart_count > 0) or
        (latency_delta is not None and latency_delta > LATENCY_DELTA_HIGH) or
        (error_rate is not None and error_rate > ERROR_RATE_HIGH)
    ):
        return "High"

    if (
        (latency_delta is not None and latency_delta > LATENCY_DELTA_MEDIUM) or
        (error_rate is not None and error_rate > ERROR_RATE_MEDIUM)
    ):
        return "Medium"

    return "Low"


# Per-experiment processing

def process_experiment(exp_dir: Path) -> tuple[list[dict], dict | None]:
    """-> (one row per snapshot T, one record of the measurements behind this run's labels)."""
    meta_path = exp_dir / "metadata.json"
    if not meta_path.exists():
        logger.warning("Skipping %s — no metadata.json", exp_dir.name)
        return [], None

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
    spans  = parse_spans(traces)
    reqs, source = target_view(spans, service)    # requests to the changed service (see span_measures)
    base_win = (base_start, base_end)

    # Baseline stats
    base_prom  = _prom_baseline(prom, service, base_start, base_end)
    base_feats = summary(window(reqs, *base_win), base_end - base_start)
    base_lat   = base_feats["latency_mean_ms"]

    # Label window (last 60s of fault phase)
    label_start  = max(fault_end - LABEL_WINDOW, fault_start)
    final_feats  = summary(window(reqs, label_start, fault_end), fault_end - label_start)
    final_lat    = final_feats["latency_mean_ms"]
    latency_measurable = enough_requests(base_feats["n"], final_feats["n"], alpha=SEVERITY_ALPHA)
    final_lat_delta    = _latency_delta(final_lat, base_lat) if latency_measurable else None
    final_error_rate   = final_feats["error_rate"]
    final_prom         = _prom_snapshot(prom, service, fault_end)
    final_restart      = final_prom.get("restart_count")
    final_replicas     = final_prom.get("replicas")
    blast_final        = blast_radius(spans, service, base_win, (label_start, fault_end))
    blast_radius_final = blast_final["count"]
    severity           = _compute_severity(
        final_lat_delta, final_error_rate, final_restart, final_replicas
    )
    record = {
        "experiment_id": meta["experiment_id"], "service": service, "fault_type": fault_type,
        "fault_value": fault_value, "traffic_level": traffic, "target_source": source,
        "n_base": base_feats["n"], "n_final": final_feats["n"], "latency_measurable": latency_measurable,
        "base_latency_mean_ms": base_lat, "final_latency_mean_ms": final_lat,
        "final_latency_delta": final_lat_delta, "final_error_rate": final_error_rate,
        "final_restart_count": final_restart, "final_replicas": final_replicas, "severity": severity,
        "blast_radius_final": blast_radius_final,
        "blast_affected": ";".join(blast_final["affected"]),
        "blast_unjudged": ";".join(blast_final["unjudged"]),
    }

    # One row per T
    rows = []
    for T in SNAPSHOT_OFFSETS:
        snap_time = fault_start + T
        if snap_time > fault_end:
            continue

        p_snap    = _prom_snapshot(prom, service, snap_time)
        sys_snap  = _system_snapshot(prom, snap_time)
        # early-window features: whatever requests were seen so far (no minimum: they are evidence, not labels)
        t_feats   = summary(window(reqs, fault_start, snap_time), T)

        cpu      = p_snap.get("cpu_usage")
        base_cpu = base_prom.get("baseline_cpu")
        mem      = p_snap.get("memory_bytes")
        base_mem = base_prom.get("baseline_memory")
        lat      = t_feats["latency_mean_ms"]

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
            "target_request_rate":   t_feats["request_rate"],
            "target_error_rate":     t_feats["error_rate"],
            "target_latency_mean_ms": lat,
            "target_latency_delta":  _latency_delta(lat, base_lat),
            "blast_radius_early":    blast_radius(spans, service, base_win, (fault_start, snap_time))["count"],
            "max_trace_depth":       _max_trace_depth(traces, service, fault_start, snap_time),

            # Labels (from final window - same for all 4 rows of this experiment)
            "severity":              severity,
            "blast_radius_final":    blast_radius_final,
            "final_latency_delta":   final_lat_delta,
            "latency_observable":    service not in NO_TRACE_SERVICES,
        })

    return rows, record


# Main

def main(raw_dir: Path, out_path: Path):
    exp_dirs = sorted(d for d in raw_dir.iterdir() if d.is_dir() and d.name.startswith("EXP_"))
    if not exp_dirs:
        logger.error("No EXP_* directories found in %s", raw_dir)
        return

    all_rows, records = [], []
    for exp_dir in exp_dirs:
        logger.info("Processing %s", exp_dir.name)
        try:
            rows, record = process_experiment(exp_dir)
            all_rows.extend(rows)
            if record:
                records.append(record)
            logger.info("  → %d rows", len(rows))
        except Exception as exc:
            logger.error("  FAILED: %s", exc)

    if not all_rows:
        logger.error("No rows generated.")
        return

    df = pd.DataFrame(all_rows)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    rec_path = out_path.parent / "label_measurements.csv"
    pd.DataFrame(records).to_csv(rec_path, index=False)
    logger.info(
        "Saved %d rows (%d experiments) → %s   (label measurements → %s)",
        len(df), df["experiment_id"].nunique(), out_path, rec_path.name,
    )
    logger.info("Severity distribution:\n%s", df["severity"].value_counts().to_string())
    logger.info("Fault type counts:\n%s", df.groupby("fault_type").size().to_string())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build proactive ML training dataset")
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW)
    parser.add_argument("--out",     type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    main(args.raw_dir, args.out)
