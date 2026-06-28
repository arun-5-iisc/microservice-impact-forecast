"""
Feature engineering layer — reads the baseline training dataset and produces an
enriched version with four groups of new features.

Reads:  dataset/processed/training_dataset.csv    (output of preprocess.py)
Writes: dataset/processed/training_dataset_fe.csv

Feature groups added
────────────────────
1. Service topology      — structural position in the Online Boutique call graph
                           (service_depth, service_downstream_count, service_is_gateway)
2. Fault severity ratio  — fault magnitude normalised by original resource limit
                           (fault_severity_ratio)
3. Log transforms        — log1p of skewed numerical features
                           (log_target_latency_mean_ms, log_target_memory_bytes,
                            log_fault_value_numeric)
4. System trace aggregates — cross-service error / latency signals
                           (system_error_rate, system_latency_ratio)
                           NOTE: these require the raw trace data; if not
                           computable they are filled with NaN (imputer handles them).

Usage:
  python scripts/feature_engineer.py
  python scripts/feature_engineer.py --in dataset/processed/training_dataset.csv \\
                                      --out dataset/processed/training_dataset_fe.csv
"""
import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger(__name__)

REPO_ROOT    = Path(__file__).parent.parent
DEFAULT_IN   = REPO_ROOT / "dataset" / "processed" / "training_dataset.csv"
DEFAULT_OUT  = REPO_ROOT / "dataset" / "processed" / "training_dataset_fe.csv"
RAW_DIR      = REPO_ROOT / "dataset" / "raw"


# ─────────────────────────────────────────────────────────────────────────────
# Static domain knowledge
# ─────────────────────────────────────────────────────────────────────────────

# Online Boutique call graph (static, from architecture docs)
# depth=1: entry points called directly by user traffic
# downstream_count: number of distinct services this service calls
SERVICE_TOPOLOGY = {
    "frontend":              {"depth": 1, "downstream_count": 6, "is_gateway": 1},
    "checkoutservice":       {"depth": 2, "downstream_count": 5, "is_gateway": 0},
    "recommendationservice": {"depth": 2, "downstream_count": 1, "is_gateway": 0},
    "productcatalogservice": {"depth": 2, "downstream_count": 0, "is_gateway": 0},
    "cartservice":           {"depth": 2, "downstream_count": 1, "is_gateway": 0},
    "paymentservice":        {"depth": 3, "downstream_count": 0, "is_gateway": 0},
    "emailservice":          {"depth": 3, "downstream_count": 0, "is_gateway": 0},
    "shippingservice":       {"depth": 3, "downstream_count": 0, "is_gateway": 0},
    "currencyservice":       {"depth": 3, "downstream_count": 0, "is_gateway": 0},
    "adservice":             {"depth": 1, "downstream_count": 0, "is_gateway": 0},
}

# Original Helm resource limits per service
# CPU in millicores (m), memory in MiB
ORIGINAL_LIMITS = {
    "recommendationservice": {"cpu": 200,  "memory": 450},
    "productcatalogservice": {"cpu": 200,  "memory": 128},
    "checkoutservice":       {"cpu": 200,  "memory": 128},
    "frontend":              {"cpu": 200,  "memory": 128},
    "cartservice":           {"cpu": 300,  "memory": 256},
    "adservice":             {"cpu": 300,  "memory": 300},
    "currencyservice":       {"cpu": 200,  "memory": 256},
    "emailservice":          {"cpu": 200,  "memory": 128},
    "paymentservice":        {"cpu": 200,  "memory": 128},
    "shippingservice":       {"cpu": 200,  "memory": 128},
}


# ─────────────────────────────────────────────────────────────────────────────
# Feature 1: Service topology
# ─────────────────────────────────────────────────────────────────────────────

def add_topology_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["service_depth"]            = df["service"].map(lambda s: SERVICE_TOPOLOGY.get(s, {}).get("depth"))
    df["service_downstream_count"] = df["service"].map(lambda s: SERVICE_TOPOLOGY.get(s, {}).get("downstream_count"))
    df["service_is_gateway"]       = df["service"].map(lambda s: SERVICE_TOPOLOGY.get(s, {}).get("is_gateway"))
    unknown = df["service_depth"].isna().sum()
    if unknown:
        logger.warning("%d rows have unknown service topology (NaN filled)", unknown)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Feature 2: Fault severity ratio
# ─────────────────────────────────────────────────────────────────────────────

def _fault_severity_ratio(row) -> float | None:
    svc        = row["service"]
    fault_type = row["fault_type"]
    fval       = row["fault_value_numeric"]
    if fval is None or np.isnan(fval):
        return None

    limits = ORIGINAL_LIMITS.get(svc, {})

    if fault_type == "cpu":
        original = limits.get("cpu")
        if original:
            # ratio = how much headroom is gone (0=no fault, 1=limit=0)
            return 1.0 - fval / original
    elif fault_type == "memory":
        original = limits.get("memory")
        if original:
            # fault_value_numeric is in Mi (strip from string "128Mi" → 128)
            return 1.0 - fval / original
    elif fault_type == "latency":
        # Normalise to [0, 1]: 1000ms → ratio=1.0 (severe), 100ms → 0.1
        return min(fval / 1000.0, 1.0)
    elif fault_type == "replicas":
        # 0 replicas → ratio=1.0 (complete outage)
        return 1.0 - fval   # fval=0 → 1.0

    return None


def add_fault_severity_ratio(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["fault_severity_ratio"] = df.apply(_fault_severity_ratio, axis=1)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Feature 3: Log transforms on skewed features
# ─────────────────────────────────────────────────────────────────────────────

def add_log_transforms(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["log_target_latency_mean_ms"] = np.log1p(df["target_latency_mean_ms"].fillna(0))
    df["log_target_memory_bytes"]    = np.log1p(df["target_memory_bytes"].fillna(0))
    df["log_fault_value_numeric"]    = np.log1p(df["fault_value_numeric"].fillna(0))
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Feature 4: System-level trace aggregates from raw data
# ─────────────────────────────────────────────────────────────────────────────

def _load_traces(exp_dir: Path) -> dict:
    path = exp_dir / "traces.json"
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)


def _span_start_s(span: dict) -> float:
    return span["startTime"] / 1_000_000


def _is_error_span(span: dict) -> bool:
    for tag in span.get("tags", []):
        k, v = tag.get("key"), tag.get("value")
        if k == "otel.status_code" and str(v).upper() == "ERROR":
            return True
        if k == "error" and v is True:
            return True
    return False


def _root_duration_ms(trace: dict) -> float | None:
    span_ids   = {s["spanID"] for s in trace["spans"]}
    referenced = {
        ref["spanID"]
        for s in trace["spans"]
        for ref in s.get("references", [])
        if ref.get("spanID") in span_ids
    }
    roots = [s for s in trace["spans"] if s["spanID"] not in referenced]
    if not roots:
        return None
    return roots[0]["duration"] / 1000


def _svc_error_rate(traces: dict, svc: str, t_start: float, t_end: float) -> float | None:
    svc_traces = traces.get(svc, [])
    in_window  = [
        t for t in svc_traces
        if any(t_start <= _span_start_s(s) <= t_end for s in t.get("spans", []))
    ]
    if not in_window:
        return None
    total  = sum(len(t["spans"]) for t in in_window)
    errors = sum(1 for t in in_window for s in t["spans"] if _is_error_span(s))
    return errors / max(total, 1)


def _svc_latency_mean(traces: dict, svc: str, t_start: float, t_end: float) -> float | None:
    svc_traces = traces.get(svc, [])
    in_window  = [
        t for t in svc_traces
        if any(t_start <= _span_start_s(s) <= t_end for s in t.get("spans", []))
    ]
    if not in_window:
        return None
    durations = [d for t in in_window if (d := _root_duration_ms(t)) is not None]
    return sum(durations) / len(durations) if durations else None


def _compute_system_trace_aggregates(exp_dir: Path, fault_start: float, snap_time: float) -> dict:
    """
    Compute mean error rate and latency ratio across ALL services in the early
    fault window [fault_start, snap_time].
    Returns system_error_rate and system_latency_ratio (vs. baseline).
    """
    traces = _load_traces(exp_dir)
    if not traces:
        return {"system_error_rate": None, "system_latency_ratio": None}

    meta_path = exp_dir / "metadata.json"
    if not meta_path.exists():
        return {"system_error_rate": None, "system_latency_ratio": None}
    with open(meta_path) as f:
        meta = json.load(f)
    base_start = meta["phases"]["baseline"]["start"]
    base_end   = meta["phases"]["baseline"]["end"]

    error_rates     = []
    latency_ratios  = []

    for svc in traces:
        er = _svc_error_rate(traces, svc, fault_start, snap_time)
        if er is not None:
            error_rates.append(er)

        fault_lat   = _svc_latency_mean(traces, svc, fault_start, snap_time)
        baseline_lat = _svc_latency_mean(traces, svc, base_start, base_end)
        if fault_lat is not None and baseline_lat and baseline_lat > 0:
            latency_ratios.append(fault_lat / baseline_lat)

    return {
        "system_error_rate":    np.mean(error_rates)   if error_rates    else None,
        "system_latency_ratio": np.mean(latency_ratios) if latency_ratios else None,
    }


def add_system_trace_aggregates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute system_error_rate and system_latency_ratio for every row.
    Reads raw trace files from RAW_DIR. Skipped rows get NaN.
    """
    df = df.copy()
    sys_error_rates    = []
    sys_latency_ratios = []

    # Cache per (experiment_id, time_into_fault_s) to avoid re-reading traces
    cache: dict[tuple, dict] = {}

    for _, row in df.iterrows():
        exp_id  = row["experiment_id"]
        T       = row["time_into_fault_s"]
        key     = (exp_id, T)

        if key not in cache:
            exp_dir = RAW_DIR / exp_id
            meta_path = exp_dir / "metadata.json"
            if meta_path.exists():
                with open(meta_path) as f:
                    meta = json.load(f)
                fault_start = meta["phases"]["fault"]["start"]
                snap_time   = fault_start + T
                cache[key]  = _compute_system_trace_aggregates(exp_dir, fault_start, snap_time)
            else:
                cache[key] = {"system_error_rate": None, "system_latency_ratio": None}

        result = cache[key]
        sys_error_rates.append(result["system_error_rate"])
        sys_latency_ratios.append(result["system_latency_ratio"])

    df["system_error_rate"]    = sys_error_rates
    df["system_latency_ratio"] = sys_latency_ratios
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main(in_path: Path, out_path: Path):
    logger.info("Loading baseline dataset from %s", in_path)
    df = pd.read_csv(in_path)
    n_start = len(df)
    logger.info("Loaded %d rows, %d experiments", n_start, df["experiment_id"].nunique())

    # Feature group 1: topology
    logger.info("Adding topology features…")
    df = add_topology_features(df)

    # Feature group 2: fault severity ratio
    logger.info("Adding fault severity ratio…")
    df = add_fault_severity_ratio(df)

    # Feature group 3: log transforms
    logger.info("Adding log transforms…")
    df = add_log_transforms(df)

    # Feature group 4: system trace aggregates (reads raw data)
    if RAW_DIR.exists():
        logger.info("Adding system trace aggregates (reads raw traces — may take a moment)…")
        df = add_system_trace_aggregates(df)
    else:
        logger.warning(
            "Raw data directory %s not found — skipping system trace aggregates", RAW_DIR
        )
        df["system_error_rate"]    = np.nan
        df["system_latency_ratio"] = np.nan

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)

    new_cols = [
        "service_depth", "service_downstream_count", "service_is_gateway",
        "fault_severity_ratio",
        "log_target_latency_mean_ms", "log_target_memory_bytes", "log_fault_value_numeric",
        "system_error_rate", "system_latency_ratio",
    ]
    logger.info("Saved %d rows → %s", len(df), out_path)
    logger.info("New columns added: %s", new_cols)
    logger.info("Null counts for new columns:\n%s",
                df[new_cols].isnull().sum().to_string())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Add feature engineering to training dataset")
    parser.add_argument("--in",  dest="in_path",  type=Path, default=DEFAULT_IN)
    parser.add_argument("--out", dest="out_path",  type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    main(args.in_path, args.out_path)
