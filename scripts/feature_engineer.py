"""
Feature engineering: adds four groups of features to the base training dataset.

Reads:  dataset/processed/training_dataset.csv    (from preprocess.py)
Writes: dataset/processed/training_dataset_fe.csv

1. Service topology      where the service sits in the Online Boutique call graph
                         (service_depth, service_downstream_count, service_upstream_count,
                         service_is_gateway)
2. Fault severity ratio  size of the change relative to the original limit (fault_severity_ratio)
3. Log transforms        log1p of skewed features (log_target_latency_mean_ms,
                         log_target_memory_bytes, log_fault_value_numeric) and a signed log of
                         the latency delta
4. System aggregates     error rate and latency ratio across all services (system_error_rate,
                         system_latency_ratio), measured with span_measures.py like the labels.
                         They need the raw traces; runs without them get NaN, which the
                         imputer in the training pipeline fills.

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

from span_measures import load_run_spans, system_aggregates

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


# Online Boutique call graph, checked against parent/child spans in the recorded traces (same graph
# as genai/replay.py). cartservice also calls Redis, which is outside the app and not traced.
CALL_GRAPH = {
    "frontend":              ["adservice", "cartservice", "checkoutservice", "currencyservice",
                              "productcatalogservice", "recommendationservice", "shippingservice"],
    "checkoutservice":       ["cartservice", "currencyservice", "emailservice", "paymentservice",
                              "productcatalogservice", "shippingservice"],
    "recommendationservice": ["productcatalogservice"],
    "adservice": [], "cartservice": [], "currencyservice": [], "emailservice": [],
    "paymentservice": [], "productcatalogservice": [], "shippingservice": [],
}


def _topology(graph: dict, entry: str = "frontend") -> dict:
    """
    Per service, computed from the call graph (not typed by hand):
      depth             position on the shortest call path from the entry point (frontend = 1)
      downstream_count  services it calls (fan-out)
      upstream_count    services that call it (fan-in)
      is_gateway        1 for the entry point that receives user traffic
    """
    depth, queue = {entry: 1}, [entry]
    while queue:
        svc = queue.pop(0)
        for callee in graph[svc]:
            if callee not in depth:
                depth[callee] = depth[svc] + 1
                queue.append(callee)
    return {svc: {"depth": depth[svc], "downstream_count": len(callees),
                  "upstream_count": sum(svc in c for c in graph.values()), "is_gateway": int(svc == entry)}
            for svc, callees in graph.items()}


SERVICE_TOPOLOGY = _topology(CALL_GRAPH)

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
    "paymentservice":        {"cpu": 200,  "memory": 256},
    "shippingservice":       {"cpu": 200,  "memory": 128},
}


# Feature 1: Service topology

def add_topology_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["service_depth"]            = df["service"].map(lambda s: SERVICE_TOPOLOGY.get(s, {}).get("depth"))
    df["service_downstream_count"] = df["service"].map(lambda s: SERVICE_TOPOLOGY.get(s, {}).get("downstream_count"))
    df["service_upstream_count"]   = df["service"].map(lambda s: SERVICE_TOPOLOGY.get(s, {}).get("upstream_count"))
    df["service_is_gateway"]       = df["service"].map(lambda s: SERVICE_TOPOLOGY.get(s, {}).get("is_gateway"))
    unknown = df["service_depth"].isna().sum()
    if unknown:
        logger.warning("%d rows have unknown service topology (NaN filled)", unknown)
    return df


# Feature 2: Fault severity ratio

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
            # fault_value_numeric is in Mi (strip from string "128Mi" to 128)
            return 1.0 - fval / original
    elif fault_type == "latency":
        # Normalise to [0, 1]: 1000ms toratio=1.0 (severe), 100ms to 0.1
        return min(fval / 1000.0, 1.0)
    elif fault_type == "replicas":
        # 0 replicas , ratio=1.0 (complete outage)
        return 1.0 - fval   # fval=0 to 1.0

    return None


def add_fault_severity_ratio(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["fault_severity_ratio"] = df.apply(_fault_severity_ratio, axis=1)
    return df


# Feature 3: Log transforms on skewed features

def add_log_transforms(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # NaN stays NaN for the median imputer. Filling with 0 first would make a missing value
    # look like a real zero.
    df["log_target_latency_mean_ms"] = df["target_latency_mean_ms"].apply(
        lambda x: np.log1p(x) if pd.notna(x) else np.nan
    )
    df["log_target_memory_bytes"] = df["target_memory_bytes"].apply(
        lambda x: np.log1p(x) if pd.notna(x) else np.nan
    )
    df["log_fault_value_numeric"] = df["fault_value_numeric"].apply(
        lambda x: np.log1p(x) if pd.notna(x) else np.nan
    )
    # The latency delta can be negative (faster than baseline), so log1p alone won't do:
    # sign(x) * log1p(|x|) squashes the very large High values and keeps the sign.
    df["signed_log_latency_delta"] = df["target_latency_delta"].apply(
        lambda x: np.sign(x) * np.log1p(abs(x)) if pd.notna(x) else np.nan
    )
    return df


# Feature 4: System-level trace aggregates from raw data

def _system_aggregates_for_run(exp_dir: Path, service: str, offsets) -> dict:
    """
    system_error_rate / system_latency_ratio at each snapshot T, for one run: mean error rate and
    mean latency ratio (early window [fault start, fault start + T] vs baseline) across all services,
    measured as in span_measures.py (changed service from its callers, others from their own spans).
    """
    meta_path = exp_dir / "metadata.json"
    if not meta_path.exists() or not (exp_dir / "traces.json").exists():
        return {T: {"system_error_rate": None, "system_latency_ratio": None} for T in offsets}
    meta = json.loads(meta_path.read_text())
    ph = meta["phases"]
    spans = load_run_spans(exp_dir)
    base = (ph["baseline"]["start"], ph["baseline"]["end"])
    fault_start = ph["fault"]["start"]
    return {T: system_aggregates(spans, service, base, (fault_start, fault_start + T)) for T in offsets}


def add_system_trace_aggregates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute system_error_rate and system_latency_ratio for every row.
    Reads each run's raw traces once. Runs without raw data get NaN.
    """
    df = df.copy()
    values: dict[tuple, dict] = {}
    for (exp_id, service), g in df.groupby(["experiment_id", "service"], sort=False):
        per_t = _system_aggregates_for_run(RAW_DIR / exp_id, service, sorted(g["time_into_fault_s"].unique()))
        values.update({(exp_id, T): v for T, v in per_t.items()})
    keys = list(zip(df["experiment_id"], df["time_into_fault_s"]))
    df["system_error_rate"]    = [values[k]["system_error_rate"] for k in keys]
    df["system_latency_ratio"] = [values[k]["system_latency_ratio"] for k in keys]
    return df


# Main

def main(in_path: Path, out_path: Path):
    logger.info("Loading baseline dataset from %s", in_path)
    df = pd.read_csv(in_path)
    n_start = len(df)
    logger.info("Loaded %d rows, %d experiments", n_start, df["experiment_id"].nunique())

    logger.info("Adding topology features…")
    df = add_topology_features(df)

    logger.info("Adding fault severity ratio…")
    df = add_fault_severity_ratio(df)

    logger.info("Adding log transforms…")
    df = add_log_transforms(df)

    if RAW_DIR.exists():
        logger.info("Adding system trace aggregates (reads raw traces — may take a moment)…")
        df = add_system_trace_aggregates(df)
    else:
        logger.warning(
            "Raw data directory %s not found — skipping system trace aggregates", RAW_DIR
        )
        df["system_error_rate"]    = np.nan
        df["system_latency_ratio"] = np.nan

    er_null  = df["system_error_rate"].isnull().sum()
    slr_null = df["system_latency_ratio"].isnull().sum()
    logger.info(
        "System trace aggregate coverage: system_error_rate null=%d/%d (%.0f%%), "
        "system_latency_ratio null=%d/%d (%.0f%%)",
        er_null,  len(df), 100 * er_null  / len(df),
        slr_null, len(df), 100 * slr_null / len(df),
    )
    if er_null > len(df) * 0.5:
        logger.warning(
            "system_error_rate is >50%% NaN — will be imputed with median (flat signal). "
            "Ensure raw trace data is present when running feature_engineer.py."
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)

    new_cols = [
        "service_depth", "service_downstream_count", "service_upstream_count", "service_is_gateway",
        "fault_severity_ratio",
        "log_target_latency_mean_ms", "log_target_memory_bytes", "log_fault_value_numeric",
        "signed_log_latency_delta",
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
