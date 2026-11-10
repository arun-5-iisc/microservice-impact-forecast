#!/usr/bin/env python
"""
Do the 20 repeated settings get the same label in both runs, under the midterm root-span measurement
("current" below) and a few alternative definitions? Also checks each definition when nothing changed
(baseline first half vs second half).

Labels to reproduce: midterm_labels.csv. Pass 1 reads every raw trace file once into stats.pkl; pass 2 works
from it.
Usage: python scripts/analysis/reliability.py [output folder]   (default results/reliability/reliability/)
"""
import gc
import json
import logging
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, trim_mean
from sklearn.metrics import cohen_kappa_score

REPO = Path(__file__).resolve().parents[2]          # repository root
OUT = (Path(sys.argv[1]) if len(sys.argv) > 1 else REPO / "results" / "reliability" / "reliability").resolve()
OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(REPO / "scripts"))
from preprocess import _compute_severity, _load_prometheus, _prom_snapshot  # noqa: E402

def _is_error_span(span: dict) -> bool:
    """The midterm's error-span test, kept so this script measures what the midterm labels did."""
    for tag in span.get("tags", []):
        k, v = tag.get("key"), tag.get("value")
        if k == "otel.status_code" and str(v).upper() == "ERROR":
            return True
        if k == "error" and v is True:
            return True
    return False

logging.getLogger().setLevel(logging.WARNING)

RAW = REPO / "dataset" / "raw"
EXTRA_RUNS = ["EXP_0199", "EXP_0339", "EXP_0370", "EXP_0419"]   # flagged thin baselines
STATS = {
    "mean": np.mean,
    "median": np.median,
    "trim10": lambda a: trim_mean(a, 0.10),
}
SEV = ["Low", "Medium", "High"]


def pairs_from_dataset():
    runs = pd.read_csv(REPO / "scripts" / "analysis" / "midterm_labels.csv")   # one row per run
    runs["setting"] = (runs.service + "|" + runs.fault_type + "|" + runs.fault_value.astype(str)
                       + "|" + runs.traffic_level)
    rep = runs[runs.setting.map(runs.setting.value_counts()) == 2].sort_values(["setting", "experiment_id"])
    pairs = [tuple(g.experiment_id) for _, g in rep.groupby("setting")]
    return runs.set_index("experiment_id"), pairs


# Pass 1: per-run, per-service, per-window statistics from the raw traces

def trace_table(traces_for_svc):
    """Flat arrays: span start per span (with trace index), plus per-trace root duration and error counts."""
    starts, tidx, root_ms, n_spans, n_err = [], [], [], [], []
    for i, tr in enumerate(traces_for_svc):
        spans = tr.get("spans", [])
        ids = {s["spanID"] for s in spans}
        referenced = {ref["spanID"] for s in spans for ref in s.get("references", []) if ref.get("spanID") in ids}
        roots = [s for s in spans if s["spanID"] not in referenced]
        root_ms.append(roots[0]["duration"] / 1000 if roots else np.nan)   # same as preprocess._root_duration_ms
        n_spans.append(len(spans))
        n_err.append(sum(1 for s in spans if _is_error_span(s)))
        for s in spans:
            starts.append(s["startTime"] / 1e6)
            tidx.append(i)
    return (np.array(starts), np.array(tidx, dtype=int), np.array(root_ms),
            np.array(n_spans), np.array(n_err))


def window_stats(tab, t0, t1):
    """preprocess._trace_features selection: a trace is in the window if ANY span starts in it."""
    starts, tidx, root_ms, n_spans, n_err = tab
    sel = np.unique(tidx[(starts >= t0) & (starts <= t1)])
    d = root_ms[sel]
    d = d[~np.isnan(d)]
    out = {"n": len(d), "traces": len(sel),
           "err": n_err[sel].sum() / max(n_spans[sel].sum(), 1) if len(sel) else 0.0}
    for k, f in STATS.items():
        out[k] = float(f(d)) if len(d) else None
    return out


def run_stats(exp_id):
    exp_dir = RAW / exp_id
    meta = json.loads((exp_dir / "metadata.json").read_text())
    ph = meta["phases"]
    bs, be, fs, fe = ph["baseline"]["start"], ph["baseline"]["end"], ph["fault"]["start"], ph["fault"]["end"]
    windows = {
        "base": (bs, be), "base_h1": (bs, (bs + be) / 2), "base_h2": ((bs + be) / 2, be),
        "final60": (max(fe - 60, fs), fe), "final120": (max(fe - 120, fs), fe),
    }
    windows.update({f"early{T}": (fs, fs + T) for T in (15, 30, 45, 60)})
    with open(exp_dir / "traces.json") as f:
        traces = json.load(f)
    stats = {}
    for svc, trs in traces.items():
        tab = trace_table(trs)
        stats[svc] = {w: window_stats(tab, *iv) for w, iv in windows.items()}
    del traces
    gc.collect()
    prom = _load_prometheus(exp_dir)
    fin = _prom_snapshot(prom, meta["service"], fe)
    return {"service": meta["service"], "stats": stats,
            "restart": fin.get("restart_count"), "replicas": fin.get("replicas")}


# Label definitions

def delta(st, base_w, fin_w, stat, min_n):
    b, f = st[base_w], st[fin_w]
    if b["n"] < max(min_n, 1) or f["n"] < max(min_n, 1) or not b[stat] or b[stat] <= 0 or f[stat] is None:
        return None
    return (f[stat] - b[stat]) / b[stat]


def severity(run, stat="mean", min_n=1, win="final60"):
    st = run["stats"].get(run["service"])
    d = delta(st, "base", win, stat, min_n) if st else None
    err = st[win]["err"] if st else 0.0
    return _compute_severity(d, err, run["restart"], run["replicas"])


def blast(run, stat="mean", thr=0.10, min_n=1, win="final60", base_w="base"):
    n = 0
    for svc, st in run["stats"].items():
        if svc == run["service"]:
            continue
        d = delta(st, base_w, win, stat, min_n)
        if d is not None and d > thr:
            n += 1
    return n


def icc1(a, b):
    """One-way random-effects ICC(1,1) for k=2 measurements per setting."""
    x = np.column_stack([a, b]).astype(float)
    n, k = x.shape
    grand = x.mean()
    msb = k * ((x.mean(axis=1) - grand) ** 2).sum() / (n - 1)
    msw = ((x - x.mean(axis=1, keepdims=True)) ** 2).sum() / (n * (k - 1))
    return (msb - msw) / (msb + (k - 1) * msw) if (msb + msw) > 0 else np.nan


def main():
    runs_df, pairs = pairs_from_dataset()
    cache = OUT / "stats.pkl"
    if cache.exists():
        data = pickle.loads(cache.read_bytes())
    else:
        data = {}
        ids = [e for p in pairs for e in p] + EXTRA_RUNS
        t0 = time.time()
        for i, e in enumerate(ids, 1):
            data[e] = run_stats(e)
            print(f"  [{i}/{len(ids)}] {e} ({time.time() - t0:.0f}s)", file=sys.stderr, flush=True)
        cache.write_bytes(pickle.dumps(data))

    report = []

    def out(s=""):
        print(s, flush=True)
        report.append(s)

    # reproduction check: the midterm ("current") definition must give the labels in midterm_labels.csv
    mism = [(e, severity(data[e]), runs_df.loc[e, "severity"], blast(data[e]), runs_df.loc[e, "blast_radius_final"])
            for p in pairs for e in p
            if severity(data[e]) != runs_df.loc[e, "severity"] or blast(data[e]) != runs_df.loc[e, "blast_radius_final"]]
    out(f"Reproduction check (current definition vs midterm_labels.csv, {2 * len(pairs)} runs): "
        f"{'OK' if not mism else f'{len(mism)} MISMATCHES: {mism}'}")

    # 1. Per-pair detail, current definition
    out("\n=== 1. The 20 pairs under the current definition ===")
    rows = []
    for a, b in pairs:
        ra, rb = data[a], data[b]
        sa, sb = ra["stats"].get(ra["service"]), rb["stats"].get(rb["service"])

        def why(run, st):
            if run["replicas"] == 0:
                return "replicas=0"
            if run["restart"] and run["restart"] > 0:
                return f"restarts={int(run['restart'])}"
            if not st:
                return "not traced"
            d, e = delta(st, "base", "final60", "mean", 1), st["final60"]["err"]
            return f"lat {d:+.0%} err {e:.1%}" if d is not None else f"no lat, err {e:.1%}"
        rows.append({
            "setting": runs_df.loc[a, "service"].replace("service", "") + " " + runs_df.loc[a, "fault_type"] + " "
                       + str(runs_df.loc[a, "fault_value"]),
            "runs": f"{a[4:]}/{b[4:]}",
            "severity": f"{severity(ra)}/{severity(rb)}",
            "why (run 1 | run 2)": f"{why(ra, sa)} | {why(rb, sb)}",
            "base n (target)": f"{sa['base']['n']}/{sb['base']['n']}" if sa else "—",
            "blast": f"{blast(ra)}/{blast(rb)}",
        })
    out(pd.DataFrame(rows).to_string(index=False))

    # 2. Null behaviour: no fault, baseline first half vs second half
    out("\n=== 2. With NO fault: baseline 1st half vs 2nd half (30 s each), all cached runs ===")
    nulls = []
    for e, run in data.items():
        st = run["stats"].get(run["service"])
        for stat in STATS:
            d = delta(st, "base_h1", "base_h2", stat, 1) if st else None
            nulls.append({"run": e, "stat": stat, "target_null_delta": d,
                          **{f"null_blast_{int(t * 100)}": blast(run, stat, t, 1, "base_h2", "base_h1")
                             for t in (0.10, 0.25, 0.50)}})
    N = pd.DataFrame(nulls)
    summ = N.groupby("stat").agg(
        target_abs_delta_median=("target_null_delta", lambda s: s.abs().median()),
        target_over_20pct=("target_null_delta", lambda s: (s.dropna() > 0.20).mean()),
        null_blast_10=("null_blast_10", "mean"), null_blast_25=("null_blast_25", "mean"),
        null_blast_50=("null_blast_50", "mean"))
    out(summ.to_string(float_format=lambda x: f"{x:.2f}"))
    out("(target_over_20pct = share of runs whose target 'latency delta' already exceeds the 20% Medium "
        "threshold with no fault; null_blast_X = services counted as 'affected' at threshold X with no fault)")
    N.to_csv(OUT / "null_check.csv", index=False)

    # 3. Agreement of alternative definitions across the pairs
    out("\n=== 3. Severity: agreement between the two runs of each setting, per definition ===")
    sev_rows = []
    for stat in STATS:
        for min_n in (1, 20):
            for win in ("final60", "final120"):
                la = [severity(data[a], stat, min_n, win) for a, _ in pairs]
                lb = [severity(data[b], stat, min_n, win) for _, b in pairs]
                all_l = la + lb
                sev_rows.append({
                    "stat": stat, "min_n": min_n, "window": win,
                    "same_label": f"{sum(x == y for x, y in zip(la, lb))}/{len(pairs)}",
                    "kappa_w": cohen_kappa_score(la, lb, labels=SEV, weights="quadratic"),
                    "Low/Med/High (40 runs)": "/".join(str(all_l.count(c)) for c in SEV),
                    "current": stat == "mean" and min_n == 1 and win == "final60",
                })
    S = pd.DataFrame(sev_rows)
    S.to_csv(OUT / "severity_definitions.csv", index=False)
    out(S.to_string(index=False, float_format=lambda x: f"{x:.2f}"))

    out("\n=== 4. Blast radius: agreement between the two runs of each setting, per definition ===")
    br_rows = []
    for stat in ("mean", "median"):
        for thr in (0.10, 0.25, 0.50):
            for min_n in (1, 20):
                for win in ("final60", "final120"):
                    ba = np.array([blast(data[a], stat, thr, min_n, win) for a, _ in pairs])
                    bb = np.array([blast(data[b], stat, thr, min_n, win) for _, b in pairs])
                    null = N[N.stat == stat][f"null_blast_{int(thr * 100)}"].mean() if min_n == 1 else np.nan
                    br_rows.append({
                        "stat": stat, "thr": thr, "min_n": min_n, "window": win,
                        "mean_abs_diff": np.abs(ba - bb).mean(), "same": f"{(ba == bb).sum()}/{len(pairs)}",
                        "icc": icc1(ba, bb), "mean_value": np.r_[ba, bb].mean(),
                        "null_mean": null,
                        "current": stat == "mean" and thr == 0.10 and min_n == 1 and win == "final60",
                    })
    B = pd.DataFrame(br_rows)
    B.to_csv(OUT / "blast_definitions.csv", index=False)
    out(B.to_string(index=False, float_format=lambda x: f"{x:.2f}"))
    out("(icc = share of variance that is between settings rather than between runs: 1 = perfectly "
        "reproducible, 0 = pure noise; null_mean = services counted with no fault, baseline halves)")

    # 5. Early features (model inputs): do they reproduce?
    out("\n=== 5. Early target latency delta (model input) — rank correlation between runs, per T ===")
    feat_rows = []
    for T in (15, 30, 45, 60):
        for stat in ("mean", "median"):
            xa, xb = [], []
            for a, b in pairs:
                sa, sb = data[a]["stats"].get(data[a]["service"]), data[b]["stats"].get(data[b]["service"])
                if sa and sb:
                    da, db = delta(sa, "base", f"early{T}", stat, 1), delta(sb, "base", f"early{T}", stat, 1)
                    if da is not None and db is not None:
                        xa.append(da)
                        xb.append(db)
            feat_rows.append({"T": T, "stat": stat, "pairs": len(xa), "spearman": spearmanr(xa, xb)[0]})
    out(pd.DataFrame(feat_rows).to_string(index=False, float_format=lambda x: f"{x:.2f}"))

    # 6. Baseline sample sizes
    out("\n=== 6. Baseline sample size (traces with a root span in the 60 s baseline), target service ===")
    bn = []
    for e, run in data.items():
        st = run["stats"].get(run["service"])
        if st:
            bn.append({"run": e, "service": run["service"], "base_n": st["base"]["n"],
                       "min_base_n_any_service": min(s["base"]["n"] for s in run["stats"].values())})
    BN = pd.DataFrame(bn).sort_values("base_n")
    out(BN.head(12).to_string(index=False))
    BN.to_csv(OUT / "baseline_sizes.csv", index=False)

    (OUT / "report.txt").write_text("\n".join(report) + "\n")
    print(f"\nSaved to {OUT}", file=sys.stderr)


if __name__ == "__main__":
    main()
