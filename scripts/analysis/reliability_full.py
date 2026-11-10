#!/usr/bin/env python
"""
Reliability of the midterm labels over all 244 runs: the validation behind the final measurement.

Re-measures every run with the midterm's root-span measurement and checks:
  a) that this copy gives the labels in midterm_labels.csv
  b) bootstrap: how often a run's label survives resampling its own traces
  c) no change vs change: baseline first half vs second half, against baseline vs the fault (30 s windows)
  d) the same between the two baselines of each repeated setting

Pass 1 reads every raw trace file once into cache.pkl; pass 2 works from the cache.
Usage: python scripts/analysis/reliability_full.py [output folder]   (default results/reliability/reliability_full/)
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

REPO = Path(__file__).resolve().parents[2]          # repository root
OUT = (Path(sys.argv[1]) if len(sys.argv) > 1 else REPO / "results" / "reliability" / "reliability_full").resolve()
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
MIDTERM_LABELS = REPO / "scripts" / "analysis" / "midterm_labels.csv"   # one row per run, root-span measurement
B = 200
SEV = ["Low", "Medium", "High"]
DEFS = [("mean", 0.10), ("mean", 0.25), ("mean", 0.50), ("median", 0.25)]   # blast-radius variants
rng = np.random.default_rng(42)


def trace_table(trs):
    starts, tidx, root_ms, n_spans, n_err = [], [], [], [], []
    for i, tr in enumerate(trs):
        spans = tr.get("spans", [])
        ids = {s["spanID"] for s in spans}
        referenced = {ref["spanID"] for s in spans for ref in s.get("references", []) if ref.get("spanID") in ids}
        roots = [s for s in spans if s["spanID"] not in referenced]
        root_ms.append(roots[0]["duration"] / 1000 if roots else np.nan)
        n_spans.append(len(spans))
        n_err.append(sum(1 for s in spans if _is_error_span(s)))
        for s in spans:
            starts.append(s["startTime"] / 1e6)
            tidx.append(i)
    return (np.array(starts), np.array(tidx, dtype=np.int32), np.array(root_ms, dtype=np.float64),
            np.array(n_spans, dtype=np.int32), np.array(n_err, dtype=np.int32))


def in_window(tab, t0, t1):
    starts, tidx = tab[0], tab[1]
    return np.unique(tidx[(starts >= t0) & (starts <= t1)])


def summary(d):
    d = d[~np.isnan(d)]
    return {"n": len(d), "mean": float(d.mean()) if len(d) else None,
            "median": float(np.median(d)) if len(d) else None}


def pass1(exp_id):
    exp_dir = RAW / exp_id
    meta = json.loads((exp_dir / "metadata.json").read_text())
    ph = meta["phases"]
    bs, be, fs, fe = ph["baseline"]["start"], ph["baseline"]["end"], ph["fault"]["start"], ph["fault"]["end"]
    mid, ls = (bs + be) / 2, max(fe - 60, fs)
    target = meta["service"]
    with open(exp_dir / "traces.json") as f:
        traces = json.load(f)
    svcs = {}
    for svc in list(traces):
        tab = trace_table(traces.pop(svc))
        root = tab[2]
        sel_b, sel_f = in_window(tab, bs, be), in_window(tab, ls, fe)
        entry = {
            "base": root[sel_b].astype(np.float32), "final": root[sel_f].astype(np.float32),
            "h1": summary(root[in_window(tab, bs, mid)]), "h2": summary(root[in_window(tab, mid, be)]),
            "f30": summary(root[in_window(tab, fe - 30, fe)]),
        }
        if svc == target:
            entry["final_spans"], entry["final_err"] = tab[3][sel_f], tab[4][sel_f]
        svcs[svc] = entry
        del tab
    del traces
    gc.collect()
    prom = _load_prometheus(exp_dir)
    fin = _prom_snapshot(prom, target, fe)
    return {"service": target, "svcs": svcs, "restart": fin.get("restart_count"), "replicas": fin.get("replicas")}


# label calculations on cached arrays

def _stat(d, how):
    d = d[~np.isnan(d)]
    if not len(d):
        return None
    return float(d.mean()) if how == "mean" else float(np.median(d))


def sev_from(run, base_d, fin_d, spans, errs):
    b, f = _stat(base_d, "mean"), _stat(fin_d, "mean")
    delta = (f - b) / b if (b and b > 0 and f is not None) else None
    err = errs.sum() / max(spans.sum(), 1) if len(spans) else 0.0
    return _compute_severity(delta, err, run["restart"], run["replicas"])


def severity(run):
    st = run["svcs"].get(run["service"])
    if not st:
        return _compute_severity(None, 0.0, run["restart"], run["replicas"])
    return sev_from(run, st["base"], st["final"], st["final_spans"], st["final_err"])


def blast(run, how="mean", thr=0.10, base=None, fin=None):
    n = 0
    for svc, st in run["svcs"].items():
        if svc == run["service"]:
            continue
        b = _stat(st["base"] if base is None else base[svc], how)
        f = _stat(st["final"] if fin is None else fin[svc], how)
        if b and b > 0 and f is not None and (f - b) / b > thr:
            n += 1
    return n


def blast_summ(run, w0, w1, how, thr, other=None):
    """Blast radius between two cached 30 s summaries (other = second run for cross-run null)."""
    n = 0
    for svc, st in run["svcs"].items():
        if svc == run["service"]:
            continue
        s0 = st[w0]
        s1 = (other["svcs"].get(svc) or {}).get(w1) if other else st[w1]
        if not s1 or not s0[how] or s0[how] <= 0 or s1[how] is None:
            continue
        if (s1[how] - s0[how]) / s0[how] > thr:
            n += 1
    return n


def boot(arr):
    return arr[rng.integers(0, len(arr), len(arr))] if len(arr) else arr


def bootstrap(run):
    st = run["svcs"].get(run["service"])
    sevs, blasts = [], []
    for _ in range(B):
        if st:
            k = rng.integers(0, len(st["final"]), len(st["final"])) if len(st["final"]) else np.array([], int)
            sevs.append(sev_from(run, boot(st["base"]), st["final"][k], st["final_spans"][k], st["final_err"][k]))
        else:
            sevs.append(severity(run))
        base = {s: boot(v["base"]) for s, v in run["svcs"].items()}
        fin = {s: boot(v["final"]) for s, v in run["svcs"].items()}
        blasts.append(blast(run, base=base, fin=fin))
    return sevs, np.array(blasts)


def main():
    runs = pd.read_csv(MIDTERM_LABELS).set_index("experiment_id")
    runs["setting"] = (runs.service + "|" + runs.fault_type + "|" + runs.fault_value.astype(str)
                       + "|" + runs.traffic_level)

    cache = OUT / "cache.pkl"
    data = pickle.loads(cache.read_bytes()) if cache.exists() else {}
    todo = [e for e in runs.index if e not in data]
    t0 = time.time()
    for i, e in enumerate(todo, 1):
        try:
            data[e] = pass1(e)
        except MemoryError:
            print(f"  {e}: MemoryError — skipped", file=sys.stderr, flush=True)
            gc.collect()
            continue
        if i % 20 == 0 or i == len(todo):
            cache.write_bytes(pickle.dumps(data))
        print(f"  [{i}/{len(todo)}] {e} ({time.time() - t0:.0f}s)", file=sys.stderr, flush=True)
    cache.write_bytes(pickle.dumps(data))

    report = []

    def out(s=""):
        print(s, flush=True)
        report.append(s)

    ids = [e for e in runs.index if e in data]
    mism = [e for e in ids if severity(data[e]) != runs.loc[e, "severity"]
            or blast(data[e]) != runs.loc[e, "blast_radius_final"]]
    out(f"Runs analysed: {len(ids)} of {len(runs)}.  Reproduction check vs midterm_labels.csv: "
        + ("OK" if not mism else f"{len(mism)} mismatches {mism[:10]}"))

    # b) bootstrap stability
    rows = []
    for e in ids:
        sevs, blasts = bootstrap(data[e])
        lab, br = runs.loc[e, "severity"], runs.loc[e, "blast_radius_final"]
        st = data[e]["svcs"].get(data[e]["service"])
        rows.append({"run": e, "fault_type": runs.loc[e, "fault_type"], "severity": lab,
                     "sev_stability": np.mean([s == lab for s in sevs]),
                     "decided_by": ("replicas=0" if data[e]["replicas"] == 0 else
                                    "restart" if (data[e]["restart"] or 0) > 0 else
                                    "untraced" if not st else "latency/error"),
                     "blast": br, "blast_sd": blasts.std(), "blast_same": np.mean(blasts == br),
                     "base_n": len(st["base"]) if st else np.nan})
    R = pd.DataFrame(rows)
    R.to_csv(OUT / "bootstrap_per_run.csv", index=False)
    R["stable"] = R["sev_stability"] >= 0.9
    out(f"\n=== b) Bootstrap ({B} resamples of each run's own traces): does the severity label survive? ===")
    out(f"Stable severity (same label in ≥90% of resamples): {R.stable.sum()}/{len(R)} runs ({R.stable.mean():.0%})")
    out("By label:\n" + R.groupby("severity").agg(runs=("run", "size"), stable=("stable", "mean"),
                                                   mean_stability=("sev_stability", "mean"))
        .reindex(SEV).to_string(float_format=lambda x: f"{x:.2f}"))
    out("By what decided the label:\n" + R.groupby("decided_by").agg(runs=("run", "size"), stable=("stable", "mean"))
        .to_string(float_format=lambda x: f"{x:.2f}"))
    out("By fault type:\n" + R.groupby("fault_type").agg(runs=("run", "size"), stable=("stable", "mean"),
                                                          blast_sd=("blast_sd", "mean"),
                                                          blast_same=("blast_same", "mean"))
        .to_string(float_format=lambda x: f"{x:.2f}"))
    out(f"Blast radius: bootstrap SD {R.blast_sd.mean():.2f} services on average; the recorded value is "
        f"reproduced in {R.blast_same.mean():.0%} of resamples")

    # c) null vs signal at equal (30 s) window length
    out("\n=== c) No fault vs fault, both 30 s vs 30 s: blast radius counted ===")
    nr = []
    for e in ids:
        run = data[e]
        for how, thr in DEFS:
            nr.append({"run": e, "fault_type": runs.loc[e, "fault_type"], "def": f"{how} >{int(thr * 100)}%",
                       "null": blast_summ(run, "h1", "h2", how, thr),
                       "signal": blast_summ(run, "h1", "f30", how, thr)})
    NR = pd.DataFrame(nr)
    NR.to_csv(OUT / "null_vs_signal.csv", index=False)
    piv = NR.groupby(["def", "fault_type"])[["null", "signal"]].mean().unstack("fault_type")
    out(piv.to_string(float_format=lambda x: f"{x:.2f}"))
    tot = NR.groupby("def")[["null", "signal"]].mean()
    tot["signal_minus_null"] = tot.signal - tot.null
    out("All runs:\n" + tot.to_string(float_format=lambda x: f"{x:.2f}"))

    # target latency delta null vs signal (30 s)
    tn = []
    for e in ids:
        st = data[e]["svcs"].get(data[e]["service"])
        if st and st["h1"]["mean"] and st["h2"]["mean"] is not None and st["f30"]["mean"] is not None:
            tn.append({"fault_type": runs.loc[e, "fault_type"],
                       "null": st["h2"]["mean"] / st["h1"]["mean"] - 1, "signal": st["f30"]["mean"] / st["h1"]["mean"] - 1})
    TN = pd.DataFrame(tn)
    out("\nTarget latency delta (mean, 30 s windows): share of runs above the 20% / 100% thresholds")
    out(pd.DataFrame({
        "null >20%": TN.groupby("fault_type")["null"].apply(lambda s: (s > 0.2).mean()),
        "signal >20%": TN.groupby("fault_type")["signal"].apply(lambda s: (s > 0.2).mean()),
        "null >100%": TN.groupby("fault_type")["null"].apply(lambda s: (s > 1.0).mean()),
        "signal >100%": TN.groupby("fault_type")["signal"].apply(lambda s: (s > 1.0).mean()),
    }).to_string(float_format=lambda x: f"{x:.0%}"))

    # d) cross-run baseline null for the repeat pairs (60 s vs 60 s, both fault-free)
    rep = runs[runs.setting.map(runs.setting.value_counts()) == 2].reset_index().sort_values("experiment_id")
    cross = []
    for _, g in rep.groupby("setting"):
        a, b = list(g.experiment_id)
        if a in data and b in data:
            base_b = {s: v["base"] for s, v in data[b]["svcs"].items()}
            ra = data[a]
            common = {s: v for s, v in ra["svcs"].items() if s in base_b}
            cross.append(blast({"service": ra["service"], "svcs": common},
                               base={s: v["base"] for s, v in common.items()},
                               fin={s: base_b[s] for s in common}))
    out(f"\n=== d) Repeat pairs: blast radius between the two runs' fault-free baselines (60 s vs 60 s) ===\n"
        f"mean {np.mean(cross):.2f} services counted 'affected' with no fault in either window "
        f"(current definition), over {len(cross)} pairs; values {sorted(cross)}")

    (OUT / "report.txt").write_text("\n".join(report) + "\n")
    print(f"\nSaved to {OUT}", file=sys.stderr)


if __name__ == "__main__":
    main()
