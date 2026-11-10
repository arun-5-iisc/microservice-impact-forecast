#!/usr/bin/env python
"""
Candidate blast-radius rules against no-change noise.

Latency here comes from each service's own server spans (health checks and telemetry export left out), not
from the root span of every trace it appears in. A service counts as affected when:
  D0  midterm rule: mean root-span duration more than 10% up (labels in midterm_labels.csv)
  D1  mean of its own spans more than 10% up
  D2  one-sided Mann-Whitney p < 0.01 and median more than 20% up, with at least MIN_N spans per window
      (the basis of the final rule in span_measures.py)

Reads reliability_full/cache.pkl, so run reliability_full.py first.
Usage: python scripts/analysis/blast_v2.py [output folder]   (default results/reliability/blast_v2/)
"""
import gc
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

REPO = Path(__file__).resolve().parents[2]          # repository root
OUT = (Path(sys.argv[1]) if len(sys.argv) > 1 else REPO / "results" / "reliability" / "blast_v2").resolve()
OUT.mkdir(parents=True, exist_ok=True)
RAW = REPO / "dataset" / "raw"
OLD_CACHE = OUT.parent / "reliability_full" / "cache.pkl"
MIDTERM_LABELS = REPO / "scripts" / "analysis" / "midterm_labels.csv"   # one row per run, root-span measurement
MIN_N = 10
EXCLUDE = ("health", "opentelemetry")


def kind(span):
    return next((t["value"] for t in span.get("tags", []) if t.get("key") == "span.kind"), None)


def pass1(exp_id):
    meta = json.loads((RAW / exp_id / "metadata.json").read_text())
    ph = meta["phases"]
    bs, be, fs, fe = ph["baseline"]["start"], ph["baseline"]["end"], ph["fault"]["start"], ph["fault"]["end"]
    with open(RAW / exp_id / "traces.json") as f:
        traces = json.load(f)
    spans, seen = {}, set()
    for key in list(traces):
        for tr in traces.pop(key):
            procs = tr.get("processes", {})
            for s in tr["spans"]:
                if s["spanID"] in seen or kind(s) != "server":
                    continue
                if any(x in s["operationName"].lower() for x in EXCLUDE):
                    continue
                seen.add(s["spanID"])
                svc = procs.get(s["processID"], {}).get("serviceName")
                t = s["startTime"] / 1e6
                if bs <= t <= fe:
                    spans.setdefault(svc, []).append((t, s["duration"] / 1000))
    del traces
    gc.collect()
    win = {"base": (bs, be), "h1": (bs, (bs + be) / 2), "h2": ((bs + be) / 2, be),
           "final": (max(fe - 60, fs), fe), "f30": (fe - 30, fe)}
    out = {}
    for svc, lst in spans.items():
        a = np.array(lst)
        out[svc] = {w: a[(a[:, 0] >= t0) & (a[:, 0] <= t1), 1].astype(np.float32) for w, (t0, t1) in win.items()}
    return {"service": meta["service"], "fault_type": meta["fault_type"], "svcs": out}


def affected(b, f, how):
    if how == "D1":
        return len(b) > 0 and len(f) > 0 and b.mean() > 0 and f.mean() / b.mean() - 1 > 0.10
    if len(b) < MIN_N or len(f) < MIN_N or np.median(b) <= 0:
        return False
    return (np.median(f) / np.median(b) - 1 > 0.20 and
            mannwhitneyu(f, b, alternative="greater").pvalue < 0.01)


def blast(run, w0, w1, how, other=None):
    n = 0
    for svc, st in run["svcs"].items():
        if svc == run["service"]:
            continue
        b = st[w0]
        f = (other["svcs"].get(svc) or {}).get(w1) if other else st[w1]
        if f is not None and affected(b, f, how):
            n += 1
    return n


def icc1(a, b):
    x = np.column_stack([a, b]).astype(float)
    n, k = x.shape
    grand = x.mean()
    msb = k * ((x.mean(axis=1) - grand) ** 2).sum() / (n - 1)
    msw = ((x - x.mean(axis=1, keepdims=True)) ** 2).sum() / (n * (k - 1))
    return (msb - msw) / (msb + (k - 1) * msw) if (msb + msw) > 0 else np.nan


def main():
    runs = pd.read_csv(MIDTERM_LABELS).set_index("experiment_id")
    runs["setting"] = (runs.service + "|" + runs.fault_type + "|" + runs.fault_value.astype(str)
                       + "|" + runs.traffic_level)
    cache = OUT / "cache.pkl"
    data = pickle.loads(cache.read_bytes()) if cache.exists() else {}
    t0 = time.time()
    todo = [e for e in runs.index if e not in data]
    for i, e in enumerate(todo, 1):
        data[e] = pass1(e)
        if i % 25 == 0 or i == len(todo):
            cache.write_bytes(pickle.dumps(data))
        print(f"  [{i}/{len(todo)}] {e} ({time.time() - t0:.0f}s)", file=sys.stderr, flush=True)

    report = []

    def out(s=""):
        print(s, flush=True)
        report.append(s)

    ids = list(runs.index)
    # Sample sizes: real requests per service in the 60 s baseline
    n_base = pd.DataFrame([{svc: len(st["base"]) for svc, st in data[e]["svcs"].items()} for e in ids])
    out("Real (non-health-check) server requests per service in the 60 s baseline, median over runs:")
    out(n_base.median().sort_values(ascending=False).astype(int).to_string())

    # No-fault vs fault at equal 30 s windows
    old = pickle.loads(OLD_CACHE.read_bytes())
    rows = []
    for e in ids:
        r = {"run": e, "fault_type": runs.loc[e, "fault_type"]}
        for how in ("D1", "D2"):
            r[f"{how}_null"] = blast(data[e], "h1", "h2", how)
            r[f"{how}_signal"] = blast(data[e], "h1", "f30", how)
            r[f"{how}_final60"] = blast(data[e], "base", "final", how)
        o = old[e]   # D0: the midterm definition (mean root duration, > 10%), 30 s windows
        r["D0_null"] = sum(1 for s, st in o["svcs"].items() if s != o["service"] and st["h1"]["mean"]
                           and st["h2"]["mean"] is not None and st["h2"]["mean"] / st["h1"]["mean"] - 1 > 0.10)
        r["D0_signal"] = sum(1 for s, st in o["svcs"].items() if s != o["service"] and st["h1"]["mean"]
                             and st["f30"]["mean"] is not None and st["f30"]["mean"] / st["h1"]["mean"] - 1 > 0.10)
        r["D0_final60"] = runs.loc[e, "blast_radius_final"]
        rows.append(r)
    R = pd.DataFrame(rows)
    R.to_csv(OUT / "per_run.csv", index=False)
    out("\n=== No fault vs fault, 30 s vs 30 s (mean services counted 'affected') ===")
    tab = pd.DataFrame({f"{d} {k}": R.groupby("fault_type")[f"{d}_{k}"].mean()
                        for d in ("D0", "D1", "D2") for k in ("null", "signal")})
    tab.loc["ALL"] = [R[c.replace(" ", "_")].mean() for c in tab.columns]
    out(tab.to_string(float_format=lambda x: f"{x:.2f}"))
    out("(D0 = current; D1 = own server spans of real requests, mean > 10%; "
        f"D2 = D1 spans + Mann-Whitney p < 0.01 + median > 20% + n >= {MIN_N})")

    out("\n=== Blast radius over the real 60 s windows, by fault type (mean / share of runs > 0) ===")
    out(pd.DataFrame({f"{d}": R.groupby("fault_type")[f"{d}_final60"].mean() for d in ("D0", "D1", "D2")})
        .to_string(float_format=lambda x: f"{x:.2f}"))
    out(pd.DataFrame({f"{d} >0": R.groupby("fault_type")[f"{d}_final60"].apply(lambda s: (s > 0).mean())
                      for d in ("D0", "D1", "D2")}).to_string(float_format=lambda x: f"{x:.0%}"))

    # Repeat pairs: agreement and cross-run baseline null (60 s vs 60 s, both fault-free)
    rep = runs[runs.setting.map(runs.setting.value_counts()) == 2].reset_index().sort_values("experiment_id")
    pairs = [tuple(g.experiment_id) for _, g in rep.groupby("setting")]
    out(f"\n=== Repeat pairs ({len(pairs)}) ===")
    Ri = R.set_index("run")
    for d in ("D0", "D1", "D2"):
        a = np.array([Ri.loc[x, f"{d}_final60"] for x, _ in pairs])
        b = np.array([Ri.loc[y, f"{d}_final60"] for _, y in pairs])
        cross = ([blast(data[x], "base", "base", d, other=data[y]) for x, y in pairs] if d != "D0" else None)
        out(f"{d}: same value {np.sum(a == b)}/{len(pairs)}, mean |diff| {np.abs(a - b).mean():.2f}, "
            f"ICC {icc1(a, b):.2f}, mean {np.r_[a, b].mean():.2f}"
            + (f", cross-run baseline null {np.mean(cross):.2f}" if cross is not None else ""))
    out("Per pair (D2): " + ", ".join(f"{runs.loc[x, 'service'].replace('service', '')} "
                                      f"{runs.loc[x, 'fault_type']} {runs.loc[x, 'fault_value']}: "
                                      f"{Ri.loc[x, 'D2_final60']}/{Ri.loc[y, 'D2_final60']}" for x, y in pairs))

    # Which services get counted under D2, and how often (60 s windows)
    cnt = {}
    for e in ids:
        run = data[e]
        for svc, st in run["svcs"].items():
            if svc != run["service"] and affected(st["base"], st["final"], "D2"):
                cnt[svc] = cnt.get(svc, 0) + 1
    out("\nD2: runs in which each service is counted as affected: " + str(dict(sorted(cnt.items(), key=lambda x: -x[1]))))

    (OUT / "report.txt").write_text("\n".join(report) + "\n")


if __name__ == "__main__":
    main()
