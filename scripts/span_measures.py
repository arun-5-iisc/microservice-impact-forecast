"""
Request-level measurements from the Jaeger traces.
Latency, error rate and "is this service affected?" are measured only here.
preprocess.py (labels and early features) and feature_engineer.py (system aggregates) both use it.
So features and labels are measured the same way.

What is measured
  Each span is one side of one request.
  Two views are kept for each service:
    own    its server spans: how long it took to answer the requests it received
    calls  its callers' client spans to it: how long callers waited, network included

  The changed service is measured from its callers (calls view).
  A network delay on its pod falls outside its own server span, so only the callers see it.
  adservice, cartservice and shippingservice export no spans (Online Boutique v0.10.5).
  For them, the callers are the only view.
  frontend has no traced caller (the load generator is not traced).
  So frontend is measured from its own spans.

  Every other service is measured from its own server spans.
  These include the time it spent waiting on the changed service.
  If a service has no request spans of its own, its callers' view is used instead.
  Calls made by the changed service are left out of that view.
  Otherwise the caller's own delay would be blamed on the service.

  Health checks and telemetry export are not user requests.
  They are always left out.

When a change counts (blast radius)
  A service is affected if, in the window compared with the baseline, it is:
    slower:    median up by at least MIN_LATENCY_INCREASE (20%) and MIN_LATENCY_INCREASE_MS (1 ms),
               AND Mann-Whitney one-sided p < ALPHA, or
    erroring:  error rate up by at least MIN_ERROR_INCREASE,
               AND Fisher exact one-sided p < ALPHA.
  It also needs enough requests to tell.
  With n_base and n_win requests, the most extreme outcome has probability 1 / C(n_base + n_win, n_win)
  if nothing changed. This must be below ALPHA.
  Example: 5 + 5 requests gives 0.004, enough. 4 + 4 gives 0.014, not enough.
  If there are too few requests, the result is "can't tell".

When the changed service's latency counts (severity)
  The latency thresholds of the severity rule apply only if there are enough requests.
  The check is the same, but at SEVERITY_ALPHA (5%), because it is a single test.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from math import comb
from pathlib import Path

import numpy as np
from scipy.stats import fisher_exact, mannwhitneyu

SERVICES = ["adservice", "cartservice", "checkoutservice", "currencyservice", "emailservice", "frontend",
            "paymentservice", "productcatalogservice", "recommendationservice", "shippingservice"]

# gRPC service name in client span operation names
GRPC_TO_SERVICE = {"AdService": "adservice", "CartService": "cartservice", "CheckoutService": "checkoutservice",
                   "CurrencyService": "currencyservice", "EmailService": "emailservice",
                   "PaymentService": "paymentservice", "ProductCatalogService": "productcatalogservice",
                   "RecommendationService": "recommendationservice", "ShippingService": "shippingservice"}

EXCLUDED = ("health", "opentelemetry")      # Kubernetes probes, gRPC health checks, telemetry export
PATH_TAGS = ("url.path", "http.target", "http.route")

ALPHA = 0.01                 
SEVERITY_ALPHA = 0.05        
MIN_LATENCY_INCREASE = 0.20  # same as the severity rule's Medium latency threshold
MIN_LATENCY_INCREASE_MS = 1.0 
MIN_ERROR_INCREASE = 0.01    # same as the severity rule's Medium error-rate threshold


# Parsing

def _tags(span: dict) -> dict:
    return {t.get("key"): t.get("value") for t in span.get("tags", [])}


def _is_error(tags: dict) -> bool:
    if str(tags.get("otel.status_code", "")).upper() == "ERROR" or tags.get("error") is True:
        return True
    code = tags.get("http.response.status_code", tags.get("http.status_code"))
    try:
        return code is not None and int(code) >= 500
    except (TypeError, ValueError):
        return False


def _is_excluded(op: str, tags: dict) -> bool:
    text = " ".join([op] + [str(tags[k]) for k in PATH_TAGS if k in tags]).lower()
    return any(x in text for x in EXCLUDED)


def _callee(op: str) -> str | None:
    head = op.split("/")[-2] if op.count("/") >= 1 else ""
    return GRPC_TO_SERVICE.get(head.rsplit(".", 1)[-1])


@dataclass
class RunSpans:
    """Per service: own server spans and callers' client spans, each an (n, 3) array of
    [start time (s), duration (ms), error (0/1)]; callers[svc] names who made each call."""
    own: dict[str, np.ndarray] = field(default_factory=dict)
    calls: dict[str, np.ndarray] = field(default_factory=dict)
    callers: dict[str, np.ndarray] = field(default_factory=dict)


def parse_spans(traces: dict) -> RunSpans:
    """traces.json content ({service: [trace, ...]}) -> RunSpans. Each span is counted once."""
    own: dict[str, list] = {s: [] for s in SERVICES}
    calls: dict[str, list] = {s: [] for s in SERVICES}
    callers: dict[str, list] = {s: [] for s in SERVICES}
    seen: set[str] = set()
    for trace_list in traces.values():
        for trace in trace_list:
            procs = trace.get("processes", {})
            for s in trace.get("spans", []):
                if s["spanID"] in seen:
                    continue    # the same trace is stored under every service it touches
                seen.add(s["spanID"])
                tags = _tags(s)
                op = s.get("operationName", "")
                if _is_excluded(op, tags):
                    continue
                rec = (s["startTime"] / 1e6, s["duration"] / 1000, float(_is_error(tags)))
                svc = procs.get(s.get("processID"), {}).get("serviceName")
                kind = tags.get("span.kind")
                if kind == "server" and svc in own:
                    own[svc].append(rec)
                elif kind == "client":
                    target = _callee(op)
                    if target:
                        calls[target].append(rec)
                        callers[target].append(svc or "")

    def arr(x):
        return np.array(x, dtype=float).reshape(-1, 3)
    return RunSpans(own={s: arr(v) for s, v in own.items()},
                    calls={s: arr(v) for s, v in calls.items()},
                    callers={s: np.array(v, dtype=object) for s, v in callers.items()})


def load_run_spans(exp_dir: Path) -> RunSpans:
    path = Path(exp_dir) / "traces.json"
    if not path.exists():
        return parse_spans({})
    with open(path) as f:
        traces = json.load(f)
    return parse_spans(traces)


# Views

def target_view(rs: RunSpans, service: str) -> tuple[np.ndarray, str]:
    """Requests to the CHANGED service: as its callers saw them (frontend: its own spans)."""
    if service == "frontend":
        return rs.own[service], "own"
    return rs.calls[service], "callers"


def other_view(rs: RunSpans, service: str, changed: str) -> tuple[np.ndarray, str]:
    """Requests of ANOTHER service: its own spans, or its callers' (without the changed service's calls)."""
    if len(rs.own[service]):
        return rs.own[service], "own"
    keep = rs.callers[service] != changed
    return rs.calls[service][keep], "callers"


def window(a: np.ndarray, t0: float, t1: float) -> np.ndarray:
    return a[(a[:, 0] >= t0) & (a[:, 0] <= t1)]


# Measures

def summary(a: np.ndarray, window_s: float) -> dict:
    """Request count, rate, mean latency and error rate of the requests in one window."""
    n = len(a)
    return {"n": n,
            "request_rate": n / max(window_s, 1),
            "latency_mean_ms": float(a[:, 1].mean()) if n else None,
            "error_rate": float(a[:, 2].mean()) if n else None}


def enough_requests(n_base: int, n_win: int, alpha: float = ALPHA) -> bool:
    """Could a clear change be told apart from chance at level alpha with these many requests?"""
    return n_base > 0 and n_win > 0 and 1 / comb(n_base + n_win, n_win) < alpha


def compare(base: np.ndarray, win: np.ndarray) -> dict:
    """Is `win` significantly slower or more erroring than `base`? See the module docstring."""
    nb, nw = len(base), len(win)
    out = {"n_base": nb, "n_win": nw, "judged": enough_requests(nb, nw), "slower": False, "erroring": False,
           "median_ratio": None, "error_increase": None}
    if not out["judged"]:
        return out
    mb, mw = float(np.median(base[:, 1])), float(np.median(win[:, 1]))
    if mb > 0:
        out["median_ratio"] = mw / mb
        if mw > (1 + MIN_LATENCY_INCREASE) * mb and mw - mb >= MIN_LATENCY_INCREASE_MS:
            out["slower"] = bool(mannwhitneyu(win[:, 1], base[:, 1], alternative="greater").pvalue < ALPHA)
    eb, ew = int(base[:, 2].sum()), int(win[:, 2].sum())
    out["error_increase"] = ew / nw - eb / nb
    if out["error_increase"] > MIN_ERROR_INCREASE:
        p = fisher_exact([[ew, nw - ew], [eb, nb - eb]], alternative="greater")[1]
        out["erroring"] = bool(p < ALPHA)
    return out


def blast_radius(rs: RunSpans, changed: str, base: tuple[float, float], win: tuple[float, float]) -> dict:
    """Other services significantly slower or more erroring in `win` than in `base`."""
    affected, unjudged, detail = [], [], {}
    for svc in SERVICES:
        if svc == changed:
            continue
        a, _ = other_view(rs, svc, changed)
        c = compare(window(a, *base), window(a, *win))
        detail[svc] = c
        if not c["judged"]:
            unjudged.append(svc)
        elif c["slower"] or c["erroring"]:
            affected.append(svc)
    return {"count": len(affected), "affected": affected, "unjudged": unjudged, "detail": detail}


def system_aggregates(rs: RunSpans, changed: str, base: tuple[float, float], win: tuple[float, float]) -> dict:
    """Mean error rate and mean latency ratio (window / baseline) across all services with requests."""
    errors, ratios = [], []
    for svc in SERVICES:
        a, _ = target_view(rs, svc) if svc == changed else other_view(rs, svc, changed)
        b, w = window(a, *base), window(a, *win)
        if len(w):
            errors.append(float(w[:, 2].mean()))
        if len(w) and len(b) and b[:, 1].mean() > 0:
            ratios.append(float(w[:, 1].mean() / b[:, 1].mean()))
    return {"system_error_rate": float(np.mean(errors)) if errors else None,
            "system_latency_ratio": float(np.mean(ratios)) if ratios else None}
