#!/usr/bin/env python
"""
Draw the report's diagrams 

    [REPORT_DIR=<LaTeX project>] venv/bin/python scripts/report/make_diagrams.py [--preview DIR]      # --preview also saves PNG copies to DIR
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch, Polygon, Rectangle  # noqa: E402

# the LaTeX project: REPORT_DIR if set (e.g. a clone of the Overleaf project), otherwise report/ (not in git)
OUT = Path(os.environ.get("REPORT_DIR", Path(__file__).resolve().parents[2] / "report")).expanduser() / "figures"
PREVIEW: Path | None = None

INK, INK2, EDGE = "#0b0b0b", "#52514e", "#8a8985"
# light tints, only to group the parts of the system (grey: experimental setup, blue: data and models, orange: LLM, green: live)
GREY, BLUE, ORANGE, GREEN = "#f0efec", "#e6f0fc", "#fdece4", "#e2f4ec"
BLUE_LINE, ORANGE_LINE = "#2a78d6", "#eb6834"

plt.rcParams.update({"font.size": 8, "text.color": INK, "pdf.fonttype": 42,
                     "savefig.bbox": "tight", "savefig.pad_inches": 0.02})


UNIT_PT = 4.46   # points per drawing unit, set by canvas()


def canvas(width_in: float, height_units: float):
    """A blank drawing area 100 units wide; one unit is the same length in x and y."""
    global UNIT_PT
    UNIT_PT = width_in * 72 / 100
    fig = plt.figure(figsize=(width_in, width_in * height_units / 100))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 100); ax.set_ylim(0, height_units); ax.axis("off")
    return fig, ax


def box(ax, x, y, w, h, title, body=None, fc="white", ec=EDGE, ls="-", size=7.5):
    """A rounded box with a bold title and, optionally, a few lines of detail under it, all inside the box."""
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=1.2", fc=fc, ec=ec, lw=0.8, ls=ls))
    if body:
        bsize = size - 1
        lines = body.count("\n") + 1
        title_h, line_h, gap = size * 1.15 / UNIT_PT, bsize * 1.3 / UNIT_PT, 0.9
        top = y + h / 2 + (title_h + gap + lines * line_h) / 2          # centre the whole block vertically
        ax.text(x + w / 2, top, title, ha="center", va="top", fontsize=size, weight="bold")
        ax.text(x + w / 2, top - title_h - gap, body, ha="center", va="top", fontsize=bsize, color=INK2,
                linespacing=1.3)
    else:
        ax.text(x + w / 2, y + h / 2, title, ha="center", va="center", fontsize=size, weight="bold")
    return (x, y, w, h)


def arrow(ax, p, q, text=None, tpos=0.5, toff=(0, 1.4), color=INK2, rad=0.0, ls="-", ha="center"):
    ax.annotate("", xy=q, xytext=p, arrowprops=dict(arrowstyle="-|>", color=color, lw=0.9, ls=ls,
                                                    mutation_scale=8, shrinkA=0, shrinkB=0,
                                                    connectionstyle=f"arc3,rad={rad}"))
    if text:
        ax.text(p[0] + (q[0] - p[0]) * tpos + toff[0], p[1] + (q[1] - p[1]) * tpos + toff[1], text,
                ha=ha, va="bottom", fontsize=6.8, color=INK2)


def line(ax, pts, color=INK2, ls="-"):
    xs, ys = zip(*pts)
    ax.plot(xs, ys, color=color, lw=0.9, ls=ls, solid_capstyle="butt")


def save(fig, name):
    fig.savefig(OUT / f"{name}.pdf", metadata={"CreationDate": None})   # no date: same figure, same file
    if PREVIEW:
        fig.savefig(PREVIEW / f"{name}.png", dpi=220)
    plt.close(fig)


def architecture():
    fig, ax = canvas(6.2, 50)
    W, H = 18, 12.5
    xs = [0.5, 20.9, 41.3, 61.7, 82.1]
    top, mid, bot = 36, 20.5, 3
    tb = box(ax, xs[0], top, W, H, "Experimental setup", "microservice\napplication and\nobservability (GKE)", fc=GREY, size=7)
    hn = box(ax, xs[1], top, W, H, "Experiment harness", "applies and undoes\nplanned changes,\ncollects telemetry",
             fc=GREY, size=7)
    rr = box(ax, xs[2], top, W, H, "Recorded runs", "metadata, metrics,\ntraces and events\nof each run", fc=BLUE, size=7)
    lf = box(ax, xs[3], top, W, H, "Labels and features", "final severity and\nblast radius; early\nsignals at 15–60 s", fc=BLUE, size=7)
    md = box(ax, xs[4], top, W, H, "Forecasting models", "XGBoost: severity\nand blast radius,\nagainst baselines", fc=BLUE, size=7)
    lw = box(ax, xs[0], bot, W, H, "Live watcher", "detects a change,\nsnapshots the\ncluster", fc=GREEN, size=7)
    iv = box(ax, xs[1], bot, W, H, "Incident at time t", "a replayed run, or\nthe live snapshot,\nup to t only", fc=ORANGE, size=7)
    ts = box(ax, xs[2], bot, W, H, "MCP tool server", "change notice, call\ngraph, metrics,\nevents, latency", fc=ORANGE, size=7)
    ag = box(ax, xs[3], bot, W, H, "LLM agent", "investigates; Opus,\ngpt-6-sol or Haiku", fc=ORANGE, size=7)
    sl = box(ax, xs[4], bot, W, H, "Report and alert", "severity, cause,\naction, evidence;\nSlack in live mode", fc=GREEN, size=7)
    rg = box(ax, xs[2], mid, W, 9.5, "Retrieval", "past incidents\nand runbooks", fc=ORANGE, size=7)

    def r(b):  return (b[0] + b[2], b[1] + b[3] / 2)
    def l(b):  return (b[0], b[1] + b[3] / 2)
    def t(b, f=0.5):  return (b[0] + b[2] * f, b[1] + b[3])
    def b_(b, f=0.5): return (b[0] + b[2] * f, b[1])

    arrow(ax, (hn[0], top + H / 2 + 1.5), (tb[0] + W, top + H / 2 + 1.5))
    arrow(ax, (tb[0] + W, top + H / 2 - 1.5), (hn[0], top + H / 2 - 1.5))
    arrow(ax, r(hn), l(rr)); arrow(ax, r(rr), l(lf)); arrow(ax, r(lf), l(md))
    arrow(ax, b_(tb), t(lw), "watch", toff=(1, 0), ha="left")
    arrow(ax, r(lw), l(iv)); arrow(ax, r(iv), l(ts)); arrow(ax, r(ts), l(ag)); arrow(ax, r(ag), l(sl))
    arrow(ax, b_(rr, 0.2), (iv[0] + W * 0.75, bot + H), "replay", tpos=0.35, toff=(-1, 0), ha="right")
    arrow(ax, b_(rr, 0.5), t(rg), "training runs", toff=(1, -0.6), ha="left")
    arrow(ax, (rg[0] + W, mid + 4.5), t(ag, 0.3))
    arrow(ax, b_(md, 0.35), t(ag, 0.75), "forecast and gate", tpos=0.45, toff=(1.2, 0), ha="left")
    save(fig, "architecture")


def gated():
    fig, ax = canvas(6.2, 34)
    h, y = 10, 21
    a = box(ax, 0.5, y, 13, h, "Change", "detected at\ntime 0")
    b = box(ax, 18, y, 14, h, "Snapshot", "telemetry up to\nT = 15 s")
    c = box(ax, 36.5, y, 14, h, "ML screen", "forecast of the\nfinal severity", fc="#e6f0fc")
    dx, dy = 60.5, y + h / 2
    ax.add_patch(Polygon([(dx - 6.5, dy), (dx, dy + 6), (dx + 6.5, dy), (dx, dy - 6)], fc="white", ec=EDGE, lw=0.8))
    ax.text(dx, dy, "Medium\nor High?", ha="center", va="center", fontsize=7, weight="bold")
    e = box(ax, 71.5, y, 14, h, "LLM agent", "tools, past\nincidents and\nthe forecast", fc="#fdece4")
    f = box(ax, 88.5, y, 11.3, h, "Slack alert", "cause, action\nand evidence", fc="#e2f4ec")
    d2x, d2y = 60.5, 6.5
    ax.add_patch(Polygon([(d2x - 6.5, d2y), (d2x, d2y + 5.5), (d2x + 6.5, d2y), (d2x, d2y - 5.5)], fc="white", ec=EDGE, lw=0.8))
    ax.text(d2x, d2y, "T = 60 s?", ha="center", va="center", fontsize=7, weight="bold")
    g = box(ax, 72, 2, 27.5, 9, "Cleared", "no agent call; optional\nlow-impact note to Slack", fc="#f0efec")
    arrow(ax, (a[0] + 13, dy), (b[0], dy)); arrow(ax, (b[0] + 14, dy), (c[0], dy)); arrow(ax, (c[0] + 14, dy), (dx - 6.5, dy))
    arrow(ax, (dx + 6.5, dy), (e[0], dy), "yes", toff=(0, 0.6)); arrow(ax, (e[0] + 14, dy), (f[0], dy))
    arrow(ax, (dx, dy - 6), (d2x, d2y + 5.5), "no", tpos=0.4, toff=(1, -1), ha="left")
    arrow(ax, (d2x + 6.5, d2y), (g[0], d2y), "yes", toff=(0, 0.6))
    line(ax, [(d2x - 6.5, d2y), (25, d2y)])
    arrow(ax, (25, d2y), (25, y), "no: next snapshot, T + 15 s", tpos=0.0, toff=(1.5, 1.2), ha="left")
    save(fig, "gated")


def run_timeline():
    fig, ax = canvas(6.2, 25)
    t0, t1 = -60, 264
    X = lambda s: 3 + (s - t0) / (t1 - t0) * 94           # noqa: E731
    yb, hb = 8, 5
    for s0, s1, fc, name, at in ((-60, 0, GREY, "baseline, 60 s", -30), (0, 186, BLUE, "change held, about 186 s", 63),
                                 (186, 264, GREY, "recovery, about 78 s", 225)):
        ax.add_patch(Rectangle((X(s0), yb), X(s1) - X(s0), hb, fc=fc, ec=EDGE, lw=0.8))
        ax.text(X(at), yb + hb / 2, name, ha="center", va="center", fontsize=7.5)
    ax.add_patch(Rectangle((X(126), yb), X(186) - X(126), hb, fc="none", ec=BLUE_LINE, lw=1.2, hatch="////"))
    ax.annotate("label window:\nlast 60 s of the change", (X(156), yb + hb), xytext=(X(156), 21), ha="center",
                va="top", fontsize=7, color=INK2, arrowprops=dict(arrowstyle="-", color=INK2, lw=0.7))
    for s in (15, 30, 45, 60):
        ax.plot([X(s), X(s)], [yb + hb, yb + hb + 2.2], color=ORANGE_LINE, lw=1.2)
        ax.text(X(s), yb + hb + 2.6, f"{s}", ha="center", va="bottom", fontsize=7, color=INK2)
    ax.text(X(37.5), yb + hb + 6.4, "snapshots for the forecast (s)", ha="center", va="bottom", fontsize=7, color=INK2)
    for s, name in ((0, "change\napplied"), (186, "change\nundone")):
        ax.plot([X(s), X(s)], [yb - 1.6, yb], color=INK, lw=0.9)
        ax.text(X(s), yb - 2.0, name, ha="center", va="top", fontsize=7, color=INK2)
    ax.annotate("", xy=(X(264), 3.2), xytext=(X(220), 3.2), arrowprops=dict(arrowstyle="-|>", color=INK2, lw=0.8,
                                                                           mutation_scale=8))
    ax.text(X(217), 3.2, "time", ha="right", va="center", fontsize=7, color=INK2)
    save(fig, "run_timeline")


def callers_view():
    fig, ax = canvas(5.6, 30)
    # a call from a caller to the changed service, as two spans on one time line
    ax.text(2, 25.5, "caller\n(e.g. checkout)", ha="left", va="center", fontsize=7.5, weight="bold")
    ax.text(2, 12.5, "changed service\n(e.g. payment)", ha="left", va="center", fontsize=7.5, weight="bold")
    xs, xe, ss, se = 30, 92, 50, 70
    ax.add_patch(Rectangle((xs, 23.5), xe - xs, 4, fc=BLUE, ec=BLUE_LINE, lw=0.9))
    ax.text((xs + xe) / 2, 25.5, "client span: what the caller waited", ha="center", va="center", fontsize=7.5)
    ax.add_patch(Rectangle((ss, 10.5), se - ss, 4, fc=ORANGE, ec=ORANGE_LINE, lw=0.9))
    ax.text((ss + se) / 2, 12.5, "server span", ha="center", va="center", fontsize=7.5)
    arrow(ax, (xs + 1, 23.5), (ss, 14.6)); arrow(ax, (se, 14.6), (xe - 1, 23.5))
    ax.text(34, 17.5, "network\n(+ injected delay)", ha="right", va="center", fontsize=6.8, color=INK2)
    ax.text(84.5, 18.5, "network", ha="left", va="center", fontsize=6.8, color=INK2)
    ax.text(30, 4.5, "Callers' view (used for the changed service): the client span, so a network delay is included.\n"
                     "Own view: the server span, processing only; it misses the delay entirely.",
            ha="left", va="center", fontsize=7, color=INK2, linespacing=1.4)
    save(fig, "callers_view")


def call_graph():
    fig, ax = canvas(6.2, 63)
    w, h = 17, 6
    pos = {"load generator": (9.5, 29), "frontend": (29, 29), "recommendation": (51, 50), "checkout": (51, 10),
           "ad": (74, 56), "productcatalog": (74, 47), "currency": (74, 38), "cart": (74, 29), "shipping": (74, 20),
           "payment": (74, 11), "email": (74, 2), "Redis": (93.5, 29)}
    untraced = {"load generator", "ad", "cart", "shipping", "Redis"}
    edges = [("load generator", "frontend")] + \
            [("frontend", s) for s in ("recommendation", "checkout", "ad", "productcatalog", "currency", "cart", "shipping")] + \
            [("recommendation", "productcatalog")] + \
            [("checkout", s) for s in ("productcatalog", "currency", "cart", "shipping", "payment", "email")] + [("cart", "Redis")]
    width = {n: (12 if n == "Redis" else w) for n in pos}
    for a, b in edges:
        (xa, ya), (xb, yb) = pos[a], pos[b]
        p = (xa + width[a] / 2, ya + h / 2); q = (xb - width[b] / 2, yb + h / 2)
        arrow(ax, p, q, color="#9a9994")
    for n, (x, y) in pos.items():
        box(ax, x - width[n] / 2, y, width[n], h, n, ls="--" if n in untraced else "-",
            fc="white" if n in untraced else BLUE, size=7.5)
    ax.text(1, 58, "solid: exports spans of its own\ndashed: seen only through its callers", ha="left", va="center",
            fontsize=7, color=INK2, linespacing=1.4)
    save(fig, "call_graph")


def testbed():
    """What runs where: the application and the observability stack on the cluster, and the harness outside it."""
    fig, ax = canvas(6.2, 62)
    # the cluster
    ax.add_patch(FancyBboxPatch((23, 1.5), 76.5, 59, boxstyle="round,pad=0,rounding_size=1.5", fc="#fafaf8", ec=EDGE, lw=0.9))
    ax.text(24.5, 59, "GKE Standard cluster: 3 × e2-standard-2 nodes (2 vCPUs, 8 GB each)", ha="left", va="top",
            fontsize=7.5, weight="bold")
    # Online Boutique: Redis sits under the cart service, which is the only one that uses it
    ax.add_patch(FancyBboxPatch((41, 27.5), 57.5, 27.5, boxstyle="round,pad=0,rounding_size=1.2", fc=BLUE, ec=EDGE, lw=0.8))
    ax.text(69.75, 53.5, "Online Boutique: 10 services", ha="center", va="top", fontsize=7.5, weight="bold")
    services = [["frontend", "checkout", "recommendation", "productcatalog"],
                ["cart", "currency", "payment", "shipping"],
                [None, "email", "ad"]]
    w, h, gx, gy = 13.0, 5.0, 1.15, 1.6
    x0, y_top = 42.2, 48.5
    pos = {}
    for r, row in enumerate(services):
        for c, name in enumerate(row):
            x, y = x0 + c * (w + gx), y_top - r * (h + gy) - h
            pos[name or "redis"] = (x, y)
            if name:
                box(ax, x, y, w, h, name, size=5.9)
    rx, ry = pos["redis"]
    box(ax, rx, ry, w, h, "Redis (carts)", ls="--", size=5.9)
    cx, cy = pos["cart"]
    arrow(ax, (cx + w / 2, cy), (rx + w / 2, ry + h), color="#9a9994")
    # load generator, inside the cluster
    fx, fy = pos["frontend"]
    lg = box(ax, 25, fy + h / 2 - 7, 13.5, 12, "Load generator", "Locust: 10 or\n25 simulated\nusers, HTTP", size=6.6)
    arrow(ax, (lg[0] + lg[2], fy + h / 2), (fx, fy + h / 2))
    # observability
    ow, oh, oy, ox, og = 12.2, 10, 5.5, 26.5, 1.1
    ax.add_patch(FancyBboxPatch((25, 3.5), 4 * ow + 3 * og + 3, 18.5, boxstyle="round,pad=0,rounding_size=1.2",
                                fc=GREEN, ec=EDGE, lw=0.8))
    ax.text(25 + (4 * ow + 3 * og + 3) / 2, 20.7, "Observability", ha="center", va="top", fontsize=7.5, weight="bold")
    otel = box(ax, ox, oy, ow, oh, "OTel Collector", "receives\nthe spans", size=5.9)
    jae = box(ax, ox + (ow + og), oy, ow, oh, "Jaeger", "stores the\ntraces", size=5.9)
    box(ax, ox + 2 * (ow + og), oy, ow, oh, "Prometheus", "metrics\nevery 5 s", size=5.9)
    box(ax, ox + 3 * (ow + og), oy, ow, oh, "Kubernetes API", "events", size=5.9)
    arrow(ax, (otel[0] + ow, oy + oh / 2), (jae[0], oy + oh / 2))
    arrow(ax, (58, 27.5), (52, 22), "spans, metrics, events", tpos=0.5, toff=(1.2, -0.6), ha="left")
    # Chaos Mesh
    cm = box(ax, 82, 7, 16.5, 12, "Chaos Mesh", "adds a network\ndelay to one\nservice's traffic", fc=ORANGE, size=6.6)
    arrow(ax, (cm[0] + cm[2] / 2, cm[1] + cm[3]), (cm[0] + cm[2] / 2, 27.5), "delay", toff=(1.2, 0), ha="left")
    # outside: the harness and the storage
    hn = box(ax, 0.5, 30, 19, 24, "Experiment harness", "applies and undoes\neach change with\nHelm, kubectl or\nChaos Mesh, then\ncollects the\ntelemetry", fc=GREY, size=7)
    gcs = box(ax, 0.5, 3.5, 19, 15, "Cloud Storage", "recorded runs:\nmetadata, metrics,\ntraces, events", fc=GREY, size=7)
    arrow(ax, (hn[0] + hn[2], 52), (41, 52), "change", tpos=0.35, toff=(0, 0.6))
    arrow(ax, (25, 15), (hn[0] + hn[2], 34), "telemetry", tpos=0.55, toff=(1.2, 0), ha="left")
    arrow(ax, (10, 30), (10, 18.5), "runs", toff=(1.2, 0), ha="left")
    save(fig, "testbed")


def main():
    global PREVIEW
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--preview", type=Path, help="also save PNG copies here")
    a = ap.parse_args()
    if a.preview:
        PREVIEW = a.preview
        PREVIEW.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    diagrams = (architecture, gated, run_timeline, callers_view, call_graph, testbed)
    for f in diagrams:
        f()
    print(f"{len(diagrams)} diagrams -> {OUT}")


if __name__ == "__main__":
    main()
