"""
Live mode: investigate changes on the running cluster with the frozen evaluation system, and notify Slack.

Modes
  watch        Watch the namespace for changes: a Deployment's CPU or memory limit, its replica count, or a new
               Chaos Mesh NetworkChaos (how the test latency faults are applied).
               - The ML model screens each change 15, 30, 45 and 60 s after it (the evaluation's "gated" policy). At
                 the first forecast of Medium or High, the LLM agent investigates, and the result goes to Slack.
               - --always: investigate every change at 30 s, without the ML gate.
               - A change that restores the state seen when watching started is reported as a restore, not
                 investigated.
  investigate  One change, given by hand: --service, --change-time (an ISO time, or e.g. "90s ago") and --at
               (default 30 s). Give the change itself (--fault cpu|memory|latency|replicas --value ...) to include
               the ML forecast.
  inject       Apply a test change (as the experiments did) and undo it after --hold seconds, without recording an
               experiment. Run "watch" in another terminal to see the system react.
  simulate     Replay a recorded run through the whole live path (snapshot -> features -> ML -> LLM -> Slack), to test
               it without a cluster. "--model test" uses Pydantic AI's TestModel instead of an LLM, which is free.

What is saved
  results/genai/live/<incident id>/       the incident record, ML features and forecast, the agent's prompt,
                                          transcript, tool calls and report, and the Slack payload
  results/genai/live/snapshots/<id>/      the telemetry snapshot, in the recorded-run format, so the evaluation
                                          tools work on it unchanged

The agent configuration is the frozen evaluation one (prompt v1, the same tools and effort).

Needs: kubectl access to the cluster (watch, investigate, inject), the API key as for the evaluation, and a Slack
webhook for real messages (see genai/notify.py). --dry-run prints the messages instead.

    python -m genai.live simulate --run EXP_0283 --at 30 --model test --dry-run
    python -m genai.live watch --dry-run
    python -m genai.live inject --service productcatalogservice --fault latency --value 200 --hold 180
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from genai import agent as A                       # noqa: E402
from genai import notify                           # noqa: E402
from genai.impact_model import SNAPSHOTS, ImpactModel  # noqa: E402
from genai.replay import RAW_DIR, SERVICES         # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))
LIVE_DIR = REPO_ROOT / "results" / "genai" / "live"
SNAP_DIR = LIVE_DIR / "snapshots"
LIVE_SERVER = REPO_ROOT / "genai" / "live_server.py"
NAMESPACE = "online-boutique"
BASELINE_S = 60          # as in the experiments: the 60 s before the change are the baseline
SETTLE_S = 10            # wait this long after a snapshot time so traces and metrics have been exported
POLL_S = 2
TEST_TOOLS = ["get_change_notice", "get_metrics", "get_latency_overview", "get_logs"]   # what TestModel calls


def iso(ts: float) -> str:
    return datetime.fromtimestamp(ts, timezone.utc).isoformat()


def ts_text(ts: float) -> str:
    return datetime.fromtimestamp(ts, timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


# Snapshots: telemetry from 60 s before the change up to "now", in the recorded-run format

def write_metadata(exp_dir: Path, incident_id: str, service: str, fault_type: str | None, fault_value, traffic: str,
                   base_start: float, change_ts: float, now_ts: float, **extra) -> None:
    meta = {"experiment_id": incident_id, "service": service, "fault_type": fault_type or "unknown",
            "fault_value": "" if fault_value is None else str(fault_value), "traffic_level": traffic,
            "start_time": iso(base_start), "end_time": iso(now_ts), "live": True,
            "phases": {"baseline": {"start": base_start, "end": change_ts},
                       "fault": {"start": change_ts, "end": now_ts},
                       "recovery": {"start": now_ts, "end": now_ts}}, **extra}
    (exp_dir / "metadata.json").write_text(json.dumps(meta, indent=2))


def snapshot_live(incident_id: str, service: str, fault_type, fault_value, traffic: str,
                  change_ts: float, now_ts: float) -> Path:
    """Collect Prometheus, Jaeger and Kubernetes data for [change - 60 s, now] with the experiments' collectors."""
    from jaeger_collector import JaegerCollector
    from kubernetes_collector import KubernetesCollector
    from prometheus_collector import PrometheusCollector
    exp_dir = SNAP_DIR / incident_id
    exp_dir.mkdir(parents=True, exist_ok=True)
    start = datetime.fromtimestamp(change_ts - BASELINE_S, timezone.utc)
    end = datetime.fromtimestamp(now_ts, timezone.utc)
    prom = PrometheusCollector()
    prom.save(prom.collect(start, end, step=15), exp_dir / "prometheus.csv")
    jaeger = JaegerCollector()
    jaeger.save(jaeger.collect(start, end), exp_dir / "traces.json")
    k8s = KubernetesCollector()
    k8s.save(k8s.collect(NAMESPACE, service, start, end), exp_dir / "events.json")
    write_metadata(exp_dir, incident_id, service, fault_type, fault_value, traffic, change_ts - BASELINE_S,
                   change_ts, now_ts, source="live")
    return exp_dir


def snapshot_from_recording(run_id: str, at: float, incident_id: str) -> tuple[Path, dict]:
    """A recorded run cut off `at` seconds after its change: what a live snapshot at that moment would hold."""
    src = RAW_DIR / run_id
    meta = json.loads((src / "metadata.json").read_text())
    ph = meta["phases"]
    change_ts, now_ts = ph["fault"]["start"], ph["fault"]["start"] + at
    exp_dir = SNAP_DIR / incident_id
    exp_dir.mkdir(parents=True, exist_ok=True)
    prom = pd.read_csv(src / "prometheus.csv")
    t = pd.to_datetime(prom["timestamp"], utc=True).map(lambda x: x.timestamp())
    prom[t <= now_ts].to_csv(exp_dir / "prometheus.csv", index=False)
    with open(src / "traces.json") as f:
        traces = json.load(f)
    cut = {}
    for svc, tl in traces.items():
        kept = []
        for tr in tl:
            spans = [s for s in tr.get("spans", []) if s["startTime"] / 1e6 <= now_ts]
            if spans:
                kept.append({**tr, "spans": spans})
        if kept:
            cut[svc] = kept
    del traces
    (exp_dir / "traces.json").write_text(json.dumps(cut))
    (exp_dir / "events.json").write_text((src / "events.json").read_text())   # the views cut events at "now"
    write_metadata(exp_dir, incident_id, meta["service"], meta["fault_type"], meta["fault_value"],
                   meta.get("traffic_level", "normal"), ph["baseline"]["start"], change_ts, now_ts,
                   source=f"simulated from {run_id}")
    return exp_dir, meta


# ML forecast on a snapshot: the pipeline's own feature code, then the frozen models

class LiveImpactModel(ImpactModel):
    def forecast(self, exp_dir: Path, at: float, save_to: Path | None = None) -> dict:
        import feature_engineer as fe_mod
        import preprocess
        T = max((t for t in SNAPSHOTS if t <= at), default=None)
        if T is None:
            return {"available": False, "note": f"first forecast is available {SNAPSHOTS[0]} s after the change"}
        meta = json.loads((exp_dir / "metadata.json").read_text())
        if meta.get("fault_type") in (None, "", "unknown"):
            return {"available": False, "note": "the change itself is unknown, so the ML model cannot be used"}
        rows, _ = preprocess.process_experiment(exp_dir)
        base = pd.DataFrame(rows)
        base = base[base.time_into_fault_s <= T].copy()
        fe = fe_mod.add_log_transforms(fe_mod.add_fault_severity_ratio(fe_mod.add_topology_features(base)))
        agg = fe_mod._system_aggregates_for_run(exp_dir, meta["service"], sorted(base.time_into_fault_s.unique()))
        fe["system_error_rate"] = fe.time_into_fault_s.map(lambda t: agg[t]["system_error_rate"])
        fe["system_latency_ratio"] = fe.time_into_fault_s.map(lambda t: agg[t]["system_latency_ratio"])
        if save_to:
            fe.to_csv(save_to / "features.csv", index=False)
        b, f = base[base.time_into_fault_s == T], fe[fe.time_into_fault_s == T]
        probs = self.clf.predict_proba(b[self.clf_features])[0]
        from train_model import SEVERITY_ORDER
        return {"available": True, "snapshot_s": T,
                "predicted_final_severity": SEVERITY_ORDER[int(probs.argmax())],
                "severity_probabilities": {c: round(float(p), 3) for c, p in zip(SEVERITY_ORDER, probs)},
                "predicted_blast_radius": round(max(0.0, float(self.reg.predict(f[self.reg_features])[0])), 2)}


# LLM investigation on a snapshot (the frozen agent, with the live tool server)

async def investigate_snapshot(incident_id: str, service: str, at: float, forecast: dict | None, model: str,
                               effort: str, workdir: Path, rag: bool = True, timeout_s: float = 600) -> dict:
    from fastmcp.client.transports import StdioTransport
    from pydantic_ai import Agent, UsageLimits
    from pydantic_ai.mcp import MCPToolset
    workdir.mkdir(parents=True, exist_ok=True)
    log_out = workdir / "calls.jsonl"
    log_out.unlink(missing_ok=True)
    args = [str(LIVE_SERVER), "--raw-dir", str(SNAP_DIR), "--run", incident_id, "--at", str(at),
            "--log-out", str(log_out)] + ([] if rag else ["--no-rag"])
    toolset = MCPToolset(StdioTransport(command=sys.executable, args=args, keep_alive=False,
                                        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}),
                         id="incident", init_timeout=60, read_timeout=120)
    if model == "test":
        from pydantic_ai.models.test import TestModel
        llm, settings = TestModel(call_tools=TEST_TOOLS), None
    else:
        from genai.orchestrator import KEY_SOURCES, api_key
        provider = A.provider_of(model)
        os.environ[KEY_SOURCES[provider][0]] = api_key(provider)
        llm = A.full_model_name(model)
        settings = A.model_settings(llm, effort)
    instructions = A.system_prompt()        # the frozen prompt
    prompt = A.user_prompt(service, at, forecast if forecast and forecast.get("available") else None)
    (workdir / "prompt.txt").write_text(instructions + "\n\n---\n\n" + prompt)
    agent = Agent(llm, output_type=A.InvestigationReport, instructions=instructions, toolsets=[toolset],
                  model_settings=settings, retries=2)
    t0 = time.time()
    result = await agent.run(prompt, usage_limits=UsageLimits(request_limit=30, tool_calls_limit=40))
    (workdir / "messages.json").write_bytes(result.all_messages_json())
    report = result.output.model_dump()
    (workdir / "report.json").write_text(json.dumps(report, indent=2))
    return {"report": report, "wall_s": round(time.time() - t0, 1),
            "cost_usd": None if model == "test" else A.usage_cost(llm, result.usage()),
            "tool_calls": len(log_out.read_text().splitlines()) if log_out.exists() else 0}


def finish_incident(incident_id: str, service: str, change: str | None, change_ts: float, at: float,
                    forecast: dict | None, res: dict, model: str, mode: str, dry_run: bool, **extra) -> dict:
    workdir = LIVE_DIR / incident_id
    incident = {"incident_id": incident_id, "mode": mode, "service": service, "change": change,
                "change_time": change_ts, "change_time_text": ts_text(change_ts), "decided_at_s": at,
                "forecast": forecast, "report": res["report"], "model": model, "cost_usd": res.get("cost_usd"),
                "tool_calls": res.get("tool_calls"), "wall_s": res.get("wall_s"),
                "investigation_dir": str(workdir.relative_to(REPO_ROOT)), **extra}
    (workdir / "live_incident.json").write_text(json.dumps(incident, indent=2))
    payload = notify.build_message(incident)
    (workdir / "slack.json").write_text(json.dumps(payload, indent=2))
    sent = notify.send(payload, dry_run=dry_run)
    r = res["report"]
    print(f"[{incident_id}] {service}: {r['predicted_final_severity']} / {r['probable_cause']} → "
          f"{r['recommended_action']}  ({res.get('wall_s')} s, {res.get('tool_calls')} tool calls, "
          f"${res.get('cost_usd') or 0:.3f}){'  — sent to Slack' if sent else ''}", flush=True)
    return incident


# Watching the cluster for changes

def kubectl_json(*args: str) -> dict | None:
    try:
        p = subprocess.run(["kubectl", *args, "-o", "json"], capture_output=True, text=True, timeout=20)
    except (OSError, subprocess.TimeoutExpired):
        return None
    return json.loads(p.stdout) if p.returncode == 0 and p.stdout.strip() else None


def cluster_state() -> tuple[dict, dict] | None:
    """Per service: replicas and CPU/memory limits; and the NetworkChaos objects in the namespace."""
    d = kubectl_json("get", "deploy", "-n", NAMESPACE)
    if d is None:
        return None
    deps = {}
    for item in d.get("items", []):
        name = item["metadata"]["name"]
        if name not in SERVICES:
            continue
        cs = item["spec"]["template"]["spec"]["containers"]
        c = next((x for x in cs if x.get("name") == "server"), cs[0])
        lim = c.get("resources", {}).get("limits", {})
        deps[name] = {"replicas": item["spec"].get("replicas"), "cpu": lim.get("cpu"), "memory": lim.get("memory")}
    chaos = {}
    for item in (kubectl_json("get", "networkchaos", "-n", NAMESPACE) or {}).get("items", []):
        sel = item.get("spec", {}).get("selector", {}).get("labelSelectors", {})
        lat = str(item.get("spec", {}).get("delay", {}).get("latency", ""))
        created = item["metadata"].get("creationTimestamp")
        chaos[item["metadata"]["name"]] = {
            "service": sel.get("app"), "latency_ms": "".join(ch for ch in lat if ch.isdigit()) or lat,
            "created": datetime.fromisoformat(created.replace("Z", "+00:00")).timestamp() if created else time.time()}
    return deps, chaos


def traffic_level() -> str:
    d = kubectl_json("get", "deploy", "loadgenerator", "-n", NAMESPACE)
    if not d:
        return "normal"
    env = {e["name"]: e.get("value") for e in d["spec"]["template"]["spec"]["containers"][0].get("env", [])}
    return "high" if env.get("USERS") == "25" else "normal"


def detect_changes(baseline: tuple, prev: tuple, cur: tuple) -> list[dict]:
    (bdeps, _), (pdeps, pchaos), (cdeps, cchaos) = baseline, prev, cur
    now, out = time.time(), []
    for svc, c in cdeps.items():
        p, b = pdeps.get(svc, c), bdeps.get(svc, c)
        for key, fault in (("replicas", "replicas"), ("cpu", "cpu"), ("memory", "memory")):
            if c[key] != p[key]:
                restored = c[key] == b[key]
                out.append({"service": svc, "fault": fault, "value": str(c[key]), "previous": str(p[key]),
                            "ts": now - POLL_S / 2, "restored": restored})
    for name, ch in cchaos.items():
        if name not in pchaos and ch["service"] in SERVICES:
            out.append({"service": ch["service"], "fault": "latency", "value": ch["latency_ms"], "previous": "none",
                        "ts": ch["created"], "restored": False, "chaos": name})
    for name, ch in pchaos.items():
        if name not in cchaos:
            out.append({"service": ch["service"], "fault": "latency", "value": "removed", "previous": ch["latency_ms"],
                        "ts": now, "restored": True, "chaos": name})
    return out


def change_text(ch: dict) -> str:
    what = {"cpu": "CPU limit", "memory": "memory limit", "replicas": "replicas", "latency": "network delay (ms)"}
    return f"{what[ch['fault']]}: {ch['previous']} → {ch['value']}"


async def handle_change(ch: dict, model: str, effort: str, ml: LiveImpactModel, always: bool, notify_low: bool,
                        dry_run: bool) -> None:
    svc, cts = ch["service"], ch["ts"]
    stamp = datetime.fromtimestamp(cts, timezone.utc).strftime("%Y%m%dT%H%M%S")
    traffic = traffic_level()
    print(f"change detected: {svc} {change_text(ch)} at {ts_text(cts)}", flush=True)
    last = None
    for T in ((30,) if always else SNAPSHOTS):
        await asyncio.sleep(max(0.0, cts + T + SETTLE_S - time.time()))
        iid = f"LIVE_{svc}_{stamp}_T{T}"
        exp_dir = await asyncio.to_thread(snapshot_live, iid, svc, ch["fault"], ch["value"], traffic, cts, cts + T)
        (LIVE_DIR / iid).mkdir(parents=True, exist_ok=True)
        forecast = await asyncio.to_thread(ml.forecast, exp_dir, T, LIVE_DIR / iid)
        last = forecast
        sev = forecast.get("predicted_final_severity")
        print(f"  ML screen at {T} s: {sev} {forecast.get('severity_probabilities', '')}", flush=True)
        if always or sev in ("Medium", "High"):
            res = await investigate_snapshot(iid, svc, T, forecast, model, effort, LIVE_DIR / iid)
            finish_incident(iid, svc, change_text(ch), cts, T, forecast, res, model, "watch", dry_run,
                            escalated_by="always" if always else "ml_screen",
                            change_detail={k: ch[k] for k in ("fault", "value", "previous", "chaos") if k in ch})
            return
    print(f"  {svc}: ML screen stayed Low up to 60 s; no investigation", flush=True)
    if notify_low and last:
        notify.send({"text": f":large_green_circle: Change to {svc} ({change_text(ch)}): the ML screen forecast "
                             f"no significant impact (Low) up to 60 s after it; not investigated further."}, dry_run)


async def handle_change_reporting_errors(ch: dict, *args) -> None:
    """handle_change in a background task: a failure (telemetry collection, API credit, Slack) must not pass silently,
    so it is printed and, if possible, posted to Slack."""
    try:
        await handle_change(ch, *args)
    except Exception as e:  # noqa: BLE001 - report anything, keep watching
        print(f"  {ch['service']}: FAILED ({type(e).__name__}: {str(e)[:300]})", flush=True)
        traceback.print_exc()
        try:
            notify.send({"text": f":warning: Change to {ch['service']} ({change_text(ch)}) at {ts_text(ch['ts'])}: "
                                 f"the automatic investigation failed ({type(e).__name__}). Check it by hand."},
                        dry_run=args[-1])
        except Exception as e2:  # noqa: BLE001
            print(f"  could not post the failure to Slack either: {type(e2).__name__}", flush=True)


async def watch(model: str, effort: str, always: bool, notify_low: bool, dry_run: bool) -> None:
    state = cluster_state()
    if state is None:
        sys.exit("cannot read the cluster with kubectl (is the cluster up and the context set?)")
    baseline, prev = state, state
    ml = LiveImpactModel()
    print(f"watching namespace {NAMESPACE}: {len(state[0])} services, {len(state[1])} NetworkChaos objects; "
          f"model {model}; {'always investigate' if always else 'ML gate at 15/30/45/60 s'}"
          f"{'; dry run' if dry_run else ''}. Ctrl-C to stop.", flush=True)
    tasks = set()
    while True:
        await asyncio.sleep(POLL_S)
        cur = await asyncio.to_thread(cluster_state)
        if cur is None:
            continue
        for ch in detect_changes(baseline, prev, cur):
            if ch["restored"]:
                print(f"restored: {ch['service']} {change_text(ch)}", flush=True)
                continue
            t = asyncio.create_task(handle_change_reporting_errors(ch, model, effort, ml, always, notify_low, dry_run))
            tasks.add(t)
            t.add_done_callback(tasks.discard)
        prev = cur


# Commands

def parse_time(text: str) -> float:
    m = re.fullmatch(r"\s*(\d+(?:\.\d+)?)\s*s\s+ago\s*", text)
    if m:
        return time.time() - float(m.group(1))
    return datetime.fromisoformat(text.replace("Z", "+00:00")).timestamp()


def cmd_simulate(a) -> None:
    iid = f"SIM_{a.run}_T{int(a.at)}"
    exp_dir, meta = snapshot_from_recording(a.run, a.at, iid)
    (LIVE_DIR / iid).mkdir(parents=True, exist_ok=True)
    forecast = LiveImpactModel().forecast(exp_dir, a.at, LIVE_DIR / iid)
    print(f"[{iid}] ML forecast at {forecast.get('snapshot_s')} s: {forecast.get('predicted_final_severity')} "
          f"{forecast.get('severity_probabilities', '')}", flush=True)
    res = asyncio.run(investigate_snapshot(iid, meta["service"], a.at, forecast, a.model, a.effort, LIVE_DIR / iid))
    change = f"{meta['fault_type']} {meta['fault_value']} (recorded run {a.run})"
    finish_incident(iid, meta["service"], change, meta["phases"]["fault"]["start"], a.at, forecast, res, a.model,
                    "simulate", a.dry_run, simulated_from=a.run,
                    change_detail={"fault": meta["fault_type"], "value": str(meta["fault_value"])})


def cmd_investigate(a) -> None:
    cts = parse_time(a.change_time)
    now = cts + a.at
    if now > time.time():
        sys.exit(f"--at {a.at} s is in the future for that change time; wait {now - time.time():.0f} s more")
    stamp = datetime.fromtimestamp(cts, timezone.utc).strftime("%Y%m%dT%H%M%S")
    iid = f"LIVE_{a.service}_{stamp}_T{int(a.at)}"
    exp_dir = snapshot_live(iid, a.service, a.fault, a.value, traffic_level(), cts, now)
    (LIVE_DIR / iid).mkdir(parents=True, exist_ok=True)
    forecast = LiveImpactModel().forecast(exp_dir, a.at, LIVE_DIR / iid) if a.fault else None
    res = asyncio.run(investigate_snapshot(iid, a.service, a.at, forecast, a.model, a.effort, LIVE_DIR / iid))
    change = f"{a.fault} {a.value}" if a.fault else None
    finish_incident(iid, a.service, change, cts, a.at, forecast, res, a.model, "manual", a.dry_run,
                    change_detail={"fault": a.fault, "value": a.value} if a.fault else None)


def cmd_inject(a) -> None:
    from deployment_manager import apply_fault, restore_deployment
    print(f"applying test change: {a.service} {a.fault}={a.value} at {ts_text(time.time())}", flush=True)
    state = apply_fault(a.service, a.fault, a.value)
    try:
        time.sleep(a.hold)
    finally:
        restore_deployment(a.service, state)
        print(f"restored {a.service} at {ts_text(time.time())}", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    common = argparse.ArgumentParser(add_help=False)
    # gpt-6-sol: the most accurate agent in the gated holdout evaluation, at half Opus's cost
    common.add_argument("--model", default="gpt-6-sol", help="claude-..., gpt-..., or 'test' (no LLM)")
    common.add_argument("--effort", default="medium")
    common.add_argument("--dry-run", action="store_true", help="print Slack messages instead of sending them")
    s = sub.add_parser("watch", parents=[common], help="watch the cluster and investigate changes")
    s.add_argument("--always", action="store_true", help="investigate every change at 30 s (no ML gate)")
    s.add_argument("--notify-low", action="store_true", help="also post a short note for changes screened Low")
    s = sub.add_parser("investigate", parents=[common], help="investigate one change given by hand")
    s.add_argument("--service", required=True, choices=SERVICES)
    s.add_argument("--change-time", required=True, help="ISO time, or e.g. '90s ago'")
    s.add_argument("--at", type=float, default=30)
    s.add_argument("--fault", choices=["cpu", "memory", "latency", "replicas"])
    s.add_argument("--value")
    s = sub.add_parser("inject", help="apply a test change and undo it after --hold seconds")
    s.add_argument("--service", required=True, choices=SERVICES)
    s.add_argument("--fault", required=True, choices=["cpu", "memory", "latency", "replicas"])
    s.add_argument("--value", required=True, help="e.g. 150m, 128Mi, 200 (ms), 0")
    s.add_argument("--hold", type=float, default=180)
    s = sub.add_parser("simulate", parents=[common], help="replay a recorded run through the live path")
    s.add_argument("--run", required=True, help="recorded run id, e.g. EXP_0283")
    s.add_argument("--at", type=float, default=30)
    a = p.parse_args()
    if a.cmd == "watch":
        try:
            asyncio.run(watch(a.model, a.effort, a.always, a.notify_low, a.dry_run))
        except KeyboardInterrupt:
            print("stopped watching")
    else:
        {"investigate": cmd_investigate, "inject": cmd_inject, "simulate": cmd_simulate}[a.cmd](a)


if __name__ == "__main__":
    main()
