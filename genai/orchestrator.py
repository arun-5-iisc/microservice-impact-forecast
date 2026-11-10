"""
Change-driven incident pipeline on replayed incidents, and its evaluation.

Pipeline (the "gated" system):
  change -> ML screen at 15/30/45/60 s -> first risky snapshot (forecast Medium or High)
         -> LLM investigation (genai/agent.py, tools from genai/mcp_server.py), forecast given up front
         -> structured report -> alert (recorded here; genai/live.py sends real alerts to Slack)
  Never risky by 60 s -> cleared without calling the LLM.

Systems compared on the same incidents:
  ml_only   ML forecast at 30 s; no cause; action and alert follow from the forecast
  llm_only  LLM investigates at 30 s with telemetry + RAG tools, no ML
  ml_llm    LLM investigates at 30 s with the ML forecast given up front
  gated     the pipeline above

The LLM runs through the standalone Pydantic AI agent (genai/agent.py). API keys come from ANTHROPIC_API_KEY or
~/.anthropic_key (Claude models) and OPENAI_API_KEY or ~/.openai_key (OpenAI models).

Usage:
  venv/bin/python -m genai.orchestrator --runs-file results/genai/dev_set.txt --systems llm_only \
      --out results/genai/dev/round0 --concurrency 4
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path

import pandas as pd

os.environ.setdefault("PYDANTIC_AI_NO_BANNER", "1")
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from genai import agent as pyd_agent  # noqa: E402
from genai.impact_model import SNAPSHOTS, ImpactModel  # noqa: E402
from genai.replay import RecordedRun  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
DECISION_T = 30                 # when the non-gated systems decide
SYSTEMS = ("ml_only", "llm_only", "ml_llm", "gated")
CAUSE_OF = {"cpu": "cpu_limit", "memory": "memory_limit", "latency": "network_latency", "replicas": "service_unavailable"}


def action_for(severity: str) -> str:
    return {"High": "rollback_change", "Medium": "monitor", "Low": "no_action"}[severity]


KEY_SOURCES = {"anthropic": ("ANTHROPIC_API_KEY", ".anthropic_key"), "openai": ("OPENAI_API_KEY", ".openai_key")}


def api_key(provider: str = "anthropic") -> str:
    """The provider's API key from its environment variable or ~/.<provider>_key (never printed or logged)."""
    env, fname = KEY_SOURCES[provider]
    key = os.environ.get(env)
    if not key:
        f = Path.home() / fname
        if not f.exists():
            sys.exit(f"needs {env} or ~/{fname}")
        key = f.read_text().strip()
    return key


# one system on one incident

async def run_system(system: str, run_id: str, ml: ImpactModel, cfg: argparse.Namespace) -> dict:
    base = {"system": system, "run": run_id}
    if system == "ml_only":
        f = ml.predict(run_id, DECISION_T)
        sev = f["predicted_final_severity"]
        return {**base, "decided_at_s": DECISION_T, "pred_severity": sev, "pred_cause": None,
                "pred_action": action_for(sev), "confidence": max(f["severity_probabilities"].values())}
    if system == "gated":
        esc = next(((T, f) for T in SNAPSHOTS if (f := ml.predict(run_id, T))["predicted_final_severity"] != "Low"), None)
        if esc is None:
            return {**base, "decided_at_s": SNAPSHOTS[-1], "escalated": False, "pred_severity": "Low",
                    "pred_cause": "no_clear_effect", "pred_action": "no_action"}
        at, forecast = esc
    else:
        at = DECISION_T
        forecast = ml.predict(run_id, DECISION_T) if system == "ml_llm" else None
    service = RecordedRun(run_id).changed_service
    workdir = cfg.out / system / run_id
    model = pyd_agent.full_model_name(cfg.model)
    res = await pyd_agent.investigate(run_id, service, at, forecast, model, cfg.effort, rag=not cfg.no_rag,
                                      rag_mode=cfg.rag_mode, workdir=workdir, prompt_version=cfg.prompt)
    if res["report"] and workdir.exists():
        (workdir / "report.json").write_text(json.dumps(res["report"], indent=2))
    r = res["report"] or {}
    return {**base, "decided_at_s": at, "escalated": True if system == "gated" else None,
            "pred_severity": r.get("predicted_final_severity"), "pred_cause": r.get("probable_cause"),
            "pred_action": r.get("recommended_action"), "confidence": r.get("confidence"),
            "affected": ",".join(r.get("affected_services", [])), "evidence": r.get("evidence"),
            **{k: v for k, v in res.items() if k != "report"}}


# Errors meaning the API account can't pay for more calls (out of credit, quota or spending limit). The run then
# stops instead of recording every remaining incident as failed; re-running the same command continues it.
BILLING_ERRORS = ("credit balance", "insufficient_quota", "exceeded your current quota", "billing",
                  "api usage limits", "spend limit", "usage limit reached")


def is_billing_error(error) -> bool:
    return isinstance(error, str) and any(x in error.lower() for x in BILLING_ERRORS)


def saved_prompt_version(workdir: Path) -> tuple[str, str | None]:
    """Which prompt version an earlier investigation used, from its saved prompt.txt ("unknown" if none matches).
    No prompt.txt means no LLM ran (e.g. a gated incident the ML model did not escalate): (None, None)."""
    f = workdir / "prompt.txt"
    if not f.exists():
        return None, None
    text = f.read_text().split("\n\n---\n\n")[0]
    for v in pyd_agent.prompt_versions():
        if text == pyd_agent.system_prompt(v):
            return v, pyd_agent.prompt_fingerprint(text)
    return "unknown", pyd_agent.prompt_fingerprint(text)


# scoring

def score(row: dict, truth: dict) -> dict:
    sev, cause = truth["severity"], CAUSE_OF[truth["fault_type"]]
    ps, pc, pa = row.get("pred_severity"), row.get("pred_cause"), row.get("pred_action")
    if sev == "High":
        good_actions = {"rollback_change", {"cpu_limit": "increase_resource_limits", "memory_limit": "increase_resource_limits",
                                            "service_unavailable": "scale_up_replicas"}.get(cause, "rollback_change")}
    elif sev == "Medium":
        good_actions = {"rollback_change", "monitor", "increase_resource_limits", "scale_up_replicas"}
    else:
        good_actions = {"no_action", "monitor"}
    alert, should = ps in ("Medium", "High"), sev in ("Medium", "High")
    return {"true_severity": sev, "true_cause": cause, "true_change": f"{truth['fault_type']} {truth['fault_value']}",
            "severity_correct": ps == sev,
            # cause is judged only where the change visibly did something (Medium/High): a Low run shows no cause
            "cause_correct": None if (pc is None or sev == "Low") else pc == cause,
            "action_ok": None if pa is None else pa in good_actions,
            "alerted": alert, "alert_correct": alert == should,
            "false_alert": alert and not should, "missed_alert": should and not alert}


def summarise(df: pd.DataFrame) -> pd.DataFrame:
    agg = {"n": ("run", "size"), "severity_acc": ("severity_correct", "mean"), "cause_acc": ("cause_correct", "mean"),
           "action_ok": ("action_ok", "mean"), "false_alerts": ("false_alert", "sum"), "missed_alerts": ("missed_alert", "sum")}
    if "cost_usd" in df:
        agg.update(errors=("error", lambda s: int(s.notna().sum())), mean_s=("wall_s", "mean"),
                   mean_tools=("tool_calls", "mean"), cost_usd=("cost_usd", "sum"))
    return df.groupby("system").agg(**agg)


async def main_async(cfg: argparse.Namespace):
    cfg.out = cfg.out.resolve()     # every path given to tools and subprocesses must be absolute
    cfg.out.mkdir(parents=True, exist_ok=True)
    provider = pyd_agent.provider_of(cfg.model)
    if any(s != "ml_only" for s in cfg.systems):
        os.environ[KEY_SOURCES[provider][0]] = api_key(provider)
    runs = cfg.runs or Path(cfg.runs_file).read_text().split()
    ml = ImpactModel()
    results_path = cfg.out / "results.csv"
    rows = pd.read_csv(results_path).to_dict("records") if results_path.exists() else []
    for r in rows:
        if r["system"] != "ml_only" and not isinstance(r.get("prompt"), str):
            r["prompt"], r["prompt_sha"] = saved_prompt_version(cfg.out / r["system"] / r["run"])
    other = sorted({r["prompt"] for r in rows if r["system"] != "ml_only" and isinstance(r.get("prompt"), str)
                    and r["prompt"] != cfg.prompt})
    if other:   # one results folder = one prompt version, so rounds are never mixed
        sys.exit(f"{results_path} holds answers from prompt version(s) {other}, not {cfg.prompt}; "
                 f"use a new --out folder (or --prompt {other[0]} to resume that run)")
    if cfg.retry_errors:
        rows = [r for r in rows if not isinstance(r.get("error"), str)]   # failed investigations run again
    done = {(r["system"], r["run"]) for r in rows}
    truths = {r: RecordedRun(r).ground_truth() for r in runs}
    sem, lock = asyncio.Semaphore(cfg.concurrency), asyncio.Lock()

    stop = asyncio.Event()      # set when the account is out of credit: no new investigations start

    async def one(system: str, run_id: str):
        if stop.is_set():
            return
        async with sem:
            if stop.is_set():
                return
            t0 = time.time()
            row = await run_system(system, run_id, ml, cfg)
        if is_billing_error(row.get("error")):
            if not stop.is_set():
                print(f"\nSTOPPING at {system} {run_id}: the API account is out of credit or over its limit "
                      f"({str(row['error'])[:160]})", flush=True)
            stop.set()
            return      # not saved, so re-running the same command repeats this investigation
        row.update(score(row, truths[run_id]), model=None if system == "ml_only" else cfg.model,
                   effort=cfg.effort, runner=None if system == "ml_only" else "pydantic", rag=not cfg.no_rag)
        async with lock:
            rows.append(row)
            pd.DataFrame(rows).to_csv(results_path, index=False)      # saved after every investigation
            print(f"{system:9s} {run_id}: truth {row['true_severity']}/{row['true_cause']} -> "
                  f"{row['pred_severity']}/{row['pred_cause']}/{row['pred_action']}  ({time.time() - t0:.0f}s, "
                  f"{row.get('tool_calls', '-')} tools, ${row.get('cost_usd') or 0:.3f})"
                  f"{'  ERROR: ' + str(row['error'])[:120] if row.get('error') else ''}", flush=True)

    await asyncio.gather(*(one(s, r) for r in runs for s in cfg.systems if (s, r) not in done))
    if stop.is_set():
        left = sum(1 for r in runs for s in cfg.systems if (s, r) not in {(x["system"], x["run"]) for x in rows})
        print(f"\nPaused: {len(rows)} investigations are saved in {results_path}; {left} still to do. "
              "Add credit, then re-run the same command to continue.", flush=True)
        sys.exit(3)
    # re-score every saved answer with the current rules (answers are kept; only the scoring is recomputed)
    for r in rows:
        truth = truths.get(r["run"]) or RecordedRun(r["run"]).ground_truth()
        r.update(score({k: (None if not isinstance(v, (list, dict)) and pd.isna(v) else v) for k, v in r.items()}, truth))
    df = pd.DataFrame(rows)
    df.to_csv(results_path, index=False)
    print("\n" + summarise(df).to_string(float_format=lambda x: f"{x:.2f}"))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--runs", nargs="+")
    src.add_argument("--runs-file", help="one experiment id per line, e.g. results/genai/dev_set.txt")
    p.add_argument("--systems", nargs="+", choices=SYSTEMS, default=list(SYSTEMS))
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--model", default="claude-opus-5-5")
    p.add_argument("--effort", default="medium", choices=["low", "medium", "high", "xhigh", "max"])
    p.add_argument("--rag-mode", default="hybrid", choices=["keyword", "vector", "hybrid"])
    p.add_argument("--no-rag", action="store_true")
    p.add_argument("--prompt", default=pyd_agent.FROZEN_PROMPT, choices=pyd_agent.prompt_versions(),
                   help="system prompt version: genai/prompts/system_<version>.md")
    p.add_argument("--concurrency", type=int, default=1, help="parallel investigations")
    p.add_argument("--retry-errors", action="store_true", help="re-run investigations that failed earlier")
    asyncio.run(main_async(p.parse_args()))


if __name__ == "__main__":
    main()
