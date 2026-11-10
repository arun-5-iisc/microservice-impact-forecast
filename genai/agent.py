"""
The investigation agent, using Pydantic AI.

One investigation is one Pydantic AI agent run against one MCP server process (genai/mcp_server.py, bound to one replayed incident at one moment).
Pydantic AI validates it and sends invalid answers back to the model to fix.

Works with Claude models ("anthropic:claude-opus-5-5", "anthropic:claude-haiku-5-5") and OpenAI models through the Responses API ("openai:gpt-6-sol").
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time

from pathlib import Path
from typing import Literal

from fastmcp.client.transports import StdioTransport
from pydantic import BaseModel, Field
from pydantic_ai import Agent, UsageLimits
from pydantic_ai.mcp import MCPToolset
from pydantic_ai.models.anthropic import AnthropicModelSettings
from pydantic_ai.models.openai import OpenAIResponsesModelSettings
from pydantic_ai.settings import ModelSettings

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from genai.replay import SERVICES  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
SERVER = REPO_ROOT / "genai" / "mcp_server.py"
ServiceName = Literal[tuple(SERVICES)]  # type: ignore[valid-type]

# System prompts are versioned files in genai/prompts/ (one per development round; see its README.md).
PROMPT_DIR = REPO_ROOT / "genai" / "prompts"
FROZEN_PROMPT = "v1"    # used when no version is given: frozen after round 1 


def prompt_versions() -> list[str]:
    return sorted(f.stem.removeprefix("system_") for f in PROMPT_DIR.glob("system_v*.md"))


def system_prompt(version: str = FROZEN_PROMPT) -> str:
    path = PROMPT_DIR / f"system_{version}.md"
    if not path.exists():
        raise ValueError(f"no prompt version {version!r}; available: {', '.join(prompt_versions())}")
    return path.read_text().strip()


def prompt_fingerprint(text: str) -> str:
    """Short sha256 of a prompt's text, recorded with every result."""
    return hashlib.sha256(text.encode()).hexdigest()[:12]


# USD per million tokens: input, output, cache read, cache write (5 min), cache write (1 h).
# Claude API list prices.
PRICES = {
    "claude-opus-5-5": (4.00, 20.00, 0.20, 5.00, 8.00),
    "claude-haiku-5-5": (0.10, 0.50, 0.01, 0.125, 0.20),
    # OpenAI standard tier, developers.openai.com/api/docs/pricing. OpenAI caches prompts
    # automatically with no write premium, so cache writes are priced as normal input.
    "gpt-6-sol": (2.00, 10.00, 0.20, 2.00, 2.00),
}


def provider_of(model: str) -> str:
    """'anthropic' or 'openai' for a plain model name (claude-..., gpt-...) or a 'provider:model' string."""
    name = model.split(":", 1)[-1] if ":" in model else model
    if model.startswith("anthropic:") or name.startswith("claude-"):
        return "anthropic"
    if model.startswith(("openai:", "openai-responses:")) or name.startswith("gpt-"):
        return "openai"
    raise ValueError(f"unknown model provider for {model!r} (expected claude-... or gpt-...)")


def full_model_name(model: str) -> str:
    """Pydantic AI model string: 'anthropic:claude-...' or 'openai:gpt-...' (OpenAI Responses API)."""
    return model if ":" in model else f"{provider_of(model)}:{model}"


class InvestigationReport(BaseModel):
    """The agent's final answer."""
    predicted_final_severity: Literal["Low", "Medium", "High"] = Field(
        description="severity of the changed service once the change has fully taken effect (a few minutes after it)")
    probable_cause: Literal["cpu_limit", "memory_limit", "network_latency", "service_unavailable",
                            "no_clear_effect", "unclear"] = Field(description="what the change most likely did to the service")
    affected_services: list[ServiceName] = Field(description="services that are, or will become, measurably degraded; may be empty")
    recommended_action: Literal["no_action", "monitor", "rollback_change", "increase_resource_limits",
                                "scale_up_replicas"] = Field(description="what an operator should do now")
    confidence: float = Field(ge=0, le=1, description="confidence in the severity and cause, 0-1")
    evidence: str = Field(description="the key observations behind the conclusion, briefly")


def forecast_text(f: dict) -> str:
    p = f["severity_probabilities"]
    return (f"An automated risk model screened this change {f['snapshot_s']} s after it and flagged it for investigation. "
            f"Its forecast: final severity {f['predicted_final_severity']} (probabilities: High {p['High']:.2f}, "
            f"Medium {p['Medium']:.2f}, Low {p['Low']:.2f}); about {f['predicted_blast_radius']:.1f} other services "
            f"affected (low confidence). The model can be wrong; verify with the telemetry.")


def user_prompt(service: str, at: float, forecast: dict | None) -> str:
    prompt = f"A change was applied to {service} {int(at)} seconds ago. Investigate and give your report."
    return prompt + ("\n\n" + forecast_text(forecast) if forecast else "")


def incident_toolset(run_id: str, at: float, rag: bool = True, rag_mode: str = "hybrid",
                     log_out: Path | None = None) -> MCPToolset:
    """MCP tools for one replayed incident. The ML forecast is never a tool here (it goes in the prompt)."""
    args = [str(SERVER), "--run", run_id, "--at", str(at), "--rag-mode", rag_mode]
    args += [] if rag else ["--no-rag"]
    args += ["--log-out", str(log_out)] if log_out else []
    env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}
    # init_timeout: the default 5 s is too short, the server takes about 3 s to start on its own or longer
    # keep_alive=False: otherwise fastmcp keeps every finished server running
    toolset = MCPToolset(StdioTransport(command=sys.executable, args=args, env=env, keep_alive=False), id="incident",
                         init_timeout=60, read_timeout=120)
    return toolset


def model_settings(model: str, effort: str) -> ModelSettings:
    if model.startswith("anthropic:"):
        # automatic prompt caching: each step re-reads the conversation so far at the cache-read price
        return AnthropicModelSettings(max_tokens=16000, anthropic_effort=effort, anthropic_cache=True)
    if model.startswith(("openai:", "openai-responses:")):
        # same effort level as the Claude models; OpenAI caches repeated prompt prefixes automatically
        return OpenAIResponsesModelSettings(max_tokens=16000, openai_reasoning_effort=effort)
    return ModelSettings(max_tokens=16000)


def usage_cost(model: str, usage) -> float | None:
    """Cost in USD from token counts (Pydantic AI counts cached tokens inside input_tokens)."""
    if getattr(usage, "cost", None):
        return float(usage.cost)
    price = PRICES.get(model.split(":", 1)[-1])
    if not price:
        return None
    p_in, p_out, p_read, p_w5, p_w1h = price
    w1h = usage.cache_write_1h_tokens or 0
    w5 = (usage.cache_write_tokens or 0) - w1h
    fresh = max(usage.input_tokens - usage.cache_read_tokens - (usage.cache_write_tokens or 0), 0)
    return (fresh * p_in + usage.output_tokens * p_out + usage.cache_read_tokens * p_read
            + w5 * p_w5 + w1h * p_w1h) / 1e6


async def investigate(run_id: str, service: str, at: float, forecast: dict | None, model: str,
                      effort: str = "medium", rag: bool = True, rag_mode: str = "hybrid",
                      workdir: Path | None = None, timeout_s: float = 600, prompt_version: str = FROZEN_PROMPT) -> dict:
    """Run one investigation. Returns the report (or an error) plus usage, cost and timing."""
    log_out = None
    if workdir:
        workdir.mkdir(parents=True, exist_ok=True)
        log_out = workdir / "calls.jsonl"
        log_out.unlink(missing_ok=True)
    instructions = system_prompt(prompt_version)
    agent = Agent(model, output_type=InvestigationReport, instructions=instructions,
                  toolsets=[incident_toolset(run_id, at, rag, rag_mode, log_out)],
                  model_settings=model_settings(model, effort), retries=2)
    prompt = user_prompt(service, at, forecast)
    t0 = time.time()
    out = {"report": None, "error": None}
    try:
        result = await agent.run(prompt, timeout=timeout_s,
                                 usage_limits=UsageLimits(request_limit=30, tool_calls_limit=40))
        out["report"] = result.output.model_dump()
        usage = result.usage()
        if workdir:
            (workdir / "messages.json").write_bytes(result.all_messages_json())
    except Exception as e:      # a failed investigation is recorded
        out["error"] = f"{type(e).__name__}: {e}"[:500]
        usage = None
    calls = log_out.read_text().splitlines() if log_out and log_out.exists() else []
    if workdir:
        (workdir / "prompt.txt").write_text(instructions + "\n\n---\n\n" + prompt)
    return {**out, "prompt": prompt_version, "prompt_sha": prompt_fingerprint(instructions),
            "wall_s": round(time.time() - t0, 1), "tool_calls": len(calls),
            "tools_used": ",".join(json.loads(c)["tool"] for c in calls),
            "requests": getattr(usage, "requests", None),
            "input_tokens": getattr(usage, "input_tokens", None), "output_tokens": getattr(usage, "output_tokens", None),
            "cache_read_tokens": getattr(usage, "cache_read_tokens", None),
            "cache_write_tokens": getattr(usage, "cache_write_tokens", None),
            "reasoning_tokens": getattr(usage, "output_reasoning_tokens", None),
            "cost_usd": usage_cost(model, usage) if usage else None}
