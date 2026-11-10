"""
MCP server that lets an LLM agent investigate one replayed incident.

One server process per investigation: it is bound to one recorded run and one moment
("--at" seconds after the change), so the agent can only see data up to that moment and
cannot choose or look up the run. The injected fault is never exposed (see genai/replay.py).

Tools (read-only):
  get_change_notice, get_service_map, get_metrics, get_events, get_latency, get_latency_overview
  search_past_incidents, get_runbook      retrieval / RAG (off with --no-rag)

The agent's final answer is Pydantic AI's typed output (genai/agent.py), and the ML forecast is given in
the prompt, so neither is a tool.

Run (stdio transport; genai/agent.py starts it like this for every investigation):
  venv/bin/python genai/mcp_server.py --run EXP_0307 --at 30 --log-out calls.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Annotated, Literal

from pydantic import Field

if __package__ in (None, ""):      # allow `python genai/mcp_server.py` as well as `python -m genai.mcp_server`
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mcp.server.mcpserver import MCPServer  # noqa: E402

from genai.knowledge import SEARCH_MODES, RagIndex, get_runbook as _get_runbook, list_runbooks, setting_of  # noqa: E402
from genai.replay import SERVICES, RecordedRun  # noqa: E402

Service = Literal[tuple(SERVICES)]    # type: ignore[valid-type]

INSTRUCTIONS = (
    "Tools for investigating a change to one microservice of the Online Boutique shop on Kubernetes. "
    "All data is as of the current moment; nothing after it is available. Times are seconds relative to "
    "the change (negative = before it)."
)


def build_server(run_id: str, at: float, rag: bool = True, rag_mode: str = "hybrid",
                 log_out: Path | None = None) -> MCPServer:
    run = RecordedRun(run_id)
    view = run.at(at)
    server = MCPServer("incident-replay", instructions=INSTRUCTIONS)
    started = time.time()

    def log(tool: str, args: dict):
        if log_out:
            with open(log_out, "a") as f:
                f.write(json.dumps({"t": round(time.time() - started, 2), "tool": tool, "args": args}) + "\n")

    @server.tool()
    def get_change_notice() -> dict:
        """Which service was changed, how long ago, and the current traffic level. Start here."""
        log("get_change_notice", {})
        return view.change_notice()

    @server.tool()
    def get_service_map() -> dict:
        """Which services call which (from distributed traces), and which services export no traces of their own."""
        log("get_service_map", {})
        return view.service_map()

    @server.tool()
    def get_metrics(service: Service,
                    last_seconds: Annotated[float | None, Field(description="only the last N seconds; omit for everything since the baseline began", gt=0)] = None) -> dict:
        """Prometheus samples (every 15 s) for one service: CPU (millicores), memory (MiB), replicas and container restarts."""
        log("get_metrics", {"service": service, "last_seconds": last_seconds})
        return {"service": service, "samples": view.metrics(service, last_seconds)}

    @server.tool()
    def get_events(service: Annotated[Service | None, Field(description="only events whose object name starts with this service; omit for all")] = None) -> dict:
        """Kubernetes events in the namespace since the baseline began (pod kills, restarts, probe failures, scaling).
        A null count / t_last means the event is still recurring and its later repeats are not known yet."""
        log("get_events", {"service": service})
        return {"events": view.events(service)}

    @server.tool()
    def get_latency(service: Service,
                    window_seconds: Annotated[float, Field(description="length of the recent window to compare with the baseline", ge=10, le=180)] = 30) -> dict:
        """Request latency (p50/p95 ms) and error rate of one service: the 60 s baseline before the change vs the
        most recent window. The changed service is measured as its callers saw it (time they waited, including the
        network); other services from their own request spans. """
        log("get_latency", {"service": service, "window_seconds": window_seconds})
        return view.latency(service, window_seconds)

    @server.tool()
    def get_latency_overview(window_seconds: Annotated[float, Field(ge=10, le=180)] = 30) -> dict:
        """get_latency for every service at once: a quick way to see which services are slower or failing."""
        log("get_latency_overview", {"window_seconds": window_seconds})
        return {"services": view.latency_overview(window_seconds)}

    if rag:
        index = RagIndex(exclude_setting=setting_of(run.changed_service, run._meta["fault_type"],
                                                    run._meta["fault_value"], run.traffic_level))

        @server.tool()
        def search_past_incidents(query: Annotated[str, Field(description="observed symptoms in plain words, e.g. 'pod restarted, memory rising, no other service slower'")],
                                  service: Annotated[Service | None, Field(description="only incidents where this service was changed")] = None,
                                  k: Annotated[int, Field(ge=1, le=10)] = 5) -> dict:
            """Search post-incident summaries of past changes in this system: what the change was, what was visible
            30 s after it and near the end, and how severe it became. Most similar first."""
            log("search_past_incidents", {"query": query, "service": service, "k": k})
            return {"incidents": index.search(query, service, k, mode=rag_mode)}

        @server.tool(description="Read a runbook. Available: " + ", ".join(list_runbooks()) + ". Start with 'triage'.")
        def get_runbook(name: str) -> dict:
            log("get_runbook", {"name": name})
            return {"name": name, "text": _get_runbook(name)}

    return server


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", required=True, help="experiment id, e.g. EXP_0307")
    p.add_argument("--at", type=float, required=True, help="seconds after the change to replay")
    p.add_argument("--no-rag", action="store_true", help="hide search_past_incidents and get_runbook")
    p.add_argument("--rag-mode", choices=SEARCH_MODES, default="hybrid", help="past-incident search method")
    p.add_argument("--log-out", type=Path, help="append one JSON line per tool call")
    a = p.parse_args()
    build_server(a.run, a.at, rag=not a.no_rag, rag_mode=a.rag_mode, log_out=a.log_out).run("stdio")


if __name__ == "__main__":
    main()
