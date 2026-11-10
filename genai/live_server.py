"""
MCP server for a live incident: the evaluation tools (genai/mcp_server.py) over a fresh snapshot
of the cluster that genai.live writes in the recorded-run format, and one live-only tool, get_logs.

Two deliberate differences from the evaluation server:
- search_past_incidents does not hide past incidents of the same setting. In the evaluation they are hidden, so the
  agent cannot look up the answer; live, finding a matching past incident is exactly what retrieval is for.
- get_logs reads the service's recent pod logs with kubectl. Recorded runs have no logs, so this tool exists only here.

    python genai/live_server.py --raw-dir results/genai/live/snapshots --run LIVE_... --at 30 [--log-out calls.jsonl]
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Annotated

from pydantic import Field

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import genai.mcp_server as S          # noqa: E402
import genai.replay as R              # noqa: E402
from genai.knowledge import RagIndex  # noqa: E402
from genai.mcp_server import Service  # noqa: E402

NAMESPACE = "online-boutique"
MAX_CHARS = 6000


def read_logs(service: str, since_seconds: float, max_lines: int, contains: str | None, previous: bool,
              namespace: str = NAMESPACE) -> dict:
    """Recent log lines of every pod of `service` (kubectl logs -l app=<service>)."""
    cmd = ["kubectl", "logs", "-n", namespace, "-l", f"app={service}", "--all-containers=true", "--prefix",
           "--timestamps", f"--since={max(int(since_seconds), 1)}s", f"--tail={max_lines}"]
    if previous:
        cmd.append("--previous")
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=25)
    except (OSError, subprocess.TimeoutExpired) as e:
        return {"service": service, "lines": [], "note": f"could not read logs: {type(e).__name__}"}
    lines = p.stdout.splitlines()
    if contains:
        lines = [x for x in lines if contains.lower() in x.lower()]
    out, size = [], 0
    for line in lines[-max_lines:]:
        size += len(line)
        if size > MAX_CHARS:
            break
        out.append(line)
    note = None
    if p.returncode != 0:
        note = (p.stderr.strip().splitlines() or ["kubectl failed"])[-1][:200]
    elif not lines:
        note = "no matching log lines (or no running pods)"
    return {"service": service, "since_seconds": since_seconds, "previous_container": previous,
            "lines": out, **({"note": note} if note else {})}


def make_server(raw_dir: Path, run_id: str, at: float, rag: bool, rag_mode: str, log_out: Path | None,
                namespace: str = NAMESPACE):
    raw_dir = Path(raw_dir)

    class LiveRun(R.RecordedRun):
        """A live snapshot, stored like a recorded run but outside dataset/raw."""
        def __init__(self, experiment_id: str, raw_dir_: Path = raw_dir):
            super().__init__(experiment_id, raw_dir_)

    S.RecordedRun = LiveRun                                                 # build_server looks these up by name
    S.RagIndex = lambda exclude_setting=None: RagIndex(exclude_setting=None)
    server = S.build_server(run_id, at, rag=rag, rag_mode=rag_mode, log_out=log_out)
    started = time.time()

    @server.tool()
    def get_logs(service: Service,
                 since_seconds: Annotated[float, Field(description="how far back to read, in seconds", ge=5, le=900)] = 120,
                 max_lines: Annotated[int, Field(ge=1, le=200)] = 60,
                 contains: Annotated[str | None, Field(description="keep only lines containing this text, e.g. 'error'")] = None,
                 previous: Annotated[bool, Field(description="read the previous (crashed) container instead")] = False) -> dict:
        """Recent log lines of the service's pods (live only). Useful to see errors, timeouts or crash messages."""
        if log_out:
            with open(log_out, "a") as f:
                f.write(json.dumps({"t": round(time.time() - started, 2), "tool": "get_logs",
                                    "args": {"service": service, "since_seconds": since_seconds,
                                             "max_lines": max_lines, "contains": contains, "previous": previous}}) + "\n")
        return read_logs(service, since_seconds, max_lines, contains, previous, namespace)

    return server


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--raw-dir", type=Path, required=True, help="folder holding the snapshot folder <run>/")
    p.add_argument("--run", required=True, help="snapshot (incident) id")
    p.add_argument("--at", type=float, required=True, help="seconds after the change that the snapshot ends")
    p.add_argument("--no-rag", action="store_true")
    p.add_argument("--rag-mode", choices=S.SEARCH_MODES, default="hybrid")
    p.add_argument("--log-out", type=Path)
    p.add_argument("--namespace", default=NAMESPACE)
    a = p.parse_args()
    make_server(a.raw_dir, a.run, a.at, not a.no_rag, a.rag_mode, a.log_out, a.namespace).run("stdio")


if __name__ == "__main__":
    main()
