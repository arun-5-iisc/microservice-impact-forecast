"""
Summary of the live demonstration (gpt-6-sol as the agent), one row per ML screen.

The live records themselves are not pushed (results/genai/live/ and logs/ are in .gitignore), so this summary is
the pushed record behind the live table in the report. It reads the watch logs of the demonstration, given on the
command line. Each change in a log (from "change detected" to "restored") gives:
  - if the ML screen flagged it and the agent investigated: one row with the ML forecast, the agent's verdict,
    cause, action and confidence, its tool calls, time and cost, and when the alert was sent to Slack, taken from
    the investigation's live_incident.json (the log names its folder in results/genai/live/);
  - if the ML screen kept it Low up to 60 s: one row per snapshot with the ML probabilities from the log, because a
    change that is not investigated keeps only its features.

Times are the time of day in UTC. slack_sent_utc is the time slack.json was written, which is right after the alert
is posted; change_to_slack_s is that time minus the change time.

Needs the local live records and logs.
Usage:  venv/bin/python scripts/analysis/live_summary.py LOG [LOG ...] [--out PATH]
        (LOG: a watch log in logs/live/)   -> results/genai/live_demo_summary.csv
"""
from __future__ import annotations

import argparse
import ast
import csv
import datetime as dt
import json
import os
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LIVE = ROOT / "results" / "genai" / "live"

CHANGE = re.compile(r"^change detected: (\S+) (.+) at (\d{4}-\d\d-\d\d \d\d:\d\d:\d\d) UTC$")
SCREEN = re.compile(r"^\s*ML screen at (\d+) s: (\w+) (\{.*\})$")
INVESTIGATION = re.compile(r"^\[(\S+)\] .*sent to Slack$")


def utc(t: float) -> dt.datetime:
    return dt.datetime.fromtimestamp(t, dt.timezone.utc)


def changes(log: Path) -> list[dict]:
    """The changes in one watch log, each with its ML screens and, if investigated, the investigation's folder."""
    out, cur = [], None
    for line in log.read_text().splitlines():
        if m := CHANGE.match(line):
            cur = dict(service=m.group(1), change=m.group(2), time=m.group(3), screens=[], investigation=None)
            out.append(cur)
        elif cur is None:
            continue
        elif m := SCREEN.match(line):
            cur["screens"].append((int(m.group(1)), m.group(2), ast.literal_eval(m.group(3))))
        elif m := INVESTIGATION.match(line):
            cur["investigation"] = m.group(1)
        elif line.startswith("restored:"):
            cur = None
    return out


def investigated_row(c: dict) -> dict:
    run = LIVE / c["investigation"]
    j = json.loads((run / "live_incident.json").read_text())
    f, r = j["forecast"], j["report"]
    p = f["severity_probabilities"]
    sent = os.stat(run / "slack.json").st_mtime
    return dict(
        service=j["service"], change=j["change"],
        change_time_utc=utc(j["change_time"]).strftime("%H:%M:%S"),
        snapshot_s=f["snapshot_s"], ml_severity=f["predicted_final_severity"],
        ml_prob_low=p["Low"], ml_prob_medium=p["Medium"], ml_prob_high=p["High"],
        ml_blast_radius=f["predicted_blast_radius"], investigated="yes", agent_model=j["model"],
        agent_severity=r["predicted_final_severity"], agent_cause=r["probable_cause"],
        agent_action=r["recommended_action"], agent_confidence=r["confidence"], tool_calls=j["tool_calls"],
        investigation_s=j["wall_s"], cost_usd=round(j["cost_usd"], 4),
        slack_sent_utc=utc(sent).strftime("%H:%M:%S.%f")[:-4],
        change_to_slack_s=round(sent - j["change_time"], 1),
    )


def screened_low_rows(c: dict) -> list[dict]:
    return [dict(service=c["service"], change=c["change"], change_time_utc=c["time"][11:], snapshot_s=t,
                 ml_severity=sev, ml_prob_low=p["Low"], ml_prob_medium=p["Medium"], ml_prob_high=p["High"],
                 investigated="no", cost_usd=0)
            for t, sev, p in c["screens"]]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("logs", nargs="+", type=Path, help="watch logs of the demonstration")
    ap.add_argument("--out", type=Path, default=ROOT / "results" / "genai" / "live_demo_summary.csv")
    args = ap.parse_args()

    rows = []
    for log in args.logs:
        for c in changes(log):
            rows += [investigated_row(c)] if c["investigation"] else screened_low_rows(c)
    cols = list(dict.fromkeys(k for r in rows for k in r))
    with open(args.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    print(f"{len(rows)} rows -> {args.out}")


if __name__ == "__main__":
    main()
