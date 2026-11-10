"""
Slack notification for one investigated change. The message gives:
- what changed and when;
- the forecast severity, with an alert only for Medium and High;
- the root cause, the affected services, and the recommended action with its remediation steps
  (the runbook's Response section plus concrete commands);
- the agent's evidence and the ML screen's forecast.

Sending uses a Slack Incoming Webhook (Slack app -> Incoming Webhooks -> add to a channel). Its URL is read from
SLACK_WEBHOOK_URL or ~/.slack_webhook (chmod 600) and is never printed. Without a webhook, or with dry_run=True,
the message is printed instead of sent.

    python -m genai.notify <investigation folder>            # re-send a saved live incident (live_incident.json)
    python -m genai.notify <investigation folder> --dry-run  # print the Slack payload instead
"""
from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

import requests

RUNBOOK_DIR = Path(__file__).resolve().parent / "knowledge" / "runbooks"
NAMESPACE = "online-boutique"
HELM_RELEASE = "online-boutique"

SEVERITY_ICON = {"High": ":red_circle:", "Medium": ":large_orange_circle:", "Low": ":large_green_circle:"}
CAUSE_TEXT = {
    "cpu_limit": "CPU starvation: the CPU limit is too low for the load",
    "memory_limit": "Memory pressure: the memory limit is too low",
    "network_latency": "Network delay on calls to the service",
    "service_unavailable": "Service unavailable: no running replicas",
    "no_clear_effect": "No measurable effect",
    "unclear": "Not clear yet",
}
CAUSE_RUNBOOK = {"cpu_limit": "cpu-throttling", "memory_limit": "memory-pressure",
                 "network_latency": "network-latency", "service_unavailable": "service-unavailable"}
ACTION_TEXT = {
    "rollback_change": "Roll back the change",
    "increase_resource_limits": "Raise the resource limits back to their previous values",
    "scale_up_replicas": "Scale the service back up",
    "monitor": "Monitor; no change needed yet",
    "no_action": "No action needed",
}
# Generic commands, used only when the change itself is not known (see undo_commands for the exact ones).
ACTION_COMMANDS = {
    "rollback_change": [f"kubectl rollout undo deployment/{{svc}} -n {NAMESPACE}",
                        f"helm rollback {HELM_RELEASE} -n {NAMESPACE}   # if the change was a Helm upgrade"],
    "increase_resource_limits": [f"kubectl set resources deployment/{{svc}} -c server -n {NAMESPACE} "
                                 "--limits=cpu=<previous>,memory=<previous>"],
    "scale_up_replicas": [f"kubectl scale deployment/{{svc}} -n {NAMESPACE} --replicas=1"],
}
# A replica count of 0 is not undone by "rollout undo" (it reverts the pod template only), whatever the action says.
CAUSE_COMMANDS = {
    "service_unavailable": [f"kubectl scale deployment/{{svc}} -n {NAMESPACE} --replicas=1"],
    "network_latency": [f"kubectl get networkchaos -n {NAMESPACE}   # find the delay rule on the service",
                        f"kubectl delete networkchaos <name> -n {NAMESPACE}"],
}
UNDO_ACTIONS = {"rollback_change", "increase_resource_limits", "scale_up_replicas"}


def undo_commands(svc: str, change: dict | None) -> list[str]:
    """The exact command(s) that undo the detected change. Live watching knows the change: its kind, the previous
    value and, for a network delay, the name of the Chaos Mesh rule."""
    if not change:
        return []
    fault, prev = change.get("fault"), change.get("previous")
    known = prev not in (None, "", "None", "none")
    if fault == "latency":
        if change.get("chaos"):
            return [f"kubectl delete networkchaos {change['chaos']} -n {NAMESPACE}"]
        return CAUSE_COMMANDS["network_latency"]
    if fault == "replicas":
        return [f"kubectl scale deployment/{svc} -n {NAMESPACE} --replicas={prev if known else 1}"]
    if fault in ("cpu", "memory"):
        cmds = [f"helm rollback {HELM_RELEASE} -n {NAMESPACE}   # back to the release before the change"]
        if known:
            cmds.append(f"kubectl set resources deployment/{svc} -c server -n {NAMESPACE} --limits={fault}={prev}"
                        "   # instead, if the limit was changed with kubectl")
        return cmds
    return []


def remediation_commands(svc: str, cause: str, action: str, change: dict | None) -> list[str]:
    if action not in UNDO_ACTIONS:
        return []                                   # monitor / no action: nothing to run
    return (undo_commands(svc, change) or
            [c.format(svc=svc) for c in CAUSE_COMMANDS.get(cause, ACTION_COMMANDS.get(action, []))])


def runbook_response(cause: str) -> list[str]:
    """The bullet points of the matching runbook's **Response** section (triage for an unclear cause)."""
    path = RUNBOOK_DIR / f"{CAUSE_RUNBOOK.get(cause, 'triage')}.md"
    if not path.exists():
        return []
    text = path.read_text()
    m = re.search(r"\*\*Response\*\*\s*\n((?:\s*- .*\n?)+)", text)
    return [line.strip()[2:] for line in m.group(1).splitlines() if line.strip().startswith("- ")] if m else []


def build_message(incident: dict) -> dict:
    """Slack Block Kit payload for one incident (see genai.live for the incident record)."""
    r, svc = incident["report"], incident["service"]
    sev, cause, action = r["predicted_final_severity"], r["probable_cause"], r["recommended_action"]
    alert = sev in ("Medium", "High")
    title = (f"{SEVERITY_ICON[sev]} {sev} impact forecast for {svc}" if alert
             else f"{SEVERITY_ICON[sev]} Change to {svc}: no significant impact expected")
    others = [s for s in r.get("affected_services", []) if s != svc]
    f = incident.get("forecast") or {}
    ml = (f"{f['predicted_final_severity']} ({f['severity_probabilities'][f['predicted_final_severity']]:.0%} "
          f"probability); about {f['predicted_blast_radius']:.1f} other services affected"
          if f.get("available") else "not used")
    steps = [ACTION_TEXT[action]] + remediation_commands(svc, cause, action, incident.get("change_detail"))
    guidance = runbook_response(cause)
    remediation = "\n".join([f"*{steps[0]}*"] + [f"`{c}`" for c in steps[1:]]
                            + ([f"_Runbook ({CAUSE_RUNBOOK.get(cause, 'triage')}):_"] + [f"• {g}" for g in guidance]
                               if guidance else []))
    fields = [("Changed service", svc),
              ("Changed at", incident.get("change_time_text", "")),
              ("Assessed", f"{incident.get('decided_at_s', '?')} s after the change"),
              ("Agent's verdict", f"{sev} (self-rated confidence {r.get('confidence', 0):.2f})"),
              ("Root cause", CAUSE_TEXT.get(cause, cause)),
              ("Affected (agent)", ", ".join(others) if others else "none so far"),
              ("ML forecast", ml)]
    if incident.get("change"):
        fields.insert(1, ("Change", incident["change"]))
    blocks = [
        {"type": "header", "text": {"type": "plain_text", "text": title[:150], "emoji": True}},
        {"type": "section", "fields": [{"type": "mrkdwn", "text": f"*{k}*\n{v}"} for k, v in fields[:10]]},
        {"type": "section", "text": {"type": "mrkdwn", "text": f"*Recommended action*\n{remediation}"[:3000]}},
        {"type": "section", "text": {"type": "mrkdwn", "text": f"*Evidence*\n{r.get('evidence', '')}"[:3000]}},
        {"type": "context", "elements": [{"type": "mrkdwn", "text":
            f"{incident.get('model', '')} · ${incident.get('cost_usd') or 0:.3f} · "
            "recommendation only: nothing was changed automatically"}]},
    ]
    return {"text": f"{sev} impact forecast for {svc}: {CAUSE_TEXT.get(cause, cause)}. {ACTION_TEXT[action]}.",
            "blocks": blocks}


def webhook_url() -> str | None:
    url = os.environ.get("SLACK_WEBHOOK_URL")
    f = Path.home() / ".slack_webhook"
    if not url and f.exists():
        url = f.read_text().strip()
    return url or None


def send(payload: dict, dry_run: bool = False) -> bool:
    """Post to Slack; returns True if sent. Prints the payload instead when dry_run or no webhook is set."""
    url = None if dry_run else webhook_url()
    if not url:
        print(("DRY RUN" if dry_run else "No Slack webhook (SLACK_WEBHOOK_URL or ~/.slack_webhook)") +
              ": message not sent. Payload:\n" + json.dumps(payload, indent=2))
        return False
    resp = requests.post(url, json=payload, timeout=15)
    if resp.status_code != 200:
        raise RuntimeError(f"Slack returned HTTP {resp.status_code}: {resp.text[:200]}")
    return True


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("folder", type=Path, help="investigation folder containing live_incident.json")
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args()
    incident = json.loads((a.folder / "live_incident.json").read_text())
    print("sent" if send(build_message(incident), dry_run=a.dry_run) else "")


if __name__ == "__main__":
    main()
