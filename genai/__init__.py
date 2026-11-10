"""
Phase 3: LLM incident investigation on top of the impact-forecasting models.

Offline evaluation on recorded runs
  replay           replays a recorded run as if it were happening now
  mcp_server       the agent's tools (MCP) for one replayed incident
  knowledge        retrieval (RAG): past-incident summaries and runbooks
  impact_model     the trained ML models (models/fe_tuned): the screen and the forecast in the prompt
  agent            the Pydantic AI agent and its versioned prompts (genai/prompts/)
  orchestrator     runs and scores the four systems (ml_only, llm_only, ml_llm, gated)
  dev_set          picks the development incidents
  eval_retrieval   evaluates retrieval on its own
  analyze_holdout  the holdout analysis, as planned before any holdout run

Live mode on a running cluster
  live             watch the cluster, screen changes with the ML model, investigate, alert
  live_server      the same tools over a fresh snapshot, plus pod logs
  notify           the Slack message
"""
