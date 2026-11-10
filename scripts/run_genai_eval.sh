#!/usr/bin/env bash
# run_genai_eval.sh - LLM incident analysis: build its inputs and run its evaluations.
#
# Prerequisites
#   - scripts/run_pipeline.sh has been run (dataset/processed/, models/fe_tuned/)
#   - Phase 3 packages:  venv/bin/pip install -r genai/requirements.txt
#   - stages that call an LLM: the provider's API key, never printed:
#       Claude models (claude-...):  ANTHROPIC_API_KEY or ~/.anthropic_key
#       OpenAI models (gpt-...):     OPENAI_API_KEY or ~/.openai_key
#
# Stages (each one skips work that is already done, so an interrupted stage can simply be re-run)
#   prepare     compact span cache (dataset/spans/), RAG index (models/rag_index/),
#               development set (results/genai/dev_set.txt)
#   retrieval   RAG retrieval evaluation on the holdout -> results/genai/retrieval_eval.csv
#   dev N       development round N: llm_only on the development set
#               -> results/genai/dev/roundN/   (see genai/prompts/README.md)
#   holdout     final evaluation of the frozen configuration: ml_only, llm_only, ml_llm, gated
#               on the 50 holdout incidents -> results/genai/holdout/<model>/rep<K>/
#   sensitivity prompt-sensitivity check: llm_only on the 50 holdout incidents with
#               the alternative prompt (SENS_PROMPT, default v0)
#               -> results/genai/sensitivity/<model>/prompt_<version>/
#   analyze     summary of the holdout, as planned before any holdout run (means over repeats, 95% CIs,
#               paired comparisons, pass^k, sensitivity) -> results/genai/holdout_summary/
#
# Settings (environment variables)
#   MODEL=claude-opus-5-5   EFFORT=medium   CONCURRENCY=3   REPEATS=1   PYTHON=venv/bin/python
#   PROMPT=<version>   system prompt genai/prompts/system_<version>.md; default for "dev N" is vN,
#                      for "holdout" the frozen version (FROZEN_PROMPT in genai/agent.py)
#   SENS_PROMPT=<version>   the alternative prompt for "sensitivity" (default v0)
#
# Examples
#   ./scripts/run_genai_eval.sh prepare
#   ./scripts/run_genai_eval.sh dev 0 --yes
#   REPEATS=3 ./scripts/run_genai_eval.sh holdout --yes
#   MODEL=claude-haiku-5-5 ./scripts/run_genai_eval.sh holdout --yes
#   MODEL=gpt-6-sol ./scripts/run_genai_eval.sh holdout --yes
#   MODEL=gpt-6-sol ./scripts/run_genai_eval.sh sensitivity --yes
#
# Every investigation is saved as it finishes (results.csv, plus prompt, transcript and report per incident),
# so a stopped run resumes where it left off. Add --retry-errors to re-run investigations that failed.
# If the API account runs out of credit (or hits its spending limit), the run stops by itself (exit code 3)
# without recording the remaining incidents as failed: add credit and re-run the same command to continue.
# Output is logged to logs/genai/.

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
PY="${PYTHON:-venv/bin/python}"
MODEL="${MODEL:-claude-opus-5-5}"
EFFORT="${EFFORT:-medium}"
CONCURRENCY="${CONCURRENCY:-3}"
REPEATS="${REPEATS:-1}"
export PYDANTIC_AI_NO_BANNER=1
LOG_DIR=logs/genai
mkdir -p "${LOG_DIR}"

usage() { sed -n '2,43p' "$0" | sed 's/^# \{0,1\}//'; exit 2; }
[[ $# -ge 1 ]] || usage
STAGE="$1"; shift

YES=0; RETRY=(); ARGS=()
for a in "$@"; do
	case "$a" in
		--yes)          YES=1 ;;
		--retry-errors) RETRY=(--retry-errors) ;;
		*)              ARGS+=("$a") ;;
	esac
done

need_yes() {    # need_yes <investigations> <description>
	local n="$1" what="$2"
	if (( ! YES )); then
		echo "This stage calls the LLM API: about ${n} investigations (${what}) with ${MODEL}." >&2
		echo "Every investigation is a paid API call, billed to the key's account." >&2
		echo "Re-run with --yes to start." >&2
		exit 1
	fi
	case "${MODEL}" in
		gpt-*) [[ -n "${OPENAI_API_KEY:-}" || -f "${HOME}/.openai_key" ]] || { echo "No API key: set OPENAI_API_KEY or create ~/.openai_key" >&2; exit 1; } ;;
		*)     [[ -n "${ANTHROPIC_API_KEY:-}" || -f "${HOME}/.anthropic_key" ]] || { echo "No API key: set ANTHROPIC_API_KEY or create ~/.anthropic_key" >&2; exit 1; } ;;
	esac
}

run() {         # run <log name> <command...>
	local log="${LOG_DIR}/$1.log"; shift
	printf '\n=== %s ===\n' "$*" | tee -a "${log}"
	"$@" 2>&1 | grep --line-buffered -vE '^[0-9]{4}-[0-9]{2}-[0-9]{2}T' | tee -a "${log}"
}

case "${STAGE}" in
	prepare)
		run prepare "${PY}" -m genai.replay build-cache
		[[ -f models/rag_index/incidents.jsonl ]] || run prepare "${PY}" -m genai.knowledge build
		[[ -f results/genai/dev_set.txt ]] || run prepare "${PY}" -m genai.dev_set
		;;
	retrieval)
		run retrieval "${PY}" -m genai.eval_retrieval
		;;
	dev)
		N="${ARGS[0]:?usage: run_genai_eval.sh dev <round number> --yes}"
		need_yes 40 "llm_only on the 40 development incidents"
		run "dev_round${N}" "${PY}" -m genai.orchestrator --runs-file results/genai/dev_set.txt \
			--systems llm_only --out "results/genai/dev/round${N}" --model "${MODEL}" --effort "${EFFORT}" \
			--prompt "${PROMPT:-v${N}}" --concurrency "${CONCURRENCY}" "${RETRY[@]}"
		;;
	holdout)
		need_yes "$(( 134 * REPEATS ))" "4 systems on 50 holdout incidents, ${REPEATS} repeat(s); the gate skips ~16 incidents"
		"${PY}" -c "from genai.replay import holdout_ids; print('\n'.join(holdout_ids()))" > "${LOG_DIR}/holdout_ids.txt"
		for (( k = 1; k <= REPEATS; k++ )); do
			run "holdout_${MODEL}" "${PY}" -m genai.orchestrator --runs-file "${LOG_DIR}/holdout_ids.txt" \
				--systems ml_only llm_only ml_llm gated --out "results/genai/holdout/${MODEL}/rep${k}" \
				--model "${MODEL}" --effort "${EFFORT}" ${PROMPT:+--prompt "${PROMPT}"} --concurrency "${CONCURRENCY}" "${RETRY[@]}"
		done
		;;
	analyze)
		run analyze "${PY}" -m genai.analyze_holdout
		;;
	sensitivity)
		SP="${SENS_PROMPT:-v0}"
		need_yes 50 "llm_only with prompt ${SP} on the 50 holdout incidents"
		"${PY}" -c "from genai.replay import holdout_ids; print('\n'.join(holdout_ids()))" > "${LOG_DIR}/holdout_ids.txt"
		run "sensitivity_${MODEL}" "${PY}" -m genai.orchestrator --runs-file "${LOG_DIR}/holdout_ids.txt" \
			--systems llm_only --out "results/genai/sensitivity/${MODEL}/prompt_${SP}" --model "${MODEL}" \
			--effort "${EFFORT}" --prompt "${SP}" --concurrency "${CONCURRENCY}" "${RETRY[@]}"
		;;
	*)
		usage
		;;
esac
