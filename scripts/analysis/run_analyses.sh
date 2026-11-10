#!/usr/bin/env bash
# run_analyses.sh - run the label-reliability and robustness analyses into results/reliability/<name>/,
# and the extra analysis of the agents' affected services into results/genai/affected/.
# Run scripts/run_pipeline.sh first; user_impact also needs the span cache (./scripts/run_genai_eval.sh prepare),
# and affected needs the holdout LLM results (results/genai/holdout/).
#
# Usage: ./scripts/analysis/run_analyses.sh [all | reliability_full | blast_v2 | reliability | robustness | repeat_disagreements | user_impact | structural | null_final | threshold_full | screening_confusion | retrieval_train | affected]
# "all" runs them in order (blast_v2 reads reliability_full's cache). Logs go to logs/analysis/.

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
PY="${PYTHON:-venv/bin/python}"
LOG_DIR=logs/analysis
mkdir -p "${LOG_DIR}"

run() {     # run <name> <script>
	printf '\n=== %s (%s) ===\n' "$1" "$2"
	# one trace file is in memory at a time; the cap makes a problem fail instead of swapping
	( ulimit -v 6500000; "${PY}" "scripts/analysis/$2" ) 2>&1 | tee "${LOG_DIR}/$1.log"
}

case "${1:-all}" in
	all)
		run reliability_full reliability_full.py
		run blast_v2         blast_v2.py
		run reliability      reliability.py
		run robustness       repeat_robustness.py
		run repeat_disagreements repeat_disagreements.py
		run user_impact      user_impact.py
		run structural       structural_exposure.py
		run null_final       null_check_final.py
		run threshold_full   threshold_sensitivity_full.py
		run screening_confusion screening_confusion.py
		run retrieval_train  retrieval_train_check.py
		run affected         score_affected.py
		;;
	reliability_full) run reliability_full reliability_full.py ;;
	blast_v2)         run blast_v2 blast_v2.py ;;
	reliability)      run reliability reliability.py ;;
	robustness)       run robustness repeat_robustness.py ;;
	repeat_disagreements) run repeat_disagreements repeat_disagreements.py ;;
	user_impact)      run user_impact user_impact.py ;;
	structural)       run structural structural_exposure.py ;;
	null_final)       run null_final null_check_final.py ;;
	threshold_full)   run threshold_full threshold_sensitivity_full.py ;;
	screening_confusion) run screening_confusion screening_confusion.py ;;
	retrieval_train)  run retrieval_train retrieval_train_check.py ;;
	affected)         run affected score_affected.py ;;
	*) sed -n '2,8p' "$0" | sed 's/^# \{0,1\}//'; exit 2 ;;
esac
printf '\nDone. Results: results/reliability/ and results/genai/affected/   Logs: %s/\n' "${LOG_DIR}"
