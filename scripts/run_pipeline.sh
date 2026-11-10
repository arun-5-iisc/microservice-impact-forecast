#!/usr/bin/env bash
# run_pipeline.sh - rebuild labels, features, models, figures and slides from dataset/raw/.
#
# Stages, in order:
#    1. Labels, early features   preprocess.py
#    2. Label justification      threshold_analysis.py --mode threshold   (all experiments)
#    3. Holdout                  drawn once (create_holdout.py) if dataset/processed/holdout_ids.txt
#                                doesn't exist; otherwise checked: all runs present, no setting split
#    4. Feature analysis (base)  threshold_analysis.py --mode features
#    5. Feature engineering      feature_engineer.py, then threshold_analysis.py --mode features
#    6. Untuned models           train_model.py, stages "baseline" and "fe"
#    7. Tuning (50 trials each)  tune_model.py: classifiers on base features, regressors on FE features
#    8. Final stage              winning classifier + regressor gathered into results/ and models/fe_tuned
#    9. Explainability           explain_model.py (SHAP)
#   10. Figures                  visualize.py, plot_labels.py (and the slide deck, if scripts/build_ppt.py is present)
#
# Usage:
#   ./scripts/run_pipeline.sh --yes            # all stages
#   ./scripts/run_pipeline.sh --yes --from 7   # resume from a stage
#   TRIALS=2 ./scripts/run_pipeline.sh --yes   # quick smoke test (2 tuning trials instead of 50)
#
# Overwrites dataset/processed/, results/ and models/, so it refuses to start without --yes.
# Each stage's output is logged to logs/pipeline/.

set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."
PY="${PYTHON:-venv/bin/python}"
BASE=dataset/processed/training_dataset.csv
FE=dataset/processed/training_dataset_fe.csv
LOG_DIR=logs/pipeline

# Stages whose tuned models go into fe_tuned, picked on cross-validation (results/<stage>/final_results.csv),
# never on the holdout: XGBoost for both (severity: higher macro-F1 and accuracy; blast radius: lower MAE
# and RMSE).
# Override without editing, e.g.:  CLF_STAGE=baseline_clf_rf CLF_MODEL=clf_rf_tuned.joblib ./scripts/run_pipeline.sh --yes --from 8
CLF_STAGE="${CLF_STAGE:-baseline_clf_xgb}"   CLF_MODEL="${CLF_MODEL:-clf_xgb_tuned.joblib}"
REG_STAGE="${REG_STAGE:-fe_reg_xgb}"         REG_MODEL="${REG_MODEL:-reg_xgb_tuned.joblib}"

TUNE_ARGS=()
if [[ -n "${TRIALS:-}" ]]; then TUNE_ARGS=(--trials "${TRIALS}"); fi

FROM=1
YES=0
while (( $# )); do
	case "$1" in
		--from) FROM="${2:?--from needs a stage number}"; shift 2 ;;
		--yes)  YES=1; shift ;;
		*)      echo "Unknown option: $1" >&2; exit 2 ;;
	esac
done
if (( ! YES )); then
	echo "This rebuilds and overwrites dataset/processed/, results/ and models/ in $(pwd)." >&2
	echo "Back up anything you need from them, then re-run with --yes." >&2
	exit 1
fi

mkdir -p "${LOG_DIR}"

stage() {   # stage <number> <name> <command...>
	local n="$1" name="$2"; shift 2
	if (( n < FROM )); then return; fi
	printf '\n=== [%2d] %s ===\n' "${n}" "${name}"
	"$@" 2>&1 | tee "${LOG_DIR}/$(printf '%02d' "${n}")_${name// /_}.log"
}

prepare_holdout() {   # draw once, then only ever check
	if [[ ! -f dataset/processed/holdout_ids.txt ]]; then
		"${PY}" scripts/create_holdout.py --data "${BASE}"
	fi
	"${PY}" - "${BASE}" <<'EOF'
import sys
from pathlib import Path
import pandas as pd
df = pd.read_csv(sys.argv[1]).drop_duplicates("experiment_id")
setting = df["service"] + "|" + df["fault_type"] + "|" + df["fault_value"].astype(str) + "|" + df["traffic_level"]
run_setting = dict(zip(df["experiment_id"], setting))
holdout = set(Path("dataset/processed/holdout_ids.txt").read_text().split())
missing = sorted(h for h in holdout if h not in run_setting)
if missing:
    sys.exit(f"holdout_ids.txt names {len(missing)} runs that are not in the dataset: {missing}\n"
             "It was drawn for a different dataset: move it aside, then resume with --from 3.")
held = {run_setting[h] for h in holdout}
split = sorted({s for e, s in run_setting.items() if s in held and e not in holdout})
if split:
    sys.exit(f"Repeat runs of held-out settings are in training: {split}\n"
             "Every run of a held-out setting must be listed in holdout_ids.txt.")
print(f"Holdout OK: {len(holdout)} runs ({len(held)} settings), no setting split across train/holdout.")
EOF
}

gather_final_stage() {
	mkdir -p results/fe_tuned models/fe_tuned
	rm -f models/fe_tuned/*.joblib     # only the chosen pair may be here (genai/impact_model.py loads it)
	cp "results/${CLF_STAGE}"/{oof_predictions.csv,accuracy_vs_time_tuned.csv,accuracy_vs_time_holdout.csv,holdout_report.csv,holdout_predictions.csv} results/fe_tuned/
	cp "results/${REG_STAGE}"/{holdout_regression_report.csv,holdout_regression_predictions.csv,mae_vs_time_holdout.csv} results/fe_tuned/
	cp "models/${CLF_STAGE}/${CLF_MODEL}" "models/${REG_STAGE}/${REG_MODEL}" models/fe_tuned/
	"${PY}" - "${CLF_STAGE}" "${REG_STAGE}" <<'EOF'
import sys
import pandas as pd
clf_stage, reg_stage = sys.argv[1:]
clf = pd.read_csv(f"results/{clf_stage}/final_results.csv")
reg = pd.read_csv(f"results/{reg_stage}/final_results.csv")
combined = pd.concat([clf, reg[reg["stage"] == reg_stage]], ignore_index=True)
combined.to_csv("results/fe_tuned/final_results.csv", index=False)
print(combined.to_string(index=False))

# Model vs no-ML baselines: severity rows from the classifier stage, blast-radius rows from the regressor stage
cmp = pd.concat([pd.read_csv(f"results/{clf_stage}/baseline_comparison.csv").query("task == 'severity'"),
                 pd.read_csv(f"results/{reg_stage}/baseline_comparison.csv").query("task == 'blast_radius'")],
                ignore_index=True)
cmp.to_csv("results/fe_tuned/baseline_comparison.csv", index=False)
pd.read_csv(f"results/{clf_stage}/forecasting_gain.csv").to_csv("results/fe_tuned/forecasting_gain.csv", index=False)
print("\nModel vs no-ML baselines:\n" + cmp.to_string(index=False))
EOF
}

stage 1  "preprocess"         "${PY}" scripts/preprocess.py
stage 2  "label threshold"    "${PY}" scripts/threshold_analysis.py --mode threshold
stage 3  "holdout"            prepare_holdout
stage 4  "features base"      "${PY}" scripts/threshold_analysis.py --mode features --data "${BASE}"
stage 5  "feature engineer"   "${PY}" scripts/feature_engineer.py
stage 5  "features fe"        "${PY}" scripts/threshold_analysis.py --mode features --data "${FE}"
stage 6  "train baseline"     "${PY}" scripts/train_model.py --data "${BASE}" --stage baseline
stage 6  "train fe"           "${PY}" scripts/train_model.py --data "${FE}" --stage fe
stage 7  "tune clf xgb"       "${PY}" scripts/tune_model.py --data "${BASE}" --task clf --clf-model xgb --stage baseline_clf_xgb "${TUNE_ARGS[@]}"
stage 7  "tune clf rf"        "${PY}" scripts/tune_model.py --data "${BASE}" --task clf --clf-model rf  --stage baseline_clf_rf  "${TUNE_ARGS[@]}"
stage 7  "tune reg rf"        "${PY}" scripts/tune_model.py --data "${FE}"   --task reg --reg-model rf  --stage fe_reg_rf        "${TUNE_ARGS[@]}"
stage 7  "tune reg xgb"       "${PY}" scripts/tune_model.py --data "${FE}"   --task reg --reg-model xgb --stage fe_reg_xgb       "${TUNE_ARGS[@]}"
stage 8  "final stage"        gather_final_stage
stage 9  "explain clf"        "${PY}" scripts/explain_model.py --model "models/${CLF_STAGE}/${CLF_MODEL}" --data "${BASE}" --stage "${CLF_STAGE}" --task clf
stage 9  "explain reg"        "${PY}" scripts/explain_model.py --model "models/${REG_STAGE}/${REG_MODEL}" --data "${FE}" --stage "${REG_STAGE}" --task reg
stage 10 "visualize"          "${PY}" scripts/visualize.py
stage 10 "plot labels"        "${PY}" scripts/plot_labels.py
# the slide deck script is kept outside the repository; build the deck only where it exists
if [[ -f scripts/build_ppt.py ]]; then stage 10 "slides" "${PY}" scripts/build_ppt.py; fi

printf '\nPipeline complete. Logs: %s/\n' "${LOG_DIR}"
