# Label-reliability and robustness analyses

These scripts produce the evidence in `results/reliability/` behind the report's validation of the label measurement
and its robustness checks. Run them all with `./scripts/analysis/run_analyses.sh`, after `scripts/run_pipeline.sh`.
They read the project's data and write only to `results/reliability/<name>/` (`score_affected.py`: `results/genai/affected/`).

| Script | Output folder | Question | Measurement studied |
|---|---|---|---|
| `reliability_full.py` | `reliability_full/` | How stable is each run's label under sampling noise (bootstrap of its own traces)? Does the blast radius find "affected" services when nothing changed (two baseline halves) vs during faults? | **Midterm** measurement (root span of every trace), reimplemented on purpose |
| `blast_v2.py` | `blast_v2/` | Candidate blast-radius rules (D0 midterm, D1, D2 with a significance test) against no-change noise and on the 20 repeated settings. Reads `reliability_full/cache.pkl`, so run it after `reliability_full.py`. | Midterm vs candidate rules |
| `reliability.py` | `reliability/` | Do the 20 settings that were run twice get the same label, under the midterm definition and alternatives? | Midterm vs alternatives |
| `repeat_robustness.py` | `robustness/` | Do the 20 repeated settings skew the ML results? Trains with one run per setting vs the full dataset, using the pipeline's tuned hyperparameters. | Final labels and models |
| `user_impact.py` | `user_impact/` | How often does the changed service's severity differ from the effect users see at the frontend? | Final labels (`span_measures.py`) |
| `structural_exposure.py` | `structural_exposure/` | Which services can a change reach through the call graph, how many of them were actually affected, and how well does the graph alone predict the blast radius? | Final labels and the call graph |
| `null_check_final.py` | `null_final/` | The no-change check for the **final** measurement: blast radius between the two baseline halves (should be ~0) and during the change, computed with `span_measures.py` itself. Needs the span cache (`./scripts/run_genai_eval.sh prepare`). Its numbers are the "final" row of the report's validation table. | Final labels |
| `screening_confusion.py` | `screening_confusion/` | Confusion matrices of the untuned severity classifiers on both feature sets (`train_model.py` does not save them), with a check against its reported accuracies. Used by the report's screening table and appendix. | Final labels |
| `threshold_sensitivity_full.py` | `threshold_full/` | How many final labels change when one threshold of the severity rule is moved (latency 20% and 100%, error rate 1% and 5%), applying the full rule? Complements the pipeline's latency-only threshold analysis. | Final labels |
| `retrieval_train_check.py` | `results/genai/retrieval_train/` (not under `results/reliability/`) | Keyword, vector and hybrid search compared on the training runs, each run's own setting hidden, so the choice of method does not depend on the holdout. Reads a temporary copy of the search index. | Final labels |
| `score_affected.py` | `results/genai/affected/` (not under `results/reliability/`) | Do the LLM agents name the right affected services? Compares each holdout investigation's list with the measured blast radius, against the ML regressor at 30 s. An extra analysis, not part of the pre-planned LLM evaluation; reads saved results only, no model calls. | Final labels and the holdout LLM results |

The first three scripts are the **validation of the label measurement**. They test the first, root-span measurement
(the one used for the midterm presentation) and show it was mostly noise; for example, its blast radius counted 2.65
"affected" services with no change at all, against 2.41 during real faults. That is why the final labels use the
request-level measurement in `scripts/span_measures.py`. The scripts keep their own copy of the midterm error test, so
they measure exactly what the midterm labels measured.

Their reference labels are the **midterm labels**, frozen in `midterm_labels.csv` (one row per run: the setting, the
severity and the blast radius). `training_dataset.csv` holds the final labels, so these three scripts do not read it.
Each starts with a reproduction check: its copy of the midterm measurement, run on the raw traces, must give exactly
the labels in `midterm_labels.csv` ("Reproduction check ... OK" at the top of each `report.txt`).

The final labels themselves are measured by `scripts/span_measures.py`; its module docstring explains the measurement.
