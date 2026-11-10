# LLM evaluation results

| Path | What it is |
|---|---|
| `holdout/<model>/rep<K>/results.csv` | The holdout evaluation: 50 incidents x 4 systems (ml_only, llm_only, ml_llm, gated), 3 repeats per model. One row per investigation: the answer, its evidence text, the scores, tokens and cost. |
| `sensitivity/<model>/prompt_v0/results.csv` | The prompt-sensitivity check: llm_only on the holdout with prompt v0. |
| `holdout_summary/` | The analysis of the two above (`genai/analyze_holdout.py`): `summary.md`, `main.csv`, `paired.csv`, `sensitivity.csv`, `errors.csv`. |
| `dev/round0/`, `dev/round1/` | The development rounds on the development set (`dev_set.txt`), prompts v0 and v1. |
| `retrieval_eval*.csv`, `retrieval_queries.jsonl` | Retrieval (RAG) evaluated on its own (`genai/eval_retrieval.py`). |
| `retrieval_train/` | Retrieval methods compared on the training runs, without the holdout (`scripts/analysis/retrieval_train_check.py`): `summary.csv`, `paired.csv`, `per_query.csv`. |
| `affected/` | Extra analysis, not part of the pre-planned evaluation: the agents' affected services compared with the measured blast radius (`scripts/analysis/score_affected.py`): `summary.csv`, `per_incident.csv`, `llm_only_by_true_blast_radius.csv`. |

Every LLM-evaluation number in the report is computed from these files.

## Transcripts

Each investigation also saved a folder next to its `results.csv`
(`<system>/<incident>/`) with the full prompt (`prompt.txt`), the conversation with the model
(`messages.json`), every tool call (`calls.jsonl`) and the final report (`report.json`). These 1,472
folders (holdout 1,242, prompt sensitivity 150, development rounds 80) are not in the repository; they are kept, in
the same layout, in Google Cloud Storage at `gs://ff-impact-forecast-data/genai/`.
