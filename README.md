# Proactive impact forecasting of microservice changes

Code and data for forecasting what a change to one microservice will do, using only the first 15 to 60 seconds of
telemetry after the change. Two things are forecast: the severity of the effect on the changed service (Low, Medium
or High), and the blast radius, i.e. how many other services get affected.

On top of the ML models there is an LLM agent that investigates a change using tools (over MCP) and summaries of past
incidents (RAG). It was tested on its own, with the ML forecast as extra input, and behind the ML model as a gate.

Everything runs on Google's [Online Boutique](https://github.com/GoogleCloudPlatform/microservices-demo) demo
(v0.10.5) on GKE. Prometheus and Jaeger collect the telemetry, and the changes are made with Helm, kubectl and
Chaos Mesh.

## What's where

- `deployment/`, `monitoring/` - install scripts for Online Boutique, Chaos Mesh, Jaeger and kube-prometheus-stack
  (versions are pinned in the scripts)
- `config/project.yaml` - the cluster the data was collected on
- `scripts/` - running experiments (`run_experiment.py`, `run_batch.py`) and the ML pipeline (`run_pipeline.sh`).
  `span_measures.py` does the trace measurements the labels are built from
- `scripts/analysis/` - label reliability and robustness checks (it has its own README)
- `scripts/report/` - makes the report's tables and figures from the result files
- `genai/` - the LLM agent: MCP server, RAG, replay of recorded runs, evaluation and live mode. Prompts are in
  `genai/prompts/`
- `experiments/plans/` - the experiment plan for each collection phase
- `dataset/processed/` - labels, features and the holdout list. The raw runs go in `dataset/raw/` (see [Data](#data))
- `results/`, `models/` - outputs and trained models

## Reproducing the results

With the recorded data (download it as described under [Data](#data)) you can rerun all the ML results, the label
analyses and the report tables on any machine with Python. The LLM evaluation needs API keys as well.

Collecting new data needs a GCP project with GKE. Timings differ between clusters, so new runs will be comparable to
these but not identical.

### Setup

```bash
python3 -m venv venv                              # Python 3.13.5 was used
venv/bin/pip install -r requirements.txt
venv/bin/pip install -r genai/requirements.txt    # only needed for the LLM part
```

Both requirements files pin their versions. Everything was run on a VM with 2 vCPUs and 8 GB of RAM; the scripts that
read the big trace files load one file at a time and cap their memory to stay within that.

### Cluster (only for new data)

You need gcloud, kubectl and helm 3. The data was collected on a GKE Standard cluster (Autopilot doesn't allow Chaos
Mesh) with a node pool `e2-pool` of 3 x e2-standard-2 regular VMs in us-central1-b. This gives the same setup:

```bash
gcloud container clusters create ff-2 --zone us-central1-b --num-nodes 1
gcloud container node-pools create e2-pool --cluster ff-2 --zone us-central1-b --machine-type e2-standard-2 --num-nodes 3
gcloud container node-pools delete default-pool --cluster ff-2 --zone us-central1-b
gcloud container clusters get-credentials ff-2 --zone us-central1-b

./monitoring/kube-prometheus-stack/install.sh     # Prometheus, kube-state-metrics (chart 87.4.0)
./monitoring/jaeger/install.sh                    # Jaeger all-in-one v2.19.0 (chart 4.11.1)
./deployment/chaos-mesh/install.sh                # Chaos Mesh 2.6.3
./deployment/online-boutique/install.sh           # Online Boutique v0.10.5 with the project's values.yaml
```

If you use a different namespace, change it in `config/project.yaml`.

`monitoring/kube-prometheus-stack/values.yaml` sets a Grafana admin password that is only meant for a lab cluster.
Change it if anyone else can reach yours.

### Running experiments

A run records 60 s of baseline, applies one change (CPU limit, memory limit, network latency, or replicas set to 0),
keeps it for 180 s, undoes it and records 60 s of recovery. Applying and undoing take a few seconds, so the recorded
phases end up about 186 s and 78 s long. Each run is saved in `dataset/raw/EXP_<n>/` as `metadata.json`,
`prometheus.csv`, `traces.json` and `events.json`.

```bash
python scripts/run_experiment.py --help                                   # a single run
python scripts/run_batch.py --plan experiments/plans/<plan>.csv --dry-run # a batch (resumable; --status, --limit N)
```

`run_batch.py` won't start a run unless the cluster is back in its baseline state. For long batches use a service
account, since a user login can expire halfway through.

Each collection phase has a plan file in `experiments/plans/` and a state file in `dataset/` that records which run
came from which plan row (`scripts/README.md` has the full table). Run IDs are given per attempt, so a missing number
such as EXP_0183 is an attempt that failed before it recorded anything.

### ML pipeline

```bash
./scripts/run_pipeline.sh --yes          # logs go to logs/pipeline/
```

In order, it runs: labels and early features (`preprocess.py`), the label threshold analysis, the holdout check,
feature analysis, feature engineering, the untuned models, tuning with Optuna, the final models, SHAP and the plots.

The holdout is the 50 runs in `dataset/processed/holdout_ids.txt`. It was drawn once with seed 42 before any model was
trained, and it has not been redrawn since. The draw is stratified by severity, so it depends on the labels used at
the time, the midterm labels, which are saved in `scripts/analysis/midterm_labels.csv`. To check it, redraw it and
compare:

```bash
venv/bin/python scripts/create_holdout.py --data scripts/analysis/midterm_labels.csv --check
```

### Label analyses

```bash
./scripts/analysis/run_analyses.sh       # writes to results/reliability/
```

This needs the pipeline outputs and the span cache, so run the pipeline and `./scripts/run_genai_eval.sh prepare`
first.

### LLM evaluation

API keys are read from `ANTHROPIC_API_KEY` or `~/.anthropic_key` for Claude, and from `OPENAI_API_KEY` or
`~/.openai_key` for OpenAI. They are never printed or written to the results.

```bash
./scripts/run_genai_eval.sh prepare                    # span cache, RAG index and development set
./scripts/run_genai_eval.sh retrieval                  # RAG retrieval evaluation
./scripts/run_genai_eval.sh dev 0 --yes                # development rounds 0 and 1 (prompts in genai/prompts/)
./scripts/run_genai_eval.sh dev 1 --yes
REPEATS=3 MODEL=claude-opus-5-5 ./scripts/run_genai_eval.sh holdout --yes      # same for claude-haiku-5-5, gpt-6-sol
MODEL=claude-opus-5-5 ./scripts/run_genai_eval.sh sensitivity --yes
./scripts/run_genai_eval.sh analyze                    # means over repeats, 95% CIs, paired comparisons
```

The prompt was tuned on training incidents only (see `genai/prompts/README.md`) and then frozen. The holdout was run
with the frozen setup, and how it would be analysed was decided before the first holdout run.

Every investigation saves its prompt, the full transcript, the tool calls and the agent's report in a folder next to
its `results.csv`. Only the `results.csv` files are in git; the folders are in the bucket (see
`results/genai/README.md`).

If a run stops because the API credit runs out, run the same command again and it carries on. LLM answers aren't
deterministic, so a rerun gives roughly the same scores but not the same answers. That's why the holdout was run
three times.

### Live mode with Slack (optional, needs a cluster)

The same system can watch a real cluster. When it sees a change, it screens it with the ML model, lets the agent
investigate if the forecast is Medium or High, and posts to Slack: the severity, the likely cause, the affected
services, what to do (commands and runbook steps) and the evidence. It only suggests; it never changes anything
itself.

```bash
# create a Slack app with an Incoming Webhook for a channel, then save its URL (it is never printed):
read -rsp "Slack webhook URL: " W; echo; printf '%s' "$W" > ~/.slack_webhook; chmod 600 ~/.slack_webhook; unset W

venv/bin/python -m genai.live simulate --run EXP_0284 --at 30 --model test --dry-run   # try it without a cluster or an LLM
venv/bin/python -m genai.live watch                       # terminal 1: watch the cluster
venv/bin/python -m genai.live inject --service productcatalogservice --fault latency --value 200 --hold 180   # terminal 2
```

`watch` uses gpt-6-sol by default, since it did best in the gated holdout evaluation, so it needs the OpenAI key. Use
`--model` to pick another one, and `--dry-run` to print the Slack message instead of sending it. Each incident is
saved in `results/genai/live/<incident id>/` (features, forecast, transcript, report and Slack payload), with its
telemetry snapshot in `results/genai/live/snapshots/`. None of this is in git.

### Tables and figures

```bash
venv/bin/python scripts/report/make_tables.py    # tables and numbers -> report/tables/
venv/bin/python scripts/report/make_figures.py   # data figures -> report/figures/
venv/bin/python scripts/report/make_diagrams.py  # system and pipeline diagrams -> report/figures/
```

## Data

The raw runs are about 20 GB (mostly `traces.json`), so they're not in git. The 244 runs of the final dataset are in a
public bucket. Download them into `dataset/raw/`:

```bash
gcloud storage cp -r gs://ff-impact-forecast-data/raw/* dataset/raw/
```

The LLM transcripts are in the same bucket under `genai/`. The span cache (`dataset/spans/`, about 40 MB) isn't in git
either; `./scripts/run_genai_eval.sh prepare` rebuilds it from the raw data.

adservice, cartservice and shippingservice don't export traces in Online Boutique v0.10.5, so their spans are missing
from `traces.json`. They are measured from the spans of the services that call them (`scripts/span_measures.py`).

In git: the code, the plans and their state files, `dataset/processed/`, all the result files (for the LLM runs only
`results.csv`, which has every answer, its evidence and its scores), and the final models in `models/fe_tuned/` and
`models/rag_index/`, so the agent and live mode work without retraining. The other models are rebuilt by the
pipeline.

Not in git: the raw runs, the span cache, the live-mode records, the analysis caches and the logs.
