# Proactive impact forecasting of microservice changes

When a change is applied to one service of a microservice application, this project forecasts, from only the first
15–60 seconds of telemetry:
- the **severity** of its effect on the changed service (Low / Medium / High);
- its **blast radius**: how many other services are affected.

It also evaluates LLM agents that investigate the change through tools (MCP) and retrieval (RAG), alone and combined
with the ML forecast.

The test bed is Google's [Online Boutique](https://github.com/GoogleCloudPlatform/microservices-demo) (v0.10.5) on
GKE. Prometheus and Jaeger record the telemetry, and changes are injected with Helm, `kubectl` and Chaos Mesh.

## Repository layout

| Path | Contents |
|---|---|
| `deployment/`, `monitoring/` | Cluster software: Online Boutique Helm chart and values, Chaos Mesh, Jaeger, kube-prometheus-stack (versions pinned in the install scripts) |
| `config/project.yaml` | The cluster the data was collected on (namespace, zone, node count) |
| `scripts/` | Experiment harness (`run_experiment.py`, `run_batch.py`), ML pipeline (`run_pipeline.sh` and its stages), shared measurement (`span_measures.py`) |
| `scripts/analysis/` | Label-reliability and robustness analyses (`run_analyses.sh`, see its README) |
| `genai/` | LLM incident agent: replay of recorded runs, MCP server, RAG, agent, evaluation, live mode; prompt versions in `prompts/` |
| `experiments/plans/` | The experiment plans (CSV) of the four collection phases; see `scripts/README.md` |
| `dataset/` | `processed/`: labels, features and the fixed holdout. The plan state files (which run each plan row produced). The raw runs go in `raw/` (see [Data](#data)). |
| `results/`, `models/` | Pipeline, analysis and LLM-evaluation outputs; trained models |
| `scripts/report/` | Generate the dissertation's tables, numbers, figures and diagrams from the result files |

## What can be reproduced, and how

| Level | Needs | Reproduces |
|---|---|---|
| **A. From the recorded data** | This repository, the raw data (see [Data](#data)), Python | All ML results, the label analyses and the report tables. The LLM evaluation additionally needs an API key. |
| **B. Collecting new data** | A GCP project with GKE | New experiment runs. Timings vary between clusters, so the runs are comparable but not identical. |

### 0. Setup

```bash
python3 -m venv venv                                     # Python 3.13.5 was used
venv/bin/pip install -r requirements.txt                 # ML pipeline (versions pinned)
venv/bin/pip install -r genai/requirements.txt           # LLM evaluation (versions pinned)
```

The results were produced on a VM with 2 vCPUs and 8 GB of RAM. Scripts that read the large raw trace files cap their
memory and load one file at a time.

### 1. Cluster and software (level B only)

Tools: `gcloud`, `kubectl`, `helm` 3. `config/project.yaml` records the cluster the data was collected on; set the
namespace there if you use a different one. The final dataset was collected on a **GKE Standard** cluster (not
Autopilot, which blocks Chaos Mesh) with a node pool `e2-pool` of **3 x e2-standard-2** regular VMs in
`us-central1-b`. An equivalent setup:

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

### 2. Experiments (level B only)

Each run records a 60 s baseline, applies one change (CPU limit, memory limit, network latency or replicas to 0) and
holds it for 180 s, then undoes it and records 60 s of recovery (applying and undoing take a few seconds, so the
recorded phases are about 186 s and 78 s). It saves `metadata.json`, `prometheus.csv`, `traces.json` and
`events.json` to `dataset/raw/EXP_<n>/`.

```bash
python scripts/run_experiment.py --help                                   # one run
python scripts/run_batch.py --plan experiments/plans/<plan>.csv --dry-run # a batch (resumable; --status, --limit N)
```

`run_batch.py` checks before every run that the cluster is in its baseline state. Long batches should run under a
service account, because user logins can expire mid-batch.
- **Every phase of the data collection is a plan file** in `experiments/plans/`, with a state file in `dataset/`
  recording which run each row produced (`scripts/README.md` has the table).
- **Run IDs are numbered per attempt,** so a gap in the sequence (e.g. EXP_0183) is an attempt that failed before
  recording any data.
- **Grafana:** `monitoring/kube-prometheus-stack/values.yaml` sets a lab-only Grafana admin password. Change it on
  any cluster that others can reach.

### 3. ML pipeline (level A)

```bash
./scripts/run_pipeline.sh --yes          # stages logged to logs/pipeline/
```

The stages are:
1. labels and early features (`preprocess.py`);
2. label-threshold analysis;
3. the fixed holdout check;
4. base feature analysis;
5. feature engineering;
6. untuned models;
7. tuning (Optuna);
8. the final model stage;
9. SHAP explanations;
10. plots.

The 50-run holdout is fixed in `dataset/processed/holdout_ids.txt`. It was drawn once (seed 42) before any model was
trained, and it is reused, not redrawn. The draw is stratified by severity, so it depends on the labels
it was drawn with: the midterm labels, frozen in `scripts/analysis/midterm_labels.csv`. This command redraws it and
confirms it is identical:

```bash
venv/bin/python scripts/create_holdout.py --data scripts/analysis/midterm_labels.csv --check
```

### 4. Label analyses (level A)

```bash
./scripts/analysis/run_analyses.sh       # -> results/reliability/ (needs the pipeline outputs and the span cache)
```

### 5. LLM evaluation (level A, needs an API key)

Keys are read from `ANTHROPIC_API_KEY` or `~/.anthropic_key` (Claude models), and from `OPENAI_API_KEY` or
`~/.openai_key` (OpenAI models). They are never printed or saved in the results.

```bash
./scripts/run_genai_eval.sh prepare                    # span cache, RAG index and development set
./scripts/run_genai_eval.sh retrieval                  # RAG retrieval evaluation
./scripts/run_genai_eval.sh dev 0 --yes                # development rounds 0 and 1 (prompts in genai/prompts/)
./scripts/run_genai_eval.sh dev 1 --yes
REPEATS=3 MODEL=claude-opus-5-5 ./scripts/run_genai_eval.sh holdout --yes      # likewise claude-haiku-5-5, gpt-6-sol
MODEL=claude-opus-5-5 ./scripts/run_genai_eval.sh sensitivity --yes
./scripts/run_genai_eval.sh analyze                    # summary: means over repeats, 95% CIs, paired comparisons
```

- **Protocol:** the prompt is refined on training incidents only (`genai/prompts/README.md`), then frozen; the holdout
  is run with the frozen configuration, and its analysis plan was fixed before any holdout run. The report describes
  each decision.
- **Saved evidence:** every investigation saves its prompt, full transcript, tool calls and report in a folder next
  to its `results.csv`. In this repository only the `results.csv` files are included; the folders are kept in Google
  Cloud Storage (`results/genai/README.md`).
- **Pausing:** a run that runs out of API credit stops cleanly; re-running the same command continues it.
- **Variation:** LLM answers vary from run to run, so re-runs reproduce the scores approximately, not answer for
  answer. The holdout repeats measure this variation.

### 5b. Live mode with Slack alerts (level B, optional)

The evaluated system can also watch a real cluster. It detects a change, screens it with the ML model, has the LLM
agent investigate, and posts the result to Slack: the severity, root cause, affected services, the recommended
action with commands and runbook steps, and the evidence. It only recommends; nothing is changed automatically.

```bash
# Slack: create an app with an Incoming Webhook for a channel, then save its URL (never printed):
read -rsp "Slack webhook URL: " W; echo; printf '%s' "$W" > ~/.slack_webhook; chmod 600 ~/.slack_webhook; unset W

venv/bin/python -m genai.live simulate --run EXP_0284 --at 30 --model test --dry-run   # test without a cluster or an LLM
venv/bin/python -m genai.live watch                       # terminal 1: watch the cluster, alert on Medium/High
venv/bin/python -m genai.live inject --service productcatalogservice --fault latency --value 200 --hold 180   # terminal 2
```

`watch` uses gpt-6-sol by default, the most accurate agent in the gated holdout evaluation (`--model` to change it;
it needs the OpenAI key, see step 5). Each incident is saved in
`results/genai/live/<incident id>/` (features, forecast, transcript, report and Slack payload), and its telemetry
snapshot in `results/genai/live/snapshots/`; these records are not in git. Add `--dry-run` to print the Slack messages
instead of sending them.

### 6. Report

```bash
export REPORT_DIR=<your LaTeX project>        # optional; without it the output goes to report/
venv/bin/python scripts/report/make_tables.py    # tables and numbers from results/ → $REPORT_DIR/tables/
venv/bin/python scripts/report/make_figures.py   # data figures from results/ → $REPORT_DIR/figures/
venv/bin/python scripts/report/make_diagrams.py  # system and pipeline diagrams → $REPORT_DIR/figures/
```

The dissertation's LaTeX sources are not in this repository. The scripts write into the folder given by
`REPORT_DIR` (for example a clone of the Overleaf project), or into a local `report/` folder if it is not set.

## Data

- **Final dataset:** 244 runs (1 from phase 1, 35 from phase 2 and 208 from phase 3), all on 3 x e2-standard-2.
- **Superseded:** 89 earlier runs on 2 x e2-standard-2 did not have enough CPU (pods could not be scheduled or were
  evicted), so the same settings were re-run on the final hardware (`experiments/plans/phase3_rerun.csv`).
- **Excluded:** 18 runs were set aside: 15 with data problems (traces lost after a Jaeger restart or at the trace
  download limit, a cluster not back in its baseline state, or timing that deviated from the protocol), and 3 valid
  replication runs on different hardware (3 x n1-standard-4). One interrupted run with no data (EXP_0203) was also
  set aside. The settings with data problems were re-run (`experiments/plans/phase3_targeted.csv`).
- **Tracing coverage:** adservice, cartservice and shippingservice export no traces in Online Boutique v0.10.5; they
  are measured from their callers' spans (`scripts/span_measures.py`).

The raw data is about 20 GB, mostly `traces.json`, and is **not stored in git**. The 244 runs of the final dataset
are in Google Cloud Storage at `gs://ff-impact-forecast-data/raw/`; download them into `dataset/raw/` with

```bash
gcloud storage cp -r gs://ff-impact-forecast-data/raw/* dataset/raw/
```

The bucket also holds the LLM transcripts under `genai/` (`results/genai/README.md`). The superseded and excluded
runs are not shared.
The derived span cache (`dataset/spans/`, about 40 MB) is rebuilt from the raw data with
`./scripts/run_genai_eval.sh prepare`.

**What is in git and what is not** (see `.gitignore`):
- **In git:**
  - the code, the experiment plans with their state files, and `dataset/processed/` (labels, features, the fixed
    holdout);
  - `results/`: the ML, analysis and LLM result files. For the LLM runs that means each run's `results.csv` (every
    answer, its evidence text and scores); the per-investigation transcripts are in GCS (`results/genai/README.md`);
  - `models/fe_tuned/` (the final severity and blast-radius models) and `models/rag_index/` (the RAG index), so
    the LLM evaluation and live mode run without retraining; the other model files are rebuilt by the pipeline;
  - the scripts that generate the report's tables and figures (`scripts/report/`).
- **Not in git:**
  - the dissertation's LaTeX sources (`report/`);
  - the raw runs (`dataset/raw/`) and the superseded and excluded runs;
  - the span cache and the live-mode records (`results/genai/live/`);
  - the caches of the label analyses, and `logs/`.
