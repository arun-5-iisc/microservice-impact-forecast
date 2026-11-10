| Script | What it does |
|---|---|
| `run_batch.py` | Runs all experiments in a plan file from `experiments/plans/`. Before each run it checks the credentials and that the cluster is back to normal. If it stops, it can be resumed. |
| `run_experiment.py` | Runs a single experiment: 60 s baseline, then the change for 180 s, then 60 s recovery. Output goes to `dataset/raw/EXP_<n>/`. |
| `deployment_manager.py` | Applies the change and reverts it. CPU and memory limits are changed with Helm, replicas with `kubectl scale`, and network delay with Chaos Mesh (NetworkChaos). |
| `traffic_manager.py` | Switches the load generator between normal (10 users) and high (25 users) traffic. |
| `prometheus_collector.py`, `jaeger_collector.py`, `kubernetes_collector.py` | Collect metrics, traces and Kubernetes events for a run. |

```bash
python scripts/run_batch.py --plan experiments/plans/phase3_targeted.csv --dry-run   
python scripts/run_batch.py --plan experiments/plans/phase3_targeted.csv             
python scripts/run_batch.py --plan experiments/plans/phase3_targeted.csv --status   
```

After every experiment the progress is written to `dataset/<plan name>_state.json`, so running
the same command again picks up from where it stopped. Use `--reset-failed` to retry failed
experiments and `--limit N` to stop after N experiments.

The state files in the repo show how the dataset was collected: which run came from which plan row,
and which attempts were dropped and why. That is why every plan shows as finished. To collect a plan
again on a new cluster, move its state file somewhere else first and `run_batch.py` will start a new one.

### Plans and runs

Each row in a plan is one setting (service, change, value, traffic) along with the reason for
picking it. The state files store the run ID created for each row.

## ML pipeline

`run_pipeline.sh` runs the scripts below in order. The stages are listed at the top of the script.

| Script | What it does |
|---|---|
| `span_measures.py` | Measures latency and error rate from the traces and decides if a service is affected. All other scripts use this. |
| `preprocess.py` | Creates the labels (severity, blast radius) and the early features at 15/30/45/60 s and saves them in `dataset/processed/`. |
| `threshold_analysis.py` | Checks the label thresholds and analyses the features. |
| `create_holdout.py` | Picks the fixed holdout of 50 runs. `--check` picks it again and compares. |
| `feature_engineer.py` | Adds topology, ratio, log and system features. |
| `train_model.py`, `tune_model.py` | Trains the untuned models, then tunes them with Optuna. Also computes the baselines and the holdout scores. |
| `explain_model.py` | SHAP and permutation importance. |
| `visualize.py`, `plot_labels.py` | Plots. |

```bash
./scripts/run_pipeline.sh --yes
```

