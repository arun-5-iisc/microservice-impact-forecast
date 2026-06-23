# Running Experiments

## Prerequisites

- GKE Standard cluster with Online Boutique deployed
- `kubectl` configured and pointing at the cluster
- `helm` 3.x installed
- Chaos Mesh installed (see [Installation](#chaos-mesh-installation))
- Python 3.9+ with dependencies installed
- GCS bucket `ff-dataset-bucket` with write access

---

## Chaos Mesh Installation

One-time setup on the GKE cluster. Required for latency fault injection across all services.

```bash
chmod +x deployment/chaos-mesh/install.sh
./deployment/chaos-mesh/install.sh
```

Verify it works before running experiments:

```bash
kubectl apply -f deployment/chaos-mesh/test-latency.yaml
# Check Jaeger UI — checkoutservice traces should show ~200ms latency
kubectl delete -f deployment/chaos-mesh/test-latency.yaml
```

---

## Running the Batch

The batch runner executes all 90 experiments sequentially and resumes safely if interrupted.

```bash
# 1. Install Python dependencies
pip install -r requirements.txt

# 2. Clear any stale chaos resources from testing
kubectl delete networkchaos --all -n online-boutique --ignore-not-found

# 3. Preview the full experiment plan
python scripts/run_batch.py --dry-run

# 4. Check initial status
python scripts/run_batch.py --status

# 5. Start the batch (~9 hours total)
python scripts/run_batch.py
```

Run inside `tmux` so the batch survives SSH disconnects:

```bash
tmux new -s experiments
python scripts/run_batch.py

# Detach:  Ctrl+B then D
# Reattach: tmux attach -t experiments
```

---

## Monitoring Progress

```bash
# Experiment counts (done / pending / failed)
python scripts/run_batch.py --status

# Watch GCS bucket fill up
gcloud storage ls gs://ff-dataset-bucket/raw/

# Watch active chaos injection during a latency experiment
kubectl get networkchaos -n online-boutique
```

---

## Resuming After Failure

The batch writes state to `dataset/batch_state.json` after every experiment.
If the process is killed or the VM reboots, just re-run — completed experiments are skipped automatically.

```bash
python scripts/run_batch.py             # resumes from where it stopped

python scripts/run_batch.py --reset-failed   # re-queues failed experiments, then runs
```

---

## Experiment Matrix (90 experiments × 4 snapshots = ~360 training rows)

| Fault type            | Services                          | Values                          | Traffic       | Count |
|-----------------------|-----------------------------------|---------------------------------|---------------|-------|
| CPU (limit only)      | 8 services (100m request)         | 150m, 125m, 100m                | normal        | 24    |
| CPU (limit only)      | adservice, cartservice (200m req) | 275m, 250m, 200m                | normal        | 6     |
| Memory (limit only)   | 6 services                        | Low / Med / High                | normal        | 18    |
| Replicas              | 8 services                        | 0 (complete unavailability)     | normal        | 8     |
| Latency (Chaos Mesh)  | 5 critical-path services          | 100ms, 200ms, 500ms, 1000ms     | normal        | 20    |
| Latency (Chaos Mesh)  | checkoutservice, frontend, productcatalogservice | 500ms, 1000ms   | high          | 6     |
| CPU                   | 5 selected services               | 125m or 100m                    | high          | 5     |
| Replicas              | 3 selected services               | 0                               | high          | 3     |
| **Total**             |                                   |                                 |               | **90** |

---

## After All Experiments Complete

Build the ML training dataset:

```bash
python scripts/preprocess.py
# Reads:  dataset/raw/EXP_*/
# Writes: dataset/processed/training_dataset.csv

# Verify
python -c "
import pandas as pd
df = pd.read_csv('dataset/processed/training_dataset.csv')
print(f'Total rows: {len(df)}')
print(df['severity'].value_counts())
print(df.groupby('fault_type').size())
"
```
