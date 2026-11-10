# Jaeger

Jaeger all-in-one (chart 4.11.1, Jaeger v2.19.0) with in-memory storage, installed in the `monitoring`
namespace by `install.sh` with the settings in `values.yaml`. Online Boutique's services send their spans to it
through the OpenTelemetry collector.

The version is pinned on purpose: newer releases (for example chart 4.16.1 / Jaeger 2.22) drop the legacy
`/api/traces` endpoint that `scripts/jaeger_collector.py` reads.

Traces are kept in memory only, it gets lost when pod restarts or the cluster is scaled down. 
so `scripts/run_experiment.py` saves each run's traces to `dataset/raw/` as soon as the run ends.

```bash
./monitoring/jaeger/install.sh
kubectl get pods -n monitoring
```
