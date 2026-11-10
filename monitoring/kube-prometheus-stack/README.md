# Prometheus

kube-prometheus-stack (chart 87.4.0), installed in the `monitoring` namespace by `install.sh` with the settings
in `values.yaml`. It runs Prometheus, Grafana, kube-state-metrics and node-exporter; Alertmanager and the default alerting rules are switched off.

Prometheus scrapes every 5 s. `scripts/prometheus_collector.py` reads each run's CPU, memory, replica and
restart metrics from it (15 s steps).

The Grafana admin password in `values.yaml` 

```bash
./monitoring/kube-prometheus-stack/install.sh
kubectl get pods -n monitoring
```
