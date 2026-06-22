# Prometheus

This directory contains the configuration used to deploy the monitoring stack for the dissertation experiments.
```
kube-prometheus-stack
│
├── Prometheus
├── Grafana
├── kube-state-metrics
├── node-exporter
├── Alertmanager
└── CRDs
```
The deployment uses:

- kube-prometheus-stack
- Helm

The monitoring stack is used as the primary source of Kubernetes and application metrics during experiment execution.

## Installation

```bash
./install.sh
```

## Verification

```bash
kubectl get pods -n monitoring
```