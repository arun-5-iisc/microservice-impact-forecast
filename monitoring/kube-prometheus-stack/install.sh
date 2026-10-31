#!/bin/bash

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VERSION="87.4.0"   # version used for the original dataset; pin for reproducibility

echo "Adding Prometheus Helm repository..."

helm repo add prometheus-community https://prometheus-community.github.io/helm-charts

helm repo update

echo "Installing kube-prometheus-stack..."

helm upgrade --install prometheus \
    prometheus-community/kube-prometheus-stack \
    --version "${VERSION}" \
    --namespace monitoring \
    --create-namespace \
    -f "${SCRIPT_DIR}/values.yaml"

echo ""
echo "Prometheus installation complete."