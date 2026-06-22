#!/bin/bash

set -e

echo "Adding Prometheus Helm repository..."

helm repo add prometheus-community https://prometheus-community.github.io/helm-charts

helm repo update

echo "Installing kube-prometheus-stack..."

helm upgrade --install prometheus \
    prometheus-community/kube-prometheus-stack \
    --namespace monitoring \
    --create-namespace \
    -f values.yaml

echo ""
echo "Prometheus installation complete."