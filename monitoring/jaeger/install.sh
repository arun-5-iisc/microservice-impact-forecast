#!/bin/bash

set -e

helm repo add jaegertracing https://jaegertracing.github.io/helm-charts

helm repo update

helm upgrade --install jaeger \
    jaegertracing/jaeger \
    -n monitoring \
    --create-namespace \
    -f values.yaml