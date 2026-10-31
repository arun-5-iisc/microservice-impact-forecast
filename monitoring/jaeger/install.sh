#!/bin/bash

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VERSION="4.11.1"   # Jaeger v2.19.0 all-in-one — version used for the original dataset.
                   # Newer releases (e.g. 4.16.1 / Jaeger 2.22) drop the legacy /api/traces
                   # endpoint that scripts/jaeger_collector.py relies on.

helm repo add jaegertracing https://jaegertracing.github.io/helm-charts

helm repo update

helm upgrade --install jaeger \
    jaegertracing/jaeger \
    --version "${VERSION}" \
    -n monitoring \
    --create-namespace \
    -f "${SCRIPT_DIR}/values.yaml"