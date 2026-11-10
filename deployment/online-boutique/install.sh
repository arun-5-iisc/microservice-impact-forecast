#!/usr/bin/env bash
# install.sh - install or upgrade Online Boutique v0.10.5 from the chart in helm-chart/ with the project's values.yaml, into the namespace set in config/project.yaml.
# Usage: ./deployment/online-boutique/install.sh [config file]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_FILE="${1:-${SCRIPT_DIR}/../../config/project.yaml}"

if [[ ! -f "${CONFIG_FILE}" ]]; then
	echo "Config file not found: ${CONFIG_FILE}" >&2
	exit 1
fi

yaml_value() {
	local key="$1"
	awk -F': *' -v key="${key}" '$1 == key {print $2; exit}' "${CONFIG_FILE}"
}

cluster_name="$(yaml_value cluster_name)"
namespace="$(yaml_value namespace)"

CHART_PATH="${SCRIPT_DIR}/helm-chart"
VALUES_FILE="${SCRIPT_DIR}/values.yaml"

if [[ -z "${namespace}" ]]; then
	echo "Config file must define namespace: ${CONFIG_FILE}" >&2
	exit 1
fi

if [[ ! -d "${CHART_PATH}" ]]; then
	echo "Helm chart not found: ${CHART_PATH}" >&2
	exit 1
fi

if [[ ! -f "${VALUES_FILE}" ]]; then
	echo "Values file not found: ${VALUES_FILE}" >&2
	exit 1
fi

if ! command -v helm >/dev/null 2>&1; then
	echo "helm is required to deploy this chart. Install helm and retry." >&2
	exit 1
fi

if ! command -v kubectl >/dev/null 2>&1; then
	echo "kubectl is required to deploy this chart." >&2
	exit 1
fi

echo "Using config: ${CONFIG_FILE}"
echo "Cluster: ${cluster_name}"
echo "Namespace: ${namespace}"
echo "Chart: ${CHART_PATH}"
echo "Values: ${VALUES_FILE}"

helm upgrade --install online-boutique \
	"${CHART_PATH}" \
	--namespace "${namespace}" \
	--create-namespace \
	--values "${VALUES_FILE}"

echo
echo "Online Boutique deployed successfully."