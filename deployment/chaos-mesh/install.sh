#!/usr/bin/env bash
# install.sh — Install Chaos Mesh on a GKE Standard cluster
#
# Requirements:
#   - kubectl configured and pointing at your GKE cluster
#   - helm 3.x installed
#   - GKE Standard cluster (NOT Autopilot — Autopilot blocks privileged DaemonSets)
#
# Usage:
#   chmod +x deployment/chaos-mesh/install.sh
#   ./deployment/chaos-mesh/install.sh
#
# To uninstall:
#   helm uninstall chaos-mesh -n chaos-mesh
#   kubectl delete ns chaos-mesh
#   kubectl delete crd $(kubectl get crd | grep chaos-mesh | awk '{print $1}')

set -euo pipefail

NAMESPACE="chaos-mesh"
RELEASE="chaos-mesh"
VERSION="2.6.3"      # pin for reproducibility; check https://github.com/chaos-mesh/chaos-mesh/releases

# Colours for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

info()  { echo -e "${GREEN}[INFO]${NC}  $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC}  $*"; }
error() { echo -e "${RED}[ERROR]${NC} $*" >&2; exit 1; }

# ─── Preflight checks ────────────────────────────────────────────────────────

info "Checking prerequisites..."

command -v kubectl >/dev/null 2>&1 || error "kubectl not found. Install it first."
command -v helm    >/dev/null 2>&1 || error "helm not found. Install it first."

# Verify cluster access
kubectl cluster-info --request-timeout=10s >/dev/null 2>&1 \
  || error "Cannot reach the Kubernetes cluster. Check your kubeconfig."

CLUSTER=$(kubectl config current-context)
info "Target cluster: ${CLUSTER}"

# Warn if this looks like an Autopilot cluster
if kubectl get nodes -o jsonpath='{.items[0].metadata.labels}' 2>/dev/null \
     | grep -q "cloud.google.com/gke-autopilot"; then
  error "This appears to be a GKE Autopilot cluster. Chaos Mesh requires a Standard cluster (privileged DaemonSets are not allowed on Autopilot)."
fi

# ─── Skip if already installed ───────────────────────────────────────────────

if helm status "${RELEASE}" -n "${NAMESPACE}" >/dev/null 2>&1; then
  INSTALLED_VERSION=$(helm list -n "${NAMESPACE}" -o json \
    | python3 -c "import json,sys; print(next(r['chart'] for r in json.load(sys.stdin) if r['name']=='${RELEASE}'))")
  warn "Chaos Mesh is already installed (${INSTALLED_VERSION}). Skipping."
  warn "To upgrade: helm upgrade ${RELEASE} chaos-mesh/chaos-mesh -n ${NAMESPACE} --version ${VERSION}"
  exit 0
fi

# ─── Add Helm repo ───────────────────────────────────────────────────────────

info "Adding chaos-mesh Helm repo..."
helm repo add chaos-mesh https://charts.chaos-mesh.org 2>/dev/null || true
helm repo update chaos-mesh

# ─── Install ─────────────────────────────────────────────────────────────────

info "Installing Chaos Mesh ${VERSION} in namespace '${NAMESPACE}'..."

helm install "${RELEASE}" chaos-mesh/chaos-mesh \
  --namespace "${NAMESPACE}" \
  --create-namespace \
  --version "${VERSION}" \
  --set chaosDaemon.runtime=containerd \
  --set chaosDaemon.socketPath=/run/containerd/containerd.sock \
  --set dashboard.securityMode=false \
  --wait \
  --timeout 5m

# ─── Verify ──────────────────────────────────────────────────────────────────

info "Waiting for all Chaos Mesh pods to be Running..."
kubectl wait --for=condition=Ready pod \
  -l app.kubernetes.io/instance=chaos-mesh \
  -n "${NAMESPACE}" \
  --timeout=120s

info "Chaos Mesh pods:"
kubectl get pods -n "${NAMESPACE}"

echo ""
info "Verifying CRDs are registered..."
EXPECTED_CRDS="networkchaos podchaos iochaos timechaos"
for crd in ${EXPECTED_CRDS}; do
  if kubectl get crd "${crd}.chaos-mesh.org" >/dev/null 2>&1; then
    echo "  ✓ ${crd}.chaos-mesh.org"
  else
    warn "  ✗ ${crd}.chaos-mesh.org not found"
  fi
done

echo ""
info "Installation complete."
info "Chaos Mesh is ready to inject network latency into Online Boutique services."
echo ""
echo "  Run a quick test:"
echo "    kubectl apply -f deployment/chaos-mesh/test-latency.yaml"
echo "    kubectl delete -f deployment/chaos-mesh/test-latency.yaml"
