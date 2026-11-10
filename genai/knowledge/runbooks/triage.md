# Triage after a change

Use this first when a change has just been applied to a service.

1. **Is the changed service up?** Check replicas and restarts in its metrics, and Kubernetes
   events (Killing, BackOff, Unhealthy / readiness probe failures, OOMKilled).
2. **Is it slower or failing?** Compare its latency and error rate with the baseline. For
   services with little traffic, a handful of requests is weak evidence: look at the raw values.
3. **Is it spreading?** Services that call the changed one (see the service map) slow down or
   fail when it does. Check the callers before unrelated services.
4. **How bad will it get?** Early signs often grow: memory creeping toward a limit ends in
   an OOM restart; CPU pinned at a low limit keeps requests queued; a service with 0 replicas
   stays down until someone acts.
5. **Decide.** If the change clearly harms the service or its callers, roll it back. If the
   effect is small and stable, monitor. If nothing changed, no action is needed.

Severity guide (final state of the changed service):
- **High**: unavailable (0 replicas), crashing (restarts), errors above ~5%, or latency more
  than double the baseline.
- **Medium**: latency up more than ~20%, or errors above ~1%.
- **Low**: no meaningful change.
