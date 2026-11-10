# Network latency (slow network path to a service)

**Typical cause:** a change adds delay on the service's network path (proxy, sidecar or
network policy misconfiguration, cross-zone routing).

**Signs**
- Latency of the service jumps by roughly a fixed amount per request (e.g. +100 ms or more),
  while its CPU and memory stay normal or even drop (it receives fewer requests per second).
- Callers' latency rises too, on the requests that go through it.
- Large delays can make health checks time out: readiness probe failures (Unhealthy events),
  restarts or 0 ready replicas follow.

**Response**
- Latency more than double the baseline, or health checks failing: roll the change back. High.
- Smaller, stable increase (20–100 %): Medium; roll back if callers are affected.
