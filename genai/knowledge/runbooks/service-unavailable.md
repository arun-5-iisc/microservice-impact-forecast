# Service unavailable (no running replicas)

**Typical cause:** a change scales a deployment to 0 replicas, or its pods cannot start.

**Signs**
- Replicas drop to 0; events show ScalingReplicaSet "Scaled down ... to 0", Killing,
  SuccessfulDelete.
- The service's own traces stop (no requests reach it).
- Callers fail fast: their latency may DROP while their error rate rises, because requests
  end with an error instead of waiting.

**Response**
- Always High severity while it lasts: restore replicas (roll the change back) immediately.
