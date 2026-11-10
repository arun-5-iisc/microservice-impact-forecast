# Memory pressure (memory limit too low)

**Typical cause:** a change lowers a container's memory limit close to, or below, its working set.

**Signs**
- Memory usage sits close to the limit, or climbs toward it.
- When it crosses the limit the container is OOM-killed: a restart appears in the metrics,
  with Killing / BackOff / Started events, and a short outage while it restarts.
- After a restart, memory climbs again: restarts repeat (CrashLoopBackOff) if the limit is too low.
- Before the first kill, latency may be normal: the risk shows in memory, not in latency.

**Response**
- Restarts or crash loops: roll the change back, or raise the memory limit. Severity is High.
- Memory near the limit without restarts yet: expect an OOM kill; roll back or raise the limit.
- Memory well below the limit: monitor.
