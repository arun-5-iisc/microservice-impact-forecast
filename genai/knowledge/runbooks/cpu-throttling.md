# CPU throttling (CPU limit too low)

**Typical cause:** a change lowers a container's CPU limit below what it needs under load.

**Signs**
- CPU usage flattens at, or just under, the new limit instead of following traffic.
- Latency of the service rises, more under high traffic; callers slow down if they wait on it.
- Usually no restarts and no OOM events: throttling slows a process but does not kill it.
- At low traffic the effect can be invisible.

**Response**
- Clear slowdown of the service or its callers: roll the change back, or restore the limit.
- Small, stable effect: monitor; re-check under higher traffic.
