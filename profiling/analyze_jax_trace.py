#!/usr/bin/env python
"""Aggregate device-side kernel time from a jax.profiler Chrome trace.

Reads the ``*.trace.json.gz`` written by ``jax.profiler.trace`` and sums
duration by event name on GPU device tracks (CUPTI sees kernels inside CUDA
graphs, unlike a default nsys capture). Optionally maps kernel/HLO names to
pySES functions via the optimized-HLO metadata using analyze_hlo_hotspots.
"""
import argparse
import gzip
import json
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze_hlo_hotspots import parse_hlo, attribute, group_key, busy_time  # noqa: E402


def main():
  p = argparse.ArgumentParser()
  p.add_argument("--trace", required=True, help="*.trace.json.gz path")
  p.add_argument("--hlo", default=None)
  p.add_argument("--steps", type=int, default=3)
  p.add_argument("--top", type=int, default=50)
  p.add_argument("--out", default=None)
  args = p.parse_args()

  with gzip.open(args.trace, "rt") as f:
    data = json.load(f)
  events = data["traceEvents"] if isinstance(data, dict) else data

  # Identify device-track pids from process_name metadata.
  pid_names = {}
  for ev in events:
    if ev.get("ph") == "M" and ev.get("name") == "process_name":
      pid_names[ev["pid"]] = ev.get("args", {}).get("name", "")
  gpu_pids = {pid for pid, name in pid_names.items()
              if "GPU" in name or "gpu" in name}
  print("device tracks:", {pid: pid_names[pid] for pid in gpu_pids})

  by_name = defaultdict(lambda: [0.0, 0])
  intervals = []
  for ev in events:
    if ev.get("ph") != "X" or ev.get("pid") not in gpu_pids:
      continue
    dur = ev.get("dur", 0)  # microseconds
    name = ev.get("name", "?")
    by_name[name][0] += dur
    by_name[name][1] += 1
    intervals.append((ev["ts"], ev["ts"] + dur))

  if not intervals:
    print("no device events found")
    return
  total_us = sum(v[0] for v in by_name.values())
  n_events = sum(v[1] for v in by_name.values())
  wall_us = max(e for _, e in intervals) - min(s for s, _ in intervals)
  busy_us = busy_time([(s, e) for s, e in intervals])

  hlo_map = parse_hlo(args.hlo) if args.hlo and os.path.exists(args.hlo) else {}
  by_group = defaultdict(lambda: [0.0, 0])
  rows = []
  for name, (tot, cnt) in sorted(by_name.items(), key=lambda kv: -kv[1][0]):
    op_name, src = attribute(name.replace(".", "_"), hlo_map)
    rows.append({"name": name, "total_ms": tot / 1e3, "count": cnt,
                 "mean_us": tot / cnt, "pct": 100.0 * tot / total_us,
                 "op_name": op_name, "src": src})
    by_group[group_key(op_name)][0] += tot
    by_group[group_key(op_name)][1] += cnt

  report = {
      "trace_wall_ms": wall_us / 1e3,
      "device_busy_ms": busy_us / 1e3,
      "device_occupancy_pct": 100.0 * busy_us / wall_us,
      "device_event_time_ms": total_us / 1e3,
      "num_device_events": n_events,
      "events_per_step": n_events / args.steps,
      "mean_event_us": total_us / n_events,
      "by_function": sorted(
          [{"function": g, "total_ms": t / 1e3, "count": c,
            "pct": 100.0 * t / total_us} for g, (t, c) in by_group.items()],
          key=lambda r: -r["total_ms"]),
      "top_events": rows[:args.top],
  }
  print("=== device summary (trace) ===")
  for k in ("trace_wall_ms", "device_busy_ms", "device_occupancy_pct",
            "device_event_time_ms", "num_device_events", "events_per_step",
            "mean_event_us"):
    print(f"  {k}: {report[k]:.2f}")
  print("\n=== device time by pySES function ===")
  for r in report["by_function"]:
    print(f"  {r['function']:44s} {r['total_ms']:9.2f} ms  {r['pct']:5.1f}%  ({r['count']})")
  print(f"\n=== top {args.top} device events ===")
  for r in report["top_events"]:
    src = f"  [{r['src']}]" if r["src"] else ""
    op = f"  {r['op_name']}" if r["op_name"] else ""
    print(f"  {r['total_ms']:8.2f} ms {r['pct']:5.1f}% x{r['count']:5d} "
          f"@{r['mean_us']:8.1f}us  {r['name'][:60]}{op[:130]}{src}")
  if args.out:
    with open(args.out, "w") as f:
      json.dump(report, f, indent=2)
    print(f"\nreport written to {args.out}")


if __name__ == "__main__":
  main()
