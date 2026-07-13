#!/usr/bin/env python
"""Join nsys kernel timings with XLA HLO metadata to attribute GPU time to source.

Inputs:
  --sqlite  nsys export (``nsys export --type sqlite`` or ``nsys stats`` side
            effect) of a capture that contains only steady-state coupling steps.
  --hlo     optimized HLO text of the coupling step saved by
            ``profile_baro_wave.py --mode cost`` (same compiled program, via the
            shared JAX compilation cache).
  --steps   number of coupling steps inside the capture window (divides counts).

Outputs a human-readable hotspot report and a JSON with:
  * device busy/idle within the capture (kernel-level occupancy),
  * kernels per step, duration histogram,
  * top kernels by total time, mapped to HLO op_name (jax name stack) and
    source file/line,
  * total time aggregated by pySES function (second jit(...) level of op_name).
"""
import argparse
import json
import os
import re
import sqlite3
from collections import defaultdict


def load_kernels(sqlite_path):
  con = sqlite3.connect(sqlite_path)
  cur = con.cursor()
  rows = []
  for table, kind in (("CUPTI_ACTIVITY_KIND_KERNEL", "kernel"),):
    try:
      cur.execute(
          f"SELECT k.start, k.end, s.value FROM {table} k "
          f"JOIN StringIds s ON k.shortName = s.id")
      rows += [(st, en, name, kind) for st, en, name in cur.fetchall()]
    except sqlite3.OperationalError as e:
      print(f"  ({table}: {e})")
  memcpy = []
  for table in ("CUPTI_ACTIVITY_KIND_MEMCPY", "CUPTI_ACTIVITY_KIND_MEMSET"):
    try:
      cur.execute(f"SELECT start, end FROM {table}")
      memcpy += [(st, en, table.rsplit('_', 1)[-1].lower(), "mem")
                 for st, en in cur.fetchall()]
    except sqlite3.OperationalError:
      pass
  con.close()
  return rows, memcpy


def busy_time(intervals):
  if not intervals:
    return 0
  ivs = sorted((s, e) for s, e, *_ in intervals)
  total = 0
  cur_s, cur_e = ivs[0]
  for s, e in ivs[1:]:
    if s > cur_e:
      total += cur_e - cur_s
      cur_s, cur_e = s, e
    else:
      cur_e = max(cur_e, e)
  total += cur_e - cur_s
  return total


_META = re.compile(r'metadata=\{([^}]*)\}')
_OPNAME = re.compile(r'op_name="([^"]*)"')
_SRC = re.compile(r'source_file="([^"]*)"')
_LINE = re.compile(r'source_line=(\d+)')


def parse_hlo(hlo_path):
  """Map normalized instruction name -> (op_name, source_file:line).

  XLA GPU kernel names are the HLO instruction names with '.' -> '_'
  (e.g. %loop_add_fusion.7 -> loop_add_fusion_7).
  """
  mapping = {}
  with open(hlo_path) as f:
    for line in f:
      line = line.strip()
      m = re.match(r'(?:ROOT )?%?([\w.\-]+) = ', line)
      if not m:
        continue
      meta = _META.search(line)
      if not meta:
        continue
      body = meta.group(1)
      op = _OPNAME.search(body)
      src = _SRC.search(body)
      ln = _LINE.search(body)
      norm = m.group(1).replace(".", "_")
      src_str = None
      if src:
        src_str = src.group(1)
        if ln:
          src_str += f":{ln.group(1)}"
      mapping[norm] = (op.group(1) if op else None, src_str)
  return mapping


def attribute(name, hlo_map):
  """Look up a kernel name in the HLO map (exact, then prefix-trimmed)."""
  if name in hlo_map:
    return hlo_map[name]
  # kernel names occasionally get numeric suffixes: try trimming trailing _N
  base = re.sub(r'_\d+$', '', name)
  for cand in (base, name.split("(")[0]):
    if cand in hlo_map:
      return hlo_map[cand]
  return (None, None)


def group_key(op_name):
  """Aggregate op_name (jax name-stack path) to a pySES function level."""
  if not op_name:
    return "(unattributed)"
  jits = re.findall(r'jit\(([^)]*)\)', op_name)
  if len(jits) >= 2:
    return jits[1]
  parts = op_name.split("/")
  if len(parts) >= 2:
    return parts[1]
  return op_name


def main():
  p = argparse.ArgumentParser()
  p.add_argument("--sqlite", required=True)
  p.add_argument("--hlo", default=None)
  p.add_argument("--steps", type=int, default=3)
  p.add_argument("--top", type=int, default=40)
  p.add_argument("--out", default=None)
  args = p.parse_args()

  kernels, mem = load_kernels(args.sqlite)
  if not kernels:
    print("no kernels found in sqlite export")
    return
  hlo_map = parse_hlo(args.hlo) if args.hlo and os.path.exists(args.hlo) else {}

  t_lo = min(s for s, *_ in kernels + mem)
  t_hi = max(e for _, e, *_ in kernels + mem)
  wall = t_hi - t_lo
  busy = busy_time(kernels + mem)
  n_kern = len(kernels)

  by_name = defaultdict(lambda: [0, 0])   # name -> [total_ns, count]
  for s, e, name, _ in kernels:
    by_name[name][0] += e - s
    by_name[name][1] += 1
  mem_total = sum(e - s for s, e, *_ in mem)

  total_kern_ns = sum(v[0] for v in by_name.values())
  by_group = defaultdict(lambda: [0, 0])
  rows = []
  for name, (tot, cnt) in sorted(by_name.items(), key=lambda kv: -kv[1][0]):
    op_name, src = attribute(name, hlo_map)
    rows.append({"kernel": name, "total_ms": tot / 1e6, "count": cnt,
                 "mean_us": tot / cnt / 1e3, "pct_of_kernel_time":
                 100.0 * tot / total_kern_ns, "op_name": op_name, "src": src})
    by_group[group_key(op_name)][0] += tot
    by_group[group_key(op_name)][1] += cnt

  report = {
      "capture_wall_ms": wall / 1e6,
      "device_busy_ms": busy / 1e6,
      "device_occupancy_pct": 100.0 * busy / wall,
      "kernel_time_ms": total_kern_ns / 1e6,
      "memcpy_memset_ms": mem_total / 1e6,
      "num_kernel_launches": n_kern,
      "kernels_per_step": n_kern / args.steps,
      "mean_kernel_us": total_kern_ns / n_kern / 1e3,
      "hlo_ops_mapped": len(hlo_map),
      "top_kernels": rows[:args.top],
      "by_function": sorted(
          [{"function": g, "total_ms": t / 1e6, "count": c,
            "pct_of_kernel_time": 100.0 * t / total_kern_ns}
           for g, (t, c) in by_group.items()],
          key=lambda r: -r["total_ms"]),
  }

  print("=== device summary (capture window) ===")
  for k in ("capture_wall_ms", "device_busy_ms", "device_occupancy_pct",
            "kernel_time_ms", "memcpy_memset_ms", "num_kernel_launches",
            "kernels_per_step", "mean_kernel_us"):
    print(f"  {k}: {report[k]:.2f}")
  print("\n=== GPU time by pySES function (from op_name metadata) ===")
  for r in report["by_function"]:
    print(f"  {r['function']:40s} {r['total_ms']:9.2f} ms  {r['pct_of_kernel_time']:5.1f}%  "
          f"({r['count']} launches)")
  print(f"\n=== top {args.top} kernels ===")
  for r in report["top_kernels"]:
    src = f"  [{r['src']}]" if r["src"] else ""
    op = f"  {r['op_name']}" if r["op_name"] else ""
    print(f"  {r['total_ms']:8.2f} ms {r['pct_of_kernel_time']:5.1f}% x{r['count']:5d} "
          f"@{r['mean_us']:8.1f}us  {r['kernel'][:70]}{op[:120]}{src}")

  if args.out:
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
      json.dump(report, f, indent=2)
    print(f"\nreport written to {args.out}")


if __name__ == "__main__":
  main()
