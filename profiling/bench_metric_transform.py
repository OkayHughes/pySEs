#!/usr/bin/env python
"""Micro-benchmark: the 2x2 metric-tensor contraction that dominates the NX16 step.

Compares
  einsum   : jnp.einsum("bks,bgs->bkg", A, M)  (XLA may rewrite to cuBLAS)
  unrolled : broadcast-multiply + sum over the length-2 axis (always a fusion)
against a device-copy bandwidth reference, and reports whether the einsum
compiled to a cuBLAS custom call under the current XLA_FLAGS.

Also times the thin GLL derivative contraction ("fij,ki->fkj", 4x4 matrix)
for completeness.
"""
import statistics
import time

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)


def timeit(fn, *args, reps=50):
  out = fn(*args)
  jax.block_until_ready(out)
  ts = []
  for _ in range(reps):
    t0 = time.perf_counter()
    out = fn(*args)
    jax.block_until_ready(out)
    ts.append(time.perf_counter() - t0)
  return statistics.median(ts)


def uses_cublas(jitted, *args):
  txt = jitted.lower(*args).compile().as_text()
  return "__cublas" in txt


def bench_metric(B=24576, K=150, dtype=jnp.float64):
  k0, k1 = jax.random.split(jax.random.PRNGKey(0))
  A = jax.random.normal(k0, (B, K, 2), dtype=dtype)
  M = jax.random.normal(k1, (B, 2, 2), dtype=dtype)

  einsum_fn = jax.jit(lambda a, m: jnp.einsum("bks,bgs->bkg", a, m))
  unrolled_fn = jax.jit(lambda a, m: jnp.sum(a[:, :, None, :] * m[:, None, :, :],
                                             axis=-1))
  copy_fn = jax.jit(lambda a: a * 1.0)

  # sanity: same numbers
  err = float(jnp.max(jnp.abs(einsum_fn(A, M) - unrolled_fn(A, M))))
  assert err < (1e-12 if dtype == jnp.float64 else 1e-3), err

  bytes_touched = (A.size + M.size + B * K * 2) * A.dtype.itemsize
  for name, fn, args in (("einsum", einsum_fn, (A, M)),
                         ("unrolled", unrolled_fn, (A, M)),
                         ("copy_ref", copy_fn, (A,))):
    t = timeit(fn, *args)
    bw = bytes_touched / t / 1e9
    cublas = uses_cublas(fn, *args) if name != "copy_ref" else False
    print(f"  metric[B={B},K={K},{A.dtype.name}] {name:9s} "
          f"{t*1e6:9.1f} us  ~{bw:7.1f} GB/s  cublas={cublas}")


def bench_derivative(B=46080, npt=4, nfield=8, dtype=jnp.float64):
  k0, k1 = jax.random.split(jax.random.PRNGKey(1))
  F = jax.random.normal(k0, (B * nfield, npt, npt), dtype=dtype)
  D = jax.random.normal(k1, (npt, npt), dtype=dtype)

  einsum_fn = jax.jit(lambda f, d: jnp.einsum("fij,ki->fkj", f, d))
  t = timeit(einsum_fn, F, D)
  bytes_touched = 2 * F.size * F.dtype.itemsize
  print(f"  deriv [B={B*nfield},{npt}x{npt},{F.dtype.name}] einsum    "
        f"{t*1e6:9.1f} us  ~{bytes_touched/t/1e9:7.1f} GB/s  "
        f"cublas={uses_cublas(einsum_fn, F, D)}")


if __name__ == "__main__":
  import os
  print(f"XLA_FLAGS={os.environ.get('XLA_FLAGS', '')!r}")
  print(f"device: {jax.devices()[0].device_kind}")
  for dtype in (jnp.float64, jnp.float32):
    for K in (30, 150):
      bench_metric(K=K, dtype=dtype)
  bench_derivative()
  bench_derivative(dtype=jnp.float32)
