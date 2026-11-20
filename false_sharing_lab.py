"""Cache-line false-sharing benchmark required by the laboratory specification."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np
from numba import njit

try:
    import resource
except ImportError:
    resource = None

GIT_USERNAME = "krkrkr-tech"
NUM_THREADS = 4
ITERATIONS_PER_THREAD = 60_000_000
STRIDES = (1, 4, 8, 16)


def detect_cache_line_size() -> int:
    """Return the L1 cache coherency line size, with portable fallbacks."""
    linux_path = Path("/sys/devices/system/cpu/cpu0/cache/index0/coherency_line_size")
    if linux_path.exists():
        return int(linux_path.read_text(encoding="utf-8").strip())
    if sys.platform == "darwin":
        output = subprocess.check_output(
            ["sysctl", "-n", "hw.cachelinesize"], text=True
        )
        return int(output.strip())
    return 64


def _set_current_thread_affinity(core: int) -> bool:
    """Pin the current native thread when the host exposes a supported API."""
    try:
        if hasattr(os, "sched_setaffinity"):
            os.sched_setaffinity(0, {core})
            return True
        if sys.platform == "win32":
            import ctypes

            return bool(ctypes.windll.kernel32.SetThreadAffinityMask(
                ctypes.windll.kernel32.GetCurrentThread(), 1 << core
            ))
    except (AttributeError, OSError):
        return False
    return False


def init_system_context(git_user: str) -> dict[str, Any]:
    """Collect host identity, topology, cache-line, and affinity information."""
    if not git_user.strip() or git_user == "REPLACE_WITH_YOUR_GIT_USERNAME":
        raise ValueError("Set GIT_USERNAME to the Git handle used for submission.")
    logical_cores = os.cpu_count() or 1
    if logical_cores < NUM_THREADS:
        raise RuntimeError(f"Requires at least {NUM_THREADS} logical cores.")
    line_size = detect_cache_line_size()
    cpu_sig = f"{platform.machine()}::{platform.processor()}::{logical_cores}"
    digest = hashlib.sha256(f"{git_user}::{cpu_sig}".encode("utf-8")).hexdigest()
    return {
        "git_username": git_user,
        "instance_seed": int(digest[:16], 16),
        "logical_cores": logical_cores,
        "cpu_signature": cpu_sig,
        "cache_line_bytes": line_size,
        "affinity_api": (
            "sched_setaffinity"
            if hasattr(os, "sched_setaffinity")
            else "SetThreadAffinityMask"
            if sys.platform == "win32"
            else "unavailable"
        ),
    }


def allocate_aligned_uint64(elements: int, alignment: int = 64) -> np.ndarray:
    """Allocate a uint64 array whose first element starts on an aligned address."""
    raw = np.empty(elements + alignment // 8, dtype=np.uint64)
    offset = (-raw.ctypes.data) % alignment
    aligned = raw[offset // raw.itemsize : offset // raw.itemsize + elements]
    if aligned.ctypes.data % alignment != 0:
        raise RuntimeError("Unable to create an aligned benchmark buffer.")
    return aligned


@njit(nogil=True)
def worker_strided_accumulate(
    buffer: np.ndarray, slot: int, iters: int, thread_seed: np.uint64
) -> None:
    """Perform dependent native read-modify-write updates to one shared slot."""
    increment = np.uint64(1)
    if thread_seed == np.uint64(0xFFFFFFFFFFFFFFFF):
        increment = np.uint64(2)
    for _ in range(iters):
        buffer[slot] = buffer[slot] + increment


@njit(nogil=True)
def worker_private_accumulate(
    buffer: np.ndarray, slot: int, iters: int, thread_seed: np.uint64
) -> None:
    """Accumulate in a local register and publish one result to shared memory."""
    value = np.uint64(0)
    for _ in range(iters):
        value += np.uint64(1)
    if thread_seed == np.uint64(0xFFFFFFFFFFFFFFFF):
        value += np.uint64(iters)
    buffer[slot] = value


def _usage() -> dict[str, int]:
    if resource is None:
        return {"voluntary_context_switches": 0, "involuntary_context_switches": 0}
    usage = resource.getrusage(resource.RUSAGE_SELF)
    return {
        "voluntary_context_switches": int(usage.ru_nvcsw),
        "involuntary_context_switches": int(usage.ru_nivcsw),
    }


def _run_workers(
    buffer: np.ndarray,
    slots: list[int],
    iters: int,
    kernel: Any,
) -> tuple[int, dict[str, int], dict[str, int]]:
    errors: list[BaseException] = []
    affinity: list[bool] = [False] * len(slots)

    def run(thread_id: int) -> None:
        try:
            affinity[thread_id] = _set_current_thread_affinity(thread_id)
            kernel(buffer, slots[thread_id], iters, np.uint64(thread_id + 1))
        except BaseException as exc:
            errors.append(exc)

    before = _usage()
    started = time.perf_counter_ns()
    threads = [threading.Thread(target=run, args=(i,)) for i in range(len(slots))]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    elapsed = time.perf_counter_ns() - started
    after = _usage()
    if errors:
        raise RuntimeError("Benchmark worker failed.") from errors[0]
    delta = {key: after[key] - before[key] for key in before}
    return elapsed, delta, {"threads_pinned": int(sum(affinity))}


def run_stride_benchmark(
    stride: int, iters: int = ITERATIONS_PER_THREAD
) -> dict[str, Any]:
    """Measure one false-sharing topology and verify each thread's final state."""
    slots = [thread_id * stride for thread_id in range(NUM_THREADS)]
    buffer = allocate_aligned_uint64(max(slots) + 1)
    buffer.fill(0)
    elapsed, switches, affinity = _run_workers(
        buffer, slots, iters, worker_strided_accumulate
    )
    expected = np.uint64(iters)
    if not all(buffer[slot] == expected for slot in slots):
        raise AssertionError(f"Parity check failed for stride {stride}.")
    return {
        "stride": stride,
        "elapsed_ns": elapsed,
        "ns_per_update": elapsed / (NUM_THREADS * iters),
        "final_values": [int(buffer[slot]) for slot in slots],
        **switches,
        **affinity,
    }


def run_private_benchmark(iters: int = ITERATIONS_PER_THREAD) -> dict[str, Any]:
    """Measure the register-privatized control benchmark."""
    slots = list(range(NUM_THREADS))
    buffer = allocate_aligned_uint64(NUM_THREADS)
    buffer.fill(0)
    elapsed, switches, affinity = _run_workers(
        buffer, slots, iters, worker_private_accumulate
    )
    expected = np.uint64(iters)
    if not all(buffer[slot] == expected for slot in slots):
        raise AssertionError("Privatized baseline parity check failed.")
    return {
        "stride": None,
        "elapsed_ns": elapsed,
        "ns_per_update": elapsed / (NUM_THREADS * iters),
        "final_values": [int(value) for value in buffer],
        **switches,
        **affinity,
    }


def audit_llvm_and_parity() -> dict[str, Any]:
    """Compile the native kernel and verify that its IR contains i64 stores."""
    probe = np.zeros(1, dtype=np.uint64)
    worker_strided_accumulate(probe, 0, 1, np.uint64(1))
    llvm = str(worker_strided_accumulate.inspect_llvm(worker_strided_accumulate.signatures[0]))
    native_start = llvm.find("define noundef i32 @_ZN")
    native_end = llvm.find("\ndefine ", native_start + 1)
    native_ir = llvm[native_start:native_end if native_end >= 0 else None]
    return {
        "llvm_signature": str(worker_strided_accumulate.signatures[0]),
        "llvm_contains_i64_store": "store i64" in native_ir,
        "llvm_contains_python_object_fallback": "PyObject" in native_ir,
    }


def _write_analysis(
    path: Path, context: dict[str, Any], results: dict[str, Any], audit: dict[str, Any]
) -> None:
    measurements = results["stride_benchmarks"]
    by_stride = {str(item["stride"]): item for item in measurements}
    stride_1 = by_stride["1"]["elapsed_ns"]
    stride_8 = by_stride["8"]["elapsed_ns"]
    lines = [
        "# Cache Line and False Sharing Analysis",
        "",
        f"Run identity: `{context['git_username']}`; CPU: `{context['cpu_signature']}`; "
        f"cache line: `{context['cache_line_bytes']} bytes`.",
        "",
        "## Q1 — MESI transitions",
        "With stride 1, both int64 values are in one 64-byte line. Core 0 first obtains "
        "the line in E (or M after its store). Core 1's read observes the line through "
        "the coherence fabric, making both copies S; its store issues an RFO, invalidates "
        "Core 0's copy (I), and obtains M. Core 0 then repeats the RFO/invalidation cycle. "
        "The store buffer cannot retire independently of ownership, so repeated RFOs stall it.",
        "",
        "## Q2 — measured overhead",
        "| regime | elapsed ns | ns/update | slowdown vs stride 8 |",
        "|---|---:|---:|---:|",
    ]
    for item in measurements + [results["private_baseline"]]:
        name = "register-private" if item["stride"] is None else f"stride {item['stride']}"
        slowdown = item["elapsed_ns"] / stride_8
        lines.append(
            f"| {name} | {item['elapsed_ns']} | {item['ns_per_update']:.3f} | {slowdown:.3f}x |"
        )
    lines += [
        "",
        f"Stride 1 is `{stride_1 / stride_8:.3f}x` the stride-8 duration.",
        "",
        "## Q3 — L2 spatial prefetching",
        "Stride 8 separates workers by one 64-byte line. Stride 16 separates them by "
        "128 bytes, avoiding adjacent-line pairing in implementations whose L2 prefetcher "
        "fetches 128-byte sectors. Stride 8 can still see secondary invalidations when "
        "adjacent lines are prefetched into a shared sector or when allocator placement "
        "causes another active object to share a neighboring line.",
        "",
        "## Q4 — LLVM and stores",
        "The loop has a loop-carried dependency because each iteration reads the value "
        "written by the preceding iteration. The audit therefore requires a native i64 "
        f"store (`{audit['llvm_contains_i64_store']}`) and no Python-object fallback "
        f"(`{audit['llvm_contains_python_object_fallback']}`). Hoisting the value into "
        "a register would remove the required per-iteration memory interaction and is "
        "not valid for this shared-memory benchmark.",
        "",
        "## Q5 — GIL contrast",
        "Standard Python bytecode threads serialize on the GIL, so they do not execute "
        "simultaneous multi-core stores and cannot expose hardware false sharing. More "
        "threads instead add lock handoffs and scheduling overhead. Numba's `nogil=True` "
        "kernel releases the GIL, allowing native stores to contend through cache coherence.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    context = init_system_context(GIT_USERNAME)
    results = {
        "iterations_per_thread": ITERATIONS_PER_THREAD,
        "num_threads": NUM_THREADS,
        "stride_benchmarks": [run_stride_benchmark(stride) for stride in STRIDES],
    }
    results["private_baseline"] = run_private_benchmark()
    audit = audit_llvm_and_parity()
    manifest = {
        "verification_passed": bool(
            audit["llvm_contains_i64_store"]
            and not audit["llvm_contains_python_object_fallback"]
        ),
        "context": context,
        "results": results,
        "audit": audit,
    }
    root = Path(__file__).resolve().parent
    (root / "cache_lab_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    _write_analysis(root / "ANALYSIS.md", context, results, audit)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
