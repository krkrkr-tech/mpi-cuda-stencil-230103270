import os
import sys
import json
import math
import time
import hashlib
import platform
import subprocess

import numpy as np
from mpi4py import MPI
from numba import cuda, njit, float64, uint64


TILE_DIM = 16
SHARED_DIM = TILE_DIM + 2
GLOBAL_NX = 512
GLOBAL_NY = 512
STENCIL_STEPS = 300
SHM_ITERS = 40_000_000
LAMBDA_DAMP = 0.15


def get_gpu_uuid():
    out = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader"],
        encoding="utf-8",
    )
    return out.strip().splitlines()[0].strip()


def detect_cpu_line_size():
    path = "/sys/devices/system/cpu/cpu0/cache/index0/coherency_line_size"
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return int(f.read().strip())
    return 64


def init_hybrid_context(github_user):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    if size != 4:
        raise RuntimeError("Requires exactly 4 MPI ranks (2x2 grid), got %d." % size)

    cuda.select_device(0)
    gpu_uuid = comm.bcast(get_gpu_uuid() if rank == 0 else None, root=0)
    line_bytes = detect_cpu_line_size()
    cpu_sig = "%s::%d" % (platform.machine(), os.cpu_count() or 1)
    digest = hashlib.sha256(
        ("%s::%s::%s" % (github_user.strip(), cpu_sig, gpu_uuid)).encode("utf-8")
    ).digest()
    instance_seed = int.from_bytes(digest[:8], "little")

    cart_comm = comm.Create_cart((2, 2), periods=(False, False), reorder=True)
    coords = cart_comm.Get_coords(rank)
    left, right = cart_comm.Shift(1, 1)
    up, down = cart_comm.Shift(0, 1)

    node_comm = comm.Split_type(MPI.COMM_TYPE_SHARED)
    bytes_per_slot = np.dtype(np.uint64).itemsize
    win = MPI.Win.Allocate_shared(
        64 * bytes_per_slot if node_comm.Get_rank() == 0 else 0,
        bytes_per_slot,
        comm=node_comm,
    )
    base_ptr, _ = win.Shared_query(0)
    shm_view = np.frombuffer(
        MPI.memory.fromaddress(base_ptr, 64 * bytes_per_slot), dtype=np.uint64
    )

    return {
        "comm": comm,
        "cart_comm": cart_comm,
        "node_comm": node_comm,
        "win": win,
        "rank": rank,
        "size": size,
        "coords": coords,
        "neighbors": {"up": up, "down": down, "left": left, "right": right},
        "gpu_uuid": gpu_uuid,
        "cpu_sig": cpu_sig,
        "line_bytes": line_bytes,
        "instance_seed": instance_seed,
        "shm_view": shm_view,
    }


def initialize_fields(context):
    rank = context["rank"]
    coords = context["coords"]
    sx, sy = 256 * coords[1], 256 * coords[0]
    rng = np.random.default_rng(context["instance_seed"] + rank)

    u = np.zeros((258, 258), dtype=np.float64)
    kappa = np.full((258, 258), 0.08, dtype=np.float64)
    source = np.zeros((258, 258), dtype=np.float64)

    for iy in range(258):
        gy = sy + iy - 1
        for ix in range(258):
            gx = sx + ix - 1
            u[iy, ix] = 1.0 + 0.01 * rng.random()
            kappa[iy, ix] = 0.04 + 0.04 * (
                0.5 + 0.5 * math.sin((gx + 1) * math.pi / 513)
            )
            source[iy, ix] = 0.001 * math.exp(
                -(((gx - 256.0) / 180.0) ** 2 + ((gy - 256.0) / 180.0) ** 2)
            )

    return (
        cuda.to_device(u),
        cuda.to_device(kappa),
        cuda.to_device(source),
    )


@cuda.jit
def pack_halos_kernel(u, left_buf, right_buf, top_buf, bottom_buf):
    j = cuda.grid(1)
    if j < 256:
        left_buf[j] = u[j + 1, 1]
        right_buf[j] = u[j + 1, 256]
        top_buf[j] = u[1, j + 1]
        bottom_buf[j] = u[256, j + 1]


@cuda.jit
def unpack_halos_kernel(u, left_buf, right_buf, top_buf, bottom_buf):
    j = cuda.grid(1)
    if j < 256:
        u[j + 1, 0] = left_buf[j]
        u[j + 1, 257] = right_buf[j]
        u[0, j + 1] = top_buf[j]
        u[257, j + 1] = bottom_buf[j]


@cuda.jit
def tiled_stencil_kernel(u, kappa, source, u_new, dx2, dt):
    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y
    bx = cuda.blockIdx.x
    by = cuda.blockIdx.y
    tile = cuda.shared.array((SHARED_DIM, SHARED_DIM), dtype=float64)

    gx = bx * TILE_DIM + tx + 1
    gy = by * TILE_DIM + ty + 1
    sx = tx + 1
    sy = ty + 1

    if 0 <= gx < 257 and 0 <= gy < 257:
        tile[sx, sy] = u[gy, gx]
    if tx == 0:
        tile[0, sy] = u[gy, gx - 1] if gx >= 1 else 0.0
    if tx == TILE_DIM - 1:
        tile[TILE_DIM + 1, sy] = u[gy, gx + 1] if gx <= 256 else 0.0
    if ty == 0:
        tile[sx, 0] = u[gy - 1, gx] if gy >= 1 else 0.0
    if ty == TILE_DIM - 1:
        tile[sx, TILE_DIM + 1] = u[gy + 1, gx] if gy <= 256 else 0.0
    cuda.syncthreads()

    if 2 <= gx <= 255 and 2 <= gy <= 255:
        laplacian = (
            tile[sy, sx + 1] - 2.0 * tile[sy, sx] + tile[sy, sx - 1]
        ) / dx2 + (
            tile[sy + 1, sx] - 2.0 * tile[sy, sx] + tile[sy - 1, sx]
        ) / dx2
        diffusion = kappa[gy, gx] * laplacian
        u_new[gy, gx] = u[gy, gx] + dt * (
            LAMBDA_DAMP * diffusion + source[gy, gx]
        )


@cuda.jit
def boundary_stencil_kernel(u, kappa, source, u_new, dx2, dt):
    gx = cuda.grid(1)
    if gx >= 258:
        return
    for gy in range(258):
        is_border = gy < 2 or gy > 255 or gx < 2 or gx > 255
        if is_border:
            if 0 < gx < 257 and 0 < gy < 257:
                lap = (
                    u[gy, gx + 1] - 2.0 * u[gy, gx] + u[gy, gx - 1]
                ) / dx2 + (
                    u[gy + 1, gx] - 2.0 * u[gy, gx] + u[gy - 1, gx]
                ) / dx2
                u_new[gy, gx] = u[gy, gx] + dt * (
                    kappa[gy, gx] * lap * LAMBDA_DAMP + source[gy, gx]
                )


@njit(nogil=True)
def host_shm_accumulate(shm_view, rank, stride, iterations):
    index = rank * stride
    value = shm_view[index]
    i = 0
    while i < iterations:
        value = value + 1
        i += 1
    shm_view[index] = value


@njit(nogil=True)
def cpu_increment_ir(shm_view, iterations):
    shm_view[0] = shm_view[0] + iterations


def benchmark_shm_strides(context):
    node_comm = context["node_comm"]
    rank = context["rank"]
    shm_view = context["shm_view"]
    shm_view[:] = 0
    node_comm.Barrier()

    results = {}
    for stride in (1, 8, 16):
        node_comm.Barrier()
        start = time.perf_counter()
        host_shm_accumulate(shm_view, rank, stride, SHM_ITERS)
        node_comm.Barrier()
        elapsed = time.perf_counter() - start
        times = node_comm.allgather(elapsed)
        results[stride] = {
            "median_s": float(np.median(times)),
            "max_s": float(max(times)),
            "counter": int(shm_view[rank * stride]),
        }
    return results


def pack_all(context, u):
    left = cuda.pinned_array(256, dtype=np.float64)
    right = cuda.pinned_array(256, dtype=np.float64)
    top = cuda.pinned_array(256, dtype=np.float64)
    bottom = cuda.pinned_array(256, dtype=np.float64)
    left_d = cuda.device_array(256, dtype=np.float64)
    right_d = cuda.device_array(256, dtype=np.float64)
    top_d = cuda.device_array(256, dtype=np.float64)
    bottom_d = cuda.device_array(256, dtype=np.float64)
    pack_halos_kernel[1, 256](u, left_d, right_d, top_d, bottom_d)
    left_d.copy_to_host(left)
    right_d.copy_to_host(right)
    top_d.copy_to_host(top)
    bottom_d.copy_to_host(bottom)
    return left, right, top, bottom


def exchange_halos(context, u, dx):
    left_s, right_s, top_s, bottom_s = pack_all(context, u)
    left_r = cuda.pinned_array(256, dtype=np.float64)
    right_r = cuda.pinned_array(256, dtype=np.float64)
    top_r = cuda.pinned_array(256, dtype=np.float64)
    bottom_r = cuda.pinned_array(256, dtype=np.float64)
    neighbors = context["neighbors"]
    cart = context["cart_comm"]
    requests = [
        cart.Irecv(left_r, source=neighbors["left"], tag=11),
        cart.Irecv(right_r, source=neighbors["right"], tag=12),
        cart.Irecv(top_r, source=neighbors["up"], tag=13),
        cart.Irecv(bottom_r, source=neighbors["down"], tag=14),
        cart.Isend(right_s, dest=neighbors["right"], tag=11),
        cart.Isend(left_s, dest=neighbors["left"], tag=12),
        cart.Isend(bottom_s, dest=neighbors["down"], tag=13),
        cart.Isend(top_s, dest=neighbors["up"], tag=14),
    ]
    MPI.Request.Waitall(requests)
    left_d = cuda.to_device(left_r)
    right_d = cuda.to_device(right_r)
    top_d = cuda.to_device(top_r)
    bottom_d = cuda.to_device(bottom_r)
    unpack_halos_kernel[1, 256](u, left_d, right_d, top_d, bottom_d)


def step_serialized(context, u, kappa, source, u_new, dt, dx2):
    exchange_halos(context, u, dx2)
    stream = cuda.stream()
    with stream:
        tiled_stencil_kernel[(16, 16), (16, 16)](
            u, kappa, source, u_new, dx2, dt
        )
        boundary_stencil_kernel[258, 1](u, kappa, source, u_new, dx2, dt)
    stream.synchronize()
    return u_new


def step_overlapped(context, u, kappa, source, u_new, dt, dx2):
    left_s, right_s, top_s, bottom_s = pack_all(context, u)
    left_r = cuda.pinned_array(256, dtype=np.float64)
    right_r = cuda.pinned_array(256, dtype=np.float64)
    top_r = cuda.pinned_array(256, dtype=np.float64)
    bottom_r = cuda.pinned_array(256, dtype=np.float64)
    neighbors = context["neighbors"]
    cart = context["cart_comm"]
    requests = [
        cart.Irecv(left_r, source=neighbors["left"], tag=21),
        cart.Irecv(right_r, source=neighbors["right"], tag=22),
        cart.Irecv(top_r, source=neighbors["up"], tag=23),
        cart.Irecv(bottom_r, source=neighbors["down"], tag=24),
        cart.Isend(right_s, dest=neighbors["right"], tag=21),
        cart.Isend(left_s, dest=neighbors["left"], tag=22),
        cart.Isend(bottom_s, dest=neighbors["down"], tag=23),
        cart.Isend(top_s, dest=neighbors["up"], tag=24),
    ]

    interior_stream = cuda.stream()
    with interior_stream:
        tiled_stencil_kernel[(16, 16), (16, 16)](
            u, kappa, source, u_new, dx2, dt
        )
    interior_stream.synchronize()
    MPI.Request.Waitall(requests)

    boundary_stream = cuda.stream()
    with boundary_stream:
        left_d = cuda.to_device(left_r)
        right_d = cuda.to_device(right_r)
        top_d = cuda.to_device(top_r)
        bottom_d = cuda.to_device(bottom_r)
        unpack_halos_kernel[1, 256](u, left_d, right_d, top_d, bottom_d)
        boundary_stencil_kernel[258, 1](u, kappa, source, u_new, dx2, dt)
    boundary_stream.synchronize()
    return u_new


def audit_llvm_and_parity():
    sig = uint64[::1], uint64
    compiled = cpu_increment_ir.compile(sig)
    llvm = compiled.inspect_llvm()
    ptx = tiled_stencil_kernel.inspect_asm()
    return {
        "llvm_load_u64": llvm.lower().count("load i64"),
        "llvm_store_u64": llvm.lower().count("store i64"),
        "llvm_add": llvm.lower().count("add i64"),
        "ptx_st_shared_f64": ptx.count("st.shared.f64"),
        "ptx_ld_shared_f64": ptx.count("ld.shared.f64"),
        "ptx_bar_sync": ptx.count("bar.sync"),
    }


def run_stencil_suite(context, mode):
    u, kappa, source = initialize_fields(context)
    u_new = cuda.device_array((258, 258), dtype=np.float64)
    dx = 1.0 / 513.0
    dx2 = dx * dx
    kappa_max = 0.08
    dt = 0.20 * dx2 / (4.0 * kappa_max)
    energy_history = {}
    start = time.perf_counter()

    for step in range(STENCIL_STEPS + 1):
        if mode == "serialized":
            u = step_serialized(context, u, kappa, source, u_new, dt, dx2)
        else:
            u = step_overlapped(context, u, kappa, source, u_new, dt, dx2)

        if step in (0, 100, 200, 300):
            host_u = u.copy_to_host()
            local_energy = float(np.sum(host_u * host_u))
            global_energy = context["cart_comm"].allreduce(local_energy, op=MPI.SUM)
            energy_history[str(step)] = global_energy
        if step < STENCIL_STEPS:
            u, u_new = u_new, u

    elapsed_ms = (time.perf_counter() - start) * 1000.0
    return elapsed_ms, energy_history, dt, kappa_max


def main():
    github_user = sys.argv[1] if len(sys.argv) > 1 else os.getenv("GITHUB_USERNAME", "")
    if not github_user:
        raise RuntimeError("A GitHub username is required.")
    context = init_hybrid_context(github_user)

    gpu_serialized_ms, serialized_energy, dt, kappa_max = run_stencil_suite(
        context, "serialized"
    )
    gpu_overlapped_ms, overlapped_energy, _, _ = run_stencil_suite(
        context, "overlapped"
    )
    max_diff = max(
        abs(serialized_energy[str(step)] - overlapped_energy[str(step)])
        for step in (0, 100, 200, 300)
    )
    shm = benchmark_shm_strides(context)
    audits = audit_llvm_and_parity()
    dx = 1.0 / 513.0
    dtmax = dx * dx / (4.0 * kappa_max)

    manifest = {
        "github_username": github_user,
        "instance_seed": context["instance_seed"],
        "cpu_signature": context["cpu_sig"],
        "gpu_uuid": context["gpu_uuid"],
        "mpi_world_size": context["size"],
        "cartesian_topology": [2, 2],
        "cpu_cache_line_bytes": context["line_bytes"],
        "global_shape": [GLOBAL_NX, GLOBAL_NY],
        "rank_shape": [258, 258],
        "stencil_steps": STENCIL_STEPS,
        "shm_iterations": SHM_ITERS,
        "dt": dt,
        "dx": dx,
        "kappa_max": kappa_max,
        "dtmax": dtmax,
        "cfl_satisfied": bool(dt <= dtmax),
        "gpu_serialized_ms": gpu_serialized_ms,
        "gpu_overlapped_ms": gpu_overlapped_ms,
        "serialized_energy": serialized_energy,
        "overlapped_energy": overlapped_energy,
        "cross_mode_max_diff": max_diff,
        "cross_mode_parity_pass": bool(max_diff < 1e-12),
        "shm_strides": shm,
        "shm_stride_1_8_ratio": shm[1]["median_s"] / max(shm[8]["median_s"], 1e-15),
        "audits": audits,
    }

    if context["rank"] == 0:
        with open("hybrid_lab_manifest.json", "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)
        print(json.dumps(manifest, indent=2))
    context["win"].Free()


if __name__ == "__main__":
    main()
