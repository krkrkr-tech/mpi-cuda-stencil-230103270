import hashlib
import json
import os
import subprocess
import sys
import time

import numpy as np
from mpi4py import MPI
from numba import cuda, float32, float64, int32
from numba import types


TILE_DIM = 16
SHARED_DIM = TILE_DIM + 2
GLOBAL_NX = 512
GLOBAL_NY = 512
LOCAL_NX = GLOBAL_NX // 2
LOCAL_NY = GLOBAL_NY // 2
STEPS = 400
LAMBDA_DAMP = 0.15
DT = 0.00005
DX = 1.0 / 513.0
DY = 1.0 / 513.0
DX2 = DX * DX
DY2 = DY * DY


def get_gpu_uuid():
    output = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader"],
        encoding="utf-8",
    )
    return output.strip().splitlines()[0].strip()


def derive_student_seed(github_user, gpu_uuid):
    digest = hashlib.sha256(
        f"{github_user.strip()}::{gpu_uuid}".encode("utf-8")
    ).hexdigest()
    return int(digest[:16], 16)


def init_distributed_context(github_user):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    if size != 4:
        raise RuntimeError(f"Expected exactly 4 MPI ranks, got {size}")

    cuda.select_device(0)
    gpu_uuid = get_gpu_uuid() if rank == 0 else None
    gpu_uuid = comm.bcast(gpu_uuid, root=0)
    instance_seed = derive_student_seed(github_user, gpu_uuid)

    cart_comm = comm.Create_cart(
        dims=[2, 2],
        periods=[False, False],
        reorder=False,
    )
    coords = cart_comm.Get_coords(rank)
    nbr_up, nbr_down = cart_comm.Shift(0, 1)
    nbr_left, nbr_right = cart_comm.Shift(1, 1)

    return cart_comm, rank, coords, gpu_uuid, instance_seed, (
        nbr_up,
        nbr_down,
        nbr_left,
        nbr_right,
    )


def initialize_subdomains(coords, instance_seed):
    row_rank, col_rank = coords
    rng = np.random.default_rng(instance_seed & 0xFFFFFFFF)

    u = np.zeros((LOCAL_NY + 2, LOCAL_NX + 2), dtype=np.float64)
    kappa = np.zeros_like(u)
    source = np.zeros_like(u)

    base_x = col_rank * LOCAL_NX
    base_y = row_rank * LOCAL_NY

    for local_i in range(LOCAL_NY + 2):
        global_i = base_y + local_i - 1
        y = (global_i + 1.0) / (GLOBAL_NY + 1.0)
        for local_j in range(LOCAL_NX + 2):
            global_j = base_x + local_j - 1
            x = (global_j + 1.0) / (GLOBAL_NX + 1.0)
            kappa[local_i, local_j] = 0.08 + 0.16 * (
                0.5 + 0.5 * np.sin(2.0 * np.pi * x) * np.cos(2.0 * np.pi * y)
            )
            source[local_i, local_j] = (
                0.025 * np.exp(-((x - 0.30) ** 2 + (y - 0.35) ** 2) / 0.001)
                + 0.012 * np.exp(-((x - 0.75) ** 2 + (y - 0.68) ** 2) / 0.002)
            )
            for pulse in range(3):
                angle = 2.0 * np.pi * ((instance_seed >> (pulse * 12)) & 0xFFF) / 4096.0
                cx = 0.20 + 0.60 * ((np.cos(angle) + 1.0) / 2.0)
                cy = 0.20 + 0.60 * ((np.sin(angle) + 1.0) / 2.0)
                amplitude = 0.10 + 0.05 * (((instance_seed >> (pulse * 16)) & 0xFF) / 255.0)
                u[local_i, local_j] += amplitude * np.exp(
                    -((x - cx) ** 2 + (y - cy) ** 2) / 0.012
                )

    # Zero local boundary ghost/outer boundary entries.
    u[0, :] = 0.0
    u[LOCAL_NY + 1, :] = 0.0
    u[:, 0] = 0.0
    u[:, LOCAL_NX + 1] = 0.0

    if row_rank == 0:
        u[1, :] = 0.0
    if row_rank == 1:
        u[LOCAL_NY, :] = 0.0
    if col_rank == 0:
        u[:, 1] = 0.0
    if col_rank == 1:
        u[:, LOCAL_NX] = 0.0

    return (
        cuda.to_device(u),
        cuda.to_device(kappa),
        cuda.to_device(source),
    )


@cuda.jit
def pack_halos_kernel(u, send_top, send_bottom, send_left, send_right, nx, ny):
    i = cuda.grid(1)
    if i < nx:
        send_top[i] = u[1, i + 1]
        send_bottom[i] = u[ny, i + 1]
    if i < ny:
        send_left[i] = u[i + 1, 1]
        send_right[i] = u[i + 1, nx]


@cuda.jit
def unpack_halos_kernel(u, recv_top, recv_bottom, recv_left, recv_right, nx, ny):
    i = cuda.grid(1)
    if i < nx:
        u[0, i + 1] = recv_top[i]
        u[ny + 1, i + 1] = recv_bottom[i]
    if i < ny:
        u[i + 1, 0] = recv_left[i]
        u[i + 1, nx + 1] = recv_right[i]


@cuda.jit
def tiled_stencil_kernel(u, u_next, kappa, source, dt, dx2, dy2, lambda_damp,
                         r_start, r_end, c_start, c_end, perimeter_only):
    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y
    bx = cuda.blockIdx.x
    by = cuda.blockIdx.y

    sx = tx + 1
    sy = ty + 1
    smem = cuda.shared.array((SHARED_DIM, SHARED_DIM), dtype=float64)

    r0 = r_start + by * TILE_DIM
    c0 = c_start + bx * TILE_DIM

    # Load the 18x18 tile, including one-cell halo apron.
    flat_id = ty * TILE_DIM + tx
    for k in range(flat_id, SHARED_DIM * SHARED_DIM, TILE_DIM * TILE_DIM):
        smem_y = k // SHARED_DIM
        smem_x = k % SHARED_DIM
        gr = r0 + smem_y - 1
        gc = c0 + smem_x - 1
        if gr < 0 or gr > 257 or gc < 0 or gc > 257:
            smem[smem_y, smem_x] = 0.0
        else:
            smem[smem_y, smem_x] = u[gr, gc]

    cuda.syncthreads()

    r = r0 + ty
    c = c0 + tx
    if r >= r_end or c >= c_end:
        return

    if perimeter_only:
        boundary = (
            r == r_start or r == r_end - 1
            or c == c_start or c == c_end - 1
        )
        if not boundary:
            return

    center = smem[sy, sx]
    east = smem[sy, sx + 1]
    west = smem[sy, sx - 1]
    north = smem[sy - 1, sx]
    south = smem[sy + 1, sx]

    kappa_center = kappa[r, c]
    kappa_east = 0.5 * (kappa[r, c + 1] + kappa_center)
    kappa_west = 0.5 * (kappa[r, c - 1] + kappa_center)
    kappa_north = 0.5 * (kappa[r - 1, c] + kappa_center)
    kappa_south = 0.5 * (kappa[r + 1, c] + kappa_center)

    rhs = (
        (kappa_east * (east - center) - kappa_west * (center - west)) / dx2
        + (kappa_south * (south - center) - kappa_north * (center - north)) / dy2
        - lambda_damp * center * center * center
        + source[r, c]
    )
    u_next[r, c] = center + dt * rhs


def allocate_exchange_buffers():
    return (
        cuda.pinned_array((LOCAL_NX,), dtype=np.float64),
        cuda.pinned_array((LOCAL_NX,), dtype=np.float64),
        cuda.pinned_array((LOCAL_NY,), dtype=np.float64),
        cuda.pinned_array((LOCAL_NY,), dtype=np.float64),
        cuda.pinned_array((LOCAL_NX,), dtype=np.float64),
        cuda.pinned_array((LOCAL_NX,), dtype=np.float64),
        cuda.pinned_array((LOCAL_NY,), dtype=np.float64),
        cuda.pinned_array((LOCAL_NY,), dtype=np.float64),
        cuda.device_array((LOCAL_NX,), dtype=np.float64),
        cuda.device_array((LOCAL_NX,), dtype=np.float64),
        cuda.device_array((LOCAL_NY,), dtype=np.float64),
        cuda.device_array((LOCAL_NY,), dtype=np.float64),
    )


def exchange_halos_async(
    u,
    cart_comm,
    rank,
    neighbors,
    boundary_stream,
    buffers,
):
    nbr_up, nbr_down, nbr_left, nbr_right = neighbors
    (
        send_top_host,
        send_bottom_host,
        send_left_host,
        send_right_host,
        recv_top_host,
        recv_bottom_host,
        recv_left_host,
        recv_right_host,
        send_top_dev,
        send_bottom_dev,
        send_left_dev,
        send_right_dev,
    ) = buffers

    recv_top_dev = cuda.device_array((LOCAL_NX,), dtype=np.float64)
    recv_bottom_dev = cuda.device_array((LOCAL_NX,), dtype=np.float64)
    recv_left_dev = cuda.device_array((LOCAL_NY,), dtype=np.float64)
    recv_right_dev = cuda.device_array((LOCAL_NY,), dtype=np.float64)

    pack_halos_kernel[(64,), (256,), boundary_stream](
        u,
        send_top_dev,
        send_bottom_dev,
        send_left_dev,
        send_right_dev,
        LOCAL_NX,
        LOCAL_NY,
    )

    send_top_dev.copy_to_host(send_top_host, stream=boundary_stream)
    send_bottom_dev.copy_to_host(send_bottom_host, stream=boundary_stream)
    send_left_dev.copy_to_host(send_left_host, stream=boundary_stream)
    send_right_dev.copy_to_host(send_right_host, stream=boundary_stream)
    boundary_stream.synchronize()

    requests = []

    if nbr_up != MPI.PROC_NULL:
        requests.append(cart_comm.Irecv(recv_top_host, source=nbr_up, tag=1))
    if nbr_down != MPI.PROC_NULL:
        requests.append(cart_comm.Irecv(recv_bottom_host, source=nbr_down, tag=2))
    if nbr_left != MPI.PROC_NULL:
        requests.append(cart_comm.Irecv(recv_left_host, source=nbr_left, tag=3))
    if nbr_right != MPI.PROC_NULL:
        requests.append(cart_comm.Irecv(recv_right_host, source=nbr_right, tag=4))

    if nbr_up != MPI.PROC_NULL:
        requests.append(cart_comm.Isend(send_top_host, dest=nbr_up, tag=2))
    if nbr_down != MPI.PROC_NULL:
        requests.append(cart_comm.Isend(send_bottom_host, dest=nbr_down, tag=1))
    if nbr_left != MPI.PROC_NULL:
        requests.append(cart_comm.Isend(send_left_host, dest=nbr_left, tag=4))
    if nbr_right != MPI.PROC_NULL:
        requests.append(cart_comm.Isend(send_right_host, dest=nbr_right, tag=3))

    if requests:
        MPI.Request.Waitall(requests)

    if nbr_up != MPI.PROC_NULL:
        recv_top_dev.copy_to_device(recv_top_host, stream=boundary_stream)
    if nbr_down != MPI.PROC_NULL:
        recv_bottom_dev.copy_to_device(recv_bottom_host, stream=boundary_stream)
    if nbr_left != MPI.PROC_NULL:
        recv_left_dev.copy_to_device(recv_left_host, stream=boundary_stream)
    if nbr_right != MPI.PROC_NULL:
        recv_right_dev.copy_to_device(recv_right_host, stream=boundary_stream)

    unpack_halos_kernel[(64,), (256,), boundary_stream](
        u,
        recv_top_dev,
        recv_bottom_dev,
        recv_left_dev,
        recv_right_dev,
        LOCAL_NX,
        LOCAL_NY,
    )
    return recv_top_host, recv_bottom_host, recv_left_host, recv_right_host


def launch_stencil(u, u_next, kappa, source, stream, perimeter_only):
    if perimeter_only:
        r_start, r_end = 1, LOCAL_NY + 1
        c_start, c_end = 1, LOCAL_NX + 1
    else:
        r_start, r_end = 2, LOCAL_NY
        c_start, c_end = 2, LOCAL_NX

    grid = (
        () // TILE_DIM,
        (r_end - r_start + TILE_DIM - 1) // TILE_DIM,
    )
    tiled_stencil_kernel[grid, (TILE_DIM, TILE_DIM), stream](
        u,
        u_next,
        kappa,
        source,
        DT,
        DX2,
        DY2,
        LAMBDA_DAMP,
        r_start,
        r_end,
        c_start,
        c_end,
        perimeter_only,
    )


def thermal_energy(u, cart_comm):
    host = u.copy_to_host()
    local_energy = float(np.sum(host) * DX * DY)
    return float(cart_comm.Allreduce(local_energy, op=MPI.SUM))


def run_solver(github_user, overlapped=True):
    cart_comm, rank, coords, gpu_uuid, instance_seed, neighbors = (
        init_distributed_context(github_user)
    )
    u, kappa, source = initialize_subdomains(coords, instance_seed)
    u_next = cuda.to_device(np.zeros((LOCAL_NY + 2, LOCAL_NX + 2), dtype=np.float64))

    boundary_stream = cuda.stream()
    interior_stream = cuda.stream()
    exchange_buffers = allocate_exchange_buffers()

    energy_trace = {}
    start = time.perf_counter()

    for step in range(STEPS):
        if overlapped:
            exchange_halos_async(
                u,
                cart_comm,
                rank,
                neighbors,
                boundary_stream,
                exchange_buffers,
            )
            launch_stencil(u, u_next, kappa, source, interior_stream, False)
            interior_stream.synchronize()
            boundary_stream.synchronize()
            launch_stencil(u, u_next, kappa, source, boundary_stream, True)
        else:
            exchange_halos_async(
                u,
                cart_comm,
                rank,
                neighbors,
                boundary_stream,
                exchange_buffers,
            )
            boundary_stream.synchronize()
            launch_stencil(u, u_next, kappa, source, boundary_stream, False)
            boundary_stream.synchronize()
            launch_stencil(u, u_next, kappa, source, boundary_stream, True)
            boundary_stream.synchronize()

        u, u_next = u_next, u

        if step in (0, 100, 200, 300, 400):
            energy_trace[str(step)] = thermal_energy(u, cart_comm)

    # Record the final state after exactly 400 completed updates.
    energy_trace["400"] = thermal_energy(u, cart_comm)
    cuda.synchronize()
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    return u, elapsed_ms, energy_trace, instance_seed, gpu_uuid


def ptx_audit():
    signature = (
        float64[:, ::1],
        float64[:, ::1],
        float64[:, ::1],
        float64[:, ::1],
        float64,
        float64,
        float64,
        float64,
        int32,
        int32,
        int32,
        int32,
        types.boolean,
    )
    try:
        ptx = tiled_stencil_kernel.inspect_asm(signature)
    except Exception:
        ptx = tiled_stencil_kernel.inspect_asm()
        if isinstance(ptx, dict):
            ptx = "\n".join(ptx.values())
    return {
        "st.shared.f64": ptx.count("st.shared.f64"),
        "ld.shared.f64": ptx.count("ld.shared.f64"),
        "bar.sync": ptx.count("bar.sync"),
    }


def main():
    if len(sys.argv) < 2:
        raise RuntimeError("Provide GITHUB_USERNAME as the first argument")
    github_user = sys.argv[1]

    u_serialized, serialized_ms, energy_serialized, seed, gpu_uuid = run_solver(
        github_user, overlapped=False
    )
    u_overlapped, overlapped_ms, energy_overlapped, _, _ = run_solver(
        github_user, overlapped=True
    )

    serialized_host = u_serialized.copy_to_host()
    overlapped_host = u_overlapped.copy_to_host()
    parity_max_diff = float(np.max(np.abs(serialized_host - overlapped_host)))

    ptx_counts = ptx_audit()
    ptx_audit_passed = (
        ptx_counts["st.shared.f64"] > 0
        and ptx_counts["ld.shared.f64"] > 0
        and ptx_counts["bar.sync"] > 0
    )

    cart_comm = MPI.COMM_WORLD
    rank = cart_comm.Get_rank()
    if rank == 0:
        manifest = {
            "GITHUB_USERNAME": github_user,
            "INSTANCE_SEED": str(seed),
            "GPU_UUID": gpu_uuid,
            "mpi_ranks": 4,
            "cartesian_topology": [2, 2],
            "steps": STEPS,
            "dt": DT,
            "dx": DX,
            "kappa_max": 0.24,
            "dt_max": DX2 / (4.0 * 0.24),
            "ptx_audit_passed": ptx_audit_passed,
            "ptx_counts": ptx_counts,
            "serialized_ms": serialized_ms,
            "overlapped_ms": overlapped_ms,
            "parity_max_diff": parity_max_diff,
            "energy_serialized": energy_serialized,
        }
        with open("submission_manifest.json", "w", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=2)

        print("MPI ranks: 4")
        print("PTX audit:", ptx_counts)
        print(f"Serialized ms: {serialized_ms:.3f}")
        print(f"Overlapped ms: {overlapped_ms:.3f}")
        print(f"Parity max diff: {parity_max_diff:.17e}")
        print("Energy trace:", json.dumps(energy_serialized, sort_keys=True))

        if parity_max_diff >= 1e-12:
            raise RuntimeError("Serialized/overlapped parity check failed")
        if not ptx_audit_passed:
            raise RuntimeError("PTX shared-memory audit failed")


if __name__ == "__main__":
    main()
