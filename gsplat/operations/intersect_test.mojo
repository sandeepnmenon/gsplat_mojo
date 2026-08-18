"""Exact verification of the tile-intersection and depth-sort stage.

Runs the full device pipeline on the shared spread scene, then rebuilds the
expected result serially on the host and compares:

  * total intersection count
  * `tile_offsets` for all C*N_TILES tiles, entry by entry
  * the *set* of gaussians listed for every tile (as a bitmask, so a
    duplicated or dropped id is caught)
  * that depths are non-decreasing inside each tile's run, and tile ids are
    non-decreasing across the whole list -- i.e. the sort actually sorted

The host side recomputes the tile footprint independently of the device
kernel, but does reuse `Mat3`/`quat_to_rotmat` for the covariance, since the
rasterizer's phase-2 check already pins those against a scalar reference.
What is being checked here is the binning, scan, emit, sort and run-boundary
logic, which is where the parallelism lives.
"""

from std.math import ceildiv, log, sqrt
from std.sys import has_accelerator
from max.gpu.host import DeviceContext
from layout import TileTensor

from config import (
    C,
    CX,
    CY,
    DTYPE,
    FOCAL,
    IDTYPE,
    KDTYPE,
    MAX_ISECTS,
    MIN_ALPHA,
    N_MAX,
    N_TEST,
    N_TILES,
    N_TILES_X,
    N_TILES_Y,
    SCAN_BLOCK,
    SCAN_NUM_BLOCKS,
    SCAN_WIDTH,
    TILE,
    Z_NEAR,
    layout_blocksums,
    layout_cn,
    layout_cn4,
    layout_cn_flat,
    layout_cn_f,
    layout_cn_i,
    layout_intrinsics,
    layout_isects,
    layout_n3,
    layout_n4,
    layout_one,
    layout_tiles_flat,
    layout_viewmats,
)
from operations.intersect import (
    KEY_MAX,
    SPLAT_DILATION,
    add_block_offsets,
    bitonic_step,
    emit_isects,
    project_and_count,
    scan_block,
    scan_block_sums,
    write_tile_offsets,
)
from scene import (
    gauss_color,
    spread_mean,
    spread_opacity,
    spread_quat,
    spread_quat_norm,
    spread_scale,
    view_rot,
    view_trans,
)
from utils_t import Mat3, matmul3x3, quat_to_rotmat, transpose
from vec import Vec3, Vec4


def ref_bbox(g: Int) -> Tuple[Int, Int, Int, Int, Float32]:
    """Host recomputation of one gaussian's tile bbox. Returns (x0,y0,x1,y1,z);
    an empty box (x1<=x0) means the gaussian is culled."""
    var rv = Mat3()
    comptime for r in range(3):
        comptime for c in range(3):
            rv[r, c] = view_rot(r, c)
    var tv = Vec3(view_trans(0), view_trans(1), view_trans(2))
    var mean = Vec3(spread_mean(g, 0), spread_mean(g, 1), spread_mean(g, 2))
    var pc = rv * mean + tv
    var z = pc[2]
    if z < Z_NEAR:
        return (0, 0, 0, 0, Float32(0.0))

    var opacity = spread_opacity(g)
    var ratio = opacity / Float32(MIN_ALPHA)
    if ratio <= 1.0:
        return (0, 0, 0, 0, Float32(0.0))
    var kappa = sqrt(2.0 * log(ratio))

    var qn = spread_quat_norm(g)
    var quat = Vec4(
        spread_quat(g, 0) / qn,
        spread_quat(g, 1) / qn,
        spread_quat(g, 2) / qn,
        spread_quat(g, 3) / qn,
    )
    var rot = quat_to_rotmat(quat)
    var ssq = Mat3.diagonal(
        spread_scale(g, 0) * spread_scale(g, 0),
        spread_scale(g, 1) * spread_scale(g, 1),
        spread_scale(g, 2) * spread_scale(g, 2),
    )
    var cov_cam = matmul3x3(
        matmul3x3(rv, matmul3x3(matmul3x3(rot, ssq), transpose(rot))),
        transpose(rv),
    )

    var inv_z = 1.0 / z
    var j00 = FOCAL * inv_z
    var j02 = -FOCAL * pc[0] * inv_z * inv_z
    var j11 = FOCAL * inv_z
    var j12 = -FOCAL * pc[1] * inv_z * inv_z
    var jr0 = Vec3(j00, 0.0, j02)
    var jr1 = Vec3(0.0, j11, j12)
    var a = jr0.dot(cov_cam * jr0) + SPLAT_DILATION
    var b = jr0.dot(cov_cam * jr1)
    var c2 = jr1.dot(cov_cam * jr1) + SPLAT_DILATION

    var mid = 0.5 * (a + c2)
    var disc = mid * mid - (a * c2 - b * b)
    if disc < 0.0:
        disc = 0.0
    var lam = mid + sqrt(disc)
    if lam <= 0.0:
        return (0, 0, 0, 0, Float32(0.0))
    var radius = kappa * sqrt(lam)

    var sx = FOCAL * pc[0] * inv_z + CX
    var sy = FOCAL * pc[1] * inv_z + CY

    var x0 = Int((sx - radius) / Float32(TILE))
    var x1 = Int(ceildiv(sx + radius, Float32(TILE)))
    var y0 = Int((sy - radius) / Float32(TILE))
    var y1 = Int(ceildiv(sy + radius, Float32(TILE)))
    if x0 < 0:
        x0 = 0
    if y0 < 0:
        y0 = 0
    if x1 > N_TILES_X:
        x1 = N_TILES_X
    if y1 > N_TILES_Y:
        y1 = N_TILES_Y
    if x1 <= x0 or y1 <= y0:
        return (0, 0, 0, 0, Float32(0.0))
    return (x0, y0, x1, y1, z)


def main() raises:
    comptime assert has_accelerator(), "requires a GPU"
    comptime assert (
        C * N_MAX <= SCAN_BLOCK * SCAN_WIDTH
    ), "two-level scan cannot cover C*N_MAX"

    var n_gauss = N_TEST
    var ctx = DeviceContext()

    var means_buf = ctx.enqueue_create_buffer[DTYPE](N_MAX * 3)
    var scales_buf = ctx.enqueue_create_buffer[DTYPE](N_MAX * 3)
    var quats_buf = ctx.enqueue_create_buffer[DTYPE](N_MAX * 4)
    var opac_buf = ctx.enqueue_create_buffer[DTYPE](C * N_MAX)
    var view_buf = ctx.enqueue_create_buffer[DTYPE](C * 16)
    var ks_buf = ctx.enqueue_create_buffer[DTYPE](C * 9)
    var blocksum_buf = ctx.enqueue_create_buffer[IDTYPE](SCAN_NUM_BLOCKS)
    var counts_buf = ctx.enqueue_create_buffer[IDTYPE](C * N_MAX)
    var offsets_buf = ctx.enqueue_create_buffer[IDTYPE](C * N_MAX)
    var bbox_buf = ctx.enqueue_create_buffer[IDTYPE](C * N_MAX * 4)
    var depth_buf = ctx.enqueue_create_buffer[DTYPE](C * N_MAX)
    var total_buf = ctx.enqueue_create_buffer[IDTYPE](1)
    var tileoff_buf = ctx.enqueue_create_buffer[IDTYPE](C * N_TILES)

    var means_h = ctx.enqueue_create_host_buffer[DTYPE](N_MAX * 3)
    var scales_h = ctx.enqueue_create_host_buffer[DTYPE](N_MAX * 3)
    var quats_h = ctx.enqueue_create_host_buffer[DTYPE](N_MAX * 4)
    var opac_h = ctx.enqueue_create_host_buffer[DTYPE](C * N_MAX)
    var view_h = ctx.enqueue_create_host_buffer[DTYPE](C * 16)
    var ks_h = ctx.enqueue_create_host_buffer[DTYPE](C * 9)
    var total_h = ctx.enqueue_create_host_buffer[IDTYPE](1)
    ctx.synchronize()

    for g in range(n_gauss):
        var qn = spread_quat_norm(g)
        comptime for a in range(3):
            means_h[g * 3 + a] = spread_mean(g, a)
            scales_h[g * 3 + a] = spread_scale(g, a)
        comptime for a in range(4):
            quats_h[g * 4 + a] = spread_quat(g, a) / qn
    for c in range(C):
        for g in range(n_gauss):
            opac_h[c * N_MAX + g] = spread_opacity(g)
        for e in range(16):
            view_h[c * 16 + e] = 0.0
        comptime for r in range(3):
            comptime for cc in range(3):
                view_h[c * 16 + r * 4 + cc] = view_rot(r, cc)
        view_h[c * 16 + 3] = view_trans(0)
        view_h[c * 16 + 7] = view_trans(1)
        view_h[c * 16 + 11] = view_trans(2)
        view_h[c * 16 + 15] = 1.0
        for e in range(9):
            ks_h[c * 9 + e] = 0.0
        ks_h[c * 9 + 0] = FOCAL
        ks_h[c * 9 + 4] = FOCAL
        ks_h[c * 9 + 2] = CX
        ks_h[c * 9 + 5] = CY
        ks_h[c * 9 + 8] = 1.0

    ctx.enqueue_copy(dst_buf=means_buf, src_buf=means_h)
    ctx.enqueue_copy(dst_buf=scales_buf, src_buf=scales_h)
    ctx.enqueue_copy(dst_buf=quats_buf, src_buf=quats_h)
    ctx.enqueue_copy(dst_buf=opac_buf, src_buf=opac_h)
    ctx.enqueue_copy(dst_buf=view_buf, src_buf=view_h)
    ctx.enqueue_copy(dst_buf=ks_buf, src_buf=ks_h)
    counts_buf.enqueue_fill(0)  # the scan covers the whole capacity

    var means = TileTensor(means_buf, layout_n3)
    var scales = TileTensor(scales_buf, layout_n3)
    var quats = TileTensor(quats_buf, layout_n4)
    var opacities = TileTensor(opac_buf, layout_cn)
    var viewmats = TileTensor(view_buf, layout_viewmats)
    var ks = TileTensor(ks_buf, layout_intrinsics)
    var counts = TileTensor(counts_buf, layout_cn_i)
    var offsets = TileTensor(offsets_buf, layout_cn_i)
    var bboxes = TileTensor(bbox_buf, layout_cn4)
    var depths = TileTensor(depth_buf, layout_cn_f)
    var total = TileTensor(total_buf, layout_one)
    var counts_flat = TileTensor(counts_buf, layout_cn_flat)
    var offsets_flat = TileTensor(offsets_buf, layout_cn_flat)
    var block_sums = TileTensor(blocksum_buf, layout_blocksums)
    var tile_offsets = TileTensor(tileoff_buf, layout_tiles_flat)

    comptime TPB = 256
    var gblocks = ceildiv(n_gauss, TPB)

    ctx.enqueue_function[project_and_count](
        means, scales, quats, opacities, viewmats, ks,
        counts, bboxes, depths,
        Int32(n_gauss), Int32(N_TILES_X), Int32(N_TILES_Y), Int32(TILE),
        grid_dim=(gblocks, C), block_dim=TPB,
    )
    ctx.enqueue_function[scan_block](
        counts_flat, offsets_flat, block_sums, Int32(C * N_MAX),
        grid_dim=SCAN_NUM_BLOCKS, block_dim=SCAN_BLOCK,
    )
    ctx.enqueue_function[scan_block_sums](
        block_sums, total, Int32(SCAN_NUM_BLOCKS),
        grid_dim=1, block_dim=SCAN_WIDTH,
    )
    ctx.enqueue_function[add_block_offsets](
        offsets_flat, block_sums, Int32(C * N_MAX),
        grid_dim=SCAN_NUM_BLOCKS, block_dim=SCAN_BLOCK,
    )
    ctx.enqueue_copy(dst_buf=total_h, src_buf=total_buf)
    ctx.synchronize()

    var n_isects = Int(total_h[0])
    print("intersections:", n_isects, "of a", MAX_ISECTS, "bound")
    if n_isects <= 0 or n_isects > MAX_ISECTS:
        raise Error("implausible intersection count")

    var n_pow2 = 1
    while n_pow2 < n_isects:
        n_pow2 *= 2
    var keys_buf = ctx.enqueue_create_buffer[KDTYPE](n_pow2)
    var vals_buf = ctx.enqueue_create_buffer[IDTYPE](n_pow2)
    keys_buf.enqueue_fill(KEY_MAX)  # padding sorts to the end
    vals_buf.enqueue_fill(-1)
    var keys = TileTensor(keys_buf, layout_isects)
    var vals = TileTensor(vals_buf, layout_isects)

    ctx.enqueue_function[emit_isects](
        bboxes, depths, offsets, counts, keys, vals,
        Int32(n_gauss), Int32(N_TILES_X), Int32(N_TILES),
        grid_dim=(gblocks, C), block_dim=TPB,
    )

    # Bitonic sort needs a power-of-two span; the tail is already KEY_MAX.
    var sort_blocks = ceildiv(n_pow2, TPB)
    var passes = 0
    var k = 2
    while k <= n_pow2:
        var j = k // 2
        while j > 0:
            ctx.enqueue_function[bitonic_step](
                keys, vals, Int32(n_pow2), Int32(k), Int32(j),
                grid_dim=sort_blocks, block_dim=TPB,
            )
            passes += 1
            j //= 2
        k *= 2
    print("bitonic: padded to", n_pow2, "in", passes, "passes")

    tileoff_buf.enqueue_fill(Int32(n_isects))
    ctx.enqueue_function[write_tile_offsets](
        keys, tile_offsets, Int32(n_isects),
        grid_dim=ceildiv(n_isects, TPB), block_dim=TPB,
    )
    ctx.synchronize()

    # ---- host reference --------------------------------------------------
    var ref_counts = List[Int]()
    for _ in range(C * N_TILES):
        ref_counts.append(0)
    var bx0 = List[Int]()
    var by0 = List[Int]()
    var bx1 = List[Int]()
    var by1 = List[Int]()
    var bz = List[Float32]()
    for g in range(n_gauss):
        var r = ref_bbox(g)
        bx0.append(r[0]); by0.append(r[1])
        bx1.append(r[2]); by1.append(r[3]); bz.append(r[4])
    var ref_total = 0
    var culled = 0
    for g in range(n_gauss):
        if bx1[g] <= bx0[g]:
            culled += 1
            continue
        for ty in range(by0[g], by1[g]):
            for tx in range(bx0[g], bx1[g]):
                ref_counts[ty * N_TILES_X + tx] += 1
                ref_total += 1

    print("host reference:", ref_total, "intersections |", culled, "gaussians culled")

    var bad_offsets = 0
    var bad_sets = 0
    var bad_order = 0
    var nonempty = 0

    with keys_buf.map_to_host() as kh:
        with vals_buf.map_to_host() as vh:
            with tileoff_buf.map_to_host() as th:
                # tile_offsets must be the exclusive scan of the per-tile counts
                var running = 0
                for t in range(C * N_TILES):
                    if Int(th[t]) != running:
                        bad_offsets += 1
                    running += ref_counts[t]
                if running != ref_total:
                    raise Error("reference scan disagrees with itself")

                # per-tile membership and ordering
                for t in range(C * N_TILES):
                    var start = Int(th[t])
                    var end = n_isects if t == C * N_TILES - 1 else Int(th[t + 1])
                    if end - start != ref_counts[t]:
                        bad_sets += 1
                        continue
                    if end > start:
                        nonempty += 1
                    var want: Int = 0
                    for g in range(n_gauss):
                        if bx1[g] <= bx0[g]:
                            continue
                        var tx = t % N_TILES_X
                        var ty = t // N_TILES_X
                        if bx0[g] <= tx < bx1[g] and by0[g] <= ty < by1[g]:
                            want |= 1 << g
                    var got: Int = 0
                    var prev_depth: UInt64 = 0
                    for i in range(start, end):
                        got |= 1 << Int(vh[i])
                        if Int(kh[i] >> 32) != t:
                            bad_order += 1
                        var d = UInt64(kh[i] & 0xFFFFFFFF)
                        if i > start and d < prev_depth:
                            bad_order += 1
                        prev_depth = d
                    if got != want:
                        bad_sets += 1

    print("tiles with gaussians:", nonempty, "of", C * N_TILES)
    print(
        "mismatches — offsets:", bad_offsets,
        "| tile membership:", bad_sets,
        "| ordering:", bad_order,
    )
    if (
        n_isects == ref_total
        and bad_offsets == 0
        and bad_sets == 0
        and bad_order == 0
        and nonempty > 0
    ):
        print("PASS: binning and depth sort match the host reference exactly")
    else:
        raise Error("FAIL: intersection stage disagrees with the reference")
