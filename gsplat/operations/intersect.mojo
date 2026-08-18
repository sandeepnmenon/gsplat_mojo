"""Tile/gaussian intersection and depth sort.

Produces the two arrays the rasterizer consumes:

  * `flatten_ids[i]`  -- gaussian id for the i-th intersection
  * `tile_offsets[t]` -- where tile t's run begins inside `flatten_ids`

Pipeline, all on device except the two small counts read back to size the
launches:

  1. `project_and_count`  project each gaussian, bound its screen footprint,
                          clip that to the tile grid, count covered tiles
  2. `scan_block` /       two-level exclusive prefix sum over the counts ->
     `scan_block_sums` /  per-gaussian write offset into the flat list
     `add_block_offsets`
  3. `emit_isects`        write one (key, gaussian) pair per covered tile,
                          key = (tile << 32) | float_bits(depth)
  4. `bitonic_step`       sort pairs by key: groups by tile, depth-ordered
                          within a tile, which is exactly the front-to-back
                          order the rasterizer composites in
  5. `write_tile_offsets` run-boundary scan over the sorted keys

The footprint bound in step 1 is the usual EWA projection of the 3D
covariance -- it is used *only* to decide which tiles to visit. Shading still
uses the exact ray/gaussian intersection in `gsplat_forward.mojo`, so the
bound only has to be conservative, never exact.
"""

from std.gpu import block_dim, block_idx, global_idx, thread_idx
from std.math import ceildiv, log, sqrt
from max.gpu.memory import AddressSpace
from max.gpu.sync import barrier
from layout import TileTensor, row_major, stack_allocation

from config import (
    C,
    DTYPE,
    IDTYPE,
    KDTYPE,
    MIN_ALPHA,
    N_MAX,
    SCAN_BLOCK,
    SCAN_NUM_BLOCKS,
    SCAN_STEPS,
    SCAN_WIDTH,
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
from utils_t import Mat3, matmul3x3, quat_to_rotmat, transpose
from vec import Vec3, Vec4

# Sentinel key for bitonic padding: sorts after every real key.
comptime KEY_MAX = UInt64(0xFFFFFFFFFFFFFFFF)

# Dilation added to the projected 2D covariance, as in 3DGS. It only grows
# the bound, so culling stays conservative.
comptime SPLAT_DILATION: Float32 = 0.3


def _cutoff_sigma(opacity: Float32) -> Float32:
    """How many sigmas out a gaussian can still reach MIN_ALPHA.

    alpha = opacity * exp(-rho^2 / 2) >= MIN_ALPHA  <=>  rho <= this.
    Tying the bound to MIN_ALPHA means the culling agrees with the test the
    rasterizer actually applies, instead of a hard-coded 3 sigma.
    """
    var ratio = opacity / Float32(MIN_ALPHA)
    if ratio <= 1.0:
        return 0.0
    return sqrt(2.0 * log(ratio))


def project_and_count(
    means: TileTensor[DTYPE, type_of(layout_n3), MutAnyOrigin],
    scales: TileTensor[DTYPE, type_of(layout_n3), MutAnyOrigin],
    quats: TileTensor[DTYPE, type_of(layout_n4), MutAnyOrigin],
    opacities: TileTensor[DTYPE, type_of(layout_cn), MutAnyOrigin],
    viewmats: TileTensor[DTYPE, type_of(layout_viewmats), MutAnyOrigin],
    Ks: TileTensor[DTYPE, type_of(layout_intrinsics), MutAnyOrigin],
    counts: TileTensor[IDTYPE, type_of(layout_cn_i), MutAnyOrigin],
    bboxes: TileTensor[IDTYPE, type_of(layout_cn4), MutAnyOrigin],
    depths: TileTensor[DTYPE, type_of(layout_cn_f), MutAnyOrigin],
    n_gaussians: Int32,
    n_tiles_x: Int32,
    n_tiles_y: Int32,
    tile_size: Int32,
):
    comptime assert means.flat_rank == 2
    comptime assert bboxes.flat_rank == 3
    comptime assert counts.flat_rank == 2
    comptime assert Ks.flat_rank == 3

    var g = Int(global_idx.x)
    var cid = Int(block_idx.y)
    if g >= Int(n_gaussians):
        return

    var ntx = Int(n_tiles_x)
    var nty = Int(n_tiles_y)
    var tsize = Int(tile_size)

    counts[cid, g] = 0
    bboxes[cid, g, 0] = 0
    bboxes[cid, g, 1] = 0
    bboxes[cid, g, 2] = 0
    bboxes[cid, g, 3] = 0
    depths[cid, g] = 0.0

    # World -> camera.
    var rv = Mat3()
    comptime for r in range(3):
        comptime for cc in range(3):
            rv[r, cc] = rebind[Scalar[DTYPE]](viewmats[cid, r, cc])
    var tv = Vec3(
        rebind[Scalar[DTYPE]](viewmats[cid, 0, 3]),
        rebind[Scalar[DTYPE]](viewmats[cid, 1, 3]),
        rebind[Scalar[DTYPE]](viewmats[cid, 2, 3]),
    )
    var mean = Vec3(
        rebind[Scalar[DTYPE]](means[g, 0]),
        rebind[Scalar[DTYPE]](means[g, 1]),
        rebind[Scalar[DTYPE]](means[g, 2]),
    )
    var pc = rv * mean + tv
    var z = pc[2]
    if z < Z_NEAR:
        return  # behind the camera or too close: contributes nothing

    var opacity = rebind[Scalar[DTYPE]](opacities[cid, g])
    var kappa = _cutoff_sigma(opacity)
    if kappa <= 0.0:
        return  # too faint to reach MIN_ALPHA anywhere

    var fx = rebind[Scalar[DTYPE]](Ks[cid, 0, 0])
    var fy = rebind[Scalar[DTYPE]](Ks[cid, 1, 1])
    var pcx = rebind[Scalar[DTYPE]](Ks[cid, 0, 2])
    var pcy = rebind[Scalar[DTYPE]](Ks[cid, 1, 2])

    # World covariance Sigma = R S^2 R^T, then rotated into camera space.
    var quat = Vec4(
        rebind[Scalar[DTYPE]](quats[g, 0]),
        rebind[Scalar[DTYPE]](quats[g, 1]),
        rebind[Scalar[DTYPE]](quats[g, 2]),
        rebind[Scalar[DTYPE]](quats[g, 3]),
    )
    var rot = quat_to_rotmat(quat)
    var s0 = rebind[Scalar[DTYPE]](scales[g, 0])
    var s1 = rebind[Scalar[DTYPE]](scales[g, 1])
    var s2 = rebind[Scalar[DTYPE]](scales[g, 2])
    var ssq = Mat3.diagonal(s0 * s0, s1 * s1, s2 * s2)
    var cov_world = matmul3x3(matmul3x3(rot, ssq), transpose(rot))
    var cov_cam = matmul3x3(matmul3x3(rv, cov_world), transpose(rv))

    # Perspective Jacobian at pc, as a 2x3.
    var inv_z = 1.0 / z
    var inv_z2 = inv_z * inv_z
    var j00 = fx * inv_z
    var j02 = -fx * pc[0] * inv_z2
    var j11 = fy * inv_z
    var j12 = -fy * pc[1] * inv_z2

    # Sigma_2D = J Sigma_cam J^T.
    var jr0 = Vec3(j00, 0.0, j02)
    var jr1 = Vec3(0.0, j11, j12)
    var t0 = cov_cam * jr0  # Sigma_cam is symmetric, so this is J row times it
    var t1 = cov_cam * jr1
    var a = jr0.dot(t0) + SPLAT_DILATION
    var b = jr0.dot(t1)
    var c2 = jr1.dot(t1) + SPLAT_DILATION

    # Largest eigenvalue of [[a, b], [b, c2]].
    var mid = 0.5 * (a + c2)
    var det = a * c2 - b * b
    var disc = mid * mid - det
    if disc < 0.0:
        disc = 0.0
    var lambda_max = mid + sqrt(disc)
    if lambda_max <= 0.0:
        return
    var radius = kappa * sqrt(lambda_max)

    var sx = fx * pc[0] * inv_z + pcx
    var sy = fy * pc[1] * inv_z + pcy

    # Half-open tile bounds, clipped to the grid.
    var tx0 = Int((sx - radius) / Float32(tsize))
    var tx1 = Int(ceildiv(sx + radius, Float32(tsize)))
    var ty0 = Int((sy - radius) / Float32(tsize))
    var ty1 = Int(ceildiv(sy + radius, Float32(tsize)))
    if tx0 < 0:
        tx0 = 0
    if ty0 < 0:
        ty0 = 0
    if tx1 > ntx:
        tx1 = ntx
    if ty1 > nty:
        ty1 = nty
    if tx1 <= tx0 or ty1 <= ty0:
        return  # entirely off screen

    counts[cid, g] = Int32((tx1 - tx0) * (ty1 - ty0))
    bboxes[cid, g, 0] = Int32(tx0)
    bboxes[cid, g, 1] = Int32(ty0)
    bboxes[cid, g, 2] = Int32(tx1)
    bboxes[cid, g, 3] = Int32(ty1)
    depths[cid, g] = z


def scan_block(
    counts: TileTensor[IDTYPE, type_of(layout_cn_flat), MutAnyOrigin],
    offsets: TileTensor[IDTYPE, type_of(layout_cn_flat), MutAnyOrigin],
    block_sums: TileTensor[IDTYPE, type_of(layout_blocksums), MutAnyOrigin],
    n_items: Int32,
):
    """Level 1 of the scan: exclusive prefix sum within each block.

    Also records the block's own total so level 2 can offset it. `counts` must
    be zero-filled past the live gaussian count, since the scan always covers
    the full capacity.
    """
    comptime assert counts.flat_rank == 1
    var tid = Int(thread_idx.x)
    var gid = Int(block_idx.x) * SCAN_BLOCK + tid
    var n = Int(n_items)

    var sh = stack_allocation[IDTYPE, address_space = AddressSpace.SHARED](
        row_major[SCAN_BLOCK]()
    )
    var mine: Int32 = 0
    if gid < n:
        mine = rebind[Scalar[IDTYPE]](counts[gid])
    sh[tid] = mine
    barrier()

    comptime for step in range(SCAN_STEPS):
        comptime off = 1 << step
        var acc = rebind[Scalar[IDTYPE]](sh[tid])
        if tid >= off:
            acc += rebind[Scalar[IDTYPE]](sh[tid - off])
        barrier()
        sh[tid] = acc
        barrier()

    if gid < n:
        offsets[gid] = rebind[Scalar[IDTYPE]](sh[tid]) - mine
    if tid == SCAN_BLOCK - 1:
        block_sums[Int(block_idx.x)] = rebind[Scalar[IDTYPE]](sh[tid])


def scan_block_sums(
    block_sums: TileTensor[IDTYPE, type_of(layout_blocksums), MutAnyOrigin],
    total: TileTensor[IDTYPE, type_of(layout_one), MutAnyOrigin],
    n_blocks: Int32,
):
    """Level 2: scan the per-block totals in a single block, in place.

    This is what bounds the whole scan at SCAN_BLOCK * SCAN_WIDTH elements --
    beyond that this level would itself need to be multi-block.
    """
    var tid = Int(thread_idx.x)
    var nb = Int(n_blocks)

    var sh = stack_allocation[IDTYPE, address_space = AddressSpace.SHARED](
        row_major[SCAN_WIDTH]()
    )
    var mine: Int32 = 0
    if tid < nb:
        mine = rebind[Scalar[IDTYPE]](block_sums[tid])
    sh[tid] = mine
    barrier()

    comptime for step in range(SCAN_STEPS):
        comptime off = 1 << step
        var acc = rebind[Scalar[IDTYPE]](sh[tid])
        if tid >= off:
            acc += rebind[Scalar[IDTYPE]](sh[tid - off])
        barrier()
        sh[tid] = acc
        barrier()

    if tid < nb:
        block_sums[tid] = rebind[Scalar[IDTYPE]](sh[tid]) - mine
    if tid == 0:
        # Entries past nb were zeroed, so the last slot is the grand total.
        total[0] = rebind[Scalar[IDTYPE]](sh[SCAN_WIDTH - 1])


def add_block_offsets(
    offsets: TileTensor[IDTYPE, type_of(layout_cn_flat), MutAnyOrigin],
    block_sums: TileTensor[IDTYPE, type_of(layout_blocksums), MutAnyOrigin],
    n_items: Int32,
):
    """Level 3: fold each block's base offset into its elements."""
    var gid = Int(global_idx.x)
    if gid >= Int(n_items):
        return
    offsets[gid] = rebind[Scalar[IDTYPE]](offsets[gid]) + rebind[
        Scalar[IDTYPE]
    ](block_sums[gid // SCAN_BLOCK])


def emit_isects(
    bboxes: TileTensor[IDTYPE, type_of(layout_cn4), MutAnyOrigin],
    depths: TileTensor[DTYPE, type_of(layout_cn_f), MutAnyOrigin],
    offsets: TileTensor[IDTYPE, type_of(layout_cn_i), MutAnyOrigin],
    counts: TileTensor[IDTYPE, type_of(layout_cn_i), MutAnyOrigin],
    keys: TileTensor[KDTYPE, type_of(layout_isects), MutAnyOrigin],
    vals: TileTensor[IDTYPE, type_of(layout_isects), MutAnyOrigin],
    n_gaussians: Int32,
    n_tiles_x: Int32,
    n_tiles_total: Int32,
):
    var g = Int(global_idx.x)
    var cid = Int(block_idx.y)
    if g >= Int(n_gaussians):
        return
    var count = Int(rebind[Scalar[IDTYPE]](counts[cid, g]))
    if count == 0:
        return

    var base = Int(rebind[Scalar[IDTYPE]](offsets[cid, g]))
    var tx0 = Int(rebind[Scalar[IDTYPE]](bboxes[cid, g, 0]))
    var ty0 = Int(rebind[Scalar[IDTYPE]](bboxes[cid, g, 1]))
    var tx1 = Int(rebind[Scalar[IDTYPE]](bboxes[cid, g, 2]))
    var ty1 = Int(rebind[Scalar[IDTYPE]](bboxes[cid, g, 3]))
    var ntx = Int(n_tiles_x)

    # Camera-space depth as sortable bits. Depth is > 0 here (Z_NEAR clipped),
    # and for positive IEEE floats the bit pattern orders the same as the
    # value, so the raw bits can go straight into the low half of the key.
    var depth_bits = UInt64(
        rebind[Scalar[DTYPE]](depths[cid, g]).to_bits[DType.uint32]()
    )

    var w = 0
    for ty in range(ty0, ty1):
        for tx in range(tx0, tx1):
            var tile = cid * Int(n_tiles_total) + ty * ntx + tx
            keys[base + w] = (UInt64(tile) << 32) | depth_bits
            vals[base + w] = Int32(g)
            w += 1


def bitonic_step(
    keys: TileTensor[KDTYPE, type_of(layout_isects), MutAnyOrigin],
    vals: TileTensor[IDTYPE, type_of(layout_isects), MutAnyOrigin],
    n_pow2: Int32,
    k: Int32,
    j: Int32,
):
    """One compare-exchange stage of a bitonic sort over `keys`/`vals`."""
    var i = Int(global_idx.x)
    if i >= Int(n_pow2):
        return
    var partner = i ^ Int(j)
    if partner <= i:
        return  # let the lower index of each pair do the swap

    var ascending = (i & Int(k)) == 0
    var ki = rebind[Scalar[KDTYPE]](keys[i])
    var kp = rebind[Scalar[KDTYPE]](keys[partner])
    if (ki > kp) == ascending:
        keys[i] = kp
        keys[partner] = ki
        var vi = rebind[Scalar[IDTYPE]](vals[i])
        vals[i] = rebind[Scalar[IDTYPE]](vals[partner])
        vals[partner] = vi


def write_tile_offsets(
    keys: TileTensor[KDTYPE, type_of(layout_isects), MutAnyOrigin],
    tile_offsets: TileTensor[IDTYPE, type_of(layout_tiles_flat), MutAnyOrigin],
    n_isects: Int32,
):
    """Mark where each tile's run starts in the sorted intersection list.

    Every tile between the previous key's tile and this one is empty, so it
    gets the same start index -- which makes its [start, end) range empty in
    the rasterizer. Tiles past the final key keep the `n_isects` fill applied
    before launch.
    """
    var i = Int(global_idx.x)
    if i >= Int(n_isects):
        return
    var cur = Int(rebind[Scalar[KDTYPE]](keys[i]) >> 32)
    if i == 0:
        for t in range(0, cur + 1):
            tile_offsets[t] = 0
    else:
        var prev = Int(rebind[Scalar[KDTYPE]](keys[i - 1]) >> 32)
        if prev != cur:
            for t in range(prev + 1, cur + 1):
                tile_offsets[t] = Int32(i)
