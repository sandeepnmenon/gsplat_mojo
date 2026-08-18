"""Self-checking render of the forward rasterizer, in three phases.

  1. isotropic gaussians on the optical axis, identity camera, every tile
     given every gaussian -- checked against a closed form
  2. rotated / anisotropic / off-axis gaussians and a moved camera, still
     every tile given every gaussian -- checked against `_ref_rho2`, a scalar
     reference written from the definitions
  3. the same scene as 2, but with tiles and depth order computed by the real
     intersection stage -- checked against the phase 2 image, which is the
     brute-force answer

Phase 1 pins the maths, phase 2 covers the rotation/anisotropy/pose paths that
phase 1 leaves on the identity path, and phase 3 shows the culling drops only
gaussians that could not have changed the image.
"""

from std.math import ceildiv, exp, sqrt
from std.sys import has_accelerator
from max.gpu.host import DeviceContext
from layout import TileTensor

from gsplat_kernels.config import (
    C,
    CDIM,
    CX,
    CY,
    DTYPE,
    FOCAL,
    IDTYPE,
    IMG_H,
    IMG_W,
    KDTYPE,
    MAX_ALPHA,
    MIN_ALPHA,
    N_MAX,
    N_TEST,
    N_TILES,
    N_TILES_X,
    N_TILES_Y,
    RADIX,
    RADIX_EPB,
    RADIX_PASSES,
    SCAN_BLOCK,
    SCAN_NUM_BLOCKS,
    SCAN_WIDTH,
    T_EPS,
    TILE,
    layout_blocksums,
    layout_cdim,
    layout_cn,
    layout_cn4,
    layout_cn_cdim,
    layout_cn_f,
    layout_cn_flat,
    layout_cn_i,
    layout_c2,
    layout_c6,
    layout_intrinsics,
    layout_isects,
    layout_last_ids,
    layout_n3,
    layout_n4,
    layout_one,
    layout_render_alphas,
    layout_render_colors,
    layout_tiles,
    layout_tiles_flat,
    layout_viewmats,
)
from gsplat_kernels.intersect import (
    KEY_MAX,
    add_block_offsets,
    emit_isects,
    project_and_count,
    radix_sort_pairs,
    scan_block,
    scan_block_sums,
    write_tile_offsets,
)
from gsplat_kernels.rasterize import rasterize_to_pixels_from_world_3dgs_fwd
from refmath import _ref_rho2
from scene import (
    axis_opacity,
    axis_scale,
    axis_z,
    gauss_color,
    spread_mean,
    spread_opacity,
    spread_quat,
    spread_quat_norm,
    spread_scale,
    view_rot,
    view_trans,
)

comptime BG = SIMD[DTYPE, 4](0.1, 0.15, 0.2, 0.0)
comptime MASKED_TILE = 0  # one tile forced invisible, to cover the mask path
comptime TOL: Float32 = 2e-4  # GPU vs host exp() disagree in the last bits

def main() raises:
    comptime assert has_accelerator(), "gsplat forward pass requires a GPU"
    comptime assert (
        C * N_MAX <= SCAN_BLOCK * SCAN_WIDTH
    ), "two-level scan cannot cover C*N_MAX"

    var n_gauss = N_TEST
    # Brute-force binning for phases 1 and 2: every tile handed every gaussian.
    var n_isects_full = C * N_TILES * n_gauss
    var ctx = DeviceContext()

    # ---- buffers ---------------------------------------------------------
    var means_buf = ctx.enqueue_create_buffer[DTYPE](N_MAX * 3)
    var quats_buf = ctx.enqueue_create_buffer[DTYPE](N_MAX * 4)
    var scales_buf = ctx.enqueue_create_buffer[DTYPE](N_MAX * 3)
    var colors_buf = ctx.enqueue_create_buffer[DTYPE](C * N_MAX * CDIM)
    var opac_buf = ctx.enqueue_create_buffer[DTYPE](C * N_MAX)
    var bg_buf = ctx.enqueue_create_buffer[DTYPE](C * CDIM)
    var masks_buf = ctx.enqueue_create_buffer[IDTYPE](C * N_TILES)
    var view0_buf = ctx.enqueue_create_buffer[DTYPE](C * 16)
    var view1_buf = ctx.enqueue_create_buffer[DTYPE](C * 16)
    var ks_buf = ctx.enqueue_create_buffer[DTYPE](C * 9)
    var radial_buf = ctx.enqueue_create_buffer[DTYPE](C * 6)
    var tangential_buf = ctx.enqueue_create_buffer[DTYPE](C * 2)
    var thin_buf = ctx.enqueue_create_buffer[DTYPE](C * 2)
    var tileoff_buf = ctx.enqueue_create_buffer[IDTYPE](C * N_TILES)
    var flat_buf = ctx.enqueue_create_buffer[IDTYPE](C * N_TILES * N_TEST)
    var renders_buf = ctx.enqueue_create_buffer[DTYPE](C * IMG_H * IMG_W * CDIM)
    var alphas_buf = ctx.enqueue_create_buffer[DTYPE](C * IMG_H * IMG_W)
    var ids_buf = ctx.enqueue_create_buffer[IDTYPE](C * IMG_H * IMG_W)
    # intersection-stage scratch
    var counts_buf = ctx.enqueue_create_buffer[IDTYPE](C * N_MAX)
    var offsets_buf = ctx.enqueue_create_buffer[IDTYPE](C * N_MAX)
    var bbox_buf = ctx.enqueue_create_buffer[IDTYPE](C * N_MAX * 4)
    var depth_buf = ctx.enqueue_create_buffer[DTYPE](C * N_MAX)
    var total_buf = ctx.enqueue_create_buffer[IDTYPE](1)
    var blocksum_buf = ctx.enqueue_create_buffer[IDTYPE](SCAN_NUM_BLOCKS)

    radial_buf.enqueue_fill(0.0)
    tangential_buf.enqueue_fill(0.0)
    thin_buf.enqueue_fill(0.0)
    view1_buf.enqueue_fill(0.0)

    # ---- host staging ----------------------------------------------------
    var means_h = ctx.enqueue_create_host_buffer[DTYPE](N_MAX * 3)
    var quats_h = ctx.enqueue_create_host_buffer[DTYPE](N_MAX * 4)
    var scales_h = ctx.enqueue_create_host_buffer[DTYPE](N_MAX * 3)
    var colors_h = ctx.enqueue_create_host_buffer[DTYPE](C * N_MAX * CDIM)
    var opac_h = ctx.enqueue_create_host_buffer[DTYPE](C * N_MAX)
    var bg_h = ctx.enqueue_create_host_buffer[DTYPE](C * CDIM)
    var view_h = ctx.enqueue_create_host_buffer[DTYPE](C * 16)
    var ks_h = ctx.enqueue_create_host_buffer[DTYPE](C * 9)
    var masks_h = ctx.enqueue_create_host_buffer[IDTYPE](C * N_TILES)
    var tileoff_h = ctx.enqueue_create_host_buffer[IDTYPE](C * N_TILES)
    var flat_h = ctx.enqueue_create_host_buffer[IDTYPE](C * N_TILES * N_TEST)
    var total_h = ctx.enqueue_create_host_buffer[IDTYPE](1)
    # phase 2's image, kept to compare phase 3 against
    var keep_c = ctx.enqueue_create_host_buffer[DTYPE](C * IMG_H * IMG_W * CDIM)
    var keep_a = ctx.enqueue_create_host_buffer[DTYPE](C * IMG_H * IMG_W)
    ctx.synchronize()

    comptime for k in range(CDIM):
        bg_h[k] = BG[k]
    for c in range(C):
        comptime for k in range(CDIM):
            bg_h[c * CDIM + k] = BG[k]
        for g in range(n_gauss):
            comptime for k in range(CDIM):
                colors_h[(c * N_MAX + g) * CDIM + k] = gauss_color(g, k)
        for e in range(9):
            ks_h[c * 9 + e] = 0.0
        ks_h[c * 9 + 0] = FOCAL
        ks_h[c * 9 + 4] = FOCAL
        ks_h[c * 9 + 2] = CX
        ks_h[c * 9 + 5] = CY
        ks_h[c * 9 + 8] = 1.0
    ctx.enqueue_copy(dst_buf=colors_buf, src_buf=colors_h)
    ctx.enqueue_copy(dst_buf=bg_buf, src_buf=bg_h)
    ctx.enqueue_copy(dst_buf=ks_buf, src_buf=ks_h)

    # ---- tensor views ----------------------------------------------------
    var means = TileTensor(means_buf, layout_n3)
    var quats = TileTensor(quats_buf, layout_n4)
    var scales = TileTensor(scales_buf, layout_n3)
    var colors = TileTensor(colors_buf, layout_cn_cdim)
    var opacities = TileTensor(opac_buf, layout_cn)
    var backgrounds = TileTensor(bg_buf, layout_cdim)
    var masks = TileTensor(masks_buf, layout_tiles)
    var viewmats0 = TileTensor(view0_buf, layout_viewmats)
    var viewmats1 = TileTensor(view1_buf, layout_viewmats)
    var ks = TileTensor(ks_buf, layout_intrinsics)
    var radial = TileTensor(radial_buf, layout_c6)
    var tangential = TileTensor(tangential_buf, layout_c2)
    var thin_prims = TileTensor(thin_buf, layout_c2)
    var tile_offsets = TileTensor(tileoff_buf, layout_tiles)
    var tile_offsets_flat = TileTensor(tileoff_buf, layout_tiles_flat)
    var flatten_ids = TileTensor(flat_buf, layout_isects)
    var render_colors = TileTensor(renders_buf, layout_render_colors)
    var render_alphas = TileTensor(alphas_buf, layout_render_alphas)
    var last_ids = TileTensor(ids_buf, layout_last_ids)
    var counts_flat = TileTensor(counts_buf, layout_cn_flat)
    var offsets_flat = TileTensor(offsets_buf, layout_cn_flat)
    var block_sums = TileTensor(blocksum_buf, layout_blocksums)
    var counts = TileTensor(counts_buf, layout_cn_i)
    var offsets = TileTensor(offsets_buf, layout_cn_i)
    var bboxes = TileTensor(bbox_buf, layout_cn4)
    var depths = TileTensor(depth_buf, layout_cn_f)
    var total = TileTensor(total_buf, layout_one)

    comptime TPB = 256
    var gblocks = ceildiv(n_gauss, TPB)

    print(
        "render", IMG_W, "x", IMG_H, "|", n_gauss, "gaussians |",
        N_TILES_X, "x", N_TILES_Y, "tiles",
    )

    # =====================================================================
    # Phase 1 — isotropic, on-axis, identity camera. Closed form applies.
    # =====================================================================
    for g in range(n_gauss):
        means_h[g * 3 + 0] = 0.0
        means_h[g * 3 + 1] = 0.0
        means_h[g * 3 + 2] = axis_z(g)
        quats_h[g * 4 + 0] = 0.0
        quats_h[g * 4 + 1] = 0.0
        quats_h[g * 4 + 2] = 0.0
        quats_h[g * 4 + 3] = 1.0
        comptime for a in range(3):
            scales_h[g * 3 + a] = axis_scale(g)
    for c in range(C):
        for g in range(n_gauss):
            opac_h[c * N_MAX + g] = axis_opacity(g)
        for e in range(16):
            view_h[c * 16 + e] = 0.0
        view_h[c * 16 + 0] = 1.0
        view_h[c * 16 + 5] = 1.0
        view_h[c * 16 + 10] = 1.0
        view_h[c * 16 + 15] = 1.0
    # brute-force binning: every tile gets every gaussian, in index order,
    # which for this scene is already front-to-back
    for f in range(C * N_TILES):
        tileoff_h[f] = Int32(f * n_gauss)
        masks_h[f] = 1
        for g in range(n_gauss):
            flat_h[f * n_gauss + g] = Int32(g)
    masks_h[MASKED_TILE] = 0

    ctx.enqueue_copy(dst_buf=means_buf, src_buf=means_h)
    ctx.enqueue_copy(dst_buf=quats_buf, src_buf=quats_h)
    ctx.enqueue_copy(dst_buf=scales_buf, src_buf=scales_h)
    ctx.enqueue_copy(dst_buf=opac_buf, src_buf=opac_h)
    ctx.enqueue_copy(dst_buf=view0_buf, src_buf=view_h)
    ctx.enqueue_copy(dst_buf=masks_buf, src_buf=masks_h)
    ctx.enqueue_copy(dst_buf=tileoff_buf, src_buf=tileoff_h)
    ctx.enqueue_copy(dst_buf=flat_buf, src_buf=flat_h)
    renders_buf.enqueue_fill(-1.0)
    alphas_buf.enqueue_fill(-1.0)
    ids_buf.enqueue_fill(-1)

    ctx.enqueue_function[rasterize_to_pixels_from_world_3dgs_fwd](
        Int32(C), Int32(n_gauss), Int32(n_isects_full), Int32(0),
        means, quats, scales, colors, opacities, backgrounds, masks,
        Int32(1), Int32(1),
        Int32(IMG_W), Int32(IMG_H), Int32(TILE),
        Int32(N_TILES_X), Int32(N_TILES_Y),
        viewmats0, viewmats1, ks,
        Int32(0), Int32(0),
        radial, tangential, thin_prims,
        tile_offsets, flatten_ids,
        render_colors, render_alphas, last_ids,
        grid_dim=(N_TILES_X, N_TILES_Y, C), block_dim=(TILE, TILE, 1),
    )
    ctx.synchronize()

    var worst_c1: Float32 = 0.0
    var worst_a1: Float32 = 0.0
    var bad1 = 0
    var lit1 = 0
    var n_sat = 0
    with renders_buf.map_to_host() as gc:
        with alphas_buf.map_to_host() as ga:
            for c in range(C):
                for y in range(IMG_H):
                    for x in range(IMG_W):
                        var exp_a: Float32
                        var exp_c = SIMD[DTYPE, 4](0.0)
                        if ((y // TILE) * N_TILES_X + (x // TILE)) == MASKED_TILE:
                            exp_a = 0.0
                            comptime for k in range(CDIM):
                                exp_c[k] = BG[k]
                        else:
                            var u = (Float32(x) + 0.5 - CX) / FOCAL
                            var v = (Float32(y) + 0.5 - CY) / FOCAL
                            var w = u * u + v * v
                            var tr: Float32 = 1.0
                            for g in range(n_gauss):
                                var s = axis_scale(g)
                                var z = axis_z(g)
                                var rho2 = z * z * w / ((1.0 + w) * s * s)
                                var a = min(
                                    Float32(MAX_ALPHA),
                                    axis_opacity(g) * exp(-0.5 * rho2),
                                )
                                if a < MIN_ALPHA:
                                    continue
                                var nt = tr * (1.0 - a)
                                if nt < T_EPS:
                                    n_sat += 1
                                    break
                                comptime for k in range(CDIM):
                                    exp_c[k] += a * tr * gauss_color(g, k)
                                tr = nt
                            exp_a = 1.0 - tr
                            comptime for k in range(CDIM):
                                exp_c[k] += tr * BG[k]
                        if exp_a > MIN_ALPHA:
                            lit1 += 1
                        var pix = (c * IMG_H + y) * IMG_W + x
                        var miss = False
                        comptime for k in range(CDIM):
                            var d = abs(gc[pix * CDIM + k] - exp_c[k])
                            if d > worst_c1:
                                worst_c1 = d
                            if d > TOL:
                                miss = True
                        var da = abs(ga[pix] - exp_a)
                        if da > worst_a1:
                            worst_a1 = da
                        if da > TOL:
                            miss = True
                        if miss:
                            bad1 += 1
    print(
        "phase 1  closed form   | lit", lit1,
        "| max err color", worst_c1, "alpha", worst_a1,
        "| bad", bad1,
    )
    var ok1 = bad1 == 0 and lit1 > 0

    # =====================================================================
    # Phase 2 — rotated / anisotropic / off-axis, camera moved. Brute-force
    # binning again, but ordered by true camera depth so phase 3 (which sorts
    # by depth) composites the same gaussians in the same relative order.
    # =====================================================================
    var cam_z = List[Float32]()
    for g in range(n_gauss):
        var zz: Float32 = 0.0
        comptime for a in range(3):
            zz += view_rot(2, a) * spread_mean(g, a, n_gauss)
        cam_z.append(zz + view_trans(2))
    var order = List[Int]()
    for g in range(n_gauss):
        order.append(g)
    for i in range(1, n_gauss):  # insertion sort, n_gauss is tiny
        var j = i
        while j > 0 and cam_z[order[j - 1]] > cam_z[order[j]]:
            var tmp = order[j - 1]
            order[j - 1] = order[j]
            order[j] = tmp
            j -= 1

    for g in range(n_gauss):
        var qn = spread_quat_norm(g)
        comptime for a in range(3):
            means_h[g * 3 + a] = spread_mean(g, a, n_gauss)
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
    for f in range(C * N_TILES):
        tileoff_h[f] = Int32(f * n_gauss)
        masks_h[f] = 1
        for i in range(n_gauss):
            flat_h[f * n_gauss + i] = Int32(order[i])

    ctx.enqueue_copy(dst_buf=means_buf, src_buf=means_h)
    ctx.enqueue_copy(dst_buf=quats_buf, src_buf=quats_h)
    ctx.enqueue_copy(dst_buf=scales_buf, src_buf=scales_h)
    ctx.enqueue_copy(dst_buf=opac_buf, src_buf=opac_h)
    ctx.enqueue_copy(dst_buf=view0_buf, src_buf=view_h)
    ctx.enqueue_copy(dst_buf=masks_buf, src_buf=masks_h)
    ctx.enqueue_copy(dst_buf=tileoff_buf, src_buf=tileoff_h)
    ctx.enqueue_copy(dst_buf=flat_buf, src_buf=flat_h)
    renders_buf.enqueue_fill(-1.0)
    alphas_buf.enqueue_fill(-1.0)

    ctx.enqueue_function[rasterize_to_pixels_from_world_3dgs_fwd](
        Int32(C), Int32(n_gauss), Int32(n_isects_full), Int32(0),
        means, quats, scales, colors, opacities, backgrounds, masks,
        Int32(1), Int32(1),
        Int32(IMG_W), Int32(IMG_H), Int32(TILE),
        Int32(N_TILES_X), Int32(N_TILES_Y),
        viewmats0, viewmats1, ks,
        Int32(0), Int32(0),
        radial, tangential, thin_prims,
        tile_offsets, flatten_ids,
        render_colors, render_alphas, last_ids,
        grid_dim=(N_TILES_X, N_TILES_Y, C), block_dim=(TILE, TILE, 1),
    )
    ctx.enqueue_copy(dst_buf=keep_c, src_buf=renders_buf)
    ctx.enqueue_copy(dst_buf=keep_a, src_buf=alphas_buf)
    ctx.synchronize()

    var cam_x = -(view_rot(0, 0) * view_trans(0) + view_rot(1, 0) * view_trans(1) + view_rot(2, 0) * view_trans(2))
    var cam_y = -(view_rot(0, 1) * view_trans(0) + view_rot(1, 1) * view_trans(1) + view_rot(2, 1) * view_trans(2))
    var cam_zc = -(view_rot(0, 2) * view_trans(0) + view_rot(1, 2) * view_trans(1) + view_rot(2, 2) * view_trans(2))

    var worst_c2: Float32 = 0.0
    var worst_a2: Float32 = 0.0
    var bad2 = 0
    var lit2 = 0
    for c in range(C):
        for y in range(IMG_H):
            for x in range(IMG_W):
                var u = (Float32(x) + 0.5 - CX) / FOCAL
                var v = (Float32(y) + 0.5 - CY) / FOCAL
                var dwx = view_rot(0, 0) * u + view_rot(1, 0) * v + view_rot(2, 0)
                var dwy = view_rot(0, 1) * u + view_rot(1, 1) * v + view_rot(2, 1)
                var dwz = view_rot(0, 2) * u + view_rot(1, 2) * v + view_rot(2, 2)
                var tr: Float32 = 1.0
                var acc = SIMD[DTYPE, 4](0.0)
                for oi in range(n_gauss):
                    var g = order[oi]
                    var qn = spread_quat_norm(g)
                    var res = _ref_rho2(
                        spread_mean(g, 0, n_gauss), spread_mean(g, 1, n_gauss), spread_mean(g, 2, n_gauss),
                        spread_quat(g, 0) / qn, spread_quat(g, 1) / qn,
                        spread_quat(g, 2) / qn, spread_quat(g, 3) / qn,
                        spread_scale(g, 0), spread_scale(g, 1), spread_scale(g, 2),
                        cam_x, cam_y, cam_zc,
                        dwx, dwy, dwz,
                    )
                    if res[1] <= 0.0:
                        continue
                    var a = min(
                        Float32(MAX_ALPHA), spread_opacity(g) * exp(-0.5 * res[0])
                    )
                    if a < MIN_ALPHA:
                        continue
                    var nt = tr * (1.0 - a)
                    if nt < T_EPS:
                        break
                    comptime for k in range(CDIM):
                        acc[k] += a * tr * gauss_color(g, k)
                    tr = nt
                var exp_a = 1.0 - tr
                if exp_a > MIN_ALPHA:
                    lit2 += 1
                var pix = (c * IMG_H + y) * IMG_W + x
                var miss = False
                comptime for k in range(CDIM):
                    var want = acc[k] + tr * BG[k]
                    var d = abs(keep_c[pix * CDIM + k] - want)
                    if d > worst_c2:
                        worst_c2 = d
                    if d > TOL:
                        miss = True
                var da = abs(keep_a[pix] - exp_a)
                if da > worst_a2:
                    worst_a2 = da
                if da > TOL:
                    miss = True
                if miss:
                    bad2 += 1
    print(
        "phase 2  scalar ref    | lit", lit2,
        "| max err color", worst_c2, "alpha", worst_a2,
        "| bad", bad2,
    )
    var ok2 = bad2 == 0 and lit2 > 0

    # =====================================================================
    # Phase 3 — same scene, but tiles and depth order from the real
    # intersection stage instead of handing every tile every gaussian.
    # =====================================================================
    counts_buf.enqueue_fill(0)  # the scan covers the whole capacity
    ctx.enqueue_function[project_and_count](
        means, scales, quats, opacities, viewmats0, ks,
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

    # Radix sorts exactly n elements -- no power-of-two padding needed.
    var n_rblocks = ceildiv(n_isects, RADIX_EPB)
    var hist_size = RADIX * n_rblocks
    var keys_buf = ctx.enqueue_create_buffer[KDTYPE](n_isects)
    var sorted_buf = ctx.enqueue_create_buffer[IDTYPE](n_isects)
    var keys_alt_buf = ctx.enqueue_create_buffer[KDTYPE](n_isects)
    var vals_alt_buf = ctx.enqueue_create_buffer[IDTYPE](n_isects)
    var hist_buf = ctx.enqueue_create_buffer[IDTYPE](hist_size)
    var histoff_buf = ctx.enqueue_create_buffer[IDTYPE](hist_size)
    var scratch_buf = ctx.enqueue_create_buffer[IDTYPE](1)
    keys_buf.enqueue_fill(KEY_MAX)
    sorted_buf.enqueue_fill(-1)
    var keys = TileTensor(keys_buf, layout_isects)
    var sorted_ids = TileTensor(sorted_buf, layout_isects)
    var keys_alt = TileTensor(keys_alt_buf, layout_isects)
    var vals_alt = TileTensor(vals_alt_buf, layout_isects)
    var hist = TileTensor(hist_buf, layout_cn_flat)
    var hist_off = TileTensor(histoff_buf, layout_cn_flat)
    var scratch = TileTensor(scratch_buf, layout_one)

    ctx.enqueue_function[emit_isects](
        bboxes, depths, offsets, counts, keys, sorted_ids,
        Int32(n_gauss), Int32(N_TILES_X), Int32(N_TILES),
        grid_dim=(gblocks, C), block_dim=TPB,
    )
    radix_sort_pairs(
        ctx, keys, sorted_ids, keys_alt, vals_alt,
        hist, hist_off, block_sums, scratch,
        keys_buf, sorted_buf, keys_alt_buf, vals_alt_buf, n_isects,
    )
    tileoff_buf.enqueue_fill(Int32(n_isects))
    ctx.enqueue_function[write_tile_offsets](
        keys, tile_offsets_flat, Int32(n_isects),
        grid_dim=ceildiv(n_isects, TPB), block_dim=TPB,
    )
    renders_buf.enqueue_fill(-1.0)
    alphas_buf.enqueue_fill(-1.0)

    ctx.enqueue_function[rasterize_to_pixels_from_world_3dgs_fwd](
        Int32(C), Int32(n_gauss), Int32(n_isects), Int32(0),
        means, quats, scales, colors, opacities, backgrounds, masks,
        Int32(1), Int32(1),
        Int32(IMG_W), Int32(IMG_H), Int32(TILE),
        Int32(N_TILES_X), Int32(N_TILES_Y),
        viewmats0, viewmats1, ks,
        Int32(0), Int32(0),
        radial, tangential, thin_prims,
        tile_offsets, sorted_ids,
        render_colors, render_alphas, last_ids,
        grid_dim=(N_TILES_X, N_TILES_Y, C), block_dim=(TILE, TILE, 1),
    )
    ctx.synchronize()

    var worst_c3: Float32 = 0.0
    var worst_a3: Float32 = 0.0
    var differing = 0
    with renders_buf.map_to_host() as gc:
        with alphas_buf.map_to_host() as ga:
            for i in range(C * IMG_H * IMG_W):
                var miss = False
                comptime for kk in range(CDIM):
                    var d = abs(gc[i * CDIM + kk] - keep_c[i * CDIM + kk])
                    if d > worst_c3:
                        worst_c3 = d
                    if d > TOL:
                        miss = True
                var da = abs(ga[i] - keep_a[i])
                if da > worst_a3:
                    worst_a3 = da
                if da > TOL:
                    miss = True
                if miss:
                    differing += 1

    var work_full = n_isects_full
    print(
        "phase 3  real binning  |", n_isects, "intersections vs", work_full,
        "brute force  (", Int(100.0 * Float32(n_isects) / Float32(work_full)),
        "% )",
    )
    print(
        "         vs phase 2 image | max diff color", worst_c3,
        "alpha", worst_a3, "| pixels differing", differing,
    )
    # A culled gaussian is one the rasterizer would have rejected at MIN_ALPHA
    # anyway, so its weight was under 1/255; anything larger means the bound
    # is dropping gaussians that mattered.
    var ok3 = worst_c3 <= Float32(MIN_ALPHA) and worst_a3 <= Float32(MIN_ALPHA)

    print("tolerance:", TOL, "| culling budget:", Float32(MIN_ALPHA))
    if ok1 and ok2 and ok3:
        print(
            "PASS: rasterizer matches independent references, and real binning"
            " reproduces the brute-force image"
        )
    else:
        raise Error(
            String("FAIL: p1 bad=") + String(bad1)
            + " p2 bad=" + String(bad2)
            + " p3 maxdiff=" + String(worst_c3)
        )
