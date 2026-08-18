"""Tile-based 3D gaussian rasterization — forward pass.

Structure follows gsplat's `rasterize_to_pixels_from_world_3dgs_fwd`: one
thread block per (camera, tile), gaussians for the tile streamed through
shared memory in block-sized batches, then alpha-composited front-to-back.

Each gaussian is evaluated by ray/gaussian intersection rather than by an EWA
2D-splat projection: `iscl_rot` = S^-1 R^T maps world space into the frame
where the gaussian is the unit sphere, the ray's closest approach to the
origin in that frame gives rho^2, and the response is exp(-rho^2 / 2).
Contributions are composited in `flatten_ids` order, so that list must be
depth-sorted per tile -- which is what `operations/intersect.mojo` produces.

`main()` is a self-checking render in three phases:

  1. isotropic gaussians on the optical axis, identity camera, every tile
     given every gaussian -- checked against a closed form
  2. rotated / anisotropic / off-axis gaussians and a moved camera, still
     every tile given every gaussian -- checked against `_ref_rho2`, a scalar
     reference written from the definitions
  3. the same scene as 2, but with tiles and depth order computed by the real
     intersection stage -- checked against the phase 2 image, which is the
     brute-force answer

Phase 1 pins the maths, phase 2 covers the rotation/anisotropy/pose paths
phase 1 leaves on the identity path, and phase 3 shows the culling drops only
gaussians that could not have changed the image.
"""

from std.math import ceildiv, exp, sqrt
from std.gpu import block_dim, block_idx, thread_idx
from std.sys import has_accelerator
from max.gpu.host import DeviceContext
from max.gpu.memory import AddressSpace
from max.gpu.sync import barrier
from layout import TileTensor, row_major, stack_allocation

from config import (
    BLOCK_SIZE,
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
    T_EPS,
    TILE,
    layout_blocksums,
    layout_cdim,
    layout_cn,
    layout_cn_flat,
    layout_cn4,
    layout_cn_cdim,
    layout_cn_f,
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
from operations.intersect import (
    KEY_MAX,
    add_block_offsets,
    bitonic_step,
    emit_isects,
    project_and_count,
    scan_block,
    scan_block_sums,
    write_tile_offsets,
)
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
from refmath import _ref_rho2
from utils_t import Mat3, matmul3x3, quat_to_rotmat, transpose
from vec import Vec3, Vec4

comptime BG = SIMD[DTYPE, 4](0.1, 0.15, 0.2, 0.0)
comptime MASKED_TILE = 0  # one tile forced invisible, to cover the mask path
comptime TOL: Float32 = 2e-4  # GPU vs host exp() disagree in the last bits


@fieldwise_init
struct SE3(Copyable, ImplicitlyCopyable, Movable):
    var rotation: Mat3
    var translation: Vec3


def extract_se3(
    viewmats: TileTensor[DTYPE, type_of(layout_viewmats), MutAnyOrigin],
    cid: Int,
) -> SE3:
    """Pull the rotation block and translation column out of a [4, 4] view matrix."""
    comptime assert viewmats.flat_rank == 3
    var rotation = Mat3()
    comptime for r in range(3):
        comptime for c in range(3):
            rotation[r, c] = rebind[Scalar[DTYPE]](viewmats[cid, r, c])
    var translation = Vec3(
        rebind[Scalar[DTYPE]](viewmats[cid, 0, 3]),
        rebind[Scalar[DTYPE]](viewmats[cid, 1, 3]),
        rebind[Scalar[DTYPE]](viewmats[cid, 2, 3]),
    )
    return SE3(rotation=rotation, translation=translation)


@fieldwise_init
struct RollingShutterParameters(Copyable, ImplicitlyCopyable, Movable):
    """Start/end camera pose for a rolling-shutter interval.

    When there is no end pose the interval collapses to the start pose (the
    original had this test inverted, so it read the uninitialized end pose).
    """

    var t_start: Vec3
    var q_start: Vec4
    var t_end: Vec3
    var q_end: Vec4

    def __init__(out self, start: SE3, end: SE3, has_end: Bool):
        from utils_t import rotation_matrix_to_quaternion

        self.t_start = start.translation
        self.q_start = rotation_matrix_to_quaternion(start.rotation)
        if has_end:
            self.t_end = end.translation
            self.q_end = rotation_matrix_to_quaternion(end.rotation)
        else:
            self.t_end = self.t_start
            self.q_end = self.q_start


def rasterize_to_pixels_from_world_3dgs_fwd(
    n_cameras: Int32,
    n_gaussians: Int32,
    n_isects: Int32,
    packed: Int32,
    means: TileTensor[DTYPE, type_of(layout_n3), MutAnyOrigin],  # [N, 3]
    quats: TileTensor[DTYPE, type_of(layout_n4), MutAnyOrigin],  # [N, 4]
    scales: TileTensor[DTYPE, type_of(layout_n3), MutAnyOrigin],  # [N, 3]
    colors: TileTensor[DTYPE, type_of(layout_cn_cdim), MutAnyOrigin],  # [C, N, CDIM]
    opacities: TileTensor[DTYPE, type_of(layout_cn), MutAnyOrigin],  # [C, N]
    backgrounds: TileTensor[DTYPE, type_of(layout_cdim), MutAnyOrigin],  # [C, CDIM]
    masks: TileTensor[IDTYPE, type_of(layout_tiles), MutAnyOrigin],  # [C, TY, TX]
    has_backgrounds: Int32,  # Bool is not DevicePassable
    has_masks: Int32,
    image_width: Int32,
    image_height: Int32,
    tile_size: Int32,
    tile_width: Int32,  # tiles across, not pixels
    tile_height: Int32,  # tiles down, not pixels
    # camera model
    viewmats0: TileTensor[DTYPE, type_of(layout_viewmats), MutAnyOrigin],
    viewmats1: TileTensor[DTYPE, type_of(layout_viewmats), MutAnyOrigin],
    Ks: TileTensor[DTYPE, type_of(layout_intrinsics), MutAnyOrigin],  # [C, 3, 3]
    camera_model_type: Int32,
    rs_type: Int32,
    radial_coeffs: TileTensor[DTYPE, type_of(layout_c6), MutAnyOrigin],
    tangential_coeffs: TileTensor[DTYPE, type_of(layout_c2), MutAnyOrigin],
    thin_prims_coeffs: TileTensor[DTYPE, type_of(layout_c2), MutAnyOrigin],
    # intersections
    tile_offsets: TileTensor[IDTYPE, type_of(layout_tiles), MutAnyOrigin],
    flatten_ids: TileTensor[IDTYPE, type_of(layout_isects), MutAnyOrigin],
    render_colors: TileTensor[DTYPE, type_of(layout_render_colors), MutAnyOrigin],
    render_alphas: TileTensor[DTYPE, type_of(layout_render_alphas), MutAnyOrigin],
    last_ids: TileTensor[IDTYPE, type_of(layout_last_ids), MutAnyOrigin],
):
    comptime assert CDIM <= 4, "pixel accumulator is a 4-lane SIMD"
    comptime assert means.flat_rank == 2
    comptime assert quats.flat_rank == 2
    comptime assert scales.flat_rank == 2
    comptime assert colors.flat_rank == 3
    comptime assert opacities.flat_rank == 2
    comptime assert backgrounds.flat_rank == 2
    comptime assert masks.flat_rank == 3
    comptime assert Ks.flat_rank == 3
    comptime assert tile_offsets.flat_rank == 3
    comptime assert flatten_ids.flat_rank == 1
    comptime assert render_colors.flat_rank == 4
    comptime assert render_alphas.flat_rank == 4
    comptime assert last_ids.flat_rank == 3

    # Kernel scalars arrive fixed-width (Int/UInt are not DevicePassable);
    # widen once here so the rest of the body indexes with plain Int.
    var n_cams = Int(n_cameras)
    var total_isects = Int(n_isects)
    var img_w = Int(image_width)
    var img_h = Int(image_height)
    var tsize = Int(tile_size)
    var twidth = Int(tile_width)
    var theight = Int(tile_height)
    var use_bg = has_backgrounds != 0
    var use_masks = has_masks != 0

    var cid = Int(block_idx.z)
    var tile_row = Int(block_idx.y)
    var tile_col = Int(block_idx.x)
    var tile_id = tile_row * twidth + tile_col

    var i = tile_row * tsize + Int(thread_idx.y)
    var j = tile_col * tsize + Int(thread_idx.x)

    var px = Float32(j) + 0.5
    var py = Float32(i) + 0.5

    var inside = i < img_h and j < img_w
    var done = not inside

    var focal_x = rebind[Scalar[DTYPE]](Ks[cid, 0, 0])
    var focal_y = rebind[Scalar[DTYPE]](Ks[cid, 1, 1])
    var principal_x = rebind[Scalar[DTYPE]](Ks[cid, 0, 2])
    var principal_y = rebind[Scalar[DTYPE]](Ks[cid, 1, 2])

    # Pixel ray, camera space. Not normalized: both the closest-approach
    # parameter and rho^2 below are invariant to the length of the direction.
    var ray_d_cam = Vec3(
        (px - principal_x) / focal_x, (py - principal_y) / focal_y, 1.0
    )

    # Lift the ray into world space, since the gaussians are world-space.
    # viewmats0 is world->camera (x_c = R x_w + t), so the camera->world
    # rotation is R^T and the camera centre is -R^T t.
    var cam = extract_se3(viewmats0, cid)
    var rot_cam_to_world = transpose(cam.rotation)
    var rayo = -(rot_cam_to_world * cam.translation)
    var rayd = rot_cam_to_world * ray_d_cam

    # A masked-out tile is uniform across the block, so returning here cannot
    # strand some threads of the block at a later barrier().
    if use_masks and Int(rebind[Scalar[IDTYPE]](masks[cid, tile_row, tile_col])) == 0:
        if inside:
            comptime for k in range(CDIM):
                var bg: Float32 = 0.0
                if use_bg:
                    bg = rebind[Scalar[DTYPE]](backgrounds[cid, k])
                render_colors[cid, i, j, k] = bg
            render_alphas[cid, i, j, 0] = 0.0
            last_ids[cid, i, j] = 0
        return

    # Range of this tile's gaussians inside flatten_ids. The end is the next
    # tile's start, walking the flattened [C, TY, TX] order.
    var range_start = Int(rebind[Scalar[IDTYPE]](tile_offsets[cid, tile_row, tile_col]))
    var range_end: Int
    if cid == n_cams - 1 and tile_id == twidth * theight - 1:
        range_end = total_isects
    else:
        var next_tile = tile_id + 1
        var next_cid = cid
        if next_tile == twidth * theight:
            next_tile = 0
            next_cid = cid + 1
        range_end = Int(
            rebind[Scalar[IDTYPE]](
                tile_offsets[
                    next_cid, next_tile // twidth, next_tile % twidth
                ]
            )
        )

    var num_batches = ceildiv(range_end - range_start, BLOCK_SIZE)

    var id_batch = stack_allocation[
        IDTYPE, address_space = AddressSpace.SHARED
    ](row_major[BLOCK_SIZE]())
    var xyz_opacity_batch = stack_allocation[
        DTYPE, address_space = AddressSpace.SHARED
    ](row_major[BLOCK_SIZE, 4]())
    var iscl_rot_batch = stack_allocation[
        DTYPE, address_space = AddressSpace.SHARED
    ](row_major[BLOCK_SIZE, 3, 3]())

    var transmittance: Float32 = 1.0
    var cur_idx: Int32 = 0
    var pix_out = SIMD[DTYPE, 4](0.0)

    var tr = Int(thread_idx.x + thread_idx.y * block_dim.x)

    for b in range(num_batches):
        # Every thread of the block runs the same number of iterations, so the
        # barriers below are reached uniformly even by finished pixels.
        barrier()

        var batch_start = range_start + BLOCK_SIZE * b
        var idx = batch_start + tr

        if idx < range_end:
            var g = Int(rebind[Scalar[IDTYPE]](flatten_ids[idx]))
            id_batch[tr] = Int32(g)

            xyz_opacity_batch[tr, 0] = rebind[Scalar[DTYPE]](means[g, 0])
            xyz_opacity_batch[tr, 1] = rebind[Scalar[DTYPE]](means[g, 1])
            xyz_opacity_batch[tr, 2] = rebind[Scalar[DTYPE]](means[g, 2])
            xyz_opacity_batch[tr, 3] = rebind[Scalar[DTYPE]](opacities[cid, g])

            var quat = Vec4(
                rebind[Scalar[DTYPE]](quats[g, 0]),
                rebind[Scalar[DTYPE]](quats[g, 1]),
                rebind[Scalar[DTYPE]](quats[g, 2]),
                rebind[Scalar[DTYPE]](quats[g, 3]),
            )
            var rotation = quat_to_rotmat(quat)
            var inv_scale = Mat3.diagonal(
                1.0 / rebind[Scalar[DTYPE]](scales[g, 0]),
                1.0 / rebind[Scalar[DTYPE]](scales[g, 1]),
                1.0 / rebind[Scalar[DTYPE]](scales[g, 2]),
            )
            var iscl_rot = matmul3x3(inv_scale, transpose(rotation))
            comptime for r in range(3):
                comptime for c in range(3):
                    iscl_rot_batch[tr, r, c] = iscl_rot[r, c]

        barrier()

        var batch_size = min(BLOCK_SIZE, range_end - batch_start)
        var t = 0
        while t < batch_size and not done:
            var opacity = rebind[Scalar[DTYPE]](xyz_opacity_batch[t, 3])
            var mean = Vec3(
                rebind[Scalar[DTYPE]](xyz_opacity_batch[t, 0]),
                rebind[Scalar[DTYPE]](xyz_opacity_batch[t, 1]),
                rebind[Scalar[DTYPE]](xyz_opacity_batch[t, 2]),
            )
            var cur_iscl_rot = Mat3()
            comptime for r in range(3):
                comptime for c in range(3):
                    cur_iscl_rot[r, c] = rebind[Scalar[DTYPE]](
                        iscl_rot_batch[t, r, c]
                    )

            # Ray/gaussian intersection. cur_iscl_rot is S^-1 R^T, which
            # maps a world offset into the gaussian's canonical frame where
            # the gaussian is the unit sphere. In that frame the response
            # along the ray is a 1D gaussian, maximal at the ray's closest
            # approach to the origin.
            var og = cur_iscl_rot * (rayo - mean)
            var dg = cur_iscl_rot * rayd
            var dd = dg.dot(dg)
            if dd > 1e-20:
                var t_star = -og.dot(dg) / dd
                if t_star > 0.0:  # ignore gaussians behind the camera
                    var closest = og + dg * t_star
                    var rho2 = closest.dot(closest)
                    var alpha = min(
                        Float32(MAX_ALPHA), opacity * exp(-0.5 * rho2)
                    )
                    if alpha >= MIN_ALPHA:
                        var next_t = transmittance * (1.0 - alpha)
                        if next_t < T_EPS:
                            # Pixel is saturated; this gaussian is excluded.
                            done = True
                        else:
                            var gid = Int(
                                rebind[Scalar[IDTYPE]](id_batch[t])
                            )
                            var weight = alpha * transmittance
                            comptime for k in range(CDIM):
                                pix_out[k] += weight * rebind[
                                    Scalar[DTYPE]
                                ](colors[cid, gid, k])
                            transmittance = next_t
                            cur_idx = Int32(batch_start + t)
            t += 1

    if inside:
        render_alphas[cid, i, j, 0] = 1.0 - transmittance
        comptime for k in range(CDIM):
            var bg: Float32 = 0.0
            if use_bg:
                bg = rebind[Scalar[DTYPE]](backgrounds[cid, k])
            render_colors[cid, i, j, k] = pix_out[k] + transmittance * bg
        last_ids[cid, i, j] = cur_idx


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
            zz += view_rot(2, a) * spread_mean(g, a)
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
                        spread_mean(g, 0), spread_mean(g, 1), spread_mean(g, 2),
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

    var n_pow2 = 1
    while n_pow2 < n_isects:
        n_pow2 *= 2
    var keys_buf = ctx.enqueue_create_buffer[KDTYPE](n_pow2)
    var sorted_buf = ctx.enqueue_create_buffer[IDTYPE](n_pow2)
    keys_buf.enqueue_fill(KEY_MAX)
    sorted_buf.enqueue_fill(-1)
    var keys = TileTensor(keys_buf, layout_isects)
    var sorted_ids = TileTensor(sorted_buf, layout_isects)

    ctx.enqueue_function[emit_isects](
        bboxes, depths, offsets, counts, keys, sorted_ids,
        Int32(n_gauss), Int32(N_TILES_X), Int32(N_TILES),
        grid_dim=(gblocks, C), block_dim=TPB,
    )
    var k = 2
    while k <= n_pow2:
        var j = k // 2
        while j > 0:
            ctx.enqueue_function[bitonic_step](
                keys, sorted_ids, Int32(n_pow2), Int32(k), Int32(j),
                grid_dim=ceildiv(n_pow2, TPB), block_dim=TPB,
            )
            j //= 2
        k *= 2
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
