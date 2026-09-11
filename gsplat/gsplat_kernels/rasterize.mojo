"""Tile-based 3D gaussian rasterization — the forward kernel.

One thread block per (camera, tile); the gaussians for a tile are streamed
through shared memory in block-sized batches and alpha-composited
front-to-back.

Each gaussian is evaluated by ray/gaussian intersection rather than an EWA
2D-splat projection: `iscl_rot` = S^-1 R^T maps world space into the frame
where the gaussian is the unit sphere, the ray's closest approach to the
origin in that frame gives rho^2, and the response is exp(-rho^2 / 2).
Contributions are composited in `flatten_ids` order, so that list must be
depth-sorted per tile -- which is what `intersect.mojo` produces.

The self-checking render that exercises this lives in `tests/forward_test.mojo`.
"""

from std.math import ceildiv, exp, sqrt
from std.gpu import block_dim, block_idx, thread_idx
from std.sys import has_accelerator
from max.gpu.host import DeviceContext
from max.gpu.memory import AddressSpace
from max.gpu.sync import barrier
from layout import TileTensor, row_major, stack_allocation

from gsplat_kernels.config import (
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
    RADIX,
    RADIX_EPB,
    RADIX_PASSES,
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
from gsplat_kernels.utils_t import Mat3, matmul3x3, quat_to_rotmat, transpose
from gsplat_kernels.vec import Vec3, Vec4


@fieldwise_init
struct SE3(Copyable, ImplicitlyCopyable, Movable):
    var rotation: Mat3
    var translation: Vec3


def extract_se3(
    viewmats: TileTensor[DTYPE, type_of(layout_viewmats), MutAnyOrigin],
    cid: Int,
) -> SE3:
    """Pull the rotation block and translation column out of a [4, 4] view matrix.
    """
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
        from gsplat_kernels.utils_t import rotation_matrix_to_quaternion

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
    colors: TileTensor[
        DTYPE, type_of(layout_cn_cdim), MutAnyOrigin
    ],  # [C, N, CDIM]
    opacities: TileTensor[DTYPE, type_of(layout_cn), MutAnyOrigin],  # [C, N]
    backgrounds: TileTensor[
        DTYPE, type_of(layout_cdim), MutAnyOrigin
    ],  # [C, CDIM]
    masks: TileTensor[
        IDTYPE, type_of(layout_tiles), MutAnyOrigin
    ],  # [C, TY, TX]
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
    Ks: TileTensor[
        DTYPE, type_of(layout_intrinsics), MutAnyOrigin
    ],  # [C, 3, 3]
    camera_model_type: Int32,
    rs_type: Int32,
    radial_coeffs: TileTensor[DTYPE, type_of(layout_c6), MutAnyOrigin],
    tangential_coeffs: TileTensor[DTYPE, type_of(layout_c2), MutAnyOrigin],
    thin_prims_coeffs: TileTensor[DTYPE, type_of(layout_c2), MutAnyOrigin],
    # intersections
    tile_offsets: TileTensor[IDTYPE, type_of(layout_tiles), MutAnyOrigin],
    flatten_ids: TileTensor[IDTYPE, type_of(layout_isects), MutAnyOrigin],
    render_colors: TileTensor[
        DTYPE, type_of(layout_render_colors), MutAnyOrigin
    ],
    render_alphas: TileTensor[
        DTYPE, type_of(layout_render_alphas), MutAnyOrigin
    ],
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
    if (
        use_masks
        and Int(rebind[Scalar[IDTYPE]](masks[cid, tile_row, tile_col])) == 0
    ):
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
    var range_start = Int(
        rebind[Scalar[IDTYPE]](tile_offsets[cid, tile_row, tile_col])
    )
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
                tile_offsets[next_cid, next_tile // twidth, next_tile % twidth]
            )
        )

    var num_batches = ceildiv(range_end - range_start, BLOCK_SIZE)

    var id_batch = stack_allocation[IDTYPE, address_space=AddressSpace.SHARED](
        row_major[BLOCK_SIZE]()
    )
    var xyz_opacity_batch = stack_allocation[
        DTYPE, address_space=AddressSpace.SHARED
    ](row_major[BLOCK_SIZE, 4]())
    var iscl_rot_batch = stack_allocation[
        DTYPE, address_space=AddressSpace.SHARED
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
                            var gid = Int(rebind[Scalar[IDTYPE]](id_batch[t]))
                            var weight = alpha * transmittance
                            comptime for k in range(CDIM):
                                pix_out[k] += weight * rebind[Scalar[DTYPE]](
                                    colors[cid, gid, k]
                                )
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
