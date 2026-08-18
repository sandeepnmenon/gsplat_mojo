"""Tile-based 3D gaussian rasterization — forward pass.

Ported to current Mojo (`comptime`/`def`/`TileTensor`). Structure follows
gsplat's `rasterize_to_pixels_from_world_3dgs_fwd`: one thread block per
(camera, tile), gaussians for the tile streamed through shared memory in
block-sized batches, then alpha-composited front-to-back into the pixel.

STATUS: the per-gaussian ray/gaussian evaluation is still a stub — the batch
loop loads each gaussian's centre, opacity and inverse-scale-rotation into
registers but does not yet compute its 2D footprint, so `pix_out` stays zero
and transmittance stays 1. Everything around it is live: tile ranges, shared
memory batching, barriers, background compositing and write-back. With the
stub, a pixel resolves to exactly the background colour, which is what
`main()` checks.
"""

from std.math import ceildiv
from std.gpu import block_dim, block_idx, thread_idx
from std.sys import has_accelerator
from max.gpu.host import DeviceContext
from max.gpu.memory import AddressSpace
from max.gpu.sync import barrier
from layout import TileTensor, row_major, stack_allocation

from utils_t import Mat3, matmul3x3, quat_to_rotmat, transpose
from vec import Vec3, Vec4

comptime DTYPE = DType.float32
comptime IDTYPE = DType.int32

# Scene / image configuration.
comptime C = 1  # number of cameras
comptime N = 4  # number of gaussians
comptime CDIM = 3  # color channels (RGB)
comptime IMG_W = 1024
comptime IMG_H = 768
comptime TILE = 16  # tile edge, in pixels

# One thread per pixel in a tile.
comptime BLOCK_SIZE = TILE * TILE

# Tile counts covering the image. The original code conflated "tile edge" with
# "number of tiles" and sized the intersection buffers [C, TILE, TILE]; they
# are [C, N_TILES_Y, N_TILES_X] here so the tile grid and the buffers agree.
comptime N_TILES_X = ceildiv(IMG_W, TILE)
comptime N_TILES_Y = ceildiv(IMG_H, TILE)
comptime N_TILES = N_TILES_X * N_TILES_Y

# Test intersection set: every tile references every gaussian.
comptime N_ISECTS = C * N_TILES * N

comptime layout_n3 = row_major[N, 3]()
comptime layout_n4 = row_major[N, 4]()
comptime layout_cn_cdim = row_major[C, N, CDIM]()
comptime layout_cn = row_major[C, N]()
comptime layout_cdim = row_major[C, CDIM]()
comptime layout_tiles = row_major[C, N_TILES_Y, N_TILES_X]()
comptime layout_viewmats = row_major[C, 4, 4]()
comptime layout_intrinsics = row_major[C, 3, 3]()
comptime layout_c6 = row_major[C, 6]()
comptime layout_c2 = row_major[C, 2]()
comptime layout_render_colors = row_major[C, IMG_H, IMG_W, CDIM]()
comptime layout_render_alphas = row_major[C, IMG_H, IMG_W, 1]()
comptime layout_last_ids = row_major[C, IMG_H, IMG_W]()
comptime layout_isects = row_major[N_ISECTS]()


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

    # Camera ray for this pixel, in camera space.
    var rayd = Vec3(
        (px - principal_x) / focal_x, (py - principal_y) / focal_y, 1.0
    )
    var rayo = Vec3(0.0, 0.0, 0.0)

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

            # TODO: project this gaussian along `rayo`/`rayd` using
            # `cur_iscl_rot`, evaluate its 2D footprint at (px, py), and
            # alpha-composite into `pix_out`/`transmittance`. Until then the
            # pixel keeps full transmittance and picks up only the background.
            _ = opacity
            _ = mean
            _ = cur_iscl_rot

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

    comptime BG = SIMD[DTYPE, 4](0.1, 0.2, 0.3, 0.0)

    var ctx = DeviceContext()

    # ---- device buffers -------------------------------------------------
    var means_buf = ctx.enqueue_create_buffer[DTYPE](N * 3)
    var quats_buf = ctx.enqueue_create_buffer[DTYPE](N * 4)
    var scales_buf = ctx.enqueue_create_buffer[DTYPE](N * 3)
    var colors_buf = ctx.enqueue_create_buffer[DTYPE](C * N * CDIM)
    var opac_buf = ctx.enqueue_create_buffer[DTYPE](C * N)
    var backgrounds_buf = ctx.enqueue_create_buffer[DTYPE](C * CDIM)
    var masks_buf = ctx.enqueue_create_buffer[IDTYPE](C * N_TILES)
    var viewmats0_buf = ctx.enqueue_create_buffer[DTYPE](C * 16)
    var viewmats1_buf = ctx.enqueue_create_buffer[DTYPE](C * 16)
    var ks_buf = ctx.enqueue_create_buffer[DTYPE](C * 9)
    var radial_buf = ctx.enqueue_create_buffer[DTYPE](C * 6)
    var tangential_buf = ctx.enqueue_create_buffer[DTYPE](C * 2)
    var thin_prims_buf = ctx.enqueue_create_buffer[DTYPE](C * 2)
    var tile_offsets_buf = ctx.enqueue_create_buffer[IDTYPE](C * N_TILES)
    var flatten_ids_buf = ctx.enqueue_create_buffer[IDTYPE](N_ISECTS)
    var renders_buf = ctx.enqueue_create_buffer[DTYPE](C * IMG_H * IMG_W * CDIM)
    var alphas_buf = ctx.enqueue_create_buffer[DTYPE](C * IMG_H * IMG_W)
    var ids_buf = ctx.enqueue_create_buffer[IDTYPE](C * IMG_H * IMG_W)

    # Defaults for everything not explicitly staged below.
    radial_buf.enqueue_fill(0.0)
    tangential_buf.enqueue_fill(0.0)
    thin_prims_buf.enqueue_fill(0.0)
    viewmats1_buf.enqueue_fill(0.0)
    masks_buf.enqueue_fill(1)  # every tile visible
    renders_buf.enqueue_fill(-1.0)  # poison, so write-back is observable
    alphas_buf.enqueue_fill(-1.0)
    ids_buf.enqueue_fill(-1)

    # ---- staged host data ------------------------------------------------
    var means_h = ctx.enqueue_create_host_buffer[DTYPE](N * 3)
    var quats_h = ctx.enqueue_create_host_buffer[DTYPE](N * 4)
    var scales_h = ctx.enqueue_create_host_buffer[DTYPE](N * 3)
    var colors_h = ctx.enqueue_create_host_buffer[DTYPE](C * N * CDIM)
    var opac_h = ctx.enqueue_create_host_buffer[DTYPE](C * N)
    var bg_h = ctx.enqueue_create_host_buffer[DTYPE](C * CDIM)
    var view_h = ctx.enqueue_create_host_buffer[DTYPE](C * 16)
    var ks_h = ctx.enqueue_create_host_buffer[DTYPE](C * 9)
    var offsets_h = ctx.enqueue_create_host_buffer[IDTYPE](C * N_TILES)
    var flat_h = ctx.enqueue_create_host_buffer[IDTYPE](N_ISECTS)
    ctx.synchronize()

    for g in range(N):
        # Spread the gaussians along z so they are not coincident.
        means_h[g * 3 + 0] = Float32(g) * 0.5 - 0.75
        means_h[g * 3 + 1] = 0.0
        means_h[g * 3 + 2] = 2.0 + Float32(g)
        # Identity rotation, (x, y, z, w).
        quats_h[g * 4 + 0] = 0.0
        quats_h[g * 4 + 1] = 0.0
        quats_h[g * 4 + 2] = 0.0
        quats_h[g * 4 + 3] = 1.0
        scales_h[g * 3 + 0] = 0.1
        scales_h[g * 3 + 1] = 0.1
        scales_h[g * 3 + 2] = 0.1

    for c in range(C):
        for g in range(N):
            opac_h[c * N + g] = 0.5
            colors_h[(c * N + g) * CDIM + 0] = 0.9
            colors_h[(c * N + g) * CDIM + 1] = 0.4
            colors_h[(c * N + g) * CDIM + 2] = 0.1
        comptime for k in range(CDIM):
            bg_h[c * CDIM + k] = BG[k]

        # Identity extrinsics.
        for e in range(16):
            view_h[c * 16 + e] = 0.0
        view_h[c * 16 + 0] = 1.0
        view_h[c * 16 + 5] = 1.0
        view_h[c * 16 + 10] = 1.0
        view_h[c * 16 + 15] = 1.0

        # Pinhole intrinsics centred on the image.
        for e in range(9):
            ks_h[c * 9 + e] = 0.0
        ks_h[c * 9 + 0] = 600.0  # fx
        ks_h[c * 9 + 4] = 600.0  # fy
        ks_h[c * 9 + 2] = Float32(IMG_W) * 0.5  # cx
        ks_h[c * 9 + 5] = Float32(IMG_H) * 0.5  # cy
        ks_h[c * 9 + 8] = 1.0

    # Every tile sees every gaussian: tile f owns flatten_ids[f*N : (f+1)*N].
    for f in range(C * N_TILES):
        offsets_h[f] = Int32(f * N)
        for g in range(N):
            flat_h[f * N + g] = Int32(g)

    ctx.enqueue_copy(dst_buf=means_buf, src_buf=means_h)
    ctx.enqueue_copy(dst_buf=quats_buf, src_buf=quats_h)
    ctx.enqueue_copy(dst_buf=scales_buf, src_buf=scales_h)
    ctx.enqueue_copy(dst_buf=colors_buf, src_buf=colors_h)
    ctx.enqueue_copy(dst_buf=opac_buf, src_buf=opac_h)
    ctx.enqueue_copy(dst_buf=backgrounds_buf, src_buf=bg_h)
    ctx.enqueue_copy(dst_buf=viewmats0_buf, src_buf=view_h)
    ctx.enqueue_copy(dst_buf=ks_buf, src_buf=ks_h)
    ctx.enqueue_copy(dst_buf=tile_offsets_buf, src_buf=offsets_h)
    ctx.enqueue_copy(dst_buf=flatten_ids_buf, src_buf=flat_h)

    # ---- tensor views ----------------------------------------------------
    var means = TileTensor(means_buf, layout_n3)
    var quats = TileTensor(quats_buf, layout_n4)
    var scales = TileTensor(scales_buf, layout_n3)
    var colors = TileTensor(colors_buf, layout_cn_cdim)
    var opacities = TileTensor(opac_buf, layout_cn)
    var backgrounds = TileTensor(backgrounds_buf, layout_cdim)
    var masks = TileTensor(masks_buf, layout_tiles)
    var viewmats0 = TileTensor(viewmats0_buf, layout_viewmats)
    var viewmats1 = TileTensor(viewmats1_buf, layout_viewmats)
    var ks = TileTensor(ks_buf, layout_intrinsics)
    var radial = TileTensor(radial_buf, layout_c6)
    var tangential = TileTensor(tangential_buf, layout_c2)
    var thin_prims = TileTensor(thin_prims_buf, layout_c2)
    var tile_offsets = TileTensor(tile_offsets_buf, layout_tiles)
    var flatten_ids = TileTensor(flatten_ids_buf, layout_isects)
    var render_colors = TileTensor(renders_buf, layout_render_colors)
    var render_alphas = TileTensor(alphas_buf, layout_render_alphas)
    var last_ids = TileTensor(ids_buf, layout_last_ids)

    print(
        "launching:",
        IMG_W,
        "x",
        IMG_H,
        "| tiles",
        N_TILES_X,
        "x",
        N_TILES_Y,
        "| gaussians",
        N,
        "| isects",
        N_ISECTS,
    )

    ctx.enqueue_function[rasterize_to_pixels_from_world_3dgs_fwd](
        Int32(C),
        Int32(N),
        Int32(N_ISECTS),
        Int32(0),  # packed
        means,
        quats,
        scales,
        colors,
        opacities,
        backgrounds,
        masks,
        Int32(1),  # has_backgrounds
        Int32(1),  # has_masks
        Int32(IMG_W),
        Int32(IMG_H),
        Int32(TILE),
        Int32(N_TILES_X),
        Int32(N_TILES_Y),
        viewmats0,
        viewmats1,
        ks,
        Int32(0),  # camera_model_type
        Int32(0),  # rs_type
        radial,
        tangential,
        thin_prims,
        tile_offsets,
        flatten_ids,
        render_colors,
        render_alphas,
        last_ids,
        grid_dim=(N_TILES_X, N_TILES_Y, C),
        block_dim=(TILE, TILE, 1),
    )
    ctx.synchronize()

    # ---- verify ----------------------------------------------------------
    # With the ray/gaussian stub, transmittance stays 1, so every pixel must
    # come out as exactly the background and every alpha as 0.
    var bad_color = 0
    var bad_alpha = 0
    with renders_buf.map_to_host() as colors_host:
        with alphas_buf.map_to_host() as alphas_host:
            for c in range(C):
                for y in range(IMG_H):
                    for x in range(IMG_W):
                        var base = ((c * IMG_H + y) * IMG_W + x) * CDIM
                        comptime for k in range(CDIM):
                            if abs(colors_host[base + k] - BG[k]) > 1e-6:
                                bad_color += 1
                        if abs(alphas_host[(c * IMG_H + y) * IMG_W + x]) > 1e-6:
                            bad_alpha += 1
            print(
                "sample pixel (0,0)   rgb =",
                colors_host[0],
                colors_host[1],
                colors_host[2],
            )
            var mid = ((IMG_H // 2) * IMG_W + IMG_W // 2) * CDIM
            print(
                "sample pixel (512,384) rgb =",
                colors_host[mid],
                colors_host[mid + 1],
                colors_host[mid + 2],
            )

    # last_ids is the strongest signal that the batch loop really ran: it holds
    # the flatten_ids slot of the last gaussian the pixel walked. Tile f owns
    # slots [f*N, (f+1)*N), so every pixel of tile f must report f*N + N - 1.
    # A kernel that launched but skipped the inner loop would leave the -1
    # poison here even though the colours still looked right.
    var bad_ids = 0
    with ids_buf.map_to_host() as ids_host:
        for c in range(C):
            for y in range(IMG_H):
                for x in range(IMG_W):
                    var f = c * N_TILES + (y // TILE) * N_TILES_X + (x // TILE)
                    var expected = Int32(f * N + N - 1)
                    if ids_host[(c * IMG_H + y) * IMG_W + x] != expected:
                        bad_ids += 1
        print(
            "last_ids: tile 0 pixel =",
            ids_host[0],
            "| tile 1 pixel =",
            ids_host[TILE],
            "| expected",
            N - 1,
            "and",
            N + N - 1,
        )

    var total = C * IMG_H * IMG_W
    print("pixels checked:", total)
    print(
        "mismatches — color:",
        bad_color,
        "| alpha:",
        bad_alpha,
        "| last_ids:",
        bad_ids,
    )
    if bad_color == 0 and bad_alpha == 0 and bad_ids == 0:
        print(
            "PASS: every pixel written, and every tile streamed all",
            N,
            "gaussians through shared memory",
        )
    else:
        raise Error("FAIL: forward pass produced unexpected pixels")
