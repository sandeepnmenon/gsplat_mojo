"""MAX custom op wrapping the gaussian-splatting forward pass.

Ported from the retired custom-op API. The changes were structural, not
cosmetic:

  | old                             | new                                  |
  |---------------------------------|--------------------------------------|
  | `import compiler`               | `from extensibility import ...`      |
  | `@compiler.register("render")`  | `@register("render")`                |
  | `from tensor import InputTensor`| `from extensibility import ...`      |
  | `InputTensor[type=..., rank=n]` | `InputTensor[dtype=..., rank=n, static_spec=_]` |
  | `ctx: DeviceContextPtr`         | `ctx: DeviceContext`                 |
  | `ctx.get_device_context()`      | `ctx` directly                       |

The op owns all of its scratch: it projects, bins, depth-sorts and rasterizes,
then writes RGB into `img_out`. Because the intersection count is only known
after the prefix sum, there is one device-to-host sync inside the op to size
the sort — the reference CUDA implementation does the same.
"""

from std.math import ceildiv
from extensibility import InputTensor, OutputTensor, register
from max.gpu.host import DeviceContext
from layout import TileTensor

from gsplat_kernels.config import (
    C,
    CDIM,
    DTYPE,
    IDTYPE,
    IMG_H,
    IMG_W,
    KDTYPE,
    N_MAX,
    N_TILES,
    N_TILES_X,
    N_TILES_Y,
    RADIX,
    RADIX_EPB,
    SCAN_BLOCK,
    SCAN_NUM_BLOCKS,
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
from gsplat_kernels.rasterize import rasterize_to_pixels_from_world_3dgs_fwd
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


@register("render")
struct Render:
    @staticmethod
    def execute[
        target: StaticString
    ](
        img_out: OutputTensor[dtype=DType.float32, rank=3, static_spec=_],
        means: InputTensor[dtype=DType.float32, rank=2, static_spec=_],
        colors: InputTensor[dtype=DType.float32, rank=2, static_spec=_],
        opacities: InputTensor[dtype=DType.float32, rank=1, static_spec=_],
        scales: InputTensor[dtype=DType.float32, rank=2, static_spec=_],
        quats: InputTensor[dtype=DType.float32, rank=2, static_spec=_],
        viewmats: InputTensor[dtype=DType.float32, rank=3, static_spec=_],
        ks: InputTensor[dtype=DType.float32, rank=3, static_spec=_],
        ctx: DeviceContext,
    ) raises:
        comptime if target != "gpu":
            raise Error("render: only the gpu target is implemented")

        var n_gauss = Int(means.dim_size(0))
        if n_gauss <= 0 or n_gauss > N_MAX:
            raise Error("render: gaussian count outside the configured N_MAX")
        if (
            Int(img_out.dim_size(0)) != IMG_H
            or Int(img_out.dim_size(1)) != IMG_W
        ):
            raise Error("render: output size does not match the build config")
        if Int(img_out.dim_size(2)) != CDIM:
            raise Error("render: output channel count does not match CDIM")
        if Int(viewmats.dim_size(0)) != C:
            raise Error("render: camera count does not match the build config")

        # Views over the caller's buffers. The layouts are capacity-shaped;
        # every index below stays inside the live count, and with C == 1 the
        # strides agree with the caller's tighter [N, ...] shapes.
        var means_t = TileTensor(means.unsafe_ptr(), layout_n3)
        var quats_t = TileTensor(quats.unsafe_ptr(), layout_n4)
        var scales_t = TileTensor(scales.unsafe_ptr(), layout_n3)
        var colors_t = TileTensor(colors.unsafe_ptr(), layout_cn_cdim)
        var opac_t = TileTensor(opacities.unsafe_ptr(), layout_cn)
        var view_t = TileTensor(viewmats.unsafe_ptr(), layout_viewmats)
        var ks_t = TileTensor(ks.unsafe_ptr(), layout_intrinsics)
        var out_t = TileTensor(img_out.unsafe_ptr(), layout_render_colors)

        # Scratch the op owns.
        var bg_buf = ctx.enqueue_create_buffer[DTYPE](C * CDIM)
        var masks_buf = ctx.enqueue_create_buffer[IDTYPE](C * N_TILES)
        var view1_buf = ctx.enqueue_create_buffer[DTYPE](C * 16)
        var radial_buf = ctx.enqueue_create_buffer[DTYPE](C * 6)
        var tang_buf = ctx.enqueue_create_buffer[DTYPE](C * 2)
        var thin_buf = ctx.enqueue_create_buffer[DTYPE](C * 2)
        var counts_buf = ctx.enqueue_create_buffer[IDTYPE](C * N_MAX)
        var offsets_buf = ctx.enqueue_create_buffer[IDTYPE](C * N_MAX)
        var bbox_buf = ctx.enqueue_create_buffer[IDTYPE](C * N_MAX * 4)
        var depth_buf = ctx.enqueue_create_buffer[DTYPE](C * N_MAX)
        var total_buf = ctx.enqueue_create_buffer[IDTYPE](1)
        var blocksum_buf = ctx.enqueue_create_buffer[IDTYPE](SCAN_NUM_BLOCKS)
        var tileoff_buf = ctx.enqueue_create_buffer[IDTYPE](C * N_TILES)
        var alphas_buf = ctx.enqueue_create_buffer[DTYPE](C * IMG_H * IMG_W)
        var ids_buf = ctx.enqueue_create_buffer[IDTYPE](C * IMG_H * IMG_W)
        var total_h = ctx.enqueue_create_host_buffer[IDTYPE](1)

        bg_buf.enqueue_fill(0.0)
        masks_buf.enqueue_fill(1)
        view1_buf.enqueue_fill(0.0)
        radial_buf.enqueue_fill(0.0)
        tang_buf.enqueue_fill(0.0)
        thin_buf.enqueue_fill(0.0)
        counts_buf.enqueue_fill(0)
        alphas_buf.enqueue_fill(0.0)
        ids_buf.enqueue_fill(-1)

        var backgrounds = TileTensor(bg_buf, layout_cdim)
        var masks = TileTensor(masks_buf, layout_tiles)
        var viewmats1 = TileTensor(view1_buf, layout_viewmats)
        var radial = TileTensor(radial_buf, layout_c6)
        var tangential = TileTensor(tang_buf, layout_c2)
        var thin_prims = TileTensor(thin_buf, layout_c2)
        var counts = TileTensor(counts_buf, layout_cn_i)
        var offsets = TileTensor(offsets_buf, layout_cn_i)
        var counts_flat = TileTensor(counts_buf, layout_cn_flat)
        var offsets_flat = TileTensor(offsets_buf, layout_cn_flat)
        var block_sums = TileTensor(blocksum_buf, layout_blocksums)
        var bboxes = TileTensor(bbox_buf, layout_cn4)
        var depths = TileTensor(depth_buf, layout_cn_f)
        var total = TileTensor(total_buf, layout_one)
        var tile_offsets = TileTensor(tileoff_buf, layout_tiles)
        var tile_offsets_flat = TileTensor(tileoff_buf, layout_tiles_flat)
        var render_alphas = TileTensor(alphas_buf, layout_render_alphas)
        var last_ids = TileTensor(ids_buf, layout_last_ids)

        comptime TPB = 256
        var gblocks = ceildiv(n_gauss, TPB)

        ctx.enqueue_function[project_and_count](
            means_t,
            scales_t,
            quats_t,
            opac_t,
            view_t,
            ks_t,
            counts,
            bboxes,
            depths,
            Int32(n_gauss),
            Int32(N_TILES_X),
            Int32(N_TILES_Y),
            Int32(TILE),
            grid_dim=(gblocks, C),
            block_dim=TPB,
        )
        ctx.enqueue_function[scan_block](
            counts_flat,
            offsets_flat,
            block_sums,
            Int32(C * N_MAX),
            grid_dim=SCAN_NUM_BLOCKS,
            block_dim=SCAN_BLOCK,
        )
        ctx.enqueue_function[scan_block_sums](
            block_sums,
            total,
            Int32(SCAN_NUM_BLOCKS),
            grid_dim=1,
            block_dim=1024,
        )
        ctx.enqueue_function[add_block_offsets](
            offsets_flat,
            block_sums,
            Int32(C * N_MAX),
            grid_dim=SCAN_NUM_BLOCKS,
            block_dim=SCAN_BLOCK,
        )
        ctx.enqueue_copy(dst_buf=total_h, src_buf=total_buf)
        ctx.synchronize()  # the sort has to be sized before it can be issued

        var n_isects = Int(total_h[0])
        # With nothing on screen every tile range is empty and the rasterizer
        # writes pure background, so the only special-casing needed is to keep
        # the allocations non-zero and skip the sort.
        var n_alloc = n_isects if n_isects > 0 else 1
        var n_rblocks = ceildiv(n_alloc, RADIX_EPB)
        var hist_size = RADIX * n_rblocks
        var keys_buf = ctx.enqueue_create_buffer[KDTYPE](n_alloc)
        var vals_buf = ctx.enqueue_create_buffer[IDTYPE](n_alloc)
        var keys_alt_buf = ctx.enqueue_create_buffer[KDTYPE](n_alloc)
        var vals_alt_buf = ctx.enqueue_create_buffer[IDTYPE](n_alloc)
        var hist_buf = ctx.enqueue_create_buffer[IDTYPE](hist_size)
        var histoff_buf = ctx.enqueue_create_buffer[IDTYPE](hist_size)
        var scratch_buf = ctx.enqueue_create_buffer[IDTYPE](1)
        keys_buf.enqueue_fill(KEY_MAX)
        vals_buf.enqueue_fill(-1)

        var keys = TileTensor(keys_buf, layout_isects)
        var vals = TileTensor(vals_buf, layout_isects)
        var keys_alt = TileTensor(keys_alt_buf, layout_isects)
        var vals_alt = TileTensor(vals_alt_buf, layout_isects)
        var hist = TileTensor(hist_buf, layout_cn_flat)
        var hist_off = TileTensor(histoff_buf, layout_cn_flat)
        var scratch = TileTensor(scratch_buf, layout_one)

        tileoff_buf.enqueue_fill(Int32(n_isects))
        if n_isects > 0:
            ctx.enqueue_function[emit_isects](
                bboxes,
                depths,
                offsets,
                counts,
                keys,
                vals,
                Int32(n_gauss),
                Int32(N_TILES_X),
                Int32(N_TILES),
                grid_dim=(gblocks, C),
                block_dim=TPB,
            )
            radix_sort_pairs(
                ctx,
                keys,
                vals,
                keys_alt,
                vals_alt,
                hist,
                hist_off,
                block_sums,
                scratch,
                keys_buf,
                vals_buf,
                keys_alt_buf,
                vals_alt_buf,
                n_isects,
            )
            ctx.enqueue_function[write_tile_offsets](
                keys,
                tile_offsets_flat,
                Int32(n_isects),
                grid_dim=ceildiv(n_isects, TPB),
                block_dim=TPB,
            )

        ctx.enqueue_function[rasterize_to_pixels_from_world_3dgs_fwd](
            Int32(C),
            Int32(n_gauss),
            Int32(n_isects),
            Int32(0),
            means_t,
            quats_t,
            scales_t,
            colors_t,
            opac_t,
            backgrounds,
            masks,
            Int32(1),
            Int32(1),
            Int32(IMG_W),
            Int32(IMG_H),
            Int32(TILE),
            Int32(N_TILES_X),
            Int32(N_TILES_Y),
            view_t,
            viewmats1,
            ks_t,
            Int32(0),
            Int32(0),
            radial,
            tangential,
            thin_prims,
            tile_offsets,
            vals,
            out_t,
            render_alphas,
            last_ids,
            grid_dim=(N_TILES_X, N_TILES_Y, C),
            block_dim=(TILE, TILE, 1),
        )
