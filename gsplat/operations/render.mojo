import compiler
from complex import ComplexSIMD
from math import iota, sqrt
from tensor import OutputTensor, InputTensor, foreach
from runtime.asyncrt import DeviceContextPtr
from utils.index import IndexList
from layout import Layout, LayoutTensor
from gpu import thread_idx, block_idx, block_dim, barrier
alias float_dtype = DType.float32
# Kernel parameters (placeholders for illustration)
alias C = 1      # number of cameras
alias N = 149621      # number of gaussians # TODO: Get this on demand
alias CDIM = 3   # color channels (RGB)
alias IMG_W = 1024
alias IMG_H = 768
alias TILE = 16   # tile size
alias DTYPE = DType.float32
# Shared memory: for 'block_size' threads per block
alias TPB = TILE * TILE
alias n_isects = 3
alias layoutN3 = Layout.row_major(N, 3)
alias layoutN4 = Layout.row_major(N, 4)
alias layoutCNCDIM = Layout.row_major(C, N, CDIM)
alias layoutCN = Layout.row_major(C, N)
alias layoutCDIM = Layout.row_major(C, CDIM)
alias layoutCTILE = Layout.row_major(C, TILE, TILE)
alias layoutViewMatrix = Layout.row_major(C, 4, 4)
alias layoutIntrincisics = Layout.row_major(1, 3, 3)
alias layoutC6 = Layout.row_major(C, 6)
alias layoutC4 = Layout.row_major(C, 4)
alias layoutC2 = Layout.row_major(C, 2)
alias layoutRenderColors = Layout.row_major(C, IMG_H, IMG_W, CDIM)
alias layoutRenderAlphas = Layout.row_major(C, IMG_H, IMG_W, 1)
alias layoutLastIds = Layout.row_major(C, IMG_H, IMG_W)
alias layout_n_isects = Layout.row_major(n_isects)
@compiler.register("render")
struct Render:
    @staticmethod
    fn execute[
        target: StaticString,
    ](
        img_out: OutputTensor[type = DType.float32, rank=3],
        means: InputTensor[type = DType.float32, rank=2],
        colors: InputTensor[type = DType.float32, rank=2],
        opacities: InputTensor[type = DType.float32, rank=1],
        scales: InputTensor[type = DType.float32, rank=2],
        quats: InputTensor[type = DType.float32, rank=2],
        world_to_view_transform: InputTensor[type = DType.float32, rank=3],
        Ks: InputTensor[type = DType.float32, rank=3],
        ctx: DeviceContextPtr,
    ) raises:
        if target == "gpu":
            render_gpu(
                img_out,
                means,
                colors,
                opacities,
                scales,
                quats,
                world_to_view_transform,
                Ks,
                ctx
            )
        else:
            raise Error("Unsupported target device", target)
fn gaussian_2d(
    x: float,
    y: float,
    mean_x: float,
    mean_y: float,
    cov: InputTensor[type = DType.float32, rank=2]
):
fn render_gpu(
    img_out: OutputTensor[type = DType.float32, rank=3],
    means: InputTensor[type = DType.float32, rank=2],
    colors: InputTensor[type = DType.float32, rank=2],
    opacities: InputTensor[type = DType.float32, rank=1],
    scales: InputTensor[type = DType.float32, rank=2],
    quats: InputTensor[type = DType.float32, rank=2],
    world_to_view_transform: InputTensor[type = DType.float32, rank=3],
    Ks: InputTensor[type = DType.float32, rank=3],
    ctx: DeviceContextPtr,
) raises:
    # Define grid and block dimensions
    grid_dim = ( (IMG_W + TILE - 1) // TILE,
                        (IMG_H + TILE - 1) // TILE,
                        C )
    block_dim = (TILE, TILE, 1)
    means_tensor = LayoutTensor[mut=False, DTYPE, layoutN3](means.unsafe_ptr())
    quats_tensor = LayoutTensor[mut=False, DTYPE, layoutN4](quats.unsafe_ptr())
    scales_tensor = LayoutTensor[mut=False, DTYPE, layoutN3](scales.unsafe_ptr())
    colors_tensor = LayoutTensor[mut=False, DTYPE, layoutCNCDIM](colors.unsafe_ptr())
    opacities_tensor = LayoutTensor[mut=False, DTYPE, layoutCN](opacities.unsafe_ptr())
    Ks_tensor = LayoutTensor[mut=False, DTYPE, layoutIntrincisics](Ks.unsafe_ptr())
    world_to_view_transform_tensor = LayoutTensor[mut=False, DTYPE, layoutViewMatrix](world_to_view_transform.unsafe_ptr())
    # Launch kernel
    ctx.get_device_context().enqueue_function[rasterize_to_pixels_from_world_3dgs_fwd](
        C,
        N,
        means,
        quats,
        scales,
        colors,
        opacities,
        world_to_view_transform,
        Ks_tensor,
        grid_dim=grid_dim, block_dim=block_dim)
@parameter
fn rasterize_to_pixels_from_world_3dgs_fwd(
    C: Int,
    N: Int,
    # n_isects: Int,
    # packed: Int,
    means: LayoutTensor[mut=False, DTYPE, layoutN3],  # [N, 3]
    quats: LayoutTensor[mut=False, DTYPE, layoutN4],   # [N, 4]
    scales: LayoutTensor[mut=False, DTYPE, layoutN3],  # [N, 3]
    colors: LayoutTensor[mut=False, DTYPE, layoutCNCDIM], # [C * N * CDIM]
    opacities: LayoutTensor[mut=False, DTYPE, layoutCN], # [C * N]
    # backgrounds: LayoutTensor[mut=False, DTYPE, layoutCDIM], # [C * CDIM]
    # masks: LayoutTensor[mut=False, DTYPE, layoutCTILE], # [C * TILE * TILE]
    # image_width: Int,
    # image_height: Int,
    # tile_size: Int,
    # tile_width: Int,
    # tile_height: Int,
    # camera model
    world_to_view_transform: LayoutTensor[mut=False, DTYPE, layoutViewMatrix], # [C, 4, 4]
    # viewmats0: LayoutTensor[mut=False, DTYPE, layoutViewMatrix], # [C, 4, 4]
    # viewmats1: LayoutTensor[mut=False, DTYPE, layoutViewMatrix], # [C, 4, 4]
    Ks: LayoutTensor[mut=False, DTYPE, layoutIntrincisics], # [C, 3, 3]
    # camera_model_type: Int,
    # rs_type: Int,
    # radial_coeffs: LayoutTensor[mut=False, DTYPE, layoutC6], # [C, 6]
    # tangential_coeffs: LayoutTensor[mut=False, DTYPE, layoutC2], # [C, 2]
    # thin_prims_coeffs: LayoutTensor[mut=False, DTYPE, layoutC2], # [C, 2]
    # intersections
    # tile_offsets: LayoutTensor[mut=True, DTYPE, layoutCTILE], # [C, TILE, TILE]
    # flatten_ids: LayoutTensor[mut=True, DTYPE, layout_n_isects], # [n_isects]
    # render_colors: LayoutTensor[mut=True, DTYPE, layoutRenderColors], # [C, IMG_H, IMG_W, CDIM]
    # render_alphas: LayoutTensor[mut=True, DTYPE, layoutRenderAlphas], # [C, IMG_H, IMG_W, 1]
    # last_ids: LayoutTensor[mut=True, DTYPE, layoutLastIds], # [C, IMG_H, IMG_W]
):
    var thread_id = block_dim.x * block_idx.x + thread_idx.x
    var thread_id_y = block_dim.y * block_idx.y + thread_idx.y
    # Calculate the pixel coordinates
    var pixel_x = thread_id % IMG_W
    var pixel_y = thread_id_y % IMG_H
    var fx = Ks[0, 0, 0]
    var fy = Ks[0, 1, 0]
    var cx = Ks[0, 1, 2]
    var cy = Ks[0, 1, 2]
    var dir_x = (pixel_x - cx) / fx
    var dir_y = (pixel_y - cy) / fy
    var dir_z = 1.0
    if pixel_x == 0 and pixel_y == 0:
        print(
            Ks[0, 0, 0],
            Ks[0, 0, 1],
            Ks[0, 0, 2],
            Ks[0, 1, 0],
            Ks[0, 1, 1],
            Ks[0, 1, 2],
            Ks[0, 2, 0],
            Ks[0, 2, 1],
            Ks[0, 2, 2])