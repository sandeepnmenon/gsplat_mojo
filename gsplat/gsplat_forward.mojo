from gpu import thread_idx, block_idx, block_dim, barrier
from gpu.host import DeviceContext
from gpu.memory import AddressSpace
from layout import Layout, LayoutTensor
from layout.tensor_builder import LayoutTensorBuild as tb
from sys import sizeof
from memory import UnsafePointer, stack_allocation
from vec import Vec3, Vec4
from utils_t import rotation_matrix_to_quaternion, quat_to_rotmat, transpose, matmul3x3

alias DTYPE = DType.float32

# Kernel parameters (placeholders for illustration)
alias C = 1      # number of cameras
alias N = 1      # number of gaussians
alias CDIM = 3   # color channels (RGB)
alias IMG_W = 1024
alias IMG_H = 768
alias TILE = 16   # tile size

# Shared memory: for 'block_size' threads per block
alias block_size = TILE * TILE

alias n_isects = 3

alias layoutN3 = Layout.row_major(N, 3)
alias layoutN4 = Layout.row_major(N, 4)
alias layoutCNCDIM = Layout.row_major(C, N, CDIM)
alias layoutCN = Layout.row_major(C, N)
alias layoutCDIM = Layout.row_major(C, CDIM)
alias layoutCTILE = Layout.row_major(C, TILE, TILE)
alias layoutViewMatrix = Layout.row_major(C, 4, 4)
alias layoutIntrincisics = Layout.row_major(C, 3, 3)
alias layoutC6 = Layout.row_major(C, 6)
alias layoutC4 = Layout.row_major(C, 4)
alias layoutC2 = Layout.row_major(C, 2)
alias layoutRenderColors = Layout.row_major(C, IMG_H, IMG_W, CDIM)
alias layoutRenderAlphas = Layout.row_major(C, IMG_H, IMG_W, 1)
alias layoutLastIds = Layout.row_major(C, IMG_H, IMG_W)
alias layout_n_isects = Layout.row_major(n_isects)

@value
struct SE3:
    var rotation: LayoutTensor[mut=True, DTYPE, Layout.row_major(3, 3), MutableAnyOrigin]
    var translation: Vec3

fn extract_se3(se3_matrix: LayoutTensor[DTYPE, Layout.row_major(3, 4)]) -> SE3:
    alias layout = Layout.row_major(3, 3)
    var rotation = InlineArray[Scalar[DTYPE], layout.size()](fill=0.0)
    var rotation_tensor = LayoutTensor[DTYPE, layout, MutableAnyOrigin](rotation.unsafe_ptr())
    @parameter
    for i in range(3):
        @parameter
        for j in range(3):
            rotation_tensor[i, j] = se3_matrix[i, j]

    # Extract translation vector (last column)
    var translation = Vec3()
    @parameter
    for i in range(3):
        translation.e[i] = se3_matrix[Int(i), 3][0]

    return SE3(rotation=rotation_tensor, translation=translation)


struct RollingShutterParameters:
    var t_start: Vec3
    var q_start: LayoutTensor[mut=False, DTYPE, layoutN4, MutableAnyOrigin] 
    var t_end: Vec3
    var q_end: LayoutTensor[mut=False, DTYPE, layoutN4, MutableAnyOrigin]

    fn __init__(out self,se3_start: LayoutTensor[DType.float32, Layout.row_major(3, 4)],
                      se3_end: LayoutTensor[DType.float32, Layout.row_major(3, 4)]):
        var start = extract_se3(se3_start)
        self.t_start = start.translation
        self.q_start = rotation_matrix_to_quaternion(start.rotation)

        if se3_end.ptr:
            self.t_end = self.t_start
            self.q_end = self.q_start
        else:
            var end = extract_se3(se3_end)
            self.t_end = end.translation
            self.q_end = rotation_matrix_to_quaternion(end.rotation)
        
    
# GPU Kernel: rasterize gaussians to pixels
fn rasterize_to_pixels_from_world_3dgs_fwd(
    C: Int,
    N: Int,
    n_isects: Int32,
    packed: Int,
    means: LayoutTensor[mut=False, DTYPE, layoutN3],  # [N, 3]
    quats: LayoutTensor[mut=False, DTYPE, layoutN4],   # [N, 4]
    scales: LayoutTensor[mut=False, DTYPE, layoutN3],  # [N, 3]
    colors: LayoutTensor[mut=False, DTYPE, layoutCNCDIM], # [C * N * CDIM]
    opacities: LayoutTensor[mut=False, DTYPE, layoutCN], # [C * N]
    backgrounds: LayoutTensor[mut=False, DTYPE, layoutCDIM], # [C * CDIM]
    masks: LayoutTensor[mut=False, DTYPE, layoutCTILE], # [C * TILE * TILE]
    image_width: Int,
    image_height: Int,
    tile_size: Int,
    tile_width: Int,
    tile_height: Int,
    # camera model
    viewmats0: LayoutTensor[mut=False, DTYPE, layoutViewMatrix], # [C, 4, 4]
    viewmats1: LayoutTensor[mut=False, DTYPE, layoutViewMatrix], # [C, 4, 4]
    Ks: LayoutTensor[mut=False, DTYPE, layoutIntrincisics], # [C, 3, 3]
    camera_model_type: Int,
    rs_type: Int, 
    radial_coeffs: LayoutTensor[mut=False, DTYPE, layoutC6], # [C, 6]
    tangential_coeffs: LayoutTensor[mut=False, DTYPE, layoutC2], # [C, 2]
    thin_prims_coeffs: LayoutTensor[mut=False, DTYPE, layoutC2], # [C, 2]
    # intersections
    tile_offsets: LayoutTensor[mut=True, DTYPE.int32, layoutCTILE], # [C, TILE, TILE]
    flatten_ids: LayoutTensor[mut=True, DTYPE, layout_n_isects], # [n_isects]
    render_colors: LayoutTensor[mut=True, DTYPE, layoutRenderColors], # [C, IMG_H, IMG_W, CDIM]
    render_alphas: LayoutTensor[mut=True, DTYPE, layoutRenderAlphas], # [C, IMG_H, IMG_W, 1]
    last_ids: LayoutTensor[mut=True, DTYPE, layoutLastIds], # [C, IMG_H, IMG_W]
):
    cid = block_idx.x
    tile_id = block_idx.y * tile_width + block_idx.z
    i = block_idx.y * tile_size + thread_idx.y
    j = block_idx.z * tile_size + thread_idx.x

    # CUDA
    # tile_offsets += cid * tile_height * tile_width
    # render_colors += cid * image_height * image_width * CDIM
    # render_alphas += cid * image_height * image_width
    # last_ids += cid * image_height * image_width
    # Mojo
    # tile_offsets[cid, tile_height, tile_width]
    # render_colors[cid, image_height, image_width, CDIM]
    # render_alphas[cid, image_height, image_width, 1]
    # last_ids[cid, image_height, image_width]

    # backgrounds[cid, CDIM]
    # masks[cid, tile_height, tile_width]

    px = j+0.5
    py = i+0.5
    pix_id = i * image_width + j
    
    # rs_params = RollingShutterParameters(
    #     viewmats0 + cid * 16,
    #     viewmats1 == nullptr ? nullptr : viewmats1 + cid * 16
    # )

    focal_length = (Ks[cid, 0,0] , Ks[cid, 1,1])
    principal_point = (Ks[cid, 0,2], Ks[cid, 1,2])

    rayd = (0.0, 0.0, 1.0)
    rayo = (0.0, 0.0, 0.0)
    inside = 0
    if (i < image_height and j < image_width):
        inside = 1
    done = not inside

    if( inside and not masks[tile_id]):
        for k in range(CDIM):
            if(backgrounds.ptr):
                render_colors[pix_id, k] = 0
            else:
                render_colors[pix_id, k] = backgrounds[cid, k]
        return
    
    range_start = tile_offsets[cid, tile_id, tile_id]
    if (cid == C - 1) and (tile_id == tile_width * tile_height - 1):
        range_end = n_isects
    else:
        range_end = rebind[Int32](tile_offsets[cid, tile_id + 1, tile_id + 1])
        
    num_batches = (range_end - range_start + block_size - 1) / block_size

    var id_batch = stack_allocation[
        block_size,
        DTYPE,
        address_space = AddressSpace.SHARED,
    ]()
    var xyz_opacity_batch = stack_allocation[
        block_size,
        Vec4,
        address_space = AddressSpace.SHARED,
    ]()
    var _iscl_rot_batch = stack_allocation[
        block_size*3*3,
        DTYPE,
        address_space = AddressSpace.SHARED,
    ]()
    var iscl_rot_batch = LayoutTensor[DTYPE, Layout.row_major(block_size,3, 3), _, address_space=AddressSpace.SHARED](_iscl_rot_batch)

    T = 1.0
    cur_idx = 0

    tr = thread_idx.x + thread_idx.y * block_dim.x + thread_idx.z * block_dim.x * block_dim.y
    
    pix_out = InlineArray[Scalar[DTYPE], CDIM](fill=0.0)
    for b in range(num_batches):
        # if (__syncthreads_count(done) >= block_size):
        #     break
        
        batch_start = range_start + block_size * b
        idx = batch_start + tr

        if (idx < range_end):
            g = flatten_ids[Int(idx)]
            id_batch[tr] = g[0]
            xyz = means[Int(g)]
            opac = opacities[Int(g)][0]
            xyz_ptr = xyz_opacity_batch.bitcast[Float32]()
            xyz_ptr[tr * 4 + 0] = xyz[0]
            xyz_ptr[tr * 4 + 1] = xyz[1]
            xyz_ptr[tr * 4 + 2] = xyz[2]
            xyz_ptr[tr * 4 + 3] = opac

            quat = quats[Int(g)]
            scale = scales[Int(g)]

            R = quat_to_rotmat(quat)
            S = LayoutTensor[mut=True, DTYPE, Layout.row_major(3, 3), MutableAnyOrigin](InlineArray[Scalar[DTYPE], 9](
                1.0 / scale[0],
                0.0,
                0.0,
                0.0,
                1.0 / scale[1],
                0.0,
                0.0,
                0.0,
                1.0 / scale[2]
            ).unsafe_ptr())
            iscl_rot = matmul3x3(S, transpose(R))
            iscl_rot_batch[tr] = iscl_rot

            barrier()

            batch_size = min(block_size, range_end - batch_start)
            t=0
            while t < Int(batch_size) and not done:
                xyz_opac = xyz_opacity_batch[t]
                opac = xyz_opac.e[3]
                xyz = Vec3(xyz_opac.e[0], xyz_opac.e[1], xyz_opac.e[2])
                # iscl_rot = iscl_rot_batch[t]

                t += 1
    
def main():
    with DeviceContext() as ctx:
        # Allocate buffers (empty)
        means_buf = ctx.enqueue_create_buffer[DTYPE](N * 3)
        quats_buf = ctx.enqueue_create_buffer[DTYPE](N * 4)
        scales_buf = ctx.enqueue_create_buffer[DTYPE](N * 3)
        colors_buf = ctx.enqueue_create_buffer[DTYPE](C * N * CDIM)
        opac_buf = ctx.enqueue_create_buffer[DTYPE](C * N)
        backgrounds_buf = ctx.enqueue_create_buffer[DTYPE](C * CDIM)
        masks_buf = ctx.enqueue_create_buffer[DTYPE](C * TILE * TILE)
        viewmats0_buf = ctx.enqueue_create_buffer[DTYPE](C * 4 * 4)
        viewmats1_buf = ctx.enqueue_create_buffer[DTYPE](C * 4 * 4)
        Ks_buf = ctx.enqueue_create_buffer[DTYPE](C * 3 * 3)
        radial_coeffs_buf = ctx.enqueue_create_buffer[DTYPE](C * 6)
        tangential_coeffs_buf = ctx.enqueue_create_buffer[DTYPE](C * 2)
        thin_prims_coeffs_buf = ctx.enqueue_create_buffer[DTYPE](C * 2)
        tile_offsets_buf = ctx.enqueue_create_buffer[DTYPE.int32](C * TILE * TILE)
        flatten_ids_buf = ctx.enqueue_create_buffer[DTYPE](n_isects)
        renders_buf = ctx.enqueue_create_buffer[DTYPE](C * IMG_H * IMG_W * CDIM)
        alphas_buf = ctx.enqueue_create_buffer[DTYPE](C * IMG_H * IMG_W)
        ids_buf = ctx.enqueue_create_buffer[DTYPE](C * IMG_H * IMG_W)

        means_tensor = LayoutTensor[DTYPE, layoutN3, MutableAnyOrigin](means_buf.unsafe_ptr())
        quats_tensor = LayoutTensor[DTYPE, layoutN4, MutableAnyOrigin](quats_buf.unsafe_ptr())
        scales_tensor = LayoutTensor[DTYPE, layoutN3, MutableAnyOrigin](scales_buf.unsafe_ptr())
        colors_tensor = LayoutTensor[DTYPE, layoutCNCDIM, MutableAnyOrigin](colors_buf.unsafe_ptr())
        opac_tensor = LayoutTensor[DTYPE, layoutCN, MutableAnyOrigin](opac_buf.unsafe_ptr())
        backgrounds_tensor = LayoutTensor[DTYPE, layoutCDIM, MutableAnyOrigin](backgrounds_buf.unsafe_ptr())
        masks_tensor = LayoutTensor[DTYPE, layoutCTILE, MutableAnyOrigin](masks_buf.unsafe_ptr())
        viewmats0_tensor = LayoutTensor[DTYPE, layoutViewMatrix, MutableAnyOrigin](viewmats0_buf.unsafe_ptr())
        viewmats1_tensor = LayoutTensor[DTYPE, layoutViewMatrix, MutableAnyOrigin](viewmats1_buf.unsafe_ptr())
        Ks_tensor = LayoutTensor[DTYPE, layoutIntrincisics, MutableAnyOrigin](Ks_buf.unsafe_ptr())
        radial_coeffs_tensor = LayoutTensor[DTYPE, layoutC6, MutableAnyOrigin](radial_coeffs_buf.unsafe_ptr())
        tangential_coeffs_tensor = LayoutTensor[DTYPE, layoutC2, MutableAnyOrigin](tangential_coeffs_buf.unsafe_ptr())
        thin_prims_coeffs_tensor = LayoutTensor[DTYPE, layoutC2, MutableAnyOrigin](thin_prims_coeffs_buf.unsafe_ptr())
        tile_offsets_tensor = LayoutTensor[DTYPE.int32, layoutCTILE, MutableAnyOrigin](tile_offsets_buf.unsafe_ptr())
        flatten_ids_tensor = LayoutTensor[DTYPE, layout_n_isects, MutableAnyOrigin](flatten_ids_buf.unsafe_ptr())
        render_colors_tensor = LayoutTensor[DTYPE, layoutRenderColors, MutableAnyOrigin](renders_buf.unsafe_ptr())
        render_alphas_tensor = LayoutTensor[DTYPE, layoutRenderAlphas, MutableAnyOrigin](alphas_buf.unsafe_ptr())
        last_ids_tensor = LayoutTensor[DTYPE, layoutLastIds, MutableAnyOrigin](ids_buf.unsafe_ptr())

        # Define grid and block dimensions
        grid_dim = ( (IMG_W + TILE - 1) // TILE,
                         (IMG_H + TILE - 1) // TILE,
                         C )
        block_dim = (TILE, TILE, 1)

        # Launch kernel
        ctx.enqueue_function[rasterize_to_pixels_from_world_3dgs_fwd](
            C,
            N,
            n_isects,
            0,  # packed
            means_tensor,
            quats_tensor,
            scales_tensor,
            colors_tensor,
            opac_tensor,
            backgrounds_tensor,
            masks_tensor,
            IMG_W,
            IMG_H,
            TILE,
            TILE,  # tile_width
            TILE,  # tile_height
            # camera model
            viewmats0_tensor,
            viewmats1_tensor,
            Ks_tensor,
            0,  # camera_model_type
            0,  # rs_type
            radial_coeffs_tensor,
            tangential_coeffs_tensor,
            thin_prims_coeffs_tensor,
            # intersections
            tile_offsets_tensor,
            flatten_ids_tensor,
            render_colors_tensor,
            render_alphas_tensor,
            last_ids_tensor
        , grid_dim=grid_dim, block_dim=block_dim)

        ctx.synchronize()
        # (Optionally map to host for inspection)
        with renders_buf.map_to_host() as render_host:
            print("Render buffer:", render_host)