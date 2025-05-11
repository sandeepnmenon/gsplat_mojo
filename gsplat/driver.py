from pathlib import Path
from max.driver import CPU, Accelerator, Tensor, accelerator_count
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef, Graph, TensorType, ops
import numpy as np
import matplotlib.pyplot as plt
import plyfile
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm
# https://en.wikipedia.org/wiki/Table_of_spherical_harmonics
C0 = (1/2)*np.sqrt(1/np.pi)
def read_ply(file_path):
    ply_data = plyfile.PlyData.read(file_path)
    vertex_data = ply_data['vertex']
    means = np.stack([
        vertex_data['x'],
        vertex_data['y'],
        vertex_data['z'],
    ], axis=1)
    colors = np.stack([
        vertex_data['f_dc_0'],
        vertex_data['f_dc_1'],
        vertex_data['f_dc_2'],
    ], axis=1) * C0 + 0.5
    opacities = vertex_data['opacity']
    scales = np.stack([
        vertex_data['scale_0'],
        vertex_data['scale_1'],
        vertex_data['scale_2'],
    ], axis=1)
    quats = np.stack([
        vertex_data['rot_0'],
        vertex_data['rot_1'],
        vertex_data['rot_2'],
        vertex_data['rot_3'],
    ], axis=1)
    return means.astype(np.float32), colors.astype(np.float32), opacities.astype(np.float32), scales.astype(np.float32), quats.astype(np.float32)
def output_image(image: Tensor, path: str):
    """Save the image to a file."""
    image_np = image.to_numpy()
    print(image_np)
    plt.imshow(image_np)
    plt.axis("off")
    plt.savefig(path, bbox_inches="tight", pad_inches=0)
    plt.close()
def create_render_graph(
    width: int,
    height: int,
    means: Tensor,
    colors: Tensor,
    opacities: Tensor,
    scales: Tensor,
    quats: Tensor,
    world_to_view_transform: Tensor,
    Ks: Tensor,
    device: DeviceRef,
) -> Graph:
    """Configure a graph to run a Mandelbrot kernel."""
    output_dtype = DType.float32
    mojo_kernels = Path(__file__).parent / "operations"
    # Configure our simple one-operation graph.
    graph = Graph(
        "render_graph",
        # The custom Mojo operation is referenced by its string name, and we
        # need to provide inputs as a list as well as expected output types.
        forward=lambda means, colors, opacities, scales, quats, world_to_view_transform, Ks: ops.custom(
            name="gsplat_forward",
            values=[means, colors, opacities, scales, quats, world_to_view_transform, Ks],
            out_types=[
                TensorType(
                    dtype=DType.float32,
                    shape=[height, width, 4],
                    device=DeviceRef.from_device(device),
                )
            ],
        )[0].tensor,
        input_types=[
            TensorType(
                DType.float32,
                shape=means.shape,
                device=DeviceRef.from_device(device),
            ),
            TensorType(
                DType.float32,
                shape=colors.shape,
                device=DeviceRef.from_device(device),
            ),
            TensorType(
                DType.float32,
                shape=opacities.shape,
                device=DeviceRef.from_device(device),
            ),
            TensorType(
                DType.float32,
                shape=scales.shape,
                device=DeviceRef.from_device(device),
            ),
            TensorType(
                DType.float32,
                shape=quats.shape,
                device=DeviceRef.from_device(device),
            ),
            TensorType(
                DType.float32,
                shape=[1, 4, 4],
                device=DeviceRef.from_device(device),
            ),
            TensorType(
                DType.float32,
                shape=[1, 3, 3],
                device=DeviceRef.from_device(device),
            )
        ],
        custom_extensions=[mojo_kernels],
    )
    # Set up an inference session for running the graph.
    session = InferenceSession(devices=[device])
    # Compile the graph.
    model = session.load(graph)
       # Perform the calculation on the target device.
    result = model.execute(means, colors, opacities, scales, quats, world_to_view_transform, Ks)[0]
    # Copy values back to the CPU to be read.
    assert isinstance(result, Tensor)
    result = result.to(CPU())
    return result
def get_cov_3d(scales, quats):
    """
    Compute the 3D covariance matrix for each gaussian
    """
    cov_3d = np.zeros((scales.shape[0], 3, 3), dtype=np.float32)
    for i in tqdm(range(scales.shape[0])):
        scale = scales[i]
        quat = quats[i]
        cov_3d[i] = np.diag(scale) @ R.from_quat(quat).as_matrix()
    # cov_3d = scales @ np.array([R.from_quat(quat).as_matrix() for quat in quats])
    return cov_3d
if __name__ == "__main__":
    # Establish Mandelbrot set ranges.
    WIDTH = 1920
    HEIGHT = 1080
    device = CPU() if accelerator_count() == 0 else Accelerator()
    means, color, opacity, scale, quats = read_ply("christmas_tree.ply")
    covs = get_cov_3d(scale, quats)
    print(means.shape, color.shape, opacity.shape, scale.shape, quats.shape)
    print("covs", covs.shape)
    means = Tensor.from_numpy(means).to(device)
    color = Tensor.from_numpy(color).to(device)
    opacity = Tensor.from_numpy(opacity).to(device)
    scale = Tensor.from_numpy(scale).to(device)
    quats = Tensor.from_numpy(quats).to(device)
    print(means.shape, color.shape, opacity.shape, scale.shape, quats.shape)
    world_to_view_transform = np.array([[
        [ 1.,          0. ,         0.    ,      0.     ,   ],
        [ 0.   ,       0.98480777 ,-0.17364819 ,-0.86824093],
        [ 0.   ,       0.17364819 , 0.98480777 , 4.92403887],
        [ 0.   ,       0.       ,   0.       ,   1.    ,    ]]]).astype(np.float32)
    world_to_view_transform = Tensor.from_numpy(world_to_view_transform).to(device)
    Ks = np.array([[
        [309.01933598 ,  0.     ,    128.     ,   ],
        [  0.      ,   309.01933598, 128.        ],
        [  0.     ,      0.       ,    1.        ]]]).astype(np.float32)
    Ks = Tensor.from_numpy(Ks).to(device)
    result = create_render_graph(
        WIDTH,
        HEIGHT,
        means,
        color,
        opacity,
        scale,
        quats,
        world_to_view_transform,
        Ks,
        device,
    )
    output_image(result, "render.png")