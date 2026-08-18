"""Deterministic test scene shared by the rasterizer and intersection tests.

No RNG state: every quantity is a pure function of the gaussian index, so the
device staging code and the host reference always agree, and a failure is
reproducible.
"""

from std.math import cos, sin, sqrt

from gsplat_kernels.config import CDIM, DTYPE

comptime GOLDEN_ANGLE: Float32 = 2.399963  # spreads points evenly on a disc


def rnd(g: Int, salt: Int) -> Float32:
    """Deterministic pseudo-random value in [0, 1)."""
    var x = (g * 1103515 + salt * 12347 + 4711) % 100003
    return Float32(x) / 100003.0


# ---------------------------------------------------------------------------
# Scene A: isotropic gaussians stacked on the optical axis, identity camera.
# The one configuration with a short closed form for the rendered image.
# Opacities are low so all N composite before transmittance saturates.
# ---------------------------------------------------------------------------
def axis_z(g: Int) -> Float32:
    return 2.0 + 0.15 * Float32(g)


def axis_scale(g: Int) -> Float32:
    return 0.04 + 0.004 * Float32(g)


def axis_opacity(g: Int) -> Float32:
    return 0.10 + 0.02 * rnd(g, 3)


# ---------------------------------------------------------------------------
# Scene B: rotated, anisotropic gaussians spread across the frame, viewed by a
# yawed and translated camera. Exercises the quaternion, anisotropy, off-axis
# and camera-pose paths, and gives the tile binning a non-trivial workload.
# ---------------------------------------------------------------------------
def spread_depth(g: Int) -> Float32:
    return 2.0 + 3.0 * rnd(g, 11)


def spread_mean(g: Int, axis: Int, n_total: Int) -> Float32:
    """Mean placed so the gaussian lands at a spiral position on screen."""
    var z = spread_depth(g)
    var ang = Float32(g) * GOLDEN_ANGLE
    var rad = 0.55 * sqrt(Float32(g + 1) / Float32(n_total))
    if axis == 0:
        return rad * cos(ang) * z
    elif axis == 1:
        return rad * sin(ang) * z * 0.75  # frame is wider than it is tall
    return z


def spread_scale(g: Int, axis: Int) -> Float32:
    var base = 0.035 + 0.05 * rnd(g, 20 + axis)
    return base * spread_depth(g) * 0.4


def spread_quat(g: Int, comp: Int) -> Float32:
    """Unnormalized quaternion component, (x, y, z, w)."""
    var v = 2.0 * rnd(g, 40 + comp) - 1.0
    if comp == 3:
        v += Float32(1.0)  # bias w so the quaternion is never near-degenerate
    return v


def spread_quat_norm(g: Int) -> Float32:
    var acc: Float32 = 0.0
    for c in range(4):
        var v = spread_quat(g, c)
        acc += v * v
    return sqrt(acc)


def spread_opacity(g: Int) -> Float32:
    return 0.25 + 0.7 * rnd(g, 60)


def gauss_color(g: Int, k: Int) -> Float32:
    return 0.15 + 0.8 * rnd(g, 70 + k)


# ---------------------------------------------------------------------------
# Camera pose for scene B: yaw about y, plus a small translation.
# Stored world -> camera, i.e. x_cam = R x_world + t.
# ---------------------------------------------------------------------------
comptime CAM_YAW: Float32 = 0.15


def view_rot(r: Int, c: Int) -> Float32:
    var ca = cos(CAM_YAW)
    var sa = sin(CAM_YAW)
    if r == 0:
        return ca if c == 0 else (sa if c == 2 else 0.0)
    elif r == 1:
        return 1.0 if c == 1 else 0.0
    return -sa if c == 0 else (ca if c == 2 else 0.0)


def view_trans(a: Int) -> Float32:
    if a == 0:
        return 0.10
    elif a == 1:
        return -0.05
    return 0.20
