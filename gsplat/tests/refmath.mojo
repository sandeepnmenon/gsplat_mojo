"""Host-side reference for the ray/gaussian intersection.

Deliberately written with plain scalars, straight from the definitions, so it
shares no code with the kernel's `Mat3`/`Vec3` path and can independently
catch an error in `quat_to_rotmat`, `transpose` or `matmul3x3`.
"""


def _ref_rho2(
    mx: Float32, my: Float32, mz: Float32,
    qx: Float32, qy: Float32, qz: Float32, qw: Float32,
    sx: Float32, sy: Float32, sz: Float32,
    ox: Float32, oy: Float32, oz: Float32,
    dx: Float32, dy: Float32, dz: Float32,
) -> Tuple[Float32, Float32]:
    """Host reference for the ray/gaussian intersection.

    Deliberately written out with plain scalars, straight from the
    definitions, so it shares no code with the kernel's Mat3/Vec3 path and
    can catch an error in `quat_to_rotmat`, `transpose` or `matmul3x3`.
    Returns `(rho^2, t_star)`.
    """
    # Rotation matrix for the quaternion (x, y, z, w).
    var r00 = 1.0 - 2.0 * (qy * qy + qz * qz)
    var r01 = 2.0 * (qx * qy - qz * qw)
    var r02 = 2.0 * (qx * qz + qy * qw)
    var r10 = 2.0 * (qx * qy + qz * qw)
    var r11 = 1.0 - 2.0 * (qx * qx + qz * qz)
    var r12 = 2.0 * (qy * qz - qx * qw)
    var r20 = 2.0 * (qx * qz - qy * qw)
    var r21 = 2.0 * (qy * qz + qx * qw)
    var r22 = 1.0 - 2.0 * (qx * qx + qy * qy)

    # M = S^-1 R^T, i.e. M[r][c] = R[c][r] / s[r].
    var m00 = r00 / sx
    var m01 = r10 / sx
    var m02 = r20 / sx
    var m10 = r01 / sy
    var m11 = r11 / sy
    var m12 = r21 / sy
    var m20 = r02 / sz
    var m21 = r12 / sz
    var m22 = r22 / sz

    var ex = ox - mx
    var ey = oy - my
    var ez = oz - mz

    var og0 = m00 * ex + m01 * ey + m02 * ez
    var og1 = m10 * ex + m11 * ey + m12 * ez
    var og2 = m20 * ex + m21 * ey + m22 * ez

    var dg0 = m00 * dx + m01 * dy + m02 * dz
    var dg1 = m10 * dx + m11 * dy + m12 * dz
    var dg2 = m20 * dx + m21 * dy + m22 * dz

    var dd = dg0 * dg0 + dg1 * dg1 + dg2 * dg2
    if dd <= 1e-20:
        return (Float32(0.0), Float32(-1.0))

    var t_star = -(og0 * dg0 + og1 * dg1 + og2 * dg2) / dd
    var p0 = og0 + t_star * dg0
    var p1 = og1 + t_star * dg1
    var p2 = og2 + t_star * dg2
    return (p0 * p0 + p1 * p1 + p2 * p2, t_star)


def _ref_rho2_f64(
    mx: Float64, my: Float64, mz: Float64,
    qx: Float64, qy: Float64, qz: Float64, qw: Float64,
    sx: Float64, sy: Float64, sz: Float64,
    ox: Float64, oy: Float64, oz: Float64,
    dx: Float64, dy: Float64, dz: Float64,
) -> Tuple[Float64, Float64]:
    """Float64 twin of `_ref_rho2`, used to size the float32 noise floor.

    The ray/gaussian intersection ends in `p = og + t_star * dg`, where both
    terms are large and nearly cancel whenever a gaussian is small relative to
    its distance from the camera. That cancellation costs several digits, so
    a float32 evaluation carries real uncertainty; comparing against this
    tells us how much.
    """
    var r00 = 1.0 - 2.0 * (qy * qy + qz * qz)
    var r01 = 2.0 * (qx * qy - qz * qw)
    var r02 = 2.0 * (qx * qz + qy * qw)
    var r10 = 2.0 * (qx * qy + qz * qw)
    var r11 = 1.0 - 2.0 * (qx * qx + qz * qz)
    var r12 = 2.0 * (qy * qz - qx * qw)
    var r20 = 2.0 * (qx * qz - qy * qw)
    var r21 = 2.0 * (qy * qz + qx * qw)
    var r22 = 1.0 - 2.0 * (qx * qx + qy * qy)

    var m00 = r00 / sx
    var m01 = r10 / sx
    var m02 = r20 / sx
    var m10 = r01 / sy
    var m11 = r11 / sy
    var m12 = r21 / sy
    var m20 = r02 / sz
    var m21 = r12 / sz
    var m22 = r22 / sz

    var ex = ox - mx
    var ey = oy - my
    var ez = oz - mz

    var og0 = m00 * ex + m01 * ey + m02 * ez
    var og1 = m10 * ex + m11 * ey + m12 * ez
    var og2 = m20 * ex + m21 * ey + m22 * ez

    var dg0 = m00 * dx + m01 * dy + m02 * dz
    var dg1 = m10 * dx + m11 * dy + m12 * dz
    var dg2 = m20 * dx + m21 * dy + m22 * dz

    var dd = dg0 * dg0 + dg1 * dg1 + dg2 * dg2
    if dd <= 1e-40:
        return (Float64(0.0), Float64(-1.0))

    var t_star = -(og0 * dg0 + og1 * dg1 + og2 * dg2) / dd
    var p0 = og0 + t_star * dg0
    var p1 = og1 + t_star * dg1
    var p2 = og2 + t_star * dg2
    return (p0 * p0 + p1 * p1 + p2 * p2, t_star)
