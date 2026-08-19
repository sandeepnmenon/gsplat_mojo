"""Matrix / quaternion utilities.

These used to hand back `LayoutTensor`s that pointed at a stack array owned by
the function that built them, so every returned matrix dangled the moment it
was returned. They are value types now: `Mat3`/`Mat2` are SIMD-backed, so they
are returned by value, stay in registers inside GPU kernels, and carry no
origin to outlive.

Quaternion layout is `(x, y, z, w)` — the vector part in lanes 0..2 and the
scalar part in lane 3. `quat_to_rotmat` and `rotation_matrix_to_quaternion` are
inverses of each other under that convention.
"""

from std.math import exp, pi, sqrt

from gsplat_kernels.vec import Vec3, Vec4

comptime DTYPE = DType.float32


@fieldwise_init
struct Mat3(Copyable, ImplicitlyCopyable, Movable, Writable):
    """Row-major 3x3 matrix held in the first 9 lanes of a 16-wide SIMD."""

    var m: SIMD[DTYPE, 16]

    def __init__(out self):
        self.m = SIMD[DTYPE, 16](0.0)

    def __getitem__(self, r: Int, c: Int) -> Float32:
        return self.m[r * 3 + c]

    def __setitem__(mut self, r: Int, c: Int, value: Float32):
        self.m[r * 3 + c] = value

    @staticmethod
    def identity() -> Self:
        var out = Self()
        out[0, 0] = 1.0
        out[1, 1] = 1.0
        out[2, 2] = 1.0
        return out

    @staticmethod
    def diagonal(d0: Float32, d1: Float32, d2: Float32) -> Self:
        var out = Self()
        out[0, 0] = d0
        out[1, 1] = d1
        out[2, 2] = d2
        return out

    def row(self, r: Int) -> Vec3:
        return Vec3(self[r, 0], self[r, 1], self[r, 2])

    def col(self, c: Int) -> Vec3:
        return Vec3(self[0, c], self[1, c], self[2, c])

    def __mul__(self, v: Vec3) -> Vec3:
        """Matrix-vector product."""
        return Vec3(self.row(0).dot(v), self.row(1).dot(v), self.row(2).dot(v))

    def write_to(self, mut w: Some[Writer]):
        comptime for r in range(3):
            w.write(self[r, 0], " ", self[r, 1], " ", self[r, 2])
            comptime if r < 2:
                w.write("\n")


@fieldwise_init
struct Mat2(Copyable, ImplicitlyCopyable, Movable, Writable):
    """Row-major 2x2 matrix: lanes are (a, b, c, d)."""

    var m: SIMD[DTYPE, 4]

    def __init__(out self, a: Float32, b: Float32, c: Float32, d: Float32):
        self.m = SIMD[DTYPE, 4](a, b, c, d)

    def __getitem__(self, r: Int, c: Int) -> Float32:
        return self.m[r * 2 + c]

    def __setitem__(mut self, r: Int, c: Int, value: Float32):
        self.m[r * 2 + c] = value

    def det(self) -> Float32:
        return self.m[0] * self.m[3] - self.m[1] * self.m[2]

    def write_to(self, mut w: Some[Writer]):
        w.write(self.m[0], " ", self.m[1], "\n", self.m[2], " ", self.m[3])


def rotation_matrix_to_quaternion(m: Mat3) -> Vec4:
    """Convert a rotation matrix to a quaternion `(x, y, z, w)`."""
    var q = Vec4()
    var trace = m[0, 0] + m[1, 1] + m[2, 2]
    if trace > 0.0:
        var s = sqrt(trace + 1.0) * 2.0
        q[3] = 0.25 * s
        q[0] = (m[2, 1] - m[1, 2]) / s
        q[1] = (m[0, 2] - m[2, 0]) / s
        q[2] = (m[1, 0] - m[0, 1]) / s
    else:
        if m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
            var s = sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
            q[3] = (m[2, 1] - m[1, 2]) / s
            q[0] = 0.25 * s
            q[1] = (m[0, 1] + m[1, 0]) / s
            q[2] = (m[0, 2] + m[2, 0]) / s
        elif m[1, 1] > m[2, 2]:
            var s = sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
            q[3] = (m[0, 2] - m[2, 0]) / s
            q[0] = (m[0, 1] + m[1, 0]) / s
            q[1] = 0.25 * s
            q[2] = (m[1, 2] + m[2, 1]) / s
        else:
            var s = sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
            q[3] = (m[1, 0] - m[0, 1]) / s
            q[0] = (m[0, 2] + m[2, 0]) / s
            q[1] = (m[1, 2] + m[2, 1]) / s
            q[2] = 0.25 * s
    return q


def quat_to_rotmat(q: Vec4) -> Mat3:
    """Convert a quaternion `(x, y, z, w)` to a rotation matrix."""
    var mat = Mat3()

    var q0 = q[0]
    var q1 = q[1]
    var q2 = q[2]
    var q3 = q[3]

    mat[0, 0] = 1.0 - 2.0 * (q1 * q1 + q2 * q2)
    mat[0, 1] = 2.0 * (q0 * q1 - q2 * q3)
    mat[0, 2] = 2.0 * (q0 * q2 + q1 * q3)
    mat[1, 0] = 2.0 * (q0 * q1 + q2 * q3)
    mat[1, 1] = 1.0 - 2.0 * (q0 * q0 + q2 * q2)
    mat[1, 2] = 2.0 * (q1 * q2 - q0 * q3)
    mat[2, 0] = 2.0 * (q0 * q2 - q1 * q3)
    mat[2, 1] = 2.0 * (q1 * q2 + q0 * q3)
    mat[2, 2] = 1.0 - 2.0 * (q0 * q0 + q1 * q1)
    return mat


def transpose(mat: Mat3) -> Mat3:
    var tmat = Mat3()
    comptime for r in range(3):
        comptime for c in range(3):
            tmat[r, c] = mat[c, r]
    return tmat


def matmul3x3(mat1: Mat3, mat2: Mat3) -> Mat3:
    var mat = Mat3()
    comptime for r in range(3):
        comptime for c in range(3):
            mat[r, c] = (
                mat1[r, 0] * mat2[0, c]
                + mat1[r, 1] * mat2[1, c]
                + mat1[r, 2] * mat2[2, c]
            )
    return mat


def g_scalar(
    cov2d: Mat2,
    mean_x: Float32,
    mean_y: Float32,
    point_x: Float32,
    point_y: Float32,
) -> Float32:
    """Evaluate a normalized 2D gaussian with covariance `cov2d` at a point."""
    var det = cov2d.det()
    var inv_a = cov2d[1, 1] / det
    var inv_b = -cov2d[0, 1] / det
    var inv_c = -cov2d[1, 0] / det
    var inv_d = cov2d[0, 0] / det
    var dx = point_x - mean_x
    var dy = point_y - mean_y
    var exponent = -0.5 * (
        dx * (inv_a * dx + inv_b * dy) + dy * (inv_c * dx + inv_d * dy)
    )
    var coeff = 1.0 / (2.0 * Float32(pi) * sqrt(det))
    return coeff * exp(exponent)
