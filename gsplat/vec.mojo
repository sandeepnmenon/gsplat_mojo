"""Vec3 / Vec4 math primitives.

Backed by SIMD rather than a heap/stack array so the types stay
register-passable and implicitly copyable, which is what GPU kernel code needs.
`Vec3` keeps lane 3 at zero; all 3D math is written over lanes 0..2 explicitly
so a stray `w` can never contaminate a result.
"""

from std.math import sqrt
from std.random import random_float64

comptime DTYPE = DType.float32

# Point3 is just an alias for Vec3, but useful for geometric clarity in the code.
comptime Point3 = Vec3


def _random_f32(min_v: Float32, max_v: Float32) -> Float32:
    return Float32(random_float64(Float64(min_v), Float64(max_v)))


@fieldwise_init
struct Vec3(Copyable, ImplicitlyCopyable, Movable, Writable):
    var e: SIMD[DTYPE, 4]

    def __init__(out self, x: Float32 = 0, y: Float32 = 0, z: Float32 = 0):
        self.e = SIMD[DTYPE, 4](x, y, z, 0.0)

    def __add__(self, rhs: Self) -> Self:
        return Self(e=self.e + rhs.e)

    def __sub__(self, rhs: Self) -> Self:
        return Self(e=self.e - rhs.e)

    def __mul__(self, rhs: Self) -> Self:
        return Self(e=self.e * rhs.e)

    def __mul__(self, t: Float32) -> Self:
        return Self(e=self.e * t)

    def __rmul__(self, t: Float32) -> Self:
        return self * t

    def __truediv__(self, t: Float32) -> Self:
        return self * (1.0 / t)

    def __neg__(self) -> Self:
        return Self(e=-self.e)

    def __getitem__(self, idx: Int) -> Float32:
        return self.e[idx]

    def __setitem__(mut self, idx: Int, value: Float32):
        self.e[idx] = value

    def __iadd__(mut self, rhs: Self):
        self.e += rhs.e

    def __isub__(mut self, rhs: Self):
        self.e -= rhs.e

    def __imul__(mut self, t: Float32):
        self.e *= t

    def __itruediv__(mut self, t: Float32):
        self.e /= t

    def x(self) -> Float32:
        return self.e[0]

    def y(self) -> Float32:
        return self.e[1]

    def z(self) -> Float32:
        return self.e[2]

    def length(self) -> Float32:
        return sqrt(self.length_squared())

    def length_squared(self) -> Float32:
        return self.dot(self)

    def write_to(self, mut w: Some[Writer]):
        w.write(self.e[0], " ", self.e[1], " ", self.e[2])

    def dot(self, rhs: Self) -> Float32:
        return (
            self.e[0] * rhs.e[0] + self.e[1] * rhs.e[1] + self.e[2] * rhs.e[2]
        )

    def cross(self, rhs: Self) -> Self:
        return Self(
            self.e[1] * rhs.e[2] - self.e[2] * rhs.e[1],
            self.e[2] * rhs.e[0] - self.e[0] * rhs.e[2],
            self.e[0] * rhs.e[1] - self.e[1] * rhs.e[0],
        )

    def unit_vector(self) -> Self:
        return self / self.length()

    @staticmethod
    def random_in_unit_disk() -> Self:
        while True:
            var p = Self(_random_f32(-1, 1), _random_f32(-1, 1), 0)
            if p.length_squared() < 1:
                return p

    @staticmethod
    def random(min_v: Float32 = 0.0, max_v: Float32 = 1.0) -> Self:
        return Self(
            _random_f32(min_v, max_v),
            _random_f32(min_v, max_v),
            _random_f32(min_v, max_v),
        )

    @staticmethod
    def random_unit_vector() -> Self:
        while True:
            var p = Self.random(-1, 1)
            var lensq = p.length_squared()
            if lensq > Float32(1e-30) and lensq <= 1.0:
                return p / sqrt(lensq)

    @staticmethod
    def random_on_hemisphere(normal: Vec3) -> Vec3:
        var on_unit_sphere = Self.random_unit_vector()
        # In the same hemisphere as the normal
        if on_unit_sphere.dot(normal) > 0.0:
            return on_unit_sphere
        else:
            return -on_unit_sphere

    def near_zero(self) -> Bool:
        """Return true if the vector is close to zero in all dimensions."""
        comptime s = 1e-8
        return (
            (abs(self.e[0]) < s)
            and (abs(self.e[1]) < s)
            and (abs(self.e[2]) < s)
        )

    def reflect(self, n: Vec3) -> Vec3:
        return self - 2 * self.dot(n) * n

    def refract(self, n: Self, etai_over_etat: Float32) -> Self:
        var cos_theta = min(-self.dot(n), Float32(1.0))
        var r_out_perp = etai_over_etat * (self + cos_theta * n)
        var r_out_parallel = -sqrt(abs(1.0 - r_out_perp.length_squared())) * n
        return r_out_perp + r_out_parallel


@fieldwise_init
struct Vec4(Copyable, ImplicitlyCopyable, Movable, Writable):
    var e: SIMD[DTYPE, 4]

    def __init__(
        out self,
        x: Float32 = 0,
        y: Float32 = 0,
        z: Float32 = 0,
        w: Float32 = 0,
    ):
        self.e = SIMD[DTYPE, 4](x, y, z, w)

    def __add__(self, rhs: Self) -> Self:
        return Self(e=self.e + rhs.e)

    def __sub__(self, rhs: Self) -> Self:
        return Self(e=self.e - rhs.e)

    def __mul__(self, rhs: Self) -> Self:
        return Self(e=self.e * rhs.e)

    def __mul__(self, t: Float32) -> Self:
        return Self(e=self.e * t)

    def __rmul__(self, t: Float32) -> Self:
        return self * t

    def __truediv__(self, t: Float32) -> Self:
        return self * (1.0 / t)

    def __neg__(self) -> Self:
        return Self(e=-self.e)

    def __getitem__(self, idx: Int) -> Float32:
        return self.e[idx]

    def __setitem__(mut self, idx: Int, value: Float32):
        self.e[idx] = value

    def __iadd__(mut self, rhs: Self):
        self.e += rhs.e

    def __isub__(mut self, rhs: Self):
        self.e -= rhs.e

    def __imul__(mut self, t: Float32):
        self.e *= t

    def __itruediv__(mut self, t: Float32):
        self.e /= t

    def x(self) -> Float32:
        return self.e[0]

    def y(self) -> Float32:
        return self.e[1]

    def z(self) -> Float32:
        return self.e[2]

    def w(self) -> Float32:
        return self.e[3]

    def xyz(self) -> Vec3:
        return Vec3(self.e[0], self.e[1], self.e[2])

    def length(self) -> Float32:
        return sqrt(self.length_squared())

    def length_squared(self) -> Float32:
        return self.dot(self)

    def write_to(self, mut w: Some[Writer]):
        w.write(self.e[0], " ", self.e[1], " ", self.e[2], " ", self.e[3])

    def dot(self, rhs: Self) -> Float32:
        return (self.e * rhs.e).reduce_add()

    def unit_vector(self) -> Self:
        return self / self.length()

    @staticmethod
    def random(min_v: Float32 = 0.0, max_v: Float32 = 1.0) -> Self:
        return Self(
            _random_f32(min_v, max_v),
            _random_f32(min_v, max_v),
            _random_f32(min_v, max_v),
            _random_f32(min_v, max_v),
        )

    @staticmethod
    def random_unit_vector() -> Self:
        while True:
            var p = Self.random(-1, 1)
            var lensq = p.length_squared()
            if lensq > Float32(1e-30) and lensq <= 1.0:
                return p / sqrt(lensq)

    def near_zero(self) -> Bool:
        """Return true if the vector is close to zero in all dimensions."""
        comptime s = 1e-8
        return (
            (abs(self.e[0]) < s)
            and (abs(self.e[1]) < s)
            and (abs(self.e[2]) < s)
            and (abs(self.e[3]) < s)
        )
