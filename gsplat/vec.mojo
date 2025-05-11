from collections import InlineArray
from math import sqrt

# Point3 is just an alias for Vec3, but useful for geometric clarity in the code.
alias Point3 = Vec3


@value
struct Vec3:
    var e: InlineArray[Float32, 3]

    @always_inline
    fn __init__(out self, x: Float32 = 0, y: Float32 = 0, z: Float32 = 0):
        self.e = InlineArray[Float32, 3](x, y, z)

    @always_inline
    fn __add__(self, rhs: Self) -> Self:
        return Self(
            self.e[0] + rhs.e[0], self.e[1] + rhs.e[1], self.e[2] + rhs.e[2]
        )

    @always_inline
    fn __sub__(self, rhs: Self) -> Self:
        return Self(
            self.e[0] - rhs.e[0], self.e[1] - rhs.e[1], self.e[2] - rhs.e[2]
        )

    @always_inline
    fn __mul__(self, rhs: Self) -> Self:
        return Self(
            self.e[0] * rhs.e[0], self.e[1] * rhs.e[1], self.e[2] * rhs.e[2]
        )

    @always_inline
    fn __mul__(self, t: Float32) -> Self:
        return Self(self.e[0] * t, self.e[1] * t, self.e[2] * t)

    @always_inline
    fn __rmul__(self, t: Float32) -> Self:
        return self * t

    @always_inline
    fn __truediv__(self, t: Float32) -> Self:
        return (1.0 / t) * self

    @always_inline
    fn __neg__(self) -> Self:
        return Self(-self.e[0], -self.e[1], -self.e[2])

    @always_inline
    fn __getitem__[I: Indexer](self, idx: I) -> ref [self.e] Float32:
        return self.e[idx]

    @always_inline
    fn __iadd__(mut self, rhs: Self):
        self.e[0] += rhs.e[0]
        self.e[1] += rhs.e[1]
        self.e[2] += rhs.e[2]

    @always_inline
    fn __imul__(mut self, t: Float32):
        self.e[0] *= t
        self.e[1] *= t
        self.e[2] *= t

    @always_inline
    fn __itruediv__(mut self, t: Float32):
        self.e[0] /= t
        self.e[1] /= t
        self.e[2] /= t

    @always_inline
    fn __ifloordiv__(mut self, t: Float32):
        self *= 1.0 / t

    @always_inline
    fn x(self) -> Float32:
        return self.e[0]

    @always_inline
    fn y(self) -> Float32:
        return self.e[1]

    @always_inline
    fn z(self) -> Float32:
        return self.e[2]

    @always_inline
    fn length(self) -> Float32:
        return sqrt(self.length_squared())

    @always_inline
    fn length_squared(self) -> Float32:
        return (
            self.e[0] * self.e[0]
            + self.e[1] * self.e[1]
            + self.e[2] * self.e[2]
        )

    @always_inline
    fn write_to[W: Writer](self, mut w: W):
        w.write(self.e[0], " ", self.e[1], " ", self.e[2])

    @always_inline
    fn dot(self, rhs: Self) -> Float32:
        return (
            self.e[0] * rhs.e[0] + self.e[1] * rhs.e[1] + self.e[2] * rhs.e[2]
        )

    @always_inline
    fn cross(self, rhs: Self) -> Self:
        return Self(
            self.e[1] * rhs.e[2] - self.e[2] * rhs.e[1],
            self.e[2] * rhs.e[0] - self.e[0] * rhs.e[2],
            self.e[0] * rhs.e[1] - self.e[1] * rhs.e[0],
        )

    @always_inline
    fn unit_vector(self) -> Self:
        return self / self.length()

    @always_inline
    @staticmethod
    fn random_in_unit_disk() -> Self:
        while True:
            p = Self(random_Float32(-1, 1), random_Float32(-1, 1), 0)
            if p.length_squared() < 1:
                return p

    @staticmethod
    @always_inline
    fn random(min: Float32 = 0.0, max: Float32 = 1.0) -> Self:
        return Self(
            random_Float32(min, max),
            random_Float32(min, max),
            random_Float32(min, max),
        )

    @staticmethod
    fn random_unit_vector() -> Self:
        while True:
            p = Self.random(-1, 1)
            lensq = p.length_squared()
            if 1e-160 < lensq <= 1:
                return p / sqrt(lensq)

    @staticmethod
    fn random_on_hemisphere(normal: Vec3) -> Vec3:
        on_unit_sphere = Self.random_unit_vector()
        # In the same hemisphere as the normal
        if on_unit_sphere.dot(normal) > 0.0:
            return on_unit_sphere
        else:
            return -on_unit_sphere

    @always_inline
    fn near_zero(self) -> Bool:
        """Return true if the vector is close to zero in all dimensions."""
        alias s = 1e-8
        return (
            (abs(self.e[0]) < s)
            and (abs(self.e[1]) < s)
            and (abs(self.e[2]) < s)
        )

    @always_inline
    fn reflect(self, n: Vec3) -> Vec3:
        return self - 2 * self.dot(n) * n

    @always_inline
    fn refract(self, n: Self, etai_over_etat: Float32) -> Self:
        cos_theta = min(-self.dot(n), 1.0)
        r_out_perp = etai_over_etat * (self + cos_theta * n)
        r_out_parallel = -sqrt(abs(1.0 - r_out_perp.length_squared())) * n
        return r_out_perp + r_out_parallel



@value
struct Vec4:
    var e: InlineArray[Float32, 4]

    @always_inline
    fn __init__(out self, x: Float32 = 0, y: Float32 = 0, z: Float32 = 0, w: Float32 = 0):
        self.e = InlineArray[Float32, 4](x, y, z, w)

    @always_inline
    fn __add__(self, rhs: Self) -> Self:
        return Self(
            self.e[0] + rhs.e[0], self.e[1] + rhs.e[1], self.e[2] + rhs.e[2], self.e[3] + rhs.e[3]
        )

    @always_inline
    fn __sub__(self, rhs: Self) -> Self:
        return Self(
            self.e[0] - rhs.e[0], self.e[1] - rhs.e[1], self.e[2] - rhs.e[2], self.e[3] - rhs.e[3]
        )

    @always_inline
    fn __mul__(self, rhs: Self) -> Self:
        return Self(
            self.e[0] * rhs.e[0], self.e[1] * rhs.e[1], self.e[2] * rhs.e[2], self.e[3] * rhs.e[3]
        )

    @always_inline
    fn __mul__(self, t: Float32) -> Self:
        return Self(self.e[0] * t, self.e[1] * t, self.e[2] * t, self.e[3] * t)

    @always_inline
    fn __rmul__(self, t: Float32) -> Self:
        return self * t

    @always_inline
    fn __truediv__(self, t: Float32) -> Self:
        return (1.0 / t) * self

    @always_inline
    fn __neg__(self) -> Self:
        return Self(-self.e[0], -self.e[1], -self.e[2])

    @always_inline
    fn __getitem__[I: Indexer](self, idx: I) -> ref [self.e] Float32:
        return self.e[idx]

    @always_inline
    fn __iadd__(mut self, rhs: Self):
        self.e[0] += rhs.e[0]
        self.e[1] += rhs.e[1]
        self.e[2] += rhs.e[2]
        self.e[3] += rhs.e[3]

    @always_inline
    fn __imul__(mut self, t: Float32):
        self.e[0] *= t
        self.e[1] *= t
        self.e[2] *= t
        self.e[3] *= t

    @always_inline
    fn __itruediv__(mut self, t: Float32):
        self.e[0] /= t
        self.e[1] /= t
        self.e[2] /= t
        self.e[3] /= t

    @always_inline
    fn __ifloordiv__(mut self, t: Float32):
        self *= 1.0 / t

    @always_inline
    fn x(self) -> Float32:
        return self.e[0]

    @always_inline
    fn y(self) -> Float32:
        return self.e[1]

    @always_inline
    fn z(self) -> Float32:
        return self.e[2]

    @always_inline
    fn length(self) -> Float32:
        return sqrt(self.length_squared())

    @always_inline
    fn length_squared(self) -> Float32:
        return (
            self.e[0] * self.e[0]
            + self.e[1] * self.e[1]
            + self.e[2] * self.e[2]
            + self.e[3] * self.e[3]
        )

    @always_inline
    fn write_to[W: Writer](self, mut w: W):
        w.write(self.e[0], " ", self.e[1], " ", self.e[2], " ", self.e[3])

    @always_inline
    fn dot(self, rhs: Self) -> Float32:
        return (
            self.e[0] * rhs.e[0] + self.e[1] * rhs.e[1] + self.e[2] * rhs.e[2] + self.e[3] * rhs.e[3]
        )

    @always_inline
    fn cross(self, rhs: Self) -> Self:
        return Self(
            self.e[1] * rhs.e[2] - self.e[2] * rhs.e[1],
            self.e[2] * rhs.e[0] - self.e[0] * rhs.e[2],
            self.e[0] * rhs.e[1] - self.e[1] * rhs.e[0],
        )

    @always_inline
    fn unit_vector(self) -> Self:
        return self / self.length()

    @always_inline
    @staticmethod
    fn random_in_unit_disk() -> Self:
        while True:
            p = Self(random_Float32(-1, 1), random_Float32(-1, 1), 0)
            if p.length_squared() < 1:
                return p

    @staticmethod
    @always_inline
    fn random(min: Float32 = 0.0, max: Float32 = 1.0) -> Self:
        return Self(
            random_Float32(min, max),
            random_Float32(min, max),
            random_Float32(min, max),
        )

    @staticmethod
    fn random_unit_vector() -> Self:
        while True:
            p = Self.random(-1, 1)
            lensq = p.length_squared()
            if 1e-160 < lensq <= 1:
                return p / sqrt(lensq)

    @staticmethod
    fn random_on_hemisphere(normal: Vec3) -> Vec3:
        on_unit_sphere = Self.random_unit_vector()
        # In the same hemisphere as the normal
        if on_unit_sphere.dot(normal) > 0.0:
            return on_unit_sphere
        else:
            return -on_unit_sphere

    @always_inline
    fn near_zero(self) -> Bool:
        """Return true if the vector is close to zero in all dimensions."""
        alias s = 1e-8
        return (
            (abs(self.e[0]) < s)
            and (abs(self.e[1]) < s)
            and (abs(self.e[2]) < s)
            and (abs(self.e[3]) < s)
        )

    @always_inline
    fn reflect(self, n: Vec3) -> Vec3:
        return self - 2 * self.dot(n) * n

    @always_inline
    fn refract(self, n: Self, etai_over_etat: Float32) -> Self:
        cos_theta = min(-self.dot(n), 1.0)
        r_out_perp = etai_over_etat * (self + cos_theta * n)
        r_out_parallel = -sqrt(abs(1.0 - r_out_perp.length_squared())) * n
        return r_out_perp + r_out_parallel
        