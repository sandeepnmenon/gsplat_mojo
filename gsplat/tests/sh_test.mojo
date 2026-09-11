"""Verify the spherical-harmonic colour pre-pass.

`assets/christmas_tree.ply` is degree 0, so it cannot exercise this path at
all. This test drives it directly with degree-3 coefficients and checks:

  1. degree 0 reduces exactly to `C0 * c0 + 0.5`, the constant-colour path
  2. degree 3 matches a host evaluation whose basis constants are written as
     closed forms (`0.5*sqrt(1/pi)` and friends) rather than the decimal
     literals the kernel uses -- so a mistyped constant is caught
  3. the result actually depends on view direction: moving the camera changes
     the colours, which a constant-colour implementation would not do
"""

from std.math import ceildiv, pi, sqrt
from std.sys import has_accelerator
from max.gpu.host import DeviceContext
from layout import TileTensor

from gsplat_kernels.config import (
    C,
    CDIM,
    DTYPE,
    N_MAX,
    SH_COEFFS,
    layout_cn_cdim,
    layout_n3,
    layout_sh,
    layout_viewmats,
)
from gsplat_kernels.spherical_harmonics import compute_colors_from_sh
from scene import rnd

comptime N_SH_TEST = 512
comptime TOL: Float32 = 1e-5


def _shk(sh: List[Float32], g: Int, i: Int, ch: Int) -> Float32:
    return sh[(g * SH_COEFFS + i) * 3 + ch]


def host_sh(
    sh: List[Float32],
    g: Int,
    deg: Int,
    dx: Float32,
    dy: Float32,
    dz: Float32,
    ch: Int,
) -> Float32:
    """Independent SH evaluation; constants come from their closed forms."""
    var c0 = 0.5 * sqrt(1.0 / Float32(pi))
    var c1 = 0.5 * sqrt(3.0 / Float32(pi))
    var c20 = 0.5 * sqrt(15.0 / Float32(pi))
    var c22 = 0.25 * sqrt(5.0 / Float32(pi))
    var c24 = 0.25 * sqrt(15.0 / Float32(pi))
    var c30 = 0.25 * sqrt(35.0 / (2.0 * Float32(pi)))
    var c31 = 0.5 * sqrt(105.0 / Float32(pi))
    var c32 = 0.25 * sqrt(21.0 / (2.0 * Float32(pi)))
    var c33 = 0.25 * sqrt(7.0 / Float32(pi))
    var c35 = 0.25 * sqrt(105.0 / Float32(pi))

    var v = c0 * _shk(sh, g, 0, ch)
    if deg >= 1:
        v += (
            -c1 * dy * _shk(sh, g, 1, ch)
            + c1 * dz * _shk(sh, g, 2, ch)
            - c1 * dx * _shk(sh, g, 3, ch)
        )
        if deg >= 2:
            var xx = dx * dx
            var yy = dy * dy
            var zz = dz * dz
            v += c20 * dx * dy * _shk(sh, g, 4, ch)
            v += -c20 * dy * dz * _shk(sh, g, 5, ch)
            v += c22 * (2.0 * zz - xx - yy) * _shk(sh, g, 6, ch)
            v += -c20 * dx * dz * _shk(sh, g, 7, ch)
            v += c24 * (xx - yy) * _shk(sh, g, 8, ch)
            if deg >= 3:
                v += -c30 * dy * (3.0 * xx - yy) * _shk(sh, g, 9, ch)
                v += c31 * dx * dy * dz * _shk(sh, g, 10, ch)
                v += -c32 * dy * (4.0 * zz - xx - yy) * _shk(sh, g, 11, ch)
                v += (
                    c33
                    * dz
                    * (2.0 * zz - 3.0 * xx - 3.0 * yy)
                    * _shk(sh, g, 12, ch)
                )
                v += -c32 * dx * (4.0 * zz - xx - yy) * _shk(sh, g, 13, ch)
                v += c35 * dz * (xx - yy) * _shk(sh, g, 14, ch)
                v += -c30 * dx * (xx - 3.0 * yy) * _shk(sh, g, 15, ch)
    v += 0.5
    if v < 0.0:
        v = 0.0
    return v


def compare(
    got: List[Float32],
    sh_ref: List[Float32],
    means: List[Float32],
    n: Int,
    deg: Int,
    ex: Float32,
    ey: Float32,
    ez: Float32,
) -> Float32:
    var worst: Float32 = 0.0
    for g in range(n):
        var dx = means[g * 3 + 0] - ex
        var dy = means[g * 3 + 1] - ey
        var dz = means[g * 3 + 2] - ez
        var L = sqrt(dx * dx + dy * dy + dz * dz)
        dx /= L
        dy /= L
        dz /= L
        comptime for ch in range(CDIM):
            var want = host_sh(sh_ref, g, deg, dx, dy, dz, ch)
            var d = abs(got[g * CDIM + ch] - want)
            if d > worst:
                worst = d
    return worst


def main() raises:
    comptime assert has_accelerator(), "requires a GPU"
    var n = N_SH_TEST
    var ctx = DeviceContext()

    var sh_buf = ctx.enqueue_create_buffer[DTYPE](N_MAX * SH_COEFFS * 3)
    var means_buf = ctx.enqueue_create_buffer[DTYPE](N_MAX * 3)
    var view_buf = ctx.enqueue_create_buffer[DTYPE](C * 16)
    var colors_buf = ctx.enqueue_create_buffer[DTYPE](C * N_MAX * CDIM)

    var sh_h = ctx.enqueue_create_host_buffer[DTYPE](N_MAX * SH_COEFFS * 3)
    var means_h = ctx.enqueue_create_host_buffer[DTYPE](N_MAX * 3)
    var view_h = ctx.enqueue_create_host_buffer[DTYPE](C * 16)
    ctx.synchronize()

    # deterministic coefficients and positions
    var sh_ref = List[Float32]()
    for g in range(n):
        for i in range(SH_COEFFS):
            comptime for ch in range(3):
                var v = 2.0 * rnd(g * SH_COEFFS + i, 100 + ch) - 1.0
                sh_h[(g * SH_COEFFS + i) * 3 + ch] = v
                sh_ref.append(v)
        means_h[g * 3 + 0] = 4.0 * rnd(g, 7) - 2.0
        means_h[g * 3 + 1] = 4.0 * rnd(g, 8) - 2.0
        means_h[g * 3 + 2] = 2.0 + 3.0 * rnd(g, 9)

    var eye_x: Float32 = 0.4
    var eye_y: Float32 = -0.3
    var eye_z: Float32 = -1.5

    var means_ref = List[Float32]()
    for g in range(n * 3):
        means_ref.append(means_h[g])

    # identity rotation, so world -> camera is just x_c = x_w - eye
    for c in range(C):
        for e in range(16):
            view_h[c * 16 + e] = 0.0
        view_h[c * 16 + 0] = 1.0
        view_h[c * 16 + 5] = 1.0
        view_h[c * 16 + 10] = 1.0
        view_h[c * 16 + 15] = 1.0
        view_h[c * 16 + 3] = -eye_x
        view_h[c * 16 + 7] = -eye_y
        view_h[c * 16 + 11] = -eye_z

    ctx.enqueue_copy(dst_buf=sh_buf, src_buf=sh_h)
    ctx.enqueue_copy(dst_buf=means_buf, src_buf=means_h)
    ctx.enqueue_copy(dst_buf=view_buf, src_buf=view_h)

    var sh = TileTensor(sh_buf, layout_sh)
    var means = TileTensor(means_buf, layout_n3)
    var viewmats = TileTensor(view_buf, layout_viewmats)
    var colors = TileTensor(colors_buf, layout_cn_cdim)

    comptime TPB = 256
    var blocks = ceildiv(n, TPB)

    # ---- 1. degree 0 must be exactly the constant-colour path ----
    ctx.enqueue_function[compute_colors_from_sh](
        sh,
        means,
        viewmats,
        colors,
        Int32(n),
        Int32(0),
        grid_dim=(blocks, C),
        block_dim=TPB,
    )
    ctx.synchronize()
    var got0 = List[Float32]()
    with colors_buf.map_to_host() as h:
        for i in range(n * CDIM):
            got0.append(h[i])
    var w0 = compare(got0, sh_ref, means_ref, n, 0, eye_x, eye_y, eye_z)
    var c0const = 0.5 * sqrt(1.0 / Float32(pi))
    var worst_const: Float32 = 0.0
    for g in range(n):
        comptime for ch in range(CDIM):
            var want = c0const * sh_ref[g * SH_COEFFS * 3 + ch] + 0.5
            if want < 0.0:
                want = 0.0
            var d = abs(got0[g * CDIM + ch] - want)
            if d > worst_const:
                worst_const = d
    print("degree 0 vs reference        : max err", w0)
    print("degree 0 vs C0*c0+0.5        : max err", worst_const)

    # ---- 2. degree 3 against the closed-form host basis ----
    ctx.enqueue_function[compute_colors_from_sh](
        sh,
        means,
        viewmats,
        colors,
        Int32(n),
        Int32(3),
        grid_dim=(blocks, C),
        block_dim=TPB,
    )
    ctx.synchronize()
    var got3 = List[Float32]()
    with colors_buf.map_to_host() as h:
        for i in range(n * CDIM):
            got3.append(h[i])
    var w3 = compare(got3, sh_ref, means_ref, n, 3, eye_x, eye_y, eye_z)
    print("degree 3 vs closed-form basis: max err", w3)

    # ---- 3. a different viewpoint must give different colours ----
    var ex2: Float32 = -2.0
    var ey2: Float32 = 1.5
    var ez2: Float32 = -4.0
    for c in range(C):
        view_h[c * 16 + 3] = -ex2
        view_h[c * 16 + 7] = -ey2
        view_h[c * 16 + 11] = -ez2
    ctx.enqueue_copy(dst_buf=view_buf, src_buf=view_h)
    ctx.enqueue_function[compute_colors_from_sh](
        sh,
        means,
        viewmats,
        colors,
        Int32(n),
        Int32(3),
        grid_dim=(blocks, C),
        block_dim=TPB,
    )
    ctx.synchronize()
    var got3b = List[Float32]()
    with colors_buf.map_to_host() as h:
        for i in range(n * CDIM):
            got3b.append(h[i])
    var w3b = compare(got3b, sh_ref, means_ref, n, 3, ex2, ey2, ez2)
    var moved: Float32 = 0.0
    for i in range(n * CDIM):
        var d = abs(got3b[i] - got3[i])
        if d > moved:
            moved = d
    print("degree 3 from a second view  : max err", w3b)
    print(
        "colour change between views  :",
        moved,
        "(0 would mean view-independent)",
    )

    if (
        w0 <= TOL
        and worst_const <= TOL
        and w3 <= TOL
        and w3b <= TOL
        and moved > 0.01
    ):
        print(
            "PASS: SH matches the closed-form basis at degree 0 and 3,"
            " and is view-dependent"
        )
    else:
        raise Error("FAIL: spherical harmonics check")
