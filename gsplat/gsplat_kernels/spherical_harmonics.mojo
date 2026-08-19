"""View-dependent colour from spherical-harmonic coefficients.

3DGS stores each gaussian's colour as SH coefficients rather than a fixed RGB,
so the colour depends on the direction the gaussian is seen from. This is a
per-camera pre-pass: it evaluates the SH for the current view and writes plain
RGB into the same `colors` buffer the rasterizer already reads, which keeps the
rasterizer itself unchanged.

Degree 0 reduces to `C0 * c0 + 0.5`, exactly the constant-colour path, so a
degree-0 file renders identically whether or not this pass runs.
"""

from std.gpu import block_idx, global_idx
from std.math import sqrt
from layout import TileTensor

from gsplat_kernels.config import (
    CDIM,
    DTYPE,
    SH_COEFFS,
    layout_cn_cdim,
    layout_n3,
    layout_sh,
    layout_viewmats,
)
from gsplat_kernels.utils_t import Mat3, transpose
from gsplat_kernels.vec import Vec3

# Real spherical harmonic basis constants, matching the reference 3DGS.
comptime SH_C0: Float32 = 0.28209479177387814
comptime SH_C1: Float32 = 0.4886025119029199
comptime SH_C2_0: Float32 = 1.0925484305920792
comptime SH_C2_1: Float32 = -1.0925484305920792
comptime SH_C2_2: Float32 = 0.31539156525252005
comptime SH_C2_3: Float32 = -1.0925484305920792
comptime SH_C2_4: Float32 = 0.5462742152960396
comptime SH_C3_0: Float32 = -0.5900435899266435
comptime SH_C3_1: Float32 = 2.890611442640554
comptime SH_C3_2: Float32 = -0.4570457994644658
comptime SH_C3_3: Float32 = 0.3731763325901154
comptime SH_C3_4: Float32 = -0.4570457994644658
comptime SH_C3_5: Float32 = 1.445305721320277
comptime SH_C3_6: Float32 = -0.5900435899266435


def _sh_coeff(
    sh: TileTensor[DTYPE, type_of(layout_sh), MutAnyOrigin], g: Int, i: Int
) -> Vec3:
    return Vec3(
        rebind[Scalar[DTYPE]](sh[g, i, 0]),
        rebind[Scalar[DTYPE]](sh[g, i, 1]),
        rebind[Scalar[DTYPE]](sh[g, i, 2]),
    )


def compute_colors_from_sh(
    sh: TileTensor[
        DTYPE, type_of(layout_sh), MutAnyOrigin
    ],  # [N, SH_COEFFS, 3]
    means: TileTensor[DTYPE, type_of(layout_n3), MutAnyOrigin],
    viewmats: TileTensor[DTYPE, type_of(layout_viewmats), MutAnyOrigin],
    colors: TileTensor[DTYPE, type_of(layout_cn_cdim), MutAnyOrigin],
    n_gaussians: Int32,
    degree: Int32,
):
    comptime assert sh.flat_rank == 3
    comptime assert colors.flat_rank == 3
    comptime assert CDIM == 3, "SH evaluation assumes RGB"

    var g = Int(global_idx.x)
    var cid = Int(block_idx.y)
    if g >= Int(n_gaussians):
        return

    # Camera centre in world space: viewmats is world -> camera, so the eye is
    # at -R^T t.
    var rot = Mat3()
    comptime for r in range(3):
        comptime for c in range(3):
            rot[r, c] = rebind[Scalar[DTYPE]](viewmats[cid, r, c])
    var tv = Vec3(
        rebind[Scalar[DTYPE]](viewmats[cid, 0, 3]),
        rebind[Scalar[DTYPE]](viewmats[cid, 1, 3]),
        rebind[Scalar[DTYPE]](viewmats[cid, 2, 3]),
    )
    var eye = -(transpose(rot) * tv)

    var mean = Vec3(
        rebind[Scalar[DTYPE]](means[g, 0]),
        rebind[Scalar[DTYPE]](means[g, 1]),
        rebind[Scalar[DTYPE]](means[g, 2]),
    )
    var d = mean - eye
    var dlen = d.length()
    if dlen > 0.0:
        d = d / dlen

    var deg = Int(degree)

    var acc = _sh_coeff(sh, g, 0) * SH_C0

    if deg >= 1:
        var x = d[0]
        var y = d[1]
        var z = d[2]
        acc += _sh_coeff(sh, g, 1) * (-SH_C1 * y)
        acc += _sh_coeff(sh, g, 2) * (SH_C1 * z)
        acc += _sh_coeff(sh, g, 3) * (-SH_C1 * x)

        if deg >= 2:
            var xx = x * x
            var yy = y * y
            var zz = z * z
            var xy = x * y
            var yz = y * z
            var xz = x * z
            acc += _sh_coeff(sh, g, 4) * (SH_C2_0 * xy)
            acc += _sh_coeff(sh, g, 5) * (SH_C2_1 * yz)
            acc += _sh_coeff(sh, g, 6) * (SH_C2_2 * (2.0 * zz - xx - yy))
            acc += _sh_coeff(sh, g, 7) * (SH_C2_3 * xz)
            acc += _sh_coeff(sh, g, 8) * (SH_C2_4 * (xx - yy))

            if deg >= 3:
                acc += _sh_coeff(sh, g, 9) * (SH_C3_0 * y * (3.0 * xx - yy))
                acc += _sh_coeff(sh, g, 10) * (SH_C3_1 * xy * z)
                acc += _sh_coeff(sh, g, 11) * (
                    SH_C3_2 * y * (4.0 * zz - xx - yy)
                )
                acc += _sh_coeff(sh, g, 12) * (
                    SH_C3_3 * z * (2.0 * zz - 3.0 * xx - 3.0 * yy)
                )
                acc += _sh_coeff(sh, g, 13) * (
                    SH_C3_4 * x * (4.0 * zz - xx - yy)
                )
                acc += _sh_coeff(sh, g, 14) * (SH_C3_5 * z * (xx - yy))
                acc += _sh_coeff(sh, g, 15) * (SH_C3_6 * x * (xx - 3.0 * yy))

    # The trainer's convention: SH encodes colour offset from mid grey, and
    # negative radiance is clipped away.
    comptime for k in range(CDIM):
        var v = acc[k] + 0.5
        if v < 0.0:
            v = 0.0
        colors[cid, g, k] = v
