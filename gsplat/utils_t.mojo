from layout import Layout, LayoutTensor
from math import sqrt

fn rotation_matrix_to_quaternion(m: LayoutTensor[DType.float32, Layout.row_major(3, 3)]) -> LayoutTensor[mut=True, DType.float32, Layout.row_major(4, 1), MutableAnyOrigin]:
    alias layout = Layout.row_major(4, 1)
    var qarray = InlineArray[Scalar[DType.float32], layout.size()](fill=0.0)
    var q = LayoutTensor[DType.float32, layout, MutableAnyOrigin](qarray.unsafe_ptr())
    
    var trace = m[0,0] + m[1,1] + m[2,2]
    if trace > 0.0:
        var s = sqrt(trace + 1.0) * 2.0
        q[3] = 0.25 * s
        q[0] = (m[2,1] - m[1,2]) / s
        q[1] = (m[0,2] - m[2,0]) / s
        q[2] = (m[1,0] - m[0,1]) / s
    else:
        if m[0,0] > m[1,1] and m[0,0] > m[2,2]:
            var s = sqrt(1.0 + m[0,0] - m[1,1] - m[2,2]) * 2.0
            q[3] = (m[2,1] - m[1,2]) / s
            q[0] = 0.25 * s
            q[1] = (m[0,1] + m[1,0]) / s
            q[2] = (m[0,2] + m[2,0]) / s
        elif m[1,1] > m[2,2]:
            var s = sqrt(1.0 + m[1,1] - m[0,0] - m[2,2]) * 2.0
            q[3] = (m[0,2] - m[2,0]) / s
            q[0] = (m[0,1] + m[1,0]) / s
            q[1] = 0.25 * s
            q[2] = (m[1,2] + m[2,1]) / s
        else:
            var s = sqrt(1.0 + m[2,2] - m[0,0] - m[1,1]) * 2.0
            q[3] = (m[1,0] - m[0,1]) / s
            q[0] = (m[0,2] + m[2,0]) / s
            q[1] = (m[1,2] + m[2,1]) / s
            q[2] = 0.25 * s
    return q

fn quat_to_rotmat(q: LayoutTensor[DType.float32, Layout.row_major(4, 1), MutableAnyOrigin]) -> LayoutTensor[mut=True, DType.float32, Layout.row_major(3, 3), MutableAnyOrigin]:
    var _mat = InlineArray[Scalar[DType.float32], 9](fill=0.0)
    var mat = LayoutTensor[DType.float32, Layout.row_major(3, 3), MutableAnyOrigin](_mat.unsafe_ptr())
    
    var q0 = q[0]
    var q1 = q[1]
    var q2 = q[2]
    var q3 = q[3]
    
    mat[0,0] = 1.0 - 2.0 * (q1 * q1 + q2 * q2)
    mat[0,1] = 2.0 * (q0 * q1 - q2 * q3)
    mat[0,2] = 2.0 * (q0 * q2 + q1 * q3)
    mat[1,0] = 2.0 * (q0 * q1 + q2 * q3)
    mat[1,1] = 1.0 - 2.0 * (q0 * q0 + q2 * q2)
    mat[1,2] = 2.0 * (q1 * q2 - q0 * q3)
    mat[2,0] = 2.0 * (q0 * q2 - q1 * q3)
    mat[2,1] = 2.0 * (q1 * q2 + q0 * q3)
    mat[2,2] = 1.0 - 2.0 * (q0 * q0 + q1 * q1)
    return mat 

fn transpose(mat: LayoutTensor[DType.float32, Layout.row_major(3, 3), MutableAnyOrigin]) -> LayoutTensor[mut=True, DType.float32, Layout.row_major(3, 3), MutableAnyOrigin]:
    var _mat = InlineArray[Scalar[DType.float32], 9](fill=0.0)
    var tmat = LayoutTensor[DType.float32, Layout.row_major(3, 3), MutableAnyOrigin](_mat.unsafe_ptr())
    
    tmat[0,0] = mat[0,0]
    tmat[0,1] = mat[1,0]
    tmat[0,2] = mat[2,0]
    tmat[1,0] = mat[0,1]
    tmat[1,1] = mat[1,1]
    tmat[1,2] = mat[2,1]
    tmat[2,0] = mat[0,2]
    tmat[2,1] = mat[1,2]
    tmat[2,2] = mat[2,2]
    return tmat
    
fn matmul3x3(mat1: LayoutTensor[DType.float32, Layout.row_major(3, 3), MutableAnyOrigin], mat2: LayoutTensor[DType.float32, Layout.row_major(3, 3), MutableAnyOrigin]) -> LayoutTensor[mut=True, DType.float32, Layout.row_major(3, 3), MutableAnyOrigin]:
    var _mat = InlineArray[Scalar[DType.float32], 9](fill=0.0)
    var mat = LayoutTensor[DType.float32, Layout.row_major(3, 3), MutableAnyOrigin](_mat.unsafe_ptr())
    
    mat[0,0] = mat1[0,0] * mat2[0,0] + mat1[0,1] * mat2[1,0] + mat1[0,2] * mat2[2,0]
    mat[0,1] = mat1[0,0] * mat2[0,1] + mat1[0,1] * mat2[1,1] + mat1[0,2] * mat2[2,1]
    mat[0,2] = mat1[0,0] * mat2[0,2] + mat1[0,1] * mat2[1,2] + mat1[0,2] * mat2[2,2]
    mat[1,0] = mat1[1,0] * mat2[0,0] + mat1[1,1] * mat2[1,0] + mat1[1,2] * mat2[2,0]
    mat[1,1] = mat1[1,0] * mat2[0,1] + mat1[1,1] * mat2[1,1] + mat1[1,2] * mat2[2,1]
    mat[1,2] = mat1[1,0] * mat2[0,2] + mat1[1,1] * mat2[1,2] + mat1[1,2] * mat2[2,2]
    mat[2,0] = mat1[2,0] * mat2[0,0] + mat1[2,1] * mat2[1,0] + mat1[2,2] * mat2[2,0]
    mat[2,1] = mat1[2,0] * mat2[0,1] + mat1[2,1] * mat2[1,1] + mat1[2,2] * mat2[2,1]
    mat[2,2] = mat1[2,0] * mat2[0,2] + mat1[2,1] * mat2[1,2] + mat1[2,2] * mat2[2,2]
    return mat

alias N = 100
from math import pi, sqrt, exp
fn g_scalar(cov2d: LayoutTensor[DType.float32, Layout.row_major(2, 2), MutableAnyOrigin], mean2d: LayoutTensor[DType.float32, Layout.row_major(N, 2), MutableAnyOrigin], point: InlineArray[Float32, 2]) -> Float32:
    var a = cov2d[0,0]
    var b = cov2d[0,1]
    var c = cov2d[1,0]
    var d = cov2d[1,1]
    var det = a * d - b * c
    var inv_a = d / det
    var inv_b = -b / det
    var inv_c = -c / det
    var inv_d = a / det
    var dx = point[0] - mean2d[0]
    var dy = point[1] - mean2d[1]
    var exponent = -0.5 * (dx * (inv_a * dx + inv_b * dy) + dy * (inv_c * dx + inv_d * dy))
    var coeff = 1.0 / (2.0 * pi * sqrt(det))
    return coeff[0]*exp(exponent[0])