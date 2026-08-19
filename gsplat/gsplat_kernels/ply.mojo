"""Loader for INRIA-format 3D Gaussian Splatting PLY files.

Handles `format binary_little_endian` with all-float32 vertex properties,
which is what the reference 3DGS trainer emits. The header is parsed straight
out of the byte buffer rather than via `String`, since the file as a whole is
not valid UTF-8.

The stored values are not the rendering parameters -- the trainer keeps them
in unconstrained form so they can be optimized freely. This undoes that:

  * `opacity`   is a logit          -> sigmoid
  * `scale_i`   is log-scale        -> exp
  * `f_dc_i`    is the order-0 SH   -> C0 * f_dc + 0.5
  * `rot_*`     is (w, x, y, z)     -> reordered to (x, y, z, w) and normalized

That last one matters: the file's quaternion convention is not the one
`quat_to_rotmat` uses, and getting it wrong silently rotates every gaussian.
"""

from std.math import exp, sqrt

comptime SH_C0: Float32 = 0.28209479177387814  # order-0 spherical harmonic

# Vertex properties we care about, in the slot order used below.
comptime P_X = 0
comptime P_Y = 1
comptime P_Z = 2
comptime P_DC0 = 3
comptime P_DC1 = 4
comptime P_DC2 = 5
comptime P_OPACITY = 6
comptime P_SCALE0 = 7
comptime P_SCALE1 = 8
comptime P_SCALE2 = 9
comptime P_ROT0 = 10
comptime P_ROT1 = 11
comptime P_ROT2 = 12
comptime P_ROT3 = 13
comptime N_WANTED = 14

comptime WANTED_NAMES = StaticString(
    "x|y|z|f_dc_0|f_dc_1|f_dc_2|opacity|scale_0|scale_1|scale_2|"
    "rot_0|rot_1|rot_2|rot_3|"
)


@fieldwise_init
struct PlyGaussians(Movable):
    """Render-ready gaussian parameters, flattened per attribute."""

    var count: Int
    var means: List[Float32]  # 3 per gaussian
    var quats: List[Float32]  # 4 per gaussian, (x, y, z, w), normalized
    var scales: List[Float32]  # 3 per gaussian, linear
    var colors: List[Float32]  # 3 per gaussian, in [0, 1] — the degree-0 view
    var opacities: List[Float32]  # 1 per gaussian, in [0, 1]
    var sh: List[Float32]  # sh_coeffs * 3 per gaussian, raw coefficients
    var sh_coeffs: Int  # (degree + 1)^2
    var sh_degree: Int


def _find_header_end(raw: List[UInt8]) raises -> Int:
    """Byte offset just past the "end_header\n" line."""
    comptime needle = StaticString("end_header")
    var nb = needle.as_bytes()
    var limit = len(raw) - needle.byte_length()
    if limit > 65536:
        limit = 65536  # the header is tiny; do not scan a 22 MB payload
    for i in range(limit):
        var hit = True
        for k in range(needle.byte_length()):
            if raw[i + k] != nb[k]:
                hit = False
                break
        if hit:
            var j = i + needle.byte_length()
            while j < len(raw) and raw[j] != UInt8(10):  # newline
                j += 1
            return j + 1
    raise Error("ply: no end_header found in the first 64 KiB")


def load_ply(path: String) raises -> PlyGaussians:
    var f = open(path, "r")
    var raw = f.read_bytes()
    f.close()

    var payload_start = _find_header_end(raw)

    # The header is ASCII, so it is safe to lift into a String and parse
    # there; the payload after it is not valid UTF-8 and stays as bytes.
    var header = String("")
    for i in range(payload_start):
        header += chr(Int(raw[i]))

    var n_verts = 0
    var n_props = 0
    var saw_binary_le = False
    var index_of_slot = List[Int]()
    for _ in range(N_WANTED):
        index_of_slot.append(-1)
    var wanted = WANTED_NAMES.split("|")
    # f_rest_<k> holds the higher-order SH coefficients, if the file has any.
    var frest_index = List[Int]()

    for line_slice in header.split("\n"):
        var line = String(line_slice).strip()
        if line.byte_length() == 0:
            continue
        var tok = line.split(" ")
        if len(tok) == 0:
            continue
        var head = String(tok[0])
        if head == "format":
            if len(tok) > 1 and String(tok[1]) == "binary_little_endian":
                saw_binary_le = True
        elif head == "element":
            if len(tok) > 2 and String(tok[1]) == "vertex":
                n_verts = Int(String(tok[2]))
        elif head == "property":
            if len(tok) < 3 or String(tok[1]) != "float":
                raise Error(
                    "ply: only all-float32 vertex properties are supported,"
                    " got: "
                    + line
                )
            var name = String(tok[2])
            for slot in range(len(wanted)):
                if String(wanted[slot]) == name:
                    index_of_slot[slot] = n_props
            if name.startswith("f_rest_"):
                var k = Int(String(name.removeprefix("f_rest_")))
                while len(frest_index) <= k:
                    frest_index.append(-1)
                frest_index[k] = n_props
            n_props += 1
        elif head == "end_header":
            break

    if payload_start < 0:
        raise Error("ply: no end_header")
    if not saw_binary_le:
        raise Error("ply: only binary_little_endian is supported")
    if n_verts <= 0:
        raise Error("ply: no vertices")
    for slot in range(N_WANTED):
        if index_of_slot[slot] < 0:
            raise Error("ply: file is missing a required 3DGS property")

    var need = payload_start + n_verts * n_props * 4
    if need > len(raw):
        raise Error("ply: payload is shorter than the header claims")
    if payload_start % 4 != 0:
        raise Error("ply: payload is not 4-byte aligned")

    # Higher-order SH is optional. The trainer writes features_rest as
    # [N, 3, K-1] flattened, i.e. channel-major, so f_rest_i belongs to
    # channel i // (K-1) at coefficient 1 + i % (K-1).
    var n_frest = len(frest_index)
    if n_frest % 3 != 0:
        raise Error("ply: f_rest_* count is not a multiple of 3")
    for i in range(n_frest):
        if frest_index[i] < 0:
            raise Error("ply: f_rest_* indices are not contiguous")
    var rest_per_channel = n_frest // 3
    var sh_coeffs = rest_per_channel + 1
    var sh_degree = 0
    while (sh_degree + 1) * (sh_degree + 1) < sh_coeffs:
        sh_degree += 1
    if (sh_degree + 1) * (sh_degree + 1) != sh_coeffs:
        raise Error("ply: f_rest_* count does not form a whole SH degree")

    # x86 and the file are both little-endian, so the payload can be read as
    # float32 directly.
    var fp = raw.unsafe_ptr().unsafe_bitcast[Float32]()
    var base_f = payload_start // 4

    var means = List[Float32](capacity=n_verts * 3)
    var quats = List[Float32](capacity=n_verts * 4)
    var scales = List[Float32](capacity=n_verts * 3)
    var colors = List[Float32](capacity=n_verts * 3)
    var opacities = List[Float32](capacity=n_verts)
    var sh = List[Float32](capacity=n_verts * sh_coeffs * 3)

    var ix = index_of_slot[P_X]
    var iy = index_of_slot[P_Y]
    var iz = index_of_slot[P_Z]
    var idc0 = index_of_slot[P_DC0]
    var idc1 = index_of_slot[P_DC1]
    var idc2 = index_of_slot[P_DC2]
    var iop = index_of_slot[P_OPACITY]
    var is0 = index_of_slot[P_SCALE0]
    var is1 = index_of_slot[P_SCALE1]
    var is2 = index_of_slot[P_SCALE2]
    var ir0 = index_of_slot[P_ROT0]
    var ir1 = index_of_slot[P_ROT1]
    var ir2 = index_of_slot[P_ROT2]
    var ir3 = index_of_slot[P_ROT3]

    for v in range(n_verts):
        var b = base_f + v * n_props

        means.append(fp[unsafe_offset=b + ix])
        means.append(fp[unsafe_offset=b + iy])
        means.append(fp[unsafe_offset=b + iz])

        # log-scale -> linear
        scales.append(exp(fp[unsafe_offset=b + is0]))
        scales.append(exp(fp[unsafe_offset=b + is1]))
        scales.append(exp(fp[unsafe_offset=b + is2]))

        # (w, x, y, z) in the file -> (x, y, z, w) here, normalized
        var qw = fp[unsafe_offset=b + ir0]
        var qx = fp[unsafe_offset=b + ir1]
        var qy = fp[unsafe_offset=b + ir2]
        var qz = fp[unsafe_offset=b + ir3]
        var qn = sqrt(qw * qw + qx * qx + qy * qy + qz * qz)
        if qn <= 0.0:
            qx = 0.0
            qy = 0.0
            qz = 0.0
            qw = 1.0
            qn = 1.0
        quats.append(qx / qn)
        quats.append(qy / qn)
        quats.append(qz / qn)
        quats.append(qw / qn)

        # Raw SH coefficients, coefficient-major: [coeff][channel].
        sh.append(fp[unsafe_offset=b + idc0])
        sh.append(fp[unsafe_offset=b + idc1])
        sh.append(fp[unsafe_offset=b + idc2])
        for c in range(rest_per_channel):
            for ch in range(3):
                sh.append(
                    fp[unsafe_offset=b + frest_index[ch * rest_per_channel + c]]
                )

        # order-0 SH -> linear RGB
        colors.append(
            (SH_C0 * fp[unsafe_offset=b + idc0] + 0.5).clamp(0.0, 1.0)
        )
        colors.append(
            (SH_C0 * fp[unsafe_offset=b + idc1] + 0.5).clamp(0.0, 1.0)
        )
        colors.append(
            (SH_C0 * fp[unsafe_offset=b + idc2] + 0.5).clamp(0.0, 1.0)
        )

        # logit -> probability
        opacities.append(1.0 / (1.0 + exp(-fp[unsafe_offset=b + iop])))

    return PlyGaussians(
        count=n_verts,
        means=means^,
        quats=quats^,
        scales=scales^,
        colors=colors^,
        opacities=opacities^,
        sh=sh^,
        sh_coeffs=sh_coeffs,
        sh_degree=sh_degree,
    )
