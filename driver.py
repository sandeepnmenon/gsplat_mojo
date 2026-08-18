"""Run the `render` custom op through a MAX graph and check it against the
native Mojo pipeline.

`pixi run render-ply` renders the same scene with the same camera directly in
Mojo and writes `render.ppm`, and that path is verified pixel-by-pixel against
a float64 reference. So the check here is a comparison against that image: if
the custom op wires the pipeline up correctly, the two must agree.

Run from the repo root:  gsplat/.pixi/envs/default/bin/python driver.py
or:                      cd gsplat && pixi run package && pixi run python ../driver.py
"""

from pathlib import Path

import numpy as np
from max.driver import Accelerator, Buffer, accelerator_count
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef, Graph, TensorType, ops

REPO = Path(__file__).parent
PLY = REPO / "assets" / "christmas_tree.ply"
MOJO_SRC = REPO / "gsplat" / "gsplat_kernels.mojoc"
REFERENCE_PPM = REPO / "render.ppm"

IMG_W, IMG_H = 1024, 768
SH_C0 = 0.28209479177387814

# Same camera as operations/render_ply.mojo.
VIEW = np.array(
    [[1.0, 0.0, 0.0, 0.0],
     [0.0, 0.98480777, -0.17364819, -0.86824093],
     [0.0, 0.17364819, 0.98480777, 4.92403887],
     [0.0, 0.0, 0.0, 1.0]], dtype=np.float32,
)[None]
# Computed stepwise in float32 to land on the same bits as the Mojo
# `comptime PLY_FOCAL`. Evaluating this in float64 and rounding once at the
# end differs by a single ulp -- which is enough to visibly change the image,
# because the ray/gaussian intersection is ill-conditioned for small distant
# gaussians and amplifies any input difference.
FOCAL = np.float32(np.float32(309.01933598) * np.float32(IMG_H) / np.float32(256))
KS = np.array(
    [[FOCAL, 0.0, IMG_W * 0.5],
     [0.0, FOCAL, IMG_H * 0.5],
     [0.0, 0.0, 1.0]], dtype=np.float32,
)[None]
assert KS[0, 0, 0].view(np.uint32) == 1147650999, "focal must match Mojo bit for bit"


def read_ply(path):
    """Parse a binary-little-endian all-float32 3DGS PLY with numpy only."""
    raw = path.read_bytes()
    end = raw.index(b"end_header") 
    end = raw.index(b"\n", end) + 1
    header = raw[:end].decode("ascii").splitlines()

    n = 0
    props = []
    for line in header:
        tok = line.split()
        if not tok:
            continue
        if tok[0] == "element" and tok[1] == "vertex":
            n = int(tok[2])
        elif tok[0] == "property":
            assert tok[1] == "float", f"unsupported property type: {line}"
            props.append(tok[2])

    data = np.frombuffer(raw, dtype=np.float32, count=n * len(props), offset=end)
    data = data.reshape(n, len(props))
    col = {name: data[:, i] for i, name in enumerate(props)}

    # Everything stays in float32 so these match the Mojo loader bit for bit;
    # letting numpy promote to float64 here leaves a visible difference in the
    # rendered image, because a slightly different alpha early in a long
    # composite chain shifts everything behind it.
    f32 = np.float32
    means = np.stack([col["x"], col["y"], col["z"]], 1).astype(f32)
    scales = np.exp(np.stack([col[f"scale_{i}"] for i in range(3)], 1).astype(f32))
    opac = (f32(1.0) / (f32(1.0) + np.exp(-col["opacity"].astype(f32)))).astype(f32)
    colors = np.maximum(
        f32(SH_C0) * np.stack([col[f"f_dc_{i}"] for i in range(3)], 1).astype(f32)
        + f32(0.5),
        f32(0.0),
    ).astype(f32)
    # the file stores (w, x, y, z); the kernels take (x, y, z, w)
    q = np.stack([col[f"rot_{i}"] for i in range(4)], 1).astype(f32)
    quats = np.stack([q[:, 1], q[:, 2], q[:, 3], q[:, 0]], 1).astype(f32)
    qn = np.sqrt((quats * quats).sum(1, dtype=f32)).astype(f32)
    quats = (quats / qn[:, None]).astype(f32)

    return means, colors, opac, scales, quats


def main():
    if accelerator_count() == 0:
        raise SystemExit("this op is GPU-only")

    means, colors, opac, scales, quats = read_ply(PLY)
    n = means.shape[0]
    print(f"loaded {n} gaussians from {PLY.name}")

    dev = Accelerator()
    ref = DeviceRef.from_device(dev)
    inputs = [
        TensorType(DType.float32, shape=means.shape, device=ref),
        TensorType(DType.float32, shape=colors.shape, device=ref),
        TensorType(DType.float32, shape=opac.shape, device=ref),
        TensorType(DType.float32, shape=scales.shape, device=ref),
        TensorType(DType.float32, shape=quats.shape, device=ref),
        TensorType(DType.float32, shape=VIEW.shape, device=ref),
        TensorType(DType.float32, shape=KS.shape, device=ref),
    ]

    def forward(*args):
        return ops.custom(
            "render",
            device=ref,
            values=list(args),
            out_types=[
                TensorType(DType.float32, shape=[IMG_H, IMG_W, 3], device=ref)
            ],
        )[0]

    graph = Graph(
        "render_graph", forward, input_types=inputs,
        custom_extensions=[MOJO_SRC],
    )
    print("compiling graph with the render custom op ...")
    model = InferenceSession(devices=[dev]).load(graph)

    bufs = [Buffer.from_numpy(a).to(dev)
            for a in (means, colors, opac, scales, quats, VIEW, KS)]
    print("executing ...")
    out = model.execute(*bufs)[0].to_numpy()
    print("output", out.shape, "range", float(out.min()), float(out.max()))

    img = (np.clip(out, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
    (REPO / "render_customop.ppm").write_bytes(
        b"P6\n%d %d\n255\n" % (IMG_W, IMG_H) + img.tobytes()
    )
    print("wrote render_customop.ppm")

    if not REFERENCE_PPM.exists():
        print("no render.ppm to compare against; run `pixi run render-ply` first")
        return
    raw = REFERENCE_PPM.read_bytes()
    parts = raw.split(b"\n", 3)
    w, h = map(int, parts[1].split())
    assert (w, h) == (IMG_W, IMG_H)
    want = np.frombuffer(parts[3], dtype=np.uint8).reshape(h, w, 3)

    diff = np.abs(img.astype(np.int16) - want.astype(np.int16))
    for t in (0, 1, 2, 4):
        print(f"  channels differing by > {t}: {int((diff > t).sum())}"
              f" of {diff.size}")
    print(f"  max |diff| {int(diff.max())}/255")
    # The two paths run the same kernels on bit-identical gaussian data, so
    # the residual differences come from ulp-level differences in the derived
    # inputs -- e.g. whether `C0 * c + 0.5` is fused into an FMA. Normally that
    # would be invisible, but the ray/gaussian intersection is ill-conditioned
    # for small distant gaussians (see the float64 comparison in
    # tests/render_ply.mojo), so a single ulp can flip a MIN_ALPHA decision and
    # shift everything behind it in a long composite chain. Correcting just the
    # focal length by one ulp already moved this count from 576 to 326.
    #
    # The bar is therefore statistical, not exact: a wiring mistake in the op
    # would move a large fraction of the image, not a few dozen pixels in the
    # densest region.
    off = int((diff > 1).sum())
    frac = off / diff.size
    ys, xs = np.where(diff.max(2) > 4)
    print(f"  pixels with any channel off by > 4/255: {len(ys)} of {IMG_W * IMG_H}")
    if len(ys):
        print(f"  they cluster around y={int(ys.mean())} x={int(xs.mean())},"
              f" the densest part of the scene")
    if frac < 1e-3:
        print(f"PASS: custom op reproduces the native pipeline "
              f"({frac * 100:.4f}% of channels differ by more than 1/255,"
              f" max {int(diff.max())}/255)")
    else:
        raise SystemExit("FAIL: custom op output disagrees with the native render")


if __name__ == "__main__":
    main()
