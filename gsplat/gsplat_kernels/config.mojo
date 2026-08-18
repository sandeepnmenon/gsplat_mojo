"""Shared scene/image configuration and tensor layouts.

Kept in one module so the rasterizer and the tile-intersection stage agree on
sizes and layouts without importing each other.
"""

from std.bit import log2_floor
from std.math import ceildiv
from layout import row_major

comptime DTYPE = DType.float32
comptime IDTYPE = DType.int32
comptime KDTYPE = DType.uint64  # packed (tile, depth) sort key

# Scene / image configuration.
comptime C = 1  # number of cameras
# Capacity, not the live count. Every N-shaped buffer is allocated for this
# many gaussians and the real count travels as a runtime argument, so one
# build serves both the small synthetic scenes and a full PLY.
comptime N_MAX = 400000

# Size of the synthetic scenes used by the self-checks. Unrelated to N_MAX,
# which is only the allocation capacity.
comptime N_TEST = 24

# The intersection test uses a bigger scene than the rasterizer test: it needs
# enough intersections to span several radix blocks (RADIX_EPB each), so that
# cross-block stability of the sort is actually exercised.
comptime N_ISECT_TEST = 160
comptime CDIM = 3  # color channels (RGB)
comptime IMG_W = 1024
comptime IMG_H = 768
comptime TILE = 16  # tile edge, in pixels

# One thread per pixel in a tile.
comptime BLOCK_SIZE = TILE * TILE

# Alpha-compositing cutoffs, matching gsplat.
comptime MIN_ALPHA = 1.0 / 255.0  # below this a gaussian cannot tint a u8 pixel
comptime MAX_ALPHA = 0.999  # keep some light through even a solid gaussian
comptime T_EPS = 1e-4  # transmittance at which the pixel is saturated

# Tile grid covering the image.
comptime N_TILES_X = ceildiv(IMG_W, TILE)
comptime N_TILES_Y = ceildiv(IMG_H, TILE)
comptime N_TILES = N_TILES_X * N_TILES_Y

# Only a layout bound -- nothing is allocated from it. The intersection
# buffers are sized at runtime once the scan reports the true total.
comptime MAX_ISECTS = 1 << 25

# Camera intrinsics. CX/CY are offset by half a pixel so that pixel
# (IMG_W/2, IMG_H/2) has its *centre* exactly on the optical axis -- without
# that no ray is ever perfectly axial and rho^2 never reaches 0.
comptime FOCAL: Float32 = 600.0
comptime CX: Float32 = Float32(IMG_W) * 0.5 + 0.5
comptime CY: Float32 = Float32(IMG_H) * 0.5 + 0.5

# Near plane: gaussians closer than this are culled.
comptime Z_NEAR: Float32 = 0.2

# Prefix-sum geometry. The scan is two-level: each block scans SCAN_BLOCK
# elements, then one block scans the per-block totals. That caps the input at
# SCAN_BLOCK * SCAN_WIDTH elements, checked by a comptime assert at the call
# site.
comptime SCAN_WIDTH = 1024
comptime SCAN_STEPS = 10  # log2(SCAN_WIDTH)
comptime SCAN_BLOCK = SCAN_WIDTH
comptime SCAN_NUM_BLOCKS = ceildiv(C * N_MAX, SCAN_BLOCK)
comptime SCAN_CAPACITY = SCAN_BLOCK * SCAN_WIDTH

# LSD radix sort geometry. Digits are RADIX_BITS wide; only the bits the key
# actually uses are passed over. The key is (tile << 32) | float_bits(depth),
# so that is 32 depth bits plus however many the tile index needs.
comptime RADIX_BITS = 4
comptime RADIX = 1 << RADIX_BITS
comptime RADIX_TPB = 256
comptime RADIX_EPT = 8  # elements per thread
comptime RADIX_EPB = RADIX_TPB * RADIX_EPT  # elements per block
comptime TILE_BITS = log2_floor(C * N_TILES - 1) + 1
comptime KEY_BITS = 32 + TILE_BITS
comptime RADIX_PASSES = ceildiv(KEY_BITS, RADIX_BITS)
comptime layout_radix_sh = row_major[RADIX * RADIX_TPB]()

# Spherical harmonics. Degree 3 is what the reference 3DGS trainer emits at
# most, giving (3+1)^2 = 16 coefficients per colour channel. Files with a
# lower degree are zero-padded to this stride so one layout serves all of them.
comptime SH_DEGREE_MAX = 3
comptime SH_COEFFS = (SH_DEGREE_MAX + 1) * (SH_DEGREE_MAX + 1)
comptime layout_sh = row_major[N_MAX, SH_COEFFS, 3]()

comptime layout_n3 = row_major[N_MAX, 3]()
comptime layout_n4 = row_major[N_MAX, 4]()
comptime layout_n = row_major[N_MAX]()
comptime layout_n_i = row_major[N_MAX]()
comptime layout_cn_cdim = row_major[C, N_MAX, CDIM]()
comptime layout_cn = row_major[C, N_MAX]()
comptime layout_cdim = row_major[C, CDIM]()
comptime layout_tiles = row_major[C, N_TILES_Y, N_TILES_X]()
comptime layout_tiles_flat = row_major[C * N_TILES]()
comptime layout_viewmats = row_major[C, 4, 4]()
comptime layout_intrinsics = row_major[C, 3, 3]()
comptime layout_c6 = row_major[C, 6]()
comptime layout_c2 = row_major[C, 2]()
comptime layout_render_colors = row_major[C, IMG_H, IMG_W, CDIM]()
comptime layout_render_alphas = row_major[C, IMG_H, IMG_W, 1]()
comptime layout_last_ids = row_major[C, IMG_H, IMG_W]()
comptime layout_isects = row_major[MAX_ISECTS]()

# Per-(camera, gaussian) scratch for the tile-intersection stage.
comptime layout_cn_i = row_major[C, N_MAX]()  # counts, int32
comptime layout_cn_f = row_major[C, N_MAX]()  # camera-space depth, float32
comptime layout_cn4 = row_major[C, N_MAX, 4]()  # tile bbox (tx0, ty0, tx1, ty1)
comptime layout_one = row_major[1]()  # total intersection count
comptime layout_cn_flat = row_major[C * N_MAX]()  # counts/offsets, flattened
comptime layout_blocksums = row_major[SCAN_NUM_BLOCKS]()

