"""
openICE (Image Correction and Enhancement) implementation
Ported from nkscan/src/dust.rs to Python for PSIX
"""

import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass

# ----- constants

# The maximum value of a 16-bit sample
M: float = 65535.0

# The raw IR value above which film counts as "clear"
TAU: float = 8847.23

# Only consider dye leakage between these values (+/-). Another strange Nikon constant.
SLOPE_LIMIT: float = 0.2

# Weight floor
W_FLOOR: float = 0.02

# Clean-film margin anchor: `b = D(floor(0.98M)) - M`
B_ANCHOR: float = 0.98

# Dust floor anchor: `phi = D(floor(0.065M))`
PHI_ANCHOR: float = 0.065

# What resolution we switch to the weird nikon horizontally-adjecent pixel thing
MIN3_DPI: int = 550

# Detail-band gain, indexed by [`Quality`]
DETAIL_GAIN: List[float] = [1.25, 1.0]

# Dither amplitude per channel
ALPHA: List[float] = [0.015, 0.015, 0.025]

# How many samples one rayon task takes on a whole-plane pass
CHUNK: int = 1 << 16


def anchor(fraction: float) -> int:
    """`floor(fraction * M)`"""
    return int(M * fraction)


def density(v: float) -> float:
    """`D(v) = M/16 * log2(v + 1)`"""
    return np.log2(v + 1.0) * (M / 16.0)


def from_density_scalar(d: float) -> int:
    """`D^-1(d) = 2^(16d/M) - 1`, rounded and clamped to a 16-bit sample"""
    v = np.exp2(d * (16.0 / M)) - 1.0
    return int(np.round(np.clip(v, 0.0, M)))


def to_density(samples: np.ndarray) -> np.ndarray:
    """Step 1: a whole plane of samples in log-density"""
    # Precompute LUT for performance
    lut = np.array([density(float(i)) for i in range(65536)], dtype=np.float32)
    return lut[samples]


def from_density(values: np.ndarray) -> np.ndarray:
    """Step 9: a whole plane of densities back to linear samples"""
    lut = np.array([from_density_scalar(float(v)) for v in np.linspace(0, 65535/16.0, 65536)], dtype=np.uint16)
    # This is approximate - for better precision we'd need to invert the density function properly
    # But for now, let's use the direct computation
    return np.clip(np.exp2(values * (16.0 / M)) - 1.0, 0, 65535).astype(np.uint16)


# ----- Model and Quality enums

class Model:
    Ls9000 = 0
    Ls5000 = 1
    Ls50 = 2


class Quality:
    Normal = 0
    Fine = 1


@dataclass
class Profile:
    """The constants that differ between scanners"""
    theta: float  # Gate bias
    gamma: List[float]  # IR-reference gain per channel [R, G, B]
    dither: List[float]  # Dither band edges, as fractions of full scale [lo, hi]
    contrast_dpi: int  # At or below this resolution beta collapses to the center
    ramp: float  # Weight-ramp anchor, `D(floor(0.85M)) - M`
    a: List[List[Tuple[float, float]]]  # Soft-threshold coefficients, `[[channel][band] = (a_hi, a_lo)]`


# Indexed by [`Model`]
PROFILES: List[Profile] = [
    # Kind 7, LS-9000
    Profile(
        theta=0.0,
        gamma=[1.100, 1.100, 1.100],
        dither=[0.04, 0.96],
        contrast_dpi=950,
        ramp=-960.42,
        a=[
            [(1.360, 1.320), (1.370, 1.300), (1.340, 1.250)],
            [(1.370, 1.300), (1.350, 1.290), (1.300, 1.240)],
            [(1.340, 1.250), (1.320, 1.250), (1.250, 1.210)],
        ],
    ),
    # Kind 8, LS-5000
    Profile(
        theta=1.0,
        gamma=[1.100, 1.100, 1.100],
        dither=[0.01, 0.99],
        contrast_dpi=1600,
        ramp=-960.42,
        a=[
            [(1.210, 1.090), (1.170, 1.080), (1.040, 0.960)],
            [(1.230, 1.130), (1.140, 1.050), (0.930, 0.840)],
            [(1.130, 1.040), (1.080, 1.020), (0.970, 0.890)],
        ],
    ),
    # Kind 9, LS-50
    Profile(
        theta=1.0,
        gamma=[1.000, 1.000, 1.000],
        dither=[0.01, 0.99],
        contrast_dpi=2500,
        ramp=-960.52,
        a=[
            [(2.210, 2.090), (2.170, 2.080), (2.040, 1.960)],
            [(2.230, 2.130), (2.140, 2.050), (1.930, 1.840)],
            [(2.130, 2.040), (2.080, 2.020), (1.970, 1.890)],
        ],
    ),
]


# ----- Geometry constants

# Row spans `(dy, dx_lo, dx_hi)` of the 9x9 octagonal box: `max(|dx|,|dy|) <= 4` and `|dx| + |dy| <= 6`
LEVEL0: List[Tuple[int, int, int]] = [
    (-4, -2, 2),
    (-3, -3, 3),
    (-2, -4, 4),
    (-1, -4, 4),
    (0, -4, 4),
    (1, -4, 4),
    (2, -4, 4),
    (3, -3, 3),
    (4, -2, 2),
]

# The 5x5 octagon: `max(|dx|,|dy|) <= 2` and `|dx| + |dy| <= 3`
LEVEL1: List[Tuple[int, int, int]] = [(-2, -1, 1), (-1, -2, 2), (0, -2, 2), (1, -2, 2), (2, -1, 1)]

# The 3x3 binomial tent, weighted `T[dy] * T[dx] / 16`
TENT: List[float] = [1.0, 2.0, 1.0]


@dataclass
class Options:
    """What to run the pipeline as"""
    model: int  # Model enum value
    quality: int  # Quality enum value
    dpi: int  # scan resolution
    metering_target: float  # Fraction of full scale this crate's AE metered the IR channel to.


@dataclass
class Params:
    """All the parameters needed to complete the pass"""
    c: float  # R->IR leakage slope
    ir_ref: float  # Clear-film IR density with the leakage taken out
    theta: float  # Gate bias
    ramp_bias: float  # `IR_ref + b`
    ramp_s: float  # `s`, the ramp's reciprocal slope
    phi: float  # Dust floor
    eta: List[float]  # Dither band edges, in density [lo, hi]
    gamma: List[float]  # IR-reference gain per channel
    a: List[List[Tuple[float, float]]]  # Soft-threshold coefficients
    detail_gain: float  # Detail-band gain
    clamp_l3: bool  # Clamp to L_3 (Off in Fine)
    min3: bool  # Feed `w` the minimum of three horizontal gates instead
    cross_beta: bool  # Measure beta over the 5-point cross rather than the center

    @classmethod
    def new(cls, opts: Options, cal) -> 'Params':
        profile = PROFILES[opts.model]
        return cls(
            c=cal.c,
            ir_ref=cal.ir_ref,
            theta=profile.theta + theta_for_metering_target(opts.metering_target),
            ramp_bias=cal.ir_ref + density(anchor(B_ANCHOR)) - M,
            ramp_s=1.0 / profile.ramp,
            phi=density(anchor(PHI_ANCHOR)),
            eta=[density(anchor(f)) for f in profile.dither],
            gamma=profile.gamma,
            a=profile.a,
            detail_gain=DETAIL_GAIN[opts.quality],
            clamp_l3=(opts.quality == Quality.Normal),
            min3=(opts.dpi > MIN3_DPI),
            cross_beta=(opts.dpi > profile.contrast_dpi),
        )

    def weight(self, gate: float) -> float:
        """`w = clamp(1 + (IR_ref + b - g)s, w_floor, 1)`"""
        return np.clip(1.0 + (self.ramp_bias - gate) * self.ramp_s, W_FLOOR, 1.0)


def theta_for_metering_target(target: float) -> float:
    """Compute the gate-bias term for our AE target."""
    return density(anchor(np.clip(target, 0.0, 1.0))) - M


# ----- Calibration

@dataclass
class Prescan:
    """The low-resolution view of the frame calibration measures against"""
    red: np.ndarray
    ir: np.ndarray
    rows: int
    cols: int


@dataclass
class Calibration:
    """The IR calibration terms measured off a prescan"""
    c: float  # R->IR leakage slope
    ir_ref: float  # Clear-film IR density with that leakage removed


def calibrate(prescan: Prescan) -> Optional[Calibration]:
    """
    Step 2: measure `c` and `IR_ref` from a low-resolution scan of the frame.
    
    Returns None when the prescan holds no clear film to measure against,
    which would otherwise divide by zero and poison the whole pass.
    """
    # Log-densities of the red and IR channels
    d_r = to_density(prescan.red)
    d_ir = to_density(prescan.ir)

    # 1. The two reference levels, IR^2-weighted so the mean leans toward the clearest pixels
    # Find all the "clear" film by thresholding from TAU
    clear_mask = prescan.ir > TAU
    if not np.any(clear_mask):
        return None
        
    # Weighted average using IR^2 as weight
    weights = prescan.ir[clear_mask].astype(np.float64) ** 2
    num_r = np.sum(d_r[clear_mask] * weights)
    num_ir = np.sum(d_ir[clear_mask] * weights)
    den = np.sum(weights)
    
    if den == 0.0:
        return None
    
    # Average red density of the IR-clear pixels
    r_ref = num_r / den
    # Average IR density of the same, but contaminated with red leakage
    ir_raw = num_ir / den

    # 2. The dye->IR crosstalk, a weighted least-squares slope over the 4x4 quadrants 
    # of every 8x8 tile that is clear film all the way through
    rows, cols = prescan.rows, prescan.cols
    col_tiles = cols // 8
    
    if col_tiles == 0 or rows < 8:
        # Not enough tiles, assume no leak
        c = 0.0
    else:
        num = 0.0
        den = 0.0
        
        for tile_y in range(rows // 8):
            for tile_x in range(col_tiles):
                row0 = tile_y * 8
                col0 = tile_x * 8
                
                # Extract 8x8 tile
                tile_ir = prescan.ir[row0:row0+8, col0:col0+8]
                
                # Only process 8x8 tiles that are "clear"
                if not np.all(tile_ir > TAU):
                    continue
                    
                # Extract red and IR densities for this tile
                tile_r = d_r[row0:row0+8, col0:col0+8]
                tile_ir_d = d_ir[row0:row0+8, col0:col0+8]
                
                # The four 4x4 quadrants (subtiles) of the 8x8 tile
                corners = [(0, 0), (0, 4), (4, 0), (4, 4)]
                for dy0, dx0 in corners:
                    # Extract 4x4 quadrant
                    q_r = tile_r[dy0:dy0+4, dx0:dx0+4]
                    q_ir = tile_ir_d[dy0:dy0+4, dx0:dx0+4]
                    
                    # Calculate means
                    mean_r = np.mean(q_r)
                    mean_ir = np.mean(q_ir)
                    tile_mean_r = np.mean(tile_r)
                    tile_mean_ir = np.mean(tile_ir_d)
                    
                    # Calculate deltas
                    delta_r = mean_r - tile_mean_r
                    delta_ir = mean_ir - tile_mean_ir
                    
                    if abs(delta_r) > 1e-6:  # Avoid division by zero
                        slope = delta_ir / delta_r
                        # Throw out the obvious outliers
                        if np.isfinite(slope) and abs(slope) <= SLOPE_LIMIT:
                            weight = (delta_r * delta_r) * (np.mean(tile_ir) ** 2)
                            num += slope * weight
                            den += weight
        
        # No clear tile survived the slope filter
        # So, assume no leak rather than hand back a NaN that flags the whole frame
        c = num / den if den > 0.0 else 0.0

    ir_ref = (ir_raw - c * r_ref) / (1.0 - c) if c != 1.0 else ir_raw
    
    return Calibration(c=c, ir_ref=ir_ref)


# ----- Gate

def gate(red: np.ndarray, ir: np.ndarray, p: Params) -> np.ndarray:
    """
    Strip the dye crosstalk out of IR, leaving `g`, which responds to defects only.
    Fused with step 1, so red never needs a density plane of its own to avoid an alloc
    """
    debug_assert(red.shape == ir.shape, "one red and one IR sample per pixel")
    
    # Densities add, so killing the dye is a subtraction, not a division
    c, inv_c = p.c, 1.0 / (1.0 - p.c)
    g = (to_density(ir) - c * to_density(red)) * inv_c - p.theta
    
    return g


def debug_assert(condition: bool, msg: str):
    """Simple debug assertion"""
    if not condition:
        raise AssertionError(msg)


# ----- Confidence

def confidence(g: np.ndarray, cols: int, p: Params) -> np.ndarray:
    """
    Compute the clean-confidence weight, in `[w_floor, 1]`.
    """
    w = np.zeros_like(g)
    g_flat = g.flatten()
    w_flat = w.flatten()
    
    if not p.min3:
        # Simple case: weight each pixel independently
        for i in range(len(g_flat)):
            w_flat[i] = p.weight(g_flat[i])
        return w.reshape(g.shape)
    
    # Handle min3 case (horizontal 3-pixel window)
    rows = g.shape[0]
    g_2d = g.reshape(rows, cols)
    w_2d = w.reshape(rows, cols)
    
    # Process each row
    for y in range(rows):
        row = g_2d[y, :]
        w_row = w_2d[y, :]
        
        # Handle edge columns
        if cols >= 1:
            w_row[0] = p.weight(np.min([row[0], row[min(1, cols-1)]]))
            if cols >= 2:
                w_row[cols-1] = p.weight(np.min([row[cols-1], row[max(0, cols-2)]]))
        
        # Handle interior with 3-pixel window
        if cols >= 3:
            for x in range(1, cols-1):
                w_row[x] = p.weight(np.min([row[x-1], row[x], row[x+1]]))
    
    return w


# ----- Decision

def and3_cols(src: np.ndarray, cols: int, k: int) -> np.ndarray:
    """AND of `src` over columns `x - k`, `x`, `x + k`, clamped at the edges"""
    src_2d = src.reshape(-1, cols)
    out_2d = np.zeros_like(src_2d, dtype=bool)
    
    for y in range(src_2d.shape[0]):
        row = src_2d[y, :]
        out_row = out_2d[y, :]
        
        for x in range(cols):
            x1 = max(0, x - k)
            x2 = x
            x3 = min(cols - 1, x + k)
            out_row[x] = row[x1] and row[x2] and row[x3]
    
    return out_2d.flatten()


def and3_rows(src: np.ndarray, rows: int, cols: int, k: int) -> np.ndarray:
    """AND of `src` over rows `y - k`, `y`, `y + k`, clamped at the edges"""
    src_2d = src.reshape(rows, cols)
    out_2d = np.zeros_like(src_2d, dtype=bool)
    
    for x in range(cols):
        col = src_2d[:, x]
        out_col = out_2d[:, x]
        
        for y in range(rows):
            y1 = max(0, y - k)
            y2 = y
            y3 = min(rows - 1, y + k)
            out_col[y] = col[y1] and col[y2] and col[y3]
    
    return out_2d.flatten()


def decide(g: np.ndarray, w: np.ndarray, rows: int, cols: int, p: Params) -> np.ndarray:
    """
    Decide which pixels are worth reconstructing.
    """
    debug_assert(g.shape == w.shape, "one gate and one weight per pixel")
    debug_assert(g.size == rows * cols, "rows * cols must cover the plane")
    
    # Dark pixels (below dust floor)
    dark = g < p.phi
    
    # 3x3 and 7x7 dilation of dark pixels
    row_dark = and3_cols(and3_cols(dark, cols, 1), cols, 3)
    col_dark = and3_rows(and3_rows(dark, rows, cols, 1), rows, cols, 3)
    
    # Create mask - True means we should reconstruct this pixel
    mask = np.ones(g.shape, dtype=bool)
    mask_flat = mask.flatten()
    row_dark_2d = row_dark.reshape(rows, cols)
    col_dark_2d = col_dark.reshape(rows, cols)
    w_2d = w.reshape(rows, cols)
    
    for y in range(rows):
        for x in range(cols):
            idx = y * cols + x
            if p.clamp_l3 and w_2d[y, x] >= 1.0:
                continue  # Skip if weight is maxed and we're clamping
            
            # Check if surrounded by dark pixels
            above = row_dark_2d[max(0, y-4), x] if y >= 4 else False
            below = row_dark_2d[min(rows-1, y+4), x] if y+4 < rows else False
            left = col_dark_2d[y, max(0, x-4)] if x >= 4 else False
            right = col_dark_2d[y, min(cols-1, x+4)] if x+4 < cols else False
            
            mask_flat[idx] = not (above or below or left or right)
    
    return mask


# ----- Pyramid operations

class Levels:
    """The four normalized convolutions at one pixel"""
    def __init__(self):
        self.c: List[float] = [0.0, 0.0, 0.0, 0.0]  # Confidence mass
        self.p: List[float] = [0.0, 0.0, 0.0, 0.0]  # Gate pyramid
        self.l: List[List[float]] = [[0.0, 0.0, 0.0, 0.0] for _ in range(3)]  # Per channel


class Sums:
    """One level's running numerator and denominator"""
    def __init__(self):
        self.kw: float = 0.0
        self.kwg: float = 0.0
        self.kwd: List[float] = [0.0, 0.0, 0.0]
    
    def span(self, f: 'Frame', k: float, lo: int, hi: int):
        """Accumulate the run of cells `lo..=hi`, all at kernel weight `k`"""
        if lo > hi:
            return
            
        # Extract the relevant slices
        w_slice = f.w[lo:hi+1]
        g_slice = f.g[lo:hi+1]
        color_slices = [plane[lo:hi+1] for plane in f.colors]
        
        # Compute weights
        kw = w_slice * k
        self.kw += np.sum(kw)
        self.kwg += np.sum(kw * g_slice)
        
        # Color channels
        for d, plane in zip(self.kwd, color_slices):
            d += np.sum(kw * plane)
    
    def finish(self, out: Levels, level: int):
        """Normalize and store results"""
        inv = 1.0 / self.kw if self.kw > 0.0 else 0.0
        out.c[level] = self.kw
        out.p[level] = self.kwg * inv
        for l, d in zip(out.l, self.kwd):
            l[level] = d * inv


class Frame:
    """The planes and geometry every tap reads"""
    def __init__(self, g: np.ndarray, w: np.ndarray, colors: List[np.ndarray], 
                 lut: np.ndarray, rows: int, cols: int):
        self.g = g
        self.w = w
        self.colors = colors  # [R, G, B] planes
        self.lut = lut
        self.rows = rows
        self.cols = cols


def offset(base: int, delta: int, length: int) -> Optional[int]:
    """`base + delta`, or `None` where that falls off a plane of `len`"""
    v = base + delta
    return v if 0 <= v < length else None


def clamped(base: int, delta: int, length: int) -> int:
    """`base + delta`, clamped into a plane of `len`"""
    return max(0, min(length - 1, base + delta))


def pyramids_at(f: Frame, y: int, x: int) -> Levels:
    """The pyramids at one pixel"""
    out = Levels()
    rows, cols = f.rows, f.cols
    
    # Levels 0 and 1: the 9x9 and 5x5 boxes, every cell at unit weight
    for level, (spans, k) in enumerate([(LEVEL0, 1.0 / 69.0), (LEVEL1, 1.0 / 21.0)]):
        sums = Sums()
        for dy, dx_lo, dx_hi in spans:
            ny = offset(y, dy, rows)
            if ny is None:
                continue
            base = ny * cols
            lo_idx = base + clamped(x, dx_lo, cols)
            hi_idx = base + clamped(x, dx_hi, cols)
            sums.span(f, k, lo_idx, hi_idx)
        sums.finish(out, level)
    
    # Level 2: the 3x3 binomial tent, weight t[dy] * t[dx] / 16
    sums = Sums()
    for dy in [-1, 0, 1]:
        ny = offset(y, dy, rows)
        if ny is None:
            continue
        for dx in [-1, 0, 1]:
            nx = offset(x, dx, cols)
            if nx is None:
                continue
            i = ny * cols + nx
            k = TENT[dy + 1] * TENT[dx + 1] / 16.0
            sums.span(f, k, i, i)
    sums.finish(out, 2)
    
    # Level 3: the raw pixel, no kernel
    i = y * cols + x
    out.c[3] = f.w[i]
    out.p[3] = f.g[i]
    for ch in range(3):
        out.l[ch][3] = f.lut[f.colors[ch][i]]
    
    return out


# ----- Dithering

def uniform(pixel: int, channel: int) -> float:
    """
    Uniform `[0, 1)`, fixed per (pixel, channel).
    Using a simple hash function instead of RNG for performance.
    """
    # Stafford's Mix13 adapted for Python
    x = ((pixel << 2) | channel) & 0xFFFFFFFF
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9
    x = (x ^ (x >> 27)) * 0x94d049bb133111eb
    x = x ^ (x >> 31)
    return (x & 0x00FFFFFF) / (1 << 24)


def dither(x: float, pixel: int, channel: int, p: Params) -> float:
    """Apply dithering if within the dither band"""
    [lo, hi] = p.eta
    if x <= lo or x >= hi:
        return 0.0
    
    envelope = 4.0 / ((hi - lo) ** 2) * (x - lo) * (hi - x)
    d = envelope * (uniform(pixel, channel) - 0.5) * ALPHA[channel] * x
    
    # Only apply if it keeps us within the band
    if x + d > lo and x + d < hi:
        return d
    return 0.0


# ----- Reconstruction

@dataclass
class Patch:
    """What we rebuilt"""
    at: List[int]  # Linear indices of reconstructed pixels
    density: List[np.ndarray]  # [R, G, B] density planes for reconstructed pixels


def reconstruct_core(g: np.ndarray, w: np.ndarray, colors: List[np.ndarray], 
                     mask: np.ndarray, p: Params, rows: int, cols: int) -> Patch:
    """
    Pyramids blended into a reconstructed density per channel, dithered and clamped
    """
    # Create frame object
    lut = np.array([density(float(i)) for i in range(65536)], dtype=np.float32)
    frame = Frame(g, w, colors, lut, rows, cols)
    
    # Get list of pixels to reconstruct
    mask_flat = mask.flatten()
    at = [i for i, v in enumerate(mask_flat) if v]
    
    if not at:
        return Patch(at=[], density=[np.array([]) for _ in range(3)])
    
    # Prepare output arrays
    density_arrays = [np.zeros(len(at), dtype=np.float32) for _ in range(3)]
    keep = [True] * len(at)
    
    # Process each pixel to reconstruct
    for idx, linear_idx in enumerate(at):
        y = linear_idx // cols
        x = linear_idx % cols
        
        # Get pyramid values at this pixel
        center = pyramids_at(frame, y, x)
        
        # Compute lo and hi bounds for detail bands
        lo = [0.0, 0.0, 0.0, 0.0]
        hi = [0.0, 0.0, 0.0, 0.0]
        for level in range(1, 4):  # Levels 1, 2, 3
            lo[level] = center.p[level] - center.p[level-1]
            hi[level] = lo[level]  # Start with same value
        
        # If cross_beta, check horizontal neighbors for min/max of detail bands
        if p.cross_beta:
            for dy, dx in [(0, -1), (0, 1), (-1, 0), (1, 0)]:
                ny = offset(y, dy, rows)
                nx = offset(x, dx, cols)
                if ny is None or nx is None:
                    continue
                neighbor = pyramids_at(frame, ny, nx)
                for level in range(1, 4):
                    d = neighbor.p[level] - neighbor.p[level-1]
                    lo[level] = min(lo[level], d)
                    hi[level] = max(hi[level], d)
        
        # Process each color channel
        for ch in range(3):
            acc = 0.0
            
            # Blurred base plus lost light
            a = center.l[ch][0] + p.gamma[ch] * (p.ir_ref - center.p[0])
            
            # Detail bands with soft thresholding
            for level in range(1, 4):  # Levels 1, 2, 3
                detail = (center.l[ch][level] - center.l[ch][level-1]) * p.detail_gain
                a_hi, a_lo = p.a[ch][level-1]
                
                # Sort so a_lo <= a_hi
                if hi[level] < 0.0:
                    a_lo, a_hi = a_hi, a_lo
                
                lo_t = a_lo * lo[level]
                hi_t = a_hi * hi[level]
                
                # Apply dead zone
                if detail < lo_t:
                    r = detail - lo_t
                elif detail > hi_t:
                    r = detail - hi_t
                else:
                    r = 0.0
                
                # Apply confidence weighting
                if level == 1:
                    confidence = min(2.0 * center.c[1], 1.0)
                elif level == 2:
                    confidence = center.c[2]
                else:  # level == 3
                    confidence = center.c[3] * center.c[3]
                
                acc += r * confidence
            
            # Add dithering
            acc += dither(acc, linear_idx, ch, p)
            
            # Apply clamping if needed
            if p.clamp_l3:
                if acc <= 0.0:
                    keep[idx] = False
                    break
                acc = max(acc, center.l[ch][3])  # Only fill, never darken
            
            density_arrays[ch][idx] = acc
    
    # Filter out any pixels we decided not to keep
    if not all(keep):
        filtered_at = [at[i] for i in range(len(at)) if keep[i]]
        filtered_density = [
            np.array([density_arrays[ch][i] for i in range(len(at)) if keep[i]])
            for ch in range(3)
        ]
        return Patch(at=filtered_at, density=filtered_density)
    
    return Patch(at=at, density=density_arrays)


# ----- Main openICE function

def clean(color: List[np.ndarray], ir: np.ndarray, cal: Calibration, 
          rows: int, cols: int, opts: Options) -> int:
    """
    Remove dust from a frame like magic, returning how many pixels it rebuilt
    
    Args:
        color: [R, G, B] planes as uint16 arrays
        ir: IR plane as uint16 array
        cal: Calibration from prescan
        rows, cols: Image dimensions
        opts: Algorithm options
        
    Returns:
        Number of pixels that were reconstructed
    """
    [red, green, blue] = color
    p = Params.new(opts, cal)
    
    # Steps 1 and 3: Log-density, then IR gating
    g = gate(red, ir, p)
    
    # Step 4: Confidence weight
    w = confidence(g, cols, p)
    
    # Step 5: Decide which pixels to reconstruct
    mask = decide(g, w, rows, cols, p)
    
    # Steps 6-8: Pyramid reconstruction
    patch = reconstruct_core(g, w, [red, green, blue], mask, p, rows, cols)
    
    # Step 9: Back to linear and apply to color planes
    for plane, density_vals in zip([red, green, blue], patch.density):
        if len(density_vals) > 0 and len(patch.at) > 0:
            # Convert density back to linear
            linear_vals = from_density(density_vals)
            
            # Apply the changes
            for idx, linear_val in zip(patch.at, linear_vals):
                if p.clamp_l3:
                    plane[idx] = max(plane[idx], linear_val)
                else:
                    plane[idx] = linear_val
    
    return len(patch.at)


# ----- Helper for creating Options from PSIX parameters

def create_ice_options(model: str = "Ls9000", quality: str = "Normal", 
                       dpi: int = 2000, metering_target: float = 0.95) -> Options:
    """
    Create openICE options from PSIX-friendly parameters
    
    Args:
        model: Scanner model ("Ls9000", "Ls5000", or "Ls50")
        quality: Quality mode ("Normal" or "Fine")
        dpi: Scan resolution in DPI
        metering_target: AE target as fraction of full scale (0.0-1.0)
    """
    model_map = {"Ls9000": Model.Ls9000, "Ls5000": Model.Ls5000, "Ls50": Model.Ls50}
    quality_map = {"Normal": Quality.Normal, "Fine": Quality.Fine}
    
    return Options(
        model=model_map[model],
        quality=quality_map[quality],
        dpi=dpi,
        metering_target=metering_target
    )