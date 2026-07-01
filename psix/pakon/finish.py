#!/usr/bin/env python3
"""pakon_finish — P0 negative->positive "density backbone".

STANDALONE. Does NOT touch pakon_decode.py / pakon_scan2.py. Implements P0
(§3 steps 1,2,4,6) on the project's clean RAW negative
(the flat-fielded, linear, 16-bit, orange-mask-intact *_rgb16.tiff from pakon_decode).

P0+P1 = density backbone + dye-decouple. NOT yet: OEM LUTs (P2), Ansel scene balance (P3),
IR-ICE defect removal, per-frame split (reg0x1e).

Pipeline (per the plan):
  1.  SAMPLE   per-channel film-base Dmin from the film rebate/clearest film (orange base).
  2.+4 DENSITY D = -log10(raw / base) (>=0; base->0); per-channel base subtract removes the orange mask.
  P1a DECOUPLE 3x3 neutral-preserving matrix in density space -> unmix dye crosstalk (de-muddy).
  P1b CROSSOVER per-channel density scale -> straighten the neutral axis (align per-layer gamma).
  6.  PRINT    per-frame white point -> s = normalized scene log-exposure; scene-linear master;
               log-domain display tone (base->black, highlight->white).
OUTPUT (plan §6): a 16-bit SCENE-REFERRED LINEAR master (neutral, for grading) AND an 8-bit
baked positive (the look = a display transform on top, never baked into the master).

Usage:
  pakon_finish.py captures/scan_<ts>_rgb16.tiff [--out-prefix captures/pos]
                  [--decouple 1.0] [--no-balance] [--balance-pct 95]
                  [--slr 2.4] [--display-gamma 2.2] [--base-pct 99.7] [--frame-lines 2500]
"""
import os
import sys

import numpy as np
import tifffile

EPS = 1.0 / 65535.0          # density floor (avoid log(0)) in normalized [0,1] terms


def build_decouple(strength, leaks=None):
    """P1 dye-decouple matrix M (3x3, DENSITY space, NEUTRAL-PRESERVING). Dye crosstalk is linear in
    density (densities add), so we model the measurement as A @ D_true, where A is the inter-channel
    leak: each channel picks up `leak` of its neighbours' density. A's rows sum to 1, so a neutral
    (equal D) is untouched -> grays stay gray; only colours get unmixed. M = inv(A) removes it.
    `strength` scales the leaks (0 -> identity = P0). leaks = [rg,rb, gr,gb, br,bg] off-diagonals.
    NOTE: these default leak fractions are our principled C-41 estimate; accurate per-scanner values
    need a colour-target (IT8) scan or measured dye x LED spectra (our independent calibration path).
    The STRUCTURE (remove crosstalk, preserve neutrals) is correct regardless of the exact numbers."""
    if leaks is None:
        leaks = [0.08, 0.05, 0.07, 0.07, 0.04, 0.09]      # rg,rb, gr,gb, br,bg (C-41 narrowband estimate)
    rg, rb, gr, gb, br, bg = [strength * x for x in leaks]
    A = np.array([[1.0 - rg - rb, rg, rb],
                  [gr, 1.0 - gr - gb, gb],
                  [br, bg, 1.0 - br - bg]], dtype=np.float64)
    return np.linalg.inv(A).astype(np.float32)


def _asset(name):
    """Path to a packaged asset under psix/assets/ (works installed or in-tree)."""
    from importlib.resources import files
    return str(files("psix") / "assets" / name)


# oem_tone_shape.npy is OURS (a measured neg->pos transfer) and ships in psix/assets/.
# The Kodak OEM matrix/LUT (.txt) are NOT shipped (Kodak IP) — psix uses the parametric
# decouple + measured tone; these constants exist only for the CLI --decouple-matrix/--tone oem paths.
OEM_DECOUPLE = _asset("_ClientColNegMat.txt")


def load_oem_decouple(path):
    """Parse the OEM Kodak ClientColNegMat (_ClientColNegMat.txt):
    `coeff_r_c: v` lines -> the 3x3 dye-decouple matrix (cols 0..2) + a per-row offset (col 3).
    This is Kodak's ACTUAL per-scanner-class crosstalk solution (vs build_decouple's principled guess).
    We apply the 3x3 crosstalk-unmix in our base-relative DENSITY domain. The col-3 offset is in the
    OEM's absolute 14-bit scan-code domain (its Dmin/bias handling) and is DROPPED: our pipeline already
    zeroes the base per channel (DEMASK) and re-neutralises per frame (scene balance), so a code-domain
    constant has no meaning in our density domain. Returns M (3x3 float32)."""
    import re
    v = {}
    for ln in open(path):
        m = re.match(r'\s*coeff_(\d)_(\d)\s*:\s*(-?[\d.eE+]+)', ln)
        if m:
            v[(int(m.group(1)), int(m.group(2)))] = float(m.group(3))
    if not all((r, c) in v for r in range(3) for c in range(3)):
        raise ValueError("OEM decouple matrix incomplete: %s" % path)
    return np.array([[v[(r, c)] for c in range(3)] for r in range(3)], np.float32)


OEM_TONE_LUT = _asset("_ClientColNegLut.txt")


def load_oem_tone_lut(path):
    """Parse the OEM Kodak ClientColNegLut (_ClientColNegLut.txt): 16384 rows
    `index<TAB>value`, 14-bit. This is Kodak's negative->positive PRINT transfer = invert + log + paper
    tone in one. INPUT = the negative LINEAR transmittance code (idx 0 = densest negative = scene
    HIGHLIGHT -> value 16383 = white; idx 16383 = clear base = scene SHADOW -> value 0 = black);
    monotonic-decreasing = the inversion. The steep convexity is the print's contrast/gamma. Returns a
    normalized 1-D table lut[16384] in [0,1] (value/16383) — index it by linear transmittance*16383."""
    d = np.loadtxt(path)
    lut = (d[np.argsort(d[:, 0]), 1] / 16383.0).astype(np.float32)
    return np.clip(lut, 0.0, 1.0)


OEM_TONE_SHAPE = _asset("oem_tone_shape.npy")


def load_oem_tone_shape(path):
    """The OEM negative→positive LUMINANCE tone SHAPE, MEASURED from the OEM's own stage dumps
    we have 12.raw (scan) and 12_rpd.raw (=OEM after colour
    correction) of the SAME frame, so the OEM neg→pos transfer is the per-pixel curve between them.
    The stage-by-stage diagnostic proved (a) that transfer is PER-CHANNEL (no 3×3 matrix — ClientColNegMat
    is per-scanner CAPTURE calibration, not a post-process), and (b) the OEM curve is ~100× GENTLER than our
    old ClientColNegLut (which was far too convex/harsh). This asset is the averaged (WB-neutral) luminance
    shape in OUR tn domain: lut[0]=highlight→white(1), lut[N-1]=base/shadow→black(0). Colour stays with
    scene_balance. Endpoints (black floor / white cap) are set by the grade, per the OEM's [~10..~250]."""
    return np.load(path).astype(np.float32)


def _rgb2hsv(rgb):
    """Vectorized RGB(0..1)->HSV(0..1). Returns (h, s, v) arrays."""
    r = rgb[..., 0]; g = rgb[..., 1]; b = rgb[..., 2]
    mx = np.maximum(np.maximum(r, g), b)
    mn = np.minimum(np.minimum(r, g), b)
    df = mx - mn
    h = np.zeros_like(mx)
    safe = df > 1e-12
    m = safe & (mx == r); h[m] = ((g[m] - b[m]) / df[m]) % 6.0
    m = safe & (mx == g); h[m] = ((b[m] - r[m]) / df[m]) + 2.0
    m = safe & (mx == b); h[m] = ((r[m] - g[m]) / df[m]) + 4.0
    h = (h / 6.0) % 1.0
    s = np.where(mx > 1e-12, df / np.maximum(mx, 1e-12), 0.0)
    return h, s, mx


def _hsv2rgb(h, s, v):
    """Vectorized HSV(0..1)->RGB(0..1)."""
    i = np.floor(h * 6.0).astype(int)
    f = h * 6.0 - i
    p = v * (1.0 - s); q = v * (1.0 - f * s); t = v * (1.0 - (1.0 - f) * s)
    i = i % 6
    r = np.choose(i, [v, q, p, p, t, v])
    g = np.choose(i, [t, v, v, q, p, p])
    b = np.choose(i, [p, p, t, v, v, q])
    return np.stack([r, g, b], axis=-1)


def apply_hsl(disp, bands, half_width=40.0):
    """Per-hue HSL (Lightroom Color-Mixer style). `bands` = list of
    (center_deg, hue_shift_deg, sat_frac, lum_frac). Each band's effect is
    weighted by circular hue proximity to its centre and gated by saturation/
    value, so neutrals are untouched and bands blend smoothly. One HSV
    round-trip for all bands. hue_shift<0 = toward the lower-hue neighbour
    (e.g. yellow→orange)."""
    h, s, v = _rgb2hsv(disp)
    hue_deg = h * 360.0
    gate = np.clip(s / 0.2, 0.0, 1.0) * np.clip(v / 0.1, 0.0, 1.0)
    # Luminance is the one axis that's fully visible on near-NEUTRAL pixels (warm
    # walls/sand read as low-sat yellow/orange), so it needs a much stronger
    # saturation gate or it relights the whole frame. Hue/Sat stay on `gate`.
    lum_gate = np.clip((s - 0.20) / 0.30, 0.0, 1.0)            # ~0 below s=0.2, full by s=0.5
    hue_delta = np.zeros_like(h)
    sat_mult = np.ones_like(h)
    lum_mult = np.ones_like(h)
    changed = False
    for center, dh, ds, dl in bands:
        if dh == 0 and ds == 0 and dl == 0:
            continue
        changed = True
        d = np.abs(((hue_deg - center + 180.0) % 360.0) - 180.0)   # circular hue distance
        wj = np.clip(1.0 - d / half_width, 0.0, 1.0) * gate
        if dh:
            hue_delta = hue_delta + dh * wj
        if ds:
            sat_mult = sat_mult * (1.0 + ds * wj)
        if dl:
            lum_mult = lum_mult * (1.0 + dl * wj * lum_gate)
    if not changed:
        return disp
    h = (h + hue_delta / 360.0) % 1.0
    s = np.clip(s * sat_mult, 0.0, 1.0)
    v = np.clip(v * lum_mult, 0.0, 1.0)
    return np.clip(_hsv2rgb(h, s, v), 0.0, 1.0)


def tone_saturation(rgb, s_shadow, s_mid, s_high):
    """LUMINANCE-DEPENDENT saturation (the OEM 'satplus' look = the 'punch'). Scale each pixel's chroma
    (its deviation from its own luminance) by a factor that depends on luminance: DESATURATE shadows
    (s_shadow<1 -> clean, not muddy), BOOST midtones (s_mid>1 -> pop), modest highlights (s_high). This
    matches the measured OEM 12.jpg per-tone saturation profile (shadow/mid/high ≈ 19/44/15) — vs our flat
    /shadow-heavy distribution (29/31/13). rgb in [0,1]; piecewise-linear scale through (Y=0,.5,1)."""
    Y = 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
    # HOLD each scale across its tone BAND (shadow Y<1/3, mid 1/3..2/3, high >2/3) with smooth transitions
    # at 1/3 and 2/3 — so the shadow band is actually desaturated (a single ramp leaves the upper shadows
    # at mid-scale, which is why shadows didn't drop). Control points centred on each band.
    s = np.interp(Y, [0.0, 0.26, 0.40, 0.60, 0.74, 1.0],
                  [s_shadow, s_shadow, s_mid, s_mid, s_high, s_high]).astype(np.float32)
    return np.clip(Y[..., None] + (rgb - Y[..., None]) * s[..., None], 0.0, 1.0)


def region_balance(Dl, fc, M, p, G):
    """Region-aware ROBUST illuminant estimate = the classical Frontier/Ansel-class subject-failure
    rejection (no trained clusters needed). Tile the frame into a G×G grid; per tile take the shades-of-gray
    illuminant; the GLOBAL cast = the MEDIAN per-channel cast across tiles. Why it beats one global
    gray-world: a coloured SUBJECT (grass, a wall, a jacket) is spatially LOCALISED → its tiles are OUTLIERS
    in the per-tile distribution, so the median rejects them; the true ILLUMINANT cast is present in EVERY
    tile → it survives the median. Global gray-world instead lets a big colour region drag the balance
    (= 'subject failure'). Returns the per-channel density offset o (mean-centred). (Fails only if a single
    subject fills >half the frame — that is what the trained priors are for; not handled here.)"""
    hr, wq, _ = Dl.shape
    ys = np.linspace(0, hr, G + 1).astype(int)
    xs = np.linspace(0, wq, G + 1).astype(int)
    casts = []
    for i in range(G):
        for j in range(G):
            tfc = fc[xs[j]:xs[j + 1]]
            if not tfc.any():
                continue
            tp = Dl[ys[i]:ys[i + 1], xs[j]:xs[j + 1], :][:, tfc, :].reshape(-1, 3)
            if len(tp) < 50:
                continue
            e = np.power(np.maximum(np.power(10.0, (tp - M) * p).mean(0), 1e-30), 1.0 / p)
            le = np.log10(np.maximum(e, 1e-30))
            casts.append(le - le.mean())                # per-tile colour cast (neutral-removed)
    if not casts:
        return np.zeros(3, np.float32)
    o = np.median(np.array(casts), axis=0)              # robust global cast (subject-failure rejected)
    return (o - o.mean()).astype(np.float32)


def scene_balance(D, fmask, frame_lines, p, whole_frame=False, regions=0, target=None):
    """P3 per-frame auto white balance (OUR OWN; the Ansel-class goal). Estimate the illuminant with robust
    SHADES-OF-GRAY (Minkowski p-norm) and remove it as a per-channel ADDITIVE density offset — the correct
    WB model in this domain: after the calib fixes the per-channel crossover (a multiplicative term), the
    illuminant is a per-channel linear scale = an additive shift in density. p=1 = plain gray-world; p~5
    weights brighter pixels -> robust to large flat colour regions (foliage, sky).
    whole_frame=True (a SINGLE extracted frame, the §3 per-frame raw-neg): use ONE GLOBAL illuminant for
    the whole frame. The strip path below uses a SLIDING WINDOW (win ≈ one frame) along transport to
    balance each frame of a multi-frame strip on its own — applied to a single frame that window makes the
    WB vary spatially across the width and DIVERGE at the (dark/short-window) edges => a colour band.
    Returns (D_balanced, film-mean offset for logging)."""
    h, w, _ = D.shape
    fc = fmask[0]                                       # per-transport-column film boolean
    Dl = D[::4]                                         # subsample CCD rows for the estimate (speed/RAM)
    if Dl.size:                                         # HARDEN: clamp per-channel density outliers before the
        # p-norm. M (the Minkowski shift) was Dl.max(); a single extreme pixel (e.g. a near-zero neg pixel ->
        # huge density) would then define M and the p=5 power collapses the illuminant onto that one pixel ->
        # runaway colour cast. Clamp to the per-channel 99.9th pct = robust high-density level, spike-proof.
        Dl = np.minimum(Dl, np.percentile(Dl, 99.9, axis=(0, 1), keepdims=True))
    M = float(Dl.max()) if Dl.size else 0.0
    if whole_frame:
        if regions and regions > 1:
            o = region_balance(Dl, fc, M, p, int(regions))                 # region-aware robust (subject-failure)
        else:
            ps = np.power(10.0, (Dl[:, fc, :] - M) * p) if fc.any() else np.power(10.0, (Dl - M) * p)
            e = np.power(np.maximum(ps.mean(axis=(0, 1)), 1e-30), 1.0 / p)  # (3,) ONE global illuminant
            o = np.log10(np.maximum(e, 1e-30)); o = o - o.mean()           # (3,) single per-channel offset
        if target is not None:                                             # leave a deliberate cast (e.g. the
            o = o - np.asarray(target, np.float32)                         # OEM warm target: don't over-neutralise
        Db = np.clip(D - o[None, None, :], 0.0, None)
        return Db, o
    pm = np.power(10.0, (Dl - M) * p).mean(axis=0)      # (w,3) per-column mean of linear^p (M-shift = stable)
    pm = pm * fc[:, None]
    win = max(8, int(frame_lines))
    csum = np.concatenate([np.zeros((1, 3)), np.cumsum(pm, 0)])
    ccnt = np.concatenate([[0.0], np.cumsum(fc.astype(np.float64))])
    idx = np.arange(w); lo = np.maximum(idx - win // 2, 0); hi = np.minimum(idx + win // 2 + 1, w)
    wm = (csum[hi] - csum[lo]) / np.maximum((ccnt[hi] - ccnt[lo])[:, None], 1.0)   # windowed p-mean / film
    e = np.power(np.maximum(wm, 1e-30), 1.0 / p)        # (w,3) per-channel illuminant estimate (relative)
    loge = np.log10(np.maximum(e, 1e-30))
    o = loge - loge.mean(axis=1, keepdims=True)         # (w,3) mean-centered per-channel density offset
    o[~fc] = 0.0
    Db = np.clip(D - o[None, :, :], 0.0, None)          # remove illuminant; base stays >=0 (neutral black)
    return Db, (o[fc].mean(axis=0) if fc.any() else np.zeros(3))


def channel_balance(D, fmask, pct):
    """P1 per-channel crossover/gamma align. The three dye layers have different contrast (gamma), so
    neutral balance drifts across the tonal range (e.g. shadows cyan, highlights warm) = crossover.
    Base is already density 0 at the bottom; we scale each channel so a high density percentile matches
    across channels at the top -> the neutral (gray) axis is straightened. Data-driven + independent
    (P3 scene-balance will refine this PER FRAME; P1 is a single global gray-axis align)."""
    p = np.maximum(np.percentile(D[fmask], pct, axis=0), 1e-3)
    return (float(p.mean()) / p).astype(np.float32)


def _box_blur(a, r):
    """Separable box blur (radius r) via cumsum, O(n). a: (h,w). float32, 2D, low memory (no moveaxis
    or float64 temporaries — important: this runs on large arrays)."""
    a = np.asarray(a, np.float32)
    n0, n1 = a.shape
    cs = np.empty((n0, n1 + 1), np.float32); cs[:, 0] = 0.0; np.cumsum(a, axis=1, out=cs[:, 1:])
    idx = np.arange(n1); lo = np.maximum(idx - r, 0); hi = np.minimum(idx + r + 1, n1)
    a = (cs[:, hi] - cs[:, lo]) / (hi - lo).astype(np.float32)
    cs = np.empty((n0 + 1, n1), np.float32); cs[0, :] = 0.0; np.cumsum(a, axis=0, out=cs[1:, :])
    idx = np.arange(n0); lo = np.maximum(idx - r, 0); hi = np.minimum(idx + r + 1, n0)
    return (cs[hi, :] - cs[lo, :]) / (hi - lo).astype(np.float32)[:, None]


def chroma_denoise(rgb, radius, amount):
    """P4 CHROMA noise reduction. The shadows carry amplified CCD COLOUR noise (the blue channel is
    starved of light by the orange mask -> noisiest; the decouple/crossover then decorrelate it into
    chroma speckle). Real film grain is LUMINANCE, so we keep luma sharp and blur only the chroma:
    split luma Y + per-channel colour offset (rgb - Y), low-pass the colour (it's naturally smooth, so
    this is near-invisible to real detail), recombine. Kills colour speckle, preserves grain + detail."""
    if amount <= 0:
        return rgb
    Y = 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
    ch = rgb - Y[..., None]                             # per-channel chroma (colour offset from luma)
    chb = np.stack([_box_blur(_box_blur(ch[..., c], radius), radius) for c in range(3)], -1)
    ch = (1.0 - amount) * ch + amount * chb
    return np.clip(Y[..., None] + ch, 0.0, 1.0)


def luma_denoise(disp, amount, radius, threshold):
    """Gentle EDGE-PRESERVING luminance noise reduction (the OEM does this; we previously did none, so our
    output is grainier than 12.jpg). Split luma detail = Y - blur(Y); the SUB-threshold part is film/sensor
    GRAIN (smooth it), the ABOVE-threshold part is real EDGES (keep fully, so sharpness is preserved). We
    attenuate only the grain by `amount`. Applied equally to all channels (chroma untouched). Pairs with
    sharpen_image (which boosts the above-threshold edges) — complementary, they don't fight."""
    if amount <= 0:
        return disp
    Y = 0.299 * disp[..., 0] + 0.587 * disp[..., 1] + 0.114 * disp[..., 2]
    blur = _box_blur(_box_blur(Y, radius), radius)
    detail = Y - blur
    edge = np.sign(detail) * np.maximum(np.abs(detail) - threshold, 0.0)   # real edges (above threshold) - keep
    grain = detail - edge                                                  # sub-threshold grain
    Ynr = blur + edge + (1.0 - amount) * grain                             # attenuate grain only
    return np.clip(disp + (Ynr - Y)[..., None], 0.0, 1.0)


def sharpen_image(disp, amount, radius, threshold):
    """P4 adaptive capture sharpening. Unsharp mask on LUMINANCE (no colour fringing), with a soft
    threshold (deadzone) so flat-area film grain/noise isn't amplified — only real edges are. Two box
    passes ≈ Gaussian. The luminance detail is added equally to all channels (hue-preserving)."""
    if amount <= 0:
        return disp
    Y = 0.299 * disp[..., 0] + 0.587 * disp[..., 1] + 0.114 * disp[..., 2]
    blur = _box_blur(_box_blur(Y, radius), radius)
    detail = Y - blur
    if threshold > 0:                                   # suppress sub-threshold (grain) detail
        detail = np.sign(detail) * np.maximum(np.abs(detail) - threshold, 0.0)
    return np.clip(disp + (amount * detail)[..., None], 0.0, 1.0)


def highlight_shoulder(s, amt):
    """Roll off highlights instead of hard-clipping them. s = per-frame normalized log-exposure with the
    UPPER CLIP REMOVED, so content brighter than the white point (e.g. a white wall, s>1) keeps its
    gradient. Below the knee: linear (midtones untouched). Above: a smooth tanh shoulder that asymptotes
    to 1, so the wall's tonal gradient is compressed into the top rather than flattened to pure white.
    amt: 0 = old hard clip at 1; higher = lower knee = more highlight headroom/roll-off."""
    if amt <= 0:
        return np.clip(s, 0.0, 1.0)
    k = 1.0 - 0.45 * amt                                # knee descends from 1.0 (amt 0) toward ~0.55
    w = max(1.0 - k, 1e-3)
    rolled = np.where(s <= k, s, k + w * np.tanh((s - k) / w))
    return np.clip(rolled, 0.0, 1.0)


def filmic_look(g, strength):
    """P2 print/film TONE CURVE (our own — not the OEM LUT). A gentle filmic S (Perlin smootherstep)
    blended over the plain encode: it STEEPENS the midtones for print 'pop' and gives a SOFT SHOULDER
    (highlights roll off toward white instead of hard-clipping) and a soft TOE. Smootherstep has zero
    slope at 0 and 1 -> genuinely soft ends, no harsh clip. strength: 0 = plain gamma (P0/P1 look),
    1 = full S. Applied to the DISPLAY only; the scene-linear master stays look-free (reversible)."""
    g = np.clip(g, 0.0, 1.0)
    s = g * g * g * (g * (g * 6.0 - 15.0) + 10.0)      # smootherstep: 6g^5-15g^4+10g^3
    return (1.0 - strength) * g + strength * s


def sliding_white(colp, frame_lines):
    """Per-frame highlight density: a sliding-window high value of the per-transport-column highlight
    profile `colp`. Approximates per-frame exposure normalization without exact frame boundaries
    (window ≈ one frame). Implemented as smooth -> downsample -> windowed max -> upsample."""
    n = len(colp)
    k = 51
    sm = np.convolve(colp, np.ones(k) / k, 'same')         # denoise the column profile
    step = 50
    ds = sm[::step]                                          # downsample (cheap to window)
    r = max(1, (frame_lines // step) // 2)                   # half-window in downsampled units
    m = np.empty_like(ds)
    for i in range(len(ds)):                                 # windowed max ≈ local (per-frame) highlight
        m[i] = ds[max(0, i - r):i + r + 1].max()
    return np.interp(np.arange(n), np.arange(len(ds)) * step, m).astype(np.float32)


def film_region(rgb, base_pct, whole_frame=False):
    """Return (base, film_mask2d, n_cols). Classify per TRANSPORT COLUMN (axis 1, the scan direction):
    the lamp-off leader/trailer/gaps and the empty gate are WHOLE transport positions (entire dark or
    neutral-bright lines), whereas a blown scene highlight is an isolated dark spot INSIDE a film line
    (densest = darkest on the negative) and must be KEPT (it -> white). So we mask whole non-film
    columns only, never individual dark pixels. base = per-channel orange Dmin from non-saturated film
    pixels. (plan §1.2 / §4.1).
    whole_frame=True: the input is a SINGLE already-extracted frame (the §3 per-frame raw-neg), so the
    ENTIRE image is film — disable gate/dark column masking. (The empty-gate heuristic 'bright+neutral'
    catastrophically misfires on a single negative frame: scene SHADOWS are clear/bright+orange-neutral on
    the negative and would be wrongly masked to black -> vertical black strokes. So mask nothing here.)"""
    h, w = rgb.shape[:2]
    if whole_frame:
        fp = rgb.reshape(-1, 3); fp = fp[fp.max(1) < 65000]
        base = np.maximum(np.percentile(fp, base_pct, axis=0), 1.0).astype(np.float32)
        return base, np.ones((h, w), bool), w
    cR = rgb[:, :, 0].mean(0); cG = rgb[:, :, 1].mean(0); cB = rgb[:, :, 2].mean(0)   # per-column means
    col_lum = (cR + cG + cB) / 3.0
    cmx = np.maximum(np.maximum(cR, cG), cB); cmn = np.minimum(np.minimum(cR, cG), cB)
    hi = float(np.percentile(col_lum, 99.0))
    neutral = (cmx - cmn) <= 0.10 * np.maximum(cmx, 1.0)
    empty_gate = (col_lum > 0.55 * hi) & neutral      # open-gate columns (no film: neutral & bright)
    dark = col_lum < 0.05 * hi                         # lamp-off leader/trailer/gaps columns
    film_cols = ~(empty_gate | dark)
    fmask = np.broadcast_to(film_cols[None, :], (h, w))
    fp = rgb[:, film_cols, :].reshape(-1, 3)
    fp = fp[fp.max(1) < 65000]                         # base from non-saturated film pixels
    if len(fp) < 1000:                                 # e.g. a single cropped frame
        fp = rgb.reshape(-1, 3); fp = fp[fp.max(1) < 65000]
    base = np.maximum(np.percentile(fp, base_pct, axis=0), 1.0).astype(np.float32)
    return base, fmask, int(film_cols.sum())


def finish(path, out_prefix, slr, display_gamma, base_pct, frame_lines, decouple, balance, balance_pct,
           calib=None, look=0.3, wb_p=5.0, sharpen=0.3, sharpen_radius=2, sharpen_thresh=0.008,
           shoulder=0.5, chroma_nr=0.8, chroma_radius=3, headroom=1.18, black_lift=0.05,
           wb_trim=(1.0, 1.0, 1.0), balance_trim=(0.0, 0.0, 0.0),
           region_trim_sh=(0.0, 0.0, 0.0), region_trim_mid=(0.0, 0.0, 0.0),
           region_trim_hi=(0.0, 0.0, 0.0),
           decouple_matrix=None, tone='oemshape', tone_lut=None, whole_frame=False,
           grade=True, grade_lo=1.0, grade_hi=99.2, contrast=0.0, grade_black=10.0, grade_white=250.0,
           density_gamma=1.0,
           tone_shape=None, scene_offset=None, analyze_only=False, balance_regions=0, balance_target=None,
           luma_nr=0.5, luma_radius=2, luma_thresh=0.02,
           saturation=False, sat_shadow=0.62, sat_mid=1.45, sat_high=1.18,
           hsl=None, return_array=False, cached_base=None, return_analysis=False):
    # FAST LIVE-GRADE PATH (psix): the analysis stages (film-base sample + scene-balance illuminant) are
    # slider-INDEPENDENT and ~50ms. `return_analysis=True` runs them and returns (base, scene_offset, fmask)
    # so the caller can cache them; subsequent slider moves pass `cached_base=` (skip film-base) +
    # `scene_offset=` (skip scene-balance) -> only the cheap print stage re-runs. Behaviour-preserving:
    # the cached values are byte-identical to what the inline path computes for the same neg+decouple.
    rgb = path if isinstance(path, np.ndarray) else tifffile.imread(path)   # accept an in-memory neg too
    if rgb.ndim != 3 or rgb.shape[2] < 3:
        print("ERROR: expected an RGB 16-bit TIFF (the *_rgb16.tiff raw negative).", file=sys.stderr)
        return 1
    rgb = rgb[:, :, :3].astype(np.float32)
    h, w = rgb.shape[:2]

    # 1. SAMPLE film base (orange Dmin) + the film region (exclude empty gate + dark lamp-off ends).
    #    whole_frame=True (single extracted frame from §3): no gate/dark masking — the whole image is film.
    if cached_base is not None and whole_frame:        # FAST PATH: reuse cached film-base (whole-frame mask=all film)
        base = np.asarray(cached_base, np.float32); fmask = np.ones((h, w), bool); nfilm = w
    else:
        base, fmask, nfilm = film_region(rgb, base_pct, whole_frame=whole_frame)
        print("  film base (orange Dmin) R/G/B = %.0f/%.0f/%.0f  (from %d film %s, %gpct)%s"
              % (base[0], base[1], base[2], nfilm, 'columns' if whole_frame else 'px', base_pct,
                 '  [WHOLE-FRAME: no gate mask]' if whole_frame else ''))

    # 2.+4. DENSITY + DEMASK: per-channel density above each channel's own base. base -> 0; the
    #       uniform orange mask is removed because each channel is normalized to its own Dmin.
    t = np.clip(rgb / base, EPS, None)         # transmittance relative to the film base, per channel
    D = np.clip(-np.log10(t), 0.0, None)       # negative density above base (>=0; high = scene highlight)
    del rgb, t                                  # free the inputs (~1.4GB) — only D is needed downstream

    # P1.5 GRAY-CARD CALIBRATION (GROUND TRUTH, from pakon_graycard.py). Per-channel density map
    #      D'_c = alpha_c*D_c + beta_c, fit so NEUTRAL gray cards render neutral — corrects both the
    #      balance (the cyan cast) and the per-channel crossover (different layer gammas). Replaces the
    #      gray-world guess. Applied to the RAW density (same domain it was fit in).
    if calib is not None:
        al = np.asarray(calib['alpha'], np.float32)
        # Apply ALPHA (the per-channel crossover/gamma) only — that's the universal stock+scanner
        # property. NOT beta: beta is the card's absolute white balance, which carries its illuminant
        # (here open-shade daylight, ~7500K) and would blue-shift other scenes. WB stays per-image (P1b
        # gray-world now, P3 scene balance later). --calib-full bakes beta too (same-illuminant only).
        D = D * al[None, None, :]
        if calib.get('_full'):
            be = np.asarray(calib['beta'], np.float32)
            D = np.clip(D + be[None, None, :], 0.0, None)
        print("  gray-card calib: crossover alpha R/G/B=%.3f/%.3f/%.3f%s  (%s)"
              % (al[0], al[1], al[2], '  +beta(illuminant WB)' if calib.get('_full') else '', calib.get('source', '?')))

    # P1a. DYE-DECOUPLE MATRIX (density space) — unmix inter-channel dye crosstalk = the main
    #      "de-muddy"/colour-separation step. OEM matrix (Kodak ClientColNegMat, ground truth) when a
    #      path is given; else the parametric build_decouple guess (identity when --decouple 0 = P0).
    if decouple > 0 and decouple_matrix and os.path.exists(decouple_matrix):
        M = load_oem_decouple(decouple_matrix)
        s = min(max(float(decouple), 0.0), 1.0)         # strength: blend identity <-> OEM matrix
        if s < 1.0:
            M = (1.0 - s) * np.eye(3, dtype=np.float32) + s * M
        D = np.clip(D.reshape(-1, 3) @ M.T, 0.0, None).reshape(h, w, 3)
        print("  decouple: OEM matrix %s @ strength %.2f  rows-sum=%.3f/%.3f/%.3f  M=[[%.3f %.3f %.3f][%.3f %.3f %.3f][%.3f %.3f %.3f]]"
              % (os.path.basename(decouple_matrix), s, M[0].sum(), M[1].sum(), M[2].sum(), *M.ravel()))
    elif decouple > 0:
        M = build_decouple(decouple)
        D = np.clip(D.reshape(-1, 3) @ M.T, 0.0, None).reshape(h, w, 3)
        print("  decouple: strength=%.2f (parametric guess)  M=[[%.2f %.2f %.2f][%.2f %.2f %.2f][%.2f %.2f %.2f]]"
              % (decouple, *M.ravel()))

    # P3. SCENE BALANCE — per-frame auto white balance (our own; Ansel-class). Robust shades-of-gray
    #     illuminant estimate per frame, removed as a per-channel ADDITIVE density offset. Runs after the
    #     calib crossover, so it sets only the per-image/per-frame WB (open shade vs sun each neutralised
    #     on its own) — this is what the gray card couldn't give. Replaces the global gray-world guess.
    mo = np.zeros(3, np.float32)                       # per-channel scene-balance offset (for return_analysis)
    if scene_offset is not None:
        # ANSEL roll-balance: apply the ROLL-CONSISTENT per-channel density offset computed across all
        # frames by pakon_roll (overrides the per-frame-independent estimate -> frames stay consistent).
        mo = np.asarray(scene_offset, np.float32)
        D = np.clip(D - mo[None, None, :], 0.0, None)
        print("  scene balance: APPLIED roll-consistent offset R/G/B = %+.3f/%+.3f/%+.3f (Ansel)" % tuple(mo))
    elif balance:
        D, mo = scene_balance(D, fmask, frame_lines, wb_p, whole_frame=whole_frame,
                              regions=balance_regions, target=balance_target)
        print("  scene balance (%s p=%g, per-frame): density offset R/G/B = %+.3f/%+.3f/%+.3f%s"
              % ('region-aware G=%d' % balance_regions if balance_regions > 1 else 'shades-of-gray', wb_p,
                 mo[0], mo[1], mo[2], '  +warm-target' if balance_target is not None else ''))
        if analyze_only:                                # pass-1 (AddScene/AnalyzeScene): emit estimate, skip render
            print("ANSEL_OFFSET=%.6f,%.6f,%.6f" % (mo[0], mo[1], mo[2]))
            return 0
    elif analyze_only:
        print("ANSEL_OFFSET=0,0,0")
        return 0

    if return_analysis:                                # FAST LIVE-GRADE: hand back the cacheable analysis
        return base, mo, fmask                         # (slider-independent); caller re-feeds via cached_base+scene_offset

    # MANUAL CMY COLOUR BALANCE (printer points) — per-channel density trim on top
    # of the auto white balance, so 0 = the auto-neutral. Density domain = the
    # optical-printer domain: +R density prints redder/warmer, etc. (RGB offsets;
    # the UI presents them as CMY, the subtractive complements.)
    if tuple(balance_trim) != (0.0, 0.0, 0.0):
        bt = np.asarray(balance_trim, np.float32)
        D = np.clip(D + bt[None, None, :], 0.0, None)
        print("  CMY balance trim: density R/G/B = %+.3f/%+.3f/%+.3f" % tuple(bt))

    # 6a. PER-FRAME WHITE POINT. A strip-wide white point makes bright frames clip and dark frames flat.
    # Compute a sliding-window (≈ one frame) highlight density along the transport axis. (Approximates
    # per-frame normalization without needing exact frame boundaries — reg0x1e DX split would be exact.)
    Dl = D.mean(2)                                          # luminance density (h, w)
    # ROBUST highlight (ported from pakon_invert.render_positive's normalize): a
    # crushed/clipped negative (e.g. an under-exposed IR 4-ch scan) pins many
    # pixels at the EPS density floor; if those land in the 99th-pct they drag
    # the white point WAY up (saw 5.68 vs a real ~1.2) and the whole frame washes
    # out. So compute the highlight from REAL content only — pixels NOT pinned at
    # the floor. For a well-exposed neg (no floor pixels) this is identical to before.
    _dcap = -np.log10(EPS)                                  # max density a floored pixel reaches
    _real = Dl < 0.97 * _dcap
    if whole_frame:
        # single extracted frame -> ONE global white point (the sliding per-frame window is for strips)
        src = Dl[_real] if _real.any() else Dl
        wd = float(np.percentile(src, 99.0))
        white_col = np.full(w, wd * headroom, np.float32)
        clipped = 100.0 * float((~_real).mean())
        print("  whole-frame white density: %.2f (single global highlight, headroom %.2f%s)"
              % (white_col[0], headroom, "" if clipped < 0.5 else "; %.0f%% floored excluded" % clipped))
    else:
        colp = np.percentile(Dl, 99.0, axis=0)              # per-transport-column highlight density (w,)
        colp[~fmask[0]] = 0.0                               # ignore non-film columns
        white_col = sliding_white(colp, frame_lines)        # per-column (per-frame) white density
        gmax = float(np.percentile(white_col[white_col > 0], 90)) if np.any(white_col > 0) else 1.0
        white_col = np.maximum(white_col, 0.35 * gmax) * headroom   # headroom>1 -> darker + more highlight room
        print("  per-frame white density: median=%.2f range[%.2f..%.2f] (window=%d lines)"
              % (np.median(white_col), white_col.min(), white_col.max(), frame_lines))

    # 6b. PRINT — STREAMED in transport CHUNKS so the spatial ops (chroma NR / sharpen) never build
    # full-image temporaries (this stage previously OOM'd on the ~730MB strip). Per chunk we compute:
    #   s = per-frame normalized log-exposure (0 = base -> BLACK, 1 = highlight -> WHITE), with highlight
    #     HEADROOM (no upper clip) rolled off by a shoulder so bright walls keep gradient; then the
    #     scene-linear MASTER (look-free) and the DISPLAY (P2 filmic tone -> P4 chroma NR -> P4 sharpen).
    # Each chunk reads a haloed slice of D (the blur reach) and writes only its interior, so no seams.
    del Dl
    fc = fmask[0]                                       # per-transport-column film boolean (1D)
    master = None if return_array else np.empty((h, w, 3), np.uint16)   # scene-linear master: not needed for
    out8 = np.empty((h, w, 3), np.uint8)                                 # the live preview (return_array) -> skip
    halo = 4 * max(chroma_radius, 1) + 2 * max(sharpen_radius, 1) + 8   # blur reach (2 box passes each)
    CH = 4096
    if chroma_nr > 0:
        print("  chroma denoise: amount=%.2f radius=%d (luma kept sharp; colour speckle removed)"
              % (chroma_nr, chroma_radius))
    if luma_nr > 0:
        print("  luma denoise: amount=%.2f radius=%d threshold=%.3f (grain smoothing; edges preserved)"
              % (luma_nr, luma_radius, luma_thresh))
    if sharpen > 0:
        print("  sharpen: amount=%.2f radius=%d threshold=%.3f (edge-aware luminance unsharp)"
              % (sharpen, sharpen_radius, sharpen_thresh))
    oem_lut = None
    oem_shape = None
    if tone == 'oemshape':
        oem_shape = load_oem_tone_shape(tone_shape or OEM_TONE_SHAPE)
        print("  tone: OEM MEASURED tone shape %s (N=%d; gentle neg→pos transfer measured from 12.raw→12_rpd; "
              "perceptual encode included — no display-gamma)" % (os.path.basename(tone_shape or OEM_TONE_SHAPE), len(oem_shape)))
    elif tone == 'oem':
        oem_lut = load_oem_tone_lut(tone_lut or OEM_TONE_LUT)
        print("  tone: OEM print LUT %s (16384-entry invert+log+paper-gamma; fed per-frame-normalized "
              "linear transmittance)" % os.path.basename(tone_lut or OEM_TONE_LUT))
    else:
        print("  tone: filmic (display-gamma %.2f + look %.2f)" % (display_gamma, look))
    # 3-WAY colour balance: per-channel density offsets weighted by tonal region
    # (shadows / mids / highlights). Triangular weights over the per-frame-normalized
    # luminance partition to 1 everywhere, so a frame with no region trim is unchanged.
    tsh = np.asarray(region_trim_sh, np.float32)[None, None, :]
    tmid = np.asarray(region_trim_mid, np.float32)[None, None, :]
    thi = np.asarray(region_trim_hi, np.float32)[None, None, :]
    _regions = bool(tsh.any() or tmid.any() or thi.any())
    if _regions:
        print("  3-way trims: S=%+.3f/%+.3f/%+.3f M=%+.3f/%+.3f/%+.3f H=%+.3f/%+.3f/%+.3f"
              % (*region_trim_sh, *region_trim_mid, *region_trim_hi))
    for c0 in range(0, w, CH):
        c1 = min(c0 + CH, w)
        a0 = max(c0 - halo, 0); a1 = min(c1 + halo, w)
        lc = c0 - a0; rc = lc + (c1 - c0)               # interior to keep inside the haloed slice
        fcc = fc[c0:c1]
        Dchunk = D[:, a0:a1, :]
        if _regions:
            wc0 = np.maximum(white_col[None, a0:a1, None], 1e-3)
            pos = np.clip(Dchunk.mean(2, keepdims=True) / wc0, 0.0, 1.0)   # 0 shadow … 1 highlight
            w_sh = np.maximum(0.0, 1.0 - 2.0 * pos)
            w_hi = np.maximum(0.0, 2.0 * pos - 1.0)
            w_mid = 1.0 - w_sh - w_hi
            Dchunk = np.clip(Dchunk + w_sh * tsh + w_mid * tmid + w_hi * thi, 0.0, None)
        # `s` (per-frame normalized + shoulder) feeds the scene-linear MASTER and the legacy filmic tone.
        # The OEM tone paths build their own input (tn) below, so for a preview with an OEM tone we can skip
        # s + the master entirely (saves a 10^ power + uint16 cast per chunk).
        if master is not None or (oem_shape is None and oem_lut is None):
            s = highlight_shoulder(np.maximum(Dchunk / white_col[None, a0:a1, None], 0.0), shoulder)
        if master is not None:
            mlin = np.clip(np.power(10.0, (s[:, lc:rc] - 1.0) * slr), 0.0, 1.0)   # scene-linear master chunk
            mlin[:, ~fcc] = 0.0
            master[:, c0:c1] = (mlin * 65535.0 + 0.5).astype(np.uint16)
        if oem_shape is not None or oem_lut is not None:
            # Per-frame-normalized LINEAR transmittance tn (the native tone-curve input). t = 10^-D;
            # per-frame white -> tw = 10^-Wc. tn: base(t=1)->1 (shadow→black), highlight(t=tw)->0 (→white).
            Dc = Dchunk; Wc = white_col[None, a0:a1, None]
            tw = np.power(10.0, -np.maximum(Wc, 1e-3))
            tn = np.clip((np.power(10.0, -Dc) - tw) / np.maximum(1.0 - tw, 1e-6), 0.0, 1.0)
            if oem_shape is not None:                                         # MEASURED OEM tone shape (gentle)
                disp = oem_shape[(tn * (len(oem_shape) - 1) + 0.5).astype(np.int32)]
            else:                                                            # legacy ClientColNegLut (harsh)
                disp = oem_lut[(tn * 16383.0 + 0.5).astype(np.int32)]
                disp = np.power(np.clip(disp, 0.0, 1.0), 1.0 / display_gamma)
        else:
            disp = filmic_look(np.power(s, 1.0 / display_gamma), look)        # P2 tone (display-only)
        if black_lift > 0:                              # open the shadows (film-scan toe; the OEM/SP3000
            disp = black_lift + (1.0 - black_lift) * disp   # keeps shadows off true-black). non-film is
        # masked to 0 below, so the frame BORDER stays black while SCENE shadows lift.
        if wb_trim != (1.0, 1.0, 1.0):                  # manual per-channel WB trim (default = no-op)
            disp = np.clip(disp * np.asarray(wb_trim, np.float32), 0.0, 1.0)
        if chroma_nr > 0:
            disp = chroma_denoise(disp, chroma_radius, chroma_nr)
        if luma_nr > 0:                                 # edge-preserving luma NR (grain smoothing; keeps edges)
            disp = luma_denoise(disp, luma_nr, luma_radius, luma_thresh)
        if saturation:                                  # P6 OEM-profile saturation (clean shadows, punchy mids)
            disp = tone_saturation(disp, sat_shadow, sat_mid, sat_high)
        if hsl:                                          # 8-band per-hue HSL (Lightroom-style)
            disp = apply_hsl(disp, hsl)
        if sharpen > 0:
            disp = sharpen_image(disp, sharpen, sharpen_radius, sharpen_thresh)
        d = disp[:, lc:rc]; d[:, ~fcc] = 0.0
        out8[:, c0:c1] = (np.clip(d, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)

    # P5. GRADE — the PRINT contrast expansion + endpoint anchoring (the RA4-paper second stage). The OEM
    #     grade assets we have (filmLut, fugc-generic0225) are IDENTITY/neutral, so they carry no contrast;
    #     the negative is low-gamma (~0.6) and without this stage the positive is washed (blacks ~p40, no
    #     true black/white) and flat. Anchor shadow->~0 / highlight->~255 (per-frame percentiles, same
    #     stretch all channels to keep WB) + a tanh S-curve for system contrast. Applied to the 8-bit
    #     DISPLAY positive only; the 16-bit master stays scene-referred (look-free, for external grading).
    if grade:
        sel = fmask.astype(bool)
        lum = out8.mean(2)
        vals = lum[sel] if sel.any() else lum.ravel()
        lo = float(np.percentile(vals, grade_lo)); hi = float(np.percentile(vals, grade_hi))
        g = np.clip((out8.astype(np.float32) - lo) / max(hi - lo, 1.0), 0.0, 1.0)   # endpoint stretch
        if contrast and contrast > 0:                                                # tanh S-curve contrast
            k = float(contrast)
            g = 0.5 + np.tanh(k * (g - 0.5)) / (2.0 * np.tanh(k * 0.5))
        if density_gamma and density_gamma != 1.0:                                   # midtone density (endpoints anchored)
            g = np.power(np.clip(g, 0.0, 1.0), density_gamma)
        # Endpoint range = the OEM's MEASURED [~10..~250] (12.raw→12.jpg lands here): the OEM LIFTS blacks
        # (~10, not 0) and CAPS whites (~250, not 255) = the soft "lab-print" range. Border stays true 0.
        out8 = (np.clip(g, 0.0, 1.0) * (grade_white - grade_black) + grade_black + 0.5).astype(np.uint8)
        out8[~sel] = 0                                  # keep the non-film border black
        print("  grade: anchor [p%.1f=%.0f, p%.1f=%.0f]->[%.0f,%.0f] + contrast S=%.1f (OEM print range)"
              % (grade_lo, lo, grade_hi, hi, grade_black, grade_white, contrast))

    if return_array:
        # In-process grading (e.g. the psix preview re-grade): hand back the baked
        # 8-bit positive without writing the master/positive TIFFs to disk.
        return out8

    mpath = '%s_master16.tiff' % out_prefix
    tifffile.imwrite(mpath, master, photometric='rgb')
    print("wrote %s  (%dx%d, 16-bit SCENE-REFERRED LINEAR master, slr=%.1f)" % (mpath, h, w, slr))
    ppath = '%s_positive8.tiff' % out_prefix
    tifffile.imwrite(ppath, out8, photometric='rgb')
    print("wrote %s  (%dx%d, 8-bit baked positive, display_gamma=%.2f)" % (ppath, h, w, display_gamma))
    print("  positive: %%black=%.2f %%white=%.2f per-ch means R/G/B=%.0f/%.0f/%.0f"
          % (100 * (out8 < 4).mean(), 100 * (out8 > 251).mean(),
             out8[..., 0].mean(), out8[..., 1].mean(), out8[..., 2].mean()))
    return 0
