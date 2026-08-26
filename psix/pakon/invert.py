#!/usr/bin/env python3
"""pakon_invert — negative->positive, frame-split, STREAMED from the raw .bin (low memory).

Why the .bin and not the decoded tiff: the EP6 .bin is line-sequential (transport order), so ONE FRAME
is a CONTIGUOUS byte range -> we memmap it and process one frame at a time (peak ~hundreds of MB, never
the whole roll). The decoded tiff is row-major, so a frame is a scattered column slice that forces the
whole 700MB file into RAM. This also matches the OEM: the scan buffer is the roll; you select+process
frames out of it.

Pipeline (per frame, all from the .bin):
  read frame's line-range (+halo) -> deinterleave RGB -> flat-field (gentle, per-column PRNU) ->
  channel-register (trilinear roll) -> orient (rot90 cw) -> normalize (exposure+gray-world WB) ->
  global colour transform (OEM-derived poly) -> write.

Frame detection (format-agnostic: 3:2 / half-frame / XPan): per-line gapness (clear+uniform+low-detail)
-> scipy.signal.find_peaks with min-distance = the autocorrelation frame PITCH. Handles variable spacing.

Usage:
  pakon_invert.py --bin captures/scan.bin --flatref captures/..._flatref.npz \
                  --transform captures/oem_global_coef.npy --out-prefix captures/frame [--pitch N]
  pakon_invert.py <rgb16.tiff> [--split] [--transform ...]      # legacy: from a decoded tiff
"""
import json
import os
import sys

import numpy as np
import tifffile
from scipy.signal import find_peaks
from scipy.ndimage import uniform_filter1d, gaussian_filter

from .decode import find_line_phase, apply_flatfield, compute_flatfield_auto  # reuse decode logic

POLY = None
P_LINE = 6000                                            # samples/line = 2000 px x 3 ch interleaved


# ---------- colour / tone ----------
def normalize(rgb):
    """Standardize a frame to a neutral [0,1] positive: per-channel density -> gray-world WB -> white pt."""
    f = rgb.reshape(-1, 3)
    base = np.percentile(f, 99.7, axis=0)
    D = np.clip(-np.log10(np.clip(rgb / base, 1e-4, None)), 0.0, None)
    m = np.maximum(np.percentile(D.reshape(-1, 3), 50, axis=0), 1e-3)
    D *= (m.mean() / m)[None, None, :]
    return np.clip(D / max(float(np.percentile(D, 99.5)), 1e-3), 0.0, 1.0)


def poly_feats(x):
    r, g, b = x[:, 0], x[:, 1], x[:, 2]
    return np.stack([np.ones_like(r), r, g, b, r * r, g * g, b * b, r * g, r * b, g * b,
                     r**3, g**3, b**3, r * r * g, r * r * b, g * g * r, g * g * b, b * b * r, b * b * g, r * g * b], 1)


def chroma_denoise(rgb, radius, amount):
    """Remove amplified CCD COLOUR noise (the magenta/green shadow blotches) while keeping luminance
    detail: split luma + per-channel colour, gaussian-blur ONLY the colour, recombine."""
    if amount <= 0:
        return rgb
    Y = 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
    ch = rgb - Y[..., None]
    chb = np.stack([gaussian_filter(ch[..., k], radius) for k in range(3)], -1)
    return np.clip(Y[..., None] + ((1.0 - amount) * ch + amount * chb), 0.0, 1.0)


def render_positive(rgb, gamma, contrast, chroma_nr=0.85, chroma_radius=5, shoulder=0.5):
    """Clean per-frame finishing: gentle gray-world normalize -> gamma -> soft filmic shoulder (highlight
    roll-off, no harsh clip) -> brightness-neutral contrast -> CHROMA noise reduction. The overfit one-frame
    POLY transform is OFF by default (it amplifies shadow chroma noise); pass --transform only to test it."""
    s = normalize(rgb)
    if POLY is not None:
        flat = s.reshape(-1, 3); out = np.empty_like(flat)
        for i in range(0, len(flat), 1_000_000):
            out[i:i + 1_000_000] = poly_feats(flat[i:i + 1_000_000]) @ POLY
        out = np.clip(out, 0.0, 1.0).reshape(s.shape)
    else:
        out = np.power(s, 1.0 / gamma)
        if shoulder > 0:                                  # soft toe+shoulder so highlights roll off
            sm = out * out * out * (out * (out * 6.0 - 15.0) + 10.0)
            out = (1.0 - shoulder) * out + shoulder * sm
        if contrast > 0:                                  # brightness-neutral contrast S
            p = min(max(float(np.median(out)), 0.05), 0.95); c = 1.0 + contrast
            out = np.where(out <= p, p * np.power(np.clip(out / p, 0, 1), c),
                           1.0 - (1.0 - p) * np.power(np.clip((1.0 - out) / (1.0 - p), 0, 1), c))
    out = chroma_denoise(out, chroma_radius, chroma_nr)
    return (np.clip(out, 0, 1) * 255.0 + 0.5).astype(np.uint8)


# ---------- channel registration (trilinear CCD offset along transport) ----------
def _best_offset(C, R, rng=40):
    cm = C - C.mean(); rm = R - R.mean(); best = (-9.0, 0)
    for d in range(-rng, rng + 1):
        x = np.roll(cm, d, 0)[rng:-rng].ravel(); y = rm[rng:-rng].ravel()
        v = float((x * y).sum() / (np.sqrt((x * x).sum() * (y * y).sum()) + 1e-9))
        if v > best[0]: best = (v, d)
    return best[1]


# Measured CCD dark floor (bias offset) of this F135, per channel R,G,B, from the 3-channel
# calibration sidecars (z['dark'] medians ~309/294/254, ~0.5% of the open-gate white). It is a flat
# ~constant bias, NOT per-column. Used when a scan has no calibrated dark of its own (the IR/4-channel
# flatref capture does not store one — see below).
SENSOR_DARK_RGB = (309.0, 294.0, 254.0)


def _flat_dark(planes):
    """A FLAT per-channel dark at the true sensor floor (SENSOR_DARK_RGB), as (width,) arrays.

    NB: do NOT fall back to compute_flatfield_auto's dark when a scan lacks a calibrated dark.
    That estimator takes the per-column 2nd-percentile of the FRAME, i.e. the darkest *scene* content
    per column. On a dim IR/4-channel scan the darkest content is ~3500 (10-14x the true ~280 sensor
    dark) AND carries the scene's per-column structure, so subtracting it injects a per-column band
    (a translucent striped 'band' visible especially in skies/flats). A flat sensor-floor dark removes
    that band while keeping a physically correct (small) dark subtraction."""
    w = planes[0].shape[1]
    return [np.full(w, SENSOR_DARK_RGB[c], np.float32) for c in range(3)]


def _flat_whites(flatref_path, sample_planes):
    """Load the gentle (per-column PRNU) white/dark references; replicate pakon_decode's exposure scaling."""
    if not flatref_path:
        return compute_flatfield_auto(sample_planes)[0], _flat_dark(sample_planes)
    z = np.load(flatref_path)
    whites = [z['white'][c].astype(np.float32) for c in range(3)]
    if max(float(w.max()) for w in whites) > 64000:                 # clipped (IR scan) -> auto white
        return compute_flatfield_auto(sample_planes)[0], _flat_dark(sample_planes)
    if whites[0].shape != sample_planes[0][0].shape:                # sidecar from another geometry
        print("  flat-field: sidecar width %d != stream %d — ignoring (auto)"
              % (whites[0].shape[0], sample_planes[0][0].shape[0]))
        return compute_flatfield_auto(sample_planes)[0], _flat_dark(sample_planes)
    if 'exp_open' in z and 'exp_film' in z:
        r = z['exp_film'].astype(np.float32) / np.maximum(z['exp_open'].astype(np.float32), 1.0)
        whites = [whites[c] * r[c] for c in range(3)]
    # dark may be from the calibration-pass geometry (different width than transport) —
    # fall back to a self-derived dark rather than rejecting the whole sidecar
    if 'dark' in z and z['dark'][0].shape == sample_planes[0][0].shape:
        darks = [z['dark'][c].astype(np.float32) for c in range(3)]
    else:
        darks = _flat_dark(sample_planes)
    return whites, darks


def _rebate_refs(binpath, phase, frames, P, vis_samples):
    """Per-column clear-film references measured on the REBATE gaps between frames (content-free,
    unlike percentile-over-the-strip estimates which are biased by bright scene areas). Returns
    (whites [3 x W], white_ir [W_ir]) or (None, None) when the strip has no usable gaps.
    Only the per-column SHAPE matters: apply_flatfield normalises each channel to its own median."""
    spans = []
    for (a1, b1), (a2, b2) in zip(frames, frames[1:]):
        lo, hi = b1 + 20, a2 - 20
        if hi - lo > 60:
            spans.append((lo, min(hi, lo + 400)))
    if not spans:
        return None, None
    acc = [[], [], []]; acc_ir = []
    for lo, hi in spans:
        seg = _read_lines(binpath, phase, lo, hi, P).astype(np.float32)
        for c in range(3):
            acc[c].append(seg[:, :vis_samples][:, c::3])
        if P > vis_samples:                                # IR block = trailing samples
            acc_ir.append(seg[:, vis_samples:P])
    whites = [np.percentile(np.vstack(a), 80, axis=0).astype(np.float32) for a in acc]
    white_ir = np.percentile(np.vstack(acc_ir), 80, axis=0).astype(np.float32) if acc_ir else None
    return whites, white_ir


# ---------- frame detection (clear-rebate gap method on the green channel) ----------
def detect_frames_oem(green_mean, green_std, detail, pitch_override=None):
    """OEM-style 35mm frame detection on the GREEN channel (no sprockets):
      - the inter-frame REBATE gap = clear unexposed film = the BRIGHTEST, most UNIFORM green (film-base
        Dmin transmits the most light, above ANY scene). Detect gaps as green within a tight ABSOLUTE band
        around that clear level AND low CCD-variance (uniform) — NOT a relative 'bright' signal (which
        confuses dark scenes for gaps). [DetectWhite_G + LoLim/HiLim band + Variance check.]
      - Stage 1 'LookForNicePictures': group gap lines -> gap centres = confident frame boundaries.
      - Stage 2 'FramingLookInBetweenEnds': estimate the pitch from confident spacing and fill any span
        wider than ~1.6 pitch with evenly-spaced boundaries (handles missed/weak gaps; any format)."""
    gm = uniform_filter1d(green_mean.astype(np.float32), 9)
    gs = uniform_filter1d(green_std.astype(np.float32), 9)
    n = len(gm)
    film = detail > 0.15 * np.percentile(detail, 80)          # film extent (excl. leader/open-gate/dark tail)
    fi = np.where(film)[0]
    if len(fi) < 50:
        return [(0, n)], 0
    fa, fb = int(fi[0]), int(fi[-1])
    # rebate ("white") level from the MIDDLE of the span: the open-gate leader/tail rows
    # (clipped at 65534) sit at the extremes and would drag p99 to the clip level, making
    # the real rebates fail the gap band (measured on an F-135 roll: rebates ~30k, leader 65k).
    lo = fa + (fb - fa) * 20 // 100
    hi = fa + (fb - fa) * 80 // 100
    white = float(np.percentile(gm[lo:hi], 99)) if hi > lo else float(np.percentile(gm[fa:fb], 99))
    umed = float(np.median(gs[fa:fb]))
    # uniformity as COEFFICIENT OF VARIATION (std/level): absolute std scales with
    # brightness, so a dark film row always "beats" a bright rebate row and the gap
    # band never fires on a low-light scan (measured on an F-135: rebate CV 0.12 vs
    # film 0.23 — clear film IS the more uniform surface, relatively).
    cv = gs / np.maximum(gm, 1.0)
    cmed = float(np.median(cv[fa:fb]))
    gapline = (gm >= 0.85 * white) & (gm <= 1.25 * white) & (cv < 0.85 * cmed)  # clear band + uniform (gap)
    gapline[:fa] = False; gapline[fb + 1:] = False
    # gap RUNS as (centre, conf, gstart, gend) — gstart/gend are the rebate edges = the frame's image edges
    runs = []
    i = fa
    while i <= fb:
        if gapline[i]:
            j = i
            while j <= fb and gapline[j]:
                j += 1
            runs.append((int((i + j) // 2), float(gm[i:j].mean() / white) * (j - i), int(i), int(j))); i = j
        else:
            i += 1
    if pitch_override:
        pitch = int(pitch_override)
    else:
        d = np.diff(sorted(r[0] for r in runs)); d = d[d > 200]
        pitch = int(np.median(d)) if len(d) else (fb - fa)
    medw = int(np.median([r[3] - r[2] for r in runs])) if runs else 80   # typical rebate (gap) width
    # ends are zero-width "gaps" at the film extent (so partial first/last frames are preserved)
    items = sorted([(fa, 1e18, fa, fa)] + runs + [(fb, 1e18, fb, fb)])
    kept = [items[0]]                                         # confidence merge of INTERIOR gaps only:
    for it in items[1:]:                                      # if two are < 0.6 pitch apart keep the STRONGER
        if it[0] - kept[-1][0] < 0.6 * pitch and it[1] < 1e17 and kept[-1][1] < 1e17:
            if it[1] > kept[-1][1]:
                kept[-1] = it
        else:
            kept.append(it)
    full = [kept[0]]                                          # fill big spans with synthetic gaps (by pitch)
    for k in range(1, len(kept)):
        span = kept[k][0] - full[-1][0]
        if span > 1.6 * pitch:
            ndiv = max(1, int(round(span / float(pitch)))); step = span // ndiv
            for m in range(1, ndiv):
                c = full[-1][0] + step
                full.append((c, 0.0, c - medw // 2, c + medw // 2))   # synthetic gap of typical width
        full.append(kept[k])
    # FRAME = image extent between gaps: previous gap's END (gend) -> next gap's START (gstart).
    frames = [(full[k][3], full[k + 1][2]) for k in range(len(full) - 1)
              if full[k + 1][2] - full[k][3] > 0.5 * pitch]
    return frames, pitch


# ---------- §3 polish: format auto-classify + detection sanity guard ----------
# 35mm geometry: the 2000px line dimension = the film HEIGHT (~24mm across the strip); a frame's
# transport length (pitch) = its WIDTH along the film. So aspect = pitch/2000 = frame_width / 24mm.
# Named formats by that ratio (image ~1.5:1 etc.; pitch carries a small rebate, so bands are generous):
FORMATS = [                                                # (name, image_aspect, lo, hi) on pitch/2000
    ('half-frame', 0.75, 0.60, 0.98),                      # 18x24mm
    ('full-frame 35mm (3:2)', 1.50, 1.30, 1.95),           # 36x24mm  (203719 measured 1.60)
    ('XPan / panoramic', 2.71, 2.40, 3.05),                # 65x24mm
]


def classify_format(pitch, frames, nlines, line_px=2000):
    """Map the measured pitch to a named 35mm format + judge whether the detection is PLAUSIBLE.
    Returns (format_name, aspect, plausible: bool, reason). A 'detection' that is one strip-spanning
    segment (no interior rebate gaps) or an out-of-range aspect is flagged degenerate, NOT a real frame."""
    aspect = pitch / float(line_px)
    if len(frames) <= 1 and pitch > 0.7 * nlines:
        return ('unknown', aspect, False, 'single strip-spanning segment — no inter-frame rebate gaps found')
    for name, _ideal, lo, hi in FORMATS:
        if lo <= aspect <= hi:
            return (name, aspect, True, 'aspect %.2f:1 in %s band' % (aspect, name))
    return ('unknown', aspect, False, 'aspect %.2f:1 outside any known format band (0.6..3.05)' % aspect)


# ---------- IR-ICE dust/scratch removal (the OEM's 'scratch removal') ----------
# Dust/scratches are OPAQUE to infra-red while the film dyes are TRANSPARENT to it, so the IR channel
# shows ONLY physical defects (sharp drops in IR transmission) on an otherwise image-independent field.
def ir_defect_mask(ir, white_ir, thresh, kernel=41, min_size=3):
    """Detect dust/scratches in the IR plane via OpenCV BLACK-HAT morphology — the standard dark-defect-on-
    bright-background detector, and what produced the validated ir_ff_overlay detection (RE-confirmed
    2026-06-26: cv2 black-hat reproduces that overlay; the old grey_closing(5,5) baseline only caught defect
    EDGES and missed the body of anything >~5px, so dust survived the inpaint).

    ir: (nlines, 2000) raw IR plane; white_ir: (2000,) open-gate IR per-column white. Per-column flat-field
    (PRNU) -> 8-bit normalised to the clear-film level -> black-hat (= morphological closing − image) with an
    ELLIPSE kernel >= the largest defect: closing fills dust/scratches with the surrounding CLEAR-film level,
    so black-hat = how much DARKER each pixel is than its local clear neighbourhood, across the whole defect
    BODY (not just edges). Returns a bool mask. Three knobs:
      thresh   — defect DEPTH as a fraction of the clear level (bh/255); higher = only deeper/surer defects.
      kernel   — black-hat window (px); the widest defect it can fully resolve. Must exceed the widest real
                 defect (too small misses big-defect bodies, e.g. the old 5px).
      min_size — discard connected defect blobs SMALLER than this many px (8-connectivity), so single-pixel
                 IR noise/grain specks are not flagged/inpainted. 1 = no size filter."""
    import cv2
    ff = ir / np.maximum(white_ir[None, :], 1.0) * float(np.median(white_ir))   # remove IR PRNU stripes
    base = float(np.percentile(ff, 95))                                          # clear-film IR level
    ir8 = np.clip(ff / max(base, 1.0) * 255.0, 0, 255).astype(np.uint8)
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (int(kernel), int(kernel)))
    bh = cv2.morphologyEx(ir8, cv2.MORPH_BLACKHAT, ker)                          # 0..255 = local defect depth
    mask = (bh.astype(np.float32) / 255.0) > thresh
    if min_size and int(min_size) > 1 and mask.any():                            # drop sub-min_size noise blobs
        n, lab, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
        keep = stats[:, cv2.CC_STAT_AREA] >= int(min_size)
        keep[0] = False                                                          # label 0 = background
        mask = keep[lab]
    return mask


def inpaint_rgb(planes, mask, radius=3, grain=True):
    """Fill defect (dust/scratch) pixels with OpenCV Telea inpainting, PER CHANNEL on the 16-bit linear
    plane. Telea propagates the surrounding image structure/gradient into the hole (smoother + edge-aware
    vs the old nearest-neighbour distance-transform fill). Done per channel because cv2.inpaint supports
    16-bit *single*-channel but NOT 16-bit 3-channel — this keeps the archival neg at full 16-bit precision.
    Only masked pixels are replaced; every other pixel keeps its full-precision float value. Mask dilated
    1px to cover the defect halo. planes: list of (h,w) float32; mask: (h,w) bool.

    GRAIN: Telea gives a SMOOTH fill, which reads as plasticky against film grain. So we re-inject matched
    grain into the healed pixels — synthetic zero-mean noise scaled to the LOCAL grain amplitude, estimated
    per channel as a robust MAD of the high-frequency residual (filled − gaussian) over the non-defect area
    (MAD ignores edges, so it tracks the grain floor, not image detail). Deterministic seed -> stable
    re-develops. (Idea borrowed conceptually from NegPy's grain-synthesis healing; clean-room impl.)"""
    if not mask.any():
        return planes
    import cv2
    ker = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))                      # 4-conn, == old binary_dilation
    m8 = cv2.dilate(mask.astype(np.uint8), ker, iterations=1) * np.uint8(255)
    m = m8 > 0
    nonm = ~m
    rng = np.random.default_rng(0)                                               # deterministic grain
    out = []
    for p in planes:
        src = np.clip(p, 0, 65535).astype(np.uint16)
        filled = cv2.inpaint(src, m8, radius, cv2.INPAINT_TELEA).astype(np.float32)
        q = p.copy()
        q[m] = filled[m]
        if grain and nonm.any():
            hp = filled - cv2.GaussianBlur(filled, (0, 0), 1.0)                  # high-freq = grain (+ a few edges)
            r = hp[nonm]
            sigma = 1.4826 * float(np.median(np.abs(r - np.median(r))))          # robust grain amplitude (MAD)
            if sigma > 0:
                # bound grained pixels to the REAL data range, NOT [0,65535]: an unbounded clip lets grain
                # push a dark healed pixel to 0, which becomes infinite density downstream and wrecks the
                # max-based scene balance (-> runaway colour cast). lo>=1 guarantees no zero/spike pixels.
                lo = max(1.0, float(np.percentile(p[nonm], 0.1)))
                hi = float(np.percentile(p[nonm], 99.9))
                q[m] = np.clip(q[m] + rng.standard_normal(int(m.sum())).astype(np.float32) * sigma, lo, hi)
        out.append(q)
    return out


# ---------- per-frame-from-.bin ----------
def _read_lines(binpath, phase, a, b, P):
    """Read lines [a,b) from the .bin via explicit np.fromfile (no memmap) -> (nlines, P) uint16."""
    raw = np.fromfile(binpath, dtype='<u2', count=(b - a) * P, offset=(phase + a * P) * 2)
    return raw[:len(raw) // P * P].reshape(-1, P)


def write_raw_negative(rgb, out_prefix, idx, meta, ir_plane=None):
    """Write the §3 ARCHIVAL raw-negative for one frame: 16-bit LINEAR, flat-fielded (gentle PRNU),
    channel-registered, oriented, ORANGE-MASK INTACT (no base-subtract, no invert, no tone). This is
    the stable interface between scan and positive conversion — pakon_finish consumes it directly
    (it expects exactly this: 'flat-fielded, linear, 16-bit, orange-mask-intact RGB'). Provenance is
    embedded in the TIFF ImageDescription (JSON) + a sidecar .json. Returns the path."""
    neg = np.clip(rgb, 0, 65535).round().astype('<u2')
    p = '%s_f%02d_neg.tiff' % (out_prefix, idx)
    desc = json.dumps(meta, separators=(',', ':'))
    tifffile.imwrite(p, neg, photometric='rgb', description=desc)
    with open('%s_f%02d_neg.json' % (out_prefix, idx), 'w') as fh:
        json.dump(meta, fh, indent=2)
    if ir_plane is not None:                               # archive the IR plane too (re-run ICE later)
        iru = np.clip(ir_plane, 0, 65535).round().astype('<u2')
        tifffile.imwrite('%s_f%02d_neg_ir.tiff' % (out_prefix, idx), iru)
    return p


def process_bin(binpath, flatref, out_prefix, gamma, contrast, pitch_override, ir_thresh=0.04, ir_offset=0,
                raw_neg=False, neg_only=False, skip_blank=False, ir_kernel=41, ir_min_size=3,
                ir=None):
    nbytes = os.path.getsize(binpath)
    head = np.fromfile(binpath, dtype='<u2', count=8000 * 3000)
    mk = np.flatnonzero(head & 1); d = np.diff(mk); d = d[(d > 1000) & (d < 9000)]
    P = 6000
    if len(d):
        m = int(np.bincount(d).argmax())
        if 3000 <= m <= 9000:
            P = m          # the stream's own line period (idx5-dependent: 8000 @ 2073, 5910 @ legacy 2043)
    # ir: None = infer from the line period (P==8000 -> IR). Callers that know the scan mode
    # pass it explicitly.
    ir_mode = (P == 8000) if ir is None else bool(ir)
    # IR line layout: [visible*3 RGB][trailing 2000-sample IR block] (HW-measured 2026-08-26,
    # roll23: phase breaks exactly at P-2000). 7880 = legacy idx5=2043 captures (5910+1970).
    IR_RGB = 5910 if P == 7880 else (P - 2000)
    W = (IR_RGB if ir_mode else P) // 3                   # visible width in px
    phase, on_ph, total = find_line_phase(head[:P * 3000], P); phase = phase or 0; del head
    nlines = (nbytes // 2 - phase) // P
    white_ir = None
    if ir_mode and flatref:
        z = np.load(flatref)
        if 'white_ir' in z:
            white_ir = z['white_ir'].astype(np.float32)
            if white_ir.shape[0] != P - IR_RGB:            # sidecar sliced with a stale layout
                print("  flat-field: white_ir width %d != stream %d — ignoring (auto)"
                      % (white_ir.shape[0], P - IR_RGB))
                white_ir = None
    if ir_mode and white_ir is None:
        white_ir = 'rebate'                             # deferred: derived from the gap rows after detection
        # (the old per-column percentile over the whole strip was content-biased: columns through
        #  bright scene areas got inflated whites -> false ICE columns -> banded inpaints)
    print("line=%d (%s), phase=%d, ~%d lines%s" % (P, "RGB+IR" if ir_mode else "RGB", phase, nlines,
          "" if not ir_mode else (" | IR-ICE %s" % ("ON" if white_ir is not None else "no white_ir->OFF"))))

    bright = np.empty(nlines, np.float32); detail = np.empty(nlines, np.float32); unif = np.empty(nlines, np.float32)
    for a in range(0, nlines, 2000):
        b = min(a + 2000, nlines)
        rgb_vis = IR_RGB if ir_mode else W * 3
        g = _read_lines(binpath, phase, a, b, P)[:, :rgb_vis][:, 1::3].astype(np.float32)   # green of RGB block
        bright[a:b] = g.mean(1); detail[a:b] = np.abs(np.diff(g, axis=1)).mean(1); unif[a:b] = g.std(1)
        del g

    frames, pitch = detect_frames_oem(bright, unif, detail, pitch_override)
    # transport-axis rescale for isotropic output: a full 35mm frame is 36mm ALONG the
    # strip, so the median detected frame length directly gives lines/mm (no assumption
    # about rebate width — the gap varies by camera). Across the strip the window is the
    # 24mm image height. Plus: ~3000 lines/36mm vs ~83 px/mm -> factor 1.0 (no-op);
    # F-135 base16: ~4126 lines/36mm = 114.6 lines/mm vs 82.1 px/mm -> x0.716.
    iso_f = None
    if frames and len(frames) >= 2:
        # (a single strip-spanning "frame" gives a meaningless length — skip)
        # Across px/mm: the window reads the FULL 24mm frame height in every mode
        # (base16 = 2000px across per pakon-reference image-stream.md; scans are never
        # cropped below 24mm). Transport lines/mm comes from the measured frame pitch
        # (a full frame is 36mm): at the OEM base16 motor rate the transport samples
        # ~114.6 lines/mm — 1.376x denser than across. The OEM oversamples the same way
        # and resamples to its 2000x3000 output; the isotropic rescale below reproduces
        # exactly that downsampling (output 2000x~3003 = TLX geometry).
        px_per_mm = W / 24.0
        lens = sorted(b - a for a, b in frames)
        med = lens[len(lens) // 2]
        good = [L for L in lens if 0.7 * med <= L <= 1.4 * med]      # drop partials/outliers
        if good and med > 300:
            lines_per_mm = (sum(good) / len(good)) / 36.0
            f_ = px_per_mm / lines_per_mm
            if 0.2 < f_ < 2.0 and abs(f_ - 1.0) > 0.02:
                iso_f = f_
                print("  isotropic scale: frames %d lines / 36mm = %.1f lines/mm vs %.1f px/mm"
                      " across -> transport x%.3f" % (sum(good) / len(good), lines_per_mm, px_per_mm, iso_f))
    if not frames:
        print("no film content found"); return 1
    fmt, aspect, plausible, why = classify_format(
        pitch, frames, nlines, line_px=W / iso_f if iso_f else W)   # post-rescale width
    print("pitch=%d px (%.2f:1); %d frames; format=%s" % (pitch, aspect, len(frames), fmt))
    if not plausible:
        print("  !! DEGENERATE DETECTION: %s" % why)
        print("     -> this is NOT a normal multi-frame roll. Treating the strip as ONE segment.")
        print("     -> if it really is multi-frame, pass --pitch N (e.g. full-frame ~%d)." % (int(1.6 * 2000)))

    # per-frame BLANK (unexposed: near-zero image detail) + PARTIAL (clipped first/last) flags
    fdetail = [float(detail[a:b].mean()) for a, b in frames]
    detmed = float(np.median(fdetail)) if fdetail else 0.0
    blank_flags = [d < 0.20 * detmed for d in fdetail]     # << typical content = unexposed/clear
    last = len(frames) - 1
    partial_flags = [(b - a) < 0.6 * pitch and (k in (0, last)) for k, (a, b) in enumerate(frames)]
    nblank = sum(blank_flags); npart = sum(partial_flags)
    if nblank or npart:
        print("  frame flags: %d blank (unexposed), %d partial (clipped edge)%s"
              % (nblank, npart, "; --skip-blank ON" if skip_blank else ""))

    cs, ce = frames[len(frames) // 2]; cw = min(10000, ce - cs)
    seg = _read_lines(binpath, phase, cs, cs + cw, P).astype(np.float32)
    spl = [seg[:, :W*3][:, c::3] for c in range(3)]
    off = [0, _best_offset(spl[1], spl[0]), _best_offset(spl[2], spl[0])]
    whites, darks = _flat_whites(flatref, spl)
    # rebate-derived references: content-free per-column whites. The percentile-over-sample
    # auto white is scene-biased (columns through bright scene areas get inflated whites ->
    # ±10% across-axis gain bands). Override the AUTO path only — a valid open-gate sidecar
    # still wins for RGB; white_ir is only 'rebate' when the sidecar's was invalid.
    sidecar_ok = False
    if flatref:
        z = np.load(flatref)
        if 'white' in z and z['white'].shape[-1] == W and max(float(z['white'].max()), 1) <= 64000:
            sidecar_ok = True
    want_rebate_ir = isinstance(white_ir, str) and white_ir == 'rebate'
    if not sidecar_ok or want_rebate_ir:
        rw, rw_ir = _rebate_refs(binpath, phase, frames, P, W * 3)
        if not sidecar_ok and rw is not None:
            whites = rw
            print("  flat-field: rebate-derived per-column white (%d gap span(s))" % (len(frames) - 1))
        if want_rebate_ir:
            white_ir = rw_ir
            if white_ir is not None:
                print("  flat-field: rebate-derived white_ir (%d cols)" % white_ir.shape[0])
    if isinstance(white_ir, str):                      # no usable gaps -> no ICE (never leak the sentinel)
        white_ir = None
    order = sorted(range(3), key=lambda c: off[c])
    halo = max(abs(o) for o in off) + abs(ir_offset) + 5
    del seg, spl

    ndef = 0
    for i, (a, b) in enumerate(frames):
        if skip_blank and blank_flags[i]:
            print("  frame %2d: lines %6d..%6d -> SKIPPED (blank/unexposed)" % (i, a, b))
            continue
        flag = ("  [BLANK]" if blank_flags[i] else "") + ("  [PARTIAL]" if partial_flags[i] else "")
        a2 = max(0, a - halo); b2 = min(nlines, b + halo)
        seg = _read_lines(binpath, phase, a2, b2, P).astype(np.float32)
        planes = apply_flatfield([seg[:, :W*3][:, c::3] for c in range(3)], whites, darks, balance=False)
        planes = [np.roll(planes[c], off[c], axis=0) for c in range(3)]
        dusty_planes = None
        if ir_mode and white_ir is not None:               # openICE: detect defects, inpaint RGB (pre-orient)
            # Use openICE if available, otherwise fall back to black-hat
            try:
                from .ice import clean, create_ice_options, Calibration
                import numpy as np
                
                # Estimate calibration from current frame (simplified)
                # In a full implementation, this would come from a prescan
                ir_float = ir.astype(np.float64)
                # Find clear pixels (top 5% by IR value)
                flat_ir = ir_float.flatten()
                threshold_idx = max(1, len(flat_ir) // 20)  # Top 5%
                clear_pixels = np.partition(flat_ir, -threshold_idx)[-threshold_idx:]
                ir_ref_est = np.mean(clear_pixels)
                
                # Estimate c (R->IR leakage) by correlating R and IR in clear areas
                # For simplicity, use a typical value or estimate from data
                # TODO: Better estimation of c
                c_est = 0.1  # Typical R->IR leakage
                
                # Create calibration
                cal = Calibration(c=c_est, ir_ref=float(ir_ref_est))
                
                # Create options (using defaults that match PSIX usage)
                opts = create_ice_options(
                    model="Ls9000",  # Assume LS-9000 as default
                    quality="Normal", 
                    dpi=2000,  # Will be overridden if we had actual DPI
                    metering_target=0.95
                )
                
                # Apply openICE
                # planes is [R, G, B] as float32 arrays after flatfield
                # Convert to uint16 for the clean function (approximate)
                color_planes = []
                for plane in planes:
                    # Clip and convert to uint16 range
                    clipped = np.clip(plane, 0, 65535)
                    color_planes.append(clipped.astype(np.uint16))
                
                # Apply the openICE reconstruction
                pixels_fixed = clean(
                    color=color_planes,
                    ir=np.ascontiguousarray(ir.astype(np.uint16)),
                    cal=cal,
                    rows=planes[0].shape[0],
                    cols=planes[0].shape[1],
                    opts=opts
                )
                
                # Convert back to float32 and store
                for i in range(3):
                    planes[i] = color_planes[i].astype(np.float32)
                
                ndef += pixels_fixed
                if pixels_fixed > 0:
                    dusty_planes = [p.copy() for p in planes]  # Keep copy for sidecar
                    
            except Exception as e:
                # Fall back to original black-hat ICE if openICE fails
                print(f"  Warning: openICE failed ({e}), falling back to black-hat")
                ir = np.roll(seg[:, IR_RGB:P], ir_offset, axis=0)
                mask = ir_defect_mask(ir, white_ir, ir_thresh, ir_kernel, ir_min_size)
                ndef += int(mask.sum())
                if mask.any():
                    dusty_planes = planes                       # keep pre-inpaint for the ICE before/after sidecar
                    planes = inpaint_rgb(planes, mask)
        cr = slice(a - a2, a - a2 + (b - a))

        def _orient(pl):                                    # planes -> oriented RGB neg (channel reorder + rot)
            # rot90 k=3 for BOTH generations (validated against PSI output 2026-08-25;
            # the earlier F-135 "wrong orientation" was the resize axis-swap bug, not this)
            return np.ascontiguousarray(np.rot90(
                np.dstack([pl[order[2]][cr], pl[order[1]][cr], pl[order[0]][cr]]), 3)).astype(np.float32)
        rgb = _orient(planes)
        # ISOTROPIC RESCALE: the two axes sample the film at different px/mm unless the
        # transport rate happens to match the CCD (true on a Plus, NOT on an F-135, whose
        # base16 runs ~118 lines/mm vs 82 px/mm across -> frames 1.44x too tall). The
        # scale comes from the film stock itself (35.00mm width, 4.234mm perforation
        # pitch — both standards, no scanner-specific gap assumption); falls back to the
        # measured-pitch/38mm estimate when the perforations aren't detectable.
        try:
            import cv2
            f = iso_f
            if f and 0.2 < f < 2.0 and abs(f - 1.0) > 0.02:
                # transport axis = cols on the oriented neg; cv2 dsize is (width, height)
                nh = max(1, int(round(rgb.shape[1] * f)))
                rgb = np.ascontiguousarray(cv2.resize(rgb, (nh, rgb.shape[0]), interpolation=cv2.INTER_AREA))
                print("  frame %2d: isotropic rescale x%.3f (transport -> %d px)" % (i, f, rgb.shape[1]))
        except Exception:                                  # noqa: BLE001 — never fail the neg for a resize
            pass
        if raw_neg or neg_only:                            # §3 archival 16-bit raw-negative per frame
            ir_arch = None
            if ir_mode:                                    # oriented IR plane (independent of ICE)
                ir_arch = np.ascontiguousarray(np.rot90(
                    np.roll(seg[:, IR_RGB:P], ir_offset, axis=0)[cr], 3)).astype(np.float32)
            meta = {'kind': 'raw-negative', 'space': 'linear', 'mask': 'orange-intact',
                    'flatfield': 'gentle-prnu', 'invert': False, 'tone': False,
                    'frame': i, 'lines': [int(a), int(b)], 'pitch_px': int(pitch),
                    'aspect': round(pitch / 2000.0, 3), 'format': fmt, 'plausible': bool(plausible),
                    'blank': bool(blank_flags[i]), 'partial': bool(partial_flags[i]), 'line_px': int(P),
                    'channels': 4 if ir_mode else 3, 'order': 'RGB', 'reg_offsets': [int(o) for o in off],
                    'orient': 'cw', 'source_bin': os.path.basename(binpath),
                    'flatref': os.path.basename(flatref) if flatref else None,
                    'consumer': 'pakon_finish.py (expects linear 16-bit orange-intact RGB)'}
            np_path = write_raw_negative(rgb, out_prefix, i, meta, ir_arch)
            if dusty_planes is not None:                   # ICE before/after sidecar: the dusty pixels ICE replaced
                rgb_dusty = _orient(dusty_planes)
                if rgb_dusty.shape != rgb.shape:           # rgb was isotropically rescaled above — match it
                    try:
                        import cv2
                        rgb_dusty = np.ascontiguousarray(cv2.resize(
                            rgb_dusty, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_AREA))
                    except Exception:                      # noqa: BLE001 — sidecar only, never fail the neg
                        rgb_dusty = None
                if rgb_dusty is not None:
                    d = np.any(np.abs(rgb - rgb_dusty) > 0.5, axis=2)
                    ys, xs = np.where(d)
                    if len(ys):
                        np.savez('%s_f%02d_neg_ice.npz' % (out_prefix, i),
                                 yx=np.stack([ys, xs]).astype(np.int32),
                                 vals=np.clip(rgb_dusty[ys, xs], 0, 65535).round().astype('<u2'))
            print("  frame %2d: lines %6d..%6d -> %dx%d  RAW-NEG %s%s%s"
                  % (i, a, b, rgb.shape[1], rgb.shape[0], np_path,
                     "  (+IR)" if ir_arch is not None else "", flag))
        if not neg_only:
            img = render_positive(rgb, gamma, contrast)
            p = '%s_f%02d.tiff' % (out_prefix, i)
            tifffile.imwrite(p, img, photometric='rgb')
            print("  frame %2d: lines %6d..%6d -> %dx%d  %s%s" % (i, a, b, img.shape[1], img.shape[0], p, flag))
            del img
        del seg, planes, rgb
    if ir_mode and white_ir is not None:
        print("IR-ICE: %d defect pixels inpainted (%.3f%% of film)" % (ndef, 100.0 * ndef / (nlines * 2000)))
    return 0
