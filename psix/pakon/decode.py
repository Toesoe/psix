#!/usr/bin/env python3
"""pakon_decode: decode a raw EP6 capture into an RGB TIFF.

EP6 format (from USB captures):
  - 16-bit little-endian CCD samples
  - RGB interleaved per pixel: R,G,B,R,G,B,...  (IR adds a 4th channel when enabled)
  - line = width px x channels; transport (film motion) is the long axis (one line per CCD read)

Writes a full 16-bit RGB TIFF and a downsampled, percentile-scaled 8-bit preview (the replayed
calibration over-exposes, so the preview auto-levels). Note: for colour NEGATIVE film the result
looks like a negative (orange mask, inverted) — that's expected at this stage.

Usage: pakon_decode.py <ep6.bin> [--width 2000] [--channels 3] [--order RGB] [--out-prefix captures/scan]
"""
import os
import sys

import numpy as np
import tifffile


def find_line_phase(raw, P):
    """Recover line-0 from the hardware LINE-SYNC MARKER (from the F135 image protocol): the firmware sets
    bit0 of the FIRST sample of every line. The OEM scans for that bit to frame lines (it does NOT
    byte-count from the trigger, and does NOT image-analyse). Verified empirically: set-bits sit
    exactly P samples apart. We take the dominant (position mod P) as the phase — robust to dark-region
    LSB noise (saturated leaders are even=no marker; dark noise is spread across all phases, the true
    markers all share one phase). Returns (phase, n_markers_on_phase, total_markers)."""
    mk = np.flatnonzero(raw & 1)
    if len(mk) == 0:
        return None, 0, 0
    hist = np.bincount(mk % P, minlength=P)
    phase = int(hist.argmax())
    return phase, int(hist[phase]), int(len(mk))


def detect_line_period(raw):
    """Dominant spacing between line-sync markers (bit0) = the EP6 line length in samples:
    6000 = 3-channel (RGB interleaved), 8000 = 4-channel IR (RGB interleaved 6000 + IR block 2000)."""
    mk = np.flatnonzero(raw & 1)
    if len(mk) < 10:
        return None
    vals, counts = np.unique(np.diff(mk), return_counts=True)
    return int(vals[counts.argmax()])


def compute_flatfield_auto(planes, sat=64000):
    """Self-derived per-column flat-field references (used when no calibration sidecar exists).
    OEM does (raw-dark)*gain, gain=target/(white-dark), per column per channel. Here white[col] =
    90th-pct of lit, non-clipped lines
    (~ the open-gate clear-base response = illumination*PRNU), dark[col] = 2nd-pct (~ the dark offset).
    Returns (whites, darks): per-phase (width,) arrays. Approximate vs a real open-gate reference, but
    it removes the same per-column border falloff + PRNU stripes."""
    whites, darks = [], []
    for p in planes:
        lm = p.mean(1)
        lit = (lm > np.percentile(lm, 70)) & (p.max(1) < sat)
        rows = p[lit] if lit.sum() > 50 else p
        whites.append(np.percentile(rows, 90, axis=0))
        darks.append(np.percentile(p, 2, axis=0))
    return whites, darks


def apply_flatfield(planes, whites, darks, balance=True):
    """Per-column flat-field: corrected = (raw - dark) * target/(white - dark).
    The OEM uses a SINGLE gain constant for ALL channels, so the open-gate white maps
    to the SAME level in every channel -> NEUTRAL white (it also corrects the raw RGB-LED per-channel
    imbalance, e.g. B ~2.8x R). balance=True replicates that: one common target across channels.
    balance=False uses each channel's own median (preserves the raw colour cast -> e.g. blue open gate).
    Margin/optical-black columns (white << median) get a neutral scalar gain (no noise amplification);
    gain clamped (the OEM clamps its 16.16 gain to 0x3ffff ~ 4x; we allow a bit more for white balance)."""
    denoms = [w - d for w, d in zip(whites, darks)]
    meds = [float(np.median(dn[dn > 0])) if np.any(dn > 0) else 1.0 for dn in denoms]
    target = max(meds) if balance else None          # common -> brightest channel ~unchanged, others lifted
    out = []
    for p, dark, dn, med in zip(planes, darks, denoms, meds):
        t = target if balance else med
        gain = np.where(dn > 0.3 * med, t / np.maximum(dn, 1.0), t / med).astype(np.float32)
        np.clip(gain, 0.0, 6.0, out=gain)
        out.append((p - dark.astype(np.float32)) * gain)
    return out


def decode(path, width, channels, order, out_prefix, preview_step, two_tap=False, tap_split=763,
           reg='auto', orient='cw', flatfield=None, wb=True):
    raw = np.fromfile(path, dtype='<u2')
    ir_plane = None
    period = detect_line_period(raw)
    if period == 8000:                                   # 4-channel IR line: [RGB interleaved 6000][IR 2000]
        width, channels, P = 2000, 3, 8000
        phase, on_ph, total = find_line_phase(raw, P)
        if phase is None:
            phase = 0
        print("  IR scan: 4-channel line (%d samples) -> RGB(6000 interleaved) + IR(2000); phase=%d, ~%d lines"
              % (P, phase, (len(raw) - phase) // P))
        raw = raw[phase:]
        nl = len(raw) // P
        if nl == 0:
            print("ERROR: capture too small for an 8000-sample IR line", file=sys.stderr); return 1
        lines = raw[:nl * P].reshape(nl, P)
        rgb_block = lines[:, :6000]
        planes = [rgb_block[:, c::3].astype(np.float32) for c in range(3)]          # R,G,B each (nl, 2000)
        ir_plane = lines[:, 6000:8000].astype(np.float32)                            # IR (nl, 2000)
    else:
        P = width * channels
        phase, on_ph, total = find_line_phase(raw, P)
        if phase is None:
            print("WARNING: no line-sync markers (bit0) found — falling back to offset 0.", file=sys.stderr)
            phase = 0
        else:
            print("  line-sync marker: phase=%d (line-0 at sample %d); %d/%d markers on-phase, ~%d lines"
                  % (phase, phase, on_ph, total, (len(raw) - phase) // P))
        raw = raw[phase:]                               # align to line-0 (the marked sample)
        nl = len(raw) // P
        if nl == 0:
            print("ERROR: capture too small for width=%d channels=%d (line=%d samples)" % (width, channels, P),
                  file=sys.stderr)
            return 1
        lines = raw[:nl * P].reshape(nl, P)
        planes = [lines[:, c::channels].astype(np.float32) for c in range(channels)]   # each (nl, width)
    idx = {'R': 0, 'G': 1, 'B': 2, 'I': 3}

    # FLAT-FIELD (fixed-pattern) correction — removes the per-column border falloff + PRNU stripes that
    # the OEM corrects in host software (the EP6 stream is RAW). Applied per
    # phase/column BEFORE trilinear registration (independent axis).
    if flatfield:
        if flatfield == 'auto':
            whites, darks = compute_flatfield_auto(planes)
            print("  flat-field: auto (self-derived per-column white/dark from this scan)")
        else:
            z = np.load(flatfield)
            whites = [z['white'][c].astype(np.float32) for c in range(channels)]
            # IR scans capture the RGB white at the OEM IR exposure where the open gate CLIPS -> unusable
            # for the RGB flat-field; fall back to auto for RGB (the IR plane still uses its own white_ir).
            if max(float(w.max()) for w in whites) > 64000:
                whites, darks = compute_flatfield_auto(planes)
                print("  flat-field: RGB white in sidecar is clipped (IR scan) -> RGB auto; IR uses white_ir")
                z = None
            # The white ref is captured at the OPEN-GATE exposure; the film may be scanned at a brighter
            # per-channel FILM exposure (orange-mask boost). Scale the white to the film exposure so the
            # gain = K/(white-dark) is consistent (PRNU is multiplicative -> exposure-independent).
            if z is not None and 'exp_open' in z and 'exp_film' in z:
                eo = z['exp_open'].astype(np.float32); ef = z['exp_film'].astype(np.float32)
                ratio = ef / np.maximum(eo, 1.0)
                whites = [whites[c] * ratio[c] for c in range(channels)]
                print("  flat-field: white scaled to film exposure (R/G/B x%.2f/%.2f/%.2f)"
                      % (ratio[0], ratio[1], ratio[2]))
            if z is None:
                pass                                         # clipped fallback: whites+darks already set (auto)
            elif 'dark' in z:
                darks = [z['dark'][c].astype(np.float32) for c in range(channels)]
                print("  flat-field: sidecar %s (open-gate white + dark references)" % flatfield)
            else:
                _, darks = compute_flatfield_auto(planes)   # white is real; derive dark from the scan
                print("  flat-field: sidecar %s (open-gate white; dark self-derived)" % flatfield)
        planes = apply_flatfield(planes, whites, darks, balance=wb)
        if wb:
            print("  flat-field: white-balanced (open gate -> neutral, common target across channels)")
    # TRILINEAR CCD: R/G/B rows are offset along transport (~8-line spacing). MUST register, but the
    # EP6 stream start-phase varies per scan, so which deinterleave phase is R/G/B (and the offsets)
    # ROTATES between scans -> auto-detect per scan instead of hard-coding.
    if reg == 'auto' and channels == 3:
        lm = lines.astype(np.float32).mean(axis=1)
        sm = np.convolve(lm, np.ones(200) / 200, 'same')
        # measure on REAL CONTENT: not dark, not saturated/clipped (the bright leader is flat -> useless)
        content = np.where((sm > 4000) & (sm < 58000))[0]
        if len(content) > 4000:
            c = content[len(content) // 2]                  # centre of the content
            a, b = max(0, c - 5000), min(nl, c + 5000)
        else:
            a, b = 0, min(nl, 12000)

        def _best(C, R, rng=40):
            cm = C[a:b] - C[a:b].mean(); rm = R[a:b] - R[a:b].mean(); bb = (-9.0, 0)
            for d in range(-rng, rng + 1):
                x = np.roll(cm, d, 0)[rng:-rng].ravel(); y = rm[rng:-rng].ravel()
                v = float((x * y).sum() / (np.sqrt((x * x).sum() * (y * y).sum()) + 1e-9))
                if v > bb[0]: bb = (v, d)
            return bb[1]
        off = [0, _best(planes[1], planes[0]), _best(planes[2], planes[0])]
        planes = [np.roll(planes[c], off[c], axis=0) for c in range(3)]
        o = sorted(range(3), key=lambda c: off[c])          # ascending transport offset = B,G,R
        rgb = np.dstack([planes[o[2]], planes[o[1]], planes[o[0]]])
        print("  auto-register: phase offsets vs p0 = %s -> R=p%d G=p%d B=p%d" % (off, o[2], o[1], o[0]))
    else:
        if reg and reg != 'auto':
            planes = [np.roll(planes[c], reg[c], axis=0) if c < len(reg) else planes[c]
                      for c in range(channels)]
        rgb = np.dstack([planes[idx[ch]] for ch in order if idx[ch] < channels])  # (nl, width, len(order))

    if two_tap:
        # LEGACY fallback only. The "two-tap split at 763 + swap" was a LINE-MISALIGNMENT artifact:
        # once lines are marker-aligned (find_line_phase), the column profile shows NO discontinuity
        # at 763 (verified) and the OEM read path consumes width consecutive RGB
        # triplets with no swap/reversal. Leave off unless debugging an unmarked capture.
        rgb = np.concatenate([rgb[:, tap_split:, :], rgb[:, :tap_split, :]], axis=1)

    # Orientation: the raster is (transport_lines, CCD_width). The transport axis is the FILM-LENGTH
    # (36mm, long) edge and CCD width is the 24mm edge, so a single 35mm frame is landscape only after
    # a 90deg rotation. Confirmed visually on real frames: 90deg CLOCKWISE, no mirror ('cw').
    rot = {'none': 0, 'cw': 3, 'ccw': 1, '180': 2}.get(orient.replace('_mir', ''), 3)
    if rot:
        rgb = np.rot90(rgb, rot)
    if orient.endswith('_mir'):
        rgb = rgb[:, ::-1, :]
    rgb = np.ascontiguousarray(rgb)

    rgb = np.clip(rgb, 0, 65535).astype(np.uint16)
    full = '%s_rgb16.tiff' % out_prefix
    tifffile.imwrite(full, rgb, photometric='rgb')
    print("wrote %s  (%d lines x %d px x %d ch, 16-bit) = %.0f MB"
          % (full, nl, width, rgb.shape[2], os.path.getsize(full) / 1e6))

    # 8-bit auto-leveled preview, downsampled on BOTH axes (keeps the landscape aspect ratio)
    prev = rgb[::preview_step, ::preview_step]
    p1, p99 = np.percentile(prev, [1, 99])
    scaled = np.clip((prev.astype(np.float32) - p1) / max(1.0, p99 - p1) * 255.0, 0, 255).astype(np.uint8)
    pv = '%s_preview8.tiff' % out_prefix
    tifffile.imwrite(pv, scaled, photometric='rgb')
    print("wrote %s  (%d x %d, 8-bit, transport/%d, levels [%.0f..%.0f])"
          % (pv, scaled.shape[0], scaled.shape[1], preview_step, p1, p99))
    print("  (open the preview; for colour negatives it will look inverted/orange — expected)")

    # IR plane (4-channel scan): separate 16-bit grayscale, same orientation as RGB. Flat-fielded per
    # column (same CCD -> same PRNU/vignette stripes as RGB).
    if ir_plane is not None:
        irp = ir_plane                                       # (nl, 2000), pre-orientation
        if flatfield:
            wI = dI = None
            if flatfield != 'auto':                          # sidecar: prefer the OPEN-GATE IR white ref
                z = np.load(flatfield)
                if 'white_ir' in z:
                    wI = [z['white_ir'].astype(np.float32)]
                    _, dI = compute_flatfield_auto([irp])
                    print("  IR: flat-field from sidecar OPEN-GATE IR white reference")
            if wI is None:
                wI, dI = compute_flatfield_auto([irp])       # auto: IR clear base (IR ~flat -> clean)
                print("  IR: flat-field auto (per-column, from the IR clear base)")
            irp = apply_flatfield([irp], wI, dI, balance=False)[0]

        # DEFECT MAP — physical defects (dust/scratch) are LOCAL dips below the surrounding clear film,
        # NOT global dark regions (gaps/edges/faint image bleed). Baseline = block-mean of the IR (follows
        # slow variation); defect = (baseline - IR)/baseline where positive. Dark non-film regions (where
        # the baseline itself is low) are masked out.
        irbase = float(np.percentile(irp, 95))               # clear-film IR level (post flat-field)
        f = 16
        hp, wp = (-irp.shape[0]) % f, (-irp.shape[1]) % f
        xp = np.pad(irp, ((0, hp), (0, wp)), mode='edge')
        blk = xp.reshape(xp.shape[0] // f, f, xp.shape[1] // f, f).mean(axis=(1, 3))
        base = np.repeat(np.repeat(blk, f, 0), f, 1)[:irp.shape[0], :irp.shape[1]]
        base = np.maximum(base, 1.0)
        defect = np.clip((base - irp) / base, 0.0, 1.0)
        defect[base < 0.4 * irbase] = 0.0                    # ignore dark non-film regions (gaps/leader/dark)

        def _orient(x):
            if rot:
                x = np.rot90(x, rot)
            if orient.endswith('_mir'):
                x = x[:, ::-1]
            return np.ascontiguousarray(x)

        iro = _orient(irp)
        iru = np.clip(iro, 0, 65535).astype(np.uint16)
        irf = '%s_ir16.tiff' % out_prefix
        tifffile.imwrite(irf, iru)
        print("wrote %s  (IR plane, %d x %d, 16-bit, flat-fielded)" % (irf, iru.shape[0], iru.shape[1]))

        # FIXED-SCALE IR preview (0..clear-base): honest — clear film ~white, defects dark, no auto-stretch.
        ip = np.clip(iru[::preview_step, ::preview_step].astype(np.float32) / max(irbase, 1.0) * 255.0,
                     0, 255).astype(np.uint8)
        tifffile.imwrite('%s_ir_preview8.tiff' % out_prefix, ip)
        print("  IR preview (fixed scale 0..%.0f): %s_ir_preview8.tiff" % (irbase, out_prefix))

        # DEFECT map outputs (8-bit: white = defect).
        dm = (_orient(defect) * 255.0).astype(np.uint8)
        tifffile.imwrite('%s_defect.tiff' % out_prefix, dm[::preview_step, ::preview_step])
        frac = 100.0 * float((defect > 0.15).mean())
        print("  defect map: %s_defect.tiff  (%.2f%% of pixels flagged as defect >15%% dip)" % (out_prefix, frac))
    return 0
