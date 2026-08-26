"""Develop + grade scanned frames into web preview images.

Two stages, so colour grading is cheap to redo:
  * develop  — driver.develop() detects frames + writes per-frame raw-negative
               TIFFs (cached under the roll's negs/). Expensive; done once.
  * grade    — driver.finish_frame() re-renders each cached negative to an 8-bit
               positive with the roll's grade settings. Cheap; redo on demand.

The user-facing grade (the slider values) is mapped here to pakon_finish's
parameters.  Previews are JPEGs sized for the carousel; the cached negatives +
raw .bin remain the source for a full-quality export later.
"""

import io
import os
from pathlib import Path

PREVIEW_MAX = 1600          # longest-side px for the committed preview JPEGs
JPEG_QUALITY = 85
LIVE_MAX = 800              # longest-side px for the interactive (live-slider) preview
LIVE_QUALITY = 80

# User-facing grade — the minilab/printer-points model (what the sliders edit).
# Density + CMY balance are "points" (~ -50..+50); 0 = the scan's auto-neutral.
DEFAULT_GRADE = {
    # master (global)
    "density": 0.0,         # + = darker (per-frame white point)
    "cyan": 0.0,            # + = more cyan / − = more red    (R-channel density)
    "magenta": 0.0,         # + = more magenta / − = more green (G-channel density)
    "yellow": 0.0,          # + = more yellow / − = more blue  (B-channel density)
    "contrast": 0.6,        # tanh S-curve strength
    "saturation": 1.0,      # 1.0 = neutral; <1 muted, >1 punch
    "decouple": 0.0,        # dye-crosstalk unmix 0..1 (colour separation / purer primaries)
    "rotate": 0,            # display rotation, degrees clockwise: 0 / 90 / 180 / 270
    # 3-way region balance (shadows / mids / highlights), same point units
    "sh_density": 0.0, "sh_cyan": 0.0, "sh_magenta": 0.0, "sh_yellow": 0.0,
    "mid_density": 0.0, "mid_cyan": 0.0, "mid_magenta": 0.0, "mid_yellow": 0.0,
    "hi_density": 0.0, "hi_cyan": 0.0, "hi_magenta": 0.0, "hi_yellow": 0.0,
}

# 8-band per-hue HSL (Lightroom Color-Mixer): name -> hue-wheel centre (degrees).
HSL_BANDS = [("red", 0), ("orange", 30), ("yellow", 60), ("green", 120),
             ("aqua", 180), ("blue", 240), ("purple", 290), ("magenta", 330)]
for _b, _c in HSL_BANDS:                    # hue (deg), sat (frac), lum (frac); 0 = neutral
    DEFAULT_GRADE["%s_hue" % _b] = 0.0
    DEFAULT_GRADE["%s_sat" % _b] = 0.0
    DEFAULT_GRADE["%s_lum" % _b] = 0.0

_DENSITY_PER_POINT = 0.15 / 50.0   # density units per CMY/density "point"

# ICE (dust/scratch detection) — the black-hat knobs the UI exposes.
ICE_DEFAULTS = {"ir_thresh": 0.04, "ir_kernel": 41, "ir_min_size": 3}


def grade_with_defaults(grade):
    g = dict(DEFAULT_GRADE)
    g.update(grade or {})
    return g


def _finish_kwargs(grade):
    """Map the user-facing grade to pakon_finish.finish() keyword overrides."""
    g = grade_with_defaults(grade)
    density = float(g["density"])
    sat = float(g["saturation"])
    k = _DENSITY_PER_POINT
    # CMY points -> per-channel density trim (printer points, in the print domain).
    # +cyan = less red (−R density); +magenta = less green; +yellow = less blue.
    balance_trim = (-k * float(g["cyan"]), -k * float(g["magenta"]), -k * float(g["yellow"]))
    kw = {
        # Density darkening, two parts:
        #  • headroom (per-frame white point) — the gentle part; saturates and is
        #    partly re-anchored by the grade stage, so it can't darken much alone.
        #  • a grade-stage midtone gamma that ONLY engages past ±50, so 0..±50
        #    render exactly as before, but ±50..±100 actually darken/brighten.
        "headroom": max(0.60, min(1.90, 1.15 + (density / 50.0) * 0.25)),
        "density_gamma": 1.0 + (1.0 if density >= 0 else -1.0)
                         * max(0.0, (abs(density) - 50.0) / 50.0) * 0.5,
        "contrast": float(g["contrast"]),
        "balance_trim": balance_trim,
        "decouple": float(g["decouple"]),       # 0 = off; OEM matrix blended by strength
        "hsl": [(center, float(g["%s_hue" % n]), float(g["%s_sat" % n]), float(g["%s_lum" % n]))
                for n, center in HSL_BANDS],
    }
    if abs(sat - 1.0) > 0.02:                       # 1.0 = leave the tone-saturation stage off
        kw["saturation"] = True
        kw["sat_shadow"] = max(0.2, min(1.2, 0.62 * sat))
        kw["sat_mid"] = max(0.2, 1.45 * sat)
        kw["sat_high"] = max(0.2, 0.2 + 0.8 * sat)

    # 3-way: per-region per-channel density trim (CMY + the region's own density).
    def _region_trim(prefix):
        d = float(g.get(prefix + "_density", 0.0))
        c = float(g.get(prefix + "_cyan", 0.0))
        m = float(g.get(prefix + "_magenta", 0.0))
        y = float(g.get(prefix + "_yellow", 0.0))
        u = -k * d                                  # +density -> darker (subtract from D)
        return (u - k * c, u - k * m, u - k * y)
    kw["region_trim_sh"] = _region_trim("sh")
    kw["region_trim_mid"] = _region_trim("mid")
    kw["region_trim_hi"] = _region_trim("hi")
    return kw


def _apply_rotate(rgb8, grade):
    """Apply the per-frame display rotation (0/90/180/270° clockwise) to a rendered
    RGB array. One helper for every output path (preview, live, ICE, export) so they
    always agree on orientation. Portrait frames come off the scanner sideways."""
    import numpy as np

    deg = int(float((grade or {}).get("rotate", 0))) % 360
    if deg:
        rgb8 = np.ascontiguousarray(np.rot90(rgb8, k=-(deg // 90)))   # k<0 = clockwise
    return rgb8


def _save_jpeg(rgb8, path):
    from PIL import Image

    im = Image.fromarray(rgb8, "RGB")
    w, h = im.size
    scale = PREVIEW_MAX / float(max(w, h))
    if scale < 1.0:
        im = im.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.LANCZOS)
    im.save(str(path), "JPEG", quality=JPEG_QUALITY)


def _grade_negs(driver, negs, previews_dir, base, grade, on_event):
    def emit(event, data=None):
        if on_event:
            try:
                on_event(event, data)
            except Exception:
                pass

    previews_dir = Path(previews_dir)
    previews_dir.mkdir(parents=True, exist_ok=True)
    kw = _finish_kwargs(grade)
    names = []
    for i, neg in enumerate(negs):
        emit("phase", {"phase": "processing", "message": "grading %d/%d…" % (i + 1, len(negs))})
        rgb8 = _apply_rotate(driver.finish_frame(neg, **kw), grade)
        jpg = "%s_f%02d.jpg" % (base, i)
        _save_jpeg(rgb8, previews_dir / jpg)
        names.append(jpg)
    return names


def develop_previews(driver, bin_path, flatref_path, negs_dir, previews_dir, base,
                     grade=None, on_event=None, ir_thresh=None, ir_kernel=None, ir_min_size=None,
                     ir=None):
    """Develop the raw capture into cached negatives, then grade → preview JPEGs.
    Returns (frame_jpeg_names, neg_paths). ir_* override ICE detection."""
    def emit(event, data=None):
        if on_event:
            try:
                on_event(event, data)
            except Exception:
                pass

    negs_dir = Path(negs_dir)
    negs_dir.mkdir(parents=True, exist_ok=True)
    emit("phase", {"phase": "processing", "message": "developing (detecting frames)…"})
    negs = driver.develop(bin_path, flatref_path, str(negs_dir / base), ir=ir,
                          ir_thresh=ir_thresh, ir_kernel=ir_kernel, ir_min_size=ir_min_size)
    names = _grade_negs(driver, negs, previews_dir, base, grade, on_event)
    return names, negs


def render_committed_frame(driver, neg_path, previews_dir, jpg_name, grade):
    """Full-quality grade of ONE frame (NR + sharpen on) → committed preview JPEG."""
    rgb8 = _apply_rotate(driver.finish_frame(str(neg_path), **_finish_kwargs(grade)), grade)
    _save_jpeg(rgb8, Path(previews_dir) / jpg_name)


def export_frame(driver, neg_path, out_path, grade, quality=95):
    """Full-RESOLUTION export: grade ONE frame (full quality, NR+sharpen on) and save the
    positive at native resolution (no preview downscale), 4:4:4 JPEG. Returns the path."""
    from PIL import Image

    rgb8 = _apply_rotate(driver.finish_frame(str(neg_path), **_finish_kwargs(grade)), grade)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgb8, "RGB").save(str(out_path), "JPEG", quality=quality, subsampling=0)
    return str(out_path)


def ensure_live_neg(neg_path):
    """Lazily cache a downsized copy of a raw-negative for fast interactive
    grading (NR/sharpen on ~1024px instead of full-res). Returns its path."""
    p = Path(neg_path)
    small = p.with_name(p.stem + "_live.tiff")
    if small.exists():
        return str(small)
    import cv2
    import tifffile

    arr = tifffile.imread(str(p))
    h, w = arr.shape[:2]
    scale = LIVE_MAX / float(max(h, w))
    if scale < 1.0:
        arr = cv2.resize(arr, (max(1, int(w * scale)), max(1, int(h * scale))),
                         interpolation=cv2.INTER_AREA)
    tifffile.imwrite(str(small), arr, photometric="rgb")
    return str(small)


def ice_view_jpeg(driver, neg_path, ir_plane_path, flatref_path, ice_npz, grade,
                  ice_on, show_mask, ir_thresh, ir_kernel, ir_min_size=3):
    """ICE interface render: the frame's graded RGB positive at the current grade, with ICE
    on/off (de-dusted vs original) and an optional live red detection overlay → JPEG bytes."""
    from PIL import Image

    kw = _finish_kwargs(grade)
    kw.update(chroma_nr=0.0, luma_nr=0.0, sharpen=0.0)        # live path: skip slow detail stages
    rgb8 = driver.ice_view(neg_path, ir_plane_path, flatref_path, ice_npz,
                           ice_on=ice_on, show_mask=show_mask, max_side=LIVE_MAX,
                           ir_thresh=ir_thresh, ir_kernel=ir_kernel, ir_min_size=ir_min_size, **kw)
    rgb8 = _apply_rotate(rgb8, grade)
    im = Image.fromarray(rgb8, "RGB")
    w, h = im.size
    scale = LIVE_MAX / float(max(w, h))
    if scale < 1.0:
        im = im.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=LIVE_QUALITY)
    return buf.getvalue()


def ice_preview_jpeg(driver, ir_plane_path, flatref_path, ir_thresh, ir_kernel, ir_min_size=3):
    """Render the ICE dust-preview (IR channel + red detected-dust overlay) for one
    frame at the given detection params → downscaled JPEG bytes (the live path)."""
    from PIL import Image

    rgb8 = driver.ice_overlay(ir_plane_path, flatref_path,
                              ir_thresh=ir_thresh, ir_kernel=ir_kernel, ir_min_size=ir_min_size)
    im = Image.fromarray(rgb8, "RGB")
    w, h = im.size
    scale = LIVE_MAX / float(max(w, h))
    if scale < 1.0:
        im = im.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=LIVE_QUALITY)
    return buf.getvalue()


# RAM cache of the slider-INDEPENDENT analysis (film-base + scene-balance offset) for the live path.
# Kept tiny (one entry: the active frame + decouple), so dragging non-decouple sliders reuses it and only
# the cheap print stage re-runs. Keyed by (live-neg path, mtime, decouple) so a re-develop invalidates it.
_ANALYSIS_CACHE = {}


def grade_preview_jpeg(driver, neg_path, grade):
    """Grade ONE frame's downsized negative → JPEG bytes (the live-slider path)."""
    import numpy as np
    from PIL import Image

    small = ensure_live_neg(neg_path)
    kw = _finish_kwargs(grade)
    # Live path: skip the detail-only stages (chroma/luma NR + sharpen) — they're
    # the slow part and don't affect the colour the user is judging. The committed
    # Apply (full-res) keeps them.
    kw.update(chroma_nr=0.0, luma_nr=0.0, sharpen=0.0)
    # Reuse the cached analysis (film-base + scene offset) unless the frame, its data, or decouple changed.
    decouple = kw.get("decouple", 0.0)
    try:
        key = (small, os.path.getmtime(small), round(float(decouple), 4))
    except OSError:
        key = None
    cached = _ANALYSIS_CACHE.get(key) if key is not None else None
    if cached is None:
        base, mo, _ = driver.finish_frame(small, return_analysis=True, decouple=decouple)
        cached = (np.asarray(base, np.float32), np.asarray(mo, np.float32))
        if key is not None:
            _ANALYSIS_CACHE.clear()                     # keep only the active frame+decouple
            _ANALYSIS_CACHE[key] = cached
    base, mo = cached
    rgb8 = _apply_rotate(driver.finish_frame(small, cached_base=base, scene_offset=mo, **kw), grade)
    buf = io.BytesIO()
    Image.fromarray(rgb8, "RGB").save(buf, "JPEG", quality=LIVE_QUALITY)
    return buf.getvalue()
