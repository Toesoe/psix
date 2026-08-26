"""The single boundary between psix and the F135 driver.

Every hardware/driver operation goes through ScannerDriver; no other part of
the app imports a ``pakon_*`` module.  Driver modules are imported lazily inside
methods so importing this module has no heavy side effects and mock-mode never
needs the hardware libraries.

The driver owns one lock: device operations (identify / scan / load) are
mutually exclusive.  ``identify(blocking=False)`` returns ``None`` when the
device is busy (e.g. a scan is running), so callers back off cleanly.
"""

import glob
import threading

VID = 0x0F05
LOADED_PID = 0xF135
UNLOADED_PID = 0xF235

# Film exposure defaults (the calibrated, HW-validated values used by the CLI).
_FILM_BOOST = (1.6, 3.6, 5.3)
_INVERT_GAMMA = 2.2
_INVERT_CONTRAST = 0.6


class ScannerError(Exception):
    pass


def _identity(health):
    """Map PakonScanner.health() to the flat identity dict the UI renders."""
    model = health.get("model") or "F135+"
    return {
        "product": "Pakon %s" % model,
        "model": model,
        "firmware": health.get("usb_fw"),
        "usb_fw": health.get("usb_fw"),
        "serial_picl": health.get("serial_picl"),
        "serial_sub": health.get("serial_sub"),
        "host_status": health.get("host_status"),
        "ready": bool(health.get("ready")),
        "temp_adc": health.get("temp_adc"),
        "eeprom_bytes": health.get("eeprom_bytes"),
        "state": health.get("state"),
        "comm_ok": bool(health.get("comm_ok")),
    }


class ScannerDriver:
    LOADED_PID = LOADED_PID
    UNLOADED_PID = UNLOADED_PID

    def __init__(self):
        self._lock = threading.Lock()

    @property
    def busy(self):
        if self._lock.acquire(blocking=False):
            self._lock.release()
            return False
        return True

    # ---- presence (lock-free; enumeration only) --------------------------

    def present_pids(self):
        """Set of F135 (0f05) product IDs currently on the USB bus."""
        import usb1

        pids = set()
        try:
            with usb1.USBContext() as ctx:
                for dev in ctx.getDeviceList(skip_on_error=True):
                    try:
                        if dev.getVendorID() == VID:
                            pids.add(dev.getProductID())
                    except Exception:
                        pass
        except Exception:
            pass
        return pids

    # ---- exclusive device operations -------------------------------------

    def load_firmware(self, log=None):
        """Upload firmware to an unloaded (f235) scanner. Returns True on success."""
        from .pakon import load as pakon_load
        with self._lock:
            return bool(pakon_load.load_firmware(log=log))

    def move_film(self, seconds=6.0, reverse=False, rate=None):
        """Motor-only film move (no lamp, no scan): forward to advance/eject,
        reverse to rewind. Full device lifecycle (open → init → move → shutdown).
        Returns True if the move ran. Raises ScannerError if the device is busy
        or unavailable."""
        from .pakon import scanner as pakon_scanner
        from .pakon import scan as pakon_scan2
        if not self._lock.acquire(blocking=False):
            raise ScannerError("scanner is busy")
        try:
            sc = pakon_scanner.PakonScanner(verbose=False)
            sc.open()
            sc.initialize()
            try:
                if rate is None:
                    from .pakon import unitdata as pakon_unitdata
                    rate = (pakon_unitdata.transport_rate(getattr(sc.dev, 'unit_info', None))
                            or pakon_scan2.MOTOR_RATE_PLUS)
                return bool(pakon_scan2.move_film(sc.dev, seconds, rate, reverse))
            finally:
                try:
                    sc.shutdown()
                except Exception:
                    pass
        except Exception as exc:
            raise ScannerError(str(exc))
        finally:
            self._lock.release()

    def firmware_status(self):
        """Filesystem-only check of the firmware dir (no device access, no lock)."""
        from .pakon import load as pakon_load
        return pakon_load.firmware_status()

    def save_firmware(self, filename, data):
        """Save a user-supplied firmware file (e.g. pakon8.hex) into the firmware dir.
        Validates the name/extension (no path traversal); returns the updated status.
        Raises ValueError on a rejected file."""
        import os
        from .pakon import load as pakon_load
        name = os.path.basename(filename or "").strip()
        if not name or name.startswith("."):
            raise ValueError("invalid filename")
        if not name.lower().endswith((".hex", ".ihex")):
            raise ValueError("not a firmware file (.hex / .ihex expected)")
        if not data:
            raise ValueError("empty file")
        fwdir = pakon_load.FWDIR
        os.makedirs(fwdir, exist_ok=True)
        with open(os.path.join(fwdir, name), "wb") as f:
            f.write(data)
        return pakon_load.firmware_status()

    def identify(self, blocking=False):
        """Initialize the scanner and return its identity dict (open → init →
        health → shutdown). Returns None if the device is busy."""
        from .pakon import scanner as pakon_scanner
        if not self._lock.acquire(blocking=blocking):
            return None
        try:
            sc = pakon_scanner.PakonScanner(verbose=False)
            try:
                sc.open()
                sc.initialize()
                return _identity(sc.health())
            finally:
                try:
                    sc.shutdown()
                except Exception:
                    pass
        finally:
            self._lock.release()

    def scan(self, out_path, *, ir=False, seconds=300.0, on_progress=None):
        """Run a full roll scan to out_path. Returns (bytes, packets).

        Owns the whole device lifecycle: open → initialize → transport →
        teardown. The transport itself is the HW-validated shared entry point.
        """
        from .pakon import scanner as pakon_scanner
        from .pakon import scan as pakon_scan2
        with self._lock:
            sc = pakon_scanner.PakonScanner(verbose=True)
            sc.open()
            sc.initialize()                 # HW-validated load/init; guarantees HOST ready
            dev = sc.dev
            try:
                return pakon_scan2.run_transport_scan(
                    dev, out_path, ir=ir, seconds=seconds, on_progress=on_progress,
                    exp_servo=True, film_boost=_FILM_BOOST, film_duty=None,
                    fixed_duty=None, dx_eof=True, dx_gap=2.0, eject_seconds=2.0,
                )
            finally:
                self._teardown(dev)
                try:
                    sc.shutdown()
                except Exception:
                    pass

    def develop(self, bin_path, flatref_path, out_prefix, *, ir_thresh=None, ir_kernel=None,
                ir_min_size=None, ir=None):
        """Detect frames and write per-frame raw-negative TIFFs (the cached
        'digital negatives' that grading re-renders). Returns their paths.
        This is the expensive step (flat-field + frame detect + IR-ICE); run it once.
        `ir_thresh`/`ir_kernel`/`ir_min_size` override the ICE detection params (else
        process_bin defaults) — re-developing with new values commits a chosen dust-removal."""
        from .pakon import invert as pakon_invert
        ice_kw = {}
        if ir_thresh is not None:
            ice_kw["ir_thresh"] = float(ir_thresh)
        if ir_kernel is not None:
            ice_kw["ir_kernel"] = int(ir_kernel)
        if ir_min_size is not None:
            ice_kw["ir_min_size"] = int(ir_min_size)
        pakon_invert.process_bin(
            str(bin_path),
            str(flatref_path) if flatref_path else None,
            str(out_prefix),
            _INVERT_GAMMA, _INVERT_CONTRAST, None, neg_only=True, ir=ir, **ice_kw,
        )
        return sorted(glob.glob("%s_f[0-9][0-9]_neg.tiff" % out_prefix))

    def ice_overlay(self, ir_plane_tiff, flatref_path, *, ir_thresh, ir_kernel, ir_min_size=3):
        """Build the ICE dust-preview: the frame's IR plane rendered grayscale with
        DETECTED dust/scratches overlaid in red, at the given black-hat detection
        params. Returns an 8-bit RGB array. Read-only (no device) — the detection is
        recomputed live so the sliders can explore it. `ir_thresh` = black-hat depth
        (bh/255; lower = more sensitive); `ir_kernel` = max defect width (px);
        `ir_min_size` = discard blobs smaller than this (px)."""
        import numpy as np
        import tifffile

        from .pakon import invert as pakon_invert
        ir_o = tifffile.imread(str(ir_plane_tiff)).astype(np.float32)   # (2000, frame_lines), oriented cw
        z = np.load(str(flatref_path))
        if "white_ir" not in z:
            raise ScannerError("flatref has no white_ir (not a 4-channel/IR scan)")
        white_ir = z["white_ir"].astype(np.float32)
        ir_raw = np.rot90(ir_o, 1)                                     # back to (frame_lines, 2000) for detect
        # PRNU flat-field for DISPLAY too (else the clear film shows per-column stripes/vignette);
        # same correction ir_defect_mask uses internally for detection.
        ff_raw = ir_raw / np.maximum(white_ir[None, :], 1.0) * float(np.median(white_ir))
        mask = pakon_invert.ir_defect_mask(ir_raw, white_ir, float(ir_thresh), int(ir_kernel), int(ir_min_size))
        ff_o = np.rot90(ff_raw, 3)                                     # re-orient flat IR to match ir_o
        mask_o = np.rot90(mask, 3)
        base = max(float(np.percentile(ff_o, 95)), 1.0)               # clear-film level -> fixed white point
        g = np.clip(ff_o / base * 255.0, 0, 255).astype(np.uint8)
        rgb = np.repeat(g[:, :, None], 3, axis=2)
        rgb[mask_o] = (255, 0, 0)
        return rgb

    def finish_frame(self, neg, **grade):
        """Grade one cached raw-negative frame into an 8-bit positive RGB array.

        `neg` is a path to a raw-negative TIFF, or an in-memory neg ndarray.
        `grade` overrides any pakon_finish parameter (contrast, shoulder,
        black_lift, wb_trim, saturation, …); everything else uses the
        HW-validated OEM-parity defaults. Returns the pixels (no files written).
        """
        import numpy as np

        from .pakon import finish as pakon_finish
        src = neg if isinstance(neg, np.ndarray) else str(neg)
        params = dict(
            out_prefix="<array>" if isinstance(neg, np.ndarray) else str(neg),  # unused under return_array
            slr=2.4, display_gamma=2.2, base_pct=99.7, frame_lines=2500,
            decouple=0.0, balance=True, balance_pct=50.0,
            tone="oemshape", whole_frame=True, return_array=True,
        )
        params.update(grade)
        return pakon_finish.finish(src, **params)

    def ice_view(self, neg_tiff, ir_neg_tiff, flatref_path, ice_npz, *,
                 ice_on, show_mask, ir_thresh, ir_kernel, ir_min_size, max_side=0, **grade):
        """Render the frame's graded RGB positive for the ICE interface, in any toggle state:
          ice_on=True  -> the committed (de-dusted) neg;  ice_on=False -> the ORIGINAL dusty neg
            (reconstructed from the develop-time ICE sidecar = the pixels ICE replaced).
          show_mask=True -> overlay the live black-hat detection (current slider params) in red.
        `max_side` downsizes the neg before grading (the slow step) so the toggles feel live, like the
        colour live-preview. Returns an 8-bit RGB array."""
        import os

        import cv2
        import numpy as np
        import tifffile

        from .pakon import invert as pakon_invert
        neg = tifffile.imread(str(neg_tiff))
        base = neg
        if not ice_on and ice_npz and os.path.exists(str(ice_npz)):
            z = np.load(str(ice_npz))
            base = neg.copy()
            base[z["yx"][0], z["yx"][1]] = z["vals"]            # restore the dusty pixels ICE had replaced
        if max_side and max(base.shape[:2]) > max_side:         # grade a downsized neg -> fast (full-res ~4s)
            s = max_side / float(max(base.shape[:2]))
            base = cv2.resize(base, (max(1, int(base.shape[1] * s)), max(1, int(base.shape[0] * s))),
                              interpolation=cv2.INTER_AREA)
        rgb8 = self.finish_frame(base, **grade)
        if show_mask:
            ir_o = tifffile.imread(str(ir_neg_tiff)).astype(np.float32)
            z = np.load(str(flatref_path))
            if "white_ir" in z:
                white_ir = z["white_ir"].astype(np.float32)
                ir_raw = np.rot90(ir_o, 1)
                mask = pakon_invert.ir_defect_mask(ir_raw, white_ir, float(ir_thresh),
                                                   int(ir_kernel), int(ir_min_size))
                mask_o = np.rot90(mask, 3)
                if mask_o.shape != rgb8.shape[:2]:              # match the (downsized) grade output
                    mask_o = cv2.resize(mask_o.astype(np.uint8), (rgb8.shape[1], rgb8.shape[0]),
                                        interpolation=cv2.INTER_NEAREST) > 0
                rgb8 = rgb8.copy()
                rgb8[mask_o] = (255, 0, 0)
        return rgb8

    # ---- internals -------------------------------------------------------

    def _teardown(self, dev):
        """Stop the motor, return to idle (lamp off); named abort as fallback."""
        from .pakon import scan as pakon_scan2
        from .pakon import scanstart as pakon_scanstart
        try:
            dev.write_reg(dev.AD_SUB, 0xa5, b"\x00\x00")
            dev.write2(dev.AD_SUB, 0xa0)
            dev.write2(dev.AD_SUB, 0xa2)
        except Exception:
            pass
        try:
            if not pakon_scan2.reset_to_idle(dev):
                for pkt in pakon_scanstart.build_abort_steps():
                    try:
                        dev.send_raw(pkt, timeout=1500)
                    except Exception:
                        pass
        except Exception:
            pass
