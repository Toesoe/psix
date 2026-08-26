"""Background scan/processing runner + SSE event bus.

One job (scan or preview-processing) runs at a time.  start_scan() spawns a
worker that writes the raw capture into the roll's folder and publishes
lifecycle/progress events to subscribed SSE clients.

All hardware work goes through the injected ScannerDriver:
  * mock mode     writes a small synthetic .bin + placeholder SVG frames.
  * hardware mode driver.scan() captures the roll, then the capture is inverted
                  into per-frame preview JPEGs (see processing.generate_previews).
"""

import os
import queue
import threading
import time
from pathlib import Path

from . import processing


class ScanBusyError(Exception):
    pass


MOCK_FRAMES_PER_SCAN = 6


def _frame_svg(roll_label, frame_no, hue):
    """A clean film-frame placeholder for mock mode (no image deps)."""
    holes = "".join(
        '<rect x="%d" y="6" width="22" height="14" rx="2" fill="#0b0c0e"/>'
        '<rect x="%d" y="380" width="22" height="14" rx="2" fill="#0b0c0e"/>' % (x, x)
        for x in range(24, 580, 40)
    )
    return (
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 600 400">'
        '<defs><linearGradient id="g" x1="0" y1="0" x2="1" y2="1">'
        '<stop offset="0" stop-color="hsl(%d,45%%,34%%)"/>'
        '<stop offset="1" stop-color="hsl(%d,52%%,16%%)"/></linearGradient></defs>'
        '<rect width="600" height="400" fill="#0b0c0e"/>'
        '<rect x="14" y="26" width="572" height="348" rx="4" fill="url(#g)"/>'
        '%s'
        '<text x="300" y="196" text-anchor="middle" font-family="monospace" '
        'font-size="34" fill="#e7ebf0">Roll %s</text>'
        '<text x="300" y="238" text-anchor="middle" font-family="monospace" '
        'font-size="22" fill="#f0a429">Frame %02d</text>'
        '</svg>'
    ) % (hue, (hue + 40) % 360, holes, roll_label, frame_no)


class ScanRunner:
    def __init__(self, config, rolls, driver):
        self.config = config
        self.rolls = rolls
        self.driver = driver
        self.monitor = None                          # bound after the monitor exists
        self._lock = threading.Lock()
        self._subscribers = []
        self._thread = None
        self._stop = threading.Event()
        self._state = {
            "busy": False, "roll_id": None, "roll_label": None, "phase": "idle",
            "bytes": 0, "message": "", "error": None, "mock": True,
        }

    def bind_monitor(self, monitor):
        """Share the DeviceMonitor so scans can pause its auto-prep."""
        self.monitor = monitor

    # ---- SSE pub/sub -----------------------------------------------------

    def subscribe(self):
        q = queue.Queue()
        with self._lock:
            self._subscribers.append(q)
        return q

    def unsubscribe(self, q):
        with self._lock:
            if q in self._subscribers:
                self._subscribers.remove(q)

    def _publish(self, event, data):
        with self._lock:
            subs = list(self._subscribers)
        for q in subs:
            q.put((event, data))

    def state(self):
        with self._lock:
            return dict(self._state)

    def _set_state(self, **changes):
        with self._lock:
            self._state.update(changes)
            snap = dict(self._state)
        self._publish("state", snap)

    # ---- control ---------------------------------------------------------

    def start_scan(self, roll_id, ir=False):
        with self._lock:
            if self._state["busy"]:
                raise ScanBusyError("a scan is already running")
        roll = self.rolls.get(roll_id)
        if roll is None:
            raise KeyError("roll not found")

        mock = bool(self.config.get("mock_mode"))
        out_path = self.rolls.new_scan_path(roll, ir=ir)
        self._stop.clear()
        self._set_state(busy=True, roll_id=roll_id, roll_label=roll["label"],
                        phase="starting", bytes=0, message="preparing", error=None, mock=mock)
        self._thread = threading.Thread(
            target=self._run, args=(roll, out_path, ir, mock), daemon=True)
        self._thread.start()
        return out_path

    def stop_scan(self):
        self._stop.set()
        self._set_state(phase="stopping", message="stop requested")

    def process_roll(self, roll_id):
        """(Re)generate preview JPEGs for every scan in a roll that lacks them.
        Runs in the background; shares the busy state with scanning."""
        with self._lock:
            if self._state["busy"]:
                raise ScanBusyError("a scan or processing job is already running")
        roll = self.rolls.get(roll_id)
        if roll is None:
            raise KeyError("roll not found")
        self._set_state(busy=True, roll_id=roll_id, roll_label=roll["label"],
                        phase="processing", bytes=0, message="processing…",
                        error=None, mock=bool(self.config.get("mock_mode")))
        threading.Thread(target=self._process, args=(roll,), daemon=True).start()

    def apply_ice(self, roll_id, ir_thresh, ir_kernel, ir_min_size):
        """Commit a chosen ICE detection (threshold + kernel + min-size): persist it on
        the roll and RE-DEVELOP every IR scan's negatives at those params (so the cached
        negs are de-dusted exactly as previewed), then re-grade previews. Background job."""
        roll = self.rolls.get(roll_id)
        if roll is None:
            raise KeyError("roll not found")
        if not self.rolls.has_ir(roll):
            raise ValueError("roll has no IR (4-channel) scan — ICE does not apply")
        with self._lock:
            if self._state["busy"]:
                raise ScanBusyError("a scan or processing job is already running")
        self.rolls.set_ice_params(roll_id, {"ir_thresh": float(ir_thresh), "ir_kernel": int(ir_kernel),
                                            "ir_min_size": int(ir_min_size)})
        self._set_state(busy=True, roll_id=roll_id, roll_label=roll["label"],
                        phase="processing", bytes=0, message="applying ICE…",
                        error=None, mock=bool(self.config.get("mock_mode")))
        threading.Thread(target=self._reprocess_ice,
                         args=(roll, float(ir_thresh), int(ir_kernel), int(ir_min_size)), daemon=True).start()

    def _reprocess_ice(self, roll, ir_thresh, ir_kernel, ir_min_size):
        try:
            import glob
            done = 0
            fg = roll.get("frame_grades", {})
            nd = self.rolls.neg_dir(roll)
            pdir = self.rolls.preview_dir(roll)
            for scan in roll.get("scans", []):
                if not scan.get("ir"):
                    continue                                  # ICE only on 4-channel scans
                out_path = str(self.rolls.roll_dir(roll) / scan["filename"])
                if not os.path.exists(out_path):
                    continue
                # stale downsized live-grade caches must go so re-grading shows the new negs
                base = os.path.splitext(scan["filename"])[0]
                for lv in glob.glob(str(nd / ("%s_f*_neg_live.tiff" % base))):
                    try:
                        os.remove(lv)
                    except OSError:
                        pass
                frames = self._develop(roll, out_path, ir_thresh=ir_thresh, ir_kernel=ir_kernel,
                                       ir_min_size=ir_min_size)
                # re-developing re-grades previews at DEFAULT; re-apply each frame's stored grade
                # so committed colour survives the de-dust re-develop.
                for i, jpg in enumerate(frames):
                    grade = fg.get(jpg)
                    if not grade:
                        continue
                    neg = nd / ("%s_f%02d_neg.tiff" % (base, i))
                    if neg.exists():
                        processing.render_committed_frame(self.driver, str(neg), pdir, jpg, grade)
                self.rolls.set_scan_frames(roll["id"], scan["filename"], frames)
                done += 1
            self._set_state(busy=False, phase="done",
                            message="ICE applied (%d scan(s))" % done, error=None)
            self._publish("done", {"roll_id": roll["id"], "reprocessed": done})
        except Exception as exc:                              # noqa: BLE001
            self._set_state(busy=False, phase="failed", message="ICE failed", error=str(exc))
            self._publish("error", {"roll_id": roll["id"], "error": str(exc)})

    def apply_frame_grade(self, roll_id, idx, grade):
        """Persist ONE frame's grade and re-render just that frame full-quality."""
        roll = self.rolls.get(roll_id)
        if roll is None:
            raise KeyError("roll not found")
        frames = self.rolls.frames(roll)
        negs = self.rolls.frame_negs(roll)
        if idx < 0 or idx >= len(frames) or idx >= len(negs):
            raise IndexError("frame out of range")
        jpg = frames[idx]["filename"]
        neg = negs[idx]
        self.rolls.set_frame_grade(roll_id, jpg, grade)
        if self.config.get("mock_mode") or not neg.exists():
            self._publish("done", {"roll_id": roll_id, "frame": idx})   # nothing to re-render
            return
        with self._lock:
            if self._state["busy"]:
                raise ScanBusyError("a scan or processing job is already running")
        self._set_state(busy=True, roll_id=roll_id, roll_label=roll["label"],
                        phase="processing", bytes=0,
                        message="applying grade to frame %02d…" % (idx + 1),
                        error=None, mock=False)
        threading.Thread(target=self._apply_frame,
                         args=(roll, idx, jpg, str(neg), grade), daemon=True).start()

    def _apply_frame(self, roll, idx, jpg, neg, grade):
        try:
            processing.render_committed_frame(
                self.driver, neg, self.rolls.preview_dir(roll), jpg, grade)
            self._set_state(busy=False, phase="done",
                            message="grade applied (frame %02d)" % (idx + 1), error=None)
            self._publish("done", {"roll_id": roll["id"], "frame": idx})
        except Exception as exc:                          # noqa: BLE001
            self._set_state(busy=False, phase="failed", message="grade failed", error=str(exc))
            self._publish("error", {"roll_id": roll["id"], "error": str(exc)})

    # ---- workers ---------------------------------------------------------

    def _run(self, roll, out_path, ir, mock):
        try:
            self._set_state(phase="scanning", message="scanning")
            if mock:
                nbytes, npkts = self._mock_scan(out_path)
            else:
                # Pause the monitor's auto-prep; the driver serializes the device.
                if self.monitor:
                    self.monitor.set_scanning(True)
                try:
                    nbytes, npkts = self.driver.scan(
                        out_path, ir=ir, on_progress=self._on_event)
                finally:
                    if self.monitor:
                        self.monitor.set_scanning(False)

            record = {
                "filename": os.path.basename(out_path),
                "bytes": nbytes, "packets": npkts, "ir": ir, "mock": mock,
                "created_at": int(time.time()),
            }
            sidecar = os.path.splitext(out_path)[0] + "_flatref.npz"
            if os.path.exists(sidecar):
                record["flatref"] = os.path.basename(sidecar)

            if mock:
                record["frames"] = self._make_mock_previews(roll, out_path)
            else:
                self._set_state(phase="processing", message="generating previews…")
                try:
                    record["frames"] = self._develop(roll, out_path)
                except Exception as exc:              # noqa: BLE001 — keep the scan even if previews fail
                    record["frames"] = []
                    self._publish("error", {"roll_id": roll["id"],
                                            "error": "preview generation failed: %s" % exc})

            self.rolls.add_scan(roll["id"], record)
            self._set_state(busy=False, phase="done", bytes=nbytes,
                            message="captured %d bytes" % nbytes, error=None)
            self._publish("done", {"roll_id": roll["id"], **record})
        except Exception as exc:                      # noqa: BLE001 — report to the UI
            self._set_state(busy=False, phase="failed", message="failed", error=str(exc))
            self._publish("error", {"roll_id": roll["id"], "error": str(exc)})

    def _process(self, roll):
        try:
            done = 0
            for scan in roll.get("scans", []):
                if scan.get("frames"):
                    continue
                out_path = str(self.rolls.roll_dir(roll) / scan["filename"])
                if not os.path.exists(out_path):
                    continue
                frames = self._develop(roll, out_path)
                self.rolls.set_scan_frames(roll["id"], scan["filename"], frames)
                done += 1
            self._set_state(busy=False, phase="done",
                            message="previews ready (%d scan(s))" % done, error=None)
            self._publish("done", {"roll_id": roll["id"], "processed": done})
        except Exception as exc:                      # noqa: BLE001
            self._set_state(busy=False, phase="failed", message="processing failed", error=str(exc))
            self._publish("error", {"roll_id": roll["id"], "error": str(exc)})

    def _develop(self, roll, out_path, ir_thresh=None, ir_kernel=None, ir_min_size=None):
        """Develop a capture into cached negatives + graded preview JPEGs.
        ICE detection params come from the explicit args, else the roll's stored
        ICE settings, else the develop defaults."""
        sidecar = os.path.splitext(out_path)[0] + "_flatref.npz"
        flatref = sidecar if os.path.exists(sidecar) else None
        base = os.path.splitext(os.path.basename(out_path))[0]
        ice = roll.get("ice") or {}
        t = ir_thresh if ir_thresh is not None else ice.get("ir_thresh")
        k = ir_kernel if ir_kernel is not None else ice.get("ir_kernel")
        m = ir_min_size if ir_min_size is not None else ice.get("ir_min_size")
        names, _ = processing.develop_previews(
            self.driver, out_path, flatref, self.rolls.neg_dir(roll),
            self.rolls.preview_dir(roll), base, grade=None, ir=roll.get("ir"),
            on_event=self._on_event, ir_thresh=t, ir_kernel=k, ir_min_size=m)
        return names

    def _on_event(self, event, data):
        """Progress callback for the driver (transport / invert).

          ('phase',    {'phase', 'message', …})       -> phase + message
          ('progress', {'bytes', 'mb', 'mean', …})    -> live byte count
          ('bytes',    int)                           -> final byte count
        """
        if event == "bytes":
            self._set_state(bytes=int(data))
        elif event == "progress":
            self._set_state(bytes=int((data or {}).get("bytes", 0)))
        elif event == "phase":
            data = data or {}
            self._set_state(phase=data.get("phase", "scanning"),
                            message=data.get("message", ""))
        else:
            self._publish(event, data)

    def _make_mock_previews(self, roll, out_path):
        """Write placeholder film-frame SVGs into the roll's previews/ dir."""
        pdir = self.rolls.preview_dir(roll)
        pdir.mkdir(parents=True, exist_ok=True)
        base = os.path.splitext(os.path.basename(out_path))[0]
        start = sum(len(s.get("frames", [])) for s in roll.get("scans", []))
        names = []
        for i in range(MOCK_FRAMES_PER_SCAN):
            fno = start + i + 1
            name = "%s_f%02d.svg" % (base, i)
            (pdir / name).write_text(_frame_svg(roll["label"], fno, (fno * 37) % 360))
            names.append(name)
        return names

    def _mock_scan(self, out_path):
        """Write a small synthetic .bin with simulated byte progress."""
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        total = 8 * 1024 * 1024
        chunk = 256 * 1024
        written = 0
        with open(out_path, "wb") as f:
            while written < total:
                if self._stop.is_set():
                    self._set_state(message="stopped")
                    break
                f.write(b"\x00" * min(chunk, total - written))
                written += chunk
                self._set_state(bytes=written)
                time.sleep(0.05)
        return written, written // 0x5000
