"""Roll registry + per-roll output folders.

A "roll" is one loaded film roll.  Each gets an auto-incrementing Roll ID
(OEM-style: 0001, 0002, …) and its own subfolder under the configured output
directory.  The raw scan data (.bin + _flatref.npz sidecar) is written into
that subfolder.

Registry layout (under <output_dir>):
    rolls.json                  {"next_id": N, "rolls": [ {roll}, ... ]}
    0001_my-roll/
        roll.json               this roll's metadata + scan records
        scan_<ts>.bin           raw EP6 capture
        scan_<ts>_flatref.npz   flat-field sidecar (hardware scans)

The output directory is read from config on every call, so changing it on the
Settings page takes effect immediately for new rolls.
"""

import json
import os
import re
import threading
import time
from pathlib import Path


def _slug(name):
    s = re.sub(r"[^A-Za-z0-9._-]+", "-", (name or "").strip()).strip("-")
    return s[:48] or "roll"


def _now():
    return int(time.time())


def _stamp():
    return time.strftime("%Y%m%d_%H%M%S")


class RollStore:
    def __init__(self, config):
        self.config = config
        self._lock = threading.Lock()

    # ---- paths -----------------------------------------------------------

    def output_dir(self):
        return Path(self.config.get("output_dir"))

    def _registry_path(self):
        return self.output_dir() / "rolls.json"

    def roll_dir(self, roll):
        return self.output_dir() / roll["dir"]

    def preview_dir(self, roll):
        return self.roll_dir(roll) / "previews"

    def neg_dir(self, roll):
        """Cached per-frame raw-negative TIFFs (re-graded by the finishing step)."""
        return self.roll_dir(roll) / "negs"

    def export_dir(self, roll):
        """Full-resolution exported positives (user 'Export image')."""
        return self.roll_dir(roll) / "export"

    def preview_path(self, roll, filename):
        """Safe path to a preview file inside the roll (None if missing/escaping)."""
        base = self.preview_dir(roll).resolve()
        path = (base / filename).resolve()
        if base not in path.parents or not path.exists():
            return None
        return path

    def frames(self, roll):
        """Flatten every scan's frame previews into one ordered carousel list."""
        items = []
        fg = roll.get("frame_grades", {})
        for scan in roll.get("scans", []):
            for fname in scan.get("frames", []):
                items.append({
                    "filename": fname,
                    "scan": scan.get("filename"),
                    "index": len(items),
                    "label": "%02d" % (len(items) + 1),
                    "grade": fg.get(fname),
                })
        return items

    def frame_negs(self, roll):
        """Cached raw-negative paths in the SAME order as frames() — so a carousel
        frame index maps to its negative for live grading."""
        nd = self.neg_dir(roll)
        out = []
        for scan in roll.get("scans", []):
            base = os.path.splitext(scan["filename"])[0]
            for i in range(len(scan.get("frames", []))):
                out.append(nd / ("%s_f%02d_neg.tiff" % (base, i)))
        return out

    def frame_ir_sources(self, roll):
        """(ir_plane_tiff, flatref_path) per frame in frames() order — the inputs the
        ICE dust-preview needs (the archived per-frame IR plane + the scan's flatref,
        which carries white_ir). flatref is None for non-IR scans."""
        nd = self.neg_dir(roll)
        rd = self.roll_dir(roll)
        out = []
        for scan in roll.get("scans", []):
            base = os.path.splitext(scan["filename"])[0]
            fr = scan.get("flatref")
            flat = (rd / fr) if fr else None
            for i in range(len(scan.get("frames", []))):
                out.append((nd / ("%s_f%02d_neg_ir.tiff" % (base, i)), flat))
        return out

    def has_ir(self, roll):
        """True if any scan in the roll is a 4-channel/IR capture (ICE applies)."""
        return any(s.get("ir") for s in roll.get("scans", []))

    # ---- registry I/O ----------------------------------------------------

    def _read_registry(self):
        path = self._registry_path()
        if not path.exists():
            return {"next_id": 1, "rolls": []}
        try:
            with path.open("r") as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError):
            return {"next_id": 1, "rolls": []}
        data.setdefault("next_id", 1)
        data.setdefault("rolls", [])
        return data

    def _write_registry(self, data):
        path = self._registry_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        with tmp.open("w") as f:
            json.dump(data, f, indent=2)
        os.replace(tmp, path)

    def _write_roll_meta(self, roll):
        d = self.roll_dir(roll)
        d.mkdir(parents=True, exist_ok=True)
        tmp = d / "roll.json.tmp"
        with tmp.open("w") as f:
            json.dump(roll, f, indent=2)
        os.replace(tmp, d / "roll.json")

    # ---- public API ------------------------------------------------------

    def next_label(self):
        """The Roll ID label the next roll would get, e.g. '0002'."""
        return "%04d" % self._read_registry()["next_id"]

    def list_rolls(self):
        rolls = self._read_registry()["rolls"]
        return sorted(rolls, key=lambda r: r["id"], reverse=True)

    def get(self, roll_id):
        for r in self._read_registry()["rolls"]:
            if r["id"] == roll_id:
                return r
        return None

    def create_roll(self, name=None):
        """Allocate the next Roll ID, create its subfolder, return the roll."""
        with self._lock:
            reg = self._read_registry()
            rid = reg["next_id"]
            label = "%04d" % rid
            name = (name or "").strip() or label
            roll = {
                "id": rid,
                "label": label,
                "name": name,
                "dir": "%s_%s" % (label, _slug(name)),
                "created_at": _now(),
                "scans": [],
            }
            self.roll_dir(roll).mkdir(parents=True, exist_ok=True)
            self._write_roll_meta(roll)
            reg["next_id"] = rid + 1
            reg["rolls"].append(roll)
            self._write_registry(reg)
            return roll

    def new_scan_path(self, roll, ir=False):
        """Absolute .bin path for a fresh scan in this roll's folder."""
        tag = "scan_ir_%s.bin" if ir else "scan_%s.bin"
        return str(self.roll_dir(roll) / (tag % _stamp()))

    def set_frame_grade(self, roll_id, filename, grade):
        """Persist one frame's colour-grade settings (keyed by its preview file)."""
        with self._lock:
            reg = self._read_registry()
            for r in reg["rolls"]:
                if r["id"] == roll_id:
                    r.setdefault("frame_grades", {})[filename] = grade
                    self._write_registry(reg)
                    self._write_roll_meta(r)
                    return r
        return None

    def set_ice_params(self, roll_id, params):
        """Persist the roll's ICE detection params (ir_thresh, ir_kernel). Future
        develops (and re-develops) of this roll's IR scans use them."""
        with self._lock:
            reg = self._read_registry()
            for r in reg["rolls"]:
                if r["id"] == roll_id:
                    r["ice"] = dict(params)
                    self._write_registry(reg)
                    self._write_roll_meta(r)
                    return r
        return None

    def set_scan_frames(self, roll_id, scan_filename, frames):
        """Attach preview frame filenames to an already-recorded scan."""
        with self._lock:
            reg = self._read_registry()
            for r in reg["rolls"]:
                if r["id"] == roll_id:
                    for s in r.get("scans", []):
                        if s.get("filename") == scan_filename:
                            s["frames"] = frames
                            self._write_registry(reg)
                            self._write_roll_meta(r)
                            return r
        return None

    def add_scan(self, roll_id, record):
        """Append a completed-scan record to the roll (registry + roll.json)."""
        with self._lock:
            reg = self._read_registry()
            for r in reg["rolls"]:
                if r["id"] == roll_id:
                    r.setdefault("scans", []).append(record)
                    self._write_registry(reg)
                    self._write_roll_meta(r)
                    return r
        return None
