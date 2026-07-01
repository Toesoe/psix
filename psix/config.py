"""Persistent app configuration (JSON-backed).

Mirrors the piPalette Config pattern: a thread-safe, atomically-written JSON
file with a fixed DEFAULTS schema.  Unknown keys are ignored on load and never
written back.
"""

import json
import os
import sys
import threading
from pathlib import Path


def user_data_dir():
    """Platform user data dir for psix (config + rolls). Overridable via $PSIX_DATA_DIR."""
    env = os.environ.get("PSIX_DATA_DIR")
    if env:
        return Path(env).expanduser()
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support" / "psix"
    if os.name == "nt":
        return Path(os.environ.get("APPDATA", Path.home())) / "psix"
    return Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local" / "share")) / "psix"


# Default output dir (rolls + raw scans) — a user-writable location, settable on Settings.
DEFAULT_OUTPUT_DIR = str(user_data_dir() / "output")


DEFAULTS = {
    # HW-safety: default to mock so the whole UI is usable before anyone
    # points psix at the real scanner.  Flip to Hardware on the Settings page.
    "mock_mode": True,
    # Where roll subfolders + raw scans are written.  Defaults to <root>/output.
    "output_dir": DEFAULT_OUTPUT_DIR,
    # When the scanner is powered on (hardware mode), auto-load firmware +
    # initialize it so it becomes ready hands-free.  Off = detect only.
    "auto_prepare": True,
}


class Config:
    """JSON file persisted under data/config.json."""

    def __init__(self, path):
        self._path = Path(path)
        self._lock = threading.Lock()
        self._values = dict(DEFAULTS)
        self._load()

    def _load(self):
        if not self._path.exists():
            return
        try:
            with self._path.open("r") as f:
                loaded = json.load(f)
        except (json.JSONDecodeError, OSError):
            return
        for key, value in loaded.items():
            if key in DEFAULTS:
                self._values[key] = value

    def _save(self):
        self._path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._path.with_suffix(".tmp")
        with tmp.open("w") as f:
            json.dump(self._values, f, indent=2)
        os.replace(tmp, self._path)

    def get(self, key):
        return self._values.get(key, DEFAULTS.get(key))

    def all(self):
        return dict(self._values)

    def update(self, **changes):
        with self._lock:
            for key, value in changes.items():
                if key in DEFAULTS:
                    self._values[key] = value
            self._save()
        return self.all()
