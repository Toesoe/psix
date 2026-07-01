"""Scanner presence monitoring + UI status.

A background DeviceMonitor polls the bus (via ScannerDriver) and drives the
device lifecycle hands-free so the app can start without the scanner and pick
it up when powered on:

    absent --(f235)--> loading firmware --(f135)--> preparing (init) --> ready

The monitor never touches the hardware directly — every operation goes through
the injected ScannerDriver, which serializes device access with its own lock.
State changes are pushed to the UI via the `publish` callback ('device' event).
"""

import threading
import time

# Synthetic identity for mock mode — same shape as ScannerDriver identity.
MOCK_INFO = {
    "product": "Pakon F135+ (mock)",
    "firmware": "mock",
    "usb_fw": "0x00,0x00",
    "serial_picl": "MOCK-PICL-0001",
    "serial_sub": "MOCK-SUB-0001",
    "host_status": "0x80",
    "ready": True,
    "temp_adc": None,
    "eeprom_bytes": 434,
    "state": "ready",
    "comm_ok": True,
}


class DeviceMonitor:
    """Background USB presence monitor + auto-preparer (hardware mode)."""

    POLL = 1.5
    PREP_BACKOFF = 5.0          # seconds between init retries after a failure
    PREP_MAX_FAILS = 3          # after this many, stop auto-retrying until Connect/replug

    def __init__(self, config, driver, publish=None):
        self.config = config
        self.driver = driver
        self._publish = publish or (lambda event, data: None)
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._force = threading.Event()
        self._thread = None
        self._scanning = False
        self._load_failed = False
        self._prep_fails = 0
        self._prep_next = 0.0
        self._st = {
            "phase": "absent", "present": False, "pid": None, "connected": False,
            "mock_mode": False, "info": None, "error": None, "display": None,
            "message": "no scanner detected", "firmware": None,
        }

    # ---- lifecycle -------------------------------------------------------

    def start(self):
        if self._thread is None:
            self._thread = threading.Thread(target=self._loop, daemon=True)
            self._thread.start()

    def stop(self):
        self._stop.set()
        self._force.set()

    def force(self):
        """Re-check now (Connect button) and clear any init backoff."""
        self._prep_fails = 0
        self._prep_next = 0.0
        self._load_failed = False
        self._force.set()

    def set_scanning(self, busy):
        with self._lock:
            self._scanning = busy
        if busy:
            self._set(phase="scanning", message="scan in progress")
        else:
            self._force.set()

    def status(self):
        with self._lock:
            return dict(self._st)

    # ---- internals -------------------------------------------------------

    def _set(self, **changes):
        with self._lock:
            self._st.update(changes)
            self._st["connected"] = self._st.get("phase") == "ready"
            snap = dict(self._st)
        self._publish("device", snap)

    def _wait(self, seconds):
        if self._force.wait(timeout=seconds):
            self._force.clear()

    def _loop(self):
        while not self._stop.is_set():
            try:
                self._tick()
            except Exception as exc:                 # noqa: BLE001 — the monitor must never die
                self._set(phase="fault", error=str(exc), message="monitor error")
            self._wait(self.POLL)

    def _tick(self):
        if self.config.get("mock_mode"):
            if self._st.get("phase") != "mock":
                self._set(phase="mock", mock_mode=True, present=False, pid=None,
                          info=None, error=None, display="Mock scanner", message="mock mode")
            return

        if self._st.get("mock_mode"):                # just left mock mode
            self._set(mock_mode=False)

        with self._lock:
            scanning = self._scanning
        if scanning:
            return                                   # leave the device to the active scan

        pids = self.driver.present_pids()
        pid_hex = lambda p: "0x%04x" % p
        if self.driver.LOADED_PID in pids:
            self._load_failed = False
            phase = self._st.get("phase")
            gave_up = phase == "fault" and self._prep_fails >= self.PREP_MAX_FAILS
            if phase != "ready" and not gave_up and time.monotonic() >= self._prep_next:
                self._prepare()
        elif self.driver.UNLOADED_PID in pids:
            self._prep_fails = 0
            unloaded = "0x%04x" % self.driver.UNLOADED_PID
            fw = self.driver.firmware_status()
            if not fw["present"] and self.config.get("auto_prepare"):
                # Scanner is connected but the user hasn't supplied its firmware image
                # yet. Guide them (first-run) instead of looping on a doomed load.
                self._load_failed = False
                if (self._st.get("phase") != "firmware_missing"
                        or self._st.get("firmware") != fw):
                    self._set(phase="firmware_missing", present=True, pid=unloaded,
                              info=None, firmware=fw, error=None,
                              message="firmware needed — add your scanner firmware file")
                return
            if not self._load_failed:
                self._load()
        else:
            self._load_failed = False
            self._prep_fails = 0
            if self._st.get("phase") != "absent":
                self._set(phase="absent", present=False, pid=None, info=None,
                          error=None, display=None, message="no scanner detected")

    def _prepare(self):
        loaded = "0x%04x" % self.driver.LOADED_PID
        self._set(phase="preparing", present=True, pid=loaded,
                  error=None, message="initializing scanner…")
        try:
            info = self.driver.identify(blocking=False)
        except Exception as exc:                     # noqa: BLE001
            self._prep_fails += 1
            self._prep_next = time.monotonic() + self.PREP_BACKOFF
            msg = ("initialize failed — click Connect to retry"
                   if self._prep_fails >= self.PREP_MAX_FAILS else "initialize failed — retrying…")
            self._set(phase="fault", present=True, pid=loaded, info=None,
                      error=str(exc), message=msg)
            return
        if info is None:
            return                                   # device busy (scan) — retry next tick
        self._prep_fails = 0
        self._set(phase="ready", present=True, pid=loaded, info=info, error=None,
                  display="%s · %s" % (info.get("product"), info.get("serial_picl") or "—"),
                  message="ready")

    def _load(self):
        unloaded = "0x%04x" % self.driver.UNLOADED_PID
        if not self.config.get("auto_prepare"):
            self._set(phase="unloaded", present=True, pid=unloaded, info=None,
                      message="unloaded — auto-prepare is off")
            return
        self._set(phase="loading", present=True, pid=unloaded, error=None,
                  message="loading firmware…")
        try:
            ok = self.driver.load_firmware(log=lambda *_a: None)
            err = None
        except Exception as exc:                     # noqa: BLE001
            ok, err = False, str(exc)
        if ok:
            self._set(phase="preparing", message="firmware loaded — initializing…")
            self._force.set()
        else:
            self._load_failed = True
            self._set(phase="fault", present=True, pid=unloaded,
                      error=err, message="firmware load failed — power-cycle to retry")


class ScannerManager:
    """UI-facing status facade. Mock is synthetic; hardware reflects the monitor."""

    def __init__(self, config, driver, publish=None):
        self.config = config
        self.monitor = DeviceMonitor(config, driver, publish=publish)

    def start(self):
        self.monitor.start()

    def stop(self):
        self.monitor.stop()

    def status(self):
        if self.config.get("mock_mode"):
            return {
                "connected": True, "mock_mode": True, "display": "Mock scanner",
                "info": MOCK_INFO, "error": None, "phase": "mock", "message": "mock mode",
            }
        return self.monitor.status()

    def connect(self):
        """Manual 'Connect' = re-check now (the monitor does the prep)."""
        self.monitor.force()
        return self.status()
