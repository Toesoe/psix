#!/usr/bin/env python3
"""pakon_scanner: §1 device-lifecycle state machine for the loaded Pakon F135.

Models the OEM device lifecycle as ONE explicit state machine over libpakon, replacing the
scattered load/init/probe scripts with a single object:

    CLOSED --open()--> OPEN --initialize()--> READY <--begin_scan()/end_scan()--> SCANNING
       ^                                        |
       +----------------- shutdown() -----------+         (FAULT --recover()--> READY | raises)

Backed by the F135 device protocol:
  - initialize()  = the RECONSTRUCTED init (pakon_initseq, byte-identical to OEM dev36, HW-validated).
  - wait_ready()  = poll HOST reg0x02 until bit7 (0x80) appears (the proven ready signal; dip-tolerant
                    — the firmware pulses bit7, so we poll until it shows rather than reading once).
  - health()      = structured identify + status snapshot (versions, serials,
                    subsystem status, temp ADC).
  - shutdown()    = safe-idle (stop motor reg0xa5=0/0xa0/0xa2 + light disable reg0x80=0) then release.
                    Idempotent; there is NO power-down opcode — stop+disable+free.
  - recover()     = re-init after a fault; raises if the device dropped (needs pakon_load again).
  - readiness/fault aggregate over {PICL+ 0x40, sub44 0x44, sub28 0x28}, fault mask 0x5f5fbff.

Prereq: run pakon_load.py first (device must be the loaded 0f05:f135). This object is non-actuating
through initialize()/health(); shutdown() only stops/disables (safe even if nothing was running).

CLI:  python pakon_scanner.py --demo     # open -> initialize -> health -> shutdown (on hardware)
      python pakon_scanner.py --health    # open -> health -> shutdown (read-only-ish)
"""
import enum
import json
import os
import sys
import time

from . import device as libpakon
from .device import PakonDevice, PakonError
from . import initseq as pakon_initseq
import usb1


class State(enum.Enum):
    CLOSED = 'closed'        # no USB handle
    OPEN = 'open'            # device opened (loaded f135), not yet initialized
    READY = 'ready'          # initialized + reached ready (idle, no fault)
    SCANNING = 'scanning'    # a scan/transport is running (actual transport = §2)
    FAULT = 'fault'          # a fault latched; recover() or shutdown()
    SHUTDOWN = 'shutdown'    # torn down


# Readiness / status aggregation. Model 'D' polls these subsystems.
READY_SUBSYS = (PakonDevice.AD_PICL_PLUS, PakonDevice.AD_SUB, 0x28)  # PICL+, sub44, sub28
FAULT_MASK = 0x5f5fbff       # OEM any-fault mask over the aggregated status word
READY_BIT = 0x80             # HOST reg0x02 bit7 = ready/service signal
BUSY_BIT = 0x01              # reg0x02 bit0 = busy


class PakonScanner:
    """Explicit §1 lifecycle for the loaded F135. Use as a context manager for guaranteed shutdown."""

    def __init__(self, verbose=False):
        self.dev = None
        self.state = State.CLOSED
        self.verbose = verbose
        self.eeprom = None              # raw EEPROM bytes read during initialize() (LED cal etc.)
        self.init_anomalies = []
        self.first_fault = None         # latched first fault (mirrors the OEM error handler)
        self.last_status = None         # last HOST reg0x02 status byte

    # ---- fault sink --------------------------------------------------------
    def _fault(self, msg):
        """Latch the FIRST fault (never overwrite) and enter FAULT, like the OEM error handler."""
        if self.first_fault is None:
            self.first_fault = msg
        self.state = State.FAULT
        if self.verbose:
            print("  [FAULT] %s" % msg)
        return PakonError(msg)

    def _require(self, *states):
        if self.state not in states:
            raise PakonError("illegal transition from %s (need one of %s)"
                             % (self.state.value, [s.value for s in states]))

    # ---- transitions -------------------------------------------------------
    def open(self):
        """CLOSED -> OPEN: claim the loaded f135. Raises if not present (run pakon_load.py first)."""
        self._require(State.CLOSED, State.SHUTDOWN)
        try:
            self.dev = PakonDevice().open()
        except PakonError as e:
            raise self._fault("open failed: %s" % e)
        self.first_fault = None
        self.state = State.OPEN
        if self.verbose:
            d = self.dev.handle.getDevice()
            print("opened %04x:%04x bus%d addr%d"
                  % (libpakon.VID, libpakon.PID_LOADED, d.getBusNumber(), d.getDeviceAddress()))
        return self

    def initialize(self):
        """OPEN -> READY: run the reconstructed InitializeScanner, then wait for ready."""
        self._require(State.OPEN, State.FAULT)
        seq = pakon_initseq.build_init_steps()
        self.eeprom, self.init_anomalies = pakon_initseq.run_steps(self.dev, seq, self.verbose)
        if self.verbose:
            print("  init: %d steps, EEPROM %d B, %d anomalies"
                  % (len(seq), len(self.eeprom), len(self.init_anomalies)))
        status = self.wait_ready()
        if status is None or not (status & READY_BIT):
            raise self._fault("init did not reach ready (HOST=%s)"
                              % ('0x%02x' % status if status is not None else 'n/a'))
        self.state = State.READY
        return self

    def wait_ready(self, timeout=12.0, interval=0.2):
        """Poll HOST reg0x02 until bit7 (ready) appears, up to `timeout`. Returns the status byte.
        Dip-tolerant: the firmware pulses bit7 post-init, so we poll until it shows (the single-read
        approach false-negatived at 0x00)."""
        if self.dev is None:
            raise PakonError("wait_ready: device not open")
        end = time.monotonic() + timeout
        status = None
        while time.monotonic() < end:
            try:
                status = self.dev.poll_status(self.dev.AD_HOST)
            except (PakonError, usb1.USBError):
                status = None
            if status is not None:
                self.last_status = status
                if status & READY_BIT:
                    return status
            time.sleep(interval)
        return status

    def is_ready(self):
        """True iff HOST shows ready (bit7). This is the PROVEN ready signal (the init verdict, HW-
        validated). Subsystem status is advisory only — we do NOT gate on a reconstructed aggregate,
        because the exact OEM bit-packing (mask 0x5f5fbff) isn't faithfully reproduced
        here and a bare idle read of PICL+/sub28 isn't reliable without the OEM poll context."""
        if self.dev is None:
            return False
        try:
            host = self.dev.poll_status(self.dev.AD_HOST)
        except (PakonError, usb1.USBError):
            return False
        self.last_status = host
        return bool(host is not None and (host & READY_BIT))

    def subsystem_status(self):
        """Type-3 POLL of {PICL+, sub44, sub28} -> {addr: status_byte|None}. POLL is the reliable
        status query (Type-1 reg0x02 reads need the OEM poll context); bit0=busy, bit7=service.
        Advisory health info, not a fault verdict."""
        per = {}
        for addr in READY_SUBSYS:
            try:
                per[addr] = self.dev.poll_status(addr)
            except (PakonError, usb1.USBError):
                per[addr] = None
        return per

    def _read_serial(self, addr):
        """Latch reg0x03=1 then read the ASCII serial at reg0x0c x7 (the identify sub-sequence)."""
        try:
            self.dev.write_reg(addr, 0x03, b'\x01')
            self.dev.poll_status(addr)
            b = bytes(self.dev.read_reg(addr, 0x0c, 7, retries=2))
            s = ''.join(chr(c) for c in b if 32 <= c < 127)
            return s or None
        except (PakonError, usb1.USBError):
            return None

    def health(self):
        """Structured identify + status snapshot (the identify reads + temp ADC). Every read is
        best-effort so a single non-responding register never throws."""
        if self.dev is None:
            return {'state': self.state.value, 'error': 'device not open'}

        def rd(addr, reg, n):
            try:
                return bytes(self.dev.read_reg(addr, reg, n, retries=2))
            except (PakonError, usb1.USBError):
                return None

        host = None
        try:
            host = self.dev.poll_status(self.dev.AD_HOST)
        except (PakonError, usb1.USBError):
            pass
        per = self.subsystem_status()
        usb_fw = rd(self.dev.AD_HOST, 0x03, 2)            # USB firmware version
        temp_adc = rd(self.dev.AD_PICL_PLUS, 0x88, 4)     # MB/LB temp ADC

        return {
            'state': self.state.value,
            'host_status': None if host is None else '0x%02x' % host,
            'ready': bool(host is not None and (host & READY_BIT)),
            'subsystems': {hex(a): (None if v is None else '0x%02x' % v) for a, v in per.items()},
            'comm_ok': all(v is not None for v in per.values()),
            'usb_fw': None if usb_fw is None else '0x%02x,0x%02x' % (usb_fw[0], usb_fw[1]),
            'serial_picl': self._read_serial(self.dev.AD_PICL_PLUS),
            'serial_sub': self._read_serial(self.dev.AD_SUB),
            'temp_adc': None if temp_adc is None else temp_adc.hex(),
            'eeprom_bytes': len(self.eeprom) if self.eeprom else 0,
            'first_fault': self.first_fault,
            'init_anomalies': len(self.init_anomalies),
        }

    def begin_scan(self):
        """READY -> SCANNING. State transition only; actual lamp+motor transport is §2 (pakon_scan2)."""
        self._require(State.READY)
        self.state = State.SCANNING
        return self

    def end_scan(self):
        """SCANNING -> READY (after the §2 transport has stopped)."""
        self._require(State.SCANNING)
        self.state = State.READY
        return self

    def shutdown(self):
        """Any -> CLOSED. Safe-idle then release. Idempotent. (No power-down opcode —
        stop motor [reg0xa5=0 + a0 + a2] + light disable [reg0x80=0] + free host resources.)"""
        if self.state in (State.CLOSED, State.SHUTDOWN) or self.dev is None:
            self.state = State.SHUTDOWN
            return self
        try:
            with self.dev._lock:
                # stop film drive (the proven motor-stop; harmless if nothing is running)
                try:
                    self.dev.write_reg(self.dev.AD_SUB, 0xa5, b'\x00\x00')
                    self.dev.write2(self.dev.AD_SUB, 0xa0)
                    self.dev.write2(self.dev.AD_SUB, 0xa2)
                except (PakonError, usb1.USBError):
                    pass
                # light/master disable
                try:
                    self.dev.write_reg(self.dev.AD_PICL_PLUS, 0x80, b'\x00')
                except (PakonError, usb1.USBError):
                    pass
        finally:
            try:
                self.dev.close()
            except Exception:        # noqa: BLE001
                pass
            self.dev = None
            self.state = State.SHUTDOWN
        if self.verbose:
            print("  shutdown: motor stopped, light disabled, device released")
        return self

    def recover(self):
        """FAULT -> READY by re-initializing. Raises (and goes CLOSED) if the device dropped off the
        bus (needs pakon_load.py again). USB stalls/busy are already retried inside libpakon."""
        self._require(State.FAULT, State.OPEN, State.READY)
        # is the loaded device still present?
        try:
            present = usb1.USBContext()
            present.open()
            h = present.openByVendorIDAndProductID(libpakon.VID, libpakon.PID_LOADED, skip_on_error=True)
            present.close()
        except Exception:            # noqa: BLE001
            h = None
        if h is None:
            self.state = State.CLOSED
            self.dev = None
            raise PakonError("device not present (0f05:f135 gone) — re-run pakon_load.py")
        if self.dev is None:
            self.open()
        self.state = State.OPEN       # allow initialize()'s precondition
        self.initialize()            # re-run the reconstructed init
        return self

    # ---- context manager ---------------------------------------------------
    def __enter__(self):
        return self.open()

    def __exit__(self, *exc):
        self.shutdown()
        return False
