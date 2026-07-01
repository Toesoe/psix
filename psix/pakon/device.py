#!/usr/bin/env python3
"""libpakon: userspace transport for the loaded Pakon F135 scanner (libusb / usb1).

Run pakon_load.py first so the device is the loaded 0f05:f135 (3 bulk endpoints:
EP0x01 OUT command, EP0x81 IN response, EP0x86 IN image).

Command protocol (verified against a live USB capture):
  command packet : [Type, PktLen, Address, payload...]   where PktLen == 1 + len(payload)
  response packet: [RType, Len, Address, data/status...]
  Types:  1 READ  [reg, count]              -> resp [0x01, len, addr, <count bytes>]
          2 WRITE  [count, reg, <val bytes>] -> resp [0x07, 0x02, addr, status]   (0x07=write-ack)
          3 POLL   [] (address only)         -> resp [0x03, 0x02, addr, status]
          4 WRITE2 [0x00, val]               -> resp [0x07, 0x02, addr, status]
  Addresses: 0x10 AD_HOST, 0x20 AD_PICL, 0x40 AD_PICL_PLUS, 0x44 sub-board.

This module is pure transport: register read/write/poll + raw replay + EP6 async
image streaming. Scan sequencing/poll-until-ready logic lives one layer up.
"""
import sys
import threading
import time

import usb1

VID = 0x0F05
PID_LOADED = 0xF135


class PakonError(Exception):
    pass


class PakonDevice:
    EP_CMD_OUT = 0x01
    EP_CMD_IN = 0x81
    EP_IMG_IN = 0x86
    IMG_PACKET = 0x5000          # 20480-byte image ring packets (confirmed from capture)
    RESP_MAX = 64                # driver used a 64-byte response buffer

    AD_HOST = 0x10
    AD_PICL = 0x20
    AD_PICL_PLUS = 0x40
    AD_SUB = 0x44

    RESP_READ = 0x01
    RESP_POLL = 0x03
    RESP_WRITE_ACK = 0x07

    def __init__(self):
        self.ctx = usb1.USBContext()
        self.ctx.open()
        self.handle = None
        self._lock = threading.RLock()   # serialize EP1 command transactions (driver's crit-section)

    # ---- lifecycle -------------------------------------------------------
    def open(self):
        self.handle = self.ctx.openByVendorIDAndProductID(VID, PID_LOADED, skip_on_error=True)
        if self.handle is None:
            raise PakonError(f"loaded scanner {VID:04x}:{PID_LOADED:04x} not found "
                             f"(run pakon_load.py first; check udev/permissions)")
        try:
            self.handle.setAutoDetachKernelDriver(True)
        except usb1.USBError:
            pass
        self.handle.claimInterface(0)
        return self

    def close(self):
        if self.handle is not None:
            try:
                self.handle.releaseInterface(0)
            except usb1.USBError:
                pass
            self.handle.close()
            self.handle = None
        self.ctx.close()

    def __enter__(self):
        return self.open()

    def __exit__(self, *exc):
        self.close()

    # ---- command channel (EP1) ------------------------------------------
    @staticmethod
    def build_packet(ptype, addr, payload=b''):
        payload = bytes(payload)
        return bytes([ptype, 1 + len(payload), addr]) + payload

    def transact(self, ptype, addr, payload=b'', timeout=2000):
        """Write one command packet to EP1-OUT, read the response from EP1-IN."""
        pkt = self.build_packet(ptype, addr, payload)
        with self._lock:
            self.handle.bulkWrite(self.EP_CMD_OUT, pkt, timeout=timeout)
            return bytes(self.handle.bulkRead(self.EP_CMD_IN, self.RESP_MAX, timeout=timeout))

    def send_raw(self, packet, timeout=2000):
        """Replay a raw, pre-built command packet (used by capture replay)."""
        with self._lock:
            self.handle.bulkWrite(self.EP_CMD_OUT, bytes(packet), timeout=timeout)
            return bytes(self.handle.bulkRead(self.EP_CMD_IN, self.RESP_MAX, timeout=timeout))

    # ---- closed-loop control-word servicing (F135 status/control protocol) ----
    @staticmethod
    def control_word(c, s):
        """Recompute the 16-bit CCD control word from status `s`: addr_sel = s&0xc0
        selects a table, val = s&0x1f indexes it."""
        addr_sel, val = s & 0xc0, s & 0x1f
        if addr_sel == 0x00:
            return {0: (c & 0xfcff) | 0x0c, 1: (c & 0xfcfb) | 0x08, 2: (c & 0xfcf7) | 0x04,
                    3: (c & 0xfcf3), 4: (c & 0xfdfb) | 0x108, 8: (c & 0xfef7) | 0x204,
                    0xc: (c & 0xfff3) | 0x300}.get(val, c)
        if addr_sel == 0x40:
            return {0: (c & 0xff3f) | 3, 1: (c & 0xff3e) | 2, 2: (c & 0xff3d) | 1,
                    3: (c & 0xff3c), 4: (c & 0xff7e) | 0x42, 8: (c & 0xffbd) | 0x81,
                    0xc: (c & 0xfffc) | 0xc0}.get(val, c)
        return c

    def read_reg(self, addr, reg, count, timeout=2000, retries=3):
        """Type 1 READ with the device's retry semantics: on a retryable status
        (type 0x07 status 3/6/9, i.e. busy/bus-busy) poll-until-ready and retry up to
        `retries` times before giving up."""
        last = None
        for _ in range(max(1, retries)):
            r = self.transact(1, addr, bytes([reg, count]), timeout)
            last = r
            if len(r) >= 3 and r[0] == self.RESP_READ:
                n = r[1] - 1               # Len counts addr + data; data = Len-1 bytes
                return r[3:3 + n]
            st = r[3] if len(r) >= 4 else None
            if st in (1, 3, 6, 9):         # retryable: wait for ready, then retry
                self.poll_until_ready(addr, max_tries=20, timeout=timeout)
                continue
            break
        raise PakonError(f"READ {hex(addr)} reg0x{reg:02x} failed: {last.hex() if last else None}")

    def write_reg(self, addr, reg, value, timeout=2000):
        """Type 2 WRITE: value is bytes; returns status byte (0 = OK)."""
        value = bytes(value)
        r = self.transact(2, addr, bytes([len(value), reg]) + value, timeout)
        if len(r) < 4 or r[0] != self.RESP_WRITE_ACK:
            raise PakonError(f"unexpected WRITE response: {r.hex()}")
        return r[3]

    def write2(self, addr, val, timeout=2000):
        """Type 4 WRITE2: [0x00, val]; returns status byte."""
        r = self.transact(4, addr, bytes([0, val & 0xFF]), timeout)
        if len(r) < 4 or r[0] != self.RESP_WRITE_ACK:
            raise PakonError(f"unexpected WRITE2 response: {r.hex()}")
        return r[3]

    def poll(self, addr, timeout=2000):
        """Type 3 POLL: returns the subsystem status byte."""
        r = self.transact(3, addr, b'', timeout)
        if len(r) < 4 or r[0] != self.RESP_POLL:
            raise PakonError(f"unexpected POLL response: {r.hex()}")
        return r[3]

    def poll_raw(self, addr, timeout=2000):
        """Type 3 POLL, returning the raw response (no validation)."""
        return self.transact(3, addr, b'', timeout)

    def poll_status(self, addr, timeout=2000):
        """Send a Type-3 poll; return the status byte (response[3]) regardless of
        response type. Bit 0x01 = busy/not-ready; bits 0x36 = data/fifo/error flags."""
        r = self.transact(3, addr, b'', timeout)
        return r[3] if len(r) >= 4 else None

    def poll_until_ready(self, addr, max_tries=44, timeout=2000):
        """Poll until status bit 0x01 (busy) clears (F135 busy/ready protocol).
        Returns (ready: bool, last_status, accumulated_flags(&0x36))."""
        flags = 0
        s = None
        for i in range(max_tries):
            s = self.poll_status(addr, timeout)
            if s is None:
                time.sleep(0.002); continue
            flags |= (s & 0x36)
            if (s & 0x01) == 0:
                return True, s, flags                 # ready
            time.sleep(min(0.002 * (i + 1), 0.05))    # incremental backoff
        return False, s, flags

    def read_reg_raw(self, addr, reg, count, timeout=2000):
        """Type 1 READ, raw response (no validation)."""
        return self.transact(1, addr, bytes([reg, count]), timeout)

    def poll_until(self, addr, predicate, timeout=30.0, interval=0.2):
        """Poll `addr` until predicate(status) is true or timeout (seconds)."""
        deadline = time.monotonic() + timeout
        last = None
        while time.monotonic() < deadline:
            last = self.poll(addr)
            if predicate(last):
                return last
            time.sleep(interval)
        raise PakonError(f"poll_until on 0x{addr:02x} timed out (last status=0x{last:02x})")

    # ---- image channel (EP6, async ring) --------------------------------
    def stream_images(self, on_packet, n_transfers=16, packet_size=None, stop=None):
        """Drain EP6 with `n_transfers` in-flight async transfers to keep the pipe
        full. `on_packet(bytes)` is called per completed transfer; return False from
        it (or supply stop()->True) to finish. Returns total bytes received."""
        packet_size = packet_size or self.IMG_PACKET
        state = {'inflight': 0, 'bytes': 0, 'done': False}

        def callback(transfer):
            st = transfer.getStatus()
            if st == usb1.TRANSFER_COMPLETED:
                n = transfer.getActualLength()
                state['bytes'] += n
                data = bytes(transfer.getBuffer()[:n])
                if not state['done']:
                    keep = on_packet(data)
                    if keep is False or (stop and stop()):
                        state['done'] = True
                if not state['done']:
                    try:
                        transfer.submit()
                        return
                    except usb1.USBError:
                        pass
            state['inflight'] -= 1

        transfers = []
        for _ in range(n_transfers):
            t = self.handle.getTransfer()
            t.setBulk(self.EP_IMG_IN, packet_size, callback=callback, timeout=0)
            t.submit()
            state['inflight'] += 1
            transfers.append(t)
        try:
            while state['inflight'] > 0:
                self.ctx.handleEvents()
        except usb1.USBErrorInterrupted:
            pass
        return state['bytes']


class PollService(threading.Thread):
    """Background closed-loop status poll/service (F135 heartbeat protocol).
    Continuously polls the subsystems and reads status reg0x02; if `service` is on,
    recomputes and writes the reg0x82 control word on change. The scanner firmware
    appears to require this heartbeat to scan.

    Thread-safe with the main thread via PakonDevice._lock (EP1 transactions).
    """
    def __init__(self, dev, period=0.05, service=False, verbose=False,
                 heartbeat=PakonDevice.AD_HOST, subsys=(PakonDevice.AD_PICL_PLUS, PakonDevice.AD_SUB)):
        super().__init__(daemon=True)
        self.dev = dev
        self.period = period
        self.service = service
        self.verbose = verbose
        self.heartbeat = heartbeat       # always-valid poll target (HOST 0x10) — the firmware heartbeat
        self.subsys = subsys             # subsystems whose reg0x02 status drives servicing
        self._stop = threading.Event()
        self.hb_status = None            # last HOST poll status byte
        self.valid = {}                  # subsys addr -> last valid reg0x02 byte
        self.cached = {a: 0 for a in subsys}
        self.cycles = 0
        self.hb_ok = 0
        self.busy = 0                    # subsystem reads that returned busy/bus-error
        self.errors = 0
        self.serviced = 0

    def run(self):
        while not self._stop.is_set():
            # 1) heartbeat: poll HOST (always valid) — this is what keeps the firmware alive
            try:
                r = self.dev.poll_raw(self.heartbeat)
                if len(r) >= 4 and r[0] == PakonDevice.RESP_POLL:
                    self.hb_status = r[3]; self.hb_ok += 1
            except Exception:
                self.errors += 1
            # 2) subsystem status reg0x02 (tolerant); service reg0x82 on valid status
            for a in self.subsys:
                try:
                    r = self.dev.read_reg_raw(a, 0x02, 1)
                    if len(r) >= 4 and r[0] == PakonDevice.RESP_READ:
                        s = r[3]
                        self.valid[a] = s
                        if self.service:
                            new = self.dev.control_word(self.cached[a], s)
                            if new != self.cached[a]:
                                self.dev.write_reg(a, 0x82, bytes([new & 0xff, (new >> 8) & 0xff]))
                                self.cached[a] = new
                                self.serviced += 1
                                if self.verbose:
                                    print("  [pollsvc] 0x%02x reg0x82<-%04x (status 0x%02x)" % (a, new, s))
                    else:
                        self.busy += 1          # bus-error/busy (e.g. 0x07../0x09) — tolerate
                except Exception:
                    self.errors += 1
            self.cycles += 1
            self._stop.wait(self.period)

    def snapshot(self):
        return {'HOST_hb': (hex(self.hb_status) if self.hb_status is not None else None),
                'subsys_valid': {hex(a): hex(s) for a, s in self.valid.items()},
                'hb_ok': self.hb_ok, 'busy': self.busy, 'serviced': self.serviced}

    def stop(self):
        self._stop.set()
        self.join(timeout=2)


def _selftest():
    """READ-ONLY probe of the command channel: poll + register reads. No writes,
    no motor, no lamp. Confirms libpakon can talk to the loaded scanner."""
    with PakonDevice() as dev:
        d = dev.handle.getDevice()
        print(f"opened {VID:04x}:{PID_LOADED:04x} bus{d.getBusNumber()} addr{d.getDeviceAddress()}")
        print("-- POLL (status) --")
        for name, addr in (('AD_HOST', dev.AD_HOST), ('AD_PICL_PLUS', dev.AD_PICL_PLUS),
                           ('AD_SUB', dev.AD_SUB)):
            try:
                print(f"  poll {name:13} 0x{addr:02x} -> status 0x{dev.poll(addr):02x}")
            except (PakonError, usb1.USBError) as e:
                print(f"  poll {name:13} 0x{addr:02x} -> {e}")
        print("-- READ (registers seen in capture) --")
        for name, addr, reg, cnt in (('HOST  reg02', dev.AD_HOST, 0x02, 0x03),
                                     ('PICL+ reg0c', dev.AD_PICL_PLUS, 0x0c, 0x07),
                                     ('SUB   reg0c', dev.AD_SUB, 0x0c, 0x07)):
            try:
                data = dev.read_reg(addr, reg, cnt)
                printable = ''.join(chr(b) if 32 <= b < 127 else '.' for b in data)
                print(f"  read {name}: {data.hex()}  '{printable}'")
            except (PakonError, usb1.USBError) as e:
                print(f"  read {name}: {e}")
