#!/usr/bin/env python3
"""pakon_dutyprobe: characterise the F-135's reg0x82 LED-duty field (lamp-only, no motor).

psix's DUTY_WRAP=900 came from a Plus duty sweep; on a base F-135 at the SAFE
current ceilings (R6/G8/B8) the servo pins at 900 and the stream shows line-to-line
brightness banding — so 900 sits in an unstable/beating region on this generation.
This probe sweeps the DUTY only (currents fixed at the board ceilings, so average
LED power never exceeds the vendor limit; the OEM itself drives duties up to ~0.97
fraction) and measures the CCD response per duty: level + line-to-line stability.

From its table we read:
  - the true duty period (output wraps over where the field rolls over)
  - the stable region (low line-mean variability) to constrain the servo

Run (film out, gate empty; the lamp will be ON):
    .venv/bin/python -m psix.pakon.dutyprobe
Read-only on the scanners' registers except the lamp/exposure writes themselves.
"""
import threading
import time

import numpy as np

from .scanner import PakonScanner
from .scanstart import build_scanstart_steps
from .scan import build_exposure, build_current
from .device import PakonError

DUTIES = (278, 400, 550, 707, 896, 1100, 1353, 1600, 1800, 2000, 2040)
CURRENTS = (2, 3, 3)          # OEM F-135 capture operating currents (R, G, B) — reg0x81 [B,IR,R,0,G]=3,2,2,3
BASE = 1858                   # OEM F-135 exposure base field (Plus: 0x03d6)
TRIGGER = b'\x10\x00\x01'     # OEM F-135 trigger word (Plus: 3c0001)
WINDOW = 0.4                  # seconds of EP6 sampled per duty step


def _arm(dev):
    dev.write_reg(dev.AD_HOST, 0x84, b'\x02')
    dev.write2(dev.AD_PICL_PLUS, 0x8a)


def probe(currents=CURRENTS, duties=DUTIES, base=BASE, window=WINDOW):
    sc = PakonScanner(verbose=True)
    sc.open()
    try:
        sc.initialize()
    except PakonError:
        # e.g. a previous session left the acquisition open — one recovery re-init,
        # then give up with a power-cycle hint.
        print("init did not reach ready — attempting recover() (power-cycle the scanner if this fails too)")
        sc.recover()
    dev = sc.dev
    try:
        # scan-start skeleton UP TO the motor GO (includes trigger#1 -> EP6 streams,
        # and lamp-on), then stop: no motor, no transport trigger.
        steps = build_scanstart_steps(known_only=True,
                                      light=dev.AD_PICL_PLUS, motor=dev.AD_SUB)
        cut = len(steps)
        for i, s in enumerate(steps):
            b = bytes.fromhex(s['data'])
            if b[0] == 4 and len(b) >= 5 and b[2] == dev.AD_SUB and b[4] == 0xa0:
                cut = i
                break
        # EP6 listener FIRST (transfers queued before trigger#1), then replay the
        # steps PACED BY THEIR rel SCHEDULE (the device needs the settle times; the
        # calibration pass after trigger#1 is what starts the image stream).
        buf = bytearray()
        done = [False]

        def cb(pkt):
            buf.extend(pkt)
            return True

        t = threading.Thread(target=dev.stream_images, args=(cb,),
                             kwargs={'stop': lambda: done[0]}, daemon=True)
        t.start()
        time.sleep(0.3)
        print("sending %d scan-start steps (paced, up to motor GO, excluded)" % cut)
        prev = steps[0]['rel']
        for s in steps[:cut]:
            gap = s['rel'] - prev
            prev = s['rel']
            if gap > 0:
                end = time.monotonic() + min(gap, 3.0)
                while time.monotonic() < end:
                    try:
                        dev.poll_status(dev.AD_HOST)          # flow-control heartbeat
                    except Exception:                          # noqa: BLE001
                        pass
                    time.sleep(0.01)
            try:
                dev.send_raw(bytes.fromhex(s['data']))
            except Exception as e:                            # noqa: BLE001
                print("  send: %s" % e)
        print("scan-start sent; EP6 bytes so far: %d (thread_alive=%s)"
              % (len(buf), t.is_alive()))

        dev.write_reg(dev.AD_PICL_PLUS, 0x81, build_current(*currents))
        _arm(dev)
        # start CONTINUOUS acquisition (the trigger#2 word — a pure register write,
        # the motor is NOT needed). trigger#1's pass is one-shot; without this the
        # CCD streams nothing after the calibration pass drains.
        dev.write_reg(dev.AD_PICL_PLUS, 0x91, TRIGGER)
        end = time.monotonic() + 1.0
        while time.monotonic() < end:
            try:
                dev.poll_status(dev.AD_HOST)
            except Exception:                             # noqa: BLE001
                pass
            time.sleep(0.01)
        print("continuous acquisition on; EP6 bytes: %d" % len(buf))
        print("\n  duty   R_mean   G_mean   B_mean   R_var%%   G_var%%   B_var%%   (var = line-to-line p95/p50 spread)")
        results = []
        start = 0
        for d in duties:
            mark = len(buf)
            dev.write_reg(dev.AD_PICL_PLUS, 0x82,
                          build_exposure(d, d, d, base=base, ir=0))
            _arm(dev)                                      # commit; takes effect live on the stream
            end = time.monotonic() + window
            while time.monotonic() < end:
                try:
                    dev.poll_status(dev.AD_HOST)           # flow-control heartbeat
                except Exception:                          # noqa: BLE001
                    pass
                time.sleep(0.01)
            raw = np.frombuffer(bytes(buf[mark:]), dtype='<u2')
            P = 6000
            mk = np.flatnonzero(raw & 1)
            if len(mk) < 20:
                print("%5d   (no EP6 lines: %d bytes, %d markers, thread_alive=%s)"
                      % (d, len(raw), len(mk), t.is_alive()))
                continue
            ph = int(np.bincount(mk % P, minlength=P).argmax())
            r = raw[ph:]
            nl = len(r) // P
            if nl < 5:
                print("%5d   (too few lines: %d)" % (d, nl))
                continue
            lines = r[:nl * P].reshape(nl, P).astype(np.float32)
            out = []
            for c in range(3):
                lm = lines[:, c::3].mean(axis=1)
                med = float(np.median(lm))
                spread = float(np.percentile(np.abs(np.diff(lm)), 95)) / max(med, 1.0) * 100.0
                out.append((med, spread))
            results.append((d, out))
            print("%5d  %7.0f  %7.0f  %7.0f   %6.1f   %6.1f   %6.1f"
                  % (d, out[0][0], out[1][0], out[2][0], out[0][1], out[1][1], out[2][1]))
        done[0] = True
        t.join(timeout=3)
        # teardown: end acquisition, then re-run InitializeScanner — the ONLY reliable
        # lamp-off (a bare reg0x80=00, armed or not, does not clear a lit lamp).
        try:
            dev.write2(dev.AD_PICL_PLUS, 0x92)                # EndAcquisition
        except Exception:                                    # noqa: BLE001
            pass
        from . import scan as pakon_scan2
        if not pakon_scan2.reset_to_idle(dev):
            print("WARNING: reset_to_idle failed — lamp may stay lit; run a psix Connect to clear it.")
        print("total EP6 bytes: %d" % len(buf))
        return results
    finally:
        sc.shutdown()


if __name__ == '__main__':
    probe()
