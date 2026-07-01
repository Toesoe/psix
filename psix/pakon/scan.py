#!/usr/bin/env python3
"""pakon_scan2: F135 scan driver — calibrate() phase only (motor-free, safe).

Pipeline so far:  pakon_load.py  ->  pakon_init.py  ->  pakon_scan2.py --calibrate

calibrate() reproduces the OEM calibration pre-scan (the automatic "scanner corrections":
LED warm-up + gain/exposure reference), captured in a USB capture (dev17) and extracted to
captures/dev17_calib.json. Sequence: trigger#1 -> DARK reference pass (lamp off) -> lamp ON
-> WHITE/warm-up reference pass. It then captures the calibration line-CCD image off EP6 to
captures/calib_ep6.bin so we can finally decode the raster format (usbmon can't capture EP6).

SAFETY:
  * The LED light source turns ON (required for the white reference). Gate: --i-understand-lamp.
  * NO motor runs — verified: the extracted window (15.0-22.7s) contains 0 motor commands
    (reg0xa5/a0/a1); the motor spin-up at 25.6s is excluded. Nothing transports film.
  * On exit (always): lamp OFF (reg0x80=00) + the abort/teardown sequence.
  * Requires the device already initialized (run pakon_init.py first); refuses if HOST not ready.

This is a faithful REPLAY of the OEM calibration (not yet the reactive gain/exposure servo) —
enough to exercise the CCD passes safely and capture real EP6 data. The reactive servo and the
transport scan (motor) come later.
"""
import json
import os
import sys
import time

import usb1
from . import device as libpakon
from .device import PakonDevice, PakonError
from . import initseq as pakon_initseq
from . import scanstart as pakon_scanstart

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CALIB_SEQ = os.path.join(ROOT, 'captures', 'dev17_calib.json')
ABORT = os.path.join(ROOT, 'captures', 'abort_sequence.json')
EP6_OUT = os.path.join(ROOT, 'captures', 'calib_ep6.bin')
INIT_SEQ = os.path.join(ROOT, 'captures', 'dev36_init.json')


def reset_to_idle(dev):
    """Reset the device to the idle/ready state (lamp OFF) by re-running InitializeScanner.
    The lamp is held on by the active scan state; a bare reg0x80=00 (or the open-loop abort replay)
    does NOT clear it. Re-running the init sequence returns the device to the post-pakon_init idle
    state, which is lamp-off. This is the reliable teardown (grounded: that's the confirmed init result).

    Uses the RE-derived init builder (pakon_initseq.build_init_steps, byte-identical to the old
    dev36_init.json replay) instead of replaying the opaque blob,
    so the teardown is fully named/understood. run_steps() reads the live EEPROM record headers itself."""
    try:
        seq = pakon_initseq.build_init_steps()
        pakon_initseq.run_steps(dev, seq)
        return True
    except Exception:
        return False


def is_lamp(pkt):
    return len(pkt) >= 5 and pkt[0] == 2 and pkt[2] == 0x40 and pkt[4] == 0x80
def is_trigger(pkt):
    return len(pkt) >= 5 and pkt[0] == 2 and pkt[2] == 0x40 and pkt[4] == 0x91


def calibrate(dev, steps, out_path, hold, pump, max_bytes):
    """Flow-controlled calibration: replay the OEM calib commands, but start EP6 only AT
    the trigger (not free-running) and pace it with a continuous ~100 Hz HOST-poll heartbeat
    (the firmware's flow control), capped at max_bytes. Returns (bytes, pkts).

    Rationale: the first run free-drained 96 MB (60 MB before the trigger)
    open-loop and the scanner faulted (red). The OEM reads EP6 only after the trigger and
    paces image delivery via the HOST poll; this mirrors that.
    """
    img = {'bytes': 0, 'pkts': 0}
    out = open(out_path, 'wb')
    stop = [False]
    transfers = []

    def cb(tr):
        if tr.getStatus() == usb1.TRANSFER_COMPLETED:
            n = tr.getActualLength()
            if n:
                out.write(tr.getBuffer()[:n]); img['bytes'] += n; img['pkts'] += 1
        if not stop[0] and img['bytes'] < max_bytes:
            try:
                tr.submit()
            except usb1.USBError:
                pass

    def open_ep6(n=16):
        for _ in range(n):
            tr = dev.handle.getTransfer()
            tr.setBulk(dev.EP_IMG_IN, dev.IMG_PACKET, callback=cb, timeout=0)
            tr.submit(); transfers.append(tr)

    # locate the trigger; pre-trigger commands run with NO EP6 draining
    trig_idx = next((i for i, s in enumerate(steps)
                     if is_trigger(bytes.fromhex(s['data']))), len(steps) - 1)
    t_start = time.monotonic()
    try:
        # --- Phase A: pre-trigger setup, no image ---
        prev = steps[0]['rel']
        for s in steps[:trig_idx]:
            gap = s['rel'] - prev; prev = s['rel']
            if gap > 0:
                time.sleep(min(gap, 2.0))
            dev.send_raw(bytes.fromhex(s['data']))

        # --- Phase B: trigger -> open EP6 -> flow-controlled drain + scheduled replay ---
        trig = bytes.fromhex(steps[trig_idx]['data'])
        dev.send_raw(trig)
        open_ep6(16)
        print("  [%5.1fs] TRIGGER reg0x91=%s — EP6 started (flow-controlled, cap %d MB)"
              % (time.monotonic() - t_start, trig[5:5 + trig[3]].hex(), max_bytes // (1 << 20)))

        sched = [(s['rel'], bytes.fromhex(s['data'])) for s in steps[trig_idx + 1:]]
        base = steps[trig_idx]['rel']
        loop0 = time.monotonic()
        si = 0
        end = loop0 + (steps[-1]['rel'] - base) + hold
        last_report = 0
        while time.monotonic() < end and img['bytes'] < max_bytes:
            now = base + (time.monotonic() - loop0)
            # issue any due replay commands
            while si < len(sched) and sched[si][0] <= now:
                pkt = sched[si][1]; si += 1
                if is_lamp(pkt):
                    print("  [%5.1fs] LAMP %s (EP6 %d B)"
                          % (time.monotonic() - t_start, 'ON' if pkt[5] else 'OFF', img['bytes']))
                try:
                    dev.send_raw(pkt)
                except usb1.USBError as e:
                    print("  send error %s: %s" % (pkt.hex(), e))
            # flow-control heartbeat: HOST poll (also pumps EP6 completions) + event pump
            try:
                dev.poll_status(dev.AD_HOST)
            except usb1.USBError:
                pass
            dev.ctx.handleEventsTimeout(pump)
            if img['bytes'] - last_report >= 8 << 20:
                last_report = img['bytes']
                print("  [%5.1fs] EP6 %d MB" % (time.monotonic() - t_start, img['bytes'] >> 20))
        print("  >>> done (EP6 %d B, cap hit=%s)" % (img['bytes'], img['bytes'] >= max_bytes))
    finally:
        stop[0] = True
        for tr in transfers:
            try:
                tr.cancel()
            except usb1.USBError:
                pass
        end = time.monotonic() + 0.3
        while time.monotonic() < end:
            dev.ctx.handleEventsTimeout(pump)
        out.close()
    return img['bytes'], img['pkts']


def build_exposure(r, g, b, base=0x03d6, ir=0):
    """PICL+ reg0x82 12-byte exposure packet. Layout:
    [B16, IR16, R16, 0000, G16, base16], little-endian."""
    import struct
    return struct.pack('<HHHHHH', b & 0xffff, ir & 0xffff, r & 0xffff, 0, g & 0xffff, base & 0xffff)


def build_current(r, g, b, ir=0):
    """PICL+ reg0x81 5-byte per-channel LED CURRENT (coarse brightness). Layout
    (the 0x81 write): [B, IR, R, 0, G], 1 byte/channel. Ceilings: R<=8, G,B<=20, IR<=8
    (G/B higher to punch the orange C-41 mask)."""
    return bytes([b & 0xff, ir & 0xff, r & 0xff, 0, g & 0xff])


# --- §2c resolution / scan geometry ------------------------------------------
# Clean-room control of the CCD geometry = the registers the OEM changes per base tier (a selector
# picks the geometry + rate). The per-tier CONSTANTS were extracted from
# the device's registry defaults — the defaults used when the
# \DpiBase{4,8,16}_35 registry keys are absent (our registry-less case). base16 = native = the init
# baseline + what our captures scan. Each preset: CCD width/offset/integration + MODE_8C + MODE_90
# (=PICL+ reg0x89 readout mode) + the OEM output frame size (W x H).
GEOM_BASE4 = dict(name='base4', ires=0, width=1000, offset=0x1f, integration=0x753,   # 1875
                  mode8c=1, mode90=0, frame=(1000, 1500))
GEOM_BASE8 = dict(name='base8', ires=1, width=2000, offset=0x3e, integration=0xafd,   # 2813
                  mode8c=0, mode90=1, frame=(1500, 2250))
GEOM_BASE16 = dict(name='base16', ires=2, width=2000, offset=0x3e, integration=0xffd,  # 4093 (native)
                   mode8c=0, mode90=0, frame=(2000, 3000))
GEOM_TIERS = {'base4': GEOM_BASE4, 'base8': GEOM_BASE8, 'base16': GEOM_BASE16}
GEOM_NATIVE = GEOM_BASE16


def set_scan_geometry(dev, width, offset, integration, mode8c=0, mode90=0, motor_rate=None):
    """Write the CCD geometry = exactly the registers the OEM programs per resolution tier.
    Horizontal resolution = width (CCD pixels) via sub44 reg0x82
    idx4=offset / idx5=offset+width / idx6=integration + PICL+ reg0x89=mode90 (readout mode). MODE_8C
    sets bit1 of the idx0 enable mask (best-effort RMW; also selects the half pixel-end clamp). Optional
    transport rate -> sub44 reg0xa5. Returns the values written. (clamp: offset+width <= 0x848 / 2120,
    or 0x424 / 1060 when mode8c.)"""
    end = offset + width
    clamp = 0x424 if mode8c else 0x848
    if end > clamp:
        raise ValueError("offset+width=%d exceeds CCD clamp 0x%x (mode8c=%d)" % (end, clamp, mode8c))
    if integration > 0xffd:
        raise ValueError("integration %d exceeds max 0xffd" % integration)
    SUB = dev.AD_SUB
    dev.write_reg(SUB, 0x82, bytes([0x06, integration & 0xff, (integration >> 8) & 0xff]))  # idx6 integ
    dev.write_reg(SUB, 0x82, bytes([0x04, offset & 0xff, (offset >> 8) & 0xff]))             # idx4 offset
    dev.write_reg(SUB, 0x82, bytes([0x05, end & 0xff, (end >> 8) & 0xff]))                   # idx5 end
    dev.write_reg(dev.AD_PICL_PLUS, 0x89, bytes([mode90 & 0xff]))                            # readout mode
    if mode8c:                                          # idx0 enable-mask bit1 (best-effort RMW)
        try:
            cur = bytes(dev.read_reg(SUB, 0x83, 3, timeout=400, retries=1))   # indexed read-back of idx0
            m = (cur[1] | (cur[2] << 8)) if len(cur) >= 3 else 0
            m |= 0x2
            dev.write_reg(SUB, 0x82, bytes([0x00, m & 0xff, (m >> 8) & 0xff]))
        except Exception:                               # noqa: BLE001 (non-fatal; clamp already covers base4 dims)
            pass
    if motor_rate is not None:
        dev.write_reg(SUB, 0xa5, bytes([motor_rate & 0xff, (motor_rate >> 8) & 0xff]))       # transport rate
    return dict(width=width, offset=offset, end=end, integration=integration,
                mode8c=mode8c, mode90=mode90, motor_rate=motor_rate)


# --- §2b principled FILM exposure --------------------------------------------
DUTY_WRAP = 900                       # reg0x82 duty ceiling before the 12-bit wrap (per the duty sweep)
FILM_BASE_TARGET = 59000              # ~90% of full-scale: land the clear film base (Dmin) here, with
                                      # ~10% headroom so the base (brightest signal on a negative) never clips.


def measure_film_base(binpath):
    """Measure the per-channel CLEAR FILM BASE (Dmin) level from a scan .bin — the brightest stable
    signal on a C-41 negative (the inter-frame rebate). Returns base[3] (R,G,B) or None. Uses the
    rebate gaps between detected frames; falls back to the high percentile of the film region."""
    import numpy as np
    from pakon_invert import find_line_phase, _read_lines, detect_frames_oem
    P = 6000
    head = np.fromfile(binpath, dtype='<u2', count=P * 3000)
    phase, _, _ = find_line_phase(head, P); phase = phase or 0; del head
    nlines = (os.path.getsize(binpath) // 2 - phase) // P
    bright = np.empty(nlines, np.float32); detail = np.empty(nlines, np.float32); unif = np.empty(nlines, np.float32)
    for a in range(0, nlines, 4000):
        b = min(a + 4000, nlines)
        g = _read_lines(binpath, phase, a, b, P)[:, :6000][:, 1::3].astype(np.float32)
        bright[a:b] = g.mean(1); detail[a:b] = np.abs(np.diff(g, axis=1)).mean(1); unif[a:b] = g.std(1); del g
    frames, pitch = detect_frames_oem(bright, unif, detail)
    # rebate = the clear gaps BETWEEN frames (true film base). Sample them per channel.
    rebates = [(frames[k][1], frames[k + 1][0]) for k in range(len(frames) - 1)
               if frames[k + 1][0] - frames[k][1] > 20]
    base = None
    if rebates:
        vals = [[], [], []]
        for a, b in rebates[:12]:
            seg = _read_lines(binpath, phase, a, b, P)[:, :6000].astype(np.float32)
            for c in range(3):
                vals[c].append(float(np.percentile(seg[:, c::3], 50)))   # rebate is uniform clear base
            del seg
        base = [float(np.median(v)) for v in vals]
        src = "%d rebate gaps" % len(rebates)
    else:                                            # fallback: high pct of the modulated film region
        fi = np.where(detail > 0.15 * np.percentile(detail, 80))[0]
        if len(fi) < 50:
            return None, "no film region found"
        fa, fb = int(fi[0]), int(fi[-1])
        seg = _read_lines(binpath, phase, fa, min(fb, fa + 8000), P)[:, :6000].astype(np.float32)
        base = [float(np.percentile(seg[:, c::3], 99.5)) for c in range(3)]   # clear base ~ top of film
        src = "film-region p99.5 (no clean rebates)"; del seg
    return base, src


def film_duty_for_target(exp_film, base, target=FILM_BASE_TARGET, wrap=DUTY_WRAP):
    """Compute the per-channel film DUTY that lands the measured film base at `target`. Level≈k·duty at
    fixed current (linear until clip), so duty = exp_film·target/base.
    Returns (duties[3], limited[3]) where limited[c]=True means the channel hit the wrap and can't reach
    target (green is typically duty-limited by the orange mask). exp_film = the duty USED for `base`."""
    out, limited = [], []
    for c in range(3):
        if base[c] <= 0:
            out.append(wrap); limited.append(True); continue
        d = exp_film[c] * float(target) / float(base[c])
        out.append(max(1, min(wrap, int(round(d)))))
        limited.append(d > wrap)
    return out, limited


def read_phases_sync(dev, nlines=64, settle_pkts=24):
    """Synchronous EP6 read, MARKER-ALIGNED (bit0 line-sync) -> per-deinterleave-phase 99th-pct level.
    Discards settle_pkts (let the new current/duty take effect), reads ~nlines, aligns on the line
    marker (stable phase<->channel), returns [phase0, phase1, phase2] or None."""
    import numpy as np
    for _ in range(settle_pkts):
        try:
            dev.handle.bulkRead(dev.EP_IMG_IN, dev.IMG_PACKET, timeout=1000)
        except usb1.USBError:
            break
    P = 2000 * 3
    need = nlines * P * 2
    buf = bytearray()
    while len(buf) < need:
        try:
            buf += dev.handle.bulkRead(dev.EP_IMG_IN, dev.IMG_PACKET, timeout=1000)
        except usb1.USBError:
            break
    a = np.frombuffer(bytes(buf), dtype='<u2')
    mk = np.flatnonzero(a & 1)
    if len(mk) == 0:
        return None
    ph = int(np.bincount(mk % P, minlength=P).argmax())
    m = a[ph: ph + ((len(a) - ph) // P) * P]
    if len(m) < P:
        return None
    lines = m.reshape(-1, P).astype(np.float32)
    return [float(np.percentile(lines[:, c::3], 99)) for c in range(3)]


def read_profiles_sync(dev, nlines=128, settle_pkts=40):
    """Like read_phases_sync but returns the per-phase per-COLUMN MEAN profile (3 x width). Used to
    measure whether the LED's per-column spatial profile changes with duty (the duty-mismatch question)."""
    import numpy as np
    for _ in range(settle_pkts):
        try:
            dev.handle.bulkRead(dev.EP_IMG_IN, dev.IMG_PACKET, timeout=1000)
        except usb1.USBError:
            break
    Pn = 2000 * 3
    need = nlines * Pn * 2
    buf = bytearray()
    while len(buf) < need:
        try:
            buf += dev.handle.bulkRead(dev.EP_IMG_IN, dev.IMG_PACKET, timeout=1000)
        except usb1.USBError:
            break
    a = np.frombuffer(bytes(buf), dtype='<u2')
    mk = np.flatnonzero(a & 1)
    if len(mk) == 0:
        return None
    ph = int(np.bincount(mk % Pn, minlength=Pn).argmax())
    m = a[ph: ph + ((len(a) - ph) // Pn) * Pn]
    if len(m) < Pn:
        return None
    lines = m.reshape(-1, Pn).astype(np.float32)
    return [lines[:, c::3].mean(axis=0) for c in range(3)]      # 3 x width per-column mean


def read_ccd_max(dev, meas_pkts=24, settle_pkts=8):
    """Drain EP6 (synchronous): discard `settle_pkts` (let the new exposure take effect),
    then read `meas_pkts` and return per-interleave-phase max (phase0,phase1,phase2).
    Data is RGB interleaved 16-bit LE; max over samples[phase::3]. Phase->R/G/B mapping
    is assumed identity for v1 (logged so we can remap if a channel doesn't respond)."""
    import numpy as np
    for _ in range(settle_pkts):
        try:
            dev.handle.bulkRead(dev.EP_IMG_IN, dev.IMG_PACKET, timeout=1000)
        except usb1.USBError:
            pass
    buf = bytearray()
    for _ in range(meas_pkts):
        try:
            buf += dev.handle.bulkRead(dev.EP_IMG_IN, dev.IMG_PACKET, timeout=1000)
        except usb1.USBError:
            break
    if len(buf) < 6:
        return (0, 0, 0), 0
    a = np.frombuffer(bytes(buf), dtype='<u2')
    a = a[:len(a) // 3 * 3]
    return (int(a[0::3].max()), int(a[1::3].max()), int(a[2::3].max())), len(buf)


def servo(dev, steps, target=0xfa00, tol=0x80, max_iter=40, start_exp=0x0200, verbose=True):
    """Reactive gain/exposure calibration servo. Replays the pre-trigger
    CCD setup, triggers, lamp ON, then converges per-channel exposure so each interleave
    phase's CCD max lands in [target-? .. target] (~0xfa00, just under the 0xffff clip),
    starting LOW and ramping up (never saturates). NO motor."""
    import struct
    # pre-trigger setup (no EP6) up to and incl. the trigger
    trig_idx = next((i for i, s in enumerate(steps)
                     if is_trigger(bytes.fromhex(s['data']))), len(steps) - 1)
    prev = steps[0]['rel']
    for s in steps[:trig_idx + 1]:
        gap = s['rel'] - prev; prev = s['rel']
        if gap > 0:
            time.sleep(min(gap, 2.0))
        dev.send_raw(bytes.fromhex(s['data']))
    def arm():
        # commit pending writes + clock an exposure-controlled CCD read pass
        dev.write_reg(dev.AD_HOST, 0x84, b'\x02')
        dev.write2(dev.AD_PICL_PLUS, 0x8a)

    # a few dark-pass arm pulses (lamp off) like the OEM, then lamp ON
    for _ in range(4):
        arm(); time.sleep(0.02)
    dev.write_reg(dev.AD_PICL_PLUS, 0x80, b'\x01')
    arm()
    print("  trigger + lamp ON (arm-pulse driven); starting servo (target=0x%04x ±0x%x)" % (target, tol))

    exp = [start_exp, start_exp, start_exp]      # phase0, phase1, phase2 exposure counts
    done = [False, False, False]
    lo, hi = target - tol, min(0xffff, target)
    for it in range(max_iter):
        dev.write_reg(dev.AD_PICL_PLUS, 0x82, build_exposure(exp[0], exp[1], exp[2]))
        arm()                                     # ARM: commit exposure + clock a read pass
        (m0, m1, m2), nb = read_ccd_max(dev)
        m = [m0, m1, m2]
        for c in range(3):
            if lo <= m[c] <= hi:
                done[c] = True
            else:
                done[c] = False
                if m[c] == 0:
                    exp[c] = min(exp[c] * 2, 0xffff)
                else:
                    exp[c] = max(1, min(0xffff, int(exp[c] * target / m[c])))
        if verbose:
            print("  it%02d exp=[%5d %5d %5d] max=[%5d %5d %5d] done=%s"
                  % (it, exp[0], exp[1], exp[2], m0, m1, m2, ''.join('Y' if d else '.' for d in done)))
        if all(done):
            print("  >>> CONVERGED in %d iterations. exposures=%s" % (it + 1, exp))
            return True, exp
    print("  >>> did NOT converge in %d iterations (last exp=%s, max=%s)" % (max_iter, exp, m))
    return False, exp


def read_dx_status(dev, addr=0x40):
    """DX/film hardware status — PPB_READ_DX_CODE (reg0x90, 30-byte response) from the DX subsystem.
    (bDrvGetHardwareStatusDx). Response =
    [pos_hi, pos_lo, count, header-flags, count x 5-byte DX events]. header bits 4-7 = the 4 DX edge
    sensors (TopClock/TopData/BottomClock/BottomData); OEM end-of-film = (header & 0x30) while scanning.
    The DX scan is already started by the scan-start reg0x91=3c0001 (= PPB_START_DX_SCAN speed/format).
    Returns dict or None (best-effort read; never throws)."""
    try:
        b = bytes(dev.read_reg(addr, 0x90, 0x1e, timeout=400, retries=1))
    except Exception:                                  # noqa: BLE001
        return None
    if len(b) < 4:
        return None
    return {'pos': (b[0] << 8) | b[1], 'count': b[2], 'header': b[3],
            'exit30': bool(b[3] & 0x30), 'sensors': b[3] & 0xf0, 'raw': b[:min(8, len(b))].hex()}


def dx_get_sensors(dev, addr=0x40):
    """Read the 4 raw DX edge-sensor levels — bDrvDxGetHardware = READ reg0x93 x4
    [TopClock, TopData, BottomClock, BottomData]. Best-effort, never throws."""
    try:
        b = bytes(dev.read_reg(addr, 0x93, 4, timeout=400, retries=1))
        return list(b[:4]) if len(b) >= 4 else None
    except Exception:                                   # noqa: BLE001
        return None


def dx_set_pots(dev, pots, addr=0x40):
    """Set the 4 DX digital pots (sensor gain, 0-31) — bDrvDxChangePots = WRITE reg0x96 x4
    Pure software, no physical adjustment."""
    dev.write_reg(addr, 0x96, bytes(max(0, min(31, int(p))) for p in pots))


def dx_program_thresholds(dev, thr8, addr=0x40):
    """Program the 8 DX / no-film thresholds — bDrvDxPutHardware = WRITE reg0x94 x8."""
    dev.write_reg(addr, 0x94, bytes(max(0, min(255, int(t))) for t in thr8))


def dx_read_events(dev, addr=0x40):
    """Read the DX event block from reg0x1e — the register the OEM polls during a scan (from
    a USB capture; NOT reg0x90). read_reg strips the [01,20,40] header and returns
    [status=0x08, pos_hi, pos_lo, count, then count x 5-byte events: code, data, epos_hi, epos_lo, term=cc].
    Returns {pos, count, events:[(code,data,epos)], raw} or None."""
    try:
        b = bytes(dev.read_reg(addr, 0x1e, 0x90, timeout=400, retries=1))
    except Exception:                                   # noqa: BLE001
        return None
    if len(b) < 4:
        return None
    pos = (b[1] << 8) | b[2]
    count = b[3]
    events = []
    off = 4
    for _ in range(min(count, 8)):
        if off + 4 >= len(b):
            break
        events.append((b[off] & 0x0f, b[off + 1], (b[off + 2] << 8) | b[off + 3]))
        off += 5
    return {'pos': pos, 'count': count, 'events': events, 'raw': b[:min(20, len(b))].hex()}


def dx_commit(dev, addr=0x40):
    """Latch/commit the DX pot+threshold config — WRITE reg0x97=0xff."""
    dev.write_reg(addr, 0x97, b'\xff')


def dx_read_codes(dev, addr=0x40):
    """Read the firmware-decoded DX-code event stream — PPB_READ_DX_CODE = READ reg0x90 x30
    Response = [pos_hi, pos_lo, count, then count events x 5 bytes:
    code, data, epos_hi, epos_lo, _]. The code low-nibble (&0x0f) is the DX symbol; events carry the
    scan position. Returns {pos, count, header, events:[(code,data,epos)], raw} or None."""
    try:
        b = bytes(dev.read_reg(addr, 0x90, 0x1e, timeout=400, retries=1))
    except Exception:                                   # noqa: BLE001
        return None
    if len(b) < 4:
        return None
    pos = (b[0] << 8) | b[1]
    count = b[2]
    events = []
    off = 3
    for _ in range(min(count, 5)):
        if off + 4 >= len(b):
            break
        events.append((b[off] & 0x0f, b[off + 1], (b[off + 2] << 8) | b[off + 3]))
        off += 5
    return {'pos': pos, 'count': count, 'header': b[3], 'events': events, 'raw': b.hex()}


def _dx_codename(code):
    return {1: 'cartridgeA', 2: 'cartridgeB', 3: 'frame#', 4: 'marker', 5: 'clock',
            6: 'fatbit', 7: 'sync', 8: 'perf'}.get(code, 'idle/?')


def transport(dev, steps, out_path, seconds, max_bytes, pump, film_thresh=4000, end_gap=3.0,
              exp_servo=True, film_boost=(1.6, 3.6, 5.3), film_duty=None, ir=False, fixed_duty=None,
              dx_eof=True, dx_gap=2.0, eject_seconds=2.0, dx_calibrate=False, dx_commit=False,
              dx_codes=False, dx_pots=None, dx_log=False, on_progress=None):
    """Faithful replay of the OEM scan-start (dev17_scanstart.json: full calibration ->
    motor rate reg0xa5 -> WRITE2 [00 a0] go-forward -> reg0x91 trigger#2), then run the
    transport loop (POLL HOST flow-control + EP6 drain) for `seconds`. MOTOR RUNS + LAMP ON.
    EP6 -> out_path (capped). Teardown (master disable -> stops motor+lamp) is in main's finally.
    This is grounded in the capture, not guessed; first run is NO FILM (motor turns empty)."""
    import numpy as np
    img = {'bytes': 0, 'pkts': 0, 'mean': 0.0, 'film_seen': False, 'last_film_t': 0.0,
           'cmax': (0, 0, 0), 'writing': False, 'total': 0,
           'grab': None, 'grab_need': 0, 'og': None}
    LINE_S = 2000 * 3                                  # samples/line (3ch)

    def _emit(event, data=None):
        """Optional progress callback (on_progress(event, data)). No-op + fully
        isolated when no callback is given, so the proven scan path is unchanged."""
        if on_progress is not None:
            try:
                on_progress(event, data)
            except Exception:                          # noqa: BLE001 — never let a UI hook break a scan
                pass

    _emit('phase', {'phase': 'calibrating', 'message': 'calibration + exposure servo'})
    out = open(out_path, 'wb')
    stop = [False]
    transfers = []

    def cb(tr):
        if tr.getStatus() == usb1.TRANSFER_COMPLETED:
            n = tr.getActualLength()
            if n:
                b = bytes(tr.getBuffer()[:n])
                a = np.frombuffer(b, dtype='<u2'); a = a[:len(a) // 3 * 3]
                if len(a) >= 3:                              # per-phase max (for the exposure servo)
                    img['cmax'] = (int(a[0::3].max()), int(a[1::3].max()), int(a[2::3].max()))
                img['total'] += n                            # absolute stream position since ring open
                if img['grab'] is not None and len(img['grab']) < img['grab_need']:
                    img['grab'] += b                         # marker-aligned measurement/white capture
                if img['writing']:                           # only capture to file after the servo
                    # NO byte-alignment needed: every CCD line carries a hardware LINE-SYNC MARKER
                    # (bit0 of its first sample). pakon_decode.find_line_phase()
                    # recovers line-0 from those markers regardless of where this file starts, so the
                    # servo-drain / ring-open offset is irrelevant. Just write the stream verbatim.
                    out.write(b); img['bytes'] += n; img['pkts'] += 1
                    img['mean'] = float(a.mean()) if len(a) else 0.0
                    if img['mean'] > film_thresh:            # END-OF-FILM tracking
                        img['film_seen'] = True; img['last_film_t'] = time.monotonic()
        if not stop[0] and (not img['writing'] or img['bytes'] < max_bytes):
            try:
                tr.submit()
            except usb1.USBError:
                pass

    ep6_open = [False]

    def open_ep6():
        if ep6_open[0]:
            return
        ep6_open[0] = True
        for _ in range(16):
            tr = dev.handle.getTransfer()
            tr.setBulk(dev.EP_IMG_IN, dev.IMG_PACKET, callback=cb, timeout=0)
            tr.submit(); transfers.append(tr)

    t0 = time.monotonic()
    ntrig = [0]
    dark_ref = [None]                                  # §2b: per-column lamp-off dark reference (3,2000)
    dark_ir_ref = [None]                               # §2b (4-ch): IR-plane lamp-off dark (2000,)
    dark_done = [False]
    DARK_LINES = 128                                   # OEM bCalibrateFixedPatternDark averages 0x80 lines
    try:
        prev = steps[0]['rel']
        for s in steps:
            gap = s['rel'] - prev; prev = s['rel']
            if gap > 0:
                end = time.monotonic() + min(gap, 3.0)
                while time.monotonic() < end:
                    dev.ctx.handleEventsTimeout(pump) if ep6_open[0] else time.sleep(pump)
            pkt = bytes.fromhex(s['data'])
            t = pkt[0]; addr = pkt[2] if len(pkt) > 2 else 0
            # annotate + act on the actuating commands
            # ---- §2b DARK-FRAME PASS. The lamp is OFF until
            # this first reg0x80-enable; EP6 has been streaming dark lines since trigger#1. Grab 128 fresh
            # dark lines NOW (lamp still off), marker-align + mean per column -> Dark[col]. Saved to the
            # flatref so pakon_invert subtracts a MEASURED dark instead of self-deriving.
            #   3-ch:  trigger#1 reg0x91=0x3c.. -> first reg0x80=01 -> 6000-sample dark lines.
            #   4-ch (IR): trigger#1 reg0x91=0x31.. (idx 23, 8000-sample mode) -> first reg0x80=0x03 (idx 111,
            #     ~88 steps later, lamp off the whole window, BEFORE motor-go idx 395). So the SAME dark
            #     window exists; the lines are 8000 samples = [RGB 6000 interleaved][IR 2000]. We average per
            #     column -> dark (3,2000) [+ dark_ir (2000,)]. The OEM does this too (it fills the
            #     IR dark slot calbuf+0x2c when this+0x378/IR is set). The grab is READ-ONLY (side buffer
            #     img['grab']) and fires before the motor starts -> no actuation, cannot stick film; if it
            #     captures nothing, decode falls back to the flat SENSOR_DARK_RGB. ⚠ HW-UNVALIDATED for IR
            #     (validate the window yields ~300/ch dark 8000-sample lines on a no-film/throwaway run before
            #     trusting it on a real roll). ----
            if (not dark_done[0] and ep6_open[0] and t == 2 and len(pkt) >= 6
                    and addr == 0x40 and pkt[4] == 0x80 and pkt[5] != 0x00):
                dark_done[0] = True
                Pd = (2000 * 4) if ir else LINE_S          # 8000-sample 4-ch line vs 6000 3-ch
                img['grab'] = bytearray(); img['grab_need'] = DARK_LINES * Pd * 2
                dend = time.monotonic() + 1.5
                while len(img['grab']) < img['grab_need'] and time.monotonic() < dend:
                    try:
                        dev.poll_status(dev.AD_HOST)
                    except usb1.USBError:
                        pass
                    dev.ctx.handleEventsTimeout(pump)
                dbuf = bytes(img['grab']); img['grab'] = None
                da = np.frombuffer(dbuf, dtype='<u2'); mk = np.flatnonzero(da & 1)
                if len(mk):
                    ph = int(np.bincount(mk % Pd, minlength=Pd).argmax())
                    mm = da[ph: ph + ((len(da) - ph) // Pd) * Pd]
                    if len(mm) >= Pd:
                        ln = mm.reshape(-1, Pd).astype(np.float32)
                        dark_ref[0] = np.stack([ln[:, :6000][:, c::3].mean(axis=0) for c in range(3)])  # (3,2000)
                        if ir:
                            dark_ir_ref[0] = ln[:, 6000:8000].mean(axis=0)                              # (2000,)
                        print("  [%5.1fs] DARK ref: %d lamp-off %s lines, per-ch mean R/G/B=%.0f/%.0f/%.0f%s"
                              % (time.monotonic() - t0, ln.shape[0], "4-ch" if ir else "3-ch",
                                 dark_ref[0][0].mean(), dark_ref[0][1].mean(), dark_ref[0][2].mean(),
                                 ("  IR=%.0f" % dark_ir_ref[0].mean()) if ir else ""))
                if dark_ref[0] is None:
                    print("  [%5.1fs] DARK ref: no lamp-off lines captured -> decode uses flat SENSOR_DARK"
                          % (time.monotonic() - t0))
            if t == 2 and len(pkt) >= 5 and addr == 0x40 and pkt[4] == 0x80:
                print("  [%5.1fs] LAMP reg0x80=%s" % (time.monotonic() - t0, pkt[5:5 + pkt[3]].hex()))
            if t == 2 and len(pkt) >= 5 and addr == 0x44 and pkt[4] == 0xa5:
                print("  [%5.1fs] MOTOR rate=%s" % (time.monotonic() - t0, pkt[5:5 + pkt[3]].hex()))
            if t == 4 and len(pkt) >= 5 and addr == 0x44 and pkt[4] == 0xa0:
                print("  [%5.1fs] MOTOR GO forward" % (time.monotonic() - t0)); open_ep6()
            if t == 2 and len(pkt) >= 5 and addr == 0x40 and pkt[4] == 0x91:
                ntrig[0] += 1
                print("  [%5.1fs] TRIGGER#%d (%s)" % (time.monotonic() - t0, ntrig[0],
                      "calibration" if ntrig[0] == 1 else "TRANSPORT begins"))
                open_ep6()
            try:
                dev.send_raw(pkt)
            except usb1.USBError as e:
                print("  send error %s: %s" % (pkt.hex(), e))
        open_ep6()

        # ---- shared helpers: arm, set per-channel exposure, and a MARKER-ALIGNED grab ----
        def _arm():
            dev.write_reg(dev.AD_HOST, 0x84, b'\x02'); dev.write2(dev.AD_PICL_PLUS, 0x8a)

        def _set(e):                                       # e = [eR, eG, eB] per-channel exposure (build_exposure args)
            dev.write_reg(dev.AD_PICL_PLUS, 0x82, build_exposure(e[0], e[1], e[2])); _arm()

        def _grab(nlines, timeout=4.0):
            """Capture nlines off EP6, MARKER-ALIGN (bit0 line-sync), deinterleave -> 3 per-phase planes.
            The marker gives a STABLE phase<->channel mapping (the per-packet phase rotates, which is why
            the old global-max servo couldn't do per-channel). Returns [plane0,plane1,plane2] or None."""
            img['grab'] = bytearray(); img['grab_need'] = nlines * LINE_S * 2
            end = time.monotonic() + timeout
            while len(img['grab']) < img['grab_need'] and time.monotonic() < end:
                try:
                    dev.poll_status(dev.AD_HOST)
                except usb1.USBError:
                    pass
                dev.ctx.handleEventsTimeout(pump)
            buf = bytes(img['grab']); img['grab'] = None
            arr = np.frombuffer(buf, dtype='<u2')
            mk = np.flatnonzero(arr & 1)
            if len(mk) == 0:
                return None
            ph = int(np.bincount(mk % LINE_S, minlength=LINE_S).argmax())
            m = arr[ph: ph + ((len(arr) - ph) // LINE_S) * LINE_S]
            if len(m) < LINE_S:
                return None
            lines = m.reshape(-1, LINE_S).astype(np.float32)
            return [lines[:, c::3] for c in range(3)]      # 3 planes, each (nlines, 2000)

        def _levels(nlines=48):
            pl = _grab(nlines)
            return None if pl is None else [float(np.percentile(p, 99)) for p in pl]

        def _grab_ir(nlines, timeout=4.0):
            """READ-ONLY 4-channel (8000-sample) grab → [R,G,B] planes from the 6000 RGB block (IR is the
            trailing 2000). Used by the IR open-gate exposure servo to read RGB levels. Does NOT write regs."""
            IRP = 2000 * 4
            img['grab'] = bytearray(); img['grab_need'] = nlines * IRP * 2
            end = time.monotonic() + timeout
            while len(img['grab']) < img['grab_need'] and time.monotonic() < end:
                try:
                    dev.poll_status(dev.AD_HOST)
                except usb1.USBError:
                    pass
                dev.ctx.handleEventsTimeout(pump)
            buf = bytes(img['grab']); img['grab'] = None
            arr = np.frombuffer(buf, dtype='<u2'); mk = np.flatnonzero(arr & 1)
            if len(mk) == 0:
                return None
            ph = int(np.bincount(mk % IRP, minlength=IRP).argmax())
            m = arr[ph: ph + ((len(arr) - ph) // IRP) * IRP]
            if len(m) < IRP:
                return None
            blk = m.reshape(-1, IRP)[:, :6000].astype(np.float32)
            return [blk[:, c::3] for c in range(3)]

        def _levels_ir(nlines=48):
            pl = _grab_ir(nlines)
            return None if pl is None else [float(np.percentile(p, 99)) for p in pl]

        # ============================================================================================
        # DX DIGI-POT CALIBRATION (replicates the OEM Film Track Test).
        # The DX edge sensors (TopClock/TopData/BottomClock/BottomData) are read raw via reg0x93; their
        # gain is 4 digital pots (0-31) written via reg0x96; the OEM servos the pots so the CLEAR-film
        # (bright) reading sits at a good level, then checks the film-vs-clear voltage swing (>=0x50) and
        # programs the no-film thresholds (reg0x94). This is the one-time bring-up that makes the DX
        # film/no-film signal meaningful -> the prerequisite for OEM-faithful end-of-film + §2d-4.
        # ============================================================================================
        if dx_calibrate:
            DXA = 0x40
            def _beat():
                try:
                    dev.poll_status(dev.AD_HOST)
                except usb1.USBError:
                    pass
                dev.ctx.handleEventsTimeout(pump)
            print("\n  >>> DX DIGI-POT CALIBRATION (read reg0x93 sensors, set reg0x96 pots). RE: Film Track Test.")
            # bracket the gain range on the (currently empty/leader) gate
            dx_set_pots(dev, [0, 0, 0, 0], DXA);   [_beat() for _ in range(4)]; lo = dx_get_sensors(dev, DXA)
            dx_set_pots(dev, [31, 31, 31, 31], DXA); [_beat() for _ in range(4)]; hi = dx_get_sensors(dev, DXA)
            print("  open-gate raw DX: pots=0 -> %s   pots=31 -> %s" % (lo, hi))
            if lo is None or hi is None:
                print("  !! reg0x93 not responding on 0x%02x — DX sensors unreachable; aborting cal." % DXA)
                return 0, 0
            # servo each pot so the CLEAR (bright, no-film) reading sits near a non-saturated target,
            # then watch film go through and record the film-vs-clear swing (OEM success = swing>=0x50).
            CLEAR_TGT = 0xc0                                   # bright target, headroom below 0xff
            pots = [16, 16, 16, 16]; dx_set_pots(dev, pots, DXA); [_beat() for _ in range(3)]
            print("\n  >>> FEED FILM NOW — DX cal runs %ds (servo pots on clear gate, then film swing)\n" % seconds)
            t0c = time.monotonic(); last = 0.0
            clear_max = [0, 0, 0, 0]; film_min = [255, 255, 255, 255]; servo_done = False
            while time.monotonic() - t0c < seconds:
                _beat()
                now = time.monotonic()
                if now - last < 0.25:
                    time.sleep(0.03); continue
                last = now
                s = dx_get_sensors(dev, DXA)
                if s is None:
                    continue
                # servo phase (first ~3s, on the clear gate): pull each clear reading toward CLEAR_TGT
                if not servo_done and now - t0c < 3.0:
                    for i in range(4):
                        if s[i] > CLEAR_TGT + 8 and pots[i] > 0:
                            pots[i] -= 1
                        elif s[i] < CLEAR_TGT - 8 and pots[i] < 31:
                            pots[i] += 1
                    dx_set_pots(dev, pots, DXA)
                else:
                    servo_done = True
                    for i in range(4):                         # track film(low)/clear(high) extremes
                        clear_max[i] = max(clear_max[i], s[i]); film_min[i] = min(film_min[i], s[i])
                print("  [%5.1fs] DX raw=%s pots=%s%s" % (now - t0c, s, pots, '' if servo_done else '  (servo)'))
            swing = [clear_max[i] - film_min[i] for i in range(4)]
            ok = all(sw >= 0x50 for sw in swing)
            print("\n  >>> DX CAL RESULT: pots=%s" % pots)
            print("      clear_max=%s  film_min=%s  swing=%s  (OEM wants each >=0x50=80)" % (clear_max, film_min, swing))
            print("      verdict: %s" % ("GOOD swing on all 4 sensors" if ok else "WEAK swing — sensors need more film contrast / pot tuning"))
            if dx_commit and ok:
                # no-film thresholds = midpoint between film and clear per sensor (the OEM uses quartiles;
                # midpoint is a safe first cut). 8 values = 4 DX + 4 no-film (here: same midpoints).
                mid = [max(0, min(255, (clear_max[i] + film_min[i]) // 2)) for i in range(4)]
                dx_program_thresholds(dev, mid + mid, DXA)
                print("      >>> COMMITTED no-film thresholds reg0x94 = %s (midpoints)" % mid)
            elif dx_commit:
                print("      (skipped threshold commit — swing too weak to trust)")
            else:
                print("      (observe-only; pass --dx-commit to program reg0x94 thresholds once swing is good)")
            return 0, 0

        # ============================================================================================
        # DX-CODE CAPTURE (§2d-4). The firmware decodes the DX film-edge barcode from the edge sensors and
        # reports it via reg0x90 as a stream of [code-nibble, data, position] events.
        # We poll reg0x90 fast during transport, dedupe events by position, and log the code-nibble
        # sequence + positions (+ raw reg0x93 sensors for diagnostics). Optionally set+commit the DX pots
        # (reg0x96 + reg0x97). This is the raw DX symbol stream; decode -> frame numbers per the DX
        # film-edge standard is the next step.
        # ============================================================================================
        if dx_codes:
            DXA = 0x40
            def _beat():
                try:
                    dev.poll_status(dev.AD_HOST)
                except usb1.USBError:
                    pass
                dev.ctx.handleEventsTimeout(pump)
            if dx_pots is not None:
                try:
                    dx_set_pots(dev, dx_pots, DXA); dx_commit(dev, DXA)
                    print("  >>> DX pots set %s + committed (reg0x96/0x97)" % (list(dx_pots),))
                except Exception as e:                   # noqa: BLE001
                    print("  >>> DX pot set failed: %s" % e)
            print("\n  >>> DX-CODE CAPTURE (poll reg0x90 events + reg0x93 sensors). >>> FEED FILM NOW <<< (%ds)\n"
                  % seconds)
            seen = {}                                    # epos -> (code, data) : dedupe by position
            order = []                                   # capture order of unique events
            t0c = time.monotonic(); last_log = 0.0; last_raw = None
            while time.monotonic() - t0c < seconds:
                _beat()
                now = time.monotonic()
                c = dx_read_codes(dev, DXA)
                if c is not None and c['count'] > 0:
                    for (code, data, epos) in c['events']:
                        if epos not in seen:
                            seen[epos] = (code, data); order.append((epos, code, data))
                            print("  [%6.1fs] DX EVENT pos=%-6d code=0x%x data=0x%02x  (hdr=0x%02x cnt=%d)"
                                  % (now - t0c, epos, code, data, c['header'], c['count']))
                # periodic raw-sensor + header heartbeat (~1 Hz) so we see sensor state even with no events
                if now - last_log > 1.0:
                    last_log = now
                    s = dx_get_sensors(dev, DXA)
                    if s != last_raw:
                        last_raw = s
                        hdr = c['header'] if c else None
                        print("  [%6.1fs] DX raw=%s hdr=%s pos=%s events_so_far=%d"
                              % (now - t0c, s, ('0x%02x' % hdr) if hdr is not None else '--',
                                 c['pos'] if c else '--', len(order)))
                time.sleep(0.02)
            print("\n  >>> DX-CODE CAPTURE DONE: %d unique events" % len(order))
            if order:
                codes = ''.join('%x' % code for _, code, _ in order)
                print("  >>> code-nibble sequence (in position order):")
                print("      %s" % codes)
                print("  >>> events (pos, code, data):")
                for epos, code, data in order[:200]:
                    print("      pos=%-6d code=0x%x data=0x%02x" % (epos, code, data))
            else:
                print("  >>> NO DX events on reg0x90 — try --dx-pots 'a,b,c,d' (set+commit gain) or check threading.")
            return 0, 0

        # ---- EXPOSURE: max LED CURRENT (reg0x81 coarse) + per-channel DUTY servo (reg0x82 fine) ----
        # From the current sweep: LED brightness = current x duty; phase0=R / phase1=G /
        # phase2=B (FIXED under marker alignment, confirmed by the sweep -> no probe needed); per-channel
        # current ceilings R8/G20/B20 (G/B high BY DESIGN to push light through the orange mask). Set the
        # currents to the CEILINGS (coarse, full LED drive), then servo the DUTY (fine). At max current the
        # duty needed stays UNDER the reg0x82 wrap (~1000) — which is exactly why duty-alone (low current)
        # wrapped before and this won't.
        TARGET, DUTY_WRAP = 0xf000, 900
        CUR = (8, 20, 20)                                  # R/G/B current ceilings (model 'D')
        exp = [40, 40, 40]                                 # start low (no clip at max current), servo up
        if ir:
            # IR OPEN-GATE EXPOSURE SERVO (from the ir_scanstart packet map). PICL+
            # reg0x82 LED exposure = [duty_b, duty_ir, duty_r, 0, duty_g, base] (6×u16). The OEM scan-start
            # servos RGB duty DOWN on the STATIC open gate (the motor starts LAST), then BOOSTS to a film
            # exposure (r/g/b≈534/597/421) before the motor — so replaying the boosted values makes the
            # open-gate white ref CLIP. FIX: re-run that servo on the open gate, HOLDING the IR duty at the OEM
            # 0x01df=479 (the failed --ir-servo jammed 0x9c00 — the IR *target level*, not a duty — into this
            # field and KILLED the IR channel), capture a NON-CLIPPING white ref, then restore the film
            # exposure for the scan. base=0x0257. We only touch PICL+ reg0x82 (LED duties); the 4-channel mode
            # (reg0x80=0x03) and sub44 CCD geometry are untouched. Servo-fail -> keep OEM exposure (safe).
            # NB: raising the LED duty/target does NOT brighten the 4-channel FILM scan (tried 2026-06-25,
            # roll5: open-gate servo reached 0xf000 but the neg stayed ~half-level/29% crushed, identical to
            # roll4 — because IR mode integrates LESS light PER LINE; the real exposure lever is the IR
            # motor/line-rate, NOT reg0x82 duty). So keep the OEM-faithful values here.
            IRP = 2000 * 4
            IR_DUTY, EXP_BASE, OPEN_TGT = 0x01df, 0x0257, 0xd000
            FILM_RGB = (0x0216, 0x0255, 0x01a5)             # OEM film duty (r,g,b); RGB well-exposed on film
            exp = [0x017f, 0x00ef, 0x0053]                  # start at the OEM converged open-gate duty (r,g,b)
            print("\n  >>> IR scan: servoing RGB duty on the open gate (IR duty 0x%03x HELD, target 0x%04x)..." % (IR_DUTY, OPEN_TGT))
            servo_ok = False
            for it in range(8):
                dev.write_reg(dev.AD_PICL_PLUS, 0x82, build_exposure(exp[0], exp[1], exp[2], base=EXP_BASE, ir=IR_DUTY)); _arm()
                time.sleep(0.05)
                lv = _levels_ir()
                if lv is None:
                    print("       (no 4-ch markers — keeping OEM exposure; white-ref will clip)"); break
                servo_ok = True; done = True
                for c in range(3):
                    if not (OPEN_TGT * 0.9 <= lv[c] <= OPEN_TGT * 1.05):
                        done = False
                    exp[c] = max(1, min(DUTY_WRAP, int(exp[c] * OPEN_TGT / max(lv[c], 1.0))))
                print("       it%02d duty(r/g/b)=%4d/%4d/%4d  R/G/B=%6.0f/%6.0f/%6.0f %s" % (it, exp[0], exp[1], exp[2], lv[0], lv[1], lv[2], 'OK' if done else ''))
                if done:
                    break
            exp_open = list(exp) if servo_ok else list(FILM_RGB)
            if servo_ok:
                dev.write_reg(dev.AD_PICL_PLUS, 0x82, build_exposure(exp_open[0], exp_open[1], exp_open[2], base=EXP_BASE, ir=IR_DUTY)); _arm()
            print("  >>> capturing open-gate 4-channel WHITE reference (%s exposure)..." % ("servoed" if servo_ok else "OEM"))
            img['grab'] = bytearray(); img['grab_need'] = 224 * IRP * 2
            wend = time.monotonic() + 6.0
            while len(img['grab']) < img['grab_need'] and time.monotonic() < wend:
                try:
                    dev.poll_status(dev.AD_HOST)
                except usb1.USBError:
                    pass
                dev.ctx.handleEventsTimeout(pump)
            gbuf = bytes(img['grab']); img['grab'] = None
            try:
                arr = np.frombuffer(gbuf, dtype='<u2'); mkk = np.flatnonzero(arr & 1)
                wph = int(np.bincount(mkk % IRP, minlength=IRP).argmax())
                wl = arr[wph: wph + ((len(arr) - wph) // IRP) * IRP].reshape(-1, IRP).astype(np.float32)
                wrgb = np.stack([wl[:, :6000][:, c::3].mean(axis=0) for c in range(3)])
                wir = wl[:, 6000:8000].mean(axis=0)
                sidecar = os.path.splitext(out_path)[0] + '_flatref.npz'
                dk = {}                                     # §2b measured dark (lamp-off pass); flat fallback if absent
                if dark_ref[0] is not None:
                    dk['dark'] = dark_ref[0]
                if dark_ir_ref[0] is not None:
                    dk['dark_ir'] = dark_ir_ref[0]
                np.savez(sidecar, white=wrgb, white_ir=wir, exp_open=np.array(exp_open),
                         exp_film=np.array(FILM_RGB), ir=True, **dk)
                print("  >>> WHITE ref saved: %s (%d lines; IR mean=%.0f, RGB %.0f/%.0f/%.0f)%s%s"
                      % (sidecar, wl.shape[0], wir.mean(), wrgb[0].mean(), wrgb[1].mean(), wrgb[2].mean(),
                         "  [STILL CLIPPED — servo failed]" if wrgb.max() > 64000 else "  [non-clipped ✓]",
                         "  +DARK" if 'dark' in dk else "  (dark=flat SENSOR_DARK)"))
                print("      decode with:  pakon_decode.py %s --flatfield %s" % (out_path, sidecar))
            except Exception as e:
                print("  WARNING: IR white-ref capture failed (%s) — decode with --flatfield auto." % e)
            # restore the OEM FILM exposure for the scan (RGB boosted -> film well-exposed; IR duty HELD)
            dev.write_reg(dev.AD_PICL_PLUS, 0x82, build_exposure(FILM_RGB[0], FILM_RGB[1], FILM_RGB[2], base=EXP_BASE, ir=IR_DUTY)); _arm()
            print("  >>> restored FILM exposure r/g/b=%d/%d/%d (IR 0x%03x) for the scan" % (FILM_RGB[0], FILM_RGB[1], FILM_RGB[2], IR_DUTY))
            # END-OF-FILM og = open-gate level AT FILM exposure (matches the running img['mean'] = whole-line mean);
            # re-grab briefly. The film exposure clips the open gate (~58k), which is the validated EOF reference.
            img['grab'] = bytearray(); img['grab_need'] = 80 * IRP * 2
            wend = time.monotonic() + 3.0
            while len(img['grab']) < img['grab_need'] and time.monotonic() < wend:
                try:
                    dev.poll_status(dev.AD_HOST)
                except usb1.USBError:
                    pass
                dev.ctx.handleEventsTimeout(pump)
            ogbuf = bytes(img['grab']); img['grab'] = None
            try:
                a2 = np.frombuffer(ogbuf, dtype='<u2'); m2 = np.flatnonzero(a2 & 1)
                p2 = int(np.bincount(m2 % IRP, minlength=IRP).argmax())
                ogv = float(a2[p2: p2 + ((len(a2) - p2) // IRP) * IRP].reshape(-1, IRP).astype(np.float32).mean())
            except Exception:
                ogv = 0.0
            if ogv > 40000.0:
                img['og'] = ogv
                print("  >>> end-of-film ARMED: open-gate@film level=%.0f (image returns near it = film end)" % ogv)
            else:
                print("  >>> open-gate@film level=%.0f (<=40000) — end-of-film stays TIMED (safe)" % ogv)
        elif fixed_duty is not None:
            # GATE-NOT-EMPTY path: skip the open-gate servo (it would mis-calibrate on the leader/film)
            # and the white-ref re-capture. Set the known-good exposure from a prior empty-gate scan;
            # decode by reusing that scan's _flatref.npz. Just move the motor forward and capture.
            dev.write_reg(dev.AD_PICL_PLUS, 0x81, build_current(*CUR)); _arm()
            _set(fixed_duty)
            print("\n  >>> FIXED exposure (no servo — gate not empty): currents R8/G20/B20, duty R/G/B=%d/%d/%d"
                  % tuple(fixed_duty))
            print("  >>> reuse a prior empty-gate _flatref.npz at decode (--flatfield <prior>_flatref.npz).")
        else:
            dev.write_reg(dev.AD_PICL_PLUS, 0x81, build_current(*CUR)); _arm()
            print("\n  >>> LED currents -> ceilings reg0x81 R=%d G=%d B=%d (coarse). Servoing duty..." % CUR)
        if exp_servo and not ir and fixed_duty is None:
            for it in range(16):
                _set(exp); time.sleep(0.05)
                lv = _levels()
                if lv is None:
                    print("       (no line markers in grab — using last duty)"); break
                done = True
                for c in range(3):                          # phase c == channel c (R/G/B): fixed mapping
                    L = lv[c]
                    if not (TARGET * 0.92 <= L <= TARGET * 1.02):
                        done = False
                    exp[c] = max(1, min(DUTY_WRAP, int(exp[c] * TARGET / max(L, 1.0))))
                print("       it%02d duty(R/G/B)=%4d/%4d/%4d  R/G/B=%5.0f/%5.0f/%5.0f %s"
                      % (it, exp[0], exp[1], exp[2], lv[0], lv[1], lv[2], 'OK' if done else ''))
                if done:
                    break
            _set(exp)
            print("  >>> open-gate duty: R=%d G=%d B=%d (currents R8/G20/B20, target 0x%04x)"
                  % (exp[0], exp[1], exp[2], TARGET))

        # ---- OPEN-GATE WHITE REFERENCE for the flat-field, at the converged per-channel exposure ----
        # pakon_decode uses the per-column per-channel mean as the white reference -> gain[col]=
        # target/(white[col]-dark[col]) removes border falloff + PRNU (EP6 is raw).
        # (Skipped for IR: the 4-channel line is 8000 samples so the 3ch _grab/sidecar don't apply, and
        #  _set(exp_film) would clobber the OEM IR exposure. IR film decodes with --flatfield auto for now.)
        if not ir and fixed_duty is None:
            print("\n  >>> capturing open-gate WHITE reference for flat-field...")
            exp_open = list(exp)
            try:
                pl = _grab(256, timeout=6.0)
                if pl is None:
                    raise RuntimeError("no markers")
                white = np.stack([p.mean(axis=0) for p in pl])               # (3, 2000)

                # ---- FILM-BASE EXPOSURE (leaderless, OEM-faithful) ----
                # The orange C-41 mask attenuates G/B, so a neutral OPEN GATE leaves the FILM BASE dim in
                # G/B. The OEM uses pre-stored per-channel FILM duty cycles (DutyCycleFilm). PRINCIPLED path
                # (--film-duty, calibrated via --calc-film-duty so the film base lands at ~90% headroom):
                # use those exact duties. LEGACY path (--film-boost): a
                # fixed multiplier over the open-gate exposure (a guess — under-drives/clips per the analysis).
                # Either way the flat-field stays exact: the white ref is captured at exp_open and SCALED by
                # exp_film/exp_open at decode (PRNU is multiplicative -> exposure-independent).
                if film_duty is not None:
                    exp_film = [max(1, min(DUTY_WRAP, int(round(film_duty[c])))) for c in range(3)]
                else:
                    exp_film = [max(1, min(DUTY_WRAP, int(round(exp_open[c] * film_boost[c])))) for c in range(3)]
                sidecar = os.path.splitext(out_path)[0] + '_flatref.npz'
                extra = {'dark': dark_ref[0]} if dark_ref[0] is not None else {}   # §2b measured dark
                np.savez(sidecar, white=white, exp_open=np.array(exp_open), exp_film=np.array(exp_film),
                         cur=np.array(CUR), **extra)
                print("  >>> WHITE reference saved: %s (%d lines, per-col mean R/G/B=%.0f/%.0f/%.0f)%s"
                      % (sidecar, pl[0].shape[0], white[0].mean(), white[1].mean(), white[2].mean(),
                         "  +DARK" if dark_ref[0] is not None else "  (dark self-derived)"))
                if exp_film != exp_open:
                    _set(exp_film)
                    print("  >>> FILM exposure boost (orange-mask comp): R=%d G=%d B=%d (open-gate was %d/%d/%d)"
                          % (exp_film[0], exp_film[1], exp_film[2], exp_open[0], exp_open[1], exp_open[2]))
                # OPEN-GATE brightness at the FILM exposure = the end-of-film reference.
                # White is measured at exp_open; the empty gate at exp_film scales by exp_film/exp_open (clamped).
                og = [min(65535.0, float(white[c].mean()) * exp_film[c] / max(exp_open[c], 1)) for c in range(3)]
                img['og'] = float(np.mean(og))
                print("  >>> open-gate level @ film exposure ~%.0f (end-of-film = image returns near it)" % img['og'])
                print("      decode with:  pakon_decode.py %s --flatfield %s" % (out_path, sidecar))
            except Exception as e:
                print("  WARNING: white-reference / film-exposure setup failed (%s) — decode with --flatfield auto." % e)

        # start capturing the actual film scan from here (servo data not written to file)
        img['writing'] = True
        img['film_seen'] = False; img['last_film_t'] = 0.0; img['bytes'] = 0; img['pkts'] = 0
        img['mean'] = 65535.0           # gate is OPEN (no film yet) -> don't let a stale low mean arm gate_filled
        _emit('phase', {'phase': 'scanning', 'message': 'feed film — capturing'})
        print("\n  >>> SCAN-START DONE. >>> FEED FILM NOW <<< (film LED blinking; emulsion-in, DX-code up,")
        print("      lowest frame first). Auto-stops ~%.0fs after the film ends. Do NOT pull film while moving.\n"
              % end_gap)
        t_loop = time.monotonic()
        maxend = t_loop + seconds
        last = 0
        reason = "max-timeout"
        # ---- DX-based end-of-film (DX subsystem protocol). Default ON (dx_eof; --no-eof disables).
        # TWO mechanisms, primary + fallback:
        #  PRIMARY (OEM-faithful): reg0x90 firmware header & 0x30 = the DX-exit bit (top sensor pair).
        #    CLEAR=film present, SET=no-film/exit. This is exactly the OEM's own exit test — but it needs
        #    the DX sensors CALIBRATED (Film Track Test: reg0x94 thresholds + reg0x96 pots). On an
        #    uncalibrated unit reg0x90 is frozen at 0x3c (exit30 always set) so film-present is never seen.
        #  FALLBACK (works uncalibrated, HW-validated): the raw DX TopClock level (reg0x93 byte0) reads LOW
        #    (~40) with film and HIGH (~168) clear — same physical sensor, raw instead of firmware-decoded.
        # We auto-select: if reg0x90 ever shows film-present (calibrated/live) it is PRIMARY; else TopClock.
        # Stop = the chosen signal CLEAR/exit SUSTAINED dx_gap s after film was engaged (== i_uiNoFilmTimeOut;
        # must exceed an inter-frame clear gap). Dark-frame + safety timeout remain as backstops.
        # Set the DX pots so the TopClock fallback swings; always LOGS the sensors.
        dx_addr = 0x40
        DX_FILM, DX_CLEAR = 80, 120                       # TopClock: <80 = film present; >120 = clear/no-film
        if dx_eof and not ir:
            try:
                dx_set_pots(dev, [27, 27, 27, 27], dx_addr)  # gain that makes TopClock swing 40..168
            except Exception:                            # noqa: BLE001
                pass
        dx_trace = []
        dx_last = 0.0
        dx_film_seen = False            # DX TopClock raw level (reg0x93) saw film — LOGGING ONLY (unreliable)
        dx_clear_since = None
        dx90_film_seen = False          # OEM reg0x90 header&0x30 CLEAR = film present (only live if DX-calibrated)
        dx90_exit_since = None          # reg0x90 header&0x30 SET (DX-exit) after film was present
        # PRIMARY end-of-film (image-based, OEM green-band semantics):
        gate_filled = False             # image mean dropped INTO the film band (gate filled with film)
        bright_since = None             # image mean RETURNED to ~open-gate brightness (film-end) since this t
        ejecting_until = None           # drive-to-eject deadline: keep motor running to clear the tail past the exit
        # ---- DX-CODE LOGGING during the full scan (lamp ON + arm-pulse train = the OEM ScanPictures
        # context, from a USB capture). Program the OEM reg0x94 thresholds + enable (reg0xd2=00); the DX
        # scan (reg0x91) was already issued by the scan-start. Then poll reg0x1e for decoded code events. ----
        dx_events = {}
        dx_arm_last = 0.0
        if dx_log:
            try:
                dx_program_thresholds(dev, [0x7c, 0x7c, 0x7c, 0x7b, 0xc9, 0xc8, 0xbe, 0xb9], dx_addr)  # OEM reg0x94
                dev.write_reg(dev.AD_PICL_PLUS, 0xd2, b'\x00')                                          # enable
                print("  >>> DX-LOG ON: reg0x94 OEM thresholds + reg0xd2=00; polling reg0x1e (lamp ON, arm-pulse train).")
            except (PakonError, usb1.USBError) as e:
                print("  >>> DX-LOG setup failed: %s" % e)
        print("  >>> END-OF-FILM detection %s: PRIMARY = IMAGE returns to OPEN-GATE brightness (~%.0f) for\n"
              "      %.1fs after the gate was seen FILLED with film (OEM green-band). reg0x90\n"
              "      DX-exit also armed (only fires if DX-calibrated). TopClock = LOGGING only. Safety timeout %.0fs.\n"
              % ("ARMED" if dx_eof else "OFF (--no-eof)", img['og'] or 65535.0, dx_gap, seconds))
        while img['bytes'] < max_bytes:
            now = time.monotonic()
            if now >= maxend:
                print("  >>> SAFETY MAX-TIMEOUT (%.0fs) — stopping (raise --seconds if film wasn't done)." % seconds)
                break
            # ---- DX poll (~3 Hz) ----
            if now - dx_last > 0.3:
                dx_last = now
                _emit('progress', {'bytes': img['bytes'], 'mb': img['bytes'] >> 20,
                                   'mean': round(img['mean'], 0), 'elapsed': round(now - t_loop, 1)})
                # PRIMARY (OEM): reg0x90 firmware DX-exit bit. header&0x30 CLEAR = film present (top
                # sensors blocked); SET = no-film/exit. On an UNCALIBRATED unit reg0x90 is frozen at 0x3c
                # (exit30 always True) so film-present is never seen -> dx90_film_seen stays False and we
                # auto-fall-back to TopClock. On a calibrated unit this is the OEM-faithful signal.
                st = read_dx_status(dev, dx_addr)
                if st is not None:
                    if not st['exit30']:                 # header&0x30 clear = film present
                        dx90_film_seen = True; dx90_exit_since = None
                    elif dx90_film_seen:                 # exit after film was present
                        dx90_exit_since = dx90_exit_since or now
                # FALLBACK: TopClock raw sensor level (reg0x93)
                s = dx_get_sensors(dev, dx_addr)
                if s is not None:
                    tc = s[0]                            # TopClock = the reliable film/no-film sensor
                    if tc < DX_FILM:
                        dx_film_seen = True; dx_clear_since = None
                    elif tc > DX_CLEAR:
                        dx_clear_since = dx_clear_since or now
                    dx_trace.append((round(now - t_loop, 1), s[0]))
                    src = "reg0x90-OEM" if dx90_film_seen else "TopClock"
                    print("  [%5.1fs] EP6 %dMB mean=%5.0f | DX sensors=%s film_seen=%s [%s]%s"
                          % (now - t0, img['bytes'] >> 20, img['mean'], s, dx_film_seen or dx90_film_seen,
                             src, '  <CLEAR>' if (dx_clear_since or dx90_exit_since) else ''))
            # ---- DX-CODE logging (reg0x1e events) + arm-pulse train, during the lit scan ----
            if dx_log:
                if now - dx_arm_last > 0.14:               # ~7 Hz arm pulse (drives the DX decode SM)
                    dx_arm_last = now
                    try:
                        _arm()
                    except usb1.USBError:
                        pass
                c = dx_read_events(dev, dx_addr)            # reg0x1e
                if c:
                    for code, data, epos in c['events']:
                        if code != 0 and (epos, code) not in dx_events:
                            dx_events[(epos, code)] = data
                            print("  [%5.1fs] DX EVENT pos=%-6d code=%d (%s) data=0x%02x"
                                  % (now - t0, epos, code, _dx_codename(code), data))
            # ---- STOP conditions ----
            # (1) PRIMARY (image-based, OEM green-band): film ATTENUATES -> image mean well
            #     BELOW the open-gate level; film-end -> the gate empties and the mean RETURNS to ~open-gate
            #     (img['og'], the measured white scaled to the film exposure). Armed only AFTER the gate was
            #     seen FILLED (mean dropped into the film band) — so a clear stretch MID-roll (still far below
            #     open-gate) does NOT trip (that was the old TopClock bug). reg0x90 DX-exit also accepted when
            #     a DX-calibrated unit makes it live (it never trips on our frozen-0x3c unit).
            eof = False
            if dx_eof and ejecting_until is None:
                og_level = img['og'] or 65535.0            # measured open-gate@film; else assume saturation
                FILL_LVL = 0.80 * og_level                 # gate filled with film (mean clearly below open-gate)
                BRIGHT_LVL = 0.92 * og_level               # gate empty / film-end (mean back near open-gate)
                if img['mean'] < FILL_LVL:
                    gate_filled = True; bright_since = None
                elif gate_filled and img['mean'] >= BRIGHT_LVL:
                    bright_since = bright_since or now
                if gate_filled and bright_since is not None and (now - bright_since) > dx_gap:
                    reason = "end-of-film(open-gate return %.1fs)" % dx_gap
                    print("  >>> END OF FILM: image returned to open-gate brightness (mean %.0f >= %.0f, "
                          "sustained %.1fs)." % (img['mean'], BRIGHT_LVL, dx_gap)); eof = True
                elif dx90_film_seen and dx90_exit_since is not None and (now - dx90_exit_since) > dx_gap:
                    reason = "end-of-film(OEM reg0x90 DX-exit %.1fs)" % dx_gap
                    print("  >>> END OF FILM via OEM reg0x90 header&0x30 DX-exit (sustained %.1fs)." % dx_gap); eof = True
            # DRIVE-TO-EJECT: the sensor/gate is UPSTREAM of the film exit, so keep the motor running forward
            # `eject_seconds` more to push the tail fully out, THEN stop (instead of cutting it mid-path).
            if eof:
                _emit('phase', {'phase': 'end-of-film', 'message': 'end of film — ' + reason})
                if eject_seconds > 0:
                    _emit('phase', {'phase': 'ejecting', 'message': 'ejecting film tail'})
                    print("  >>> driving forward %.1fs to EJECT the tail before stopping..." % eject_seconds)
                    img['writing'] = False                 # don't append the empty-gate tail to the .bin
                    ejecting_until = now + eject_seconds
                else:
                    break
            if ejecting_until is not None and now >= ejecting_until:
                print("  >>> eject drive complete — stopping motor (film should be clear of the gate).")
                break
            try:
                dev.poll_status(dev.AD_HOST)            # flow-control heartbeat
            except usb1.USBError:
                pass
            dev.ctx.handleEventsTimeout(pump)
        _emit('phase', {'phase': 'stopping', 'message': 'stopping motor + teardown',
                        'bytes': img['bytes'], 'reason': reason})
        print("  >>> transport done (%s; EP6 %d B, film_seen=%s)" % (reason, img['bytes'], img['film_seen']))
        if dx_log:
            try:
                dev.write2(dev.AD_PICL_PLUS, 0x92); dev.write_reg(dev.AD_PICL_PLUS, 0xd2, b'\x01')  # DX stop+disable
            except (PakonError, usb1.USBError):
                pass
            codes = sorted({code for (_, code) in dx_events})
            print("  >>> DX-LOG: %d events; codes seen: %s (%s)"
                  % (len(dx_events), codes, ', '.join('%d=%s' % (c, _dx_codename(c)) for c in codes)))
            if dx_events:
                seq = ''.join(str(code) for _, code in sorted(dx_events.keys()))
                print("  >>> DX code sequence (position order): %s" % seq)
    finally:
        # STOP THE MOTOR PROPERLY FIRST. reg0x80=00 master-disable does NOT stop the sub44 drive;
        # rate->0 + apply + reg0xa2 (idle) does (confirmed on hardware). Do this before anything else.
        try:
            dev.write_reg(dev.AD_SUB, 0xa5, b'\x00\x00')   # motor rate = 0
            dev.write2(dev.AD_SUB, 0xa0)                    # apply (go with rate 0)
            dev.write2(dev.AD_SUB, 0xa2)                    # idle/halt
            print("  motor stopped (rate=0 + reg0xa2).")
        except Exception as e:
            print("  WARNING: motor-stop failed (%s) — POWER-CYCLE if it's still running." % e)
        stop[0] = True
        for tr in transfers:
            try:
                tr.cancel()
            except usb1.USBError:
                pass
        end = time.monotonic() + 0.3
        while time.monotonic() < end:
            dev.ctx.handleEventsTimeout(pump)
        out.close()
    return img['bytes'], img['pkts']


def run_transport_scan(dev, out_path, *, ir=False, seconds=300.0, on_progress=None,
                       scanstart_path=None, steps=None, known_only=True, max_mb=2048, pump=0.01,
                       **transport_kw):
    """Shared roll-scan entry point: build the certified scan-start, then run
    transport() to out_path. Returns (nbytes, npkts).

    Used by BOTH the CLI (main() --transport) and the psix app, so there is one
    proven path. The caller owns the device lifecycle (open/initialize/teardown)
    and the HOST-ready check, exactly as main() does — this function only does
    the scan-start + transport.

    `steps` lets a caller pass a pre-built/raw scan-start (the --replay-scanstart-raw
    escape hatch); otherwise it is built from `scanstart_path` (or the default
    dev17_scanstart.json / ir_scanstart.json). `on_progress(event, data)` receives
    'phase' and 'progress' events; omit it for the original silent behaviour.
    """
    if steps is None:
        # Scan-start from the IN-CODE sequence (no capture file). DEFAULT (known_only) => send only the
        # surely-known DET packets; the live servo produces exposure. HW-validated on a no-film run AND a
        # real roll. Escape hatch PSIX_SCANSTART_FULL=1 sends the full captured sequence (DET + the reactive
        # servo-seeds) for diagnostics. scanstart_path = debug json override.
        ko = known_only and os.environ.get('PSIX_SCANSTART_FULL') != '1'
        if ko:
            print("  >>> SCAN-START: deterministic packets only (%s); reactive exposure live from servo."
                  % ("IR" if ir else "3ch"))
        else:
            print("  >>> SCAN-START: full captured sequence (%s) incl. reactive servo-seeds (PSIX_SCANSTART_FULL)."
                  % ("IR" if ir else "3ch"))
        steps = pakon_scanstart.build_scanstart_steps(ir=ir, known_only=ko, capture_path=scanstart_path)
    return transport(dev, steps, out_path, seconds, max_mb << 20, pump,
                     ir=ir, on_progress=on_progress, **transport_kw)
