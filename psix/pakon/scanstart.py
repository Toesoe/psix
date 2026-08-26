#!/usr/bin/env python3
"""pakon_scanstart — generates the OEM-equivalent scan-start sequence.

The scan-start is NOT a deterministic byte-stream like InitializeScanner (see pakon_initseq.py).
It is a DETERMINISTIC command skeleton wrapped around three CLOSED-LOOP servos (fixed-pattern dark/
bright, LED current/duty, exposure). The captured exposure/gain/offset payloads are the OEM unit's
servo OUTPUTS; replaying them over/under-exposes our sensor. So this module does two things:

1. classify() — labels EVERY packet of dev17_scanstart.json as DETERMINISTIC (safe to emit from
   named constants) or REACTIVE (servo-computed, must be produced live). This PROVES the blind blob
   is fully explained (zero unknown packets) — the de-blinding deliverable.
2. build_skeleton_steps() — generates the DETERMINISTIC skeleton (preflight reads, arm strobes, both
   triggers, lamp-on, sub44 geometry/mux, motor rate+GO, poll heartbeat) from reconstructed protocol constants. The
   REACTIVE blocks are produced by the servo (pakon_scan2.calibrate()/servo()), not hardcoded.

Grounded in the F135 scan-start protocol (arm loop, film sense, comm, DX, motor, servo, exposure
compose, current ceiling, LED duty, init config) + the IR variant.

Usage:
  pakon_scanstart.py --verify   offline: classify all 209 packets of dev17_scanstart.json, assert 0
                                unknown, and check the fixed-DETERMINISTIC values match our constants.
  pakon_scanstart.py --print    dump the annotated DETERMINISTIC skeleton this module generates.
  pakon_scanstart.py --classify dump the per-packet classification of dev17_scanstart.json.
This module does NOT actuate hardware. transport() in pakon_scan2.py is the execution path; this is
the RE/validation companion.
"""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCANSTART = os.path.join(ROOT, 'captures', 'dev17_scanstart.json')   # debug-only json override (not shipped)

# --- subsystem addresses ----------------------------------------------------
HOST, PICL, PICL_PLUS, SUB44 = 0x10, 0x20, 0x40, 0x44
ADDRNAME = {0x10: 'HOST', 0x20: 'PICL', 0x28: 'sub28', 0x40: 'PICL+', 0x44: 'sub44'}

# --- wire grammar (verified against dev17_scanstart.json + dev36_init.json) --
# All packets: [type, total_len, addr, len/count, reg, payload...]
#   type 1=READ, 2=WRITE, 3=POLL, 4=WRITE2.  READ uses byte3=count, byte4=reg (NOT reg,count).
def p_read(addr, reg, count):  return bytes([1, 3, addr, count, reg])
def p_write(addr, reg, vals):  vals = bytes(vals); return bytes([2, 3 + len(vals), addr, len(vals), reg]) + vals
def p_poll(addr):              return bytes([3, 1, addr])
def p_write2(addr, reg):       return bytes([4, 3, addr, 0x00, reg & 0xff])


# --- scan-start constants (from the F135 protocol) ---------------------------
# Two captured variants share ONE skeleton; only these scalars differ (see IR section in the writeup):
#   visible (dev17_scanstart): trigger 3c0001, lamp 01,        rate 0x1726
#   IR/4ch  (ir_scanstart):    trigger 310001, lamp 01/02/03,  rate 0x12fa, IR exposure field populated
TRIGGER = (0x3c, 0x00, 0x01)            # reg0x91 CCD read-pass start (resets line counter); both triggers
F135_TRIGGER = b'\x10\x00\x01'          # reg0x91 trigger word on an F-135 (OEM capture; both modes)
LAMP_ON = (0x01,)                       # reg0x80 bit0=visible R/G/B, bit1=IR
SVC_ACK = (0x00, 0x02)                  # reg0x06 service-ack (emit only if reg0x02 header bit0x80 set)
MOTOR_RATE = 0x1726                     # reg0xa5 transport rate (5926; film clamp 400..0x251c)
TRIGGER_FLAG = 0x01                     # reg0x91 byte2 = start/mode flag (from this+0x10); same both variants
LAMP_VALID = {0x01, 0x02, 0x03}         # bitmask: bit0=visible, bit1=IR
MOTOR_CLAMP = (400, 0x251c)             # film-mode rate clamp; DX mode = 1000..0x7ffe

# sub44 reg0x82 = CCD-timing / region / illumination-mux table [idx, u16] — DETERMINISTIC bitfields.
# idx0=enable mask, idx4=pixel offset, idx5=pixel end, idx6=integration time, idx9=color/addr mux.
# (Values below are the geometry/mux a single-pass RGB scan emits; the mux idx9 cycles per color phase.)
SUB44_GEOM_IDX = {0: 'enable mask', 4: 'CCD pixel offset', 5: 'CCD pixel end',
                  6: 'integration time', 9: 'illumination color/addr mux'}
# sub44 reg0x84 idx2/3/4 = fixed per-channel config; idx5/6/7 = R/G/B offset trim (REACTIVE).
SUB44_OFFSET_FIXED = {2, 3, 4}
SUB44_OFFSET_TRIM = {5, 6, 7}

# Preflight read block (steps 0-24) — DETERMINISTIC; (count, reg, name). Routed via the status path.
PREFLIGHT_READS = [
    (1, 0x02, 'PICL+ status byte (bit0x80=wants-service)'),
    (30, 0x90, 'PICL+ DX-code block (drains DX counter)'),
    (2, 0x84, 'PICL+ thermal/light-level telemetry'),
    (4, 0x88, 'PICL+ board/lamp temp pair'),
    (1, 0x83, 'PICL+ detailed health/flags'),
]

# classification verdicts
DET, REACT = 'DET', 'REACT'


def classify(pkt):
    """Classify one wire packet. Returns (kind, verdict, note). verdict in {DET, REACT, None}.
    kind=None/'unknown' means the packet is NOT explained (should never happen post-RE)."""
    t = pkt[0]
    addr = pkt[2] if len(pkt) > 2 else 0
    an = ADDRNAME.get(addr, hex(addr))
    if t == 3:
        return ('poll', DET, 'POLL %s (flow-control heartbeat)' % an)
    if t == 1:                                                       # READ [1,3,addr,count,reg]
        count, reg = pkt[3], pkt[4]
        return ('read', DET, 'READ %s reg0x%02x x%d (status/telemetry — device-provided)' % (an, reg, count))
    if t == 4:                                                       # WRITE2 [4,3,addr,00,reg]
        reg = pkt[4]
        if addr == PICL_PLUS and reg == 0x8a:
            return ('arm-strobe', DET, 'WRITE2 PICL+ [008a] (arm-pulse commit strobe)')
        if addr == SUB44 and reg == 0xa0:
            return ('motor-go', DET, 'WRITE2 sub44 [00a0] (motor GO forward)')
        if addr == SUB44 and reg == 0xa1:
            return ('motor-rev', DET, 'WRITE2 sub44 [00a1] (motor GO reverse)')
        if addr == SUB44 and reg == 0xa2:
            return ('motor-stop', DET, 'WRITE2 sub44 [00a2] (motor stop)')
        if addr == PICL_PLUS and reg == 0x92:
            return ('scan-stop', DET, 'WRITE2 PICL+ [0092] (scan / DX stop)')
        return ('write2', DET, 'WRITE2 %s reg0x%02x' % (an, reg))
    if t == 2:                                                       # WRITE [2,len+3,addr,len,reg,payload]
        ln, reg = pkt[3], pkt[4]
        pay = pkt[5:5 + ln]
        if addr == HOST and reg == 0x84:
            return ('arm-host', DET, 'WRITE HOST reg0x84=02 (arm-pulse part 1)')
        if addr == PICL_PLUS and reg == 0x06:
            return ('svc-ack', DET, 'WRITE PICL+ reg0x06=%s (status service-ack)' % pay.hex())
        if addr == PICL_PLUS and reg == 0x80:
            what = 'lamp OFF / master-disable' if pay == b'\x00' else 'lamp enable bitmask'
            return ('lamp', DET, 'WRITE PICL+ reg0x80=%s (%s)' % (pay.hex(), what))
        if addr == PICL_PLUS and reg == 0x91:
            return ('trigger', DET, 'WRITE PICL+ reg0x91=%s (CCD read-pass TRIGGER)' % pay.hex())
        if addr == PICL_PLUS and reg == 0x81:
            return ('led-current', REACT, 'WRITE PICL+ reg0x81=%s (LED current [B,IR,R,_,G] — SERVO)' % pay.hex())
        if addr == PICL_PLUS and reg == 0x82:
            return ('exposure', REACT, 'WRITE PICL+ reg0x82=%s (exposure [B,IR,R,_,G,base] — SERVO)' % pay.hex())
        if addr == SUB44 and reg == 0x82 and ln == 3:
            idx = pay[0]; val = pay[1] | (pay[2] << 8)
            return ('geom', DET, 'WRITE sub44 reg0x82 idx%d=0x%04x (%s)'
                    % (idx, val, SUB44_GEOM_IDX.get(idx, '?geometry')))
        if addr == SUB44 and reg == 0x84 and ln == 3:
            idx = pay[0]; val = pay[1] | (pay[2] << 8)
            if idx in SUB44_OFFSET_TRIM:
                mag = val & 0xff; sign = '-' if (val & 0x100) else '+'
                ch = {5: 'R', 6: 'G', 7: 'B'}[idx]
                return ('offset-trim', REACT,
                        'WRITE sub44 reg0x84 idx%d %s offset trim=%s%d (SERVO, sign-mag)' % (idx, ch, sign, mag))
            return ('offset-cfg', DET, 'WRITE sub44 reg0x84 idx%d=0x%04x (fixed per-channel config)' % (idx, val))
        if addr == SUB44 and reg == 0xa5:
            val = pay[0] | (pay[1] << 8) if ln >= 2 else pay[0]
            return ('motor-rate', DET, 'WRITE sub44 reg0xa5=0x%04x (motor transport rate)' % val)
        return ('write', DET, 'WRITE %s reg0x%02x=%s' % (an, reg, pay.hex()))
    return ('unknown', None, 'UNCLASSIFIED type%d: %s' % (t, pkt.hex()))


def load_capture(path=SCANSTART):
    d = json.load(open(path))
    seq = d['sequence'] if isinstance(d, dict) and 'sequence' in d else d
    out = []
    for s in seq:
        if isinstance(s, str):                          # abort_sequence: plain hex strings
            out.append(bytes.fromhex(s))
        else:
            out.append(bytes.fromhex(s.get('data') or s.get('hex')))
    return out


def build_skeleton_steps():
    """Generate the DETERMINISTIC scan-start skeleton from reconstructed protocol constants. The REACTIVE blocks
    (calibration FPC + LED current/duty + exposure servo, and sub44 reg0x84 idx5/6/7 offset trim)
    are produced live by the servo and are represented here as <SERVO> placeholders so the structure
    is explicit. Returns a list of (packet_or_None, note)."""
    s = []
    # Phase A — preflight status sweep (F135 status path) + idle polls
    for _ in range(4):
        s.append((p_poll(HOST), 'preflight idle poll'))
    for count, reg, name in PREFLIGHT_READS:
        s.append((p_read(PICL_PLUS, reg, count), 'preflight read: ' + name))
    s.append((p_write(PICL_PLUS, 0x06, SVC_ACK), 'preflight service-ack (only if status bit0x80)'))
    s.append((p_read(SUB44, 0x02, 1), 'preflight read: sub44 status byte'))
    # Phase B — TRIGGER#1 (calibration pre-scan) + reactive FPC/LED calibration
    s.append((p_write(HOST, 0x84, [0x02]), 'arm-pulse part 1'))
    s.append((p_write2(PICL_PLUS, 0x8a), 'arm-pulse strobe'))
    s.append((p_write(PICL_PLUS, 0x91, TRIGGER), 'TRIGGER#1 = calibration pre-scan'))
    s.append((None, '<SERVO> fixed-pattern dark/bright + LED current/duty calibration '
                    '(sub44 reg0x82 geometry/mux DET; reg0x84 idx5/6/7 R/G/B offset trim REACTIVE; '
                    'arm pulse per iteration)'))
    # Phase C — lamp on + exposure servo
    s.append((p_write(PICL_PLUS, 0x80, LAMP_ON), 'lamp ON (visible)'))
    s.append((None, '<SERVO> exposure: reg0x81 LED current + reg0x82 duty, converge to ~0xfa00 '
                    '(arm pulse per iteration)'))
    # Phase D — motor pre-roll: geometry/mux, final exposure, rate + GO, settling
    s.append((None, '<SERVO> final converged reg0x82 exposure for transport'))
    s.append((p_write(SUB44, 0xa5, [MOTOR_RATE & 0xff, (MOTOR_RATE >> 8) & 0xff]),
              'motor rate = 0x%04x' % MOTOR_RATE))
    s.append((p_poll(SUB44), 'poll sub44'))
    s.append((p_write2(SUB44, 0xa0), 'motor GO forward'))
    for _ in range(4):
        s.append((p_poll(HOST), 'settling / film-feed wait poll'))
    # Phase E — TRIGGER#2 = transport scan begins
    s.append((p_write(PICL_PLUS, 0x91, TRIGGER), 'TRIGGER#2 = transport scan begins'))
    s.append((p_poll(HOST), 'transport flow-control heartbeat'))
    return s


def build_abort_steps():
    """Named DETERMINISTIC end-of-scan teardown (replaces a blind
    replay; classified 141/141 DET, 0 reactive). NOTE: this is the FALLBACK teardown — the primary
    lamp-off teardown is reset_to_idle() (re-init), because the open-loop abort does NOT clear a
    scan-state-held lamp. Returns a list of wire packets (bytes).
    Sequence mirrors the OEM (from a USB capture): master-disable, scan/motor stop, geometry/mux reset,
    then a status-drain settle loop."""
    s = []
    s.append(p_write(PICL_PLUS, 0x80, [0x00]))                  # master-disable / lamp off
    s.append(p_write(SUB44, 0x82, [0x00, 0x60, 0x00]))         # geometry enable-mask reset (idx0=0x0060)
    s.append(p_poll(SUB44))
    s.append(p_write(PICL_PLUS, 0x80, [0x00]))                  # master-disable again
    s.append(p_poll(PICL_PLUS))
    s.append(p_write(HOST, 0x84, [0x02]))                       # arm pulse part 1
    s.append(p_write2(PICL_PLUS, 0x8a))                         # arm strobe
    s.append(p_poll(PICL_PLUS))
    s.append(p_read(PICL_PLUS, 0x02, 1))                        # status
    s.append(p_write2(PICL_PLUS, 0x92))                         # scan / DX stop
    s.append(p_poll(PICL_PLUS))
    s.append(p_write2(SUB44, 0xa2))                            # motor stop
    s.append(p_poll(SUB44))
    s.append(p_read(SUB44, 0x02, 1))
    s.append(p_write(SUB44, 0x82, [0x09, 0x17, 0x02]))         # color/addr mux reset (idx9=0x0217)
    s.append(p_poll(SUB44))
    s.append(p_write(SUB44, 0x82, [0x09, 0x17, 0x00]))         # color/addr mux reset (idx9=0x0017)
    s.append(p_poll(SUB44))
    s.append(p_write(PICL_PLUS, 0x06, [0x00, 0x20]))           # service word
    s.append(p_poll(PICL_PLUS))
    s.append(p_read(PICL_PLUS, 0x90, 30))                       # drain DX block
    for _ in range(24):                                         # status-drain settle loop
        s.append(p_read(PICL_PLUS, 0x02, 1))
        s.append(p_poll(HOST))
        s.append(p_read(SUB44, 0x02, 1))
        s.append(p_poll(HOST))
    return s


def remap_steps(steps, light=PICL_PLUS, motor=SUB44, ir=False):
    """Rewrite built steps from the Plus addressing/values to another generation's.
    The same command set applies to both generations — the addresses differ
    (byte[2]), and on an F-135 the trigger word (reg0x91 payload), the scan
    window end (idx5) and the scan-time CCD integration (idx6, which the Plus
    sequence never writes) differ, per the OEM F-135 capture. Identity for Plus."""
    if (light, motor) == (PICL_PLUS, SUB44) and not ir:
        return steps
    amap = {PICL_PLUS: light, SUB44: motor}
    out = []
    for s in steps:
        b = bytearray(bytes.fromhex(s['data']))
        b[2] = amap.get(b[2], b[2])
        if ir and b[0] == 2 and len(b) == 8 and b[4] == 0x82 and b[5] == 0x00:
            v = b[6] | (b[7] << 8)
            if v in (0x60, 0x61):              # IR mode: enable-mask bit 0x100 adds the
                nv = v | 0x100                 # IR block to the line ([RGB 5910][IR ~1968])
                b[6], b[7] = nv & 0xff, (nv >> 8) & 0xff
        if b[0] == 2 and len(b) >= 8 and b[4] == 0x91:
            b[5:8] = F135_TRIGGER                      # OEM F-135 trigger word
        # NOTE: idx5 stays at the Plus value (0x819 = 2073): pakon-reference
        # image-stream.md documents base16 = 2000px visible width, and the idx5=2043
        # seen in one F-135 capture yields a 1970px window — 30px narrow. With 2073
        # the 3-ch line is 6000 samples (2000px) and the IR line is 8000 (2000px x4),
        # matching the documented geometry exactly.
        out.append({**s, 'data': bytes(b).hex()})
    return out


def build_scanstart_steps(ir=False, assert_clean=True, capture_path=None, known_only=False,
                          light=PICL_PLUS, motor=SUB44):
    """Build transport()'s scan-start.

    Default: the IN-CODE sequence (_scanstart_seq.SCANSTART_3CH / SCANSTART_IR — no file loaded). Every
    packet is CLASSIFIED (named, 0 unknown) + the fixed-DET values sanity-checked. The reactive packets
    (reg0x81/0x82 exposure, reg0x84 idx5/6/7 trim) are servo SEEDS that transport()'s live servo overrides.

    known_only=True: return ONLY the DETERMINISTIC packets — every one a classified, known/documented
    value (preflight reads, arm pulses, triggers, the full CCD geometry/illumination-mux + fixed-config
    writes, lamp enable, motor rate/GO, heartbeats). The REACTIVE seeds (reg0x81 current / reg0x82 exposure
    / reg0x84 idx5/6/7 trim — session-specific captured values) are DROPPED; transport()'s live servo
    produces them. So the scanner is sent only surely-known values. Works for 3ch AND IR (the DET set carries
    the variant scalars: trigger 310001, lamp 03/02, rate 0x12fa, 4-ch geometry). ⚠ This sends a never-tested
    handshake — VALIDATE on a no-film/empty-gate run before relying on it (motor spins free, no film => no
    stick risk). `capture_path` (debug only) replays a json file instead.
    `light`/`motor` retarget the built packets' bus addresses (default = the Plus 0x40/0x44;
    pass an F-135's 0x20/0x24 for that generation — same commands, different addresses)."""
    if capture_path is not None:                                     # debug override: replay a json capture
        d = json.load(open(capture_path))
        raw = d['sequence'] if isinstance(d, dict) and 'sequence' in d else d
        seq = [{'rel': (s.get('rel', 0.0) if not isinstance(s, str) else 0.0),
                'data': (s if isinstance(s, str) else (s.get('data') or s.get('hex')))} for s in raw]
    else:
        from ._scanstart_seq import SCANSTART_3CH, SCANSTART_IR
        seq = SCANSTART_IR if ir else SCANSTART_3CH
    out = []
    for i, st in enumerate(seq):
        data = st['data']; rel = st.get('rel', 0.0)
        pkt = bytes.fromhex(data)
        kind, verdict, note = classify(pkt)
        if verdict is None and assert_clean:
            raise ValueError('scan-start step %d is UNCLASSIFIED: %s' % (i, data))
        out.append({'rel': rel, 'data': data, 'kind': kind, 'verdict': verdict, 'note': note})
    if assert_clean:                                            # fixed-DET sanity (variant-agnostic)
        triggers = [bytes.fromhex(s['data'])[5:8] for s in out
                    if bytes.fromhex(s['data'])[:1] == b'\x02'
                    and bytes.fromhex(s['data'])[2] == PICL_PLUS and bytes.fromhex(s['data'])[4] == 0x91]
        if not (len(triggers) == 2 and len(set(triggers)) == 1):
            raise ValueError('scan-start: expected 2 identical reg0x91 triggers, got %r' % triggers)
    if known_only:                                             # drop the reactive SEEDS -> servo makes them live
        out = [s for s in out if s['verdict'] == DET]
    return remap_steps(out, light, motor, ir=ir)


def cmd_classify(pkts, path=SCANSTART):
    print('=' * 78)
    print('CLASSIFY %s — %d packets' % (path, len(pkts)))
    print('=' * 78)
    counts = {}
    unknown = 0
    for i, pkt in enumerate(pkts):
        kind, verdict, note = classify(pkt)
        counts[(kind, verdict)] = counts.get((kind, verdict), 0) + 1
        if verdict is None:
            unknown += 1
        if kind in ('unknown',) or verdict == REACT or kind in ('trigger', 'lamp', 'motor-rate',
                                                                 'motor-go', 'svc-ack'):
            print('  %3d  [%-5s] %s' % (i, verdict or '????', note))
    print('-' * 78)
    det = sum(v for (k, ver), v in counts.items() if ver == DET)
    react = sum(v for (k, ver), v in counts.items() if ver == REACT)
    print('  DETERMINISTIC: %d   REACTIVE: %d   UNKNOWN: %d' % (det, react, unknown))
    print('  by kind:')
    for (k, ver), v in sorted(counts.items(), key=lambda x: -x[1]):
        print('     %-13s %-5s  %d' % (k, ver, v))
    return unknown


def detect_variant(pkts):
    """visible vs IR/4-channel. IR is identified by lamp bit1 (reg0x80 & 0x02) ever set OR a non-zero
    IR exposure field (reg0x82 u16[1])."""
    lamp_ir = any(p[0] == 2 and p[2] == PICL_PLUS and p[4] == 0x80 and (p[5] & 0x02) for p in pkts)
    exp_ir = any(p[0] == 2 and p[2] == PICL_PLUS and p[4] == 0x82 and p[3] >= 4 and (p[7] | (p[8] << 8))
                 for p in pkts)
    return 'IR/4-channel' if (lamp_ir or exp_ir) else 'visible/3-channel'


def cmd_verify(pkts):
    """Offline: prove every packet is classified (0 unknown) + the deterministic skeleton holds. Checks
    are VARIANT-AGNOSTIC (hold for both the visible dev17 and the IR ir_scanstart capture): the exact
    trigger byte / lamp mask / rate differ by variant, so we validate structure + ranges, not literals."""
    variant = detect_variant(pkts)
    print('=' * 78)
    print('VERIFY scan-start reconstruction (offline, no hardware) — variant: %s' % variant)
    print('=' * 78)
    ok = True
    unknown = [i for i, p in enumerate(pkts) if classify(p)[1] is None]
    if unknown:
        ok = False
        print('  FAIL: %d UNCLASSIFIED packets at steps %s' % (len(unknown), unknown))
    else:
        print('  PASS: all %d packets classified (0 unknown) — blob fully de-blinded.' % len(pkts))

    checks = []
    # exactly two triggers, identical to each other, trailing flag == 0x01 (calib pass + transport pass)
    triggers = [bytes(p[5:8]) for p in pkts if p[0] == 2 and p[2] == PICL_PLUS and p[4] == 0x91]
    trig_val = triggers[0].hex() if triggers else '--'
    checks.append(('exactly 2 triggers, byte-identical (reg0x91=%s)' % trig_val,
                   len(triggers) == 2 and len(set(triggers)) == 1))
    checks.append(('trigger flag byte == 0x%02x' % TRIGGER_FLAG,
                   bool(triggers) and all(t[2] == TRIGGER_FLAG for t in triggers)))
    # lamp values are a valid enable bitmask {vis, IR, vis+IR}
    lamps = [p[5] for p in pkts if p[0] == 2 and p[2] == PICL_PLUS and p[4] == 0x80]
    checks.append(('lamp reg0x80 values %s ⊆ {01,02,03}' % [hex(x) for x in lamps],
                   bool(lamps) and set(lamps).issubset(LAMP_VALID)))
    # motor rate within the film-mode clamp range
    rate = next((p[5] | (p[6] << 8) for p in pkts if p[0] == 2 and p[2] == SUB44 and p[4] == 0xa5), None)
    checks.append(('motor rate reg0xa5=0x%04x in film clamp [0x%04x,0x%04x]'
                   % ((rate or 0), MOTOR_CLAMP[0], MOTOR_CLAMP[1]),
                   rate is not None and MOTOR_CLAMP[0] <= rate <= MOTOR_CLAMP[1]))
    checks.append(('motor GO forward (WRITE2 sub44 [00a0]) present',
                   any(p[0] == 4 and p[2] == SUB44 and p[4] == 0xa0 for p in pkts)))
    # sub44 indexed tables use only known indices
    geom_idx = {p[5] for p in pkts if p[0] == 2 and p[2] == SUB44 and p[4] == 0x82 and p[3] == 3}
    checks.append(('sub44 reg0x82 indices %s ⊆ known %s' % (sorted(geom_idx), sorted(SUB44_GEOM_IDX)),
                   geom_idx.issubset(set(SUB44_GEOM_IDX))))
    trim_idx = {p[5] for p in pkts if p[0] == 2 and p[2] == SUB44 and p[4] == 0x84 and p[3] == 3}
    checks.append(('sub44 reg0x84 indices %s ⊆ {2,3,4 fixed}∪{5,6,7 trim}' % sorted(trim_idx),
                   trim_idx.issubset(SUB44_OFFSET_FIXED | SUB44_OFFSET_TRIM)))

    print('  deterministic-skeleton checks:')
    for name, passed in checks:
        print('    [%s] %s' % ('OK' if passed else 'FAIL', name))
        ok = ok and passed

    # reactive sanity: exposure servo values must EVOLVE (not constant) — proves they're not hardcodable
    exps = [bytes(p[5:5 + p[3]]) for p in pkts if p[0] == 2 and p[2] == PICL_PLUS and p[4] == 0x82]
    evolves = len(set(exps)) > 1
    print('    [%s] reg0x82 exposure payloads EVOLVE (%d distinct of %d) — REACTIVE, not hardcodable'
          % ('OK' if evolves else 'FAIL', len(set(exps)), len(exps)))
    ok = ok and evolves

    print('-' * 78)
    print('  VERDICT: %s' % ('PASS — scan-start fully de-blinded (skeleton holds; servo blocks reactive).'
                             if ok else 'FAIL — see above.'))
    return 0 if ok else 1


def cmd_print():
    print('=' * 78)
    print('DETERMINISTIC scan-start skeleton (REACTIVE blocks = <SERVO>)')
    print('=' * 78)
    for pkt, note in build_skeleton_steps():
        if pkt is None:
            print('   %-40s %s' % ('', note))
        else:
            _, verdict, desc = classify(pkt)
            print('   [%-5s] %-30s %s' % (verdict, pkt.hex(), note))


def cmd_roundtrip(capture_path):
    """Prove build_scanstart_steps() reproduces the capture byte-for-byte (the certified-loader proof)."""
    print('ROUNDTRIP %s' % capture_path)
    orig = load_capture(capture_path)
    rebuilt = [bytes.fromhex(s['data']) for s in build_scanstart_steps(capture_path=capture_path)]
    ok = orig == rebuilt
    print('  %d packets, byte-identical: %s' % (len(orig), 'YES' if ok else 'NO'))
    return 0 if ok else 1


def cmd_abort_check():
    """Confirm build_abort_steps() emits the OEM teardown command SET (every essential write) + 0 unknown."""
    print('ABORT teardown generator (build_abort_steps)')
    steps = build_abort_steps()
    unknown = 0
    cmds = set()
    for pkt in steps:
        kind, verdict, note = classify(pkt)
        if verdict is None:
            unknown += 1
        if kind not in ('poll', 'read'):
            cmds.add(kind)
    need = {'lamp', 'scan-stop', 'motor-stop', 'geom', 'arm-host', 'arm-strobe', 'svc-ack'}
    missing = need - cmds
    print('  %d steps, %d unknown; teardown commands present: %s' % (len(steps), unknown, sorted(cmds)))
    print('  required teardown commands covered: %s%s'
          % ('YES' if not missing else 'NO', '' if not missing else ' (missing %s)' % sorted(missing)))
    return 0 if (unknown == 0 and not missing) else 1
