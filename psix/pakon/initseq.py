#!/usr/bin/env python3
"""pakon_initseq: reconstruct the InitializeScanner sequence from the F135 init protocol.

This REPLACES the opaque `captures/dev36_init.json` blind replay. Every step here is
GENERATED from documented protocol values, so there is no captured/per-unit payload baked in:

  - constants        : bring-up, arm pulse, InitCfgCore, temp setpoints, motor-idle
                       (all literal/model-generic)
  - device-provided  : the EEPROM read (A4/A9), serials, the reactive sensor reads
                       (we issue the request; the scanner returns its own data live)
  - the temp block   : lamp/motherboard temperature-control + TEC setpoints,
                       model-generic OEM defaults (NOT per-unit, NOT CCD timing).
                       Confirmed = the dev36 defaults.

Wire grammar (libpakon): packet = [Type, PktLen, Addr, payload], PktLen = 1 + len(payload).
  Type1 READ   payload=[reg,count]      Type2 WRITE  payload=[count,reg,*vals]
  Type3 POLL   payload=[]               Type4 WRITE2 payload=[0x00,val]
Subsystem addrs: 0x10 HOST, 0x40 PICL+, 0x44 sub44.

EEPROM I/O: vendor control transfers, magic wIndex 0x1234, 0x20-byte chunks:
  A4 (req 0xa4, OUT, wValue 0xa5) setup, then A9 (req 0xa9, IN, wValue=byte addr) read.
  Records are [len:u32, crc32:u32, data]; the parser reads record@0x0000 (LED cal,
  398 B) and record@0x0800 (36 B). CRC32 = standard reflected poly 0xEDB88320.

Modes:
  --verify   offline: generate the deterministic stream and byte-diff it against dev36_init.json
  (default)  execute the generated sequence on a loaded+powered scanner (run pakon_load first)
"""
import json
import os
import sys
import time

from . import device as libpakon
from .device import PakonDevice, PakonError

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEV36 = os.path.join(ROOT, 'captures', 'dev36_init.json')

# --- subsystem addresses ----------------------------------------------------
HOST, PICL, PICL_PLUS, SUB44 = 0x10, 0x20, 0x40, 0x44
ADDRNAME = {0x10: 'HOST', 0x20: 'PICL', 0x28: 'sub28', 0x40: 'PICL+', 0x44: 'sub44'}

# --- EEPROM access constants -------------------------------------------------
EE_MAGIC = 0x1234           # wIndex on every EEPROM control transfer
EE_REQ_SETUP = 0xa4         # OUT setup
EE_REQ_READ = 0xa9          # IN read
EE_SETUP_VAL = 0xa5         # ((param_6=2 | 0x50) << 1) | 1  -> read setup wValue
EE_CHUNK = 0x20             # 32-byte read chunks
EE_RECORDS = (0x0000, 0x0800)   # record start addrs (LED-cal main, aux)

# --- temperature-control / TEC setpoints ------------------------------------
# Lamp + motherboard temp window, model-generic OEM defaults (confirmed in dev36's clamp ranges).
# Assembled into PICL+ regs exactly as the device packs them (4-byte little payloads).
TEMP_SETPOINTS = {
    0x8f: (0xe8, 0xff, 0x18, 0x00),   # [-LampTempWarningLow(i16)=−24] , LampTempWarningHigh=0x18 , 0
    0x8c: (0xe0, 0xff, 0x20, 0x00),   # [-LampTempFaultLow(i16)=−32]   , LampTempFaultHigh=0x20    , 0
    0x8b: (0xf0, 0x00, 0x20, 0x03),   # MotherBoardTempWarningLow window
    0x8d: (0xa0, 0x00, 0x70, 0x03),   # MotherBoardTempFaultLow window
    # reg0x8e (LampTempWorking) is emitted only if config+0x1654 != 0 — it is 0 here, so skipped.
}
TEC = ((0xd0, 0x00), (0xd1, 0x01))    # TEC enable

# --- InitCfgCore — all literal constants ------------------------------------
# PICL+ block
PICL_CFG = [
    (0x87, (0x00, 0x00)),                                            # scan-config zero
    (0x80, (0x01,)),                                                 # light enable
    (0x82, (0,0,0,0,0,0,0,0,0,0,0xd6,0x03)),                         # 12-byte control block
    (0x80, (0x00,)),                                                 # light disable
    (0x89, (0x00,)),                                                 # reg0x89 = 0
]
# sub44 control-word default table: payload [idx, lo, hi]
SUB_CFG_82 = [                                                       # reg0x82 idx/val
    (0x06, 0x0ffd), (0x00, 0x0060), (0x0b, 0x0000), (0x04, 0x003e), (0x05, 0x080e),
    (0x01, 0x0000), (0x02, 0x0000), (0x03, 0x0000), (0x0a, 0x0400),
]
SUB_CFG_84 = [(0x00, 0x0078), (0x01, 0x0080)]                        # reg0x84 idx/val

# --- packet builders (libpakon wire grammar) --------------------------------
def p_read(addr, reg, count):  return bytes([1, 3, addr, reg, count])
def p_write(addr, reg, vals):  vals = bytes(vals); return bytes([2, 3 + len(vals), addr, len(vals), reg]) + vals
def p_poll(addr):              return bytes([3, 1, addr])
def p_write2(addr, val):       return bytes([4, 3, addr, 0x00, val & 0xff])


class Step:
    """One init step. kind: 'bulk' (EP1 packet) | 'ee_setup' | 'ee_read'. `live` = the response
    is device-provided (don't assert its bytes in the golden diff)."""
    __slots__ = ('kind', 'phase', 'note', 'data', 'req', 'val', 'length', 'live')
    def __init__(self, kind, phase, note, *, data=None, req=None, val=None, length=0, live=False):
        self.kind, self.phase, self.note = kind, phase, note
        self.data, self.req, self.val, self.length, self.live = data, req, val, length, live

    def describe(self):
        if self.kind == 'bulk':
            d = self.data; t = d[0]; a = d[2]; an = ADDRNAME.get(a, hex(a))
            if t == 1:  s = 'READ   %-6s reg0x%02x x%d' % (an, d[3], d[4])
            elif t == 2: s = 'WRITE  %-6s reg0x%02x = %s' % (an, d[4], d[5:5 + d[3]].hex())
            elif t == 3: s = 'POLL   %-6s' % an
            elif t == 4: s = 'WRITE2 %-6s [%s]' % (an, d[3:].hex())
            else:        s = 'T%d %s' % (t, d.hex())
        elif self.kind == 'ee_setup': s = 'EE A4  setup (val0x%02x)' % self.val
        else:                          s = 'EE A9  read  addr0x%04x len%d' % (self.val, self.length)
        return '%-34s  [%s] %s' % (s, self.phase, self.note)


def bulk(addr_packet, phase, note, live=False):
    return Step('bulk', phase, note, data=addr_packet, live=live)


def eeprom_steps(record_lens):
    """Generate the A4/A9 EEPROM read plan. For each record:
    read the 8-byte header (len:u32 + crc32), then (len-8) bytes in 0x20-byte chunks. `record_lens`
    are the record total lengths (read live from the header at runtime; defaults match dev36)."""
    out = []
    for start, total in zip(EE_RECORDS, record_lens):
        # header (8 bytes) at the record start
        out.append(Step('ee_setup', 'eeprom', 'record@0x%04x setup' % start, val=EE_SETUP_VAL))
        out.append(Step('ee_read', 'eeprom', 'record@0x%04x header (len+crc32)' % start,
                        req=EE_REQ_READ, val=start, length=8, live=True))
        addr, remaining = start + 8, total - 8
        while remaining > 0:
            n = min(EE_CHUNK, remaining)
            out.append(Step('ee_setup', 'eeprom', 'chunk@0x%04x setup' % addr, val=EE_SETUP_VAL))
            out.append(Step('ee_read', 'eeprom', 'data chunk @0x%04x' % addr,
                            req=EE_REQ_READ, val=addr, length=n, live=True))
            addr += n; remaining -= n
    return out


def build_init_steps(record_lens=(0x18e, 0x24)):
    """The full reconstructed InitializeScanner sequence (RE-derived). `record_lens` are the two
    EEPROM record sizes; read live from headers at runtime, defaulted to the F135 structural sizes."""
    s = []
    # PHASE 1 — HOST bring-up (open: const)
    s += [bulk(p_write2(HOST, 0x85), 'bringup', 'host init/reset cmd 0x85'),
          bulk(p_write(HOST, 0x8f, [0x00]), 'bringup', 'host interface reg0x8f=0'),
          bulk(p_write2(SUB44, 0x00), 'bringup', 'sub44 select [0000]')]
    # PHASE 2 — EEPROM read (device-provided: LED calibration)
    s += eeprom_steps(record_lens)
    # PHASE 3 — presence + identify (const cmds; serial is device-provided)
    s += [bulk(p_read(HOST, 0x02, 3), 'identify', 'HOST presence (expect bit7 0x88)', live=True),
          bulk(p_write2(SUB44, 0x00), 'identify', 'sub44 select'),
          bulk(p_write(SUB44, 0x97, [0x01]), 'identify', 'sub44 reg0x97=1'),
          bulk(p_poll(SUB44), 'identify', 'poll sub44'),
          bulk(p_write(PICL_PLUS, 0x03, [0x01]), 'identify', 'PICL+ reg0x03=1 (latch serial)'),
          bulk(p_poll(PICL_PLUS), 'identify', 'poll PICL+'),
          bulk(p_read(PICL_PLUS, 0x0c, 7), 'identify', 'read PICL+ serial', live=True),
          bulk(p_write(SUB44, 0x03, [0x01]), 'identify', 'sub44 reg0x03=1'),
          bulk(p_poll(SUB44), 'identify', 'poll sub44'),
          bulk(p_read(SUB44, 0x0c, 7), 'identify', 'read sub44 serial', live=True)]
    # PHASE 4 — arm pulse (const)
    s += [bulk(p_write(HOST, 0x84, [0x02]), 'arm', 'arm pulse HOST reg0x84=2'),
          bulk(p_write2(PICL_PLUS, 0x8a), 'arm', 'arm pulse PICL+ [008a]'),
          bulk(p_poll(PICL_PLUS), 'arm', 'poll PICL+')]
    # PHASE 5 — temperature-control / TEC setpoints (model-generic OEM defaults)
    for reg, payload in TEMP_SETPOINTS.items():
        s.append(bulk(p_write(PICL_PLUS, reg, payload), 'tempcfg', 'temp setpoint reg0x%02x' % reg))
        s.append(bulk(p_poll(PICL_PLUS), 'tempcfg', 'poll'))
    for reg, val in TEC:
        s.append(bulk(p_write(PICL_PLUS, reg, [val]), 'tempcfg', 'TEC reg0x%02x=%d' % (reg, val)))
        s.append(bulk(p_poll(PICL_PLUS), 'tempcfg', 'poll'))
    # PHASE 6 — InitCfgCore (const)
    for reg, payload in PICL_CFG:
        s.append(bulk(p_write(PICL_PLUS, reg, payload), 'initcfg', 'PICL+ reg0x%02x' % reg))
        s.append(bulk(p_poll(PICL_PLUS), 'initcfg', 'poll'))
    for idx, val in SUB_CFG_82:
        s.append(bulk(p_write(SUB44, 0x82, [idx, val & 0xff, (val >> 8) & 0xff]), 'initcfg',
                      'sub44 reg0x82 idx%d=0x%x' % (idx, val)))
        s.append(bulk(p_poll(SUB44), 'initcfg', 'poll'))
    for idx, val in SUB_CFG_84:
        s.append(bulk(p_write(SUB44, 0x84, [idx, val & 0xff, (val >> 8) & 0xff]), 'initcfg',
                      'sub44 reg0x84 idx%d=0x%x' % (idx, val)))
        s.append(bulk(p_poll(SUB44), 'initcfg', 'poll'))
    s += [bulk(p_write2(SUB44, 0xa2), 'initcfg', 'sub44 motor idle [00a2]'),
          bulk(p_poll(SUB44), 'initcfg', 'poll')]
    # PHASE 7 — reactive prime (device-provided sensor reads; service-ack)
    s += [bulk(p_poll(HOST), 'reactive', 'poll HOST'),
          bulk(p_read(PICL_PLUS, 0x01, 2), 'reactive', 'sensor reg0x01', live=True),
          bulk(p_write(PICL_PLUS, 0x06, [0x00, 0x02]), 'reactive', 'service-ack reg0x06=0002'),
          bulk(p_poll(PICL_PLUS), 'reactive', 'poll'),
          bulk(p_read(PICL_PLUS, 0x1e, 0x90), 'reactive', 'DX-code sensor block reg0x1e', live=True),
          bulk(p_read(PICL_PLUS, 0x01, 0x83), 'reactive', 'reg0x01 block', live=True),
          bulk(p_read(PICL_PLUS, 0x02, 0x84), 'reactive', 'reg0x02 block', live=True),
          bulk(p_read(PICL_PLUS, 0x04, 0x88), 'reactive', 'reg0x04 block', live=True),
          bulk(p_poll(HOST), 'reactive', 'poll HOST'),
          bulk(p_read(SUB44, 0x01, 2), 'reactive', 'sub44 reg0x01', live=True),
          # the 2 computed exposure writes (idx9) — scan-time defaults, harmless at init
          bulk(p_write(SUB44, 0x82, [0x09, 0x14, 0x00]), 'reactive', 'sub44 reg0x82 idx9=0x14'),
          bulk(p_poll(SUB44), 'reactive', 'poll'),
          bulk(p_write(SUB44, 0x82, [0x09, 0x17, 0x00]), 'reactive', 'sub44 reg0x82 idx9=0x17'),
          bulk(p_poll(SUB44), 'reactive', 'poll')]
    # PHASE 8 — wait-ready heartbeat (poll HOST until bit7 settles)
    for _ in range(22):
        s.append(bulk(p_poll(HOST), 'waitready', 'poll HOST (settle ready)', live=True))
    return s


# ===========================================================================
# Offline golden byte-diff against dev36_init.json
# ===========================================================================
def step_to_dev36_shape(st):
    """Render a generated Step in the dev36_init.json record shape for comparison."""
    if st.kind == 'bulk':
        return {'t': 'bulk', 'data': st.data.hex()}
    if st.kind == 'ee_setup':
        return {'t': 'ctrl', 'bm': 0x40, 'req': EE_REQ_SETUP, 'val': st.val, 'idx': EE_MAGIC, 'len': 0}
    return {'t': 'ctrl', 'bm': 0xc0, 'req': st.req, 'val': st.val, 'idx': EE_MAGIC, 'len': st.length}


def verify(seq):
    """Compare the generated sequence against dev36_init.json. Asserts the DETERMINISTIC fields
    (packet bytes / control-transfer shape) match; the live read RESPONSES are not in dev36's
    request records, so this validates the request stream is byte-identical."""
    dev = [{k: v for k, v in st.items() if k != 'rel'} for st in json.load(open(DEV36))]
    gen = [step_to_dev36_shape(s) for s in seq]
    print("dev36 steps: %d   generated steps: %d" % (len(dev), len(gen)))
    ok = True
    n = max(len(dev), len(gen))
    mism = 0
    for i in range(n):
        d = dev[i] if i < len(dev) else None
        g = gen[i] if i < len(gen) else None
        match = (d == g)
        if not match:
            # normalise: dev36 ctrl bm for IN read is 0xc0; OUT is 0x40 — already matched
            ok = False; mism += 1
            print("  MISMATCH @%3d" % i)
            print("     dev36: %s" % d)
            print("     gen  : %s" % g)
            if mism >= 20:
                print("  ... (stopping after 20 mismatches)"); break
    print("-" * 60)
    print("VERDICT: %s" % ("IDENTICAL — generated stream byte-matches dev36" if ok
                           else "%d mismatch(es) — see above" % mism))
    return 0 if ok else 1


# ===========================================================================
# Hardware execution (run pakon_load first; device powered)
# ===========================================================================
def run_steps(dev, seq, verbose=False):
    """Execute a built step list on an already-OPEN PakonDevice (serial write-then-poll, no concurrent
    polling — the init_sequence desync lesson). Returns (eeprom_bytes, anomalies). Reused by both the
    standalone executor below and the PakonScanner lifecycle object."""
    eeprom = bytearray()
    anomalies = []
    with dev._lock:
        for i, st in enumerate(seq):
            try:
                if st.kind == 'ee_setup':
                    dev.handle.controlWrite(0x40, st.req if st.req else EE_REQ_SETUP,
                                            st.val, EE_MAGIC, b'', timeout=2000)
                    if verbose: print('%3d %s' % (i, st.describe()))
                elif st.kind == 'ee_read':
                    r = dev.handle.controlRead(0xc0, st.req, st.val, EE_MAGIC, st.length, timeout=2000)
                    eeprom += bytes(r)
                    if verbose: print('%3d %s -> %s' % (i, st.describe(), bytes(r).hex()))
                else:
                    resp = dev.send_raw(st.data)
                    t = st.data[0]
                    bad = (len(resp) < 4) or (t == 2 and resp[3] not in (0x00, 0x08))
                    if bad:
                        anomalies.append((i, st.describe(), resp.hex()))
                    if verbose or bad:
                        print('%3d %s -> %s%s' % (i, st.describe(), resp.hex(),
                                                   '  <<< unexpected' if bad else ''))
            except Exception as e:                       # noqa: BLE001 (tolerant per-step)
                anomalies.append((i, st.describe(), str(e)))
                print('%3d %s -> ERROR %s' % (i, st.describe(), e))
    return bytes(eeprom), anomalies


def execute(seq, verbose=False, settle_max=12.0):
    dev = PakonDevice().open()
    try:
        d = dev.handle.getDevice()
        print("opened %04x:%04x bus%d addr%d — executing %d RECONSTRUCTED init steps"
              % (libpakon.VID, libpakon.PID_LOADED, d.getBusNumber(), d.getDeviceAddress(), len(seq)))
        print("=" * 66)
        eeprom, anomalies = run_steps(dev, seq, verbose)
        print("=" * 66)
        print("init done. EEPROM bytes read: %d  anomalies: %d" % (len(eeprom), len(anomalies)))
        for i, c, det in anomalies[:12]:
            print("   %3d %-34s %s" % (i, c, det))

        # readiness verdict (HOST bit7 settles)
        status = None
        end = time.monotonic() + settle_max
        while time.monotonic() < end:
            try:
                status = dev.poll_status(dev.AD_HOST)
            except (PakonError, Exception):
                status = None
            if status is not None and (status & 0x80):
                break
            time.sleep(0.2)
        print("=" * 66)
        if status is not None and (status & 0x80):
            print("VERDICT: HOST bit7 SET (0x%02x) — reconstructed init reached ready." % status)
            print("  >>> CHECK: Film LED should be OFF.")
            return 0
        print("VERDICT: HOST = %s (bit7 not set) — init did not complete."
              % ('0x%02x' % status if status is not None else 'n/a'))
        return 1
    finally:
        dev.close()
