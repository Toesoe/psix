#!/usr/bin/env python3
"""pakon_unitdata: decode the per-unit EEPROM (section A + B) that init reads.

Layout per pakon-reference calibration.md (confirmed across five units): section A
payload = hardware version / scanner type / serial, then per-resolution-base
{Offset, MotorSpeed, MotorSpeed_IR} triples for bases 4/8/16, then the two 3x10
colour matrices. Section B = twelve u16 motor-adjust words. The EEPROM does NOT
hold the light calibration (LED currents/duties — OEM keeps those per-scan; psix
uses the OEM-capture F-135 operating point for that).

The transport rate the OEM writes is MotorSpeed scaled by the adjust words:
    rate = round(MotorSpeed * MotorAdjust / MotorAdjustDrag)
(verified: 1569 * 1000/1008 -> 1557, exactly the OEM capture's rate on an F-135).
"""
import struct

SCANNER_TYPES = {1350: 'F135', 1351: 'F135+'}
BASE16_INTEGRATION = 0x0c1a        # OEM capture: scan-time CCD integration (F-135)
BASE16_WINDOW = 2000               # active CCD window width at base16 (end = offset + width)


def parse(eeprom):
    """Decode the init-read EEPROM blob (A header+payload, then B) -> dict or None.
    Tolerates short/failed reads: returns what decoded."""
    if not eeprom or len(eeprom) < 0x26:
        return None
    a = eeprom[8:8 + 390]                              # section A payload
    if len(a) < 0x26:
        return None
    u16 = lambda off: struct.unpack_from('<H', a, off)[0]
    u32 = lambda off: struct.unpack_from('<I', a, off)[0]
    f32 = lambda off: struct.unpack_from('<f', a, off)[0]

    def base_triplet(off):
        return {'offset': u16(off), 'motor_speed': u16(off + 2), 'motor_speed_ir': u16(off + 4)}

    out = {
        'hw_version': u32(0x0),
        'scanner_type': u32(0x4),
        'serial': u32(0x8),
        'base4': base_triplet(0x14 - 8),
        'base8': base_triplet(0x1a - 8),
        'base16': base_triplet(0x20 - 8),
    }
    out['model'] = SCANNER_TYPES.get(out['scanner_type'])
    try:
        out['neg_matrix'] = [round(f32(0x26 - 8 + 4 * i), 4) for i in range(30)]
        out['pos_matrix'] = [round(f32(0x9e - 8 + 4 * i), 4) for i in range(30)]
    except struct.error:
        pass
    # section B: 12 u16 motor-adjust words (payload starts after A(398) + B header(8))
    b_off = 8 + 390 + 8
    if len(eeprom) >= b_off + 24:
        out['motor_adjust'] = list(struct.unpack_from('<12H', eeprom, b_off))
    return out


def transport_rate(unit, ir=False, base='base16'):
    """OEM transport rate for a base: MotorSpeed scaled by the motor-adjust words.
    Falls back to the raw MotorSpeed when the adjust words are missing."""
    if not unit:
        return None
    b = unit.get(base) or {}
    ms = b.get('motor_speed_ir' if ir else 'motor_speed')
    if not ms:
        return None
    adj = unit.get('motor_adjust') or []
    if len(adj) >= 12:
        # per-base pairs (fwd, IR) of [adjust, drag]: base16 = words [8:12]
        pair = adj[8:12] if base == 'base16' else adj[0:4]
        a, d = (pair[0], pair[1]) if not ir else (pair[2], pair[3])
        if a and d:
            return int(round(ms * a / d))
    return ms


def describe(unit):
    if not unit:
        return 'eeprom: not decoded'
    b16 = unit['base16']
    return ('eeprom: type=%d (%s) serial=%d hw=%d | base16 offset=%d speed=%d ir=%d'
            % (unit['scanner_type'], unit.get('model') or '?', unit['serial'],
               unit['hw_version'], b16['offset'], b16['motor_speed'], b16['motor_speed_ir']))
