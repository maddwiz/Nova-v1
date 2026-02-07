from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Optional


# ---------------------------------------
# Slot type tags
# ---------------------------------------
# 0 = DICT STRING (value -> dict id)
# 1 = INT DELTA   (value parsed as int, delta+zigzag)
# 2 = RAW STRING  (store string bytes directly, no dict)
SLOT_DICT = 0
SLOT_INT = 1
SLOT_RAW = 2


# ---------------------------------------
# Varint + ZigZag helpers
# ---------------------------------------
def uvarint_encode(x: int) -> bytes:
    if x < 0:
        raise ValueError("uvarint cannot encode negative")
    out = bytearray()
    while True:
        b = x & 0x7F
        x >>= 7
        if x:
            out.append(b | 0x80)
        else:
            out.append(b)
            break
    return bytes(out)


def uvarint_decode(buf: bytes, i: int) -> Tuple[int, int]:
    x = 0
    shift = 0
    while True:
        if i >= len(buf):
            raise ValueError("uvarint decode overflow")
        b = buf[i]
        i += 1
        x |= (b & 0x7F) << shift
        if not (b & 0x80):
            break
        shift += 7
        if shift > 63:
            raise ValueError("uvarint too large")
    return x, i


def zigzag_encode(n: int) -> int:
    # signed -> unsigned
    # works fine for our ints/deltas
    return (n << 1) ^ (n >> 63)


def zigzag_decode(z: int) -> int:
    # unsigned -> signed
    return (z >> 1) ^ -(z & 1)


def pack_str(s: str) -> bytes:
    b = s.encode("utf-8", errors="ignore")
    return uvarint_encode(len(b)) + b


def unpack_str(buf: bytes, i: int) -> Tuple[str, int]:
    n, i = uvarint_decode(buf, i)
    s = buf[i : i + n].decode("utf-8", errors="ignore")
    return s, i + n


def try_int(s: str) -> Optional[int]:
    try:
        if not s:
            return None
        if s[0] == "-" and s[1:].isdigit():
            return int(s)
        if s.isdigit():
            return int(s)
        return None
    except Exception:
        return None


# ---------------------------------------
# Type inference
# ---------------------------------------
def infer_slot_types(samples_by_slot: List[List[str]], int_min_hits: int = 50) -> List[int]:
    """
    Decide per-slot:
      - INT if enough samples parse as ints
      - else DICT (default)
    """
    out: List[int] = []
    for col in samples_by_slot:
        hits = 0
        for v in col:
            if try_int(v) is not None:
                hits += 1
        if hits >= int_min_hits:
            out.append(SLOT_INT)
        else:
            out.append(SLOT_DICT)
    return out


@dataclass
class TypedSlotsState:
    slot_types: List[int]
    dicts_v2i: List[dict[str, int]]
    dicts_i2v: List[list[str]]

    @classmethod
    def from_types(cls, slot_types: List[int]) -> "TypedSlotsState":
        dicts_v2i: List[dict[str, int]] = []
        dicts_i2v: List[list[str]] = []
        for _ in slot_types:
            dicts_v2i.append({})
            dicts_i2v.append([])
        return cls(slot_types=slot_types, dicts_v2i=dicts_v2i, dicts_i2v=dicts_i2v)


MAGIC = b"TS1"


def _dict_id_for(st: TypedSlotsState, slot_i: int, v: str) -> int:
    d = st.dicts_v2i[slot_i]
    if v in d:
        return d[v]
    idx = len(st.dicts_i2v[slot_i])
    d[v] = idx
    st.dicts_i2v[slot_i].append(v)
    return idx


def pack_params_typed(st: TypedSlotsState, events: List[List[str]]) -> bytes:
    """
    Format:
      MAGIC(3)
      nslots(uvarint)
      slot_types[nslots] (1 byte each)
      nevents(uvarint)

      For each slot:
        if DICT:
          dict_count(uvarint)
          dict strings...
          ids[nevents] as uvarint
        if INT:
          first_value (zigzag -> uvarint)
          deltas[nevents-1] (zigzag -> uvarint)
        if RAW:
          strings[nevents] (len+bytes)
    """
    if not events:
        raise ValueError("events empty")

    nslots = len(events[0])
    for row in events:
        if len(row) != nslots:
            raise ValueError("events are ragged (different slot counts)")

    if len(st.slot_types) != nslots:
        raise ValueError("state slot_types mismatch nslots")

    out = bytearray()
    out += MAGIC
    out += uvarint_encode(nslots)

    for t in st.slot_types:
        out.append(int(t) & 0xFF)

    out += uvarint_encode(len(events))

    for si, stype in enumerate(st.slot_types):
        col = [row[si] for row in events]

        if stype == SLOT_DICT:
            # build dict first
            for v in col:
                _dict_id_for(st, si, v)

            dict_vals = st.dicts_i2v[si]

            out += uvarint_encode(len(dict_vals))
            for s in dict_vals:
                out += pack_str(s)

            for v in col:
                out += uvarint_encode(_dict_id_for(st, si, v))

        elif stype == SLOT_INT:
            ints: List[int] = []
            for v in col:
                x = try_int(v)
                if x is None:
                    x = 0
                ints.append(x)

            out += uvarint_encode(zigzag_encode(ints[0]))
            prev = ints[0]
            for x in ints[1:]:
                d = x - prev
                out += uvarint_encode(zigzag_encode(d))
                prev = x

        elif stype == SLOT_RAW:
            for v in col:
                out += pack_str(v)

        else:
            raise ValueError(f"unknown slot type {stype}")

    return bytes(out)


def unpack_params_typed(blob: bytes, event_count: int | None = None) -> Tuple[List[int], List[List[str]]]:
    """
    Returns:
      slot_types, decoded events (strings)
    """
    if not blob.startswith(MAGIC):
        raise ValueError("bad typed-slots magic")

    i = len(MAGIC)

    nslots, i = uvarint_decode(blob, i)

    slot_types: List[int] = []
    for _ in range(nslots):
        slot_types.append(blob[i])
        i += 1

    nevents, i = uvarint_decode(blob, i)

    # optional mismatch check (doesn't break decode)
    if event_count is not None and nevents != event_count:
        pass

    cols: List[List[str]] = [[] for _ in range(nslots)]

    for si, stype in enumerate(slot_types):
        if stype == SLOT_DICT:
            dict_count, i = uvarint_decode(blob, i)

            dict_i2v: List[str] = []
            for _ in range(dict_count):
                s, i = unpack_str(blob, i)
                dict_i2v.append(s)

            for _ in range(nevents):
                did, i = uvarint_decode(blob, i)
                if did < len(dict_i2v):
                    cols[si].append(dict_i2v[did])
                else:
                    cols[si].append("")

        elif stype == SLOT_INT:
            z0, i = uvarint_decode(blob, i)
            first = zigzag_decode(z0)

            ints = [first]
            prev = first
            for _ in range(nevents - 1):
                zz, i = uvarint_decode(blob, i)
                d = zigzag_decode(zz)
                x = prev + d
                ints.append(x)
                prev = x

            cols[si] = [str(x) for x in ints]

        elif stype == SLOT_RAW:
            for _ in range(nevents):
                s, i = unpack_str(blob, i)
                cols[si].append(s)

        else:
            raise ValueError(f"unknown slot type {stype}")

    # transpose cols -> events
    events: List[List[str]] = []
    for ei in range(nevents):
        row = []
        for si in range(nslots):
            row.append(cols[si][ei])
        events.append(row)

    return slot_types, events
