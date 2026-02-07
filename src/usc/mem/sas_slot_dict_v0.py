from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import re
from datetime import datetime, timezone


MAGIC_SASD = b"USC_SASD5"  # dict packet v5 (tools + slotdict)
MAGIC_SASA = b"USC_SASA5"  # data packet v5 (dt + ts_style + tool_id + rid16 + payload_chunks)


# -----------------------------
# Varint helpers
# -----------------------------
def _uvarint_encode(x: int) -> bytes:
    x = int(x)
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


def _uvarint_decode(buf: bytes, off: int) -> Tuple[int, int]:
    x = 0
    shift = 0
    while True:
        if off >= len(buf):
            raise ValueError("uvarint decode overflow")
        b = buf[off]
        off += 1
        x |= (b & 0x7F) << shift
        if (b & 0x80) == 0:
            return x, off
        shift += 7
        if shift > 63:
            raise ValueError("uvarint too large")


def _bstr(s: str) -> bytes:
    return s.encode("utf-8", errors="replace")


def _s(b: bytes) -> str:
    return b.decode("utf-8", errors="replace")


# -----------------------------
# Timestamp helpers
# -----------------------------
def _iso_to_us(iso: str) -> int:
    iso = iso.strip()
    if iso.endswith("Z"):
        iso = iso[:-1] + "+00:00"
    dt = datetime.fromisoformat(iso)
    return int(dt.timestamp() * 1_000_000)


def _format_us_to_iso(ts_us: int) -> str:
    dt = datetime.fromtimestamp(ts_us / 1_000_000, tz=timezone.utc)
    return dt.isoformat(timespec="milliseconds")


# -----------------------------
# UUID helpers (binary 16 bytes)
# -----------------------------
_HEX = set("0123456789abcdefABCDEF")


def _uuid_to_16(uuid_str: str) -> bytes:
    s = uuid_str.strip().replace("-", "")
    if len(s) != 32:
        raise ValueError("bad uuid length")
    if any(c not in _HEX for c in s):
        raise ValueError("bad uuid chars")
    return bytes.fromhex(s)


def _u16_to_uuid(b16: bytes) -> str:
    if len(b16) != 16:
        raise ValueError("uuid bytes must be 16")
    h = b16.hex()
    return f"{h[0:8]}-{h[8:12]}-{h[12:16]}-{h[16:20]}-{h[20:32]}"


# -----------------------------
# Line parsers (support TWO timestamp styles)
# -----------------------------
RE_BRACKET = re.compile(r"^\[(?P<ts>[^\]]+)\]\s+(?P<body>.*)$")
RE_BARE = re.compile(r"^(?P<ts>\d{4}-\d{2}-\d{2}T[0-9:\.]+(?:Z|[+\-]\d{2}:\d{2}))\s+(?P<body>.*)$")

RE_TOOL_CALL = re.compile(
    r"^tool_call::(?P<tool>\S+)\s+rid=(?P<rid>[0-9a-fA-F\-]{36})\s+payload=(?P<payload>\{.*\})$"
)

# ts_style:
# 0 = no timestamp (raw line)
# 1 = bracketed: [ts] body
# 2 = bare: ts body
def _split_line(line: str) -> Tuple[Optional[str], str, int]:
    s = line.rstrip("\n")

    m = RE_BRACKET.match(s)
    if m:
        return m.group("ts"), m.group("body"), 1

    m = RE_BARE.match(s)
    if m:
        return m.group("ts"), m.group("body"), 2

    return None, s, 0


# -----------------------------
# Slot dictionary mining
# -----------------------------
RE_JSON_KEY = re.compile(r'"([^"\\]{1,40})"\s*:')

def _mine_slot_keys(payloads: List[str], top_k: int = 256) -> List[bytes]:
    """
    Mine common JSON keys from payload strings.
    Store as exact substring bytes: b'"key"' including quotes.
    """
    freq: Dict[str, int] = {}

    for p in payloads:
        for m in RE_JSON_KEY.finditer(p):
            k = m.group(1)
            freq[k] = freq.get(k, 0) + 1

    items = sorted(freq.items(), key=lambda kv: (-kv[1], kv[0]))
    keys = [k for k, _c in items[:top_k]]

    # store as exact substring with quotes
    out = [b'"' + _bstr(k) + b'"' for k in keys]
    return out


@dataclass
class SASDict:
    base_ts_us: int
    tool_to_id: Dict[str, int]
    id_to_tool: List[str]     # id 1..n
    slot_to_id: Dict[bytes, int]
    id_to_slot: List[bytes]   # id 1..m


def _build_dict(lines: List[str], slot_top_k: int = 256) -> SASDict:
    # Find first timestamp
    first_ts_us = None
    for ln in lines:
        ts, _body, style = _split_line(ln)
        if ts is not None and style in (1, 2):
            first_ts_us = _iso_to_us(ts)
            break
    if first_ts_us is None:
        first_ts_us = 0

    # Collect tools + payloads
    tools_set = set()
    payloads: List[str] = []

    for ln in lines:
        ts, body, _style = _split_line(ln)
        if ts is None:
            continue
        m = RE_TOOL_CALL.match(body)
        if m:
            tools_set.add(m.group("tool"))
            payloads.append(m.group("payload"))

    tools_sorted = sorted(tools_set)
    tool_to_id = {t: i + 1 for i, t in enumerate(tools_sorted)}

    # Mine slot keys
    slot_entries = _mine_slot_keys(payloads, top_k=slot_top_k)
    # sort longest-first to enable greedy match
    slot_entries = sorted(slot_entries, key=lambda b: (-len(b), b))

    id_to_slot = slot_entries[:]  # id 1..m
    slot_to_id = {b: i + 1 for i, b in enumerate(id_to_slot)}

    return SASDict(
        base_ts_us=first_ts_us,
        tool_to_id=tool_to_id,
        id_to_tool=tools_sorted,
        slot_to_id=slot_to_id,
        id_to_slot=id_to_slot,
    )


def _encode_dict_packet(d: SASDict) -> bytes:
    out = bytearray()
    out += MAGIC_SASD
    out += _uvarint_encode(d.base_ts_us)

    # tools
    out += _uvarint_encode(len(d.id_to_tool))
    for t in d.id_to_tool:
        tb = _bstr(t)
        out += _uvarint_encode(len(tb))
        out += tb

    # slot dict entries (bytes)
    out += _uvarint_encode(len(d.id_to_slot))
    for sb in d.id_to_slot:
        out += _uvarint_encode(len(sb))
        out += sb

    return bytes(out)


def _decode_dict_packet(pkt: bytes) -> SASDict:
    if not pkt.startswith(MAGIC_SASD):
        raise ValueError("bad SAS dict magic")
    off = len(MAGIC_SASD)

    base_ts_us, off = _uvarint_decode(pkt, off)

    # tools
    n_tools, off = _uvarint_decode(pkt, off)
    id_to_tool: List[str] = []
    tool_to_id: Dict[str, int] = {}
    for i in range(n_tools):
        ln, off = _uvarint_decode(pkt, off)
        tb = pkt[off:off + ln]
        off += ln
        t = _s(tb)
        tid = i + 1
        id_to_tool.append(t)
        tool_to_id[t] = tid

    # slots
    n_slots, off = _uvarint_decode(pkt, off)
    id_to_slot: List[bytes] = []
    slot_to_id: Dict[bytes, int] = {}
    for i in range(n_slots):
        ln, off = _uvarint_decode(pkt, off)
        sb = pkt[off:off + ln]
        off += ln
        sid = i + 1
        id_to_slot.append(sb)
        slot_to_id[sb] = sid

    return SASDict(
        base_ts_us=int(base_ts_us),
        tool_to_id=tool_to_id,
        id_to_tool=id_to_tool,
        slot_to_id=slot_to_id,
        id_to_slot=id_to_slot,
    )


# -----------------------------
# Payload chunk encoding (lossless substitution)
# -----------------------------
# chunk tag:
# 0 = literal bytes
# 1 = slot-dict ref (uvarint id)
def _payload_to_chunks(d: SASDict, payload: bytes) -> List[Tuple[int, bytes]]:
    """
    Greedy replace any exact slot substring b'"key"' with a ref token.
    Everything else is literal.
    """
    if not d.id_to_slot:
        return [(0, payload)]

    # quick index: first byte -> candidates (all slots start with b'"')
    candidates = d.id_to_slot

    out: List[Tuple[int, bytes]] = []
    i = 0
    n = len(payload)

    lit_start = 0

    while i < n:
        matched = None

        if payload[i:i+1] == b'"':
            # try match longest-first
            for sb in candidates:
                L = len(sb)
                if i + L <= n and payload[i:i+L] == sb:
                    matched = sb
                    break

        if matched is None:
            i += 1
            continue

        # flush literal before match
        if lit_start < i:
            out.append((0, payload[lit_start:i]))

        # emit ref chunk
        sid = d.slot_to_id.get(matched, 0)
        if sid == 0:
            # fallback literal
            out.append((0, matched))
        else:
            out.append((1, _uvarint_encode(sid)))

        i += len(matched)
        lit_start = i

    # flush tail literal
    if lit_start < n:
        out.append((0, payload[lit_start:n]))

    # merge adjacent literals
    merged: List[Tuple[int, bytes]] = []
    for tag, b in out:
        if not merged:
            merged.append((tag, b))
        else:
            pt, pb = merged[-1]
            if tag == 0 and pt == 0:
                merged[-1] = (0, pb + b)
            else:
                merged.append((tag, b))

    return merged if merged else [(0, payload)]


def _chunks_to_payload(d: SASDict, chunks: List[Tuple[int, bytes]]) -> bytes:
    out = bytearray()
    for tag, b in chunks:
        if tag == 0:
            out += b
        else:
            # b is varint bytes for slot id
            sid, _ = _uvarint_decode(b, 0)
            if 1 <= sid <= len(d.id_to_slot):
                out += d.id_to_slot[sid - 1]
            else:
                # invalid id, skip
                pass
    return bytes(out)


# -----------------------------
# Data packets
# -----------------------------
def _encode_data_packet(entries: List[Tuple[int, int, int, Optional[bytes], bytes, List[Tuple[int, bytes]]]]) -> bytes:
    """
    Each entry:
      (dt_us, ts_style, tool_id, rid16_or_none, raw_body_bytes, payload_chunks)

    tool_id:
      0 = RAW line (use raw_body_bytes)
      1..n = tool_call (use payload_chunks + rid16)
    """
    out = bytearray()
    out += MAGIC_SASA
    out += _uvarint_encode(len(entries))

    for dt_us, ts_style, tool_id, rid16, raw_body, payload_chunks in entries:
        out += _uvarint_encode(dt_us)
        out += _uvarint_encode(ts_style)
        out += _uvarint_encode(tool_id)

        if tool_id == 0:
            out += _uvarint_encode(len(raw_body))
            out += raw_body
        else:
            if rid16 is None or len(rid16) != 16:
                raise ValueError("tool entry missing rid16")

            out += rid16

            # payload chunks
            out += _uvarint_encode(len(payload_chunks))
            for tag, bb in payload_chunks:
                out += _uvarint_encode(tag)
                out += _uvarint_encode(len(bb))
                out += bb

    return bytes(out)


def _decode_data_packet(pkt: bytes) -> List[Tuple[int, int, int, Optional[bytes], bytes, List[Tuple[int, bytes]]]]:
    if not pkt.startswith(MAGIC_SASA):
        raise ValueError("bad SAS data magic")
    off = len(MAGIC_SASA)

    n, off = _uvarint_decode(pkt, off)
    out: List[Tuple[int, int, int, Optional[bytes], bytes, List[Tuple[int, bytes]]]] = []

    for _ in range(n):
        dt_us, off = _uvarint_decode(pkt, off)
        ts_style, off = _uvarint_decode(pkt, off)
        tool_id, off = _uvarint_decode(pkt, off)

        if tool_id == 0:
            ln, off = _uvarint_decode(pkt, off)
            raw = pkt[off:off + ln]
            off += ln
            out.append((dt_us, ts_style, 0, None, raw, []))
        else:
            rid16 = pkt[off:off + 16]
            off += 16

            nch, off = _uvarint_decode(pkt, off)
            chunks: List[Tuple[int, bytes]] = []
            for _j in range(nch):
                tag, off = _uvarint_decode(pkt, off)
                ln, off = _uvarint_decode(pkt, off)
                bb = pkt[off:off + ln]
                off += ln
                chunks.append((int(tag), bb))

            out.append((dt_us, ts_style, int(tool_id), rid16, b"", chunks))

    return out


# -----------------------------
# Public API
# -----------------------------
def build_sas_packets_from_text(text: str, max_lines_per_packet: int = 60, slot_top_k: int = 256) -> List[bytes]:
    lines = text.splitlines()
    d = _build_dict(lines, slot_top_k=slot_top_k)

    packets: List[bytes] = []
    packets.append(_encode_dict_packet(d))

    chunk: List[Tuple[int, int, int, Optional[bytes], bytes, List[Tuple[int, bytes]]]] = []
    last_ts_us = d.base_ts_us
    ts_us = d.base_ts_us

    for ln in lines:
        ts, body, ts_style = _split_line(ln)

        if ts is None:
            # raw no-ts line: preserve exact whitespace
            chunk.append((0, 0, 0, None, _bstr(body), []))
        else:
            ts_us = _iso_to_us(ts)
            dt_us = max(0, ts_us - last_ts_us)
            last_ts_us = ts_us

            m = RE_TOOL_CALL.match(body)
            if m:
                tool = m.group("tool")
                rid = m.group("rid")
                payload_str = m.group("payload")

                tool_id = d.tool_to_id.get(tool, 0)
                rid16 = _uuid_to_16(rid)

                payload_bytes = payload_str.encode("utf-8", errors="replace")
                chunks_payload = _payload_to_chunks(d, payload_bytes)

                chunk.append((dt_us, ts_style, tool_id, rid16, b"", chunks_payload))
            else:
                # timestamped but not a tool call -> raw body
                chunk.append((dt_us, ts_style, 0, None, _bstr(body), []))

        if len(chunk) >= max_lines_per_packet:
            packets.append(_encode_data_packet(chunk))
            chunk = []

    if chunk:
        packets.append(_encode_data_packet(chunk))

    return packets


def decode_sas_packets_to_lines(packets: List[bytes]) -> List[str]:
    if not packets:
        return []

    d = _decode_dict_packet(packets[0])
    out_lines: List[str] = []

    ts_us = d.base_ts_us

    for pkt in packets[1:]:
        entries = _decode_data_packet(pkt)
        for dt_us, ts_style, tool_id, rid16, raw_body, chunks in entries:
            # only advance if this line had a timestamp style
            if d.base_ts_us != 0 and ts_style in (1, 2):
                ts_us += dt_us

            if tool_id == 0:
                body = _s(raw_body)
            else:
                tool = d.id_to_tool[tool_id - 1] if 1 <= tool_id <= len(d.id_to_tool) else "UNKNOWN"
                rid = _u16_to_uuid(rid16 or (b"\x00" * 16))
                payload = _chunks_to_payload(d, chunks).decode("utf-8", errors="replace")
                body = f"tool_call::{tool} rid={rid} payload={payload}"

            # preserve formatting EXACTLY (no strip)
            if ts_style == 0 or d.base_ts_us == 0:
                out_lines.append(body)
            else:
                ts_str = _format_us_to_iso(ts_us)
                if ts_style == 1:
                    out_lines.append(f"[{ts_str}] {body}")
                else:
                    out_lines.append(f"{ts_str} {body}")

    return out_lines
