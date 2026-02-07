from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import re
from datetime import datetime, timezone


MAGIC_SAS_DICT = b"USC_SASD4"  # dict packet v4 (base_ts + tool dict)
MAGIC_SAS_DATA = b"USC_SASA4"  # data packet v4 (dt + tool_id + ts_style + rid16 + payload)


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

# Bare ISO timestamp lines (no brackets)
# Example:
# 2026-01-20T05:02:29.641+00:00 INFO ...
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


@dataclass
class SASDictV4:
    base_ts_us: int
    tool_to_id: Dict[str, int]
    id_to_tool: List[str]  # id 1..n


def _encode_dict_packet(d: SASDictV4) -> bytes:
    out = bytearray()
    out += MAGIC_SAS_DICT
    out += _uvarint_encode(d.base_ts_us)

    tools = d.id_to_tool[:]
    out += _uvarint_encode(len(tools))
    for t in tools:
        tb = _bstr(t)
        out += _uvarint_encode(len(tb))
        out += tb

    return bytes(out)


def _decode_dict_packet(pkt: bytes) -> SASDictV4:
    if not pkt.startswith(MAGIC_SAS_DICT):
        raise ValueError("bad SAS dict magic")
    off = len(MAGIC_SAS_DICT)

    base_ts_us, off = _uvarint_decode(pkt, off)

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

    return SASDictV4(base_ts_us=base_ts_us, tool_to_id=tool_to_id, id_to_tool=id_to_tool)


def _encode_data_packet(entries: List[Tuple[int, int, int, Optional[bytes], bytes]]) -> bytes:
    """
    Each entry:
      (dt_us, tool_id, ts_style, rid16_or_none, payload_bytes)

    tool_id:
      0 = RAW body (payload is utf8 body text)
      1..n = tool call (payload is utf8 json payload, rid16 required)

    ts_style:
      0 = no timestamp
      1 = bracketed
      2 = bare
    """
    out = bytearray()
    out += MAGIC_SAS_DATA
    out += _uvarint_encode(len(entries))

    for dt_us, tool_id, ts_style, rid16, payload in entries:
        out += _uvarint_encode(dt_us)
        out += _uvarint_encode(tool_id)
        out += _uvarint_encode(ts_style)

        if tool_id == 0:
            out += _uvarint_encode(len(payload))
            out += payload
        else:
            if rid16 is None or len(rid16) != 16:
                raise ValueError("tool entry missing rid16")
            out += rid16
            out += _uvarint_encode(len(payload))
            out += payload

    return bytes(out)


def _decode_data_packet(pkt: bytes) -> List[Tuple[int, int, int, Optional[bytes], bytes]]:
    if not pkt.startswith(MAGIC_SAS_DATA):
        raise ValueError("bad SAS data magic")
    off = len(MAGIC_SAS_DATA)

    n, off = _uvarint_decode(pkt, off)
    out: List[Tuple[int, int, int, Optional[bytes], bytes]] = []

    for _ in range(n):
        dt_us, off = _uvarint_decode(pkt, off)
        tool_id, off = _uvarint_decode(pkt, off)
        ts_style, off = _uvarint_decode(pkt, off)

        if tool_id == 0:
            ln, off = _uvarint_decode(pkt, off)
            payload = pkt[off:off + ln]
            off += ln
            out.append((dt_us, 0, ts_style, None, payload))
        else:
            rid16 = pkt[off:off + 16]
            off += 16
            ln, off = _uvarint_decode(pkt, off)
            payload = pkt[off:off + ln]
            off += ln
            out.append((dt_us, tool_id, ts_style, rid16, payload))

    return out


# -----------------------------
# Public API
# -----------------------------
def build_sas_packets_from_text(text: str, max_lines_per_packet: int = 60) -> List[bytes]:
    lines = text.splitlines()
    if not lines:
        d = SASDictV4(base_ts_us=0, tool_to_id={}, id_to_tool=[])
        return [_encode_dict_packet(d)]

    # Find first timestamp (either bracket or bare)
    first_ts_us = None
    for ln in lines:
        ts, _, style = _split_line(ln)
        if ts is not None and style in (1, 2):
            first_ts_us = _iso_to_us(ts)
            break
    if first_ts_us is None:
        first_ts_us = 0

    # Collect tools for dict
    tools_set = set()
    for ln in lines:
        ts, body, style = _split_line(ln)
        if ts is None:
            continue
        m = RE_TOOL_CALL.match(body)
        if m:
            tools_set.add(m.group("tool"))

    tools_sorted = sorted(tools_set)
    tool_to_id = {t: i + 1 for i, t in enumerate(tools_sorted)}
    d = SASDictV4(base_ts_us=first_ts_us, tool_to_id=tool_to_id, id_to_tool=tools_sorted)

    packets: List[bytes] = []
    packets.append(_encode_dict_packet(d))

    chunk: List[Tuple[int, int, int, Optional[bytes], bytes]] = []
    last_ts_us = first_ts_us

    for ln in lines:
        ts, body, ts_style = _split_line(ln)

        if ts is None:
            # no timestamp: raw (preserve whitespace)
            chunk.append((0, 0, 0, None, _bstr(body)))
        else:
            ts_us = _iso_to_us(ts)
            dt_us = max(0, ts_us - last_ts_us)
            last_ts_us = ts_us

            m = RE_TOOL_CALL.match(body)
            if m:
                tool = m.group("tool")
                rid = m.group("rid")
                payload = m.group("payload")

                tool_id = tool_to_id.get(tool, 0)
                rid16 = _uuid_to_16(rid)
                chunk.append((dt_us, tool_id, ts_style, rid16, _bstr(payload)))
            else:
                # timestamped but not tool_call
                chunk.append((dt_us, 0, ts_style, None, _bstr(body)))

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
        for dt_us, tool_id, ts_style, rid16, payload in entries:
            if d.base_ts_us != 0 and ts_style in (1, 2):
                ts_us += dt_us

            if tool_id == 0:
                body = _s(payload)
            else:
                tool = d.id_to_tool[tool_id - 1] if 1 <= tool_id <= len(d.id_to_tool) else "UNKNOWN"
                rid = _u16_to_uuid(rid16 or (b"\x00" * 16))
                body = f"tool_call::{tool} rid={rid} payload={_s(payload)}"

            # ✅ DO NOT STRIP — preserve whitespace exactly
            if ts_style == 0 or d.base_ts_us == 0:
                out_lines.append(body)
            else:
                ts_str = _format_us_to_iso(ts_us)
                if ts_style == 1:
                    out_lines.append(f"[{ts_str}] {body}")
                else:
                    out_lines.append(f"{ts_str} {body}")

    return out_lines
