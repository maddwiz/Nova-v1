from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import re
from datetime import datetime, timezone


MAGIC_SASD = b"USC_SASD6"  # dict packet v6 (tools + token dict)
MAGIC_SASA = b"USC_SASA6"  # data packet v6 (dt + ts_style + tool_id + rid16 + payload_tok)


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
# Line parsers (2 timestamp styles)
# -----------------------------
RE_BRACKET = re.compile(r"^\[(?P<ts>[^\]]+)\]\s+(?P<body>.*)$")
RE_BARE = re.compile(r"^(?P<ts>\d{4}-\d{2}-\d{2}T[0-9:\.]+(?:Z|[+\-]\d{2}:\d{2}))\s+(?P<body>.*)$")

RE_TOOL_CALL = re.compile(
    r"^tool_call::(?P<tool>\S+)\s+rid=(?P<rid>[0-9a-fA-F\-]{36})\s+payload=(?P<payload>\{.*\})$"
)

# ts_style: 0 none, 1 bracket, 2 bare
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
# Token dict mining (JSON keys + common literals)
# -----------------------------
RE_JSON_KEY = re.compile(r'"([^"\\]{1,40})"\s*:')

DEFAULT_LITERALS = [
    b'"q"', b'"query"', b'"recency"', b'"domains"', b'"search_query"', b'"open"', b'"click"',
    b'"ref_id"', b'"id"', b'"lineno"', b'"pageno"', b'"pattern"',
    b'"type"', b'"ticker"', b'"market"',
    b'"location"', b'"start"', b'"duration"',
    b'"response_length"', b'"short"', b'"medium"', b'"long"',
]

def _mine_tokens(payloads: List[str], top_k_keys: int = 256, add_literals: bool = True) -> List[bytes]:
    freq: Dict[str, int] = {}

    for p in payloads:
        for m in RE_JSON_KEY.finditer(p):
            k = m.group(1)
            freq[k] = freq.get(k, 0) + 1

    items = sorted(freq.items(), key=lambda kv: (-kv[1], kv[0]))
    keys = [k for k, _c in items[:top_k_keys]]

    # store as exact substring bytes: b'"key"'
    toks = [b'"' + _bstr(k) + b'"' for k in keys]

    if add_literals:
        toks.extend(DEFAULT_LITERALS)

    # unique + sorted longest-first
    uniq = list(dict.fromkeys(toks))
    uniq.sort(key=lambda b: (-len(b), b))
    return uniq


@dataclass
class SASDict:
    base_ts_us: int
    tool_to_id: Dict[str, int]
    id_to_tool: List[str]
    tok_to_id: Dict[bytes, int]
    id_to_tok: List[bytes]


def _build_dict(lines: List[str], tok_top_k: int = 256) -> SASDict:
    first_ts_us = None
    tools_set = set()
    payloads: List[str] = []

    for ln in lines:
        ts, body, style = _split_line(ln)
        if first_ts_us is None and ts is not None and style in (1, 2):
            first_ts_us = _iso_to_us(ts)

        if ts is not None:
            m = RE_TOOL_CALL.match(body)
            if m:
                tools_set.add(m.group("tool"))
                payloads.append(m.group("payload"))

    if first_ts_us is None:
        first_ts_us = 0

    tools_sorted = sorted(tools_set)
    tool_to_id = {t: i + 1 for i, t in enumerate(tools_sorted)}

    toks = _mine_tokens(payloads, top_k_keys=tok_top_k, add_literals=True)
    tok_to_id = {b: i + 1 for i, b in enumerate(toks)}  # id 1..m

    return SASDict(
        base_ts_us=int(first_ts_us),
        tool_to_id=tool_to_id,
        id_to_tool=tools_sorted,
        tok_to_id=tok_to_id,
        id_to_tok=toks,
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

    # tokens
    out += _uvarint_encode(len(d.id_to_tok))
    for tb in d.id_to_tok:
        out += _uvarint_encode(len(tb))
        out += tb

    return bytes(out)


def _decode_dict_packet(pkt: bytes) -> SASDict:
    if not pkt.startswith(MAGIC_SASD):
        raise ValueError("bad SAS dict magic")
    off = len(MAGIC_SASD)

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

    n_tok, off = _uvarint_decode(pkt, off)
    id_to_tok: List[bytes] = []
    tok_to_id: Dict[bytes, int] = {}
    for i in range(n_tok):
        ln, off = _uvarint_decode(pkt, off)
        tb = pkt[off:off + ln]
        off += ln
        tid = i + 1
        id_to_tok.append(tb)
        tok_to_id[tb] = tid

    return SASDict(
        base_ts_us=int(base_ts_us),
        tool_to_id=tool_to_id,
        id_to_tool=id_to_tool,
        tok_to_id=tok_to_id,
        id_to_tok=id_to_tok,
    )


# -----------------------------
# Tokenize payload with marker 0x00
# -----------------------------
MARK = 0

def _tokenize_payload(d: SASDict, payload: bytes) -> bytes:
    """
    Replace any token substring with: 0x00 + uvarint(id)
    Escape literal 0x00 bytes as: 0x00 + uvarint(0)
    """
    if not d.id_to_tok:
        return payload

    toks = d.id_to_tok  # longest-first
    out = bytearray()

    i = 0
    n = len(payload)

    while i < n:
        b = payload[i]

        # escape literal 0x00
        if b == 0:
            out.append(0)
            out += _uvarint_encode(0)
            i += 1
            continue

        matched = None
        for tb in toks:
            L = len(tb)
            if L and i + L <= n and payload[i:i+L] == tb:
                matched = tb
                break

        if matched is None:
            out.append(b)
            i += 1
        else:
            out.append(0)
            out += _uvarint_encode(d.tok_to_id[matched])
            i += len(matched)

    return bytes(out)


def _detokenize_payload(d: SASDict, tok: bytes) -> bytes:
    """
    Parse stream; 0x00 + id:
      id==0 => literal 0x00
      id>=1 => dictionary token bytes
    """
    out = bytearray()
    i = 0
    n = len(tok)

    while i < n:
        b = tok[i]
        if b != 0:
            out.append(b)
            i += 1
            continue

        # marker
        i += 1
        if i >= n:
            break
        tid, i2 = _uvarint_decode(tok, i)
        i = i2

        if tid == 0:
            out.append(0)
        else:
            if 1 <= tid <= len(d.id_to_tok):
                out += d.id_to_tok[tid - 1]
            else:
                # bad id -> ignore
                pass

    return bytes(out)


# -----------------------------
# Data packet encode/decode
# -----------------------------
def _encode_data_packet(entries: List[Tuple[int, int, int, Optional[bytes], bytes, bytes]]) -> bytes:
    """
    entries:
      (dt_us, ts_style, tool_id, rid16_or_none, raw_body_bytes, payload_tok_bytes)

    tool_id 0 => raw_body_bytes used
    tool_id >0 => rid16 + payload_tok_bytes used
    """
    out = bytearray()
    out += MAGIC_SASA
    out += _uvarint_encode(len(entries))

    for dt_us, ts_style, tool_id, rid16, raw_body, payload_tok in entries:
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
            out += _uvarint_encode(len(payload_tok))
            out += payload_tok

    return bytes(out)


def _decode_data_packet(pkt: bytes) -> List[Tuple[int, int, int, Optional[bytes], bytes, bytes]]:
    if not pkt.startswith(MAGIC_SASA):
        raise ValueError("bad SAS data magic")
    off = len(MAGIC_SASA)

    n, off = _uvarint_decode(pkt, off)
    out: List[Tuple[int, int, int, Optional[bytes], bytes, bytes]] = []

    for _ in range(n):
        dt_us, off = _uvarint_decode(pkt, off)
        ts_style, off = _uvarint_decode(pkt, off)
        tool_id, off = _uvarint_decode(pkt, off)

        if tool_id == 0:
            ln, off = _uvarint_decode(pkt, off)
            raw = pkt[off:off + ln]
            off += ln
            out.append((int(dt_us), int(ts_style), 0, None, raw, b""))
        else:
            rid16 = pkt[off:off + 16]
            off += 16
            ln, off = _uvarint_decode(pkt, off)
            payload_tok = pkt[off:off + ln]
            off += ln
            out.append((int(dt_us), int(ts_style), int(tool_id), rid16, b"", payload_tok))

    return out


# -----------------------------
# Public API
# -----------------------------
def build_sas_packets_from_text(text: str, max_lines_per_packet: int = 60, tok_top_k: int = 256) -> List[bytes]:
    lines = text.splitlines()
    d = _build_dict(lines, tok_top_k=tok_top_k)

    packets: List[bytes] = []
    packets.append(_encode_dict_packet(d))

    chunk: List[Tuple[int, int, int, Optional[bytes], bytes, bytes]] = []
    last_ts_us = d.base_ts_us
    ts_us = d.base_ts_us

    for ln in lines:
        ts, body, ts_style = _split_line(ln)

        if ts is None:
            chunk.append((0, 0, 0, None, _bstr(body), b""))
        else:
            ts_us = _iso_to_us(ts)
            dt_us = max(0, ts_us - last_ts_us)
            last_ts_us = ts_us

            m = RE_TOOL_CALL.match(body)
            if m:
                tool = m.group("tool")
                rid = m.group("rid")
                payload = m.group("payload")

                tool_id = d.tool_to_id.get(tool, 0)
                rid16 = _uuid_to_16(rid)

                payload_b = payload.encode("utf-8", errors="replace")
                payload_tok = _tokenize_payload(d, payload_b)

                chunk.append((dt_us, ts_style, tool_id, rid16, b"", payload_tok))
            else:
                chunk.append((dt_us, ts_style, 0, None, _bstr(body), b""))

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
        for dt_us, ts_style, tool_id, rid16, raw_body, payload_tok in entries:
            if d.base_ts_us != 0 and ts_style in (1, 2):
                ts_us += dt_us

            if tool_id == 0:
                body = _s(raw_body)
            else:
                tool = d.id_to_tool[tool_id - 1] if 1 <= tool_id <= len(d.id_to_tool) else "UNKNOWN"
                rid = _u16_to_uuid(rid16 or (b"\x00" * 16))
                payload_b = _detokenize_payload(d, payload_tok)
                payload = payload_b.decode("utf-8", errors="replace")
                body = f"tool_call::{tool} rid={rid} payload={payload}"

            if ts_style == 0 or d.base_ts_us == 0:
                out_lines.append(body)
            else:
                ts_str = _format_us_to_iso(ts_us)
                if ts_style == 1:
                    out_lines.append(f"[{ts_str}] {body}")
                else:
                    out_lines.append(f"{ts_str} {body}")

    return out_lines
