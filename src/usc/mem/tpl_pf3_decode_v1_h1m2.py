from __future__ import annotations

from typing import Dict, List, Tuple, Optional
import csv
import io
import re

try:
    import zstandard as zstd
except Exception:
    zstd = None

from usc.api.hdfs_template_codec_h1m2_rowmask import decode_h1m2_rowmask_blob
from usc.mem.hdfs_templates_v0 import _normalize_template


MAGIC = b"TPF3"
VERSION = 1

_WILDCARD_RE = re.compile(r"(\<\*\>|\[\*\])")


def _uvarint_decode(data: bytes, off: int = 0) -> Tuple[int, int]:
    shift = 0
    x = 0
    while True:
        if off >= len(data):
            raise ValueError("uvarint decode past end")
        b = data[off]
        off += 1
        x |= (b & 0x7F) << shift
        if (b & 0x80) == 0:
            break
        shift += 7
        if shift > 63:
            raise ValueError("uvarint too large")
    return x, off


def _bytes_decode(data: bytes, off: int) -> Tuple[bytes, int]:
    n, off = _uvarint_decode(data, off)
    end = off + n
    if end > len(data):
        raise ValueError("bytes decode past end")
    return data[off:end], end


def _zstd_decompress(buf: bytes, raw_len: int) -> bytes:
    if zstd is None:
        raise RuntimeError("zstandard missing (pip install zstandard)")
    d = zstd.ZstdDecompressor()
    out = d.decompress(buf, max_output_size=max(raw_len, 1) * 4 + 1024)
    return out


def _parse_templates_csv_text(tpl_text: str) -> Dict[int, str]:
    """
    tpl_text is the raw CSV file content stored in PF3 header.
    We map EventId -> EventTemplate.
    EventId may be 'E1', 'E000001', '1', etc.
    """
    def parse_eid(s: str) -> Optional[int]:
        if not s:
            return None
        s = s.strip()
        try:
            return int(s)
        except Exception:
            pass
        if len(s) >= 2 and s[0] in ("E", "e") and s[1:].isdigit():
            return int(s[1:])
        if s.lower().startswith("0x"):
            try:
                return int(s, 16)
            except Exception:
                return None
        return None

    mp: Dict[int, str] = {}
    f = io.StringIO(tpl_text)
    r = csv.DictReader(f)
    for row in r:
        eid_raw = (row.get("EventId") or "").strip()
        tpl = (row.get("EventTemplate") or "").strip()
        eid = parse_eid(eid_raw)
        if eid is None or not tpl:
            continue
        mp[int(eid)] = tpl
    return mp


def _render_template(tpl: str, params: List[str]) -> str:
    """
    Replace each wildcard token in order with params.
    Supports BOTH '<*>' and '[*]' wildcards (LogHub uses both).
    Must normalize template first so wildcard count matches extraction.
    """
    normalized = _normalize_template(tpl)
    out_parts: List[str] = []
    last = 0
    pidx = 0

    for m in _WILDCARD_RE.finditer(normalized):
        out_parts.append(normalized[last:m.start()])
        if pidx < len(params):
            out_parts.append(params[pidx] or "")
            pidx += 1
        else:
            out_parts.append(m.group(0))
        last = m.end()

    out_parts.append(normalized[last:])
    return "".join(out_parts)


def decode_pf3_h1m2_to_lines(pf3_blob: bytes) -> List[str]:
    """
    Decode PF3 container produced by build_tpl_pf3_blob_h1m2
    back into original lines (lossless).
    """
    off = 0
    if pf3_blob[:4] != MAGIC:
        raise ValueError("PF3: bad magic")
    off += 4

    ver, off = _uvarint_decode(pf3_blob, off)
    if ver != VERSION:
        raise ValueError(f"PF3: unsupported version {ver}")

    tpl_bytes, off = _bytes_decode(pf3_blob, off)
    tpl_text = tpl_bytes.decode("utf-8", errors="ignore")
    tpl_map = _parse_templates_csv_text(tpl_text)

    _packet_events, off = _uvarint_decode(pf3_blob, off)
    total_rows, off = _uvarint_decode(pf3_blob, off)
    chunk_count, off = _uvarint_decode(pf3_blob, off)

    out_lines: List[str] = []

    for _ in range(chunk_count):
        if off >= len(pf3_blob):
            raise ValueError("PF3: truncated chunk header")
        flags = pf3_blob[off]
        off += 1

        raw_len, off = _uvarint_decode(pf3_blob, off)
        pay_len, off = _uvarint_decode(pf3_blob, off)
        end = off + pay_len
        if end > len(pf3_blob):
            raise ValueError("PF3: chunk payload past end")
        payload = pf3_blob[off:end]
        off = end

        is_zstd = (flags & 1) == 1
        chunk_raw = _zstd_decompress(payload, raw_len) if is_zstd else payload

        rows, unknown_lines = decode_h1m2_rowmask_blob(chunk_raw)

        # Rehydrate into original lines in-order
        uidx = 0
        for r in rows:
            if r is None:
                line = unknown_lines[uidx] if uidx < len(unknown_lines) else ""
                uidx += 1
                out_lines.append(line)
            else:
                eid, params = r
                tpl = tpl_map.get(int(eid))
                if tpl is None:
                    out_lines.append(f"[EID={eid}] " + " | ".join(params))
                else:
                    out_lines.append(_render_template(tpl, params))

    if len(out_lines) > total_rows:
        out_lines = out_lines[:total_rows]

    return out_lines
