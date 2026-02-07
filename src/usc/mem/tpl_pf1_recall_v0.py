import struct
import hashlib
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

try:
    import zstandard as zstd
except Exception:
    zstd = None

from usc.api.hdfs_template_codec_v1_channels_mask import encode_template_channels_v1_mask


MAGIC = b"TPF1"
VERSION = 0


# ============================
# varint helpers
# ============================

def _uvarint_decode(buf: bytes, off: int) -> Tuple[int, int]:
    x = 0
    shift = 0
    while True:
        if off >= len(buf):
            raise ValueError("uvarint decode overflow")
        b = buf[off]
        off += 1
        x |= (b & 0x7F) << shift
        if not (b & 0x80):
            break
        shift += 7
    return x, off


def _uvarint_encode(x: int) -> bytes:
    out = bytearray()
    x = int(x)
    while True:
        b = x & 0x7F
        x >>= 7
        if x:
            out.append(b | 0x80)
        else:
            out.append(b)
            break
    return bytes(out)


def _bytes_decode(buf: bytes, off: int) -> Tuple[bytes, int]:
    n, off = _uvarint_decode(buf, off)
    if off + n > len(buf):
        raise ValueError("bytes decode overflow")
    return buf[off:off+n], off+n


def _str_decode(buf: bytes, off: int) -> Tuple[str, int]:
    b, off = _bytes_decode(buf, off)
    return b.decode("utf-8", errors="replace"), off


# ============================
# bloom helpers
# ============================

def _blake64(data: bytes) -> int:
    h = hashlib.blake2b(data, digest_size=8).digest()
    return struct.unpack("<Q", h)[0]


def bloom_make(m_bits: int) -> bytearray:
    return bytearray((m_bits + 7) // 8)


def bloom_add(bloom: bytearray, m_bits: int, k: int, token: int) -> None:
    base = token.to_bytes(8, "little", signed=False)
    h0 = _blake64(base)
    h1 = _blake64(base + b"\x01")
    for i in range(k):
        idx = (h0 + i * h1) % m_bits
        bloom[idx // 8] |= (1 << (idx % 8))


def bloom_has(bloom: bytes, m_bits: int, k: int, token: int) -> bool:
    base = token.to_bytes(8, "little", signed=False)
    h0 = _blake64(base)
    h1 = _blake64(base + b"\x01")
    for i in range(k):
        idx = (h0 + i * h1) % m_bits
        if not (bloom[idx // 8] & (1 << (idx % 8))):
            return False
    return True


# ============================
# template reconstruction
# ============================

def render_template(template: str, params: List[str]) -> str:
    # replace [*] sequentially
    out = template
    for p in params:
        out = out.replace("[*]", p, 1)
    return out


def load_template_map_from_csv_text(csv_text: str) -> Dict[int, str]:
    # CSV columns: EventId, EventTemplate (LogHub format)
    import csv
    from io import StringIO
    reader = csv.DictReader(StringIO(csv_text))
    tpl: Dict[int, str] = {}
    for row in reader:
        eid_raw = (row.get("EventId") or "").strip()
        etpl = (row.get("EventTemplate") or "").strip()
        if not eid_raw or not etpl:
            continue
        if eid_raw.startswith("E"):
            eid = int(eid_raw[1:])
        else:
            eid = int(eid_raw)
        tpl[eid] = etpl
    return tpl


# ============================
# V1M raw decode (H1M1)
# ============================

def _decode_svarint_to_int(u: int) -> int:
    # undo zigzag
    return (u >> 1) ^ (-(u & 1))


def _decode_int_stream(buf: bytes, off: int) -> Tuple[List[int], int]:
    n, off = _uvarint_decode(buf, off)
    out: List[int] = []
    if n == 0:
        return out, off
    first_u, off = _uvarint_decode(buf, off)
    out.append(_decode_svarint_to_int(first_u))
    prev = out[0]
    for _ in range(n - 1):
        du, off = _uvarint_decode(buf, off)
        d = _decode_svarint_to_int(du)
        prev = prev + d
        out.append(prev)
    return out, off


def _decode_ip_stream(buf: bytes, off: int) -> Tuple[List[str], int]:
    n, off = _uvarint_decode(buf, off)
    out: List[str] = []
    for _ in range(n):
        if off + 4 > len(buf):
            raise ValueError("ip decode overflow")
        b = buf[off:off+4]
        off += 4
        out.append(".".join(str(x) for x in b))
    return out, off


def _decode_hex_stream(buf: bytes, off: int) -> Tuple[List[str], int]:
    n, off = _uvarint_decode(buf, off)
    out: List[str] = []
    for _ in range(n):
        b, off = _bytes_decode(buf, off)
        out.append(b.hex())
    return out, off


def _decode_raw_stream(buf: bytes, off: int) -> Tuple[List[str], int]:
    n, off = _uvarint_decode(buf, off)
    out: List[str] = []
    for _ in range(n):
        s, off = _str_decode(buf, off)
        out.append(s)
    return out, off


def _decode_dict_stream(buf: bytes, off: int) -> Tuple[List[str], int]:
    nvals, off = _uvarint_decode(buf, off)
    vocab_n, off = _uvarint_decode(buf, off)
    vocab: List[str] = []
    for _ in range(vocab_n):
        s, off = _str_decode(buf, off)
        vocab.append(s)
    out: List[str] = []
    for _ in range(nvals):
        idx, off = _uvarint_decode(buf, off)
        if vocab:
            out.append(vocab[min(idx, len(vocab)-1)])
        else:
            out.append("")
    return out, off


def decode_h1m1_raw_struct(raw_struct: bytes) -> Tuple[List[Tuple[int, List[str]]], List[str]]:
    """
    Decode the raw_struct created by encode_template_channels_v1_mask().
    Returns list of (event_id, params) and unknown lines.
    """
    if raw_struct[:4] != b"H1M1":
        raise ValueError("bad H1M1 magic")

    off = 4
    _ver = struct.unpack("<I", raw_struct[off:off+4])[0]
    off += 4

    n_events, off = _uvarint_decode(raw_struct, off)
    n_unknown, off = _uvarint_decode(raw_struct, off)
    max_params, off = _uvarint_decode(raw_struct, off)

    # event ids (uvarint per row)
    eids: List[int] = []
    for _ in range(n_events):
        eid, off = _uvarint_decode(raw_struct, off)
        eids.append(int(eid))

    # init params matrix with empty strings
    params_matrix: List[List[str]] = [[""] * max_params for _ in range(n_events)]

    # each channel: mask_len, mask_bytes, type, payload
    for chan_i in range(max_params):
        mask_len, off = _uvarint_decode(raw_struct, off)
        if off + mask_len > len(raw_struct):
            raise ValueError("mask overflow")
        mask = raw_struct[off:off+mask_len]
        off += mask_len

        ctype, off = _uvarint_decode(raw_struct, off)
        payload_bytes, off = _bytes_decode(raw_struct, off)

        poff = 0
        if ctype == 1:
            vals_i, _ = _decode_int_stream(payload_bytes, poff)
            vals = [str(x) for x in vals_i]
        elif ctype == 2:
            vals, _ = _decode_hex_stream(payload_bytes, poff)
        elif ctype == 3:
            vals, _ = _decode_ip_stream(payload_bytes, poff)
        elif ctype == 4:
            vals, _ = _decode_dict_stream(payload_bytes, poff)
        else:
            vals, _ = _decode_raw_stream(payload_bytes, poff)

        # apply vals to non-empty positions
        vi = 0
        for row in range(n_events):
            if mask[row // 8] & (1 << (row % 8)):
                if vi < len(vals):
                    params_matrix[row][chan_i] = vals[vi]
                vi += 1

    events = [(eids[i], params_matrix[i]) for i in range(n_events)]

    unknown: List[str] = []
    for _ in range(n_unknown):
        s, off = _str_decode(raw_struct, off)
        unknown.append(s)

    return events, unknown


# ============================
# PF1 packet format
# ============================

@dataclass
class PF1Meta:
    blob_bytes: int
    packet_count: int
    packet_events: int
    bloom_bits: int
    bloom_k: int
    template_bytes: int


def _zstd_compress(b: bytes, level: int) -> bytes:
    if zstd is None:
        raise RuntimeError("zstandard not installed (pip install zstandard)")
    return zstd.ZstdCompressor(level=level).compress(b)


def _zstd_decompress(b: bytes) -> bytes:
    if zstd is None:
        raise RuntimeError("zstandard not installed (pip install zstandard)")
    return zstd.ZstdDecompressor().decompress(b)


def build_tpl_pf1_blob(
    events: List[Tuple[int, List[str]]],
    unknown_lines: List[str],
    template_csv_text: str,
    packet_events: int = 4096,
    zstd_level: int = 10,
    bloom_bits: int = 4096,
    bloom_k: int = 3,
) -> Tuple[bytes, PF1Meta]:
    """
    Creates a PF1 self-contained blob:
      - embedded templates CSV
      - packetized compressed event chunks
      - bloom filter per packet over EventIDs in that packet

    Layout:
      MAGIC(4) 'TPF1'
      u32 VERSION
      u32 zstd_level
      u32 packet_events
      u32 bloom_bits
      u32 bloom_k
      u32 template_csv_len
      template_csv_bytes
      uvarint packet_count
      [packet_table]
        for each packet:
          u32 offset
          u32 length
          u32 bloom_len
          bloom_bytes
      [packet_bytes...]
    """
    tpl_bytes = template_csv_text.encode("utf-8", errors="replace")

    packets: List[bytes] = []
    blooms: List[bytes] = []

    # include unknown lines in packet 0 only (simple)
    unknown_left = unknown_lines[:]

    i = 0
    n = len(events)
    while i < n:
        chunk = events[i:i+packet_events]
        i += packet_events

        # raw struct (H1M1) for this chunk
        # unknown only on first packet to keep it simple
        ul = unknown_left if packets == [] else []
        raw_struct = encode_template_channels_v1_mask(chunk, ul)

        comp = _zstd_compress(raw_struct, level=zstd_level)
        packets.append(comp)

        # bloom over event_ids in this packet
        bl = bloom_make(bloom_bits)
        for eid, _params in chunk:
            bloom_add(bl, bloom_bits, bloom_k, int(eid))
        blooms.append(bytes(bl))

    # header build
    out = bytearray()
    out += MAGIC
    out += struct.pack("<I", VERSION)
    out += struct.pack("<I", int(zstd_level))
    out += struct.pack("<I", int(packet_events))
    out += struct.pack("<I", int(bloom_bits))
    out += struct.pack("<I", int(bloom_k))
    out += struct.pack("<I", len(tpl_bytes))
    out += tpl_bytes

    out += _uvarint_encode(len(packets))

    # reserve table
    table_start = len(out)
    # each entry: offset(4) + length(4) + bloom_len(4) + bloom_bytes
    # bloom_bytes len = (bloom_bits+7)//8 fixed, but store anyway
    for p, bl in zip(packets, blooms):
        out += struct.pack("<I", 0)  # offset placeholder
        out += struct.pack("<I", len(p))
        out += struct.pack("<I", len(bl))
        out += bl

    # write packet bytes and patch offsets
    packet_offsets: List[int] = []
    for p in packets:
        packet_offsets.append(len(out))
        out += p

    # patch offsets
    off = table_start
    for po in packet_offsets:
        out[off:off+4] = struct.pack("<I", po)
        off += 4
        _plen = struct.unpack("<I", out[off:off+4])[0]
        off += 4
        blen = struct.unpack("<I", out[off:off+4])[0]
        off += 4 + blen

    meta = PF1Meta(
        blob_bytes=len(out),
        packet_count=len(packets),
        packet_events=packet_events,
        bloom_bits=bloom_bits,
        bloom_k=bloom_k,
        template_bytes=len(tpl_bytes),
    )
    return bytes(out), meta


@dataclass
class PF1Header:
    zstd_level: int
    packet_events: int
    bloom_bits: int
    bloom_k: int
    templates_csv: str
    packet_count: int
    table: List[Tuple[int, int, bytes]]  # (offset, length, bloom_bytes)


def parse_pf1_header(blob: bytes) -> PF1Header:
    if blob[:4] != MAGIC:
        raise ValueError("bad PF1 magic")

    off = 4
    _ver = struct.unpack("<I", blob[off:off+4])[0]
    off += 4
    zstd_level = struct.unpack("<I", blob[off:off+4])[0]
    off += 4
    packet_events = struct.unpack("<I", blob[off:off+4])[0]
    off += 4
    bloom_bits = struct.unpack("<I", blob[off:off+4])[0]
    off += 4
    bloom_k = struct.unpack("<I", blob[off:off+4])[0]
    off += 4
    tpl_len = struct.unpack("<I", blob[off:off+4])[0]
    off += 4
    tpl_bytes = blob[off:off+tpl_len]
    off += tpl_len

    packet_count, off = _uvarint_decode(blob, off)

    table: List[Tuple[int, int, bytes]] = []
    for _ in range(packet_count):
        pkt_off = struct.unpack("<I", blob[off:off+4])[0]
        off += 4
        pkt_len = struct.unpack("<I", blob[off:off+4])[0]
        off += 4
        blen = struct.unpack("<I", blob[off:off+4])[0]
        off += 4
        bl = blob[off:off+blen]
        off += blen
        table.append((pkt_off, pkt_len, bl))

    return PF1Header(
        zstd_level=int(zstd_level),
        packet_events=int(packet_events),
        bloom_bits=int(bloom_bits),
        bloom_k=int(bloom_k),
        templates_csv=tpl_bytes.decode("utf-8", errors="replace"),
        packet_count=int(packet_count),
        table=table,
    )


def recall_event_id(blob: bytes, event_id: int, limit: int = 50) -> List[str]:
    """
    Selective recall:
      - bloom scan packets
      - decode only candidate packets
      - reconstruct lines from templates + params
    """
    hdr = parse_pf1_header(blob)
    tmap = load_template_map_from_csv_text(hdr.templates_csv)

    hits: List[str] = []
    eid = int(event_id)

    for (pkt_off, pkt_len, bloom_bytes) in hdr.table:
        if not bloom_has(bloom_bytes, hdr.bloom_bits, hdr.bloom_k, eid):
            continue

        comp = blob[pkt_off:pkt_off+pkt_len]
        raw = _zstd_decompress(comp)
        events, unknown = decode_h1m1_raw_struct(raw)

        # unknown lines (if any)
        # (HDFS normally has 0)
        for ln in unknown:
            # unknown has no EventID; ignore for EventID recall
            pass

        for peid, params in events:
            if peid == eid:
                tpl = tmap.get(peid, "")
                if tpl:
                    hits.append(render_template(tpl, params))
                else:
                    hits.append(f"E{peid} " + " | ".join(params))
                if len(hits) >= limit:
                    return hits

    return hits
