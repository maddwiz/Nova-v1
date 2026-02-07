import struct
from dataclasses import dataclass
from typing import Dict, List, Tuple
from bisect import bisect_left

try:
    import zstandard as zstd
except Exception:
    zstd = None

from usc.api.hdfs_template_codec_v1_channels_mask import encode_template_channels_v1_mask


MAGIC = b"TPF1"
VERSION = 1


# ----------------------------
# uvarint helpers
# ----------------------------

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


def _uvarint_decode(buf: bytes, off: int) -> Tuple[int, int]:
    x = 0
    shift = 0
    while True:
        if off >= len(buf):
            raise ValueError("uvarint overflow")
        b = buf[off]
        off += 1
        x |= (b & 0x7F) << shift
        if not (b & 0x80):
            break
        shift += 7
    return x, off


def _bytes_encode(b: bytes) -> bytes:
    return _uvarint_encode(len(b)) + b


def _bytes_decode(buf: bytes, off: int) -> Tuple[bytes, int]:
    n, off = _uvarint_decode(buf, off)
    if off + n > len(buf):
        raise ValueError("bytes decode overflow")
    return buf[off:off+n], off+n


# ----------------------------
# template reconstruction
# ----------------------------

def render_template(template: str, params: List[str]) -> str:
    out = template
    for p in params:
        out = out.replace("[*]", p, 1)
    return out


def load_template_map_from_csv_text(csv_text: str) -> Dict[int, str]:
    import csv
    from io import StringIO
    reader = csv.DictReader(StringIO(csv_text))
    tmap: Dict[int, str] = {}
    for row in reader:
        eid_raw = (row.get("EventId") or "").strip()
        tpl = (row.get("EventTemplate") or "").strip()
        if not eid_raw or not tpl:
            continue
        eid = int(eid_raw[1:]) if eid_raw.startswith("E") else int(eid_raw)
        tmap[eid] = tpl
    return tmap


# ----------------------------
# decode helpers (FULL type support)
# ----------------------------

def _zigzag_decode(u: int) -> int:
    return (u >> 1) ^ (-(u & 1))


def _decode_int_stream_gen(payload: bytes):
    off = 0
    n, off = _uvarint_decode(payload, off)
    if n == 0:
        return
    first_u, off = _uvarint_decode(payload, off)
    prev = _zigzag_decode(first_u)
    yield prev
    for _ in range(n - 1):
        du, off = _uvarint_decode(payload, off)
        prev = prev + _zigzag_decode(du)
        yield prev


def _decode_raw_stream_gen(payload: bytes):
    off = 0
    n, off = _uvarint_decode(payload, off)
    for _ in range(n):
        b, off = _bytes_decode(payload, off)
        yield b.decode("utf-8", errors="replace")


def _decode_hex_stream_gen(payload: bytes):
    off = 0
    n, off = _uvarint_decode(payload, off)
    for _ in range(n):
        b, off = _bytes_decode(payload, off)
        yield b.hex()


def _decode_ip_stream_gen(payload: bytes):
    off = 0
    n, off = _uvarint_decode(payload, off)
    for _ in range(n):
        if off + 4 > len(payload):
            raise ValueError("ip decode overflow")
        b = payload[off:off+4]
        off += 4
        yield ".".join(str(x) for x in b)


def _decode_dict_stream_gen(payload: bytes):
    off = 0
    nvals, off = _uvarint_decode(payload, off)
    vocab_n, off = _uvarint_decode(payload, off)
    vocab: List[str] = []
    for _ in range(vocab_n):
        b, off = _bytes_decode(payload, off)
        vocab.append(b.decode("utf-8", errors="replace"))
    for _ in range(nvals):
        idx, off = _uvarint_decode(payload, off)
        if vocab:
            yield vocab[min(idx, len(vocab) - 1)]
        else:
            yield ""


def _bit_set(mask: bytes, i: int) -> bool:
    return bool(mask[i // 8] & (1 << (i % 8)))


def decode_h1m1_select_params(raw: bytes, target_eid: int) -> List[List[str]]:
    """
    FAST selective decode:
      - parse EventIDs
      - find row indices where eid == target_eid
      - decode ONLY values for those rows (skip everything else)
    """
    if raw[:4] != b"H1M1":
        raise ValueError("bad H1M1 magic")

    off = 4
    _ver = struct.unpack("<I", raw[off:off+4])[0]
    off += 4

    n_events, off = _uvarint_decode(raw, off)
    n_unknown, off = _uvarint_decode(raw, off)
    max_params, off = _uvarint_decode(raw, off)

    eids: List[int] = []
    hit_rows: List[int] = []
    te = int(target_eid)

    for i in range(n_events):
        eid, off = _uvarint_decode(raw, off)
        eid = int(eid)
        eids.append(eid)
        if eid == te:
            hit_rows.append(i)

    if not hit_rows:
        # still must skip channels + unknown for caller? no, caller only needs hits
        return []

    hitset = set(hit_rows)
    hits_out: Dict[int, List[str]] = {r: [""] * max_params for r in hit_rows}

    # decode channels
    for chan in range(max_params):
        mask_len, off = _uvarint_decode(raw, off)
        mask = raw[off:off+mask_len]
        off += mask_len

        ctype, off = _uvarint_decode(raw, off)
        payload, off = _bytes_decode(raw, off)

        # if none of the hit rows are active in this channel, skip payload decode entirely ✅
        any_hit = False
        for r in hit_rows:
            if _bit_set(mask, r):
                any_hit = True
                break
        if not any_hit:
            continue

        if ctype == 1:
            gen = _decode_int_stream_gen(payload)
            for i in range(n_events):
                if _bit_set(mask, i):
                    v = next(gen)
                    if i in hitset:
                        hits_out[i][chan] = str(v)
        elif ctype == 2:
            gen = _decode_hex_stream_gen(payload)
            for i in range(n_events):
                if _bit_set(mask, i):
                    v = next(gen)
                    if i in hitset:
                        hits_out[i][chan] = v
        elif ctype == 3:
            gen = _decode_ip_stream_gen(payload)
            for i in range(n_events):
                if _bit_set(mask, i):
                    v = next(gen)
                    if i in hitset:
                        hits_out[i][chan] = v
        elif ctype == 4:
            gen = _decode_dict_stream_gen(payload)
            for i in range(n_events):
                if _bit_set(mask, i):
                    v = next(gen)
                    if i in hitset:
                        hits_out[i][chan] = v
        else:
            gen = _decode_raw_stream_gen(payload)
            for i in range(n_events):
                if _bit_set(mask, i):
                    v = next(gen)
                    if i in hitset:
                        hits_out[i][chan] = v

    # skip unknown lines block (we don't need to parse them here, but we must advance)
    for _ in range(n_unknown):
        _, off = _bytes_decode(raw, off)

    # return hit params in original order
    return [hits_out[r] for r in hit_rows]


# ----------------------------
# PF1 v1 packet format
# ----------------------------

@dataclass
class PF1Packet:
    offset: int
    length: int
    eids_sorted: List[int]


@dataclass
class PF1Index:
    templates_map: Dict[int, str]
    packets: List[PF1Packet]


@dataclass
class PF1Meta:
    blob_bytes: int
    packet_count: int
    packet_events: int
    template_bytes: int


def _zstd_compress(b: bytes, level: int) -> bytes:
    if zstd is None:
        raise RuntimeError("zstandard missing")
    return zstd.ZstdCompressor(level=level).compress(b)


def _zstd_decompress(b: bytes) -> bytes:
    if zstd is None:
        raise RuntimeError("zstandard missing")
    return zstd.ZstdDecompressor().decompress(b)


def _encode_eidset(eids: List[int]) -> bytes:
    s = sorted(set(int(x) for x in eids))
    out = bytearray()
    out += _uvarint_encode(len(s))
    prev = 0
    for x in s:
        out += _uvarint_encode(x - prev)
        prev = x
    return bytes(out)


def _decode_eidset(buf: bytes) -> List[int]:
    off = 0
    n, off = _uvarint_decode(buf, off)
    out: List[int] = []
    prev = 0
    for _ in range(n):
        d, off = _uvarint_decode(buf, off)
        prev += int(d)
        out.append(prev)
    return out


def build_tpl_pf1_blob(
    events: List[Tuple[int, List[str]]],
    unknown_lines: List[str],
    template_csv_text: str,
    packet_events: int = 32768,
    zstd_level: int = 10,
) -> Tuple[bytes, PF1Meta]:
    tpl_bytes = template_csv_text.encode("utf-8", errors="replace")

    packets: List[bytes] = []
    eidsets: List[bytes] = []

    i = 0
    n = len(events)
    while i < n:
        chunk = events[i:i+packet_events]
        i += packet_events

        ul = unknown_lines if not packets else []
        raw_struct = encode_template_channels_v1_mask(chunk, ul)
        comp = _zstd_compress(raw_struct, level=zstd_level)
        packets.append(comp)

        eids = [eid for eid, _p in chunk]
        eidsets.append(_encode_eidset(eids))

    out = bytearray()
    out += MAGIC
    out += struct.pack("<I", VERSION)
    out += struct.pack("<I", int(zstd_level))
    out += struct.pack("<I", int(packet_events))
    out += struct.pack("<I", len(tpl_bytes))
    out += tpl_bytes
    out += _uvarint_encode(len(packets))

    table_start = len(out)
    for pkt, eidset in zip(packets, eidsets):
        out += struct.pack("<I", 0)
        out += struct.pack("<I", len(pkt))
        out += _bytes_encode(eidset)

    offsets: List[int] = []
    for pkt in packets:
        offsets.append(len(out))
        out += pkt

    off = table_start
    for po in offsets:
        out[off:off+4] = struct.pack("<I", po)
        off += 4
        _plen = struct.unpack("<I", out[off:off+4])[0]
        off += 4
        _eidset, off = _bytes_decode(out, off)

    meta = PF1Meta(
        blob_bytes=len(out),
        packet_count=len(packets),
        packet_events=packet_events,
        template_bytes=len(tpl_bytes),
    )
    return bytes(out), meta


def build_pf1_index(blob: bytes) -> PF1Index:
    if blob[:4] != MAGIC:
        raise ValueError("bad PF1 magic")

    off = 4
    ver = struct.unpack("<I", blob[off:off+4])[0]
    off += 4
    if ver != 1:
        raise ValueError("wrong PF1 version")

    _zlvl = struct.unpack("<I", blob[off:off+4])[0]
    off += 4
    _pkt_events = struct.unpack("<I", blob[off:off+4])[0]
    off += 4

    tpl_len = struct.unpack("<I", blob[off:off+4])[0]
    off += 4
    tpl_bytes = blob[off:off+tpl_len]
    off += tpl_len
    templates_map = load_template_map_from_csv_text(tpl_bytes.decode("utf-8", errors="replace"))

    packet_count, off = _uvarint_decode(blob, off)

    packets: List[PF1Packet] = []
    for _ in range(packet_count):
        pkt_off = struct.unpack("<I", blob[off:off+4])[0]
        off += 4
        pkt_len = struct.unpack("<I", blob[off:off+4])[0]
        off += 4
        eidset_bytes, off = _bytes_decode(blob, off)
        eids_sorted = _decode_eidset(eidset_bytes)
        packets.append(PF1Packet(offset=int(pkt_off), length=int(pkt_len), eids_sorted=eids_sorted))

    return PF1Index(templates_map=templates_map, packets=packets)


def _sorted_contains(arr: List[int], x: int) -> bool:
    i = bisect_left(arr, x)
    return i < len(arr) and arr[i] == x


def recall_event_id_index(index: PF1Index, blob: bytes, event_id: int, limit: int = 50) -> List[str]:
    hits: List[str] = []
    eid = int(event_id)

    for pkt in index.packets:
        if not _sorted_contains(pkt.eids_sorted, eid):
            continue

        comp = blob[pkt.offset:pkt.offset+pkt.length]
        raw = _zstd_decompress(comp)

        hit_params = decode_h1m1_select_params(raw, eid)
        if not hit_params:
            continue

        tpl = index.templates_map.get(eid, "")

        for params in hit_params:
            if tpl:
                hits.append(render_template(tpl, params))
            else:
                hits.append(f"E{eid} " + " | ".join(params))

            if len(hits) >= limit:
                return hits

    return hits
