from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional

from usc.mem.varint import encode_uvarint, decode_uvarint
from usc.mem.zstd_codec import zstd_compress, zstd_decompress

from drain3 import TemplateMiner
from drain3.template_miner_config import TemplateMinerConfig

from usc.mem.template_miner_drain3 import _extract_params_from_template


MAGIC_DICT = b"USDICT3D"
MAGIC_DATA = b"USDAT3D7"


def _pack_string(s: str) -> bytes:
    b = s.encode("utf-8")
    return encode_uvarint(len(b)) + b


def _unpack_string(data: bytes, offset: int) -> Tuple[str, int]:
    n, off = decode_uvarint(data, offset)
    b = data[off:off + n]
    off += n
    return b.decode("utf-8"), off


def _bits_needed(n: int) -> int:
    if n <= 1:
        return 1
    b = 0
    x = n - 1
    while x > 0:
        b += 1
        x >>= 1
    return max(1, b)


def _bitpack(values: List[int], bits: int) -> bytes:
    if bits <= 0:
        return b""
    out = bytearray()
    acc = 0
    acc_bits = 0
    mask = (1 << bits) - 1
    for v in values:
        v &= mask
        acc |= (v << acc_bits)
        acc_bits += bits
        while acc_bits >= 8:
            out.append(acc & 0xFF)
            acc >>= 8
            acc_bits -= 8
    if acc_bits > 0:
        out.append(acc & 0xFF)
    return bytes(out)


def _bitunpack(data: bytes, count: int, bits: int) -> List[int]:
    if bits <= 0:
        return [0] * count
    out: List[int] = []
    acc = 0
    acc_bits = 0
    idx = 0
    mask = (1 << bits) - 1
    for _ in range(count):
        while acc_bits < bits:
            if idx >= len(data):
                raise ValueError("bitunpack ran out of bytes")
            acc |= data[idx] << acc_bits
            idx += 1
            acc_bits += 8
        out.append(acc & mask)
        acc >>= bits
        acc_bits -= bits
    return out


def _mtf_pos_and_move(mtf: List[int], idx: int) -> int:
    pos = mtf.index(idx)
    mtf.pop(pos)
    mtf.insert(0, idx)
    return pos


def _mtf_get_and_move(mtf: List[int], pos: int) -> int:
    idx = mtf[pos]
    mtf.pop(pos)
    mtf.insert(0, idx)
    return idx


def _drain_to_format(template_mined: str) -> str:
    return template_mined.replace("<*>", "{}")


def _make_miner() -> TemplateMiner:
    cfg = TemplateMinerConfig()
    cfg.profiling_enabled = False
    return TemplateMiner(config=cfg)


def _mine_chunks_stateful(miner: TemplateMiner, chunks: List[str]) -> Tuple[List[str], List[List[str]]]:
    chunk_templates: List[str] = []
    chunk_params: List[List[str]] = []

    for ch in chunks:
        ends_with_newline = ch.endswith("\n")
        lines = ch.splitlines(keepends=False)

        line_templates: List[str] = []
        all_params: List[str] = []

        for line in lines:
            res = miner.add_log_message(line)
            tmpl = res["template_mined"]
            params = _extract_params_from_template(line, tmpl)
            line_templates.append(tmpl)
            all_params.extend(params)

        rebuilt = "\n".join(line_templates)
        if ends_with_newline:
            rebuilt += "\n"

        chunk_templates.append(rebuilt)
        chunk_params.append(all_params)

    return chunk_templates, chunk_params


def _zigzag_encode(x: int) -> int:
    return (x << 1) ^ (x >> 63)


def _zigzag_decode(z: int) -> int:
    return (z >> 1) ^ (-(z & 1))


def _looks_int(s: str) -> bool:
    if not s:
        return False
    if s[0] == "-" and s[1:].isdigit():
        return True
    return s.isdigit()


PT_NUM = 0
PT_STR = 1

@dataclass
class StreamStateV3D7:
    templates: List[str] = field(default_factory=list)
    temp_index: Dict[str, int] = field(default_factory=dict)
    mtf: List[int] = field(default_factory=list)

    # slot dicts
    slot_strings: Dict[Tuple[int, int], List[str]] = field(default_factory=dict)
    slot_index: Dict[Tuple[int, int], Dict[str, int]] = field(default_factory=dict)
    slot_mtf: Dict[Tuple[int, int], List[int]] = field(default_factory=dict)

    # numeric last values
    slot_last_num: Dict[Tuple[int, int], int] = field(default_factory=dict)

    miner: Optional[TemplateMiner] = None


def _slot_structs(state: StreamStateV3D7, tid: int, slot: int) -> Tuple[List[str], Dict[str, int], List[int]]:
    key = (tid, slot)
    if key not in state.slot_strings:
        state.slot_strings[key] = []
        state.slot_index[key] = {}
        state.slot_mtf[key] = []
    return state.slot_strings[key], state.slot_index[key], state.slot_mtf[key]


def build_dict_state_from_chunks(chunks: List[str], state: StreamStateV3D7) -> None:
    if state.miner is None:
        state.miner = _make_miner()

    mined_templates, _ = _mine_chunks_stateful(state.miner, chunks)

    for tmpl_mined in mined_templates:
        fmt_t = _drain_to_format(tmpl_mined)
        if fmt_t not in state.temp_index:
            tid = len(state.templates)
            state.templates.append(fmt_t)
            state.temp_index[fmt_t] = tid

    state.mtf = list(range(len(state.templates)))


def encode_dict_packet(state: StreamStateV3D7, level: int = 10) -> bytes:
    out = bytearray()
    out += MAGIC_DICT
    out += encode_uvarint(len(state.templates))
    for t in state.templates:
        out += _pack_string(t)
    return zstd_compress(bytes(out), level=level)


def apply_dict_packet(packet: bytes, state: StreamStateV3D7) -> None:
    raw = zstd_decompress(packet)
    if not raw.startswith(MAGIC_DICT):
        raise ValueError("not a v3d DICT packet")

    off = len(MAGIC_DICT)
    ntemp, off = decode_uvarint(raw, off)

    state.templates = []
    state.temp_index = {}
    for _ in range(ntemp):
        t, off = _unpack_string(raw, off)
        tid = len(state.templates)
        state.templates.append(t)
        state.temp_index[t] = tid

    state.mtf = list(range(len(state.templates)))

    # reset slot dicts
    state.slot_strings = {}
    state.slot_index = {}
    state.slot_mtf = {}
    state.slot_last_num = {}

    if state.miner is None:
        state.miner = _make_miner()

def encode_data_packet(chunks: List[str], state: StreamStateV3D7, level: int = 10) -> bytes:
    if state.miner is None:
        state.miner = _make_miner()

    mined_templates, mined_params = _mine_chunks_stateful(state.miner, chunks)

    # template refresh
    new_templates: List[str] = []
    for tmpl_mined in mined_templates:
        fmt_t = _drain_to_format(tmpl_mined)
        if fmt_t not in state.temp_index:
            tid = len(state.templates)
            state.templates.append(fmt_t)
            state.temp_index[fmt_t] = tid
            new_templates.append(fmt_t)

    if not state.mtf:
        state.mtf = list(range(len(state.templates)))
    else:
        cur = len(state.mtf)
        if len(state.templates) > cur:
            state.mtf.extend(range(cur, len(state.templates)))

    # template positions
    tid_positions: List[int] = []
    arities: List[int] = []
    tids_for_chunks: List[int] = []

    for tmpl_mined in mined_templates:
        fmt_t = _drain_to_format(tmpl_mined)
        tid = state.temp_index[fmt_t]
        tids_for_chunks.append(tid)
        tid_positions.append(_mtf_pos_and_move(state.mtf, tid))
        arities.append(fmt_t.count("{}"))

    t_bits = _bits_needed(max(1, len(state.templates)))
    tpos_bytes = _bitpack(tid_positions, t_bits)

    # slot refresh records + param stream
    slot_new_strings: Dict[Tuple[int, int], List[str]] = {}
    param_stream = bytearray()

    for tid, tmpl_mined, params in zip(tids_for_chunks, mined_templates, mined_params):
        fmt_t = _drain_to_format(tmpl_mined)
        arity = fmt_t.count("{}")

        if len(params) < arity:
            params = params + [""] * (arity - len(params))
        elif len(params) > arity:
            params = params[:arity]

        for slot_i, p in enumerate(params):
            key = (tid, slot_i)

            if _looks_int(p):
                curv = int(p)
                lastv = state.slot_last_num.get(key, 0)
                delta = curv - lastv
                state.slot_last_num[key] = curv

                param_stream += encode_uvarint(PT_NUM)
                param_stream += encode_uvarint(_zigzag_encode(delta))
            else:
                slist, sindex, smtf = _slot_structs(state, tid, slot_i)

                if p not in sindex:
                    sid = len(slist)
                    slist.append(p)
                    sindex[p] = sid
                    if not smtf:
                        smtf.extend(range(len(slist)))
                    else:
                        smtf.append(sid)

                    slot_new_strings.setdefault((tid, slot_i), []).append(p)

                sid = sindex[p]
                pos = _mtf_pos_and_move(smtf, sid)

                param_stream += encode_uvarint(PT_STR)
                param_stream += encode_uvarint(pos)

    out = bytearray()
    out += MAGIC_DATA
    out += encode_uvarint(len(chunks))

    # template refresh
    out += encode_uvarint(len(new_templates))
    for t in new_templates:
        out += _pack_string(t)

    # template positions
    out += encode_uvarint(t_bits)
    out += encode_uvarint(len(tpos_bytes))
    out += tpos_bytes

    # arities
    out += encode_uvarint(len(arities))
    for a in arities:
        out += encode_uvarint(a)

    # slot refresh
    out += encode_uvarint(len(slot_new_strings))
    for (tid, slot), strs in slot_new_strings.items():
        out += encode_uvarint(tid)
        out += encode_uvarint(slot)
        out += encode_uvarint(len(strs))
        for s in strs:
            out += _pack_string(s)

    # param stream
    out += encode_uvarint(len(param_stream))
    out += bytes(param_stream)

    return zstd_compress(bytes(out), level=level)


def decode_data_packet(packet: bytes, state: StreamStateV3D7) -> List[str]:
    raw = zstd_decompress(packet)
    if not raw.startswith(MAGIC_DATA):
        raise ValueError("not a v3d7 DATA packet")

    off = len(MAGIC_DATA)
    n_chunks, off = decode_uvarint(raw, off)

    # template refresh
    nnewt, off = decode_uvarint(raw, off)
    for _ in range(nnewt):
        t, off = _unpack_string(raw, off)
        if t not in state.temp_index:
            tid = len(state.templates)
            state.templates.append(t)
            state.temp_index[t] = tid

    if not state.mtf:
        state.mtf = list(range(len(state.templates)))
    else:
        cur = len(state.mtf)
        if len(state.templates) > cur:
            state.mtf.extend(range(cur, len(state.templates)))

    # template positions
    t_bits, off = decode_uvarint(raw, off)
    tpos_len, off = decode_uvarint(raw, off)
    tpos_bytes = raw[off:off + tpos_len]
    off += tpos_len
    tid_positions = _bitunpack(tpos_bytes, n_chunks, t_bits)

    # arities
    n_ar, off = decode_uvarint(raw, off)
    arities: List[int] = []
    for _ in range(n_ar):
        a, off = decode_uvarint(raw, off)
        arities.append(a)

    # slot refresh
    nslot_updates, off = decode_uvarint(raw, off)
    for _ in range(nslot_updates):
        tid, off = decode_uvarint(raw, off)
        slot, off = decode_uvarint(raw, off)
        nnew, off = decode_uvarint(raw, off)

        slist, sindex, smtf = _slot_structs(state, tid, slot)

        for _ in range(nnew):
            s, off = _unpack_string(raw, off)
            if s not in sindex:
                sid = len(slist)
                slist.append(s)
                sindex[s] = sid
                if not smtf:
                    smtf.extend(range(len(slist)))
                else:
                    smtf.append(sid)

    # param stream
    p_len, off = decode_uvarint(raw, off)
    p_raw = raw[off:off + p_len]
    poff = 0

    out_chunks: List[str] = []

    for i in range(n_chunks):
        tid = _mtf_get_and_move(state.mtf, tid_positions[i])
        template = state.templates[tid]
        arity = arities[i]

        vals: List[str] = []
        for slot_i in range(arity):
            ptype, poff = decode_uvarint(p_raw, poff)
            if ptype == PT_NUM:
                z, poff = decode_uvarint(p_raw, poff)
                delta = _zigzag_decode(z)
                key = (tid, slot_i)
                lastv = state.slot_last_num.get(key, 0)
                curv = lastv + delta
                state.slot_last_num[key] = curv
                vals.append(str(curv))
            else:
                pos, poff = decode_uvarint(p_raw, poff)
                slist, sindex, smtf = _slot_structs(state, tid, slot_i)
                sid = _mtf_get_and_move(smtf, pos)
                vals.append(slist[sid])

        out_chunks.append(template.format(*vals))

    return out_chunks
