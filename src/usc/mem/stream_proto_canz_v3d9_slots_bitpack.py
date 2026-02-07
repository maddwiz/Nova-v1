from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional

from usc.mem.varint import encode_uvarint, decode_uvarint
from usc.mem.zstd_codec import zstd_compress, zstd_decompress

from drain3 import TemplateMiner
from drain3.template_miner_config import TemplateMinerConfig

from usc.mem.template_miner_drain3 import _extract_params_from_template


MAGIC_DICT = b"USDICT3D"
MAGIC_DATA = b"USDAT3D9"


# -------------------------
# helpers
# -------------------------
def _pack_string(s: str) -> bytes:
    b = s.encode("utf-8")
    return encode_uvarint(len(b)) + b


def _unpack_string(data: bytes, offset: int) -> Tuple[str, int]:
    n, off = decode_uvarint(data, offset)
    b = data[off:off + n]
    off += n
    return b.decode("utf-8"), off


def _bits_needed_maxval(maxv: int) -> int:
    # how many bits to represent values 0..maxv
    if maxv <= 0:
        return 1
    b = 0
    x = maxv
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


def _pack_bits_1(flags: List[int]) -> bytes:
    out = bytearray()
    cur = 0
    bit = 0
    for f in flags:
        if f:
            cur |= (1 << bit)
        bit += 1
        if bit == 8:
            out.append(cur)
            cur = 0
            bit = 0
    if bit != 0:
        out.append(cur)
    return bytes(out)


def _unpack_bits_1(data: bytes, count: int) -> List[int]:
    out: List[int] = []
    idx = 0
    bit = 0
    cur = data[0] if data else 0
    for _ in range(count):
        if idx >= len(data):
            raise ValueError("unpack_bits_1 out of data")
        out.append((cur >> bit) & 1)
        bit += 1
        if bit == 8:
            idx += 1
            bit = 0
            cur = data[idx] if idx < len(data) else 0
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


# -------------------------
# state
# -------------------------
@dataclass
class StreamStateV3D9:
    templates: List[str] = field(default_factory=list)
    temp_index: Dict[str, int] = field(default_factory=dict)
    mtf: List[int] = field(default_factory=list)

    # per-slot string dicts
    slot_strings: Dict[Tuple[int, int], List[str]] = field(default_factory=dict)
    slot_index: Dict[Tuple[int, int], Dict[str, int]] = field(default_factory=dict)
    slot_mtf: Dict[Tuple[int, int], List[int]] = field(default_factory=dict)

    # numeric last values per slot
    slot_last_num: Dict[Tuple[int, int], int] = field(default_factory=dict)

    miner: Optional[TemplateMiner] = None


def _slot_structs(state: StreamStateV3D9, tid: int, slot: int) -> Tuple[List[str], Dict[str, int], List[int]]:
    key = (tid, slot)
    if key not in state.slot_strings:
        state.slot_strings[key] = []
        state.slot_index[key] = {}
        state.slot_mtf[key] = []
    return state.slot_strings[key], state.slot_index[key], state.slot_mtf[key]


# -------------------------
# dict packet
# -------------------------
def build_dict_state_from_chunks(chunks: List[str], state: StreamStateV3D9) -> None:
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


def encode_dict_packet(state: StreamStateV3D9, level: int = 10) -> bytes:
    out = bytearray()
    out += MAGIC_DICT
    out += encode_uvarint(len(state.templates))
    for t in state.templates:
        out += _pack_string(t)
    return zstd_compress(bytes(out), level=level)


def apply_dict_packet(packet: bytes, state: StreamStateV3D9) -> None:
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


# -------------------------
# data packet (bitpacked param streams)
# -------------------------
def encode_data_packet(chunks: List[str], state: StreamStateV3D9, level: int = 10) -> bytes:
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

    t_bits = _bits_needed_maxval(max(0, len(state.templates) - 1))
    tpos_bytes = _bitpack(tid_positions, t_bits)

    # slot refresh + param collections
    slot_new_strings: Dict[Tuple[int, int], List[str]] = {}

    type_flags: List[int] = []  # 1=str, 0=num
    nums: List[int] = []
    strs: List[int] = []

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
                type_flags.append(0)
                curv = int(p)
                lastv = state.slot_last_num.get(key, 0)
                delta = curv - lastv
                state.slot_last_num[key] = curv
                nums.append(_zigzag_encode(delta))
            else:
                type_flags.append(1)
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
                strs.append(pos)

    # bitpack types
    types_bytes = _pack_bits_1(type_flags)

    # bitpack nums + strs
    num_bits = _bits_needed_maxval(max(nums) if nums else 0)
    str_bits = _bits_needed_maxval(max(strs) if strs else 0)

    nums_bytes = _bitpack(nums, num_bits)
    strs_bytes = _bitpack(strs, str_bits)

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
    for (tid, slot), ss in slot_new_strings.items():
        out += encode_uvarint(tid)
        out += encode_uvarint(slot)
        out += encode_uvarint(len(ss))
        for s in ss:
            out += _pack_string(s)

    # param types bitset
    out += encode_uvarint(len(type_flags))
    out += encode_uvarint(len(types_bytes))
    out += types_bytes

    # nums payload
    out += encode_uvarint(len(nums))
    out += encode_uvarint(num_bits)
    out += encode_uvarint(len(nums_bytes))
    out += nums_bytes

    # strs payload
    out += encode_uvarint(len(strs))
    out += encode_uvarint(str_bits)
    out += encode_uvarint(len(strs_bytes))
    out += strs_bytes

    return zstd_compress(bytes(out), level=level)


def decode_data_packet(packet: bytes, state: StreamStateV3D9) -> List[str]:
    raw = zstd_decompress(packet)
    if not raw.startswith(MAGIC_DATA):
        raise ValueError("not a v3d9 DATA packet")

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

    # param types
    n_flags, off = decode_uvarint(raw, off)
    types_len, off = decode_uvarint(raw, off)
    types_bytes = raw[off:off + types_len]
    off += types_len
    type_flags = _unpack_bits_1(types_bytes, n_flags)

    # nums
    n_nums, off = decode_uvarint(raw, off)
    num_bits, off = decode_uvarint(raw, off)
    nums_len, off = decode_uvarint(raw, off)
    nums_bytes = raw[off:off + nums_len]
    off += nums_len
    nums = _bitunpack(nums_bytes, n_nums, num_bits)

    # strs
    n_strs, off = decode_uvarint(raw, off)
    str_bits, off = decode_uvarint(raw, off)
    strs_len, off = decode_uvarint(raw, off)
    strs_bytes = raw[off:off + strs_len]
    off += strs_len
    strs = _bitunpack(strs_bytes, n_strs, str_bits)

    # reconstruct
    num_i = 0
    str_i = 0
    flag_i = 0

    out_chunks: List[str] = []

    for i in range(n_chunks):
        tid = _mtf_get_and_move(state.mtf, tid_positions[i])
        template = state.templates[tid]
        arity = arities[i]

        vals: List[str] = []
        for slot_i in range(arity):
            is_str = type_flags[flag_i]
            flag_i += 1
            key = (tid, slot_i)

            if is_str:
                pos = strs[str_i]
                str_i += 1
                slist, sindex, smtf = _slot_structs(state, tid, slot_i)
                sid = _mtf_get_and_move(smtf, pos)
                vals.append(slist[sid])
            else:
                z = nums[num_i]
                num_i += 1
                delta = _zigzag_decode(z)
                lastv = state.slot_last_num.get(key, 0)
                curv = lastv + delta
                state.slot_last_num[key] = curv
                vals.append(str(curv))

        out_chunks.append(template.format(*vals))

    return out_chunks
