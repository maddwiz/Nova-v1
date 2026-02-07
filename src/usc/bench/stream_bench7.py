from usc.bench.datasets import toy_big_agent_log_varied
from usc.bench.metrics import gzip_compress
from usc.mem.chunking import chunk_by_lines

from usc.mem.templatemtf_bits_deltaonly_canon_zstd import (
    encode_chunks_with_template_mtf_bits_deltaonly_canon as CANZ_BATCH,
)

# v3 dict+data
from usc.mem.stream_proto_canz_v3 import (
    StreamStateV3,
    build_dict_state_from_chunks as build_v3,
    encode_dict_packet as dict_v3,
    apply_dict_packet as apply_v3,
    encode_data_packet as data_v3,
)

# v3b dict+data (smaller dict)
from usc.mem.stream_proto_canz_v3b import (
    StreamStateV3B,
    build_dict_state_from_chunks as build_v3b,
    encode_dict_packet as dict_v3b,
    apply_dict_packet as apply_v3b,
    encode_data_packet as data_v3b,
)


def _ratio(raw: int, comp: int) -> float:
    return raw / max(1, comp)


def run():
    raw_big = toy_big_agent_log_varied(loops=30)
    raw_bytes = raw_big.encode("utf-8")
    gz = gzip_compress(raw_bytes)

    chunks = [c.text for c in chunk_by_lines(raw_big, max_lines=25)]
    canz_batch = CANZ_BATCH(chunks)

    # ---- v3 ----
    st3_build = StreamStateV3()
    build_v3(chunks, state=st3_build)
    pkt3_dict = dict_v3(st3_build, level=10)

    st3_send = StreamStateV3()
    apply_v3(pkt3_dict, state=st3_send)
    pkt3_data = data_v3(chunks, st3_send, level=10)
    total3 = len(pkt3_dict) + len(pkt3_data)

    # ---- v3b ----
    st3b_build = StreamStateV3B()
    build_v3b(chunks, state=st3b_build)
    pkt3b_dict = dict_v3b(st3b_build, level=10)

    st3b_send = StreamStateV3B()
    apply_v3b(pkt3b_dict, state=st3b_send)
    pkt3b_data = data_v3b(chunks, st3b_send, level=10)
    total3b = len(pkt3b_dict) + len(pkt3b_data)

    print("USC Stream Bench v7 — DICT Shrink v3b")
    print("-------------------------------------------------")
    print(f"RAW bytes       : {len(raw_bytes)}")
    print(f"GZIP bytes      : {len(gz)}   (ratio {_ratio(len(raw_bytes), len(gz)):.2f}x)")
    print(f"CANZ batch      : {len(canz_batch)}   (ratio {_ratio(len(raw_bytes), len(canz_batch)):.2f}x)")
    print("-------------------------------------------------")
    print(f"v3  DICT bytes  : {len(pkt3_dict)}")
    print(f"v3  DATA bytes  : {len(pkt3_data)}")
    print(f"v3  FIRST total : {total3}")
    print("-------------------------------------------------")
    print(f"v3b DICT bytes  : {len(pkt3b_dict)}")
    print(f"v3b DATA bytes  : {len(pkt3b_data)}")
    print(f"v3b FIRST total : {total3b}")
    print("-------------------------------------------------")
    print("✅ Goal: v3b FIRST total < v3 FIRST total")
    print("✅ Stretch: v3b FIRST total < GZIP")
    print("-------------------------------------------------")


if __name__ == "__main__":
    run()
