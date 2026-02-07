from usc.bench.datasets import toy_big_agent_log_varied
from usc.bench.metrics import gzip_compress
from usc.mem.chunking import chunk_by_lines

# v3b (champion lossless + beats gzip)
from usc.mem.stream_proto_canz_v3b import (
    StreamStateV3B,
    build_dict_state_from_chunks as build_v3b,
    encode_dict_packet as dict_v3b,
    apply_dict_packet as apply_v3b,
    encode_data_packet as data_v3b,
)

# v3c typed (new)
from usc.mem.stream_proto_canz_v3c_typed import (
    StreamStateV3C,
    build_dict_state_from_chunks as build_v3c,
    encode_dict_packet as dict_v3c,
    apply_dict_packet as apply_v3c,
    encode_data_packet as data_v3c,
)

def _ratio(raw: int, comp: int) -> float:
    return raw / max(1, comp)

def run():
    raw_big = toy_big_agent_log_varied(loops=30)
    raw_bytes = raw_big.encode("utf-8")
    gz = gzip_compress(raw_bytes)

    chunks = [c.text for c in chunk_by_lines(raw_big, max_lines=25)]

    # ---- v3b ----
    st3b_build = StreamStateV3B()
    build_v3b(chunks, state=st3b_build)
    pkt3b_dict = dict_v3b(st3b_build, level=10)

    st3b_send = StreamStateV3B()
    apply_v3b(pkt3b_dict, state=st3b_send)
    pkt3b_data = data_v3b(chunks, st3b_send, level=10)

    total3b = len(pkt3b_dict) + len(pkt3b_data)

    # ---- v3c ----
    st3c_build = StreamStateV3C()
    build_v3c(chunks, state=st3c_build)
    pkt3c_dict = dict_v3c(st3c_build, level=10)

    st3c_send = StreamStateV3C()
    apply_v3c(pkt3c_dict, state=st3c_send)
    pkt3c_data = data_v3c(chunks, st3c_send, level=10)

    total3c = len(pkt3c_dict) + len(pkt3c_data)

    print("USC Stream Bench v8 — v3b vs v3c typed")
    print("-------------------------------------------------")
    print(f"RAW bytes       : {len(raw_bytes)}")
    print(f"GZIP bytes      : {len(gz)}   (ratio {_ratio(len(raw_bytes), len(gz)):.2f}x)")
    print("-------------------------------------------------")
    print(f"v3b DICT bytes  : {len(pkt3b_dict)}")
    print(f"v3b DATA bytes  : {len(pkt3b_data)}")
    print(f"v3b FIRST total : {total3b}   (ratio {_ratio(len(raw_bytes), total3b):.2f}x)")
    print("-------------------------------------------------")
    print(f"v3c DICT bytes  : {len(pkt3c_dict)}")
    print(f"v3c DATA bytes  : {len(pkt3c_data)}")
    print(f"v3c FIRST total : {total3c}   (ratio {_ratio(len(raw_bytes), total3c):.2f}x)")
    print("-------------------------------------------------")
    print("✅ Goal: v3c FIRST total <= v3b FIRST total")
    print("✅ Stretch: v3c FIRST total beats gzip by more margin")
    print("-------------------------------------------------")

if __name__ == "__main__":
    run()
