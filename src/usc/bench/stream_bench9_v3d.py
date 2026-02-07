from usc.bench.datasets import toy_big_agent_log_varied
from usc.bench.metrics import gzip_compress
from usc.mem.chunking import chunk_by_lines

# v3b champ
from usc.mem.stream_proto_canz_v3b import (
    StreamStateV3B,
    build_dict_state_from_chunks as build_v3b,
    encode_dict_packet as dict_v3b,
    apply_dict_packet as apply_v3b,
    encode_data_packet as data_v3b,
)

# v3d drain3
from usc.mem.stream_proto_canz_v3d_drain3 import (
    StreamStateV3D,
    build_dict_state_from_chunks as build_v3d,
    encode_dict_packet as dict_v3d,
    apply_dict_packet as apply_v3d,
    encode_data_packet as data_v3d,
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

    # ---- v3d ----
    st3d_build = StreamStateV3D()
    build_v3d(chunks, state=st3d_build)
    pkt3d_dict = dict_v3d(st3d_build, level=10)

    st3d_send = StreamStateV3D()
    apply_v3d(pkt3d_dict, state=st3d_send)
    pkt3d_data = data_v3d(chunks, st3d_send, level=10)

    total3d = len(pkt3d_dict) + len(pkt3d_data)

    print("USC Stream Bench v9 — v3b vs v3d (Drain3)")
    print("-------------------------------------------------")
    print(f"RAW bytes       : {len(raw_bytes)}")
    print(f"GZIP bytes      : {len(gz)}   (ratio {_ratio(len(raw_bytes), len(gz)):.2f}x)")
    print("-------------------------------------------------")
    print(f"v3b DICT bytes  : {len(pkt3b_dict)}")
    print(f"v3b DATA bytes  : {len(pkt3b_data)}")
    print(f"v3b FIRST total : {total3b}   (ratio {_ratio(len(raw_bytes), total3b):.2f}x)")
    print("-------------------------------------------------")
    print(f"v3d DICT bytes  : {len(pkt3d_dict)}")
    print(f"v3d DATA bytes  : {len(pkt3d_data)}")
    print(f"v3d FIRST total : {total3d}   (ratio {_ratio(len(raw_bytes), total3d):.2f}x)")
    print("-------------------------------------------------")
    print("✅ Goal: v3d <= v3b on VARIED logs")
    print("✅ Stretch: v3d beats gzip more consistently on messy logs")
    print("-------------------------------------------------")

if __name__ == "__main__":
    run()
