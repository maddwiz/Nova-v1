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

# v3d5 drain3 persistent + refresh
from usc.mem.stream_proto_canz_v3d_drain3 import (
    StreamStateV3D,
    build_dict_state_from_chunks as build_v3d,
    encode_dict_packet as dict_v3d,
    apply_dict_packet as apply_v3d,
    encode_data_packet as data_v3d,
)

def _ratio(raw: int, comp: int) -> float:
    return raw / max(1, comp)

def _windows(items, win):
    for i in range(0, len(items), win):
        yield items[i:i+win]

def run(window_chunks: int = 10):
    raw_big = toy_big_agent_log_varied(loops=30)
    raw_bytes = raw_big.encode("utf-8")
    gz = gzip_compress(raw_bytes)

    chunks = [c.text for c in chunk_by_lines(raw_big, max_lines=25)]

    print("USC Stream Bench v10 — multi-packet windows")
    print("-------------------------------------------------")
    print(f"RAW bytes  : {len(raw_bytes)}")
    print(f"GZIP bytes : {len(gz)} (ratio {_ratio(len(raw_bytes), len(gz)):.2f}x)")
    print("-------------------------------------------------")
    print(f"Total chunks      : {len(chunks)}")
    print(f"Chunks per packet : {window_chunks}")
    print("-------------------------------------------------")

    # ---------------- v3b ----------------
    st3b_build = StreamStateV3B()
    build_v3b(chunks, state=st3b_build)
    pkt3b_dict = dict_v3b(st3b_build, level=10)

    st3b_send = StreamStateV3B()
    apply_v3b(pkt3b_dict, state=st3b_send)

    total3b = len(pkt3b_dict)
    sizes3b = []
    for w in _windows(chunks, window_chunks):
        pkt = data_v3b(w, st3b_send, level=10)
        sizes3b.append(len(pkt))
        total3b += len(pkt)

    # ---------------- v3d5 ----------------
    st3d_build = StreamStateV3D()
    build_v3d(chunks, state=st3d_build)
    pkt3d_dict = dict_v3d(st3d_build, level=10)

    st3d_send = StreamStateV3D()
    apply_v3d(pkt3d_dict, state=st3d_send)

    total3d = len(pkt3d_dict)
    sizes3d = []
    for w in _windows(chunks, window_chunks):
        pkt = data_v3d(w, st3d_send, level=10)
        sizes3d.append(len(pkt))
        total3d += len(pkt)

    def tail_avg(xs, k=5):
        if not xs:
            return 0.0
        k = min(k, len(xs))
        return sum(xs[-k:]) / k

    print("v3b:")
    print(f"  DICT bytes     : {len(pkt3b_dict)}")
    print(f"  DATA packets   : {len(sizes3b)}")
    print(f"  DATA avg bytes : {sum(sizes3b)/len(sizes3b):.1f}")
    print(f"  DATA tail avg  : {tail_avg(sizes3b):.1f}   (last 5 packets)")
    print(f"  TOTAL bytes    : {total3b}  (ratio {_ratio(len(raw_bytes), total3b):.2f}x)")
    print("-------------------------------------------------")
    print("v3d5 (Drain3 + persistent strings + refresh):")
    print(f"  DICT bytes     : {len(pkt3d_dict)}")
    print(f"  DATA packets   : {len(sizes3d)}")
    print(f"  DATA avg bytes : {sum(sizes3d)/len(sizes3d):.1f}")
    print(f"  DATA tail avg  : {tail_avg(sizes3d):.1f}   (last 5 packets)")
    print(f"  TOTAL bytes    : {total3d}  (ratio {_ratio(len(raw_bytes), total3d):.2f}x)")
    print("-------------------------------------------------")
    print("✅ This is the real test of persistent streaming codecs.")
    print("✅ v3d5 should get better in tail avg as dictionaries warm up.")
    print("-------------------------------------------------")

if __name__ == "__main__":
    run(window_chunks=10)
