from usc.bench.metrics import gzip_compress
from usc.mem.chunking import chunk_by_lines

from usc.bench.datasets_real_agent_trace import real_agent_trace

from usc.mem.stream_proto_canz_v3b import (
    StreamStateV3B,
    build_dict_state_from_chunks as build_v3b,
    encode_dict_packet as dict_v3b,
    apply_dict_packet as apply_v3b,
    encode_data_packet as data_v3b,
)

from usc.mem.stream_proto_canz_v3d_drain3 import (
    StreamStateV3D,
    build_dict_state_from_chunks as build_v3d6,
    encode_dict_packet as dict_v3d6,
    apply_dict_packet as apply_v3d6,
    encode_data_packet as data_v3d6,
)

from usc.mem.stream_proto_canz_v3auto import (
    StreamStateV3AUTO,
    build_dict_state_from_chunks as build_auto,
    encode_dict_packet as dict_auto,
    apply_dict_packet as apply_auto,
    encode_data_packet as data_auto,
)

def _ratio(raw: int, comp: int) -> float:
    return raw / max(1, comp)

def _windows(items, win):
    for i in range(0, len(items), win):
        yield items[i:i+win]

def run(max_lines_per_chunk: int = 25, window_chunks: int = 10, loops: int = 250):
    raw_big = real_agent_trace(loops=loops, seed=7)
    raw_bytes = raw_big.encode("utf-8")
    gz = gzip_compress(raw_bytes)

    chunks = [c.text for c in chunk_by_lines(raw_big, max_lines=max_lines_per_chunk)]

    print("USC Stream Bench v11 — REAL agent trace")
    print("-------------------------------------------------")
    print(f"RAW bytes  : {len(raw_bytes)}")
    print(f"GZIP bytes : {len(gz)} (ratio {_ratio(len(raw_bytes), len(gz)):.2f}x)")
    print("-------------------------------------------------")
    print(f"Chunks          : {len(chunks)}")
    print(f"Lines per chunk : {max_lines_per_chunk}")
    print(f"Chunks/packet   : {window_chunks}")
    print("-------------------------------------------------")

    # v3b
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

    # v3d6
    st3d_build = StreamStateV3D()
    build_v3d6(chunks, state=st3d_build)
    pkt3d_dict = dict_v3d6(st3d_build, level=10)

    st3d_send = StreamStateV3D()
    apply_v3d6(pkt3d_dict, state=st3d_send)

    total3d = len(pkt3d_dict)
    sizes3d = []
    for w in _windows(chunks, window_chunks):
        pkt = data_v3d6(w, st3d_send, level=10)
        sizes3d.append(len(pkt))
        total3d += len(pkt)

    # AUTO
    stA_build = StreamStateV3AUTO()
    build_auto(chunks, state=stA_build)
    pktA_dict = dict_auto(stA_build, level=10)

    stA_send = StreamStateV3AUTO()
    apply_auto(pktA_dict, state=stA_send)

    totalA = len(pktA_dict)
    sizesA = []
    for w in _windows(chunks, window_chunks):
        pkt = data_auto(w, stA_send, level=10)
        sizesA.append(len(pkt))
        totalA += len(pkt)

    def tail_avg(xs, k=5):
        if not xs:
            return 0.0
        k = min(k, len(xs))
        return sum(xs[-k:]) / k

    print("v3b:")
    print(f"  DICT bytes     : {len(pkt3b_dict)}")
    print(f"  DATA packets   : {len(sizes3b)}")
    print(f"  DATA avg bytes : {sum(sizes3b)/len(sizes3b):.1f}")
    print(f"  DATA tail avg  : {tail_avg(sizes3b):.1f}")
    print(f"  TOTAL bytes    : {total3b}  (ratio {_ratio(len(raw_bytes), total3b):.2f}x)")
    print("-------------------------------------------------")
    print("v3d6:")
    print(f"  DICT bytes     : {len(pkt3d_dict)}")
    print(f"  DATA packets   : {len(sizes3d)}")
    print(f"  DATA avg bytes : {sum(sizes3d)/len(sizes3d):.1f}")
    print(f"  DATA tail avg  : {tail_avg(sizes3d):.1f}")
    print(f"  TOTAL bytes    : {total3d}  (ratio {_ratio(len(raw_bytes), total3d):.2f}x)")
    print("-------------------------------------------------")
    print("AUTO:")
    print(f"  DICT bytes     : {len(pktA_dict)}")
    print(f"  DATA packets   : {len(sizesA)}")
    print(f"  DATA avg bytes : {sum(sizesA)/len(sizesA):.1f}")
    print(f"  DATA tail avg  : {tail_avg(sizesA):.1f}")
    print(f"  TOTAL bytes    : {totalA}  (ratio {_ratio(len(raw_bytes), totalA):.2f}x)")
    print("-------------------------------------------------")

if __name__ == "__main__":
    run()
