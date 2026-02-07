from usc.bench.datasets import toy_big_agent_log_varied
from usc.mem.chunking import chunk_by_lines

from usc.mem.stream_proto_canz_v3b import (
    StreamStateV3B,
    build_dict_state_from_chunks,
    encode_dict_packet,
    apply_dict_packet,
    encode_data_packet,
)

# v3 decoder understands USDATAZ3 (our DATA packet magic)
from usc.mem.stream_proto_canz_v3 import decode_data_packet as decode_data_packet_v3
from usc.mem.stream_proto_canz_v3 import StreamStateV3


def run():
    raw = toy_big_agent_log_varied(loops=30)
    chunks = [c.text for c in chunk_by_lines(raw, max_lines=25)]

    # ---- Build sender dict state (v3b) ----
    st_build = StreamStateV3B()
    build_dict_state_from_chunks(chunks, state=st_build)
    dict_pkt = encode_dict_packet(st_build, level=10)

    # ---- Decode DICT packet into temp v3b receiver ----
    st_tmp = StreamStateV3B()
    apply_dict_packet(dict_pkt, state=st_tmp)

    # ---- Build v3 receiver state for DATA decoding ----
    st_recv = StreamStateV3()

    # IMPORTANT FIX:
    # Use the true arity computed during template extraction,
    # not t.count("{}") which can be wrong.
    for tid, t in enumerate(st_tmp.templates):
        st_recv.templates.append(t)
        st_recv.temp_index[t] = tid
        st_recv.arity_by_tid[tid] = st_tmp.arity_by_tid.get(tid, 0)
        st_recv.mtf.append(tid)

    # ---- Encode DATA packet with v3b encoder ----
    st_send = StreamStateV3B()
    build_dict_state_from_chunks(chunks, state=st_send)
    data_pkt = encode_data_packet(chunks, st_send, level=10)

    # ---- Decode DATA using v3 decoder ----
    out_chunks = decode_data_packet_v3(data_pkt, st_recv)

    ok = (out_chunks == chunks)
    print("ROUNDTRIP OK:", ok)

    if not ok:
        for i, (a, b) in enumerate(zip(chunks, out_chunks)):
            if a != b:
                print("FIRST MISMATCH INDEX:", i)
                print("ORIG:", repr(a))
                print("DECO:", repr(b))
                break


if __name__ == "__main__":
    run()
