from usc.bench.datasets import toy_big_agent_log_varied
from usc.mem.chunking import chunk_by_lines

from usc.mem.stream_proto_canz_v3d_drain3 import (
    StreamStateV3D,
    build_dict_state_from_chunks,
    encode_dict_packet,
    apply_dict_packet,
    encode_data_packet,
    decode_data_packet,
)


def run():
    raw = toy_big_agent_log_varied(loops=30)
    chunks = [c.text for c in chunk_by_lines(raw, max_lines=25)]

    # sender builds dict
    st_build = StreamStateV3D()
    build_dict_state_from_chunks(chunks, state=st_build)
    dict_pkt = encode_dict_packet(st_build, level=10)

    # sender encodes data
    st_send = StreamStateV3D()
    apply_dict_packet(dict_pkt, state=st_send)
    data_pkt = encode_data_packet(chunks, st_send, level=10)

    # receiver decodes
    st_recv = StreamStateV3D()
    apply_dict_packet(dict_pkt, state=st_recv)
    out_chunks = decode_data_packet(data_pkt, st_recv)

    ok = (out_chunks == chunks)
    print("ROUNDTRIP OK:", ok)

    if not ok:
        for i, (a, b) in enumerate(zip(chunks, out_chunks)):
            if a != b:
                print("FIRST MISMATCH INDEX:", i)
                print("ORIG:", repr(a))
                print("DECO:", repr(b))
                return


if __name__ == "__main__":
    run()
