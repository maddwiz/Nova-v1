from usc.bench.datasets_real_agent_trace import real_agent_trace

from usc.api.codec_odc import (
    build_v3b_packets_from_text,
    odc_encode_packets,
    odc_decode_to_packets,
)


def run():
    loops = 900
    text = real_agent_trace(loops=loops, seed=7)

    packets = build_v3b_packets_from_text(
        text,
        max_lines_per_chunk=60,
        window_chunks=1,
        level=10,
    )

    blob, meta = odc_encode_packets(
        packets,
        level=10,
        dict_target_size=8192,
        sample_chunk_size=1024,
    )

    back = odc_decode_to_packets(blob)

    ok = (packets == back)

    print("USC Bench21 — ODC roundtrip (packets exact match)")
    print("------------------------------------------------------------")
    print("packets_in :", len(packets))
    print("packets_out:", len(back))
    print("dict_bytes :", meta.dict_bytes)
    print("framed     :", meta.framed_bytes)
    print("compressed :", meta.compressed_bytes)
    print("ODC blob   :", len(blob))
    print("PASS       :", ok)
    print("------------------------------------------------------------")

    if not ok:
        # find first mismatch
        n = min(len(packets), len(back))
        for i in range(n):
            if packets[i] != back[i]:
                print("First mismatch at packet index:", i)
                print("in_len :", len(packets[i]))
                print("out_len:", len(back[i]))
                break


if __name__ == "__main__":
    run()
