from usc.bench.datasets_real_agent_trace import real_agent_trace
from usc.bench.metrics import gzip_compress
from usc.mem.chunking import chunk_by_lines

from usc.mem.stream_proto_canz_v3b import (
    StreamStateV3B,
    build_dict_state_from_chunks as build_v3b,
    encode_dict_packet as dict_v3b,
    apply_dict_packet as apply_v3b,
    encode_data_packet as data_v3b,
)

from usc.mem.outerstream_zstd import compress_outerstream


def _ratio(raw: int, comp: int) -> float:
    return raw / max(1, comp)


def _windows(items, win):
    for i in range(0, len(items), win):
        yield items[i:i + win]


def run():
    loops = 900  # best win case from your sweep
    max_lines_per_chunk = 60
    window_chunks = 1

    raw_text = real_agent_trace(loops=loops, seed=7)
    raw = raw_text.encode("utf-8")
    gz = gzip_compress(raw)

    chunks = [c.text for c in chunk_by_lines(raw_text, max_lines=max_lines_per_chunk)]

    # USC v3b packets
    st_build = StreamStateV3B()
    build_v3b(chunks, state=st_build)
    pkt_dict = dict_v3b(st_build, level=10)

    st_send = StreamStateV3B()
    apply_v3b(pkt_dict, state=st_send)

    packets = [pkt_dict]
    for w in _windows(chunks, window_chunks):
        packets.append(data_v3b(w, st_send, level=10))

    usc_stream = b"".join(packets)

    outer_blob, meta = compress_outerstream(packets, level=10)

    print("USC Bench19 — OuterStream (USC packets + 1 outer zstd pass)")
    print("------------------------------------------------------------")
    print(f"RAW bytes        : {len(raw)}")
    print(f"GZIP bytes       : {len(gz):>7} ({_ratio(len(raw), len(gz)):.2f}x)")
    print("------------------------------------------------------------")
    print(f"USC v3b stream   : {len(usc_stream):>7} ({_ratio(len(raw), len(usc_stream)):.2f}x) packets={len(packets)}")
    print(f"OuterStream blob : {len(outer_blob):>7} ({_ratio(len(raw), len(outer_blob)):.2f}x) raw_framed={meta.raw_stream_bytes} comp={meta.comp_stream_bytes}")
    print("------------------------------------------------------------")


if __name__ == "__main__":
    run()
