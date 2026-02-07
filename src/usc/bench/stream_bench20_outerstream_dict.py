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

from usc.mem.outerstream_zstd import pack_packets
from usc.mem.zstd_trained_dict import train_dict, compress_with_dict, compress_plain


def _ratio(raw: int, comp: int) -> float:
    return raw / max(1, comp)


def _windows(items, win):
    for i in range(0, len(items), win):
        yield items[i:i + win]


def _chunks(data: bytes, chunk_size: int = 1024):
    return [data[i:i + chunk_size] for i in range(0, len(data), chunk_size)]


def run():
    loops = 900
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

    framed = pack_packets(packets)

    # Plain outer zstd
    outer_plain = compress_plain(framed, level=10)

    # Outer dict zstd (train on framed chunks)
    samples = _chunks(framed, chunk_size=1024)
    bundle = train_dict(samples, dict_size=8192)
    outer_dict = compress_with_dict(framed, bundle, level=10)

    print("USC Bench20 — OuterStream framed + ZSTD plain vs ZSTD dict")
    print("------------------------------------------------------------")
    print(f"RAW bytes        : {len(raw)}")
    print(f"GZIP bytes       : {len(gz):>7} ({_ratio(len(raw), len(gz)):.2f}x)")
    print("------------------------------------------------------------")
    print(f"USC packets      : {len(packets)}")
    print(f"Framed bytes     : {len(framed):>7}")
    print("------------------------------------------------------------")
    print(f"Outer ZSTD plain : {len(outer_plain):>7} ({_ratio(len(raw), len(outer_plain)):.2f}x)")
    print(f"Outer ZSTD dict  : {len(outer_dict):>7} ({_ratio(len(raw), len(outer_dict)):.2f}x) dict={len(bundle.dict_bytes)}")
    print("------------------------------------------------------------")


if __name__ == "__main__":
    run()
