from usc.bench.datasets import toy_big_agent_log_varied
from usc.bench.metrics import gzip_compress
from usc.mem.chunking import chunk_by_lines

from usc.mem.templatemtf_bits_deltaonly_canon_zstd import (
    encode_chunks_with_template_mtf_bits_deltaonly_canon as CANZ,
)

from usc.mem.stream_window_canz import StreamState, encode_stream_window_canz


def _ratio(raw: int, comp: int) -> float:
    return raw / max(1, comp)


def run_stream_bench2():
    raw_big = toy_big_agent_log_varied(loops=30)
    raw_big_bytes = raw_big.encode("utf-8")
    gz = gzip_compress(raw_big_bytes)

    # we simulate an agent emitting chunks in small pieces
    chunks = [c.text for c in chunk_by_lines(raw_big, max_lines=10)]

    # batch (baseline)
    canz_batch = CANZ(chunks)

    # stream-window v2: encode many small packets, then measure total bytes
    st = StreamState()
    packets = []
    for ch in chunks:
        packets.append(encode_stream_window_canz([ch], state=st))
    stream_total = sum(len(p) for p in packets)

    print("USC Stream Bench v2 — VARIED BIG LOG")
    print("----------------------------------------")
    print(f"RAW bytes       : {len(raw_big_bytes)}")
    print(f"GZIP bytes      : {len(gz)}  (ratio {_ratio(len(raw_big_bytes), len(gz)):.2f}x)")
    print(f"CANZ (batch)    : {len(canz_batch)}  (ratio {_ratio(len(raw_big_bytes), len(canz_batch)):.2f}x)")
    print(f"CANZ (streamv2) : {stream_total}  (ratio {_ratio(len(raw_big_bytes), stream_total):.2f}x)")
    print("----------------------------------------")


if __name__ == "__main__":
    run_stream_bench2()
