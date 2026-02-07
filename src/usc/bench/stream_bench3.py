from usc.bench.datasets import toy_big_agent_log_varied
from usc.bench.metrics import gzip_compress
from usc.mem.chunking import chunk_by_lines

from usc.mem.templatemtf_bits_deltaonly_canon_zstd import (
    encode_chunks_with_template_mtf_bits_deltaonly_canon as CANZ,
)

from usc.mem.stream_window_canz import StreamState, encode_stream_window_canz


def _ratio(raw: int, comp: int) -> float:
    return raw / max(1, comp)


def _chunks_to_packets(chunks, window_size, state):
    packets = []
    for i in range(0, len(chunks), window_size):
        window = chunks[i : i + window_size]
        packets.append(encode_stream_window_canz(window, state=state))
    return packets


def run_stream_bench3():
    raw_big = toy_big_agent_log_varied(loops=30)
    raw_big_bytes = raw_big.encode("utf-8")
    gz = gzip_compress(raw_big_bytes)

    # small chunking, like a real agent emitting events
    chunks = [c.text for c in chunk_by_lines(raw_big, max_lines=25)]

    # baseline: batch CANZ over ALL chunks at once
    canz_batch = CANZ(chunks)

    # stream windows: send multiple packets, but keep state across them
    for window_size in [1, 5, 10, 25, 50]:
        st = StreamState()
        packets = _chunks_to_packets(chunks, window_size=window_size, state=st)
        total_bytes = sum(len(p) for p in packets)

        print(f"WINDOW={str(window_size).rjust(2)} | packets={str(len(packets)).rjust(3)} | "
              f"total={str(total_bytes).rjust(6)} | ratio={_ratio(len(raw_big_bytes), total_bytes):.2f}x")

    print("----------------------------------------")
    print("USC Stream Bench v3 — VARIED BIG LOG")
    print("----------------------------------------")
    print(f"RAW bytes     : {len(raw_big_bytes)}")
    print(f"GZIP bytes    : {len(gz)}  (ratio {_ratio(len(raw_big_bytes), len(gz)):.2f}x)")
    print(f"CANZ (batch)  : {len(canz_batch)}  (ratio {_ratio(len(raw_big_bytes), len(canz_batch)):.2f}x)")
    print("----------------------------------------")
    print("✅ Lower total bytes is better.")
    print("✅ Window 25/50 should crush window 1.")
    print("----------------------------------------")


if __name__ == "__main__":
    run_stream_bench3()
