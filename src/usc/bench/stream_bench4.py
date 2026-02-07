from usc.bench.datasets import toy_big_agent_log_varied
from usc.bench.metrics import gzip_compress
from usc.mem.chunking import chunk_by_lines

from usc.mem.templatemtf_bits_deltaonly_canon_zstd import (
    encode_chunks_with_template_mtf_bits_deltaonly_canon as CANZ,
)

from usc.mem.stream_window_canz_v2 import StreamStateV2, encode_stream_window_canz_v2


def _ratio(raw: int, comp: int) -> float:
    return raw / max(1, comp)


def _chunks_to_packets_v2(chunks, window_size, state):
    packets = []
    for i in range(0, len(chunks), window_size):
        window = chunks[i : i + window_size]
        packets.append(encode_stream_window_canz_v2(window, state=state))
    return packets


def run_stream_bench4():
    raw_big = toy_big_agent_log_varied(loops=30)
    raw_big_bytes = raw_big.encode("utf-8")
    gz = gzip_compress(raw_big_bytes)

    # ✅ best-known chunking
    chunks = [c.text for c in chunk_by_lines(raw_big, max_lines=25)]

    # baseline batch
    canz_batch = CANZ(chunks)

    # stream windows v2
    for window_size in [1, 5, 10, 25, 50]:
        st = StreamStateV2()
        packets = _chunks_to_packets_v2(chunks, window_size=window_size, state=st)
        total_bytes = sum(len(p) for p in packets)

        print(f"V2 WINDOW={str(window_size).rjust(2)} | packets={str(len(packets)).rjust(3)} | "
              f"total={str(total_bytes).rjust(6)} | ratio={_ratio(len(raw_big_bytes), total_bytes):.2f}x")

    print("----------------------------------------")
    print("USC Stream Bench v4 (CANZ v2) — VARIED BIG LOG")
    print("----------------------------------------")
    print(f"RAW bytes     : {len(raw_big_bytes)}")
    print(f"GZIP bytes    : {len(gz)}  (ratio {_ratio(len(raw_big_bytes), len(gz)):.2f}x)")
    print(f"CANZ (batch)  : {len(canz_batch)}  (ratio {_ratio(len(raw_big_bytes), len(canz_batch)):.2f}x)")
    print("----------------------------------------")
    print("✅ Lower total bytes is better.")
    print("----------------------------------------")


if __name__ == "__main__":
    run_stream_bench4()
