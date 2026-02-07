from usc.bench.datasets import toy_big_agent_log_varied
from usc.bench.metrics import gzip_compress
from usc.mem.chunking import chunk_by_lines

from usc.mem.templatemtf_bits_deltaonly_canon_zstd import (
    encode_chunks_with_template_mtf_bits_deltaonly_canon as CANZ,
)
from usc.mem.stream_tmtfdo_canz import encode_stream_tmtfdo_canz


def _ratio(raw: int, comp: int) -> float:
    return raw / max(1, comp)


def run_stream_bench():
    raw_big = toy_big_agent_log_varied(loops=30)
    raw_big_bytes = raw_big.encode("utf-8")
    gz = gzip_compress(raw_big_bytes)

    # simulate streaming chunks
    chunks = [c.text for c in chunk_by_lines(raw_big, max_lines=10)]

    # batch (current)
    canz_batch = CANZ(chunks)

    # stream
    canz_stream = encode_stream_tmtfdo_canz(chunks)

    print("USC Stream Bench — VARIED BIG LOG")
    print("----------------------------------------")
    print(f"RAW bytes      : {len(raw_big_bytes)}")
    print(f"GZIP bytes     : {len(gz)}  (ratio {_ratio(len(raw_big_bytes), len(gz)):.2f}x)")
    print(f"CANZ (batch)   : {len(canz_batch)}  (ratio {_ratio(len(raw_big_bytes), len(canz_batch)):.2f}x)")
    print(f"CANZ (stream)  : {len(canz_stream)}  (ratio {_ratio(len(raw_big_bytes), len(canz_stream)):.2f}x)")
    print("----------------------------------------")


if __name__ == "__main__":
    run_stream_bench()
