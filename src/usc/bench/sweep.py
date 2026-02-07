from usc.bench.datasets import toy_big_agent_log_varied
from usc.bench.metrics import gzip_compress
from usc.mem.chunking import chunk_by_lines

from usc.mem.templatemtf_bits_deltaonly_canon import (
    encode_chunks_with_template_mtf_bits_deltaonly_canon as CAN,
)
from usc.mem.templatemtf_bits_deltaonly_canon_zstd import (
    encode_chunks_with_template_mtf_bits_deltaonly_canon as CANZ,
)


def _ratio(raw: int, comp: int) -> float:
    return raw / max(1, comp)


def run_sweep():
    raw_big = toy_big_agent_log_varied(loops=30)
    raw_big_bytes = raw_big.encode("utf-8")
    gz_big = gzip_compress(raw_big_bytes)

    sizes = [10, 25, 50, 100, 200]

    print("USC Bench Sweep — VARIED BIG LOG")
    print("----------------------------------------")
    print(f"RAW bytes  : {len(raw_big_bytes)}")
    print(f"GZIP bytes : {len(gz_big)}  (ratio {_ratio(len(raw_big_bytes), len(gz_big)):.2f}x)")
    print("----------------------------------------")
    print("ChunkLines | CAN bytes | CANZ bytes | Gap to GZIP")
    print("----------------------------------------")

    for max_lines in sizes:
        chunks = [c.text for c in chunk_by_lines(raw_big, max_lines=max_lines)]

        can_bytes = CAN(chunks)
        canz_bytes = CANZ(chunks)

        gap = len(canz_bytes) - len(gz_big)

        print(
            f"{str(max_lines).rjust(9)} | "
            f"{str(len(can_bytes)).rjust(8)} | "
            f"{str(len(canz_bytes)).rjust(9)} | "
            f"{str(gap).rjust(10)}"
        )

    print("----------------------------------------")
    print("✅ Lower CANZ bytes is better. ✅ Smaller Gap is better.")


if __name__ == "__main__":
    run_sweep()
