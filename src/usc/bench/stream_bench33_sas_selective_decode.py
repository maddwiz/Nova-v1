import time

from usc.bench.datasets_real_agent_trace import real_agent_trace
from usc.mem.sas_dict_token_v1 import build_sas_packets_from_text, decode_sas_packets_to_lines
from usc.mem.sas_index_v0 import build_index, selective_decode_lines, packets_for_tools


def run():
    loops = 900
    text = real_agent_trace(loops=loops, seed=7)

    # Build SAS packets (dict-token v1)
    packets = build_sas_packets_from_text(
        text,
        max_lines_per_packet=60,
        tok_top_k=0,
    )

    # Build index
    t0 = time.perf_counter()
    idx = build_index(packets)
    t1 = time.perf_counter()

    # Full decode
    t2 = time.perf_counter()
    lines_full = decode_sas_packets_to_lines(packets)
    t3 = time.perf_counter()

    # Selective decode: only search_query tool calls
    want = {"web.search_query"}
    t4 = time.perf_counter()
    lines_sel = selective_decode_lines(packets, include_tools=want, include_raw_lines=False)
    t5 = time.perf_counter()

    # Build partial packet stream containing only packets that have search_query
    t6 = time.perf_counter()
    partial = packets_for_tools(packets, want)
    t7 = time.perf_counter()

    # Print results
    print("USC Bench33 — SAS selective decode (DictToken v1 stream)")
    print("------------------------------------------------------------")
    print("Total packets:", len(packets))
    print("Total lines (full decode):", len(lines_full))
    print("Selective lines (web.search_query only):", len(lines_sel))
    print("Partial stream packets (dict + matching packets):", len(partial))
    print("------------------------------------------------------------")
    print("Index build time (ms):", round((t1 - t0) * 1000, 2))
    print("Full decode time  (ms):", round((t3 - t2) * 1000, 2))
    print("Selective time    (ms):", round((t5 - t4) * 1000, 2))
    print("Partial build time(ms):", round((t7 - t6) * 1000, 2))
    print("------------------------------------------------------------")

    # Show some examples so we know it's working
    print("Examples (first 5 selective lines):")
    for ln in lines_sel[:5]:
        print("  ", ln[:120])


if __name__ == "__main__":
    run()
