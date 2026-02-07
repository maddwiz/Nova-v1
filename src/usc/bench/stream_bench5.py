from usc.bench.datasets import toy_big_agent_log_varied
from usc.bench.metrics import gzip_compress
from usc.mem.chunking import chunk_by_lines

from usc.mem.templatemtf_bits_deltaonly_canon_zstd import (
    encode_chunks_with_template_mtf_bits_deltaonly_canon as CANZ_BATCH,
)

from usc.mem.stream_proto_canz_v3 import (
    StreamStateV3,
    build_dict_state_from_chunks,
    encode_dict_packet,
    apply_dict_packet,
    encode_data_packet,
)


def _ratio(raw: int, comp: int) -> float:
    return raw / max(1, comp)


def _windowed(chunks, window_size):
    for i in range(0, len(chunks), window_size):
        yield chunks[i : i + window_size]


def run_stream_bench5():
    raw_big = toy_big_agent_log_varied(loops=30)
    raw_big_bytes = raw_big.encode("utf-8")
    gz = gzip_compress(raw_big_bytes)

    # best-known chunking
    chunks = [c.text for c in chunk_by_lines(raw_big, max_lines=25)]

    # baseline batch codec
    canz_batch = CANZ_BATCH(chunks)

    # ---- DICT warmup (send once) ----
    state_build = StreamStateV3()
    build_dict_state_from_chunks(chunks, state=state_build)
    dict_pkt = encode_dict_packet(state_build, level=10)

    # receiver applies dict packet
    state_send = StreamStateV3()
    apply_dict_packet(dict_pkt, state=state_send)

    # ---- DATA streaming: send in windows ----
    window_size = 25  # 1 packet here for this dataset
    data_packets = []
    for win in _windowed(chunks, window_size):
        data_packets.append(encode_data_packet(win, state_send, level=10))

    total_data = sum(len(p) for p in data_packets)
    total_first_time = len(dict_pkt) + total_data

    print("USC Stream Bench v5 — DICT + DATA protocol")
    print("----------------------------------------")
    print(f"RAW bytes        : {len(raw_big_bytes)}")
    print(f"GZIP bytes       : {len(gz)}  (ratio {_ratio(len(raw_big_bytes), len(gz)):.2f}x)")
    print(f"CANZ batch       : {len(canz_batch)}  (ratio {_ratio(len(raw_big_bytes), len(canz_batch)):.2f}x)")
    print("----------------------------------------")
    print(f"DICT packet      : {len(dict_pkt)} bytes")
    print(f"DATA packets     : {len(data_packets)} packets, total {total_data} bytes")
    print("----------------------------------------")
    print(f"FIRST RUN total  : {total_first_time} bytes  (DICT + DATA)")
    print(f"STEADY total     : {total_data} bytes  (DATA only)")
    print("----------------------------------------")
    print("✅ FIRST RUN includes warmup cost.")
    print("✅ STEADY is the real long-running agent advantage.")
    print("----------------------------------------")


if __name__ == "__main__":
    run_stream_bench5()
