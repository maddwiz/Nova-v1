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

from usc.mem.zstd_trained_dict import train_dict, compress_with_dict
from usc.mem.usc_warmdict import warmdict_compress_packets


def _ratio(raw: int, comp: int) -> float:
    return raw / max(1, comp)


def _windows(items, win):
    for i in range(0, len(items), win):
        yield items[i:i + win]


def _chunks(data: bytes, chunk_size: int = 1024):
    out = []
    for i in range(0, len(data), chunk_size):
        out.append(data[i:i + chunk_size])
    return out


def run():
    max_lines_per_chunk = 60

    loops_list = [400, 900, 1500]
    window_chunks_list = [1, 2, 5, 10, 20]
    warmups = [1, 2, 3, 5]

    print("USC Bench18 — WarmDict sweep (DATA-only)")
    print("------------------------------------------------------------")

    for loops in loops_list:
        raw_text = real_agent_trace(loops=loops, seed=7)
        raw = raw_text.encode("utf-8")
        gz = gzip_compress(raw)

        chunks = [c.text for c in chunk_by_lines(raw_text, max_lines=max_lines_per_chunk)]

        st_build = StreamStateV3B()
        build_v3b(chunks, state=st_build)
        pkt_dict = dict_v3b(st_build, level=10)

        st_send = StreamStateV3B()
        apply_v3b(pkt_dict, state=st_send)

        print(f"\nLOOPS={loops}  RAW={len(raw)}  GZIP={len(gz)} ({_ratio(len(raw), len(gz)):.2f}x)")
        print("------------------------------------------------------------")

        for win in window_chunks_list:
            data_packets = []
            for w in _windows(chunks, win):
                data_packets.append(data_v3b(w, st_send, level=10))

            usc_stream = pkt_dict + b"".join(data_packets)
            base_total = len(usc_stream)

            # FULL dict ceiling baseline (best-effort)
            full_total = None
            full_dict_bytes = None
            try:
                full_bundle = train_dict(_chunks(usc_stream), dict_size=8192)
                full_total = len(compress_with_dict(usc_stream, full_bundle, level=10))
                full_dict_bytes = len(full_bundle.dict_bytes)
            except Exception:
                full_total = None
                full_dict_bytes = None

            if full_total is None:
                full_str = "FULLDICT=SKIP"
            else:
                full_str = f"FULLDICT={full_total:>7} ({_ratio(len(raw), full_total):.2f}x) dict={full_dict_bytes}"

            print(
                f"win={win:>2} | packets: dict=1 + data={len(data_packets):<3} | "
                f"USC={base_total:>7} ({_ratio(len(raw), base_total):.2f}x) | {full_str}"
            )

            # WarmDict on DATA-only, add dict packet size back in
            for W in warmups:
                if len(data_packets) < 2:
                    continue

                W2 = min(max(1, W), len(data_packets) - 1)
                wd = warmdict_compress_packets(
                    data_packets,
                    warmup_packets=W2,
                    dict_target_size=8192,
                    level=10,
                )
                total = len(pkt_dict) + wd.total_bytes

                print(
                    f"       Warm W={W2}: total={total:>7} ({_ratio(len(raw), total):.2f}x) "
                    f"mode={wd.used_mode} warm={wd.warmup_bytes} rest_comp={wd.rest_compressed_bytes} dict={wd.trained_dict_bytes}"
                )

            print("")

    print("------------------------------------------------------------")


if __name__ == "__main__":
    run()
