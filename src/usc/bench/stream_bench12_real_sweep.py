from usc.bench.metrics import gzip_compress
from usc.mem.chunking import chunk_by_lines

from usc.bench.datasets_real_agent_trace import real_agent_trace

from usc.mem.stream_proto_canz_v3b import (
    StreamStateV3B,
    build_dict_state_from_chunks as build_v3b,
    encode_dict_packet as dict_v3b,
    apply_dict_packet as apply_v3b,
    encode_data_packet as data_v3b,
)


def _ratio(raw: int, comp: int) -> float:
    return raw / max(1, comp)


def _windows(items, win):
    for i in range(0, len(items), win):
        yield items[i:i + win]


def run(loops: int = 350):
    raw_big = real_agent_trace(loops=loops, seed=7)
    raw_bytes = raw_big.encode("utf-8")
    gz = gzip_compress(raw_bytes)

    print("USC Stream Bench v12 — REAL trace sweep (v3b)")
    print("-------------------------------------------------")
    print(f"RAW bytes  : {len(raw_bytes)}")
    print(f"GZIP bytes : {len(gz)} (ratio {_ratio(len(raw_bytes), len(gz)):.2f}x)")
    print("-------------------------------------------------")

    line_sizes = [10, 15, 25, 40, 60]
    win_sizes = [3, 5, 10, 20]

    best = None

    for max_lines in line_sizes:
        chunks = [c.text for c in chunk_by_lines(raw_big, max_lines=max_lines)]

        for win in win_sizes:
            st_build = StreamStateV3B()
            build_v3b(chunks, state=st_build)
            pkt_dict = dict_v3b(st_build, level=10)

            st_send = StreamStateV3B()
            apply_v3b(pkt_dict, state=st_send)

            total = len(pkt_dict)
            sizes = []

            for w in _windows(chunks, win):
                pkt = data_v3b(w, st_send, level=10)
                sizes.append(len(pkt))
                total += len(pkt)

            ratio = _ratio(len(raw_bytes), total)

            row = (ratio, total, max_lines, win, len(pkt_dict), sum(sizes) / len(sizes))
            if best is None or row[0] > best[0]:
                best = row

            print(
                f"lines={max_lines:>2} win={win:>2} | "
                f"TOTAL={total:>6} | ratio={ratio:>5.2f}x | "
                f"dict={len(pkt_dict):>5} | data_avg={sum(sizes)/len(sizes):>7.1f}"
            )

    print("-------------------------------------------------")
    br, bt, bl, bw, bd, bavg = best
    print(f"✅ BEST: lines={bl} win={bw} | TOTAL={bt} | ratio={br:.2f}x | dict={bd} | data_avg={bavg:.1f}")
    print("-------------------------------------------------")


if __name__ == "__main__":
    run()
