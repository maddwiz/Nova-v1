import json
from usc.bench.datasets_mixed_tool_trace import mixed_tool_trace
from usc.mem.sas_dict_token_v1 import build_sas_packets_from_text, decode_sas_packets_to_lines


def run():
    text = mixed_tool_trace(steps=1200, seed=7)
    packets = build_sas_packets_from_text(text, max_lines_per_packet=10, tok_top_k=0)

    lines = decode_sas_packets_to_lines(packets)

    pvals = []
    for ln in lines:
        if "tool_call::web.screenshot" not in ln:
            continue
        if "payload=" not in ln:
            continue
        try:
            payload = json.loads(ln.split("payload=", 1)[1].strip())
        except Exception:
            continue
        if "pageno" in payload:
            pvals.append(payload["pageno"])

    uniq = sorted(set(pvals))
    print("Found screenshot pageno values:", uniq)
    print("Count:", len(pvals))


if __name__ == "__main__":
    run()
