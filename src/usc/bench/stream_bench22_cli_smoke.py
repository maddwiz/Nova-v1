import os
import shutil
import tempfile

from usc.bench.datasets_real_agent_trace import real_agent_trace
from usc.api.codec_odc import build_v3b_packets_from_text, odc_decode_to_packets


def run():
    text = real_agent_trace(loops=300, seed=7)

    expected_packets = build_v3b_packets_from_text(
        text,
        max_lines_per_chunk=60,
        window_chunks=1,
        level=10,
    )

    td = tempfile.mkdtemp(prefix="usc_cli_smoke_")
    try:
        infile = os.path.join(td, "trace.txt")
        odcfile = os.path.join(td, "trace.odc")
        outdir = os.path.join(td, "decoded_packets")

        with open(infile, "w", encoding="utf-8") as f:
            f.write(text)

        cmd1 = f"python -m usc.cli.usc_cli encode --mode odc --in {infile} --out {odcfile}"
        rc1 = os.system(cmd1)

        cmd2 = f"python -m usc.cli.usc_cli decode --mode odc --in {odcfile} --outdir {outdir}"
        rc2 = os.system(cmd2)

        if os.path.exists(odcfile):
            blob = open(odcfile, "rb").read()
            decoded_packets = odc_decode_to_packets(blob)
            decoded_n = len(decoded_packets)
        else:
            decoded_n = -1

        ok = (rc1 == 0 and rc2 == 0 and decoded_n == len(expected_packets))

        print("USC Bench22 — CLI smoke test")
        print("------------------------------------------------------------")
        print("encode_rc         :", rc1)
        print("decode_rc         :", rc2)
        print("expected_packets  :", len(expected_packets))
        print("decoded_packets   :", decoded_n)
        print("PASS              :", ok)
        print("------------------------------------------------------------")

    finally:
        shutil.rmtree(td, ignore_errors=True)


if __name__ == "__main__":
    run()
