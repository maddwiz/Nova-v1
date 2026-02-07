from __future__ import annotations

import argparse
import os
from typing import List

from usc.api.codec_odc import (
    build_v3b_packets_from_text,
    odc_encode_packets,
    odc_decode_to_packets,
)

from usc.api.codec_odc2_indexed import (
    odc2_encode_packets,
    odc2_decode_all_packets,
    odc2_decode_packet_range,
)


def _read_text(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        return f.read()


def _write_bytes(path: str, data: bytes) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "wb") as f:
        f.write(data)


def _read_bytes(path: str) -> bytes:
    with open(path, "rb") as f:
        return f.read()


def _write_packet_files(outdir: str, packets: List[bytes]) -> None:
    os.makedirs(outdir, exist_ok=True)
    for i, p in enumerate(packets):
        fn = os.path.join(outdir, f"packet_{i:04d}.bin")
        with open(fn, "wb") as f:
            f.write(p)


def cmd_encode(args: argparse.Namespace) -> int:
    text = _read_text(args.infile)

    packets = build_v3b_packets_from_text(
        text,
        max_lines_per_chunk=args.max_lines,
        window_chunks=args.window_chunks,
        level=args.level,
    )

    if args.mode == "odc":
        blob, meta = odc_encode_packets(
            packets,
            level=args.level,
            dict_target_size=args.dict_size,
            sample_chunk_size=args.sample_chunk,
        )
        _write_bytes(args.outfile, blob)

        print("USC encode OK ✅")
        print("------------------------------")
        print("mode         :", args.mode)
        print("outfile      :", args.outfile)
        print("packets      :", meta.packets)
        print("dict_bytes   :", meta.dict_bytes)
        print("framed_bytes :", meta.framed_bytes)
        print("compressed   :", meta.compressed_bytes)
        print("blob_bytes   :", len(blob))
        print("------------------------------")
        return 0

    if args.mode == "odc2":
        blob, meta = odc2_encode_packets(
            packets,
            level=args.level,
            dict_target_size=args.dict_size,
            sample_chunk_size=args.sample_chunk,
            group_size=args.group_size,
        )
        _write_bytes(args.outfile, blob)

        print("USC encode OK ✅")
        print("------------------------------")
        print("mode       :", args.mode)
        print("outfile    :", args.outfile)
        print("packets    :", meta.packet_count)
        print("group_size :", meta.group_size)
        print("blocks     :", meta.block_count)
        print("dict_bytes :", meta.dict_bytes)
        print("used_mode  :", meta.used_mode)
        print("blob_bytes :", len(blob))
        print("------------------------------")
        return 0

    raise SystemExit("Unknown --mode")


def cmd_decode(args: argparse.Namespace) -> int:
    blob = _read_bytes(args.infile)

    if args.mode == "odc":
        packets = odc_decode_to_packets(blob)
        _write_packet_files(args.outdir, packets)

        print("USC decode OK ✅")
        print("------------------------------")
        print("mode    :", args.mode)
        print("infile  :", args.infile)
        print("outdir  :", args.outdir)
        print("packets :", len(packets))
        print("------------------------------")
        return 0

    if args.mode == "odc2":
        packets = odc2_decode_all_packets(blob)
        _write_packet_files(args.outdir, packets)

        print("USC decode OK ✅")
        print("------------------------------")
        print("mode    :", args.mode)
        print("infile  :", args.infile)
        print("outdir  :", args.outdir)
        print("packets :", len(packets))
        print("------------------------------")
        return 0

    raise SystemExit("Unknown --mode")


def cmd_decode_range(args: argparse.Namespace) -> int:
    blob = _read_bytes(args.infile)

    if args.mode != "odc2":
        raise SystemExit("decode-range is only supported for --mode odc2")

    packets = odc2_decode_packet_range(blob, args.start, args.end)
    _write_packet_files(args.outdir, packets)

    print("USC decode-range OK ✅")
    print("------------------------------")
    print("mode    :", args.mode)
    print("infile  :", args.infile)
    print("outdir  :", args.outdir)
    print("range   :", f"[{args.start}:{args.end}]")
    print("packets :", len(packets))
    print("------------------------------")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="usc", description="Unified State Codec CLI")
    sub = p.add_subparsers(dest="cmd", required=True)

    pe = sub.add_parser("encode", help="Encode text -> USC packets -> blob")
    pe.add_argument("--mode", default="odc", choices=["odc", "odc2"])
    pe.add_argument("--in", dest="infile", required=True, help="Input text file")
    pe.add_argument("--out", dest="outfile", required=True, help="Output file (.odc or .odc2)")
    pe.add_argument("--level", type=int, default=10)
    pe.add_argument("--dict-size", type=int, default=8192)
    pe.add_argument("--sample-chunk", type=int, default=1024)
    pe.add_argument("--max-lines", type=int, default=60)
    pe.add_argument("--window-chunks", type=int, default=1)
    pe.add_argument("--group-size", type=int, default=4, help="ODC2 only: packets per compressed block")
    pe.set_defaults(fn=cmd_encode)

    pd = sub.add_parser("decode", help="Decode blob -> packet files")
    pd.add_argument("--mode", default="odc", choices=["odc", "odc2"])
    pd.add_argument("--in", dest="infile", required=True, help="Input blob file")
    pd.add_argument("--outdir", required=True, help="Output folder for packets")
    pd.set_defaults(fn=cmd_decode)

    pr = sub.add_parser("decode-range", help="Decode only packet range (ODC2 only)")
    pr.add_argument("--mode", default="odc2", choices=["odc2"])
    pr.add_argument("--in", dest="infile", required=True, help="Input .odc2 file")
    pr.add_argument("--outdir", required=True, help="Output folder for packets")
    pr.add_argument("--start", type=int, required=True)
    pr.add_argument("--end", type=int, required=True)
    pr.set_defaults(fn=cmd_decode_range)

    return p


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return int(args.fn(args))


if __name__ == "__main__":
    raise SystemExit(main())
