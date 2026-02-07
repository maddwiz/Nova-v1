from __future__ import annotations
import argparse
from usc.codec.decode_router import decode_auto

def add_decode_auto_subcommand(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser("decode-auto", help="Auto-decode USC containers (TPF3/USCH/USCC)")
    p.add_argument("--in", dest="inp", required=True, help="Input .bin file")
    p.add_argument("--out", required=True, help="Output decoded .log file")
    p.set_defaults(func=_cmd_decode_auto)

def _cmd_decode_auto(args) -> int:
    mode = decode_auto(args.inp, args.out)
    print(f"✅ decoded ({mode}): {args.inp} → {args.out}")
    return 0
