import argparse
from usc.bench.runner import run_toy_bench

def main():
    p = argparse.ArgumentParser(prog="usc")
    sub = p.add_subparsers(dest="cmd")

    b = sub.add_parser("bench", help="Run benchmarks")
    b.add_argument("--toy", action="store_true", help="Run toy benchmark suite")

    args = p.parse_args()

    if args.cmd == "bench" and args.toy:
        run_toy_bench()
        return

    p.print_help()
