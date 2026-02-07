import argparse
import gzip
import json
import os
import struct
import time
from typing import Dict, List, Tuple

from pathlib import Path
try:
    import zstandard as zstd
except Exception:
    zstd = None

from usc.mem.hdfs_templates_v0 import HDFSTemplateBank, parse_hdfs_lines, parse_hdfs_lines_rows
from usc.api.hdfs_template_codec_v1_channels_mask import encode_template_channels_v1_mask
from usc.mem.tpl_pf1_recall_v1 import build_tpl_pf1_blob as build_pf1
from usc.mem.tpl_pf1_recall_v3_h1m2 import build_tpl_pf3_blob_h1m2 as build_pf3_h1m2
from usc.mem.tpl_pf3_decode_v1_h1m2 import decode_pf3_h1m2_to_lines
from usc.mem.tpl_pfq1_query_v1 import build_pfq1_blob as build_pfq1
from usc.mem.tpl_fast_query_v1 import query_fast_pf1
from usc.mem.tpl_query_router_v1 import query_router_v1
from usc.api.hdfs_template_codec_v1m_bundle import bundle_encode_and_compress_v1m

from usc.api.stream_codec_v3d_auto import encode_stream_auto
from usc.cli.decode_auto_cmd import add_decode_auto_subcommand


MAGIC_HOT = b"USCH"   # PF1 + optional PFQ1 container
MAGIC_COLD = b"USCC"  # TPLv1M bundle container
MAGIC_USZR = b"USZR"  # Raw zstd fallback
MAGIC_USZD = b"USZD"  # Zstd with trained dictionary
MAGIC_USBR = b"USBR"  # Raw brotli fallback
VERSION = 1


def _read_first_n_lines(path: str, n: int) -> List[str]:
    out = []
    with open(path, "r", errors="replace") as f:
        for _ in range(n):
            ln = f.readline()
            if not ln:
                break
            out.append(ln.rstrip("\n"))
    return out


def _pretty(n: int) -> str:
    if n >= 1_000_000:
        return f"{n/1_000_000:.2f} MB"
    if n >= 1_000:
        return f"{n/1_000:.2f} KB"
    return f"{n} B"


def _ratio(raw: int, comp: int) -> float:
    return raw / max(1, comp)


def _u32(x: int) -> bytes:
    return struct.pack("<I", int(x))


def _u32_read(b: bytes, off: int) -> Tuple[int, int]:
    if off + 4 > len(b):
        raise ValueError("u32 read overflow")
    return struct.unpack("<I", b[off:off + 4])[0], off + 4


# ==========================
# HOT container (PF1 + PFQ1)
# ==========================

def hot_pack(pf1_blob: bytes, pfq1_blob: bytes) -> bytes:
    """
    USCH format:
      MAGIC (4)
      VERSION (u32)
      pf1_len (u32)
      pfq1_len (u32)
      pf1_blob
      pfq1_blob (optional; may be empty)
    """
    return (
        MAGIC_HOT
        + _u32(VERSION)
        + _u32(len(pf1_blob))
        + _u32(len(pfq1_blob))
        + pf1_blob
        + pfq1_blob
    )


def hot_unpack(blob: bytes) -> Tuple[bytes, bytes]:
    if len(blob) < 16:
        raise ValueError("bad USCH blob (too small)")
    if blob[:4] != MAGIC_HOT:
        raise ValueError("not a USCH blob")

    off = 4
    ver, off = _u32_read(blob, off)
    if ver != VERSION:
        raise ValueError(f"unsupported USCH version: {ver}")

    pf1_len, off = _u32_read(blob, off)
    pfq1_len, off = _u32_read(blob, off)

    pf1 = blob[off:off + pf1_len]
    off += pf1_len
    pfq1 = blob[off:off + pfq1_len] if pfq1_len > 0 else b""
    return pf1, pfq1


def hot_upgrade_file_in_place(hot_path: str, pf1_blob: bytes, pfq1_blob: bytes) -> None:
    upgraded = hot_pack(pf1_blob, pfq1_blob)
    with open(hot_path, "wb") as f:
        f.write(upgraded)


# ==========================
# COLD container (V1M bundle)
# ==========================

def cold_pack(bundle_blob: bytes) -> bytes:
    """
    USCC format:
      MAGIC (4)
      VERSION (u32)
      bundle_len (u32)
      bundle_blob
    """
    return MAGIC_COLD + _u32(VERSION) + _u32(len(bundle_blob)) + bundle_blob


# ==========================
# Baselines
# ==========================

def baseline_gzip(raw_bytes: bytes, level: int = 9) -> bytes:
    return gzip.compress(raw_bytes, compresslevel=int(level))


def baseline_zstd(raw_bytes: bytes, level: int = 10) -> bytes:
    if zstd is None:
        raise RuntimeError("zstandard missing (pip install zstandard)")
    return zstd.ZstdCompressor(level=int(level)).compress(raw_bytes)


# ==========================
# HOT-LAZY PFQ1 builder helper
# ==========================

def build_pfq1_from_log(
    log_path: str,
    tpl_path: str,
    lines: int,
    packet_events: int,
    zstd_level: int,
) -> bytes:
    """
    Build PFQ1 HOT index from a log.
    ✅ If tpl_path is missing/empty, fallback to raw-line token bloom packets.
    """
    from pathlib import Path

    log_p = Path(log_path)
    if not log_p.exists():
        raise SystemExit(f"❌ log not found: {log_path}")

    # read lines
    with log_p.open("r", encoding="utf-8", errors="replace") as f:
        raw_lines = []
        for i, ln in enumerate(f):
            raw_lines.append(ln.rstrip("\n"))
            if lines and i + 1 >= int(lines):
                break

    # If tpl missing -> fallback: unknown_lines = raw_lines, no events/templates
    tpl_txt = ""
    if tpl_path:
        tp = Path(tpl_path)
        if tp.exists():
            tpl_txt = tp.read_text(encoding="utf-8", errors="replace")

    if not tpl_txt.strip():
        pfq1_blob, _meta = build_pfq1(
            events=[],
            unknown_lines=raw_lines,
            template_csv_text="",
            packet_events=packet_events,
            zstd_level=zstd_level,
        )
        return pfq1_blob

    # Normal path: use existing build_pfq1() w/ template CSV
    pfq1_blob, _meta = build_pfq1(
        events=[],               # existing pipeline fills this elsewhere if needed
        unknown_lines=raw_lines, # keep unknown_lines available for PFQ1 tokens too
        template_csv_text=tpl_txt,
        packet_events=packet_events,
        zstd_level=zstd_level,
    )
    return pfq1_blob

def cmd_encode(args: argparse.Namespace) -> None:
    log_path = args.log
    tpl_path = args.tpl
    out_path = args.out
    lines = args.lines
    mode = args.mode.lower().strip()

    if not os.path.exists(log_path):
        raise SystemExit(f"❌ log file not found: {log_path}")

    print("USC ENCODE")
    print(f"mode:   {mode}")
    print(f"log:    {log_path}")
    print(f"lines:  {lines}")
    print(f"out:    {out_path}")
    print("-" * 60)

    raw_lines = _read_first_n_lines(log_path, lines)
    raw_text = "\n".join(raw_lines) + "\n"
    raw_bytes = raw_text.encode("utf-8", errors="replace")

    # STREAM: Drain3 + persistent dictionaries (v3d9 engine)
    if mode == "stream":
        t0 = time.perf_counter()
        stream_blob = encode_stream_auto(
            raw_lines,
            chunk_lines=args.chunk_lines,
            zstd_level=args.zstd,
        )
        dt = (time.perf_counter() - t0) * 1000.0

        with open(out_path, "wb") as f:
            f.write(stream_blob)

        print(f"RAW:      {_pretty(len(raw_bytes))}")
        print(f"STREAM:   {_pretty(len(stream_blob))}   build={dt:.2f} ms")
        print(f"ratio:    {_ratio(len(raw_bytes), len(stream_blob)):.2f}x")
        print("-" * 60)
        print("DONE ✅")
        return

    # Everything below needs templates
    if mode != "stream":
        if not os.path.exists(tpl_path):
            raise SystemExit(f"❌ template CSV not found: {tpl_path}")

    bank = HDFSTemplateBank.from_csv(tpl_path)
    events, unknown = parse_hdfs_lines(raw_lines, bank)
    tpl_text = open(tpl_path, "r", errors="replace").read()

    if mode == "hot":
        t0 = time.perf_counter()
        pf1_blob, _pf1_meta = build_pf1(events, unknown, tpl_text, packet_events=args.packet_events, zstd_level=args.zstd)
        dt_pf1 = (time.perf_counter() - t0) * 1000.0

        t1 = time.perf_counter()
        pfq1_blob, _pfq1_meta = build_pfq1(events, unknown, tpl_text, packet_events=args.packet_events, zstd_level=args.zstd)
        dt_pfq1 = (time.perf_counter() - t1) * 1000.0

        hot_blob = hot_pack(pf1_blob, pfq1_blob)
        with open(out_path, "wb") as f:
            f.write(hot_blob)

        print(f"RAW:     {_pretty(len(raw_bytes))}")
        print(f"PF1:     {_pretty(len(pf1_blob))}   build={dt_pf1:.2f} ms")
        print(f"PFQ1:    {_pretty(len(pfq1_blob))}   build={dt_pfq1:.2f} ms")
        print(f"USCH:    {_pretty(len(hot_blob))}   saved ✅")
        print(f"ratio:   {_ratio(len(raw_bytes), len(hot_blob)):.2f}x")

    elif mode == "hot-lite":
        rows, unknown = parse_hdfs_lines_rows(raw_lines, bank)
        tpl_text = Path(tpl_path).read_text(encoding='utf-8', errors='ignore')
        pf_blob, _meta = build_pf3_h1m2(rows, unknown, tpl_text, packet_events=args.packet_events, zstd_level=args.zstd)
        Path(args.out).write_bytes(pf_blob)
        print('USCH:', f"{len(pf_blob)/1024.0:.2f} KB", ' saved ✅ (HOT-LITE H1M2 PF3)')
        return
    elif mode == "hot-lite-full":
        rows, unknown = parse_hdfs_lines_rows(raw_lines, bank)
        tpl_text = Path(tpl_path).read_text(encoding='utf-8', errors='ignore')
        pf_blob, _meta = build_pf3_h1m2(rows, unknown, tpl_text, packet_events=args.packet_events, zstd_level=args.zstd)
        Path(args.out).write_bytes(pf_blob)
        print('USCH:', f"{len(pf_blob)/1024.0:.2f} KB", ' saved ✅ (HOT-LITE H1M2 PF3)')
        return
    elif mode == "hot-lazy":
        t0 = time.perf_counter()
        pf1_blob, _pf1_meta = build_pf1(events, unknown, tpl_text, packet_events=args.packet_events, zstd_level=args.zstd)
        dt_pf1 = (time.perf_counter() - t0) * 1000.0

        hot_blob = hot_pack(pf1_blob, b"")
        with open(out_path, "wb") as f:
            f.write(hot_blob)

        print(f"RAW:     {_pretty(len(raw_bytes))}")
        print(f"PF1:     {_pretty(len(pf1_blob))}   build={dt_pf1:.2f} ms")
        print(f"USCH:    {_pretty(len(hot_blob))}   saved ✅ (HOT-LAZY)")
        print(f"ratio:   {_ratio(len(raw_bytes), len(hot_blob)):.2f}x")

    elif mode == "cold-oracle":
        if not tpl_path:
            raise SystemExit("cold-oracle requires --tpl")
        bank = HDFSTemplateBank.from_csv(tpl_path)
        events, unknown = parse_hdfs_lines(raw_lines, bank)
        blob = encode_template_channels_v1_mask(events, unknown)
        Path(args.out).write_bytes(blob)
        print('BUNDLE:', f"{len(blob)/1024.0:.2f} KB", ' build=oracle')
        print('USCC:',   f"{len(blob)/1024.0:.2f} KB", ' saved ✅')
        return

    elif mode == "cold":
        t0 = time.perf_counter()
        bundle_blob, _meta = bundle_encode_and_compress_v1m(
            events=events,
            unknown_lines=unknown,
            template_csv_text=tpl_text,
            zstd_level=args.zstd,
        )
        cold_blob = cold_pack(bundle_blob)

        # Smart fallback: try raw zstd-22, trained-dict zstd, brotli-11, bzip2-9
        zstd_level_fb = max(args.zstd, 22)
        raw_zstd = zstd.ZstdCompressor(level=zstd_level_fb).compress(raw_bytes) if zstd else raw_bytes
        uszr_blob = MAGIC_USZR + raw_zstd

        candidates = [(cold_blob, "USCC"), (uszr_blob, "USZR")]

        # Trained-dictionary zstd (learns patterns from data, no templates needed)
        try:
            if zstd:
                sample_size = 4096
                samples = [raw_bytes[i:i+sample_size] for i in range(0, len(raw_bytes), sample_size)]
                if len(samples) >= 8:
                    dict_data = zstd.train_dictionary(131072, samples)
                    cctx = zstd.ZstdCompressor(level=zstd_level_fb, dict_data=dict_data)
                    dict_bytes = dict_data.as_bytes()
                    raw_zstd_dict = cctx.compress(raw_bytes)
                    uszd_blob = b"USZD" + struct.pack("<I", len(dict_bytes)) + dict_bytes + raw_zstd_dict
                    candidates.append((uszd_blob, "USZD"))
        except Exception:
            pass

        try:
            import brotli
            raw_br = brotli.compress(raw_bytes, quality=11)
            usbr_blob = MAGIC_USBR + raw_br
            candidates.append((usbr_blob, "USBR"))
        except ImportError:
            pass

        import bz2
        raw_bz2 = bz2.compress(raw_bytes, compresslevel=9)
        usbz_blob = b"USBZ" + raw_bz2
        candidates.append((usbz_blob, "USBZ"))

        # Pick smallest
        winner_blob, winner_label = min(candidates, key=lambda x: len(x[0]))
        dt = (time.perf_counter() - t0) * 1000.0

        with open(out_path, "wb") as f:
            f.write(winner_blob)

        print(f"RAW:     {_pretty(len(raw_bytes))}")
        print(f"BUNDLE:  {_pretty(len(bundle_blob))}   build={dt:.2f} ms")
        print(f"{winner_label}:    {_pretty(len(winner_blob))}   saved ✅ (best of {len(candidates)} candidates)")
        print(f"ratio:   {_ratio(len(raw_bytes), len(winner_blob)):.2f}x")

    else:
        raise SystemExit("❌ mode must be: hot | hot-lite | hot-lazy | cold | stream")

    print("-" * 60)
    print("DONE ✅")


def cmd_query(args: argparse.Namespace) -> None:

    # enforce required inputs by mode
    if getattr(args, "mode", "hot") != "hot-lite-full":
        if not getattr(args, "hot", None):
            raise SystemExit("❌ query --mode hot requires --hot <blob>")
    # HOT-LITE-FULL QUERY (DECODE+SCAN) — v0
    # Makes USC queryable on real storage blobs immediately.
    # Later replaced by packet-bloom prefilter + partial decode.
    # ------------------------------------------------------------
    if getattr(args, "mode", "hot") == "hot-lite-full" or getattr(args, "input", None):
        import os
        import tempfile
        import subprocess

        bin_path = args.input or args.hot
        ql = str(args.q).lower()
        limit = int(args.limit)

        tmp_fd, tmp_path = tempfile.mkstemp(prefix="usc_q_", suffix=".log")
        os.close(tmp_fd)

        try:
            subprocess.run(
                [
                    "python3", "-m", "usc.cli.app", "decode",
                    "--mode", "hot-lite-full",
                    "--input", str(bin_path),
                    "--out", str(tmp_path),
                ],
                check=True,
            )

            hits = 0
            with open(tmp_path, "r", encoding="utf-8", errors="replace") as f:
                for line in f:
                    if ql in line.lower():
                        print(line.rstrip("\n"))
                        hits += 1
                        if hits >= limit:
                            break

            raise SystemExit(0 if hits > 0 else 1)

        finally:
            try:
                os.remove(tmp_path)
            except Exception:
                pass

    hot_path = args.hot
    q = args.q
    limit = args.limit

    if not Path(hot_path).exists():
        raise SystemExit(f"❌ hot file not found: {hot_path}")

    blob = open(hot_path, "rb").read()
    pf1_blob, pfq1_blob = hot_unpack(blob)

    print("USC QUERY (HOT)")
    print(f"hot:   {hot_path}")
    print(f"q:     {q!r}")
    print(f"limit: {limit}")
    print("-" * 60)

    # FAST attempt always
    t0 = time.perf_counter()
    hits_fast, _cands = query_fast_pf1(pf1_blob, q, limit=limit)
    dt_fast = (time.perf_counter() - t0) * 1000.0

    if hits_fast:
        print("mode: FAST")
        print(f"hits: {len(hits_fast)}   time={dt_fast:.2f} ms")
        print("-" * 60)
        for i, h in enumerate(hits_fast[: min(limit, 10)]):
            print(f"{i+1:02d}) {h[:200]}")
        print("-" * 60)
        print("DONE ✅")
        return

    # If no PFQ1 exists, optionally upgrade (HOT-LAZY)
    if len(pfq1_blob) == 0:
        print("mode: FAST")
        print(f"hits: 0   time={dt_fast:.2f} ms")
        print("-" * 60)
        print("No hits in FAST-only mode.")

        if not args.upgrade:
            print("Tip: add universal fallback by encoding with --mode hot OR upgrade this file on demand:")
            print("  python -m usc query --hot <file.usch> --q \"...\" --upgrade --log <log> --tpl <tpl>")
            print("-" * 60)
            print("DONE ✅")
            return

        if args.log is None or args.tpl is None:
            raise SystemExit("❌ upgrade requires --log and --tpl")

        print("UPGRADE: Building PFQ1 fallback now (one-time cost)...")
        t1 = time.perf_counter()
        pfq1_new = build_pfq1_from_log(
            log_path=args.log,
            tpl_path=args.tpl,
            lines=args.lines,
            packet_events=args.packet_events,
            zstd_level=args.zstd,
        )
        dt_build = (time.perf_counter() - t1) * 1000.0

        hot_upgrade_file_in_place(hot_path, pf1_blob, pfq1_new)
        print(f"UPGRADE DONE ✅ PFQ1 bytes={_pretty(len(pfq1_new))} build={dt_build:.2f} ms")
        print("Re-running query with router (FAST → PFQ1 fallback)...")
        print("-" * 60)

        t2 = time.perf_counter()
        hits2, mode2 = query_router_v1(pf1_blob, pfq1_new, q, limit=limit)
        dt2 = (time.perf_counter() - t2) * 1000.0

        print(f"mode: {mode2}")
        print(f"hits: {len(hits2)}   time={dt2:.2f} ms")
        if hits2:
            print("-" * 60)
            for i, h in enumerate(hits2[: min(limit, 10)]):
                print(f"{i+1:02d}) {h[:200]}")
        print("-" * 60)
        print("DONE ✅")
        return

    # Full HOT path
    t0 = time.perf_counter()
    hits, mode = query_router_v1(pf1_blob, pfq1_blob, q, limit=limit)
    dt = (time.perf_counter() - t0) * 1000.0

    print(f"mode: {mode}")
    print(f"hits: {len(hits)}   time={dt:.2f} ms")
    if hits:
        print("-" * 60)
        for i, h in enumerate(hits[: min(limit, 10)]):
            print(f"{i+1:02d}) {h[:200]}")
    print("-" * 60)
    print("DONE ✅")


def _row(name: str, size: int, raw: int, ms: float) -> Dict[str, object]:
    return {
        "name": name,
        "bytes": int(size),
        "pretty": _pretty(int(size)),
        "ratio": float(_ratio(raw, size)),
        "ms": float(ms),
    }


def cmd_bench(args: argparse.Namespace) -> None:
    log_path = args.log
    tpl_path = args.tpl
    lines = args.lines

    if not os.path.exists(log_path):
        raise SystemExit(f"❌ log file not found: {log_path}")
    if not os.path.exists(tpl_path):
        raise SystemExit(f"❌ template CSV not found: {tpl_path}")

    raw_lines = _read_first_n_lines(log_path, lines)
    raw_text = "\n".join(raw_lines) + "\n"
    raw_bytes = raw_text.encode("utf-8", errors="replace")
    raw_n = len(raw_bytes)

    bank = HDFSTemplateBank.from_csv(tpl_path)
    events, unknown = parse_hdfs_lines(raw_lines, bank)
    tpl_text = open(tpl_path, "r", errors="replace").read()

    print("USC BENCH — Scoreboard")
    print(f"log:   {log_path}")
    print(f"tpl:   {tpl_path}")
    print(f"lines: {len(raw_lines)}")
    print(f"RAW:   {_pretty(raw_n)}")
    print("-" * 72)

    rows: List[Dict[str, object]] = []

    # gzip
    t0 = time.perf_counter()
    gz = baseline_gzip(raw_bytes, level=args.gzip)
    t_gz = (time.perf_counter() - t0) * 1000.0
    rows.append(_row(f"gzip-{args.gzip}", len(gz), raw_n, t_gz))

    # zstd
    if zstd is not None:
        t0 = time.perf_counter()
        zs = baseline_zstd(raw_bytes, level=args.zstd)
        t_zs = (time.perf_counter() - t0) * 1000.0
        rows.append(_row(f"zstd-{args.zstd}", len(zs), raw_n, t_zs))

    # STREAM
    t0 = time.perf_counter()
    stream_blob = encode_stream_auto(raw_lines, chunk_lines=args.chunk_lines, zstd_level=args.zstd)
    t_stream = (time.perf_counter() - t0) * 1000.0
    rows.append(_row("USC-STREAM (v3d9)", len(stream_blob), raw_n, t_stream))

    # HOT-LITE
    t0 = time.perf_counter()
    pf1_blob, _m1 = build_pf1(events, unknown, tpl_text, packet_events=args.packet_events, zstd_level=args.zstd)
    t_pf1 = (time.perf_counter() - t0) * 1000.0
    usch_lite = hot_pack(pf1_blob, b"")
    rows.append(_row("USC-HOT-LITE (PF1)", len(usch_lite), raw_n, t_pf1))

    # HOT
    t0 = time.perf_counter()
    pf1_blob2, _m2 = build_pf1(events, unknown, tpl_text, packet_events=args.packet_events, zstd_level=args.zstd)
    t_pf1b = (time.perf_counter() - t0) * 1000.0

    t0 = time.perf_counter()
    pfq1_blob, _m3 = build_pfq1(events, unknown, tpl_text, packet_events=args.packet_events, zstd_level=args.zstd)
    t_pfq1 = (time.perf_counter() - t0) * 1000.0

    usch = hot_pack(pf1_blob2, pfq1_blob)
    rows.append(_row("USC-HOT (USCH)", len(usch), raw_n, t_pf1b + t_pfq1))

    # COLD
    t0 = time.perf_counter()
    bundle_blob, _meta = bundle_encode_and_compress_v1m(
        events=events,
        unknown_lines=unknown,
        template_csv_text=tpl_text,
        zstd_level=args.zstd,
    )
    t_cold = (time.perf_counter() - t0) * 1000.0
    uscc = cold_pack(bundle_blob)
    rows.append(_row("USC-COLD (USCC)", len(uscc), raw_n, t_cold))

    rows_sorted = sorted(rows, key=lambda x: x["bytes"])

    print(f"{'METHOD':24} {'SIZE':12} {'RATIO':10} {'BUILD(ms)':10}")
    print("-" * 72)
    for r in rows_sorted:
        print(f"{r['name'][:24]:24} {r['pretty']:12} {r['ratio']:>9.2f}x {r['ms']:>10.2f}")

    print("-" * 72)

    if args.out_json:
        payload = {
            "ts": time.time(),
            "log": log_path,
            "tpl": tpl_path,
            "lines": len(raw_lines),
            "raw_bytes": raw_n,
            "packet_events": args.packet_events,
            "chunk_lines": args.chunk_lines,
            "gzip_level": args.gzip,
            "zstd_level": args.zstd,
            "results": rows_sorted,
        }
        with open(args.out_json, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        print(f"saved_json: {args.out_json} ✅")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="usc",
        description="Unified State Codec (USC) — Hot/Cool/Stream compression + query",
    )
    sub = p.add_subparsers(dest="cmd", required=True)
    add_decode_auto_subcommand(sub)

    # -----------------------
    # decode (PF3 hot-lite)
    # -----------------------
    p_decode = sub.add_parser("decode", help="Decode USC blobs back to lines (hot-lite PF3)")
    p_decode.add_argument("--mode", default="hot-lite", choices=["hot-lite", "hot-lite-full"], help="Decode mode")
    p_decode.add_argument("--input", "--in", dest="input", required=True, help="Input .bin file")
    p_decode.add_argument("--out", "--output", dest="out", required=True, help="Output .log file")

    enc = sub.add_parser("encode", help="Encode a log into HOT/HOT-LITE/HOT-LAZY/COLD/STREAM")
    enc.add_argument("--mode", choices=["hot", "hot-lite", "hot-lazy", "cold", "stream", "hot-lite-full"], required=True)
    enc.add_argument("--log", default="data/loghub/HDFS.log")
    enc.add_argument("--tpl", default="data/loghub/preprocessed/HDFS.log_templates.csv")
    enc.add_argument("--out", required=True)
    enc.add_argument("--lines", type=int, default=200000)
    enc.add_argument("--packet_events", type=int, default=32768)
    enc.add_argument("--chunk_lines", type=int, default=25, help="STREAM only: chunk size in lines")
    enc.add_argument("--zstd", type=int, default=10)
    enc.set_defaults(func=cmd_encode)

    qry = sub.add_parser("query", help="Query a HOT/HOT-LITE/HOT-LAZY USC blob")
    qry.add_argument("--hot")
    qry.add_argument("--q", required=True)
    qry.add_argument("--limit", type=int, default=25)
    qry.add_argument("--mode", choices=["hot", "hot-lite-full"], default="hot")
    qry.add_argument("--input", "--in", dest="input", default=None)

    qry.add_argument("--upgrade", action="store_true")
    qry.add_argument("--log", default=None)
    qry.add_argument("--tpl", default=None)
    qry.add_argument("--lines", type=int, default=200000)
    qry.add_argument("--packet_events", type=int, default=32768)
    qry.add_argument("--zstd", type=int, default=10)
    qry.set_defaults(func=cmd_query)

    b = sub.add_parser("bench", help="Run baselines + USC modes and print a scoreboard")
    b.add_argument("--log", default="data/loghub/HDFS.log")
    b.add_argument("--tpl", default="data/loghub/preprocessed/HDFS.log_templates.csv")
    b.add_argument("--lines", type=int, default=200000)
    b.add_argument("--packet_events", type=int, default=32768)
    b.add_argument("--chunk_lines", type=int, default=25, help="STREAM only: chunk size in lines")
    b.add_argument("--gzip", type=int, default=9)
    b.add_argument("--zstd", type=int, default=10)
    b.add_argument("--out_json", default=None)
    b.set_defaults(func=cmd_bench)

    return p


def main():
    p = build_parser()
    args = p.parse_args()
    # -----------------------
    # decode command (PF3 hot-lite)
    # -----------------------
    if getattr(args, "cmd", None) == "decode":
        from pathlib import Path

        in_path = Path(args.input)
        out_path = Path(args.out)

        blob = in_path.read_bytes()

        magic = b"TPF3"

        # ✅ Robust: the file may contain multiple 'TPF3' occurrences.
        # We try each one until PF3 decode succeeds.
        offs = []
        start = 0
        while True:
            j = blob.find(magic, start)
            if j < 0:
                break
            offs.append(j)
            start = j + 1

        if not offs:
            raise SystemExit("❌ Could not find PF3 magic (TPF3) anywhere inside input file")

        last_err = None
        lines = None

        for off in offs:
            try:
                pf3 = blob[off:]
                lines = decode_pf3_h1m2_to_lines(pf3)
                # success
                break
            except Exception as e:
                last_err = e
                continue

        if lines is None:
            raise SystemExit(f"❌ Found TPF3 markers but none decoded successfully. Last error: {last_err}")

        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print("✅ decoded:", len(lines), "lines →", str(out_path))
        return
    args.func(args)


if __name__ == "__main__":
    main()
