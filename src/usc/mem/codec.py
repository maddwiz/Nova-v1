import json
import gzip
from dataclasses import asdict
from typing import Dict, Any, Tuple, List

from usc.mem.skeleton import extract_skeleton, render_skeleton
from usc.mem.witnesses import extract_witnesses
from usc.mem.residuals import extract_residual
from usc.mem.ecc import make_ecc, verify_ecc
from usc.mem.fingerprint import make_fingerprint
from usc.mem.probes import probe_truth_spine, confidence_score


class USCNeedsMoreBits(Exception):
    """
    Raised when we refuse to decode at a low tier because confidence is too low.
    This is USC's "do not hallucinate" safety behavior.
    """
    pass


def mem_encode(text: str, tier: int = 3) -> bytes:
    """
    USC-MEM v0.7
    - Tiers (0 = tiny, 3 = lossless)
    - Light ECC (truth spine verification)
    - Fingerprint (behavior-id)
    - Probes + confidence gates

    Packet is JSON with short keys, gzip-compressed.
    """
    sk = extract_skeleton(text)
    sk_txt = render_skeleton(sk)
    wit = extract_witnesses(text)

    if tier == 3:
        res = extract_residual(text, sk_txt)
        residual_payload = res.text
    elif tier == 0:
        residual_payload = ""
    else:
        raise ValueError("tier must be 0 or 3")

    ecc = make_ecc(sk.header, sk.goal, wit.lines)
    fp = make_fingerprint(sk.header, sk.goal, wit.lines)

    packet: Dict[str, Any] = {
        "v": "mem-v0.7",
        "t": tier,
        "sk": asdict(sk),
        "w": wit.lines,
        "r": residual_payload,
        "e": ecc,
        "f": fp,
    }

    raw = json.dumps(packet, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return gzip.compress(raw, compresslevel=9)


def mem_decode(packet_bytes: bytes, min_conf: float = 0.60) -> Tuple[str, float]:
    """
    Decode and return (text, confidence).
    If confidence < min_conf, USC refuses to silently hallucinate.
    """
    raw = gzip.decompress(packet_bytes)
    pkt = json.loads(raw.decode("utf-8"))

    tier = int(pkt.get("t", 3))
    sk = pkt["sk"]
    witnesses = pkt.get("w", [])
    ecc = pkt.get("e", "")
    fp = pkt.get("f", "")

    header = sk.get("header", "")
    goal = sk.get("goal", "")

    # 1) Truth spine verification
    if not verify_ecc(ecc, header, goal, witnesses):
        raise ValueError("ECC verification failed: truth spine mismatch")

    # 2) Fingerprint verification
    want_fp = make_fingerprint(header, goal, witnesses)
    if fp != want_fp:
        raise ValueError("Fingerprint verification failed")

    # 3) Probe verification
    probe_ok = probe_truth_spine(header, goal, witnesses)

    # 4) Confidence score
    conf = confidence_score(tier=tier, probe_ok=probe_ok)

    # 5) Confidence gate
    if conf < min_conf:
        raise USCNeedsMoreBits(
            f"Decode confidence too low ({conf:.2f} < {min_conf:.2f}). Need higher tier."
        )

    sk_txt = f"{header}\n{goal}\n"
    residual_text = pkt.get("r", "")
    return sk_txt + residual_text, conf


def mem_decode_with_fallback(
    packets_low_to_high: List[bytes],
    min_conf: float = 0.80,
) -> Tuple[str, float, int]:
    """
    Self-healing decode:
    - Try packets in order (tier 0 first, then tier 3).
    - If low tier fails confidence, automatically upgrade.

    Returns: (decoded_text, confidence, used_tier)
    """
    last_err: Exception | None = None

    for pkt_bytes in packets_low_to_high:
        try:
            decoded_text, conf = mem_decode(pkt_bytes, min_conf=min_conf)

            # figure out tier used (read packet metadata)
            raw = gzip.decompress(pkt_bytes)
            pkt = json.loads(raw.decode("utf-8"))
            used_tier = int(pkt.get("t", 3))

            return decoded_text, conf, used_tier
        except USCNeedsMoreBits as e:
            last_err = e
            continue

    # If we got here, everything failed
    if last_err:
        raise last_err
    raise ValueError("All decode attempts failed")
