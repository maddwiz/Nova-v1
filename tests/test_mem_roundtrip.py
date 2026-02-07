from usc.bench.datasets import toy_agent_log
from usc.mem.codec import mem_encode, mem_decode, mem_decode_with_fallback
from usc.mem.commit import commit_memory, load_last_commit


def test_mem_roundtrip_tier3_lossless():
    raw = toy_agent_log()
    pkt = mem_encode(raw, tier=3)

    decoded_text, conf = mem_decode(pkt)

    assert decoded_text == raw
    assert conf >= 0.90


def test_mem_decode_tier0_not_exact_but_valid():
    raw = toy_agent_log()
    pkt = mem_encode(raw, tier=0)

    decoded_text, conf = mem_decode(pkt)

    assert isinstance(decoded_text, str)
    assert len(decoded_text) > 0
    assert conf >= 0.60


def test_auto_tier_escalation_upgrades_to_tier3():
    raw = toy_agent_log()

    pkt0 = mem_encode(raw, tier=0)
    pkt3 = mem_encode(raw, tier=3)

    decoded_text, conf, used_tier = mem_decode_with_fallback(
        packets_low_to_high=[pkt0, pkt3],
        min_conf=0.80,
    )

    assert used_tier == 3
    assert decoded_text == raw
    assert conf >= 0.90


def test_commit_store_writes_and_reads(tmp_path):
    raw = toy_agent_log()

    pkt0 = mem_encode(raw, tier=0)
    pkt3 = mem_encode(raw, tier=3)

    decoded_text, conf, used_tier = mem_decode_with_fallback(
        packets_low_to_high=[pkt0, pkt3],
        min_conf=0.80,
    )

    store_path = tmp_path / "commits.jsonl"

    rec = commit_memory(
        store_path=str(store_path),
        packet_version="mem-v0.7",
        used_tier=used_tier,
        confidence=conf,
        decoded_text=decoded_text,
    )

    last = load_last_commit(str(store_path))

    assert last is not None
    assert last.text_fingerprint == rec.text_fingerprint
    assert last.used_tier == rec.used_tier
