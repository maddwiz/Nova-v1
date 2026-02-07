"""Tests for N2: Forensic Mode."""
from usc.forensic import ForensicAudit, AuditResult, ForensicDiff


def _sample_packets():
    return [b"packet-0-data", b"packet-1-data", b"packet-2-data"]


class TestIntegrityVerification:
    def test_good_data_passes(self):
        pkts = _sample_packets()
        audit = ForensicAudit(pkts)
        result = audit.verify_integrity()
        assert result.valid is True
        assert result.total_packets == 3
        assert result.verified_packets == 3
        assert result.corrupted_indices == []

    def test_tampered_packet_detected(self):
        pkts = _sample_packets()
        audit = ForensicAudit(pkts)
        # Tamper a packet after creating audit
        audit.packets[1] = b"TAMPERED"
        result = audit.verify_integrity()
        assert result.valid is False
        assert 1 in result.corrupted_indices

    def test_extra_packet_detected(self):
        pkts = _sample_packets()
        audit = ForensicAudit(pkts)
        audit.packets.append(b"extra")
        result = audit.verify_integrity()
        assert result.valid is False

    def test_empty_packets(self):
        audit = ForensicAudit([])
        result = audit.verify_integrity()
        assert result.valid is True
        assert result.total_packets == 0


class TestAuditTrail:
    def test_initial_trail_has_encode_entries(self):
        pkts = _sample_packets()
        audit = ForensicAudit(pkts)
        trail = audit.chain_of_custody()
        assert len(trail) == 3
        assert all(e.operation == "encode" for e in trail)

    def test_record_operation_adds_entry(self):
        pkts = _sample_packets()
        audit = ForensicAudit(pkts)
        audit.record_operation("verify", 0)
        trail = audit.chain_of_custody()
        assert len(trail) == 4
        assert trail[-1].operation == "verify"

    def test_entries_have_timestamps(self):
        pkts = _sample_packets()
        audit = ForensicAudit(pkts)
        trail = audit.chain_of_custody()
        for entry in trail:
            assert entry.timestamp > 0
            assert len(entry.packet_hash) == 32

    def test_trail_sorted_chronologically(self):
        pkts = _sample_packets()
        audit = ForensicAudit(pkts)
        audit.record_operation("modify", 1)
        audit.record_operation("verify", 2)
        trail = audit.chain_of_custody()
        timestamps = [e.timestamp for e in trail]
        assert timestamps == sorted(timestamps)


class TestDiffStates:
    def test_identical_packets(self):
        pkts = _sample_packets()
        diff = ForensicAudit.diff_states(pkts, pkts)
        assert diff.unchanged_count == 3
        assert diff.changed_indices == []
        assert diff.added_indices == []
        assert diff.removed_indices == []

    def test_changed_packet(self):
        pkts_a = _sample_packets()
        pkts_b = [pkts_a[0], b"CHANGED", pkts_a[2]]
        diff = ForensicAudit.diff_states(pkts_a, pkts_b)
        assert diff.unchanged_count == 2
        assert diff.changed_indices == [1]

    def test_added_packets(self):
        pkts_a = _sample_packets()
        pkts_b = pkts_a + [b"new-1", b"new-2"]
        diff = ForensicAudit.diff_states(pkts_a, pkts_b)
        assert diff.unchanged_count == 3
        assert diff.added_indices == [3, 4]
        assert diff.removed_indices == []

    def test_removed_packets(self):
        pkts_a = _sample_packets()
        pkts_b = pkts_a[:1]
        diff = ForensicAudit.diff_states(pkts_a, pkts_b)
        assert diff.unchanged_count == 1
        assert diff.removed_indices == [1, 2]
        assert diff.added_indices == []

    def test_empty_to_nonempty(self):
        diff = ForensicAudit.diff_states([], [b"a", b"b"])
        assert diff.unchanged_count == 0
        assert diff.added_indices == [0, 1]

    def test_nonempty_to_empty(self):
        diff = ForensicAudit.diff_states([b"a", b"b"], [])
        assert diff.unchanged_count == 0
        assert diff.removed_indices == [0, 1]
