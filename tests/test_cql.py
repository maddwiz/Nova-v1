"""Tests for N1: CQL — Codec Query Language."""
import pytest
from usc.cql import parse_cql, execute_cql, CQLRow, CQLResult


def _sample_rows():
    return [
        CQLRow(text="User alice logged in from 10.0.0.1",
               template="User <*> logged in from <*>",
               slots=["alice", "10.0.0.1"]),
        CQLRow(text="User bob logged in from 10.0.0.2",
               template="User <*> logged in from <*>",
               slots=["bob", "10.0.0.2"]),
        CQLRow(text="Error: connection timeout for server-1",
               template="Error: connection timeout for <*>",
               slots=["server-1"]),
        CQLRow(text="Metric: cpu_usage=95.2 host=web-01",
               template="Metric: cpu_usage=<*> host=<*>",
               slots=["95.2", "web-01"]),
        CQLRow(text="Metric: cpu_usage=72.5 host=web-02",
               template="Metric: cpu_usage=<*> host=<*>",
               slots=["72.5", "web-02"]),
        CQLRow(text="Error: disk full on server-2",
               template="Error: disk full on <*>",
               slots=["server-2"]),
    ]


class TestParser:
    def test_parse_select_lines(self):
        q = parse_cql("SELECT lines WHERE template LIKE '%error%'")
        assert q.target == "lines"
        assert q.where is not None
        assert q.limit is None

    def test_parse_select_count(self):
        q = parse_cql("SELECT count() WHERE template LIKE '%timeout%'")
        assert q.target == "count"

    def test_parse_with_limit(self):
        q = parse_cql("SELECT lines WHERE text LIKE '%user%' LIMIT 5")
        assert q.limit == 5

    def test_parse_comparison(self):
        q = parse_cql("SELECT lines WHERE slot[0] > 80")
        assert q.target == "lines"

    def test_parse_and(self):
        q = parse_cql("SELECT lines WHERE template LIKE '%error%' AND slot[0] = 'server-1'")
        assert q.target == "lines"

    def test_parse_or(self):
        q = parse_cql("SELECT lines WHERE template LIKE '%error%' OR template LIKE '%metric%'")
        assert q.target == "lines"

    def test_parse_aggregate(self):
        q = parse_cql("SELECT avg(slot[0]) WHERE template LIKE '%cpu_usage%'")
        assert q.target == "avg"
        assert q.func_field == "slot[0]"

    def test_parse_invalid_syntax(self):
        with pytest.raises(SyntaxError):
            parse_cql("INVALID QUERY")

    def test_parse_no_where(self):
        q = parse_cql("SELECT lines LIMIT 3")
        assert q.where is None
        assert q.limit == 3

    def test_parse_count_no_where(self):
        q = parse_cql("SELECT count()")
        assert q.target == "count"
        assert q.where is None


class TestExecutor:
    def test_select_lines_with_like(self):
        rows = _sample_rows()
        q = parse_cql("SELECT lines WHERE template LIKE '%error%'")
        result = execute_cql(q, rows)
        assert len(result.rows) == 2
        assert all("Error" in r for r in result.rows)

    def test_select_lines_with_limit(self):
        rows = _sample_rows()
        q = parse_cql("SELECT lines LIMIT 2")
        result = execute_cql(q, rows)
        assert len(result.rows) == 2

    def test_select_count(self):
        rows = _sample_rows()
        q = parse_cql("SELECT count() WHERE template LIKE '%logged in%'")
        result = execute_cql(q, rows)
        assert result.count == 2

    def test_select_count_no_where(self):
        rows = _sample_rows()
        q = parse_cql("SELECT count()")
        result = execute_cql(q, rows)
        assert result.count == 6

    def test_comparison_numeric(self):
        rows = _sample_rows()
        q = parse_cql("SELECT lines WHERE slot[0] > 80")
        result = execute_cql(q, rows)
        # 95.2 > 80, others are strings or < 80
        assert len(result.rows) >= 1
        assert "95.2" in result.rows[0]

    def test_and_condition(self):
        rows = _sample_rows()
        q = parse_cql("SELECT lines WHERE template LIKE '%cpu_usage%' AND slot[0] > 90")
        result = execute_cql(q, rows)
        assert len(result.rows) == 1
        assert "95.2" in result.rows[0]

    def test_or_condition(self):
        rows = _sample_rows()
        q = parse_cql("SELECT lines WHERE slot[0] = 'alice' OR slot[0] = 'bob'")
        result = execute_cql(q, rows)
        assert len(result.rows) == 2

    def test_avg_aggregate(self):
        rows = _sample_rows()
        q = parse_cql("SELECT avg(slot[0]) WHERE template LIKE '%cpu_usage%'")
        result = execute_cql(q, rows)
        assert result.aggregate is not None
        assert abs(result.aggregate - 83.85) < 0.01

    def test_min_aggregate(self):
        rows = _sample_rows()
        q = parse_cql("SELECT min(slot[0]) WHERE template LIKE '%cpu_usage%'")
        result = execute_cql(q, rows)
        assert result.aggregate == 72.5

    def test_max_aggregate(self):
        rows = _sample_rows()
        q = parse_cql("SELECT max(slot[0]) WHERE template LIKE '%cpu_usage%'")
        result = execute_cql(q, rows)
        assert result.aggregate == 95.2

    def test_no_matching_rows(self):
        rows = _sample_rows()
        q = parse_cql("SELECT lines WHERE template LIKE '%nonexistent%'")
        result = execute_cql(q, rows)
        assert result.rows == []

    def test_aggregate_no_numeric_values(self):
        rows = _sample_rows()
        q = parse_cql("SELECT avg(slot[0]) WHERE template LIKE '%logged in%'")
        result = execute_cql(q, rows)
        # "alice" and "bob" are not numeric
        assert result.aggregate is None

    def test_empty_rows(self):
        q = parse_cql("SELECT count()")
        result = execute_cql(q, [])
        assert result.count == 0
