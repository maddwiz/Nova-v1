"""
N1: CQL Executor — runs parsed CQL queries against decoded data.
"""
from __future__ import annotations

import fnmatch
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from usc.cql.parser import (
    SelectQuery,
    WhereExpr,
    LikeExpr,
    CompareExpr,
    AndExpr,
    OrExpr,
)


@dataclass
class CQLRow:
    """A row of decoded data for CQL to query."""
    text: str
    template: str = ""
    slots: List[str] = None

    def __post_init__(self):
        if self.slots is None:
            self.slots = []


@dataclass
class CQLResult:
    """Query execution result."""
    rows: List[str]
    aggregate: Optional[float] = None
    count: Optional[int] = None


def _sql_like_to_regex(pattern: str) -> re.Pattern:
    """Convert SQL LIKE pattern (% and _) to Python regex."""
    # First replace % and _ with placeholders, then escape, then restore
    result = []
    i = 0
    while i < len(pattern):
        ch = pattern[i]
        if ch == "%":
            result.append(".*")
        elif ch == "_":
            result.append(".")
        else:
            result.append(re.escape(ch))
        i += 1
    return re.compile(f"^{''.join(result)}$", re.IGNORECASE)


def _get_field_value(row: CQLRow, field: str) -> str:
    """Extract a field value from a row."""
    f = field.lower()
    if f == "template":
        return row.template
    if f == "text":
        return row.text
    # slot[N]
    m = re.match(r"slot\[(\d+)\]", f)
    if m:
        idx = int(m.group(1))
        if idx < len(row.slots):
            return row.slots[idx]
        return ""
    return row.text


def _evaluate_where(row: CQLRow, expr: WhereExpr) -> bool:
    """Evaluate a WHERE expression against a row."""
    if isinstance(expr, LikeExpr):
        value = _get_field_value(row, expr.field)
        pattern = _sql_like_to_regex(expr.pattern)
        result = bool(pattern.match(value))
        return not result if expr.negate else result

    if isinstance(expr, CompareExpr):
        value = _get_field_value(row, expr.field)
        # Try numeric comparison if the RHS looks numeric
        try:
            rhs = float(expr.value)
            lhs = float(value)
            if expr.op == ">":
                return lhs > rhs
            if expr.op == "<":
                return lhs < rhs
            if expr.op == ">=":
                return lhs >= rhs
            if expr.op == "<=":
                return lhs <= rhs
            if expr.op == "=":
                return lhs == rhs
            if expr.op == "!=":
                return lhs != rhs
        except ValueError:
            # If RHS is numeric but LHS isn't, the row doesn't match numeric comparisons
            try:
                float(expr.value)
                # RHS is numeric, LHS is not → skip this row for >, <, >=, <=
                if expr.op in (">", "<", ">=", "<="):
                    return False
            except ValueError:
                pass
            # String comparison
            if expr.op == "=":
                return value == expr.value
            if expr.op == "!=":
                return value != expr.value
            return value > expr.value if expr.op == ">" else value < expr.value

    if isinstance(expr, AndExpr):
        return _evaluate_where(row, expr.left) and _evaluate_where(row, expr.right)

    if isinstance(expr, OrExpr):
        return _evaluate_where(row, expr.left) or _evaluate_where(row, expr.right)

    return True


def execute_cql(query: SelectQuery, rows: List[CQLRow]) -> CQLResult:
    """Execute a parsed CQL query against a list of rows."""
    # Filter rows through WHERE clause
    if query.where:
        filtered = [r for r in rows if _evaluate_where(r, query.where)]
    else:
        filtered = list(rows)

    # Apply LIMIT
    if query.limit is not None:
        filtered = filtered[:query.limit]

    # Handle aggregates
    if query.target == "count":
        return CQLResult(rows=[], count=len(filtered))

    if query.target in ("min", "max", "avg"):
        values = []
        for r in filtered:
            val = _get_field_value(r, query.func_field or "text")
            try:
                values.append(float(val))
            except ValueError:
                pass

        if not values:
            return CQLResult(rows=[], aggregate=None)

        if query.target == "min":
            return CQLResult(rows=[], aggregate=min(values))
        if query.target == "max":
            return CQLResult(rows=[], aggregate=max(values))
        if query.target == "avg":
            return CQLResult(rows=[], aggregate=sum(values) / len(values))

    # Default: return lines
    return CQLResult(rows=[r.text for r in filtered])
