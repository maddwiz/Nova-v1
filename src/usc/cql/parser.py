"""
N1: CQL Parser — Codec Query Language.

Hand-rolled recursive descent parser for a simple SQL-like DSL:
    SELECT lines WHERE template LIKE '%error%' AND slot[0] > 100 LIMIT 10
    SELECT count() WHERE template LIKE '%timeout%'
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional, Union


# ---------- AST Nodes ----------

@dataclass
class LikeExpr:
    field: str
    pattern: str
    negate: bool = False


@dataclass
class CompareExpr:
    field: str
    op: str  # >, <, >=, <=, =, !=
    value: str


@dataclass
class AndExpr:
    left: "WhereExpr"
    right: "WhereExpr"


@dataclass
class OrExpr:
    left: "WhereExpr"
    right: "WhereExpr"


WhereExpr = Union[LikeExpr, CompareExpr, AndExpr, OrExpr]


@dataclass
class SelectQuery:
    target: str  # "lines", "count", "min", "max", "avg"
    func_field: Optional[str] = None  # for aggregates: field name
    where: Optional[WhereExpr] = None
    limit: Optional[int] = None


# ---------- Tokenizer ----------

_TOKEN_RE = re.compile(
    r"(?P<string>'[^']*')"
    r"|(?P<number>-?\d+(?:\.\d+)?)"
    r"|(?P<slot>slot\[\d+\])"
    r"|(?P<keyword>SELECT|WHERE|LIKE|AND|OR|NOT|LIMIT|"
    r"count|min|max|avg|lines|template|text)"
    r"|(?P<op>>=|<=|!=|>|<|=)"
    r"|(?P<paren>[()])"
    r"|(?P<ws>\s+)",
    re.IGNORECASE,
)


@dataclass
class Token:
    kind: str
    value: str


def tokenize(query: str) -> List[Token]:
    tokens: List[Token] = []
    pos = 0
    while pos < len(query):
        m = _TOKEN_RE.match(query, pos)
        if m is None:
            raise SyntaxError(f"CQL: unexpected character at position {pos}: {query[pos:]!r}")
        pos = m.end()
        for kind, value in m.groupdict().items():
            if value is not None and kind != "ws":
                tokens.append(Token(kind=kind, value=value))
                break
    return tokens


# ---------- Parser ----------

class CQLParser:
    def __init__(self, tokens: List[Token]) -> None:
        self.tokens = tokens
        self.pos = 0

    def peek(self) -> Optional[Token]:
        if self.pos < len(self.tokens):
            return self.tokens[self.pos]
        return None

    def advance(self) -> Token:
        tok = self.tokens[self.pos]
        self.pos += 1
        return tok

    def expect_keyword(self, kw: str) -> Token:
        tok = self.advance()
        if tok.kind != "keyword" or tok.value.upper() != kw.upper():
            raise SyntaxError(f"CQL: expected '{kw}', got '{tok.value}'")
        return tok

    def parse(self) -> SelectQuery:
        self.expect_keyword("SELECT")
        target, func_field = self._parse_target()

        where = None
        p = self.peek()
        if p and p.kind == "keyword" and p.value.upper() == "WHERE":
            self.advance()
            where = self._parse_or_expr()

        limit = None
        p = self.peek()
        if p and p.kind == "keyword" and p.value.upper() == "LIMIT":
            self.advance()
            tok = self.advance()
            limit = int(tok.value)

        return SelectQuery(target=target, func_field=func_field, where=where, limit=limit)

    def _parse_target(self):
        tok = self.advance()
        val = tok.value.lower()
        if val == "lines":
            return "lines", None
        if val in ("count", "min", "max", "avg"):
            # Expect ()
            p = self.peek()
            func_field = None
            if p and p.kind == "paren" and p.value == "(":
                self.advance()  # (
                nxt = self.peek()
                if nxt and nxt.kind == "paren" and nxt.value == ")":
                    self.advance()  # )
                else:
                    # field name inside parens
                    field_tok = self.advance()
                    func_field = field_tok.value
                    self.advance()  # )
            return val, func_field
        raise SyntaxError(f"CQL: unexpected target '{val}'")

    def _parse_or_expr(self) -> WhereExpr:
        left = self._parse_and_expr()
        while True:
            p = self.peek()
            if p and p.kind == "keyword" and p.value.upper() == "OR":
                self.advance()
                right = self._parse_and_expr()
                left = OrExpr(left=left, right=right)
            else:
                break
        return left

    def _parse_and_expr(self) -> WhereExpr:
        left = self._parse_primary()
        while True:
            p = self.peek()
            if p and p.kind == "keyword" and p.value.upper() == "AND":
                self.advance()
                right = self._parse_primary()
                left = AndExpr(left=left, right=right)
            else:
                break
        return left

    def _parse_primary(self) -> WhereExpr:
        p = self.peek()

        # Parenthesized expression
        if p and p.kind == "paren" and p.value == "(":
            self.advance()
            expr = self._parse_or_expr()
            self.advance()  # )
            return expr

        # NOT LIKE
        negate = False
        if p and p.kind == "keyword" and p.value.upper() == "NOT":
            self.advance()
            negate = True

        # Field name
        field_tok = self.advance()
        field = field_tok.value

        # LIKE or comparison
        op_tok = self.peek()
        if op_tok and op_tok.kind == "keyword" and op_tok.value.upper() == "LIKE":
            self.advance()
            pattern_tok = self.advance()
            pattern = pattern_tok.value.strip("'")
            return LikeExpr(field=field, pattern=pattern, negate=negate)
        elif op_tok and op_tok.kind == "op":
            self.advance()
            value_tok = self.advance()
            value = value_tok.value.strip("'")
            return CompareExpr(field=field, op=op_tok.value, value=value)
        else:
            raise SyntaxError(f"CQL: expected LIKE or comparison operator after '{field}'")


def parse_cql(query: str) -> SelectQuery:
    """Parse a CQL query string into an AST."""
    tokens = tokenize(query)
    parser = CQLParser(tokens)
    return parser.parse()
