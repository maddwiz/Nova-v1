"""
N5: OpenTelemetry Span Adapter.

Converts OTEL spans to structured log lines for USC encoding.
Only importable if opentelemetry-api is installed.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import zstandard as zstd


@dataclass
class SpanRecord:
    """Simplified span record for USC storage."""
    name: str
    trace_id: str
    span_id: str
    parent_id: str = ""
    start_time: str = ""
    end_time: str = ""
    status: str = "OK"
    attributes: Dict[str, str] = None

    def __post_init__(self):
        if self.attributes is None:
            self.attributes = {}


class OTelSpanAdapter:
    """Adapter for OpenTelemetry spans."""

    name: str = "otel"

    def ingest(self, spans: List[SpanRecord]) -> bytes:
        """Convert span records to USC packet."""
        lines = []
        for span in spans:
            attrs = " ".join(f"{k}={v}" for k, v in span.attributes.items())
            line = (
                f"SPAN {span.name} trace={span.trace_id} "
                f"span={span.span_id} parent={span.parent_id} "
                f"start={span.start_time} end={span.end_time} "
                f"status={span.status} {attrs}"
            )
            lines.append(line.strip())
        text = "\n".join(lines)
        cctx = zstd.ZstdCompressor(level=10)
        return cctx.compress(text.encode("utf-8"))

    def retrieve(self, blob: bytes) -> List[SpanRecord]:
        """Decode USC blob back to span records."""
        dctx = zstd.ZstdDecompressor()
        text = dctx.decompress(blob).decode("utf-8")
        spans = []
        for line in text.splitlines():
            line = line.strip()
            if not line or not line.startswith("SPAN "):
                continue
            span = self._parse_span_line(line)
            if span:
                spans.append(span)
        return spans

    def _parse_span_line(self, line: str) -> Optional[SpanRecord]:
        """Parse a single span line back to SpanRecord."""
        parts = line.split()
        if len(parts) < 2:
            return None

        name = parts[1]
        attrs = {}
        trace_id = span_id = parent_id = start_time = end_time = status = ""

        for part in parts[2:]:
            if "=" in part:
                k, _, v = part.partition("=")
                if k == "trace":
                    trace_id = v
                elif k == "span":
                    span_id = v
                elif k == "parent":
                    parent_id = v
                elif k == "start":
                    start_time = v
                elif k == "end":
                    end_time = v
                elif k == "status":
                    status = v
                else:
                    attrs[k] = v

        return SpanRecord(
            name=name,
            trace_id=trace_id,
            span_id=span_id,
            parent_id=parent_id,
            start_time=start_time,
            end_time=end_time,
            status=status or "OK",
            attributes=attrs,
        )

    def ingest_otel_spans(self, spans: Any) -> bytes:
        """
        Ingest OpenTelemetry ReadableSpan objects directly.
        Import at function level to avoid hard dependency.
        """
        from opentelemetry.sdk.trace import ReadableSpan

        records = []
        for span in spans:
            attrs = {}
            if span.attributes:
                attrs = {str(k): str(v) for k, v in span.attributes.items()}
            records.append(SpanRecord(
                name=span.name,
                trace_id=format(span.context.trace_id, "032x") if span.context else "",
                span_id=format(span.context.span_id, "016x") if span.context else "",
                parent_id=format(span.parent.span_id, "016x") if span.parent else "",
                status=span.status.status_code.name if span.status else "OK",
                attributes=attrs,
            ))
        return self.ingest(records)
