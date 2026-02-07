from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

from drain3 import TemplateMiner
from drain3.template_miner_config import TemplateMinerConfig


@dataclass
class Drain3Mined:
    template: str          # e.g. "User <*> logged in from <*>"
    params: List[str]      # e.g. ["Bob", "10.0.0.1"]


def make_miner() -> TemplateMiner:
    cfg = TemplateMinerConfig()
    cfg.profiling_enabled = False
    return TemplateMiner(config=cfg)


def _extract_params_from_template(message: str, template_mined: str) -> List[str]:
    """
    Drain3 template uses "<*>" wildcards.
    Extract actual wildcard substrings from the original message by matching literals.
    """
    parts = template_mined.split("<*>")
    if len(parts) == 1:
        return []

    params: List[str] = []
    pos = 0

    for i in range(len(parts) - 1):
        lit_a = parts[i]
        lit_b = parts[i + 1]

        if lit_a:
            idx_a = message.find(lit_a, pos)
            if idx_a == -1:
                return []
            pos = idx_a + len(lit_a)

        if lit_b:
            idx_b = message.find(lit_b, pos)
            if idx_b == -1:
                params.append(message[pos:])
                pos = len(message)
            else:
                params.append(message[pos:idx_b])
                pos = idx_b
        else:
            params.append(message[pos:])
            pos = len(message)

    return params


def mine_message(miner: TemplateMiner, message: str) -> Drain3Mined:
    res = miner.add_log_message(message)
    tmpl = res["template_mined"]
    params = _extract_params_from_template(message, tmpl)
    return Drain3Mined(template=tmpl, params=params)


def mine_chunk_lines(chunks: List[str]) -> Tuple[List[str], List[List[str]]]:
    """
    Treat each LINE as a log message for Drain3.
    Then rebuild per-chunk templates by joining line templates with "\\n".
    IMPORTANT: preserve trailing newline if the original chunk ended with "\\n".
    """
    miner = make_miner()

    chunk_templates: List[str] = []
    chunk_params: List[List[str]] = []

    for ch in chunks:
        ends_with_newline = ch.endswith("\n")

        # splitlines() removes the trailing newline, so we must preserve it manually
        lines = ch.splitlines(keepends=False)

        line_templates: List[str] = []
        all_params: List[str] = []

        for line in lines:
            mined = mine_message(miner, line)
            line_templates.append(mined.template)
            all_params.extend(mined.params)

        rebuilt = "\n".join(line_templates)
        if ends_with_newline:
            rebuilt += "\n"

        chunk_templates.append(rebuilt)
        chunk_params.append(all_params)

    return chunk_templates, chunk_params
