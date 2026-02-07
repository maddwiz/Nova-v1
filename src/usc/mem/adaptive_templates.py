from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from drain3 import TemplateMiner
from drain3.template_miner_config import TemplateMinerConfig

from usc.mem.template_miner_drain3 import Drain3Mined, mine_message


@dataclass
class TemplateStats:
    """Statistics about template mining quality."""
    template_count: int
    total_lines: int
    total_slots: int
    coverage_pct: float       # % of lines that matched a template
    avg_slots_per_template: float
    reuse_rate: float         # avg times each template is used


@dataclass
class AdaptiveTemplateEngine:
    """
    Wraps Drain3 with adaptive threshold tuning and template merging.

    The similarity threshold controls how aggressively Drain3 merges log lines
    into shared templates. Lower = more merging (fewer templates, more slots).
    Higher = less merging (more templates, fewer slots).
    """
    similarity_threshold: float = 0.4
    _miner: Optional[TemplateMiner] = field(default=None, repr=False)
    _template_usage: Dict[str, int] = field(default_factory=dict, repr=False)
    _total_lines: int = field(default=0, repr=False)
    _total_slots: int = field(default=0, repr=False)

    def _ensure_miner(self) -> TemplateMiner:
        if self._miner is None:
            cfg = TemplateMinerConfig()
            cfg.profiling_enabled = False
            cfg.drain_sim_th = self.similarity_threshold
            self._miner = TemplateMiner(config=cfg)
        return self._miner

    def reset(self) -> None:
        """Reset internal state for a fresh mining run."""
        self._miner = None
        self._template_usage.clear()
        self._total_lines = 0
        self._total_slots = 0

    def mine(self, message: str) -> Drain3Mined:
        """Mine a single message, tracking usage stats."""
        miner = self._ensure_miner()
        result = mine_message(miner, message)
        self._template_usage[result.template] = (
            self._template_usage.get(result.template, 0) + 1
        )
        self._total_lines += 1
        self._total_slots += len(result.params)
        return result

    def mine_chunk_lines(
        self, chunks: List[str]
    ) -> Tuple[List[str], List[List[str]]]:
        """
        Adaptive version of mine_chunk_lines — same interface as the original
        but uses adaptive threshold and tracks statistics.
        """
        chunk_templates: List[str] = []
        chunk_params: List[List[str]] = []

        for ch in chunks:
            ends_with_newline = ch.endswith("\n")
            lines = ch.splitlines(keepends=False)

            line_templates: List[str] = []
            all_params: List[str] = []

            for line in lines:
                mined = self.mine(line)
                line_templates.append(mined.template)
                all_params.extend(mined.params)

            rebuilt = "\n".join(line_templates)
            if ends_with_newline:
                rebuilt += "\n"

            chunk_templates.append(rebuilt)
            chunk_params.append(all_params)

        return chunk_templates, chunk_params

    def template_stats(self) -> TemplateStats:
        """Return current mining statistics."""
        n_templates = len(self._template_usage)
        if n_templates == 0:
            return TemplateStats(
                template_count=0,
                total_lines=0,
                total_slots=0,
                coverage_pct=0.0,
                avg_slots_per_template=0.0,
                reuse_rate=0.0,
            )
        coverage = 100.0  # Drain3 always assigns a template
        avg_slots = self._total_slots / n_templates if n_templates else 0.0
        reuse = self._total_lines / n_templates if n_templates else 0.0
        return TemplateStats(
            template_count=n_templates,
            total_lines=self._total_lines,
            total_slots=self._total_slots,
            coverage_pct=coverage,
            avg_slots_per_template=avg_slots,
            reuse_rate=reuse,
        )

    @staticmethod
    def auto_tune(
        samples: List[str],
        thresholds: Optional[List[float]] = None,
    ) -> float:
        """
        Find the similarity threshold that minimizes template count while
        keeping slot count reasonable.

        Score = template_count + 0.1 * total_slots
        Lower score = better (fewer templates, manageable slots).
        """
        if thresholds is None:
            thresholds = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]

        best_threshold = 0.4
        best_score = float("inf")

        for th in thresholds:
            engine = AdaptiveTemplateEngine(similarity_threshold=th)
            for s in samples:
                engine.mine(s)
            stats = engine.template_stats()
            score = stats.template_count + 0.1 * stats.total_slots
            if score < best_score:
                best_score = score
                best_threshold = th

        return best_threshold

    @staticmethod
    def merge_templates(t1: str, t2: str) -> str:
        """
        Merge two similar templates by replacing differing tokens with <*>.

        Example:
            t1 = "User <*> logged in from <*>"
            t2 = "User <*> logged out from <*>"
            result = "User <*> logged <*> from <*>"
        """
        tokens1 = t1.split()
        tokens2 = t2.split()

        if len(tokens1) != len(tokens2):
            return t1  # can't merge different-length templates

        merged = []
        for a, b in zip(tokens1, tokens2):
            if a == b:
                merged.append(a)
            else:
                merged.append("<*>")
        return " ".join(merged)
