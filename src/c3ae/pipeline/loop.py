"""Pipeline Loop — the full 5-step dataflow: Load → Reason → Verify → Write → Govern."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from c3ae.exceptions import GovernanceError
from c3ae.memory_spine.spine import MemorySpine
from c3ae.mre.engine import MREEngine, ReasoningSession
from c3ae.types import ReasoningEntry, SearchResult


@dataclass
class PipelineResult:
    """Result of a pipeline run."""
    session: ReasoningSession
    entries_written: list[ReasoningEntry] = field(default_factory=list)
    entries_blocked: list[tuple[ReasoningEntry, str]] = field(default_factory=list)
    search_results: list[SearchResult] = field(default_factory=list)


class PipelineLoop:
    """Orchestrates the full reasoning pipeline.

    Steps:
    1. Load — retrieve relevant context from memory
    2. Reason — produce reasoning output (MRE step)
    3. Verify — governance validation
    4. Write — persist verified results
    5. Govern — audit trail
    """

    def __init__(self, spine: MemorySpine, max_steps: int = 10) -> None:
        self.spine = spine
        self.mre = MREEngine(spine, max_steps=max_steps)
        self.max_steps = max_steps

    async def run(self, task: str,
                  steps: list[dict[str, Any]] | None = None,
                  metadata: dict[str, Any] | None = None) -> PipelineResult:
        """Execute the pipeline.

        If `steps` is provided, runs those specific steps.
        Each step dict has: query, output, new_facts?, resolved_questions?, new_questions?,
                           write_title?, write_content?, write_tags?, evidence_ids?
        """
        session = await self.mre.start_session(task, metadata)
        result = PipelineResult(session=session)

        if steps:
            for step_data in steps:
                await self._execute_step(session, step_data, result)
        else:
            # Single-step reasoning: load + search
            context = await self.spine.search(task, top_k=10)
            result.search_results = context
            step = await self.mre.step(
                session, query=task,
                output=f"Retrieved {len(context)} relevant memory entries.",
                new_facts=[f"Found {len(context)} results for: {task}"],
            )

        await self.mre.finalize(session, session.steps[-1].output if session.steps else "No steps executed")
        return result

    async def _execute_step(self, session: ReasoningSession,
                            step_data: dict[str, Any],
                            result: PipelineResult) -> None:
        """Execute a single pipeline step: Load → Reason → Verify → Write → Govern."""
        query = step_data.get("query", "")
        output = step_data.get("output", "")

        # Step 1+2: Load + Reason (MRE handles both)
        step = await self.mre.step(
            session, query=query, output=output,
            new_facts=step_data.get("new_facts"),
            resolved_questions=step_data.get("resolved_questions"),
            new_questions=step_data.get("new_questions"),
        )
        result.search_results.extend(step.context)

        # Steps 3+4+5: Verify → Write → Govern (if write requested)
        write_title = step_data.get("write_title")
        write_content = step_data.get("write_content")
        if write_title and write_content:
            entry = ReasoningEntry(
                title=write_title,
                content=write_content,
                tags=step_data.get("write_tags", []),
                evidence_ids=step_data.get("evidence_ids", []),
                session_id=session.session_id,
            )
            try:
                await self.spine.add_knowledge(
                    title=entry.title,
                    content=entry.content,
                    tags=entry.tags,
                    evidence_ids=entry.evidence_ids,
                    session_id=session.session_id,
                )
                result.entries_written.append(entry)
            except GovernanceError as e:
                result.entries_blocked.append((entry, str(e)))
