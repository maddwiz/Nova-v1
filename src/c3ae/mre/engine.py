"""MRE — Multi-step Reasoning Engine with Carry-Over State."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from uuid import uuid4

from c3ae.cos.cos import COSManager
from c3ae.memory_spine.spine import MemorySpine
from c3ae.types import SearchResult


@dataclass
class ReasoningStep:
    """One step in a multi-step reasoning chain."""
    step_number: int
    query: str
    context: list[SearchResult] = field(default_factory=list)
    output: str = ""
    new_facts: list[str] = field(default_factory=list)
    resolved_questions: list[str] = field(default_factory=list)
    new_questions: list[str] = field(default_factory=list)


@dataclass
class ReasoningSession:
    """A complete reasoning session."""
    session_id: str
    task: str
    steps: list[ReasoningStep] = field(default_factory=list)
    final_answer: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


class MREEngine:
    """Chunked reasoning loop with COS updates at each step.

    The MRE processes a task in steps:
    1. Retrieve relevant context from memory
    2. Produce a reasoning output for that step
    3. Update the Carry-Over Summary
    4. Repeat until done or budget exhausted
    """

    def __init__(self, spine: MemorySpine, max_steps: int = 10) -> None:
        self.spine = spine
        self.cos = spine.cos
        self.max_steps = max_steps

    async def start_session(self, task: str, metadata: dict[str, Any] | None = None) -> ReasoningSession:
        """Initialize a reasoning session."""
        session_id = uuid4().hex
        self.spine.start_session(session_id, metadata)
        self.cos.create(session_id, f"Task: {task}", open_questions=[task])
        return ReasoningSession(session_id=session_id, task=task, metadata=metadata or {})

    async def step(self, session: ReasoningSession, query: str,
                   output: str,
                   new_facts: list[str] | None = None,
                   resolved_questions: list[str] | None = None,
                   new_questions: list[str] | None = None) -> ReasoningStep:
        """Execute one reasoning step.

        In the full pipeline, the caller (LLM or pipeline loop) provides:
        - query: what to search for context
        - output: the reasoning result for this step
        - new_facts/resolved_questions/new_questions: COS updates
        """
        step_num = len(session.steps)

        # Retrieve context
        context = await self.spine.search(query, top_k=10)

        # Create step record
        step = ReasoningStep(
            step_number=step_num,
            query=query,
            context=context,
            output=output,
            new_facts=new_facts or [],
            resolved_questions=resolved_questions or [],
            new_questions=new_questions or [],
        )
        session.steps.append(step)

        # Update COS
        cos_summary = f"Step {step_num}: {output[:200]}"
        self.cos.update(
            session.session_id,
            cos_summary,
            new_facts=new_facts,
            resolved_questions=resolved_questions,
            new_questions=new_questions,
        )

        return step

    async def finalize(self, session: ReasoningSession, final_answer: str) -> ReasoningSession:
        """Finalize the session with a final answer."""
        session.final_answer = final_answer
        self.spine.end_session(session.session_id)

        # Store the final reasoning log
        self.spine.vault.store_raw_log(
            self._render_session_log(session),
            session.session_id,
            "reasoning_log.md",
        )
        return session

    def get_cos_prompt(self, session: ReasoningSession) -> str:
        """Get the current COS prompt for LLM context injection."""
        return self.cos.render_prompt(session.session_id)

    def _render_session_log(self, session: ReasoningSession) -> str:
        parts = [f"# Reasoning Session: {session.task}\n"]
        for step in session.steps:
            parts.append(f"## Step {step.step_number}")
            parts.append(f"**Query:** {step.query}")
            parts.append(f"**Context hits:** {len(step.context)}")
            parts.append(f"**Output:** {step.output}")
            if step.new_facts:
                parts.append(f"**New facts:** {', '.join(step.new_facts)}")
            parts.append("")
        if session.final_answer:
            parts.append(f"## Final Answer\n{session.final_answer}")
        return "\n".join(parts)
