"""Venice AI chat completion client."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import httpx

from c3ae.config import VeniceConfig


@dataclass
class Message:
    role: str  # "system", "user", "assistant"
    content: str


@dataclass
class ChatResponse:
    content: str
    model: str = ""
    usage: dict[str, int] = field(default_factory=dict)
    finish_reason: str = ""
    raw: dict[str, Any] = field(default_factory=dict)


class VeniceChat:
    """Async chat completion client for Venice AI."""

    def __init__(
        self,
        api_key: str,
        model: str = "qwen3-235b-a22b-instruct-2507",
        base_url: str = "https://api.venice.ai/api/v1",
        timeout: float = 120.0,
        temperature: float = 0.3,
        max_tokens: int = 4096,
    ) -> None:
        self.api_key = api_key
        self.model = model
        self.base_url = base_url
        self.timeout = timeout
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._client: httpx.AsyncClient | None = None
        self._total_input_tokens = 0
        self._total_output_tokens = 0
        self._call_count = 0

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                headers={"Authorization": f"Bearer {self.api_key}"},
                timeout=self.timeout,
            )
        return self._client

    async def chat(
        self,
        messages: list[Message],
        temperature: float | None = None,
        max_tokens: int | None = None,
        json_mode: bool = False,
    ) -> ChatResponse:
        """Send a chat completion request."""
        client = await self._get_client()

        body: dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "temperature": temperature if temperature is not None else self.temperature,
            "max_tokens": max_tokens or self.max_tokens,
        }
        if json_mode:
            body["response_format"] = {"type": "json_object"}

        resp = await client.post("/chat/completions", json=body)
        resp.raise_for_status()
        data = resp.json()

        choice = data["choices"][0]
        usage = data.get("usage", {})

        self._call_count += 1
        self._total_input_tokens += usage.get("prompt_tokens", 0)
        self._total_output_tokens += usage.get("completion_tokens", 0)

        content = choice["message"]["content"]
        # Strip <think>...</think> blocks from reasoning models
        if "<think>" in content:
            import re
            content = re.sub(r"<think>.*?</think>\s*", "", content, flags=re.DOTALL).strip()

        return ChatResponse(
            content=content,
            model=data.get("model", self.model),
            usage=usage,
            finish_reason=choice.get("finish_reason", ""),
            raw=data,
        )

    @property
    def stats(self) -> dict[str, Any]:
        return {
            "calls": self._call_count,
            "input_tokens": self._total_input_tokens,
            "output_tokens": self._total_output_tokens,
            "total_tokens": self._total_input_tokens + self._total_output_tokens,
        }

    async def close(self) -> None:
        if self._client and not self._client.is_closed:
            await self._client.aclose()
            self._client = None
