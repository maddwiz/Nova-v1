from __future__ import annotations

from typing import List, Protocol, runtime_checkable


@runtime_checkable
class Tokenizer(Protocol):
    """Pluggable tokenizer interface."""

    name: str

    def encode(self, text: str) -> List[int]: ...
    def decode(self, tokens: List[int]) -> str: ...


class WhitespaceTokenizer:
    """Simple whitespace-based tokenizer — no external deps."""

    name: str = "whitespace"

    def __init__(self) -> None:
        self._vocab: dict[str, int] = {}
        self._reverse: dict[int, str] = {}
        self._next_id: int = 0

    def encode(self, text: str) -> List[int]:
        tokens: List[int] = []
        for word in text.split(" "):
            if word not in self._vocab:
                self._vocab[word] = self._next_id
                self._reverse[self._next_id] = word
                self._next_id += 1
            tokens.append(self._vocab[word])
        return tokens

    def decode(self, tokens: List[int]) -> str:
        return " ".join(self._reverse[t] for t in tokens)


def get_tiktoken_adapter(encoding_name: str = "cl100k_base") -> Tokenizer:
    """Optional tiktoken adapter — raises ImportError if tiktoken not installed."""
    import tiktoken

    class TiktokenAdapter:
        name: str = f"tiktoken-{encoding_name}"

        def __init__(self) -> None:
            self._enc = tiktoken.get_encoding(encoding_name)

        def encode(self, text: str) -> List[int]:
            return self._enc.encode(text)

        def decode(self, tokens: List[int]) -> str:
            return self._enc.decode(tokens)

    return TiktokenAdapter()
