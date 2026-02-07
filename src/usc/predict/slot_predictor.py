"""
G6: Learned Slot Prediction.

Frequency-based slot value predictor — learns common patterns from training data.
When prediction matches actual value, encodes as 1 bit instead of full value.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from usc.mem.varint import encode_uvarint, decode_uvarint


@dataclass
class SlotModel:
    """Trained prediction model for slot values."""
    # slot_index -> (value -> frequency) map
    slot_freqs: Dict[int, Counter] = field(default_factory=dict)
    # slot_index -> most frequent value (cached)
    _top_values: Dict[int, str] = field(default_factory=dict)

    def top_value(self, slot_index: int) -> Optional[str]:
        """Return the most frequent value for a slot, or None if unknown."""
        if slot_index in self._top_values:
            return self._top_values[slot_index]
        if slot_index not in self.slot_freqs:
            return None
        counter = self.slot_freqs[slot_index]
        if not counter:
            return None
        top = counter.most_common(1)[0][0]
        self._top_values[slot_index] = top
        return top

    def prediction_accuracy(self, slot_index: int) -> float:
        """What fraction of values match the top prediction for this slot?"""
        if slot_index not in self.slot_freqs:
            return 0.0
        counter = self.slot_freqs[slot_index]
        total = sum(counter.values())
        if total == 0:
            return 0.0
        top_count = counter.most_common(1)[0][1]
        return top_count / total


class SlotPredictor:
    """Learns and predicts slot values for compression."""

    def __init__(self) -> None:
        self._model: Optional[SlotModel] = None

    def train(self, slot_values: List[List[str]]) -> SlotModel:
        """
        Train from a list of slot value rows.
        Each row is a list of slot values (one per slot position).
        """
        model = SlotModel()
        for row in slot_values:
            for idx, val in enumerate(row):
                if idx not in model.slot_freqs:
                    model.slot_freqs[idx] = Counter()
                model.slot_freqs[idx][val] += 1
        self._model = model
        return model

    def predict(self, slot_index: int) -> Optional[str]:
        """Predict the most likely value for a slot position."""
        if self._model is None:
            return None
        return self._model.top_value(slot_index)

    def encode_with_prediction(
        self, actual_values: List[str], model: Optional[SlotModel] = None,
    ) -> bytes:
        """
        Encode slot values using prediction.

        Wire format per slot:
            flag_byte (1B): 0x01 if prediction correct, 0x00 if not
            [if 0x00]: value_len (uvarint) + value_bytes

        Returns the encoded bytes.
        """
        m = model or self._model
        out = bytearray()
        out += encode_uvarint(len(actual_values))
        for idx, actual in enumerate(actual_values):
            predicted = m.top_value(idx) if m else None
            if predicted is not None and predicted == actual:
                out += b"\x01"  # prediction correct
            else:
                out += b"\x00"  # prediction miss
                val_bytes = actual.encode("utf-8")
                out += encode_uvarint(len(val_bytes))
                out += val_bytes
        return bytes(out)

    def decode_with_prediction(
        self, data: bytes, model: Optional[SlotModel] = None, offset: int = 0,
    ) -> Tuple[List[str], int]:
        """Decode slot values, using model to fill in predicted values."""
        m = model or self._model
        n_slots, off = decode_uvarint(data, offset)
        values: List[str] = []
        for idx in range(n_slots):
            flag = data[off]
            off += 1
            if flag == 0x01:
                # Prediction was correct — use model's top value
                val = m.top_value(idx) if m else ""
                values.append(val or "")
            else:
                # Full value encoded
                val_len, off = decode_uvarint(data, off)
                val_bytes = data[off:off + val_len]
                off += val_len
                values.append(val_bytes.decode("utf-8"))
        return values, off

    def compression_ratio(
        self, slot_values: List[List[str]], model: Optional[SlotModel] = None,
    ) -> float:
        """Estimate compression ratio: encoded_size / raw_size."""
        m = model or self._model
        raw_size = 0
        encoded_size = 0
        for row in slot_values:
            raw_bytes = b"".join(v.encode("utf-8") for v in row)
            raw_size += len(raw_bytes)
            encoded = self.encode_with_prediction(row, m)
            encoded_size += len(encoded)
        if raw_size == 0:
            return 1.0
        return encoded_size / raw_size
