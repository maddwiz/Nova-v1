"""Compression as Anomaly Detection — dedup ratio signals drift.

Upgrade #8: The dedup ratio is a natural novelty signal. When it drops
sharply (new data looks nothing like stored memories), that's an anomaly.
When it stays high, the agent is in familiar territory.

This module tracks dedup ratio as a time series and detects anomalies
using a simple z-score approach. It can wire into governance/monitoring.

Usage:
    detector = AnomalyDetector(window_size=50)

    for session in sessions:
        blob, stats = cogdedup_encode(data, store)
        ratio = len(data) / len(blob)
        alert = detector.observe(ratio, label=f"session-{i}")
        if alert:
            print(f"ANOMALY: {alert}")
"""
from __future__ import annotations

import math
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional


@dataclass
class AnomalyAlert:
    """An anomaly detected in the dedup ratio time series."""
    timestamp: float
    label: str
    ratio: float
    z_score: float
    mean: float
    std: float
    severity: str  # "low", "medium", "high"

    def __str__(self) -> str:
        return (
            f"[{self.severity.upper()}] ratio={self.ratio:.1f}x "
            f"(z={self.z_score:+.2f}, mean={self.mean:.1f}x, std={self.std:.2f}) "
            f"@ {self.label}"
        )


@dataclass
class DriftReport:
    """Summary of drift over a time window."""
    window_size: int
    current_mean: float
    current_std: float
    trend: float         # positive = improving compression, negative = degrading
    alerts_count: int
    is_drifting: bool


class AnomalyDetector:
    """Detect anomalies in compression ratio time series.

    Uses a sliding window z-score: if the current ratio deviates by
    more than `z_threshold` standard deviations from the window mean,
    it's flagged as an anomaly.

    Low ratio = novelty (data unlike anything seen before)
    High ratio = extreme dedup (suspicious duplication or system loop)

    Args:
        window_size: Number of recent observations to maintain
        z_threshold_low: Z-score below which to flag low-ratio anomaly
        z_threshold_high: Z-score above which to flag high-ratio anomaly
    """

    def __init__(
        self,
        window_size: int = 50,
        z_threshold_low: float = -2.0,
        z_threshold_high: float = 3.0,
    ) -> None:
        self._window_size = window_size
        self._z_low = z_threshold_low
        self._z_high = z_threshold_high
        self._history: Deque[float] = deque(maxlen=window_size)
        self._alerts: List[AnomalyAlert] = []
        self._observation_count: int = 0

    def observe(self, ratio: float, label: str = "") -> Optional[AnomalyAlert]:
        """Record an observation and check for anomaly.

        Args:
            ratio: Compression ratio (original_size / compressed_size)
            label: Human-readable label for this observation

        Returns:
            AnomalyAlert if anomaly detected, None otherwise.
        """
        self._observation_count += 1

        # Need enough history for meaningful stats
        if len(self._history) < 5:
            self._history.append(ratio)
            return None

        mean = sum(self._history) / len(self._history)
        variance = sum((x - mean) ** 2 for x in self._history) / len(self._history)
        std = math.sqrt(variance) if variance > 0 else 0.001

        z_score = (ratio - mean) / std

        alert = None
        if z_score < self._z_low:
            severity = "high" if z_score < self._z_low * 1.5 else "medium"
            alert = AnomalyAlert(
                timestamp=time.time(),
                label=label,
                ratio=ratio,
                z_score=z_score,
                mean=mean,
                std=std,
                severity=severity,
            )
            self._alerts.append(alert)
        elif z_score > self._z_high:
            severity = "medium" if z_score > self._z_high * 1.5 else "low"
            alert = AnomalyAlert(
                timestamp=time.time(),
                label=label,
                ratio=ratio,
                z_score=z_score,
                mean=mean,
                std=std,
                severity=severity,
            )
            self._alerts.append(alert)

        self._history.append(ratio)
        return alert

    def drift_report(self) -> DriftReport:
        """Generate a summary of current drift state."""
        if len(self._history) < 2:
            return DriftReport(
                window_size=len(self._history),
                current_mean=sum(self._history) / max(1, len(self._history)),
                current_std=0.0,
                trend=0.0,
                alerts_count=len(self._alerts),
                is_drifting=False,
            )

        mean = sum(self._history) / len(self._history)
        variance = sum((x - mean) ** 2 for x in self._history) / len(self._history)
        std = math.sqrt(variance) if variance > 0 else 0.0

        # Trend: compare first half vs second half means
        half = len(self._history) // 2
        hist_list = list(self._history)
        first_half = sum(hist_list[:half]) / max(1, half)
        second_half = sum(hist_list[half:]) / max(1, len(hist_list) - half)
        trend = second_half - first_half

        # Drifting if trend is significant relative to std
        is_drifting = abs(trend) > std if std > 0 else abs(trend) > 0.5

        return DriftReport(
            window_size=len(self._history),
            current_mean=round(mean, 2),
            current_std=round(std, 2),
            trend=round(trend, 2),
            alerts_count=len(self._alerts),
            is_drifting=is_drifting,
        )

    @property
    def alerts(self) -> List[AnomalyAlert]:
        return list(self._alerts)

    @property
    def observation_count(self) -> int:
        return self._observation_count

    def reset(self) -> None:
        self._history.clear()
        self._alerts.clear()
        self._observation_count = 0
