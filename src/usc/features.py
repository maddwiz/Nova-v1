from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class FeatureFlags:
    """Runtime feature flags for optional USC capabilities."""

    merkle: bool = False          # G2
    adaptive_templates: bool = False  # G1
    token_mode: bool = False      # G3
    fleet: bool = False           # G4
    semantic_blooms: bool = False  # G5
    slot_prediction: bool = False  # G6
    gist: bool = False            # G7
    cql: bool = False             # N1
    forensic: bool = False        # N2
    pii_crypto: bool = False      # N3
    zerocopy: bool = False        # N4
    adapters: bool = False        # N5
    diff_storage: bool = False    # N6
    timetravel: bool = False      # N7
    auto_summaries: bool = False  # N8
    dynamic_tiering: bool = False  # N9
    cold_reindex: bool = False    # N10
    budgeted_decode: bool = False  # N11
    kvcache: bool = False         # N12


# Global singleton — modules check this at import/call time
flags = FeatureFlags()


def enable(flag_name: str) -> None:
    """Enable a feature flag by name."""
    if not hasattr(flags, flag_name):
        raise ValueError(f"Unknown feature flag: {flag_name}")
    setattr(flags, flag_name, True)


def require(flag_name: str) -> None:
    """Raise if a feature flag is not enabled."""
    if not hasattr(flags, flag_name):
        raise ValueError(f"Unknown feature flag: {flag_name}")
    if not getattr(flags, flag_name):
        raise RuntimeError(
            f"Feature '{flag_name}' is not enabled. "
            f"Call usc.features.enable('{flag_name}') first."
        )


def enable_all() -> None:
    """Enable every feature flag."""
    for name in FeatureFlags.__dataclass_fields__:
        setattr(flags, name, True)
