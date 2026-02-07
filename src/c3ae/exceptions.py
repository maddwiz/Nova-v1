"""C3/Ae exceptions."""


class C3AEError(Exception):
    """Base exception."""


class StorageError(C3AEError):
    """Storage layer error."""


class GovernanceError(C3AEError):
    """Write blocked by governance rules."""


class EmbeddingError(C3AEError):
    """Embedding computation failed."""


class RetrievalError(C3AEError):
    """Search/retrieval failed."""


class VaultError(C3AEError):
    """Filesystem vault error."""


class ConfigError(C3AEError):
    """Configuration error."""
