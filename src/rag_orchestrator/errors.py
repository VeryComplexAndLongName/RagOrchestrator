class RAGError(Exception):
    """Base error for rag_orchestrator."""


class ConfigurationError(RAGError):
    """Raised when provider or pipeline is configured incorrectly."""


class ProviderDependencyError(RAGError):
    """Raised when optional provider dependencies are missing."""
