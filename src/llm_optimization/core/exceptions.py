"""Custom exceptions for the LLM Optimization Toolkit."""


class LLMOptError(Exception):
    """Base exception for all toolkit errors."""


class ConfigError(LLMOptError):
    """Raised when configuration is invalid or missing."""


class DataError(LLMOptError):
    """Raised when data loading or preprocessing fails."""


class TrainingError(LLMOptError):
    """Raised when training fails."""


class InferenceError(LLMOptError):
    """Raised when inference fails."""
