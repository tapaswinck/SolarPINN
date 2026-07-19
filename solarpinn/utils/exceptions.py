"""
Custom exceptions used throughout SolarPINN.
"""

class SolarPINNError(Exception):
    """Base exception for SolarPINN."""

class DataError(SolarPINNError):
    """Raised when dataset operation fails."""

class PhysicsError(SolarPINNError):
    """Raised when a physics computation fails."""

class ConfigurationError(SolarPINNError):
    """Raised when configuration is invalid."""

class ModelError(SolarPINNError):
    """Raised for model-related errors."""

    