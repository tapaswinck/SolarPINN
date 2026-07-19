""" 
Global project settings.

This module contains configurable defaults used thorughout the SolarPINN
project. 
"""
from dataclasses import dataclass
from typing import Literal

@dataclass(frozen=True)
class TrainingConfig:
    """ Training-related settings."""

    seed: int = 42
    device: Literal["cpu", "cuda", "mps", "auto"] = "auto"

    num_workers: int = 4

    batch_size: int = 16

    epochs: int = 100

    learning_rate: float = 1e-3

    weight_decay: float = 1e-4

@dataclass(frozen=True)
class DataConfig:
    """ Dataset configuration."""

    image_size: int = 512

    train_split: float = 0.8
    
    validation_split: float = 0.1

    test_split: float = 0.1

    normalize: bool = True

@dataclass(frozen=True)
class LoggingConfig:
    """Logging configuration."""

    level: str = "INFO"

    log_to_file: bool = True

    log_filename: str = "solarpinn.log"

@dataclass(frozen=True)
class ExperimentConfig:
    """Experiment metadata. """

    experiment_name: str = "default"

    save_checkpoints: bool = True

    save_predictions: bool = True

training = TrainingConfig()
data = DataConfig()
logging = LoggingConfig()
experiment = ExperimentConfig()


