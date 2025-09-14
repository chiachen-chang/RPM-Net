"""
RPM-Net: Reciprocal Point MLP Network for Open Set Recognition

A novel framework for open set recognition in network security threat detection
that introduces reciprocal point mechanism to learn "non-class" representations.
"""

from .model import RPMModel
from .losses import compute_rpm_loss, compute_fisher_loss_optimized
from .data_utils import load_data, prepare_data_loaders
from .evaluation import evaluate_known_classification, evaluate_unknown_detection
from .config import Config

__version__ = "1.0.0"
__author__ = "Your Name"
__email__ = "your.email@example.com"

__all__ = [
    "RPMModel",
    "compute_rpm_loss", 
    "compute_fisher_loss_optimized",
    "load_data",
    "prepare_data_loaders",
    "evaluate_known_classification",
    "evaluate_unknown_detection",
    "Config"
]
