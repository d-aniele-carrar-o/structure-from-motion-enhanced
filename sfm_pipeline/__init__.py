"""
Refactored Structure from Motion Pipeline

A clean, modular implementation of the SFM pipeline with the same functionality
as the original script-based version but with better organization and maintainability.
"""

from .config import Config
from .pipeline import Pipeline
from .media_processor import MediaProcessor
from .feature_matcher import FeatureMatcher
from .reconstructor import Reconstructor
from .logger import setup_logger

__version__ = "1.0.0"
__all__ = [
    "Config",
    "Pipeline", 
    "MediaProcessor",
    "FeatureMatcher",
    "Reconstructor",
    "setup_logger"
]