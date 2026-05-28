"""
PCA Analysis Module
==================

Pattern-based PCA parsing, processing, and visualization.
"""

from .config import (
    PCAConfig,
    PCASystemConfig,
    PCACompareConfig,
    PCAGromacsRunConfig,
    PCATerminalDetectionConfig,
)
from .parser import PCAParser
from .plotter import PCAPlotter
from .analyzer import PCAAnalyzer
from .compare import PCAComparisonAnalyzer
from .runner import PCAGromacsRunner

__all__ = [
    'PCAConfig',
    'PCASystemConfig',
    'PCACompareConfig',
    'PCAGromacsRunConfig',
    'PCATerminalDetectionConfig',
    'PCAParser',
    'PCAPlotter',
    'PCAAnalyzer',
    'PCAComparisonAnalyzer',
    'PCAGromacsRunner',
]
