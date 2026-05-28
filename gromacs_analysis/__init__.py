"""
GROMACS Analysis Toolkit
=========================

A unified toolkit for GROMACS molecular dynamics analysis and visualization.

Modules:
--------
- md: MD trajectory analysis (RMSD, RMSF, Rg, SASA, H-bonds, COM-COM)
- config: YAML configuration loading
- utils: Utility functions for data parsing and file handling
- cli: Command-line interface and interactive prompts

"""

__version__ = '1.0.0'
__author__ = 'GROMACS Analysis Team'

from .md import (
    MDAnalyzer,
    MDConfig,
    SystemConfig,
    AnalysisMetric,
    PlotConfig,
    MDDataProcessor,
    MDPlotter
)

from .mmpbsa import (
    MMPBSAAnalyzer,
    MMPBSAConfig,
    MMPBSASystemConfig,
    MMPBSAParser,
    MMPBSAProcessor,
    MMPBSAPlotter,
)

from .pca import (
    PCAAnalyzer,
    PCAConfig,
    PCASystemConfig,
    PCAGromacsRunConfig,
    PCATerminalDetectionConfig,
    PCAParser,
    PCAPlotter,
    PCAGromacsRunner,
)

from .distance import (
    DistanceAnalyzer,
    DistanceConfig,
    DistanceProcessor,
    DistancePlotter,
)

from .prolif import (
    ProlifAnalyzer,
    ProlifComparisonAnalyzer,
    ProlifConfig,
    ProlifCompareConfig,
    ProlifProcessor,
)
from .qc import QCAnalyzer, QCConfig
from .ttclust import TTClustAnalyzer, TTClustConfig

from .config import load_yaml_config, generate_yaml_template
from .cli import interactive_prompt
from .pipeline import (
    PipelineConfig,
    PipelineStage,
    load_pipeline_config,
    save_pipeline_config,
    generate_pipeline_template,
    run_pipeline,
)

__all__ = [
    # MD Analysis
    'MDAnalyzer',
    'MDConfig',
    'SystemConfig',
    'AnalysisMetric',
    'PlotConfig',
    'MDDataProcessor',
    'MDPlotter',
    # MMPBSA Analysis
    'MMPBSAAnalyzer',
    'MMPBSAConfig',
    'MMPBSASystemConfig',
    'MMPBSAParser',
    'MMPBSAProcessor',
    'MMPBSAPlotter',
    # PCA Analysis
    'PCAAnalyzer',
    'PCAConfig',
    'PCASystemConfig',
    'PCAGromacsRunConfig',
    'PCATerminalDetectionConfig',
    'PCAParser',
    'PCAPlotter',
    'PCAGromacsRunner',
    # Distance Analysis
    'DistanceAnalyzer',
    'DistanceConfig',
    'DistanceProcessor',
    'DistancePlotter',
    # ProLIF Analysis
    'ProlifAnalyzer',
    'ProlifComparisonAnalyzer',
    'ProlifConfig',
    'ProlifCompareConfig',
    'ProlifProcessor',
    # QC Analysis
    'QCAnalyzer',
    'QCConfig',
    # TTClust
    'TTClustAnalyzer',
    'TTClustConfig',
    # Utilities
    'load_yaml_config',
    'generate_yaml_template',
    'interactive_prompt',
    # Pipeline
    'PipelineConfig',
    'PipelineStage',
    'load_pipeline_config',
    'save_pipeline_config',
    'generate_pipeline_template',
    'run_pipeline',
]
