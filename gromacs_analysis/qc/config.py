"""Configuration classes for preflight quality-control checks."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional


@dataclass
class QCConfig:
    """Configuration for scientific preflight checks."""

    base_dir: Path
    output_dir: Path
    topology: Optional[Path] = None
    trajectories: List[Path] = field(default_factory=list)
    index_file: Optional[Path] = None
    expected_groups: List[str] = field(default_factory=list)
    protein_selection: str = "protein"
    ligand_selection: str = "resname UNK"
    expected_replicates: Optional[int] = None
    min_distance_cutoff: Optional[float] = None
    sample_stride: int = 10
    max_frames: Optional[int] = 500
    fail_on_error: bool = False

    def __post_init__(self) -> None:
        self.base_dir = Path(self.base_dir)
        self.output_dir = Path(self.output_dir)
        self.topology = Path(self.topology) if self.topology else None
        self.trajectories = [Path(p) for p in self.trajectories]
        self.index_file = Path(self.index_file) if self.index_file else None

        if self.sample_stride < 1:
            raise ValueError("QCConfig.sample_stride must be >= 1")
