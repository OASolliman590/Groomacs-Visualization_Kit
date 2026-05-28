"""
YAML loader for distance analysis config.
"""

from pathlib import Path
from typing import Any, Dict

import yaml

from .config import DistanceConfig
from ..md.config import PlotConfig
from ..config.yaml_loader import parse_amino_acids_config


def _resolve_path(value: str, base_path: Path) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = (base_path / path).resolve()
    return path


def load_yaml_config(yaml_path: Path) -> DistanceConfig:
    with open(yaml_path, "r") as f:
        data: Dict[str, Any] = yaml.safe_load(f) or {}

    base_path = yaml_path.parent

    base_dir = _resolve_path(data["base_dir"], base_path)
    output_dir = _resolve_path(data["output_dir"], base_path)

    topology = _resolve_path(data["topology"], base_path)
    trajectories = [
        _resolve_path(p, base_path)
        for p in data.get("trajectories", [])
    ]

    residues = data.get("residues")
    residue_ids = None
    residue_selection = None
    auto_detect_residues = bool(data.get("auto_detect_residues", False))
    auto_detect_cutoff = data.get("auto_detect_cutoff", 5.0)
    auto_detect_stride = data.get("auto_detect_stride", 1)
    auto_detect_max_frames = data.get("auto_detect_max_frames")
    auto_detect_selection = data.get("auto_detect_selection", "protein")
    auto_detect_method = data.get("auto_detect_method", "min")
    if isinstance(residues, dict):
        auto_flag = residues.get("auto")
        if auto_flag is None:
            auto_flag = residues.get("auto_detect")
        if auto_flag is not None:
            auto_detect_residues = bool(auto_flag)
            auto_detect_cutoff = residues.get("cutoff", auto_detect_cutoff)
            auto_detect_stride = residues.get("stride", auto_detect_stride)
            auto_detect_max_frames = residues.get("max_frames", auto_detect_max_frames)
            auto_detect_selection = residues.get("selection", auto_detect_selection)
            auto_detect_method = residues.get("method", auto_detect_method)
        if not auto_detect_residues:
            if "selection" in residues:
                residue_selection = residues["selection"]
            else:
                residue_ids = parse_amino_acids_config(residues)
    elif isinstance(residues, list):
        residue_ids = []
        for item in residues:
            if isinstance(item, int):
                residue_ids.append(item)
            elif isinstance(item, str):
                digits = "".join(ch for ch in item if ch.isdigit())
                if digits:
                    residue_ids.append(int(digits))

    plot_data = data.get("plot_config", {})
    plot_config = PlotConfig(
        template=plot_data.get("template", "ggplot2"),
        style=plot_data.get("style", "simple"),
        width=plot_data.get("width", 1200),
        height=plot_data.get("height", 600),
        scale=plot_data.get("scale", 2),
        font_family=plot_data.get("font_family", "Times New Roman"),
        font_size=plot_data.get("font_size", 24),
        colors=plot_data.get("colors", {}),
        save_formats=plot_data.get("save_formats", ["html", "svg"]),
    )

    return DistanceConfig(
        base_dir=base_dir,
        output_dir=output_dir,
        protein_name=data.get("protein_name", "Protein"),
        ligand_name=data.get("ligand_name", "Ligand"),
        topology=topology,
        trajectories=trajectories,
        ligand_selection=data.get("ligand_selection", "resname UNK"),
        residue_ids=residue_ids,
        residue_selection=residue_selection,
        auto_detect_residues=auto_detect_residues,
        auto_detect_cutoff=auto_detect_cutoff,
        auto_detect_stride=auto_detect_stride,
        auto_detect_max_frames=auto_detect_max_frames,
        auto_detect_selection=auto_detect_selection,
        auto_detect_method=auto_detect_method,
        method=data.get("method", "com"),
        distance_scale=data.get("distance_scale", 1.0),
        time_unit=data.get("time_unit", "auto"),
        time_step_ns=data.get("time_step_ns"),
        per_residue_plots=data.get("per_residue_plots", True),
        combined_plot=data.get("combined_plot", True),
        combined_show_std=data.get("combined_show_std", False),
        include_overall=data.get("include_overall", False),
        plot_config=plot_config,
    )
