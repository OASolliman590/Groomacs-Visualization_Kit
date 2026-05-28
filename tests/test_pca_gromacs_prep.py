from pathlib import Path

import yaml

from gromacs_analysis.pca.config import PCATerminalDetectionConfig
from gromacs_analysis.pca.prep import (
    build_index_plan,
    build_residue_map,
    detect_terminals,
    parse_gro_residues,
    parse_pdb_residues,
    parse_rmsf_file,
)
from gromacs_analysis.pipeline.yaml_loader import load_pipeline_config


def _write_gro(path: Path, count: int) -> None:
    lines = ["Synthetic GRO", str(count)]
    atom_id = 1
    for resid in range(1, count + 1):
        lines.append(f"{resid:5d}{'ALA':<5}{'CA':>5}{atom_id:5d}{0.1 * resid:8.3f}{0.0:8.3f}{0.0:8.3f}")
        atom_id += 1
    lines.append("   1.00000   1.00000   1.00000")
    path.write_text("\n".join(lines) + "\n")


def _write_pdb(path: Path, start_resid: int, count: int) -> None:
    lines = []
    atom_id = 1
    for offset in range(count):
        resid = start_resid + offset
        lines.append(
            f"ATOM  {atom_id:5d}  CA  ALA A{resid:4d}    "
            f"{float(offset):8.3f}{0.0:8.3f}{0.0:8.3f}"
            f"  1.00  0.00      PROA C"
        )
        atom_id += 1
    path.write_text("\n".join(lines) + "\n")


def _write_rmsf(path: Path, values) -> None:
    path.write_text("".join(f"{idx} {value}\n" for idx, value in enumerate(values, start=1)))


def test_rmsf_gro_native_pdb_alignment_and_terminal_detection(tmp_path: Path):
    gro_path = tmp_path / "step3_input.gro"
    pdb_path = tmp_path / "step3_input.pdb"
    rmsf_path = tmp_path / "RMSF_.dat"
    _write_gro(gro_path, 5)
    _write_pdb(pdb_path, 321, 5)
    _write_rmsf(rmsf_path, [5.0, 0.4, 0.3, 0.4, 4.0])

    residue_map = build_residue_map(
        parse_rmsf_file(rmsf_path),
        parse_gro_residues(gro_path),
        parse_pdb_residues(pdb_path),
    )
    detection = detect_terminals(
        residue_map,
        PCATerminalDetectionConfig(
            smoothing_window=1,
            mad_multiplier=2.0,
            stable_run=1,
            max_terminal_fraction=0.4,
        ),
    )

    assert len(residue_map) == 5
    assert residue_map[0].gro_resid == 1
    assert residue_map[0].native_resid == 321
    assert detection.gro_start_resid == 2
    assert detection.gro_end_resid == 4
    assert detection.native_start_resid == 322
    assert detection.native_end_resid == 324


def test_index_plan_uses_detected_range_and_generated_group_ids():
    class Detection:
        gro_start_resid = 9
        gro_end_resid = 289

    plan = build_index_plan(
        Detection(),
        existing_group_count=16,
        core_group_name="core_no_terminals",
        calpha_group_name="core_no_terminals_Calpha",
    )

    assert plan.core_group_id == 16
    assert plan.calpha_group_id == 17
    assert plan.commands == [
        "ri 9-289",
        "name 16 core_no_terminals",
        "16 & a CA",
        "name 17 core_no_terminals_Calpha",
        "q",
    ]
    assert "18" not in plan.stdin


def test_pca_batch_supports_project_root_simulation_and_pca_dirs(tmp_path: Path):
    config_path = tmp_path / "pipeline.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "project": {"name": "demo", "protein_name": "KEAP1"},
                "run": ["pca_batch"],
                "pca_batch": {
                    "enabled": True,
                    "base_dir": str(tmp_path),
                    "output_dir": str(tmp_path / "analysis_outputs" / "pca_batch"),
                    "run_gromacs": {
                        "enabled": True,
                        "gmx": "gmx",
                        "terminal_detection": {"core_residue_range": [9, 289]},
                    },
                    "systems": [
                        {
                            "name": "Benzyl",
                            "simulation_dir": "Benzyl_Holo_1_11b",
                            "pca_dir": "Benzyl_PCA",
                        }
                    ],
                },
            },
            sort_keys=False,
        )
    )

    cfg = load_pipeline_config(config_path)
    pca_cfg = cfg.stages["pca_batch"].config[0]

    assert pca_cfg.base_dir == tmp_path / "Benzyl_PCA"
    assert pca_cfg.run_gromacs.enabled is True
    assert pca_cfg.run_gromacs.input_dir == tmp_path / "Benzyl_Holo_1_11b"
    assert pca_cfg.run_gromacs.terminal_detection.core_residue_range == (9, 289)
