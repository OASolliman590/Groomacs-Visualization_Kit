from pathlib import Path

from gromacs_analysis.cli.orchestrate_cli import (
    SystemConfig,
    WorkflowConfig,
    _build_trjcat_command,
    _build_parts_qc,
    _collect_trajectory_parts,
    _extract_segment_number,
    _parse_ndx_groups,
    _segment_sort_key,
)


def test_segment_sort_key_prefers_numeric_parts():
    names = [
        "step5_10.xtc",
        "step5_2.xtc",
        "step5_1.xtc",
        "random.xtc",
    ]
    ordered = sorted(names, key=_segment_sort_key)
    assert ordered[:3] == ["step5_1.xtc", "step5_2.xtc", "step5_10.xtc"]
    assert ordered[-1] == "random.xtc"


def test_collect_trajectory_parts_filters_outputs(tmp_path: Path):
    system_dir = tmp_path / "sys"
    system_dir.mkdir()
    for name in (
        "step5_1.xtc",
        "step5_2.xtc",
        "step5_production_noPBC.xtc",
        "step5_production_cat.xtc",
    ):
        (system_dir / name).write_text("x")

    cfg = SystemConfig(
        name="sys",
        workdir=system_dir,
        ligand_resname="PAT",
        trajectory_patterns=["step5_*.xtc", "step5_production*.xtc"],
    )
    parts = _collect_trajectory_parts(cfg)
    assert [p.name for p in parts] == ["step5_1.xtc", "step5_2.xtc"]


def test_parse_ndx_groups_reads_order(tmp_path: Path):
    ndx = tmp_path / "new_index.ndx"
    ndx.write_text(
        "[ System ]\n1 2 3\n[ Protein ]\n1 2\n[ PAT ]\n3 4\n"
    )
    groups = _parse_ndx_groups(ndx)
    assert groups["System"] == 0
    assert groups["Protein"] == 1
    assert groups["PAT"] == 2


def test_extract_segment_number_supports_step_and_seg():
    assert _extract_segment_number("step5_12.xtc") == 12
    assert _extract_segment_number("seg_7.xtc") == 7
    assert _extract_segment_number("random.xtc") is None


def test_parts_qc_detects_missing_numbers(tmp_path: Path):
    system_dir = tmp_path / "sys"
    system_dir.mkdir()
    p1 = system_dir / "step5_1.xtc"
    p3 = system_dir / "step5_3.xtc"
    p1.write_text("a")
    p3.write_text("b")

    cfg = SystemConfig(
        name="sys",
        workdir=system_dir,
        ligand_resname="PAT",
        strict_segment_sequence=False,
    )
    report = _build_parts_qc(cfg, [p1, p3])
    assert report["is_split_input"] is True
    assert report["missing_segment_numbers"] == [2]


def test_build_trjcat_command_with_settime(tmp_path: Path):
    system_dir = tmp_path / "sys"
    system_dir.mkdir()
    p1 = system_dir / "step5_1.xtc"
    p2 = system_dir / "step5_2.xtc"
    p1.write_text("a")
    p2.write_text("b")

    cfg = WorkflowConfig(gmx="gmx")
    system = SystemConfig(
        name="sys",
        workdir=system_dir,
        ligand_resname="PAT",
        use_settime=True,
        settime_starts_ps=[0.0, 1000.0],
    )
    cmd, stdin = _build_trjcat_command(
        cfg=cfg,
        system=system,
        parts=[p1, p2],
        mode="dedup",
        cat_out=system_dir / "cat.xtc",
        dry_run=False,
    )
    assert "-settime" in cmd
    assert stdin == "0.0\n1000.0\n"
