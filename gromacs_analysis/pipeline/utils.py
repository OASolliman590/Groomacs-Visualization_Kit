"""
Utility task runner for pipeline extras (CSV inspection, DAT conversion, renumbering, modevectors).
"""

import importlib.util
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[4]


def _load_module(module_name: str, module_path: Path):
    if not module_path.exists():
        raise FileNotFoundError(f"Module not found: {module_path}")
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    if spec and spec.loader:
        spec.loader.exec_module(module)
    return module


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _run_examine_csv(task: Dict[str, Any]) -> Dict[str, Any]:
    files = [str(p) for p in task.get("files", [])]
    if not files:
        return {"success": False, "error": "No files provided for examine_csv"}

    module_path = _repo_root() / "_legacy_scripts" / "root_utilities" / "examine_csv_files.py"
    module = _load_module("examine_csv_files", module_path)

    num_lines = task.get("num_lines", 10)
    try:
        module.increase_csv_field_size_limit()
    except Exception:
        pass

    for path in files:
        module.examine_csv_file(path, num_lines=num_lines)

    return {"success": True, "files": files}


def _run_convert_dat(task: Dict[str, Any], output_root: Optional[Path]) -> Dict[str, Any]:
    inputs = [Path(p) for p in task.get("inputs", [])]
    if not inputs:
        return {"success": False, "error": "No inputs provided for convert_dat"}

    output_dir = task.get("output_dir")
    if output_dir:
        output_dir = Path(output_dir)
    elif output_root:
        output_dir = output_root / "converted_dat"
    else:
        output_dir = Path("./converted_dat")

    _ensure_dir(output_dir)

    module_path = _repo_root() / "_legacy_scripts" / "root_utilities" / "fixed_dat_file_parsing.py"
    module = _load_module("fixed_dat_file_parsing", module_path)

    file_type = task.get("file_type", "results")
    debug = task.get("debug", False)

    converted = []
    for input_path in inputs:
        if not input_path.exists():
            logger.warning(f"DAT file not found: {input_path}")
            continue
        if "output_files" in task:
            output_files = task.get("output_files", [])
            out_path = Path(output_files[len(converted)]) if len(output_files) > len(converted) else None
        else:
            suffix = "_decomp.csv" if file_type == "decomp" else "_results.csv"
            out_path = output_dir / f"{input_path.stem}{suffix}"

        success = module.convert_dat_to_csv(str(input_path), str(out_path), file_type=file_type, debug=debug)
        if success:
            converted.append(str(out_path))

    return {"success": True, "converted": converted}


def _run_renumber_gro(task: Dict[str, Any], output_root: Optional[Path]) -> Dict[str, Any]:
    input_file = task.get("input_file")
    start_residue = int(task.get("start_residue", 1))
    if not input_file:
        return {"success": False, "error": "No input_file provided for renumber_gro"}

    input_path = Path(input_file)
    if not input_path.exists():
        return {"success": False, "error": f"Input file not found: {input_path}"}

    output_file = task.get("output_file")
    if output_file:
        output_path = Path(output_file)
    else:
        output_dir = task.get("output_dir")
        if output_dir:
            output_dir = Path(output_dir)
        elif output_root:
            output_dir = output_root
        else:
            output_dir = input_path.parent
        _ensure_dir(output_dir)
        output_path = output_dir / f"{input_path.stem}_renumbered{input_path.suffix}"

    _renumber_gro_file(input_path, output_path, start_residue)
    return {"success": True, "output": str(output_path)}


def _renumber_gro_file(input_path: Path, output_path: Path, start_residue: int) -> None:
    with open(input_path, "r") as infile, open(output_path, "w") as outfile:
        outfile.write(infile.readline())
        outfile.write(infile.readline())

        current_residue_number = start_residue
        previous_residue = None

        for line in infile:
            if len(line.strip()) > 20:
                residue_number = line[:5].strip()
                if residue_number != previous_residue:
                    previous_residue = residue_number
                    updated_residue_number = current_residue_number
                    current_residue_number += 1
                else:
                    updated_residue_number = current_residue_number - 1

                new_line = f"{updated_residue_number:>5}" + line[5:]
                outfile.write(new_line)
            else:
                outfile.write(line)


def _run_modevectors(task: Dict[str, Any], output_root: Optional[Path]) -> Dict[str, Any]:
    output_dir = task.get("output_dir")
    if output_dir:
        output_dir = Path(output_dir)
    elif output_root:
        output_dir = output_root
    else:
        output_dir = Path("./modevectors")
    _ensure_dir(output_dir)

    first_obj = task.get("first_obj")
    last_obj = task.get("last_obj")
    if not first_obj or not last_obj:
        return {"success": False, "error": "first_obj and last_obj are required for modevectors"}

    modevectors_path = _repo_root() / "_legacy_scripts" / "root_utilities" / "modevectors.py"
    pml_name = task.get("pml_name", "modevectors.pml")
    pml_path = output_dir / pml_name

    params = task.get("params", {})
    param_str = ", ".join(f"{k}={repr(v)}" for k, v in params.items())
    if param_str:
        param_str = ", " + param_str

    pml_script = f"""
run {modevectors_path}
load {first_obj}, first_obj
load {last_obj}, last_obj
modevectors first_obj, last_obj{param_str}
"""

    pml_path.write_text(pml_script.strip() + "\n")

    return {"success": True, "pml": str(pml_path)}


def _run_dry_run(task: Dict[str, Any], output_root: Optional[Path]) -> Dict[str, Any]:
    config_path = task.get("pipeline_config")
    if not config_path:
        return {"success": False, "error": "pipeline_config is required for dry_run"}

    from ..pipeline.yaml_loader import load_pipeline_config  # noqa: WPS433
    from ..md.data_processor import MDDataProcessor  # noqa: WPS433
    from ..mmpbsa.processor import MMPBSAProcessor  # noqa: WPS433
    from ..pca.processor import PCAProcessor  # noqa: WPS433
    from ..distance.processor import DistanceProcessor  # noqa: WPS433
    from ..interactions.processor import InteractionProcessor  # noqa: WPS433

    cfg = load_pipeline_config(Path(config_path))
    summary: Dict[str, Any] = {"stages": {}}

    # Simple heuristics for unit checks
    unit_thresholds = {
        "rmsd_prot": 50,
        "rmsd_lig": 50,
        "rmsd_complex": 50,
        "rmsf": 20,
        "rog": 100,
        "sasa": 100000,
        "hbonds": 2000,
    }

    for stage in cfg.enabled_stages():
        name = stage.name
        if name in ("md", "md_compare"):
            processor = MDDataProcessor(stage.config)
            files = processor.find_data_files()
            stage_info = {"missing": [], "mismatched": [], "unit_warnings": []}
            for system_name, metrics in files.items():
                for metric_name, info in metrics.items():
                    if not info.get("files"):
                        stage_info["missing"].append(f"{system_name}:{metric_name}")
                        continue
                    lengths = []
                    for file_path in info.get("files", []):
                        _, values = processor.parse_data_file(file_path, data_format="two_column")
                        lengths.append(len(values))
                        if metric_name in unit_thresholds and len(values) > 0:
                            if float(np.nanmedian(values)) > unit_thresholds[metric_name]:
                                stage_info["unit_warnings"].append(f"{system_name}:{metric_name}")
                    if lengths and (max(lengths) - min(lengths)) > 0:
                        stage_info["mismatched"].append(f"{system_name}:{metric_name}")
            summary["stages"][name] = stage_info

        elif name == "mmpbsa":
            processor = MMPBSAProcessor(stage.config)
            stage_info = {"missing_results": [], "missing_decomp": []}
            for system in stage.config.systems:
                results_files, decomp_files = processor._find_system_files(system)  # noqa: WPS437
                if not results_files:
                    stage_info["missing_results"].append(system.name)
                if not decomp_files:
                    stage_info["missing_decomp"].append(system.name)
            summary["stages"][name] = stage_info

        elif name == "pca":
            processor = PCAProcessor(stage.config)
            detected = processor.auto_detect_files()
            stage_info = {"missing": []}
            for key in ["eigenvals", "proj_1d", "cc_holo", "cc_apo"]:
                if not detected.get(key):
                    stage_info["missing"].append(key)
            summary["stages"][name] = stage_info

        elif name == "distance":
            stage_info = {"missing": []}
            if not stage.config.topology.exists():
                stage_info["missing"].append(str(stage.config.topology))
            for traj in stage.config.trajectories:
                if not traj.exists():
                    stage_info["missing"].append(str(traj))
            summary["stages"][name] = stage_info

        elif name == "interactions":
            stage_info = {"missing": []}
            if not stage.config.topology.exists():
                stage_info["missing"].append(str(stage.config.topology))
            for traj in stage.config.trajectories:
                if not traj.exists():
                    stage_info["missing"].append(str(traj))
            summary["stages"][name] = stage_info

    if output_root:
        output_root.mkdir(parents=True, exist_ok=True)
        out_path = output_root / "dry_run_summary.json"
        try:
            import json

            out_path.write_text(json.dumps(summary, indent=2))
            summary["output"] = str(out_path)
        except Exception:
            pass

    return {"success": True, "summary": summary}


def _run_generate_upstream_scripts(task: Dict[str, Any], output_root: Optional[Path]) -> Dict[str, Any]:
    output_dir = task.get("output_dir")
    if output_dir:
        output_dir = Path(output_dir)
    elif output_root:
        output_dir = output_root / "upstream_scripts"
    else:
        output_dir = Path("./upstream_scripts")

    from .upstream import generate_upstream_scripts  # noqa: WPS433

    return generate_upstream_scripts(task, output_dir)


def _run_duivy_inspect_guide(task: Dict[str, Any], output_root: Optional[Path]) -> Dict[str, Any]:
    """Write an optional DuIvyTools guide for raw XVG/XPM inspection."""
    output_dir = Path(task.get("output_dir")) if task.get("output_dir") else (output_root or Path("./utils"))
    _ensure_dir(output_dir)

    output_file = Path(task.get("output_file")) if task.get("output_file") else (output_dir / "duivy_inspection_guide.md")
    xvg_files = [str(path) for path in task.get("xvg_files", [])]
    xpm_files = [str(path) for path in task.get("xpm_files", [])]

    lines = [
        "# DuIvyTools Raw Inspection Guide",
        "",
        "This guide is optional and intended for raw `.xvg/.xpm` sanity inspection only.",
        "It does not replace the maintained Python toolkit outputs.",
        "",
        "## Command Discovery",
        "",
        "- `dit --help`",
        "- `dit xvg_show -h`",
        "- `dit xvg_compare -h`",
        "- `dit xpm_show -h`",
        "- `dit xpm2csv -h`",
        "",
    ]

    if xvg_files:
        lines.extend(["## XVG Inputs", ""])
        for idx, xvg_file in enumerate(xvg_files, start=1):
            lines.append(f"- `{xvg_file}`")
            if idx <= 3:
                lines.append(f"  - preview: `dit xvg_show -f {xvg_file}`")
        if len(xvg_files) >= 2:
            lines.append(
                f"- compare first two: `dit xvg_compare -f {xvg_files[0]} {xvg_files[1]}`"
            )
        lines.append("")

    if xpm_files:
        lines.extend(["## XPM Inputs", ""])
        for idx, xpm_file in enumerate(xpm_files, start=1):
            lines.append(f"- `{xpm_file}`")
            if idx <= 3:
                lines.append(f"  - preview: `dit xpm_show -f {xpm_file}`")
                lines.append(f"  - export csv: `dit xpm2csv -f {xpm_file}`")
        lines.append("")

    if not xvg_files and not xpm_files:
        lines.extend(
            [
                "## Inputs",
                "",
                "No raw files were listed. Add `xvg_files` and/or `xpm_files` in this utility task to prefill commands.",
                "",
            ]
        )

    lines.extend(
        [
            "## Notes",
            "",
            "- Use these commands for quick validation and troubleshooting of raw files.",
            "- Keep ranking and scientific reporting based on maintained package outputs.",
            "",
        ]
    )

    output_file.write_text("\n".join(lines))
    return {"success": True, "guide_file": str(output_file), "xvg_count": len(xvg_files), "xpm_count": len(xpm_files)}


def run_utils_tasks(config: Dict[str, Any]) -> Dict[str, Any]:
    tasks = config.get("tasks", [])
    output_root = None
    if config.get("output_dir"):
        output_root = Path(config["output_dir"])
        _ensure_dir(output_root)

    results: List[Dict[str, Any]] = []

    for task in tasks:
        task_type = task.get("type")
        if task_type == "examine_csv":
            results.append(_run_examine_csv(task))
        elif task_type == "convert_dat":
            results.append(_run_convert_dat(task, output_root))
        elif task_type == "renumber_gro":
            results.append(_run_renumber_gro(task, output_root))
        elif task_type == "modevectors":
            results.append(_run_modevectors(task, output_root))
        elif task_type == "dry_run":
            results.append(_run_dry_run(task, output_root))
        elif task_type == "generate_upstream_scripts":
            results.append(_run_generate_upstream_scripts(task, output_root))
        elif task_type == "duivy_inspect_guide":
            results.append(_run_duivy_inspect_guide(task, output_root))
        else:
            results.append({"success": False, "error": f"Unknown utility task: {task_type}"})

    return {"success": True, "tasks": results}
