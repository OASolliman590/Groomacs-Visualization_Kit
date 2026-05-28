"""Automatic GROMACS PCA output generation."""

from __future__ import annotations

import json
import logging
import shlex
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from .config import PCAConfig
from .prep import (
    PCAIndexPlan,
    build_index_plan,
    build_residue_map,
    count_index_groups,
    detect_terminals,
    parse_gro_residues,
    parse_pdb_residues,
    parse_rmsf_file,
    write_index_plan,
    write_residue_map,
    write_terminal_detection,
)

logger = logging.getLogger(__name__)


class PCAGromacsRunner:
    """Generate PCA input/output files by running GROMACS commands."""

    def __init__(self, config: PCAConfig):
        self.config = config
        self.run_config = config.run_gromacs
        self.pca_dir = Path(config.base_dir)
        self.input_dir = Path(self.run_config.input_dir) if self.run_config.input_dir else self.pca_dir
        self.log_path = self.pca_dir / "pca_run_commands.log"

    def run(self) -> Dict[str, object]:
        """Run the complete PCA-generation workflow."""
        if not self.run_config.enabled:
            return {"success": True, "skipped": True}

        self.pca_dir.mkdir(parents=True, exist_ok=True)
        if self.log_path.exists() and self.run_config.overwrite:
            self.log_path.unlink()

        logger.info("Preparing RMSF-guided PCA index groups")
        residue_map, detection = self._prepare_residue_metadata()
        default_index = self.pca_dir / "_default_index.ndx"
        final_index = self.pca_dir / self.run_config.index

        gro_path = self._input_path(self.run_config.terminal_detection.gro_file)
        self._run([self.run_config.gmx, "make_ndx", "-f", str(gro_path), "-o", str(default_index)], stdin="q\n")
        group_count = count_index_groups(default_index)
        index_plan = build_index_plan(
            detection,
            existing_group_count=group_count,
            core_group_name=self.run_config.core_group_name,
            calpha_group_name=self.run_config.calpha_group_name,
        )
        write_index_plan(index_plan, self.pca_dir / "pca_index_plan.txt")
        self._run(
            [
                self.run_config.gmx,
                "make_ndx",
                "-f",
                str(gro_path),
                "-n",
                str(default_index),
                "-o",
                str(final_index),
            ],
            stdin=index_plan.stdin,
        )
        if default_index.exists():
            default_index.unlink()

        self._run_trajectory_preparation(final_index, index_plan)
        preflight = self._preflight_prepared_trajectory()
        self._run_covariance_and_projections()

        return {
            "success": True,
            "pca_dir": str(self.pca_dir),
            "index": str(final_index),
            "core_range_gro": [detection.gro_start_resid, detection.gro_end_resid],
            "core_range_native": [detection.native_start_resid, detection.native_end_resid],
            "residue_count": len(residue_map),
            "preflight": preflight,
        }

    def _prepare_residue_metadata(self):
        terminal_cfg = self.run_config.terminal_detection
        rmsf_path = self._input_path(terminal_cfg.rmsf_file)
        gro_path = self._input_path(terminal_cfg.gro_file)
        pdb_path = self._input_path(terminal_cfg.pdb_file)

        rmsf_values = parse_rmsf_file(rmsf_path)
        gro_residues = parse_gro_residues(gro_path)
        pdb_residues = parse_pdb_residues(pdb_path) if pdb_path.exists() else None
        residue_map = build_residue_map(rmsf_values, gro_residues, pdb_residues)
        detection = detect_terminals(residue_map, terminal_cfg)

        write_residue_map(residue_map, self.pca_dir / "residue_map.csv")
        write_terminal_detection(detection, self.pca_dir / "terminal_detection.json")
        return residue_map, detection

    def _run_trajectory_preparation(self, index_path: Path, index_plan: PCAIndexPlan) -> None:
        tpr = self._input_path(self.run_config.tpr)
        input_traj = self._input_path(self.run_config.trajectory)
        no_pbc = self.pca_dir / self.run_config.no_pbc_trajectory
        calpha_traj = self.pca_dir / "step5_production_noPBC_Calpha.xtc"
        ref_pdb = self.pca_dir / "ref.pdb"

        self._run(
            [
                self.run_config.gmx,
                "trjconv",
                "-f",
                str(input_traj),
                "-o",
                str(no_pbc),
                "-s",
                str(tpr),
                "-pbc",
                "mol",
                "-center",
                "-n",
                str(index_path),
                "-ur",
                "compact",
            ],
            stdin=f"{self.run_config.protein_group}\n{self.run_config.output_group}\n",
        )
        self._run(
            [
                self.run_config.gmx,
                "trjconv",
                "-f",
                str(no_pbc),
                "-o",
                str(calpha_traj),
                "-n",
                str(index_path),
            ],
            stdin=f"{index_plan.calpha_group_name}\n",
        )
        self._run(
            [
                self.run_config.gmx,
                "trjconv",
                "-s",
                str(tpr),
                "-f",
                str(no_pbc),
                "-dump",
                "0",
                "-o",
                str(ref_pdb),
                "-n",
                str(index_path),
            ],
            stdin=f"{index_plan.calpha_group_name}\n",
        )

    def _run_covariance_and_projections(self) -> None:
        gmx = self.run_config.gmx
        ref = "ref.pdb"
        traj = "step5_production_noPBC_Calpha.xtc"
        n_pcs = str(self.run_config.n_pcs)

        self._run([gmx, "covar", "-mwa", "-ref", "-s", ref, "-f", traj, "-v", "eigenvectors_all_ref.trr", "-o", "eigenvals_all_ref.xvg"], stdin="System\nSystem\n", cwd=self.pca_dir)
        for last in (2, 3, 10):
            self._run(
                [
                    gmx,
                    "covar",
                    "-mwa",
                    "-ref",
                    "-s",
                    ref,
                    "-f",
                    traj,
                    "-v",
                    f"eigenvectors_1-{last}_ref.trr",
                    "-o",
                    f"eigenvals_1-{last}_ref.xvg",
                    "-last",
                    str(last),
                ],
                stdin="System\nSystem\n",
                cwd=self.pca_dir,
            )

        self._run(
            [gmx, "anaeig", "-proj", "proj.xvg", "-first", "1", "-last", n_pcs, "-s", ref, "-f", traj, "-v", "eigenvectors_all_ref.trr", "-eig", "eigenvals_all_ref.xvg"],
            stdin="System\nSystem\n",
            cwd=self.pca_dir,
        )
        self._run([gmx, "analyze", "-n", n_pcs, "-cc", "cc_all.xvg", "-f", "proj.xvg"], cwd=self.pca_dir)

        if self.run_config.window_cosine_enabled:
            for end_ns in range(self.run_config.window_step_ns, self.run_config.window_max_ns + 1, self.run_config.window_step_ns):
                self._run(
                    [
                        gmx,
                        "anaeig",
                        "-proj",
                        f"proj_0_{end_ns}.xvg",
                        "-first",
                        "1",
                        "-last",
                        n_pcs,
                        "-s",
                        ref,
                        "-f",
                        traj,
                        "-v",
                        "eigenvectors_all_ref.trr",
                        "-eig",
                        "eigenvals_all_ref.xvg",
                        "-b",
                        "0",
                        "-e",
                        str(end_ns),
                        "-tu",
                        "ns",
                    ],
                    stdin="System\nSystem\n",
                    cwd=self.pca_dir,
                )
                self._run([gmx, "analyze", "-n", n_pcs, "-cc", f"coscont_0_{end_ns}.xvg", "-f", f"proj_0_{end_ns}.xvg"], cwd=self.pca_dir)

        self._run(
            [
                gmx,
                "anaeig",
                "-extr",
                "extreme.pdb",
                "-first",
                "1",
                "-last",
                "3",
                "-s",
                ref,
                "-f",
                traj,
                "-nframes",
                str(self.run_config.extreme_nframes),
                "-v",
                "eigenvectors_all_ref.trr",
                "-eig",
                "eigenvals_all_ref.xvg",
            ],
            stdin="System\nSystem\n",
            cwd=self.pca_dir,
        )

    def _preflight_prepared_trajectory(self) -> Dict[str, object]:
        """Validate prepared PCA trajectory before covar/anaeig execution."""
        no_pbc = self.pca_dir / self.run_config.no_pbc_trajectory
        calpha = self.pca_dir / "step5_production_noPBC_Calpha.xtc"
        ref_pdb = self.pca_dir / "ref.pdb"

        report: Dict[str, object] = {
            "success": True,
            "errors": [],
            "warnings": [],
            "paths": {
                "no_pbc_trajectory": str(no_pbc),
                "calpha_trajectory": str(calpha),
                "reference_pdb": str(ref_pdb),
            },
            "metrics": {},
        }

        for path in (no_pbc, calpha, ref_pdb):
            if not path.exists():
                report["errors"].append(f"Missing prepared file: {path}")
            elif path.stat().st_size == 0:
                report["errors"].append(f"Prepared file is empty: {path}")

        ref_atom_count = self._count_pdb_atoms(ref_pdb) if ref_pdb.exists() else 0
        report["metrics"]["reference_atom_count"] = ref_atom_count
        if ref_atom_count < 1:
            report["errors"].append("Reference PDB contains no ATOM/HETATM entries.")

        if calpha.exists() and ref_pdb.exists():
            try:
                import MDAnalysis as mda  # noqa: WPS433
            except Exception as exc:
                report["metrics"]["mdanalysis_enabled"] = False
                report["warnings"].append(f"MDAnalysis preflight unavailable: {exc}")
            else:
                try:
                    universe = mda.Universe(str(ref_pdb), str(calpha))
                    frame_count = len(universe.trajectory)
                    atom_count = int(universe.atoms.n_atoms)
                    timestep_ps = float(universe.trajectory.dt) if universe.trajectory.dt else None
                    report["metrics"]["calpha_frame_count"] = frame_count
                    report["metrics"]["calpha_atom_count"] = atom_count
                    report["metrics"]["trajectory_timestep_ps"] = timestep_ps
                    report["metrics"]["mdanalysis_enabled"] = True

                    if frame_count < 2:
                        report["errors"].append("Prepared C-alpha trajectory has fewer than 2 frames.")
                    if atom_count < 1:
                        report["errors"].append("Prepared C-alpha trajectory has zero atoms.")
                    if ref_atom_count and atom_count != ref_atom_count:
                        report["warnings"].append(
                            f"C-alpha trajectory atom count ({atom_count}) differs from ref PDB atoms ({ref_atom_count})."
                        )
                except Exception as exc:
                    report["errors"].append(f"Failed to parse prepared C-alpha trajectory: {exc}")

        report["success"] = len(report["errors"]) == 0
        preflight_path = self.pca_dir / "pca_preflight.json"
        preflight_path.write_text(json.dumps(report, indent=2))

        if not report["success"]:
            raise RuntimeError(f"PCA preflight failed; see {preflight_path}")

        return report

        for first, last, output in [(1, 2, "2dproj_12.xvg"), (1, 3, "2dproj_13.xvg"), (2, 3, "2dproj_23.xvg")]:
            self._run(
                [
                    gmx,
                    "anaeig",
                    "-s",
                    ref,
                    "-f",
                    traj,
                    "-2d",
                    output,
                    "-first",
                    str(first),
                    "-last",
                    str(last),
                    "-v",
                    "eigenvectors_all_ref.trr",
                    "-eig",
                    "eigenvals_all_ref.xvg",
                ],
                stdin="System\nSystem\n",
                cwd=self.pca_dir,
            )

        self._run(
            [
                gmx,
                "anaeig",
                "-s",
                ref,
                "-f",
                traj,
                "-3d",
                "3dproj_123.pdb",
                "-first",
                "1",
                "-last",
                "3",
                "-v",
                "eigenvectors_all_ref.trr",
                "-eig",
                "eigenvals_all_ref.xvg",
            ],
            stdin="System\nSystem\n",
            cwd=self.pca_dir,
        )

    def _input_path(self, name: str) -> Path:
        path = Path(name)
        if path.is_absolute():
            return path
        return self.input_dir / path

    def _run(self, command: Sequence[str], stdin: Optional[str] = None, cwd: Optional[Path] = None) -> None:
        cwd = cwd or self.input_dir
        self._log_command(command, stdin=stdin, cwd=cwd)
        logger.info("Running: %s", " ".join(shlex.quote(part) for part in command))
        subprocess.run(
            list(command),
            input=stdin,
            text=True,
            check=True,
            cwd=str(cwd),
        )

    def _log_command(self, command: Sequence[str], stdin: Optional[str], cwd: Path) -> None:
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with self.log_path.open("a") as handle:
            handle.write(f"$ cd {cwd}\n")
            if stdin:
                handle.write(f"$ printf {shlex.quote(stdin)} | ")
            else:
                handle.write("$ ")
            handle.write(" ".join(shlex.quote(part) for part in command))
            handle.write("\n\n")

    @staticmethod
    def _count_pdb_atoms(path: Path) -> int:
        count = 0
        with path.open() as handle:
            for line in handle:
                if line.startswith(("ATOM", "HETATM")):
                    count += 1
        return count
