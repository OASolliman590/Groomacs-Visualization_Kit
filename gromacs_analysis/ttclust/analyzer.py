"""TTClust analysis orchestrator."""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, List

from .config import TTClustConfig

logger = logging.getLogger(__name__)


class TTClustAnalyzer:
    """Run TTClust and collect generated figures/artifacts."""

    def __init__(self, config: TTClustConfig):
        self.config = config

    def run_analysis(self) -> Dict:
        logger.info("=" * 60)
        logger.info("Starting TTClust Analysis")
        logger.info("=" * 60)

        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        data_dir = self.config.output_dir / "data"
        plots_dir = self.config.output_dir / "plots"
        data_dir.mkdir(parents=True, exist_ok=True)
        plots_dir.mkdir(parents=True, exist_ok=True)

        topology = self._resolve_optional_path(self.config.topology)
        trajectories = [self._resolve_path(path) for path in self.config.trajectories]
        for traj in trajectories:
            if not traj.exists():
                raise FileNotFoundError(f"Trajectory not found: {traj}")
        if topology and not topology.exists():
            raise FileNotFoundError(f"Topology not found: {topology}")

        command = self._build_command(topology, trajectories)
        command = self._wrap_ttclust_command(command, data_dir)
        env = self._build_environment(data_dir)
        logger.info("Running TTClust command: %s", " ".join(command))
        proc = subprocess.run(
            command,
            cwd=str(self.config.output_dir),
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )

        (data_dir / "ttclust_stdout.log").write_text(proc.stdout or "")
        (data_dir / "ttclust_stderr.log").write_text(proc.stderr or "")
        (data_dir / "ttclust_command.txt").write_text(" ".join(command) + "\n")

        if proc.returncode != 0:
            tail = "\n".join((proc.stderr or "").splitlines()[-20:])
            raise RuntimeError(
                f"TTClust failed with exit code {proc.returncode}. "
                f"See {data_dir / 'ttclust_stderr.log'}\n{tail}"
            )

        figure_files = self._collect_figure_files()
        figure_index = data_dir / "figures_index.json"
        figure_index.write_text(json.dumps([str(path) for path in figure_files], indent=2) + "\n")

        # Keep a flat copy in plots/ for quick browsing.
        copied = []
        for fig in figure_files:
            target = plots_dir / fig.name
            if fig.resolve() == target.resolve():
                copied.append(target)
                continue
            shutil.copy2(fig, target)
            copied.append(target)

        cluster_pdb_dir = self.config.output_dir / "Cluster_PDB"
        result = {
            "success": True,
            "output_dir": str(self.config.output_dir),
            "figures": [str(p) for p in figure_files],
            "figures_flat_copy": [str(p) for p in copied],
            "figure_count": len(figure_files),
            "log_file": str(self.config.output_dir / self.config.logfile),
            "cluster_pdb_dir": str(cluster_pdb_dir) if cluster_pdb_dir.exists() else None,
            "command_file": str(data_dir / "ttclust_command.txt"),
            "stdout_file": str(data_dir / "ttclust_stdout.log"),
            "stderr_file": str(data_dir / "ttclust_stderr.log"),
            "figures_index_file": str(figure_index),
        }
        logger.info("TTClust complete; figures found: %d", len(figure_files))
        return result

    def _build_environment(self, data_dir: Path) -> Dict[str, str]:
        env = os.environ.copy()
        mpl_config = data_dir / "mplconfig"
        numba_cache = data_dir / "numba_cache"
        mpl_config.mkdir(parents=True, exist_ok=True)
        numba_cache.mkdir(parents=True, exist_ok=True)

        # TTClust relies on numba JIT decorators that can fail in constrained envs.
        env.setdefault("NUMBA_DISABLE_JIT", "1")
        env.setdefault("NUMBA_CACHE_DIR", str(numba_cache))
        env.setdefault("MPLCONFIGDIR", str(mpl_config))

        for key, value in self.config.environment.items():
            env[str(key)] = str(value)
        return env

    def _build_command(self, topology: Path | None, trajectories: List[Path]) -> List[str]:
        exe = self._resolve_executable()

        command = [exe, "-f", *[str(path) for path in trajectories]]
        if topology is not None:
            command.extend(["-t", str(topology)])
        command.extend(["-s", str(self.config.stride)])
        command.extend(["-l", self.config.logfile])
        command.extend(["-st", self.config.select_traj])
        command.extend(["-sa", self.config.select_alignment])
        command.extend(["-sr", self.config.select_rmsd])
        command.extend(["-m", self.config.method])
        command.extend(["-i", "Y" if self.config.interactive_matrix else "n"])
        if self.config.autoclust:
            command.extend(["-aa", "Y"])
        if self.config.cutoff is not None:
            command.extend(["-cc", str(self.config.cutoff)])
        if self.config.n_groups is not None:
            command.extend(["-ng", str(self.config.n_groups)])
        if self.config.axis:
            command.extend(["-axis", self.config.axis])
        if self.config.limit_matrix is not None:
            command.extend(["-limitmat", str(self.config.limit_matrix)])
        if self.config.extra_args:
            command.extend(self.config.extra_args)
        return command

    def _resolve_executable(self) -> str:
        if self.config.executable:
            return self.config.executable

        for candidate in ("ttclust.py", "ttclust"):
            resolved = shutil.which(candidate)
            if resolved:
                return resolved

        raise FileNotFoundError(
            "TTClust executable not found. Install TTClust (for example: "
            "`conda install -c tubiana -c conda-forge ttclust`) or set `ttclust.executable`."
        )

    def _wrap_ttclust_command(self, command: List[str], data_dir: Path) -> List[str]:
        exe_name = Path(command[0]).name.lower()
        if exe_name not in {"ttclust", "ttclust.py"}:
            return command

        wrapper_path = data_dir / "ttclust_wrapper.py"
        wrapper_path.write_text(
            "\n".join(
                [
                    "import numba",
                    "_original_jit = numba.jit",
                    "def _jit_no_cache(*args, **kwargs):",
                    "    kwargs['cache'] = False",
                    "    return _original_jit(*args, **kwargs)",
                    "numba.jit = _jit_no_cache",
                    "from ttclust.ttclust import main",
                    "if __name__ == '__main__':",
                    "    main()",
                ]
            )
            + "\n"
        )
        return [sys.executable, str(wrapper_path), *command[1:]]

    def _collect_figure_files(self) -> List[Path]:
        figure_files: List[Path] = []
        for pattern in ("*.png", "*.svg", "*.jpg", "*.jpeg"):
            figure_files.extend(self.config.output_dir.rglob(pattern))
        unique = sorted({path.resolve() for path in figure_files})
        return unique

    def _resolve_path(self, path: Path) -> Path:
        if path.is_absolute():
            return path
        return (self.config.base_dir / path).resolve()

    def _resolve_optional_path(self, path: Path | None) -> Path | None:
        if path is None:
            return None
        return self._resolve_path(path)
