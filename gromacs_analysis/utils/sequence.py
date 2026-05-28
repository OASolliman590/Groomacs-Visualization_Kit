"""Sequence helpers for residue labels."""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple, Dict


def load_sequence_from_topology(topology_path: Path, selection: str = "protein") -> List[Tuple[int, str]]:
    """
    Load residue IDs and names from a topology using MDAnalysis.

    Returns a list of (resid, resname) tuples in order.
    """
    try:
        import MDAnalysis as mda  # noqa: WPS433
    except ImportError as exc:
        raise ImportError("MDAnalysis is required to load sequence from topology") from exc

    topo = Path(topology_path)
    if not topo.exists():
        raise FileNotFoundError(f"Topology not found: {topo}")

    u = mda.Universe(str(topo))
    residues = u.select_atoms(selection).residues
    return [(int(res.resid), res.resname) for res in residues]


def build_residue_labels(residues: List[Tuple[int, str]], include_names: bool = True) -> List[str]:
    """Build residue labels from (resid, resname) tuples."""
    labels: List[str] = []
    for resid, resname in residues:
        if include_names:
            labels.append(f"{resname}{resid}")
        else:
            labels.append(str(resid))
    return labels


def load_sequence_from_pdb(pdb_path: Path) -> List[Tuple[int, str]]:
    """
    Load residue IDs and names from a PDB file by parsing ATOM/HETATM records.

    Keeps first-seen residue order and de-duplicates by (chain, resid, insertion code).
    """
    path = Path(pdb_path)
    if not path.exists():
        raise FileNotFoundError(f"PDB not found: {path}")

    residues: List[Tuple[int, str]] = []
    seen: Dict[Tuple[str, int, str], bool] = {}

    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            if not (line.startswith("ATOM") or line.startswith("HETATM")):
                continue
            if len(line) < 27:
                continue

            resname = line[17:20].strip()
            chain_id = line[21:22].strip()
            resid_str = line[22:26].strip()
            icode = line[26:27].strip()

            if not resid_str:
                continue
            try:
                resid = int(resid_str)
            except ValueError:
                continue

            key = (chain_id, resid, icode)
            if key in seen:
                continue
            seen[key] = True
            residues.append((resid, resname or "UNK"))

    if not residues:
        raise ValueError(f"No residues parsed from PDB: {path}")

    return residues
