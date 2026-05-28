"""Processor for interaction fingerprints."""

from __future__ import annotations

import logging
from typing import Dict, List, Tuple

import numpy as np

from .config import InteractionConfig

logger = logging.getLogger(__name__)


class InteractionProcessor:
    def __init__(self, config: InteractionConfig):
        self.config = config

    def compute_all(self) -> Dict[str, List[Dict]]:
        """Compute occupancy per residue for each interaction type."""
        try:
            import MDAnalysis as mda  # noqa: WPS433
            from MDAnalysis.lib.distances import distance_array  # noqa: WPS433
        except ImportError as exc:
            raise ImportError("MDAnalysis is required for interaction analysis") from exc

        topology = self.config.topology
        trajectories = self.config.trajectories
        if not topology.exists():
            raise FileNotFoundError(f"Topology not found: {topology}")
        for traj in trajectories:
            if not traj.exists():
                raise FileNotFoundError(f"Trajectory not found: {traj}")

        results: Dict[str, List[Dict]] = {}

        for interaction in self.config.interaction_types:
            interaction = interaction.lower()
            if interaction == "hbond":
                res = self._contact_occupancy(
                    mda,
                    distance_array,
                    protein_sel=f"{self.config.protein_selection} and (name N* O*)",
                    ligand_sel=f"({self.config.ligand_selection}) and (name N* O*)",
                    cutoff=self.config.hbond_cutoff,
                )
            elif interaction == "hydrophobic":
                res = self._contact_occupancy(
                    mda,
                    distance_array,
                    protein_sel=(
                        f"{self.config.protein_selection} and resname ALA VAL LEU ILE MET PHE TYR TRP PRO and (name C* S*)"
                    ),
                    ligand_sel=f"({self.config.ligand_selection}) and (name C* S*)",
                    cutoff=self.config.hydrophobic_cutoff,
                )
            elif interaction == "salt_bridge":
                res = self._salt_bridge_occupancy(mda, distance_array)
            elif interaction == "pi_stacking":
                res = self._pi_stacking_occupancy(mda, distance_array)
            else:
                logger.warning(f"Unknown interaction type: {interaction}")
                res = []

            results[interaction] = res

        return results

    def _contact_occupancy(
        self,
        mda,
        distance_array,
        protein_sel: str,
        ligand_sel: str,
        cutoff: float,
    ) -> List[Dict]:
        u = mda.Universe(str(self.config.topology), str(self.config.trajectories[0]))
        protein_atoms = u.select_atoms(protein_sel)
        ligand_atoms = u.select_atoms(ligand_sel)
        residues = protein_atoms.residues

        counts = {int(res.resid): 0 for res in residues}
        resnames = {int(res.resid): res.resname for res in residues}
        frames = 0

        for traj in self.config.trajectories:
            u = mda.Universe(str(self.config.topology), str(traj))
            protein_atoms = u.select_atoms(protein_sel)
            ligand_atoms = u.select_atoms(ligand_sel)
            residues = protein_atoms.residues
            if ligand_atoms.n_atoms == 0 or protein_atoms.n_atoms == 0:
                continue
            for ts in u.trajectory[:: self.config.stride]:
                frames += 1
                lig_pos = ligand_atoms.positions
                for res in residues:
                    dist = np.min(distance_array(res.atoms.positions, lig_pos))
                    if dist <= cutoff:
                        counts[int(res.resid)] = counts.get(int(res.resid), 0) + 1

        results = []
        for resid, count in counts.items():
            occ = (count / frames * 100.0) if frames else 0.0
            resname = resnames.get(resid, "UNK")
            results.append({
                "resid": resid,
                "resname": resname,
                "label": f"{resname}{resid}",
                "occupancy": occ,
            })
        return results

    def _salt_bridge_occupancy(self, mda, distance_array) -> List[Dict]:
        # Default selections: protein charged sidechains, ligand N/O atoms
        protein_pos = f"{self.config.protein_selection} and (resname LYS ARG HIS) and (name NZ NH* NE* ND*)"
        protein_neg = f"{self.config.protein_selection} and (resname ASP GLU) and (name OD* OE*)"
        ligand_pos = self.config.ligand_pos_sel or f"({self.config.ligand_selection}) and name N*"
        ligand_neg = self.config.ligand_neg_sel or f"({self.config.ligand_selection}) and name O*"

        # combine contacts from both directions
        res_pos = self._contact_occupancy(mda, distance_array, protein_pos, ligand_neg, self.config.salt_bridge_cutoff)
        res_neg = self._contact_occupancy(mda, distance_array, protein_neg, ligand_pos, self.config.salt_bridge_cutoff)

        combined: Dict[int, Dict] = {}
        for item in res_pos + res_neg:
            resid = item["resid"]
            if resid not in combined:
                combined[resid] = item
            else:
                combined[resid]["occupancy"] = max(combined[resid]["occupancy"], item["occupancy"])
        return list(combined.values())

    def _pi_stacking_occupancy(self, mda, distance_array) -> List[Dict]:
        if not self.config.ligand_ring_sel:
            logger.warning("Pi-stacking requested but ligand_ring_sel is not set; skipping.")
            return []

        protein_sel = f"{self.config.protein_selection} and resname PHE TYR TRP HIS"
        ligand_sel = self.config.ligand_ring_sel
        return self._contact_occupancy(mda, distance_array, protein_sel, ligand_sel, self.config.pi_stack_cutoff)
