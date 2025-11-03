from __future__ import annotations

"""
General grafting orchestrator.

Provides:
- PlacementConfig: parameters controlling guest placement/orientation and clash threshold
- prepare_guest: drop specified atoms from guest template and renumber
- graft_guests: place guest clones at selected host sites with torsion grid and tilt, clash screening,
  unique mol_index assignment, and merging back into host

Assumptions:
- Host: typically a slab (e.g., MXN) with nonbonded OH terminations; bonds may be None
- Guest: full molecule with internal bonds (e.g., DOP), composed from CAR+MDF beforehand
- Sites: produced by usm.ops.sites (pairing O and H by distance), with 'o_pos' as anchor coordinate
"""

from dataclasses import dataclass, asdict
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

from usm.core.model import USM
from usm.ops.renumber import renumber_atoms
from usm.ops.merge import merge_structures
from usm.ops.select import select_by_mask
from usm.ops.frames import (
    find_aid_by_name,
    anchor_axis_from_neighbor,
    rotation_matrix_from_vectors,
    rotate_about_point,
    apply_torsion_and_tilt,
    translate_to_point,
    outward_plane_axis,
)
from usm.ops.collide import extract_xyz, min_distance_between_sets
from usm.ops.label import assign_unique_mol_index
from usm.ops.sites import Site


# ---------------------------
# Config and records
# ---------------------------

@dataclass
class PlacementConfig:
    """
    Parameters controlling guest placement at each site.
    """
    torsion_step_deg: float = 15.0       # grid step around surface normal (deg)
    tilt_deg: float = 20.0               # outward tilt after torsion (deg)
    clash_cutoff: float = 1.2            # accept if min guest-host distance >= cutoff (Å)
    torsion_start_deg: float = 0.0       # starting torsion angle (deg)
    full_turn_deg: float = 360.0         # torsion scan angular span (deg)
    max_trials: Optional[int] = None     # override number of torsion trials (None => computed from span/step)
    preferred_label: Optional[str] = None  # mol_label to assign for cloned guests (None => derive from guest)
    anchor_name: str = "O1"              # anchor atom in guest template
    plane_axis_mode: str = "auto"        # "auto" => outward_plane_axis(normal)


@dataclass
class PlacementRecord:
    site_id: int
    side: str
    accepted: bool
    torsion_deg: Optional[float]
    tilt_deg: Optional[float]
    min_distance: Optional[float]
    reason: str
    o_pos: Tuple[float, float, float]


@dataclass
class GraftResult:
    usm: USM
    placements: List[PlacementRecord]

    def to_manifest(self) -> List[Dict]:
        return [asdict(p) for p in self.placements]


# ---------------------------
# Helpers
# ---------------------------

def prepare_guest(guest: USM, drop_names: Iterable[str]) -> USM:
    """
    Drop specified atoms by 'name' from the guest and associated bonds; renumber aids.
    """
    names = {str(n) for n in drop_names}
    a = guest.atoms
    keep_mask = ~(a["name"].astype("string").isin(list(names)).fillna(False))
    trimmed = select_by_mask(guest, keep_mask)
    return renumber_atoms(trimmed, order_by=["mol_index", "name"], in_place=False)


def _normal_for_side(side: str) -> np.ndarray:
    """
    Default surface normal: +z for top, -z for bottom.
    """
    s = (side or "").lower()
    return np.array([0.0, 0.0, 1.0], dtype=float) if s == "top" else np.array([0.0, 0.0, -1.0], dtype=float)


def _host_mask_excluding_aids(host: USM, exclude_aids: Iterable[int]) -> np.ndarray:
    """
    Build a boolean mask over host.atoms marking atoms to keep (exclude these aids).
    """
    excl = set(int(x) for x in exclude_aids)
    aids = host.atoms["aid"].astype(int).to_numpy()
    return ~np.isin(aids, np.array(list(excl), dtype=int))


def _guest_anchor_position(guest: USM, anchor_name: str) -> np.ndarray:
    aid = find_aid_by_name(guest, anchor_name)
    row = guest.atoms[guest.atoms["aid"] == int(aid)]
    if row.empty:
        return np.array([0.0, 0.0, 0.0], dtype=float)
    return row[["x", "y", "z"]].to_numpy(dtype=float).reshape(3,)


def _orient_guest_template(guest_template: USM,
                           config: PlacementConfig,
                           site: Site,
                           normal_dir: Optional[np.ndarray] = None) -> USM:
    """
    Rotate guest (about its anchor) so the anchor axis aligns with provided surface normal.
    Returns a new USM; translation to site.o_pos happens after torsion/tilt.
    """
    g0 = guest_template.copy()
    anchor = config.anchor_name
    p_anchor = _guest_anchor_position(g0, anchor)
    v_from = anchor_axis_from_neighbor(g0, anchor)
    n = normal_dir if normal_dir is not None else _normal_for_side(site.side)
    R_align = rotation_matrix_from_vectors(v_from, n)
    g1 = rotate_about_point(g0, R_align, origin=p_anchor, in_place=False)
    return g1


def _transform_guest_for_angles(guest_aligned: USM,
                                site: Site,
                                config: PlacementConfig,
                                normal_dir: np.ndarray,
                                torsion_deg: float) -> USM:
    """
    Apply torsion and tilt about the anchor point, then translate anchor to site.o_pos.
    """
    anchor = config.anchor_name
    p_anchor = _guest_anchor_position(guest_aligned, anchor)
    plane_axis = outward_plane_axis(normal_dir) if config.plane_axis_mode == "auto" else outward_plane_axis(normal_dir)
    g2 = apply_torsion_and_tilt(guest_aligned, anchor_point=p_anchor, normal_dir=normal_dir,
                                torsion_deg=float(torsion_deg), tilt_deg=float(config.tilt_deg),
                                plane_axis=plane_axis, in_place=False)
    g3 = translate_to_point(g2, from_point=p_anchor, to_point=np.array(site.o_pos, dtype=float), in_place=False)
    return g3


def _scan_torsion_angles(config: PlacementConfig) -> List[float]:
    step = float(max(1e-6, config.torsion_step_deg))
    start = float(config.torsion_start_deg)
    full = float(config.full_turn_deg)
    n = int(np.ceil(full / step)) if config.max_trials is None else int(config.max_trials)
    return [start + i * step for i in range(n)]


# ---------------------------
# Main grafting loop
# ---------------------------

def graft_guests(host: USM,
                 guest_template: USM,
                 sites: List[Site],
                 config: PlacementConfig,
                 drop_guest_names: Optional[Iterable[str]] = None) -> GraftResult:
    """
    Place guest clones at selected sites with torsion scan and clash threshold.
    Workflow per site:
    - Optionally drop atoms from guest (e.g., H10) via prepare_guest once
    - Pre-align guest anchor axis to surface normal (per side cache)
    - For torsion grid, apply torsion+tilt, then translate anchor to site.o_pos
    - Measure min distance to host (excluding site O/H to be removed)
    - Accept first orientation with min_distance >= clash_cutoff (else record best and reject)
    - On accept: remove site O/H from host, assign unique mol_index, merge, renumber
    """
    out = host.copy()
    placements: List[PlacementRecord] = []

    # Prepare base guest once (apply drops)
    g_base = guest_template.copy()
    if drop_guest_names:
        g_base = prepare_guest(g_base, drop_guest_names)

    # Cache aligned template per side to avoid repeated alignments
    aligned_cache: Dict[str, USM] = {}
    torsions = _scan_torsion_angles(config)

    for site in sites:
        # Prepare aligned guest for this site's side
        if site.side not in aligned_cache:
            n_side = _normal_for_side(site.side)
            aligned_cache[site.side] = _orient_guest_template(g_base, config, site, normal_dir=n_side)
        guest_aligned = aligned_cache[site.side]

        # Host without the atoms we intend to remove for this site (O and H of the old OH)
        keep_mask = _host_mask_excluding_aids(out, site.removal_aids)
        host_xyz = extract_xyz(out, mask=keep_mask)

        accepted = False
        best_md = -float("inf")
        best_t: Optional[float] = None
        best_clone: Optional[USM] = None

        normal_dir = _normal_for_side(site.side)
        for t in torsions:
            clone = _transform_guest_for_angles(guest_aligned, site, config, normal_dir, torsion_deg=float(t))
            md = min_distance_between_sets(extract_xyz(clone), host_xyz)
            # Track best min distance
            if np.isfinite(md) and md > best_md:
                best_md = float(md)
                best_t = float(t)
                best_clone = clone
            # Accept if above cutoff
            if md >= float(config.clash_cutoff):
                accepted = True
                best_t = float(t)
                best_clone = clone
                break

        if accepted and best_clone is not None:
            # Remove site atoms from host
            mask_after = _host_mask_excluding_aids(out, site.removal_aids)
            out = select_by_mask(out, mask_after)
            # Assign a unique mol_index to the accepted guest and merge
            placed = assign_unique_mol_index(best_clone, out, preferred_label=config.preferred_label, in_place=False)
            out = merge_structures([out, placed])
            out = renumber_atoms(out, order_by=["mol_index", "name"], in_place=False)
            placements.append(
                PlacementRecord(
                    site_id=int(site.site_id),
                    side=site.side,
                    accepted=True,
                    torsion_deg=float(best_t) if best_t is not None else None,
                    tilt_deg=float(config.tilt_deg),
                    min_distance=float(best_md) if np.isfinite(best_md) else None,
                    reason="accepted",
                    o_pos=tuple(site.o_pos),
                )
            )
        else:
            placements.append(
                PlacementRecord(
                    site_id=int(site.site_id),
                    side=site.side,
                    accepted=False,
                    torsion_deg=None if best_t is None else float(best_t),
                    tilt_deg=float(config.tilt_deg),
                    min_distance=None if not np.isfinite(best_md) else float(best_md),
                    reason=f"no_orientation_met_cutoff_{config.clash_cutoff:.3f}",
                    o_pos=tuple(site.o_pos),
                )
            )

    return GraftResult(usm=out, placements=placements)


__all__ = [
    "PlacementConfig",
    "PlacementRecord",
    "GraftResult",
    "prepare_guest",
    "graft_guests",
]