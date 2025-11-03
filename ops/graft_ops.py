from __future__ import annotations

"""
General grafting orchestrator (alternative module to avoid file corruption on graft.py).

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
from typing import Dict, Iterable, List, Optional, Tuple, Any
from time import perf_counter

import numpy as np
import pandas as pd

# Optional SciPy KD-tree for neighbor acceleration
try:
    from scipy.spatial import cKDTree  # type: ignore
    _HAVE_SCIPY = True
except Exception:  # pragma: no cover
    _HAVE_SCIPY = False

from usm.core.model import USM
from usm.ops.renumber import renumber_atoms
from usm.ops.merge import merge_structures, merge_preserving_first
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
from usm.ops.transform import rotation_matrix_from_axis_angle


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
    neighbor_radius: Optional[float] = None  # if set, prune host to neighbors within radius of site.o_pos
    tilt_scan_deg: Optional[List[float]] = None  # optional scan over tilt degrees; None -> use [tilt_deg]
    tilt_both_signs: bool = False        # if True, also try negative tilt; default False for baseline parity
    lift_angstrom: float = 0.0           # optional outward lift applied to non-anchor atoms along surface normal
    lift_scan_angstrom: Optional[List[float]] = None  # optional scan over outward lift distances; None -> use [lift_angstrom]


@dataclass
class PlacementRecord:
    site_id: int
    side: str
    accepted: bool
    torsion_deg: Optional[float]
    tilt_deg: Optional[float]
    lift_angstrom: Optional[float]
    min_distance: Optional[float]
    reason: str
    o_pos: Tuple[float, float, float]


@dataclass
class GraftResult:
    usm: USM
    placements: List[PlacementRecord]
    metrics: Optional[Dict[str, Any]] = None

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
                                torsion_deg: float,
                                tilt_override: Optional[float] = None,
                                lift_override: Optional[float] = None) -> USM:
    """
    Apply torsion and tilt about the anchor point, then translate anchor to site.o_pos.
    If tilt_override is provided, use that tilt in degrees instead of config.tilt_deg.
    Applies outward lift (if any) to non-anchor atoms along the surface normal, keeping anchor pinned.
    """
    anchor = config.anchor_name
    p_anchor = _guest_anchor_position(guest_aligned, anchor)
    plane_axis = outward_plane_axis(normal_dir) if config.plane_axis_mode == "auto" else outward_plane_axis(normal_dir)
    tilt_val = float(config.tilt_deg) if tilt_override is None else float(tilt_override)
    g2 = apply_torsion_and_tilt(guest_aligned, anchor_point=p_anchor, normal_dir=normal_dir,
                                torsion_deg=float(torsion_deg), tilt_deg=tilt_val,
                                plane_axis=plane_axis, in_place=False)

    # Apply outward lift to non-anchor atoms while keeping anchor pinned
    eff_lift = float(config.lift_angstrom) if (lift_override is None) else float(lift_override)
    if eff_lift > 1e-9:
        v = np.asarray(normal_dir, dtype=float)
        n = v / (np.linalg.norm(v) + 1e-12)
        shift = n * eff_lift
        # compute anchor row index
        arow = g2.atoms[g2.atoms["name"].astype("string") == str(anchor)]
        if not arow.empty:
            idx = int(arow.index[0])
            coords = g2.atoms[["x", "y", "z"]].to_numpy(dtype=float)
            coords = coords + shift[None, :]
            coords[idx, :] = coords[idx, :] - shift
            g2.atoms.loc[:, ["x", "y", "z"]] = coords

    g3 = translate_to_point(g2, from_point=p_anchor, to_point=np.array(site.o_pos, dtype=float), in_place=False)
    return g3


def _transform_coords_for_angles(coords_aligned: np.ndarray,
                                 anchor_point: np.ndarray,
                                 normal_dir: np.ndarray,
                                 torsion_deg: float,
                                 tilt_deg: float,
                                 plane_axis: np.ndarray,
                                 to_point: np.ndarray,
                                 lift_angstrom: float = 0.0) -> np.ndarray:
    """
    Fast coordinate-space transform for a pre-aligned guest:
    - Rotate about 'normal_dir' by torsion_deg around 'anchor_point'
    - Rotate about 'plane_axis' by tilt_deg around 'anchor_point'
    - Apply outward lift to non-anchor atoms along 'normal_dir' if lift_angstrom > 0
    - Translate so anchor maps to 'to_point'
    Returns a new (N,3) ndarray.
    """
    C0 = np.asarray(coords_aligned, dtype=float)
    p = np.asarray(anchor_point, dtype=float).reshape(3,)
    n = np.asarray(normal_dir, dtype=float).reshape(3,)
    a = np.asarray(plane_axis, dtype=float).reshape(3,)
    t = np.asarray(to_point, dtype=float).reshape(3,)

    # Center at anchor
    X = C0 - p[None, :]

    # Torsion about normal
    R_tor = rotation_matrix_from_axis_angle(n, float(torsion_deg))
    X = (R_tor @ X.T).T

    # Tilt about plane axis
    if abs(float(tilt_deg)) > 1e-12:
        R_tilt = rotation_matrix_from_axis_angle(a, float(tilt_deg))
        X = (R_tilt @ X.T).T

    # Optional outward lift while keeping anchor at origin
    if float(lift_angstrom) > 1e-9:
        nn = n / (np.linalg.norm(n) + 1e-12)
        shift = nn * float(lift_angstrom)
        # Anchor is the row closest to origin after centering/rotations
        norms = np.linalg.norm(X, axis=1)
        anchor_idx = int(np.argmin(norms)) if norms.size > 0 else 0
        X = X + shift[None, :]
        X[anchor_idx, :] = X[anchor_idx, :] - shift

    # Translate to target anchor position
    return X + t[None, :]


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
    t_graft_start = perf_counter()
    time_transform = 0.0
    time_neighbor_prep = 0.0
    time_distance = 0.0
    time_merge = 0.0
    angles_total = 0

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
        normal_dir = _normal_for_side(site.side)

        # Host without the atoms we intend to remove for this site (O and H of the old OH)
        keep_mask = _host_mask_excluding_aids(out, site.removal_aids)
        # Restrict collision host to slab atoms only (exclude previously placed guests)
        # Assumes slab label is 'XXXX'; fallback to full mask if none match.
        labels = out.atoms["mol_label"].astype("string")
        slab_mask = (labels == "XXXX").to_numpy(dtype=bool)
        combined_mask = keep_mask & slab_mask if slab_mask.any() else keep_mask

        t0_np = perf_counter()
        host_xyz = extract_xyz(out, mask=combined_mask)

        # Optional neighbor pruning: restrict to atoms within radius of the site anchor
        if config.neighbor_radius is not None:
            center = np.array(site.o_pos, dtype=float)
            r2 = float(config.neighbor_radius) ** 2
            dif = host_xyz - center[None, :]
            d2 = np.sum(dif * dif, axis=1)
            mloc = d2 <= r2
            if np.any(mloc):
                host_xyz = host_xyz[mloc]
            # Fallback: if no neighbors in radius (rare), keep full set to avoid false +inf distances

        # Build KD-tree once per site (optional)
        tree = cKDTree(host_xyz) if ('cKDTree' in globals() and _HAVE_SCIPY and host_xyz.size > 0) else None

        # Precompute aligned guest coordinates and rotation axes
        p_anchor = _guest_anchor_position(guest_aligned, config.anchor_name)
        plane_axis = outward_plane_axis(normal_dir) if config.plane_axis_mode == "auto" else outward_plane_axis(normal_dir)
        guest_coords_aligned = guest_aligned.atoms[["x", "y", "z"]].to_numpy(dtype=float)

        time_neighbor_prep += (perf_counter() - t0_np)

        accepted = False
        best_md = -float("inf")
        best_t: Optional[float] = None
        best_tilt: Optional[float] = None
        best_lift: Optional[float] = None

        # Determine scan lists
        tilt_list = list(config.tilt_scan_deg) if config.tilt_scan_deg is not None else [float(config.tilt_deg)]
        lifts_list = list(config.lift_scan_angstrom) if config.lift_scan_angstrom is not None else [float(config.lift_angstrom)]

        for tilt_val in tilt_list:
            # Optionally try both signs (disabled by default for baseline parity)
            tilts_to_try = [float(tilt_val)]
            if bool(config.tilt_both_signs) and abs(float(tilt_val)) > 1e-12:
                tilts_to_try.append(-float(tilt_val))
            for tilt_signed in tilts_to_try:
                for lift_val in lifts_list:
                    for t in torsions:
                        t0_tr = perf_counter()
                        coords = _transform_coords_for_angles(
                            guest_coords_aligned,
                            p_anchor,
                            normal_dir,
                            float(t),
                            float(tilt_signed),
                            plane_axis,
                            np.array(site.o_pos, dtype=float),
                            lift_angstrom=float(lift_val),
                        )
                        time_transform += (perf_counter() - t0_tr)

                        t0_d = perf_counter()
                        # Exclude anchor atom from clash check (anchor O is expected to be very close to slab)
                        to_pt = np.asarray(site.o_pos, dtype=float).reshape(3,)
                        d2g = np.sum((coords - to_pt[None, :]) ** 2, axis=1)
                        aidx = int(np.argmin(d2g)) if d2g.size > 0 else -1
                        if aidx >= 0 and coords.shape[0] > 1:
                            coords_wo_anchor = np.concatenate([coords[:aidx, :], coords[aidx + 1:, :]], axis=0)
                        else:
                            coords_wo_anchor = coords
    
                        if 'tree' in locals() and (tree is not None):
                            if coords_wo_anchor.size == 0:
                                md = float(np.inf)
                            else:
                                dists, _ = tree.query(coords_wo_anchor, k=1, workers=-1)
                                md = float(np.min(dists)) if dists.size > 0 else float(np.inf)
                        else:
                            md = min_distance_between_sets(coords_wo_anchor, host_xyz)
                        time_distance += (perf_counter() - t0_d)
                        angles_total += 1

                        # Track best min distance
                        if np.isfinite(md) and md > best_md:
                            best_md = float(md)
                            best_t = float(t)
                            best_tilt = float(tilt_signed)
                            best_lift = float(lift_val)
                        # Accept if above cutoff
                        if md >= float(config.clash_cutoff):
                            accepted = True
                            best_t = float(t)
                            best_tilt = float(tilt_signed)
                            best_lift = float(lift_val)
                            break
                    if accepted:
                        break
                if accepted:
                    break
            if accepted:
                break

        if accepted and (best_t is not None):
            t0_m = perf_counter()
            # Materialize accepted guest as USM only once (heavy op)
            best_clone = _transform_guest_for_angles(
                guest_aligned, site, config, normal_dir, torsion_deg=float(best_t), tilt_override=best_tilt, lift_override=best_lift
            )
            # Remove site atoms from host
            mask_after = _host_mask_excluding_aids(out, site.removal_aids)
            out = select_by_mask(out, mask_after)
            # Assign a unique mol_index to the accepted guest and merge
            placed = assign_unique_mol_index(best_clone, out, preferred_label=config.preferred_label, in_place=False)
            # Preserve host AIDs so subsequent site.removal_aids remain valid
            out = merge_preserving_first(out, placed)
            time_merge += (perf_counter() - t0_m)
            placements.append(
                PlacementRecord(
                    site_id=int(site.site_id),
                    side=site.side,
                    accepted=True,
                    torsion_deg=float(best_t) if best_t is not None else None,
                    tilt_deg=float(best_tilt) if best_tilt is not None else float(config.tilt_deg),
                    lift_angstrom=float(best_lift) if best_lift is not None else float(config.lift_angstrom),
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
                    lift_angstrom=None,
                    min_distance=None if not np.isfinite(best_md) else float(best_md),
                    reason=f"no_orientation_met_cutoff_{config.clash_cutoff:.3f}",
                    o_pos=tuple(site.o_pos),
                )
            )

    # Final renumber after all merges to keep AIDs stable during site processing
    out = renumber_atoms(out, order_by=["mol_index", "name"], in_place=False)

    metrics: Dict[str, Any] = {
        "graft_total_sec": float(perf_counter() - t_graft_start),
        "n_sites": int(len(sites)),
        "angles_total": int(angles_total),
        "angles_per_site_avg": (float(angles_total) / len(sites)) if len(sites) > 0 else 0.0,
        "time_transform_sec": float(time_transform),
        "time_neighbor_prep_sec": float(time_neighbor_prep),
        "time_distance_sec": float(time_distance),
        "time_merge_sec": float(time_merge),
    }
    return GraftResult(usm=out, placements=placements, metrics=metrics)


__all__ = [
    "PlacementConfig",
    "PlacementRecord",
    "GraftResult",
    "prepare_guest",
    "graft_guests",
]