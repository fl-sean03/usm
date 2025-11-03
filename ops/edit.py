from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any, Set

import numpy as np
import pandas as pd

from usm.core.model import USM
from usm.ops.renumber import renumber_atoms


@dataclass
class NeighborRemovalPolicy:
    """
    Policy for selecting which neighbor(s) to remove per replaced atom.
    - elements_priority: try these element symbols in order (e.g., ["H", "D"])
    - fallback_lowest_degree: if no neighbor matches priority, optionally remove the lowest-degree neighbor
    - max_remove_per_atom: remove at most this many neighbors per replaced atom (default 1)
    """
    elements_priority: List[str]
    fallback_lowest_degree: bool = True
    max_remove_per_atom: int = 1


def _assign_layers_by_bin(z: np.ndarray, dz: float) -> np.ndarray:
    """
    Assign integer layer ids by rounding z/dz to nearest integer.
    Deterministic and stable for a fixed dz.
    """
    if dz is None or dz <= 0:
        raise ValueError("dz must be > 0 for group_mode='bin'")
    z = np.asarray(z, dtype=float)
    z = np.where(np.isfinite(z), z, 0.0)
    # round-half-up for determinism (avoid bankers rounding)
    return np.floor(z / dz + 0.5).astype(np.int64)


def _assign_layers_by_eps(z: np.ndarray, eps: float) -> np.ndarray:
    """
    Cluster sorted z values: new layer whenever delta > eps.
    Returns an array of integer layer ids aligned to the original z order.
    """
    if eps is None or eps <= 0:
        raise ValueError("eps must be > 0 for group_mode='eps'")
    z = np.asarray(z, dtype=float)
    z = np.where(np.isfinite(z), z, 0.0)
    n = len(z)
    if n == 0:
        return np.array([], dtype=np.int64)
    order = np.argsort(z, kind="mergesort")
    inv = np.empty(n, dtype=np.int64)
    layer_sorted = np.zeros(n, dtype=np.int64)
    current_layer = 0
    prev = z[order[0]]
    layer_sorted[0] = current_layer
    for i in range(1, n):
        zi = z[order[i]]
        if abs(zi - prev) > eps:
            current_layer += 1
        layer_sorted[i] = current_layer
        prev = zi
    # invert to original order
    inv[order] = np.arange(n, dtype=np.int64)
    # build per-original-index layer
    layer_per_idx = np.empty(n, dtype=np.int64)
    layer_per_idx[order] = layer_sorted
    return layer_per_idx


def _compute_degrees(bonds: pd.DataFrame) -> Dict[int, int]:
    """Return degree (bond count) per atom id."""
    deg = {}
    if bonds is None or len(bonds) == 0:
        return deg
    a1_counts = bonds["a1"].value_counts(dropna=True)
    a2_counts = bonds["a2"].value_counts(dropna=True)
    for aid, cnt in a1_counts.items():
        if pd.isna(aid):
            continue
        deg[int(aid)] = deg.get(int(aid), 0) + int(cnt)
    for aid, cnt in a2_counts.items():
        if pd.isna(aid):
            continue
        deg[int(aid)] = deg.get(int(aid), 0) + int(cnt)
    return deg


def _neighbors_of(aid: int, bonds: pd.DataFrame) -> List[int]:
    """List neighbor aids of a given atom id."""
    if bonds is None or len(bonds) == 0:
        return []
    hits = bonds[(bonds["a1"] == aid) | (bonds["a2"] == aid)]
    if hits.empty:
        return []
    nbs = []
    for _, r in hits.iterrows():
        a1 = int(r["a1"])
        a2 = int(r["a2"])
        nbs.append(a2 if a1 == aid else a1)
    # unique neighbors deterministically sorted
    return sorted(set(nbs))


def replace_atom_types_and_remove_neighbors(
    usm: USM,
    target_types: List[str],
    replacement: Dict[str, str],
    group_mode: str = "bin",
    dz: float = 1.0,
    eps: Optional[float] = None,
    per_layer: Optional[int] = None,
    neighbor_policy: Optional[NeighborRemovalPolicy] = None,
    in_place: bool = False,
) -> USM:
    """
    Replace atom_type for selected atoms grouped by z, and remove one (or more) connected neighbor(s) per replaced atom.

    Parameters
    ----------
    usm : USM
        Structure with bonds populated (required).
    target_types : List[str]
        Atom types to search for and consider for replacement.
    replacement : Dict[str, str]
        Mapping from old_type -> new_type for replaced atoms. Types not present in this map remain unchanged.
    group_mode : str
        "bin" (use dz bin width) or "eps" (cluster consecutive z within eps).
    dz : float
        Bin width for group_mode="bin".
    eps : Optional[float]
        Epsilon for clustering when group_mode="eps".
    per_layer : Optional[int]
        If provided, limit to replacing at most this many target atoms per layer (chosen deterministically).
        If None, replace all targets in each layer.
    neighbor_policy : Optional[NeighborRemovalPolicy]
        Policy for choosing which neighbor(s) to remove per replaced atom. Defaults to H-priority, 1 neighbor.
    in_place : bool
        If True, mutate input USM; otherwise return a copy.

    Returns
    -------
    USM
        Updated structure with atom_type changes applied, selected neighbor atoms removed, bonds cleaned up, and IDs renumbered.
    """
    if usm.bonds is None or len(usm.bonds) == 0:
        raise ValueError("replace_atom_types_and_remove_neighbors requires bonds; compose MDF bonds onto CAR first.")

    policy = neighbor_policy or NeighborRemovalPolicy(elements_priority=["H"], fallback_lowest_degree=True, max_remove_per_atom=1)

    out = usm if in_place else usm.copy()
    atoms = out.atoms.copy()
    bonds = out.bonds.copy()

    # Build layer ids
    z = atoms["z"].to_numpy(dtype=float, copy=True)
    if group_mode == "bin":
        layers = _assign_layers_by_bin(z, dz=dz)
    elif group_mode == "eps":
        layers = _assign_layers_by_eps(z, eps=eps if eps is not None else 1e-3)
    else:
        raise ValueError("group_mode must be 'bin' or 'eps'")

    atoms["__layer_id"] = layers

    # Targets: atom_type in target_types
    # use string compare robustly
    a_types = atoms["atom_type"].astype("string")
    is_target = a_types.isin(pd.Series(target_types, dtype="string"))
    target_df = atoms[is_target].copy()

    if target_df.empty:
        # nothing to do; just return copy with clean-up of temp column
        out.atoms = atoms.drop(columns=["__layer_id"])
        return out

    # Deterministic selection per layer: sort by (layer_id, mol_index, name, aid)
    # name may be NA; handle by fillna
    target_df["__name_key"] = target_df["name"].fillna("").astype(str)
    target_df.sort_values(by=["__layer_id", "mol_index", "__name_key", "aid"], kind="mergesort", inplace=True)

    # Build list of aids to change respecting per_layer
    aids_to_change: List[int] = []
    for layer_id, grp in target_df.groupby("__layer_id", sort=True):
        if per_layer is None or per_layer >= len(grp):
            aids_to_change.extend(grp["aid"].astype(int).tolist())
        else:
            aids_to_change.extend(grp["aid"].astype(int).head(int(per_layer)).tolist())

    aids_to_change = sorted(set(aids_to_change))

    # Apply atom_type replacement
    old_types = atoms.set_index("aid")["atom_type"].to_dict()
    for aid in aids_to_change:
        old_t = str(old_types.get(int(aid), ""))
        new_t = replacement.get(old_t, old_t)
        # set value
        atoms.loc[atoms["aid"] == aid, "atom_type"] = new_t

    # Neighbor removal selection
    degrees = _compute_degrees(bonds)
    # map aid -> element
    elem_map: Dict[int, str] = {int(r["aid"]): ("" if pd.isna(r.get("element")) else str(r.get("element"))) for _, r in atoms[["aid", "element"]].iterrows()}

    to_remove: Set[int] = set()
    for aid in aids_to_change:
        chosen: List[int] = []
        nbs = _neighbors_of(int(aid), bonds)
        if not nbs:
            continue

        # First pass: priority elements
        if policy.elements_priority:
            for el in policy.elements_priority:
                # candidates with this element
                cand = [nb for nb in nbs if elem_map.get(int(nb), "") == el]
                if cand:
                    # choose deterministically by (degree, aid)
                    cand_sorted = sorted(cand, key=lambda x: (degrees.get(int(x), 0), int(x)))
                    chosen.append(cand_sorted[0])
                    if len(chosen) >= policy.max_remove_per_atom:
                        break

        # Fallback: lowest degree neighbor (if allowed), excluding already chosen
        if len(chosen) < policy.max_remove_per_atom and policy.fallback_lowest_degree:
            remaining = [nb for nb in nbs if nb not in chosen]
            if remaining:
                rem_sorted = sorted(remaining, key=lambda x: (degrees.get(int(x), 0), int(x)))
                # fill remaining slots
                for nb in rem_sorted:
                    chosen.append(nb)
                    if len(chosen) >= policy.max_remove_per_atom:
                        break

        # Register for deletion; never delete the replaced atom itself
        for nb in chosen:
            if int(nb) != int(aid):
                to_remove.add(int(nb))

    if to_remove:
        # Drop atoms in to_remove
        keep_mask = ~atoms["aid"].isin(list(to_remove))
        atoms = atoms.loc[keep_mask].reset_index(drop=True)

        # Drop bonds that reference removed aids
        bonds = bonds[~(bonds["a1"].isin(to_remove) | bonds["a2"].isin(to_remove))].reset_index(drop=True)

    # Clean temp column
    if "__layer_id" in atoms.columns:
        atoms.drop(columns=["__layer_id"], inplace=True, errors="ignore")
    if "__name_key" in atoms.columns:
        atoms.drop(columns=["__name_key"], inplace=True, errors="ignore")

    # Build updated USM and renumber deterministically
    out.atoms = atoms
    out.bonds = bonds

    # Add summary to provenance
    prov = dict(out.provenance or {})
    prov["edit_summary"] = {
        "targets_requested": list(target_types),
        "replaced_count": int(len(aids_to_change)),
        "removed_neighbors_count": int(len(to_remove)),
        "group_mode": group_mode,
        "dz": dz,
        "eps": eps,
        "per_layer": None if per_layer is None else int(per_layer),
        "neighbor_policy": {
            "elements_priority": policy.elements_priority,
            "fallback_lowest_degree": policy.fallback_lowest_degree,
            "max_remove_per_atom": policy.max_remove_per_atom,
        },
    }
    out.provenance = prov

    # Renumber to maintain dense aids and remap bonds
    out = renumber_atoms(out, order_by=["mol_index", "name"], in_place=True)
    return out