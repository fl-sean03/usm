from __future__ import annotations

from typing import Iterable, Dict, Any, Optional
import numpy as np
import pandas as pd

from usm.core.model import USM


def _remap_bonds(bonds: Optional[pd.DataFrame], aid_map: Dict[int, int]) -> Optional[pd.DataFrame]:
    if bonds is None or len(bonds) == 0:
        return None
    # Keep bonds where both endpoints are retained
    keep_mask = bonds["a1"].isin(aid_map) & bonds["a2"].isin(aid_map)
    if not keep_mask.any():
        return None
    new_bonds = bonds.loc[keep_mask].copy()
    new_bonds["a1"] = new_bonds["a1"].map(aid_map).astype("int32")
    new_bonds["a2"] = new_bonds["a2"].map(aid_map).astype("int32")
    # Normalize a1<a2 after remap
    a1 = new_bonds["a1"].to_numpy()
    a2 = new_bonds["a2"].to_numpy()
    swap = a1 > a2
    if swap.any():
        new_bonds.loc[swap, ["a1", "a2"]] = new_bonds.loc[swap, ["a2", "a1"]].to_numpy()
    # bid is re-assigned in USM __post_init__
    return new_bonds.reset_index(drop=True)


def select_by_mask(usm: USM, mask: pd.Series) -> USM:
    """
    Generic selection by boolean mask over atoms rows.
    Returns a new USM with reindexed aids and bonds remapped to new aids.
    """
    if mask.dtype != bool:
        mask = mask.astype(bool)
    new_atoms = usm.atoms.loc[mask].copy().reset_index(drop=True)
    # Build old_aid -> new_aid mapping
    old_aids = usm.atoms.loc[mask, "aid"].to_numpy()
    new_aids = np.arange(len(new_atoms), dtype=np.int32)
    aid_map = {int(old): int(new) for old, new in zip(old_aids, new_aids)}
    # Remap bonds
    new_bonds = _remap_bonds(usm.bonds, aid_map)
    # Molecules not recomputed in v0.1 (optional)
    return USM(
        atoms=new_atoms,
        bonds=new_bonds,
        molecules=None if usm.molecules is None else usm.molecules.copy(),
        cell=dict(usm.cell),
        provenance=dict(usm.provenance or {}),
        preserved_text=dict(usm.preserved_text or {}),
    )


def select_by_element(usm: USM, elements: Iterable[str]) -> USM:
    """
    Select atoms whose 'element' is in the provided iterable of element symbols.
    """
    elems = {str(e) for e in elements}
    mask = usm.atoms["element"].isin(elems).fillna(False)
    return select_by_mask(usm, mask)


def select_by_name(usm: USM, names: Iterable[str]) -> USM:
    """
    Select atoms whose 'name' is in the provided iterable of names.
    """
    ns = {str(n) for n in names}
    mask = usm.atoms["name"].isin(ns).fillna(False)
    return select_by_mask(usm, mask)


def select_by_molecule_index(usm: USM, mol_index: int) -> USM:
    """
    Select atoms within a specific mol_index (as parsed from CAR/MDF).
    """
    mask = (usm.atoms["mol_index"] == int(mol_index))
    return select_by_mask(usm, mask)

def select_box(usm: USM, xmin: float, xmax: float, ymin: float, ymax: float, zmin: float, zmax: float) -> USM:
    """
    Axis-aligned box selection [xmin,xmax] x [ymin,ymax] x [zmin,zmax].
    Inclusive on bounds. NaN coordinates are treated as out-of-box.
    """
    a = usm.atoms
    mask = (
        a["x"].between(xmin, xmax, inclusive="both").fillna(False)
        & a["y"].between(ymin, ymax, inclusive="both").fillna(False)
        & a["z"].between(zmin, zmax, inclusive="both").fillna(False)
    )
    return select_by_mask(usm, mask)


def select_within_radius(usm: USM, center: Iterable[float], radius: float) -> USM:
    """
    Spherical selection: keep atoms with ||r - center|| <= radius.
    NaN coordinates are treated as outside radius.
    """
    c = np.asarray(list(center), dtype=float).reshape(3,)
    coords = usm.atoms[["x", "y", "z"]].to_numpy(dtype=float)
    # Mark NaN rows as large distance
    nan_mask = ~np.isfinite(coords).all(axis=1)
    d2 = np.sum((coords - c[None, :]) ** 2, axis=1)
    d2[nan_mask] = np.inf
    mask = d2 <= float(radius) ** 2
    return select_by_mask(usm, pd.Series(mask, index=usm.atoms.index))