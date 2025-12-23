from __future__ import annotations

from typing import Iterable, List, Dict, Any, Optional
import numpy as np
import pandas as pd

from usm.core.model import USM


def _remap_bonds_for_new_aids(bonds: Optional[pd.DataFrame], aid_map: Dict[int, int]) -> Optional[pd.DataFrame]:
    if bonds is None or len(bonds) == 0:
        return None
    out = bonds.copy()
    out["a1"] = out["a1"].map(aid_map)
    out["a2"] = out["a2"].map(aid_map)
    out = out.dropna(subset=["a1", "a2"]).copy()
    # Normalize a1 < a2
    a1 = out["a1"].astype("int32").to_numpy()
    a2 = out["a2"].astype("int32").to_numpy()
    swap = a1 > a2
    if swap.any():
        tmp = a1[swap].copy()
        a1[swap] = a2[swap]
        a2[swap] = tmp
    out["a1"] = a1
    out["a2"] = a2
    if swap.any():
        for col in ["ix", "iy", "iz"]:
            if col in out.columns:
                out.loc[swap, col] = -out.loc[swap, col]
    return out.reset_index(drop=True)


def renumber_atoms(usm: USM, order_by: Optional[List[str]] = None, in_place: bool = False) -> USM:
    """
    Deterministically renumber atom IDs (aid) based on a stable sort.
    - order_by: list of column names to sort by before assigning new aids.
      Defaults to ["mol_index", "name"] when available, else current order.
    """
    out = usm if in_place else usm.copy()

    atoms = out.atoms.copy()
    if order_by is None:
        order_by = []
        if "mol_index" in atoms.columns:
            order_by.append("mol_index")
        if "name" in atoms.columns:
            order_by.append("name")

    if order_by:
        atoms = atoms.sort_values(by=order_by, kind="mergesort").reset_index(drop=True)
    else:
        atoms = atoms.reset_index(drop=True)

    old_aids = atoms["aid"].to_numpy().astype(int)
    new_aids = np.arange(len(atoms), dtype=np.int32)
    aid_map = {int(old): int(new) for old, new in zip(old_aids, new_aids)}
    atoms["aid"] = new_aids

    out.atoms = atoms
    out.bonds = _remap_bonds_for_new_aids(out.bonds, aid_map)
    # molecules table omitted in v0.1; if present, would need remap
    return out


def renumber_molecules(usm: USM, in_place: bool = False) -> USM:
    """
    Assign deterministic molecule indices (mid) by first appearance order of (mol_label, mol_index, mol_block_name).
    The molecule table is optional in v0.1; we annotate atoms with a derived 'mid' for convenience.
    """
    out = usm if in_place else usm.copy()
    a = out.atoms

    if not all(col in a.columns for col in ["mol_label", "mol_index", "mol_block_name"]):
        # Nothing to do
        return out

    keys = a[["mol_label", "mol_index", "mol_block_name"]].astype({"mol_label": "string", "mol_block_name": "string"}).copy()
    # Stable category codes give first-appearance order
    tuples = list(zip(keys["mol_label"].tolist(), keys["mol_index"].tolist(), keys["mol_block_name"].tolist()))
    first_index: Dict[tuple, int] = {}
    mids = np.empty(len(a), dtype=np.int32)
    next_id = 0
    for idx, t in enumerate(tuples):
        if t not in first_index:
            first_index[t] = next_id
            next_id += 1
        mids[idx] = first_index[t]
    out.atoms["mid"] = mids
    return out