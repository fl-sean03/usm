from __future__ import annotations

"""
Molecule labeling and indexing utilities.

These helpers manage (mol_label, mol_index) assignments so that each cloned
guest molecule receives a unique mol_index within its label namespace.
"""

from typing import Optional, Iterable, Tuple
import numpy as np
import pandas as pd

from usm.core.model import USM


def next_available_mol_index(usm: USM, mol_label: Optional[str] = None) -> int:
    """
    Return the next available mol_index for the given mol_label within the USM.

    - If mol_label is None, determine the most common label and advance its index.
    - If no matching atoms exist, returns 1.
    """
    a = usm.atoms
    if "mol_label" not in a.columns or "mol_index" not in a.columns:
        return 1

    label = mol_label
    if label is None:
        # Pick the most frequent label or default to "XXXX"
        counts = a["mol_label"].astype("string").value_counts()
        label = str(counts.index[0]) if not counts.empty else "XXXX"

    rows = a[a["mol_label"].astype("string") == str(label)]
    if rows.empty:
        return 1
    # mol_index may contain NA
    mis = pd.to_numeric(rows["mol_index"], errors="coerce").dropna()
    if mis.empty:
        return 1
    return int(np.max(mis.to_numpy())) + 1


def assign_molecule_identity(usm: USM, mol_label: Optional[str], mol_index: int, in_place: bool = False) -> USM:
    """
    Assign 'mol_label' and 'mol_index' to all atoms in the given USM.

    Returns a copy unless in_place=True.
    """
    out = usm if in_place else usm.copy()
    if mol_label is not None:
        out.atoms.loc[:, "mol_label"] = str(mol_label)
    out.atoms.loc[:, "mol_index"] = int(mol_index)
    return out


def assign_unique_mol_index(guest: USM, host: USM, preferred_label: Optional[str] = None, in_place: bool = False) -> USM:
    """
    Assign a unique mol_index to 'guest' within the namespace of 'preferred_label'
    (or the guest's current label if None), based on indices used in 'host'.

    Steps:
    - Determine target label: preferred_label or the first non-null guest mol_label else "MOL"
    - Compute next_available_mol_index against the host for that label
    - Set all guest atoms to that (mol_label, mol_index)

    Returns a copy unless in_place=True.
    """
    # Determine the label to use
    g = guest.atoms
    label = preferred_label
    if label is None:
        # pick a representative guest label
        vals = g["mol_label"].dropna().astype("string")
        label = str(vals.iloc[0]) if not vals.empty else "MOL"

    next_idx = next_available_mol_index(host, label)
    return assign_molecule_identity(guest, label, next_idx, in_place=in_place)


__all__ = [
    "next_available_mol_index",
    "assign_molecule_identity",
    "assign_unique_mol_index",
]