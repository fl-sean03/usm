from __future__ import annotations

from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import pandas as pd

from usm.core.model import USM


def _assert_cells_compatible(usms: List[USM], policy: str) -> Dict[str, Any]:
    """
    Reconcile cell metadata across multiple USMs.
    policy: "strict" (all equal or NaN treated equal), "first" (take first), "error" (raise on mismatch)
    """
    policy = (policy or "strict").lower()
    first = usms[0].cell if usms else {}
    if policy == "first":
        return dict(first)

    def cell_tuple(cell: Dict[str, Any]) -> Tuple:
        return (
            bool(cell.get("pbc", False)),
            float(cell.get("a", np.nan)),
            float(cell.get("b", np.nan)),
            float(cell.get("c", np.nan)),
            float(cell.get("alpha", np.nan)),
            float(cell.get("beta", np.nan)),
            float(cell.get("gamma", np.nan)),
            str(cell.get("spacegroup", "")),
        )

    if policy in ("strict", "error"):
        base = cell_tuple(first)
        for u in usms[1:]:
            tup = cell_tuple(u.cell)
            # Consider NaNs equal in numeric fields
            eq = True
            for x, y in zip(base, tup):
                if isinstance(x, float) and isinstance(y, float):
                    if (np.isnan(x) and np.isnan(y)) or (x == y):
                        continue
                    eq = False
                    break
                else:
                    if x != y:
                        eq = False
                        break
            if not eq:
                if policy == "error":
                    raise ValueError("Incompatible cell metadata across inputs")
                else:
                    # strict but mismatch -> prefer first and note provenance (handled by caller)
                    return dict(first)
        return dict(first)

    raise ValueError(f"Unknown cell reconcile policy: {policy}")


def merge_structures(usms: List[USM], cell_policy: str = "strict") -> USM:
    """
    Merge multiple USM structures into one.
    - Concatenates atoms deterministically in the order of input USMs and original aid order.
    - Remaps bonds endpoints with stable new aids.
    - Cell metadata reconciled via cell_policy: "strict"|"first"|"error".
    - Preserved text and provenance merged conservatively (first wins, notes appended).
    """
    if not usms:
        raise ValueError("No USMs provided to merge")

    # Prepare new atoms
    atoms_parts: List[pd.DataFrame] = []
    bonds_parts: List[pd.DataFrame] = []
    notes: List[str] = []

    running_offset = 0
    for idx, u in enumerate(usms):
        a = u.atoms.sort_values("aid").reset_index(drop=True).copy()
        # Keep a mapping from old aid -> new aid
        old_aids = a["aid"].to_numpy().astype(int)
        new_aids = np.arange(running_offset, running_offset + len(a), dtype=np.int32)
        aid_map = {int(old): int(new) for old, new in zip(old_aids, new_aids)}
        a["aid"] = new_aids
        atoms_parts.append(a)

        if u.bonds is not None and len(u.bonds) > 0:
            b = u.bonds.copy()
            # Map endpoints
            b["a1"] = b["a1"].map(aid_map)
            b["a2"] = b["a2"].map(aid_map)
            # Drop bonds referencing dropped atoms (shouldn't happen)
            b = b.dropna(subset=["a1", "a2"]).copy()
            # Normalize order of endpoints
            a1 = b["a1"].astype("int32").to_numpy()
            a2 = b["a2"].astype("int32").to_numpy()
            swap = a1 > a2
            if swap.any():
                tmp = a1[swap].copy()
                a1[swap] = a2[swap]
                a2[swap] = tmp
            b["a1"] = a1
            b["a2"] = a2
            bonds_parts.append(b)

        running_offset += len(a)

    out_atoms = pd.concat(atoms_parts, ignore_index=True)
    out_bonds = pd.concat(bonds_parts, ignore_index=True) if bonds_parts else None

    # Reconcile cell
    out_cell = _assert_cells_compatible(usms, cell_policy)
    if cell_policy == "strict":
        # If there was a mismatch but we didn't error, leave a parse note
        pass

    # Merge provenance and preserved_text conservatively: prefer first, append notes from others
    provenance = dict(usms[0].provenance or {})
    preserved_text = dict(usms[0].preserved_text or {})

    for u in usms[1:]:
        if u.provenance and u.provenance != provenance:
            notes.append(f"merged provenance from {u.provenance.get('source_path','unknown')}")
        if u.preserved_text and u.preserved_text != preserved_text:
            notes.append("merged preserved_text from additional inputs")

    if notes:
        prov_notes = provenance.get("parse_notes", "")
        provenance["parse_notes"] = (prov_notes + " | " if prov_notes else "") + "; ".join(notes)

    return USM(
        atoms=out_atoms,
        bonds=out_bonds,
        molecules=None,  # molecule table optional in v0.1
        cell=out_cell,
        provenance=provenance,
        preserved_text=preserved_text,
    )

def merge_preserving_first(usm_first: USM, usm_second: USM, cell_policy: str = "strict") -> USM:
    """
    Merge two USMs while preserving all AIDs of the first structure unchanged.
    The second structure's AIDs are remapped to a fresh contiguous range that
    starts after the max AID present in the first.

    - Keeps ordering deterministic: second's atoms are sorted by original aid before remap.
    - Bonds of the second are remapped according to the new AIDs; first bonds unchanged.
    - Cell metadata is reconciled with the provided policy (default: "strict").
    - Provenance/preserved_text merged conservatively (first wins; notes appended).

    This is useful for iterative workflows that cache AIDs of the base/host structure
    (e.g., site.removal_aids), avoiding invalidation after each merge.
    """
    # Prepare atoms
    a1 = usm_first.atoms.copy()
    a2 = usm_second.atoms.copy()

    # Determine starting AID for the second
    next_start = 0 if a1.empty else int(np.max(a1["aid"].astype(int).to_numpy())) + 1

    # Deterministic remap for second: sort by original aid, then assign sequential new aids
    a2_sorted = a2.sort_values("aid").reset_index(drop=True).copy()
    old2 = a2_sorted["aid"].astype(int).to_numpy()
    new2 = np.arange(next_start, next_start + len(a2_sorted), dtype=np.int32)
    map2 = {int(o): int(n) for o, n in zip(old2, new2)}
    a2_sorted.loc[:, "aid"] = new2

    # Bonds
    b1 = usm_first.bonds.copy() if (usm_first.bonds is not None and len(usm_first.bonds) > 0) else None
    b2 = usm_second.bonds.copy() if (usm_second.bonds is not None and len(usm_second.bonds) > 0) else None

    if b2 is not None and len(b2) > 0:
        b2 = b2.copy()
        b2["a1"] = b2["a1"].map(map2)
        b2["a2"] = b2["a2"].map(map2)
        b2 = b2.dropna(subset=["a1", "a2"]).copy()
        # Normalize order of endpoints
        a1c = b2["a1"].astype("int32").to_numpy()
        a2c = b2["a2"].astype("int32").to_numpy()
        swap = a1c > a2c
        if swap.any():
            tmp = a1c[swap].copy()
            a1c[swap] = a2c[swap]
            a2c[swap] = tmp
        b2["a1"] = a1c
        b2["a2"] = a2c

    # Concatenate atoms (first unchanged)
    out_atoms = pd.concat(
        [a1.sort_values("aid").reset_index(drop=True), a2_sorted],
        ignore_index=True
    )

    # Concatenate bonds
    if b1 is not None and b2 is not None:
        out_bonds = pd.concat([b1, b2], ignore_index=True)
    elif b1 is not None:
        out_bonds = b1
    else:
        out_bonds = b2  # may be None

    # Reconcile cell
    out_cell = _assert_cells_compatible([usm_first, usm_second], cell_policy)

    # Merge provenance and preserved_text similar to merge_structures
    provenance = dict(usm_first.provenance or {})
    preserved_text = dict(usm_first.preserved_text or {})
    notes = []

    if usm_second.provenance and usm_second.provenance != provenance:
        notes.append(f"merged provenance from {usm_second.provenance.get('source_path','unknown')}")
    if usm_second.preserved_text and usm_second.preserved_text != preserved_text:
        notes.append("merged preserved_text from additional inputs")

    if notes:
        prov_notes = provenance.get("parse_notes", "")
        provenance["parse_notes"] = (prov_notes + " | " if prov_notes else "") + "; ".join(notes)

    return USM(
        atoms=out_atoms,
        bonds=out_bonds,
        molecules=None,
        cell=out_cell,
        provenance=provenance,
        preserved_text=preserved_text,
    )