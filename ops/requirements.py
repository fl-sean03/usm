from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Union


from usm.ops._canonicalize import (
    _canonicalize_angle_key,
    _canonicalize_bond_key,
    _extract_atom_types_by_aid,
)


def _normalized_unique_bond_pairs(structure: Any, *, n_atoms: int) -> list[tuple[int, int]]:
    bonds = getattr(structure, "bonds", None)
    if bonds is None or len(bonds) == 0:
        return []

    if "a1" not in bonds.columns or "a2" not in bonds.columns:
        raise ValueError("structure.bonds: requires columns 'a1' and 'a2'")

    a1s = bonds["a1"].tolist()
    a2s = bonds["a2"].tolist()

    pairs: set[tuple[int, int]] = set()
    for i, (a1, a2) in enumerate(zip(a1s, a2s)):
        if a1 is None or a2 is None:
            raise ValueError(f"structure.bonds[{i}]: a1/a2 must be ints, got null")
        try:
            x = int(a1)
            y = int(a2)
        except Exception as e:
            raise ValueError(
                f"structure.bonds[{i}]: a1/a2 must be ints, got {type(a1).__name__}/{type(a2).__name__}"
            ) from e

        if x < 0 or x >= n_atoms or y < 0 or y >= n_atoms:
            raise ValueError(
                f"structure.bonds[{i}]: a1/a2 out of range: ({x},{y}) (n_atoms={n_atoms})"
            )
        if x == y:
            raise ValueError(f"structure.bonds[{i}]: self-bond not allowed (aid={x})")

        a, b = (x, y) if x <= y else (y, x)
        pairs.add((a, b))

    return sorted(pairs)


def derive_requirements_v0_1(structure: Any) -> dict[str, Any]:
    """Derive v0.1 Requirements JSON deterministically from a USM-like structure.

    Inputs:
      - structure.atoms with column 'atom_type' (required)
      - structure.bonds with columns 'a1','a2' (optional)

    Output (plain dict, v0.1 schema):
      - atom_types: unique + sorted strings (after strip)
      - bond_types: unique + sorted [t1,t2] with endpoints canonicalized so t1 <= t2
      - angle_types: unique + sorted [t1,t2,t3] derived from bond adjacency, endpoints canonicalized so t1 <= t3
      - dihedral_types: [] (v0.1.1: not required yet)
    """
    atom_types_by_aid = _extract_atom_types_by_aid(structure)
    n_atoms = len(atom_types_by_aid)

    atom_types = sorted(set(atom_types_by_aid))

    bond_pairs = _normalized_unique_bond_pairs(structure, n_atoms=n_atoms)

    # Bond types from endpoints' atom types
    bond_types_set: set[tuple[str, str]] = set()
    neighbors: list[list[int]] = [[] for _ in range(n_atoms)]
    for a1, a2 in bond_pairs:
        t1 = atom_types_by_aid[a1]
        t2 = atom_types_by_aid[a2]
        bond_types_set.add(_canonicalize_bond_key(t1, t2))
        neighbors[a1].append(a2)
        neighbors[a2].append(a1)

    bond_types = [list(k) for k in sorted(bond_types_set)]

    # Angle enumeration (deterministic) per DevGuide v0.1.1:
    # for each central atom j, consider sorted neighbors; enumerate pairs (i,k) with p<q.
    angle_types_set: set[tuple[str, str, str]] = set()
    for j in range(n_atoms):
        nbrs = sorted(set(neighbors[j]))
        if len(nbrs) < 2:
            continue
        tj = atom_types_by_aid[j]
        for p in range(len(nbrs) - 1):
            i = nbrs[p]
            ti = atom_types_by_aid[i]
            for q in range(p + 1, len(nbrs)):
                k = nbrs[q]
                tk = atom_types_by_aid[k]
                angle_types_set.add(_canonicalize_angle_key(ti, tj, tk))

    angle_types = [list(k) for k in sorted(angle_types_set)]

    return {
        "atom_types": atom_types,
        "bond_types": bond_types,
        "angle_types": angle_types,
        "dihedral_types": [],
    }


def write_requirements_json(structure: Any, path: Union[str, Path]) -> None:
    """Write deterministic Requirements JSON (v0.1) to `path`.

    Writer rules:
      - UTF-8
      - json.dumps(..., indent=2, sort_keys=True)
      - newline-terminated
    """
    req = derive_requirements_v0_1(structure)
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(req, indent=2, sort_keys=True)
    if not text.endswith("\n"):
        text += "\n"
    p.write_text(text, encoding="utf-8")


__all__ = ["derive_requirements_v0_1", "write_requirements_json"]