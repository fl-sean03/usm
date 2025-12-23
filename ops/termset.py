from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional, Union


def _norm_str(value: Any, *, where: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{where}: expected str, got {type(value).__name__}")
    s = value.strip()
    if not s:
        raise ValueError(f"{where}: must be a non-empty string")
    return s


def _canonicalize_bond_key(t1: str, t2: str) -> tuple[str, str]:
    a = _norm_str(t1, where="bond_types[*][0]")
    b = _norm_str(t2, where="bond_types[*][1]")
    return (a, b) if a <= b else (b, a)


def _canonicalize_angle_key(t1: str, t2: str, t3: str) -> tuple[str, str, str]:
    a = _norm_str(t1, where="angle_types[*][0]")
    b = _norm_str(t2, where="angle_types[*][1]")
    c = _norm_str(t3, where="angle_types[*][2]")
    return (a, b, c) if a <= c else (c, b, a)


def _canonicalize_dihedral_key(t1: str, t2: str, t3: str, t4: str) -> tuple[str, str, str, str]:
    a = _norm_str(t1, where="dihedral_types[*][0]")
    b = _norm_str(t2, where="dihedral_types[*][1]")
    c = _norm_str(t3, where="dihedral_types[*][2]")
    d = _norm_str(t4, where="dihedral_types[*][3]")
    fwd = (a, b, c, d)
    rev = (d, c, b, a)
    return fwd if fwd <= rev else rev


def _canonicalize_improper_key(t1: str, t2: str, t3: str, t4: str) -> tuple[str, str, str, str]:
    """Canonicalize impropers conservatively per DevGuide v0.1.2.

    Assumes t2 is central. Sort peripherals (t1,t3,t4) ascending.
    """

    a = _norm_str(t1, where="improper_types[*][0]")
    b = _norm_str(t2, where="improper_types[*][1]")
    c = _norm_str(t3, where="improper_types[*][2]")
    d = _norm_str(t4, where="improper_types[*][3]")
    p1, p2, p3 = sorted([a, c, d])
    return (p1, b, p2, p3)


def _extract_atom_types_by_aid(structure: Any) -> list[str]:
    """Aid-aware atom_type extraction.

    Mirrors the deterministic mapping logic in [`_extract_atom_types_by_aid()`](src/usm/ops/requirements.py:30),
    but kept local so this module is standalone.
    """

    if not hasattr(structure, "atoms"):
        raise ValueError("structure: missing .atoms table")
    atoms = structure.atoms
    if "atom_type" not in atoms.columns:
        raise ValueError("structure.atoms: missing required column 'atom_type'")

    n = int(len(atoms))

    # Prefer explicit aid mapping if present; otherwise assume row order is aid order.
    if "aid" in atoms.columns:
        aids = atoms["aid"].tolist()
        types_raw = atoms["atom_type"].tolist()
        by_aid: list[Optional[str]] = [None] * n
        seen: set[int] = set()
        for idx, (aid, t) in enumerate(zip(aids, types_raw)):
            if aid is None:
                raise ValueError(f"structure.atoms.aid[{idx}]: must be an int, got null")
            try:
                ai = int(aid)
            except Exception as e:
                raise ValueError(
                    f"structure.atoms.aid[{idx}]: must be an int, got {type(aid).__name__}"
                ) from e
            if ai in seen:
                raise ValueError(f"structure.atoms.aid[{idx}]: duplicate aid {ai}")
            if ai < 0 or ai >= n:
                raise ValueError(f"structure.atoms.aid[{idx}]: out of range: {ai} (n_atoms={n})")
            seen.add(ai)
            by_aid[ai] = _norm_str(t, where=f"structure.atoms.atom_type[{idx}]")

        if any(v is None for v in by_aid):
            raise ValueError("structure.atoms: aid mapping must cover a contiguous 0..n-1 range")
        return [v for v in by_aid if v is not None]

    # No aid column: use row order.
    return [
        _norm_str(t, where=f"structure.atoms.atom_type[{i}]")
        for i, t in enumerate(atoms["atom_type"].tolist())
    ]


def _normalized_unique_bond_pairs(
    structure: Any, *, n_atoms: int
) -> list[tuple[int, int, int, int, int]]:
    """Return sorted unique (a1, a2, ix, iy, iz) with normalization and validation.

    Normalization follows DevGuide v0.1.3:
    - If a1 < a2: keep (a1, a2, ix, iy, iz)
    - If a1 > a2: swap to (a2, a1, -ix, -iy, -iz)
    - If a1 == a2: ensure (ix, iy, iz) is lexicographically positive.
    """

    bonds = getattr(structure, "bonds", None)
    if bonds is None or len(bonds) == 0:
        return []

    required = ["a1", "a2"]
    for col in required:
        if col not in bonds.columns:
            raise ValueError(f"structure.bonds: missing required column '{col}'")

    a1s = bonds["a1"].tolist()
    a2s = bonds["a2"].tolist()

    # PBC image flags default to 0 if missing (backward compatibility)
    ixs = bonds["ix"].tolist() if "ix" in bonds.columns else [0] * len(bonds)
    iys = bonds["iy"].tolist() if "iy" in bonds.columns else [0] * len(bonds)
    izs = bonds["iz"].tolist() if "iz" in bonds.columns else [0] * len(bonds)

    pairs: set[tuple[int, int, int, int, int]] = set()
    for i, (a1, a2, ix, iy, iz) in enumerate(zip(a1s, a2s, ixs, iys, izs)):
        if a1 is None or a2 is None:
            raise ValueError(f"structure.bonds[{i}]: a1/a2 must be ints, got null")
        try:
            x = int(a1)
            y = int(a2)
            oxi = int(ix or 0)
            oyi = int(iy or 0)
            ozi = int(iz or 0)
        except Exception as e:
            raise ValueError(
                f"structure.bonds[{i}]: a1, a2, ix, iy, iz must be ints"
            ) from e

        if x < 0 or x >= n_atoms or y < 0 or y >= n_atoms:
            raise ValueError(
                f"structure.bonds[{i}]: a1/a2 out of range: ({x},{y}) (n_atoms={n_atoms})"
            )

        # Normalization
        if x < y:
            entry = (x, y, oxi, oyi, ozi)
        elif x > y:
            entry = (y, x, -oxi, -oyi, -ozi)
        else:
            # self-bond (e.g. across PBC)
            if oxi == 0 and oyi == 0 and ozi == 0:
                raise ValueError(f"structure.bonds[{i}]: self-bond not allowed (aid={x})")
            if (oxi, oyi, ozi) < (-oxi, -oyi, -ozi):
                entry = (x, y, oxi, oyi, ozi)
            else:
                entry = (x, y, -oxi, -oyi, -ozi)

        pairs.add(entry)

    return sorted(pairs)


def _encode_key(parts: tuple[str, ...]) -> str:
    return "|".join(parts)


def derive_termset_v0_1_2(structure: Any) -> dict[str, Any]:
    """Derive TermSet v0.1.2 deterministically from a USM-like structure.

    Inputs (duck-typed):
      - structure.atoms: DataFrame with required column 'atom_type' and optional 'aid'
      - structure.bonds: DataFrame with columns 'a1','a2' (optional)

    Canonicalization rules are specified in [`docs/DevGuides/DevGuide_v0.1.2.md`](docs/DevGuides/DevGuide_v0.1.2.md:175).
    """

    atom_types_by_aid = _extract_atom_types_by_aid(structure)
    n_atoms = len(atom_types_by_aid)

    atom_types = sorted(set(atom_types_by_aid))

    bond_pairs = _normalized_unique_bond_pairs(structure, n_atoms=n_atoms)

    # neighbors[i] stores (neighbor_id, ix, iy, iz)
    neighbors: list[list[tuple[int, int, int, int]]] = [[] for _ in range(n_atoms)]
    bond_types_counts: dict[str, int] = {}
    bond_types_set: set[tuple[str, str]] = set()

    for a1, a2, ix, iy, iz in bond_pairs:
        t1 = atom_types_by_aid[a1]
        t2 = atom_types_by_aid[a2]
        key = _canonicalize_bond_key(t1, t2)
        bond_types_set.add(key)
        bond_types_counts[_encode_key(key)] = bond_types_counts.get(_encode_key(key), 0) + 1
        neighbors[a1].append((a2, ix, iy, iz))
        neighbors[a2].append((a1, -ix, -iy, -iz))

    # normalize adjacency for deterministic enumeration
    for j in range(n_atoms):
        neighbors[j] = sorted(set(neighbors[j]))

    bond_types = [list(k) for k in sorted(bond_types_set)]

    angle_types_counts: dict[str, int] = {}
    angle_types_set: set[tuple[str, str, str]] = set()
    for j in range(n_atoms):
        nbrs = neighbors[j]
        if len(nbrs) < 2:
            continue
        tj = atom_types_by_aid[j]
        for p in range(len(nbrs) - 1):
            i, *_ = nbrs[p]
            ti = atom_types_by_aid[i]
            for q in range(p + 1, len(nbrs)):
                k, *_ = nbrs[q]
                tk = atom_types_by_aid[k]
                key = _canonicalize_angle_key(ti, tj, tk)
                angle_types_set.add(key)
                angle_types_counts[_encode_key(key)] = (
                    angle_types_counts.get(_encode_key(key), 0) + 1
                )

    angle_types = [list(k) for k in sorted(angle_types_set)]

    dihedral_types_counts: dict[str, int] = {}
    dihedral_types_set: set[tuple[str, str, str, str]] = set()
    for j, k, bj_ix, bj_iy, bj_iz in bond_pairs:
        # Enumerate i-j-k-l with i neighbor of j (excluding k) and l neighbor of k (excluding j)
        # Periodic awareness: i-j and k-l can be periodic.
        # However, TermSet v0.1.2 defines types by atom types only.
        # We must exclude the same physical bond.
        # In neighbors[j], we have (i, ix, iy, iz). k is (k, bj_ix, bj_iy, bj_iz).
        for i, iix, iiy, iiz in neighbors[j]:
            if i == k and iix == bj_ix and iiy == bj_iy and iiz == bj_iz:
                continue
            for l, lix, liy, liz in neighbors[k]:
                # j relative to k is (-bj_ix, -bj_iy, -bj_iz)
                if l == j and lix == -bj_ix and liy == -bj_iy and liz == -bj_iz:
                    continue
                key = _canonicalize_dihedral_key(
                    atom_types_by_aid[i],
                    atom_types_by_aid[j],
                    atom_types_by_aid[k],
                    atom_types_by_aid[l],
                )
                dihedral_types_set.add(key)
                dihedral_types_counts[_encode_key(key)] = (
                    dihedral_types_counts.get(_encode_key(key), 0) + 1
                )

    dihedral_types = [list(k) for k in sorted(dihedral_types_set)]

    improper_types_counts: dict[str, int] = {}
    improper_types_set: set[tuple[str, str, str, str]] = set()
    for j in range(n_atoms):
        nbrs = neighbors[j]
        if len(nbrs) < 3:
            continue
        tj = atom_types_by_aid[j]
        # combinations of 3 peripherals (deterministic because nbrs sorted)
        for a in range(len(nbrs) - 2):
            i, *_ = nbrs[a]
            for b in range(a + 1, len(nbrs) - 1):
                k, *_ = nbrs[b]
                for c in range(b + 1, len(nbrs)):
                    l, *_ = nbrs[c]
                    key = _canonicalize_improper_key(
                        atom_types_by_aid[i],
                        tj,
                        atom_types_by_aid[k],
                        atom_types_by_aid[l],
                    )
                    improper_types_set.add(key)
                    improper_types_counts[_encode_key(key)] = (
                        improper_types_counts.get(_encode_key(key), 0) + 1
                    )

    improper_types = [list(k) for k in sorted(improper_types_set)]

    return {
        "schema": "molsaic.termset.v0.1.2",
        "atom_types": atom_types,
        "bond_types": bond_types,
        "angle_types": angle_types,
        "dihedral_types": dihedral_types,
        "improper_types": improper_types,
        "counts": {
            "bond_types": {k: bond_types_counts[k] for k in sorted(bond_types_counts)},
            "angle_types": {k: angle_types_counts[k] for k in sorted(angle_types_counts)},
            "dihedral_types": {k: dihedral_types_counts[k] for k in sorted(dihedral_types_counts)},
            "improper_types": {k: improper_types_counts[k] for k in sorted(improper_types_counts)},
        },
    }


def write_termset_json(termset: dict[str, Any], path: Union[str, Path]) -> Union[str, Path]:
    """Write deterministic TermSet JSON to `path`.

    Writer rules:
      - UTF-8
      - json.dumps(..., indent=2, sort_keys=True)
      - newline-terminated
    """

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(termset, indent=2, sort_keys=True)
    if not text.endswith("\n"):
        text += "\n"
    p.write_text(text, encoding="utf-8")
    return path


def export_termset_json(structure: Any, path: Union[str, Path]) -> Union[str, Path]:
    """Convenience: derive then write TermSet JSON."""

    return write_termset_json(derive_termset_v0_1_2(structure), path)


__all__ = [
    "derive_termset_v0_1_2",
    "write_termset_json",
    "export_termset_json",
]

