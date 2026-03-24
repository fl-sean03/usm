"""Shared canonicalization utilities for topology derivation.

Used by requirements.py, termset.py, and parameterset.py to avoid
code duplication. All functions are deterministic and produce
canonical (sorted/normalized) keys.
"""
from __future__ import annotations

from typing import Any, Optional


def _norm_str(value: Any, *, where: str) -> str:
    """Validate and strip a string value."""
    if not isinstance(value, str):
        raise ValueError(f"{where}: expected str, got {type(value).__name__}")
    s = value.strip()
    if not s:
        raise ValueError(f"{where}: must be a non-empty string")
    return s


def _canonicalize_bond_key(t1: str, t2: str) -> tuple[str, str]:
    """Canonicalize bond key: t1 <= t2 lexicographically."""
    a = _norm_str(t1, where="bond_types[*][0]")
    b = _norm_str(t2, where="bond_types[*][1]")
    return (a, b) if a <= b else (b, a)


def _canonicalize_angle_key(t1: str, t2: str, t3: str) -> tuple[str, str, str]:
    """Canonicalize angle key: endpoints t1 <= t3, center t2 fixed."""
    a = _norm_str(t1, where="angle_types[*][0]")
    b = _norm_str(t2, where="angle_types[*][1]")
    c = _norm_str(t3, where="angle_types[*][2]")
    return (a, b, c) if a <= c else (c, b, a)


def _canonicalize_dihedral_key(
    t1: str, t2: str, t3: str, t4: str,
) -> tuple[str, str, str, str]:
    """Canonicalize dihedral: forward vs reversed, pick lexicographically smaller."""
    a = _norm_str(t1, where="dihedral_types[*][0]")
    b = _norm_str(t2, where="dihedral_types[*][1]")
    c = _norm_str(t3, where="dihedral_types[*][2]")
    d = _norm_str(t4, where="dihedral_types[*][3]")
    fwd = (a, b, c, d)
    rev = (d, c, b, a)
    return fwd if fwd <= rev else rev


def _canonicalize_improper_key(
    t1: str, t2: str, t3: str, t4: str,
) -> tuple[str, str, str, str]:
    """Canonicalize impropers: t2 is central, sort peripherals (t1,t3,t4)."""
    a = _norm_str(t1, where="improper_types[*][0]")
    b = _norm_str(t2, where="improper_types[*][1]")
    c = _norm_str(t3, where="improper_types[*][2]")
    d = _norm_str(t4, where="improper_types[*][3]")
    p1, p2, p3 = sorted([a, c, d])
    return (p1, b, p2, p3)


def _extract_atom_types_by_aid(structure: Any) -> list[str]:
    """Extract atom types indexed by aid from a USM structure.

    Returns list where index i is the atom_type for aid=i.
    Validates contiguous 0..N-1 aid mapping.
    """
    if not hasattr(structure, "atoms"):
        raise ValueError("structure: missing .atoms table")
    atoms = structure.atoms
    if "atom_type" not in atoms.columns:
        raise ValueError("structure.atoms: missing required column 'atom_type'")

    n = int(len(atoms))

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
                raise ValueError(f"structure.atoms.aid[{idx}]: {e}") from e
            if ai < 0 or ai >= n:
                raise ValueError(f"structure.atoms.aid[{idx}]={ai}: out of range [0, {n})")
            if ai in seen:
                raise ValueError(f"structure.atoms.aid[{idx}]={ai}: duplicate aid")
            seen.add(ai)
            by_aid[ai] = str(t).strip()
        if any(v is None for v in by_aid):
            missing = [i for i, v in enumerate(by_aid) if v is None]
            raise ValueError(f"structure.atoms.aid: missing aid values: {missing[:10]}")
        return by_aid  # type: ignore[return-value]
    else:
        return [str(t).strip() for t in atoms["atom_type"].tolist()]
