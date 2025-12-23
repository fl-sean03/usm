from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Union

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ParameterSetDerivationError(ValueError):
    """Deterministic error for ParameterSet derivation failures."""

    missing_types: tuple[str, ...] = ()
    inconsistent_types: tuple[str, ...] = ()
    details: dict[str, Any] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        object.__setattr__(self, "missing_types", tuple(sorted(self.missing_types)))
        object.__setattr__(self, "inconsistent_types", tuple(sorted(self.inconsistent_types)))
        if self.details is None:
            object.__setattr__(self, "details", {})

    def __str__(self) -> str:
        parts: list[str] = ["ParameterSet derivation failed"]
        if self.missing_types:
            parts.append(f"missing_types={list(self.missing_types)}")
        if self.inconsistent_types:
            parts.append(f"inconsistent_types={list(self.inconsistent_types)}")
        if self.details:
            # stable order for message
            keys = sorted(self.details)
            details_str = ", ".join(f"{k}={self.details[k]}" for k in keys)
            parts.append(f"details={{ {details_str} }}")
        return "; ".join(parts)


def _norm_str(value: Any, *, where: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{where}: expected str, got {type(value).__name__}")
    s = value.strip()
    if not s:
        raise ValueError(f"{where}: must be a non-empty string")
    return s


def derive_parameterset_v0_1_2(structure: Any) -> dict[str, Any]:
    """Derive ParameterSet v0.1.2 deterministically from a USM-like structure.

    Requires per-atom columns (see [`ATOMS_DTYPES`](src/usm/core/model.py:9)):
      - mass_amu
      - lj_sigma_angstrom
      - lj_epsilon_kcal_mol

    Rules:
      - For each atom_type: required columns must be present and non-null for all atoms of that type.
      - For each atom_type: required values must be exactly equal across all atoms of that type.
      - Optional: include element only if present and consistent for that type.
    """

    if not hasattr(structure, "atoms"):
        raise ValueError("structure: missing .atoms table")
    atoms = structure.atoms
    if "atom_type" not in atoms.columns:
        raise ValueError("structure.atoms: missing required column 'atom_type'")

    required_cols = ["mass_amu", "lj_sigma_angstrom", "lj_epsilon_kcal_mol"]
    missing_cols = sorted([c for c in required_cols if c not in atoms.columns])
    if missing_cols:
        raise ParameterSetDerivationError(details={"missing_columns": missing_cols})

    # Normalize atom_type strings (strip) for deterministic grouping.
    atom_types = [
        _norm_str(t, where=f"structure.atoms.atom_type[{i}]")
        for i, t in enumerate(atoms["atom_type"].tolist())
    ]

    # Iterate types deterministically.
    unique_types = sorted(set(atom_types))

    missing_types: list[str] = []
    inconsistent_types: list[str] = []
    details: dict[str, Any] = {"missing": {}, "inconsistent": {}}

    # Build per-type record.
    out_atom_types: dict[str, dict[str, Any]] = {}

    # Precompute row indices for each type using deterministic row order.
    type_to_rows: dict[str, list[int]] = {t: [] for t in unique_types}
    for idx, t in enumerate(atom_types):
        type_to_rows[t].append(idx)

    def _values_for(col: str, rows: list[int]) -> list[Any]:
        return [atoms.iloc[r][col] for r in rows]

    for t in unique_types:
        rows = type_to_rows[t]

        rec: dict[str, Any] = {}

        # Required numeric fields
        type_missing_cols: list[str] = []
        type_inconsistent_cols: list[str] = []

        for col in required_cols:
            vals = _values_for(col, rows)

            # treat NaN/NA as missing
            if any(pd.isna(v) for v in vals):
                type_missing_cols.append(col)
                continue

            first = float(vals[0])
            if any(float(v) != first for v in vals[1:]):
                type_inconsistent_cols.append(col)
                continue

            rec[col] = first

        if type_missing_cols:
            missing_types.append(t)
            details["missing"][t] = {"columns": sorted(type_missing_cols)}
            continue
        if type_inconsistent_cols:
            inconsistent_types.append(t)
            details["inconsistent"][t] = {"columns": sorted(type_inconsistent_cols)}
            continue

        # Optional element
        if "element" in atoms.columns:
            elems = _values_for("element", rows)
            # include only if present (non-null), stringy, and consistent
            if elems and all(isinstance(e, str) and e.strip() for e in elems) and not any(pd.isna(e) for e in elems):
                e0 = elems[0].strip()
                if all(e.strip() == e0 for e in elems[1:]):
                    rec["element"] = e0

        out_atom_types[t] = rec

    if missing_types or inconsistent_types:
        # remove empty subblocks for stable output
        if not details["missing"]:
            details.pop("missing", None)
        if not details.get("inconsistent"):
            details.pop("inconsistent", None)
        raise ParameterSetDerivationError(
            missing_types=tuple(sorted(missing_types)),
            inconsistent_types=tuple(sorted(inconsistent_types)),
            details=details,
        )

    return {
        "schema": "upm.parameterset.v0.1.2",
        "atom_types": {k: out_atom_types[k] for k in sorted(out_atom_types)},
    }


def write_parameterset_json(pset: dict[str, Any], path: Union[str, Path]) -> Union[str, Path]:
    """Write deterministic ParameterSet JSON to `path`."""

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(pset, indent=2, sort_keys=True)
    if not text.endswith("\n"):
        text += "\n"
    p.write_text(text, encoding="utf-8")
    return path


def export_parameterset_json(structure: Any, path: Union[str, Path]) -> Union[str, Path]:
    """Convenience: derive then write ParameterSet JSON."""

    return write_parameterset_json(derive_parameterset_v0_1_2(structure), path)


__all__ = [
    "ParameterSetDerivationError",
    "derive_parameterset_v0_1_2",
    "write_parameterset_json",
    "export_parameterset_json",
]
