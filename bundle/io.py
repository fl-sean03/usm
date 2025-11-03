from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Any, Optional

import pandas as pd

# Top-level vendored USM package (src/usm)
from ..core.model import USM


VERSION = "0.1"

DEFAULT_UNITS = {
    "coordinates": "angstrom",
    "angles": "degree",
    "charge": "e",
}

DEFAULT_TOLERANCES = {
    "coord_abs": 1e-5,
    "cell_abs": 1e-8,
}


def _dtype_map(df: Optional[pd.DataFrame]) -> Dict[str, str]:
    if df is None:
        return {}
    out: Dict[str, str] = {}
    for c in df.columns:
        try:
            out[c] = str(df.dtypes[c])
        except Exception:
            out[c] = "object"
    return out


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def save_bundle(usm: USM, folder: str) -> str:
    """
    Serialize USM into a 'bundle' folder with tables and a manifest.json.
    Preferred format is Parquet; falls back to CSV if no parquet engine available.
    Layout:
      - atoms.parquet or atoms.csv
      - bonds.parquet or bonds.csv (optional)
      - molecules.parquet or molecules.csv (optional)
      - manifest.json
    """
    out_dir = Path(folder)
    _ensure_dir(out_dir)

    # Atoms
    atoms_path = out_dir / "atoms.parquet"
    try:
        usm.atoms.to_parquet(atoms_path, index=False)
    except Exception:
        atoms_path = out_dir / "atoms.csv"
        usm.atoms.to_csv(atoms_path, index=False)

    # Bonds
    bonds_path = None
    if getattr(usm, "bonds", None) is not None and len(usm.bonds) > 0:
        bonds_path = out_dir / "bonds.parquet"
        try:
            usm.bonds.to_parquet(bonds_path, index=False)
        except Exception:
            bonds_path = out_dir / "bonds.csv"
            usm.bonds.to_csv(bonds_path, index=False)

    # Molecules
    molecules_path = None
    if getattr(usm, "molecules", None) is not None and len(usm.molecules) > 0:
        molecules_path = out_dir / "molecules.parquet"
        try:
            usm.molecules.to_parquet(molecules_path, index=False)
        except Exception:
            molecules_path = out_dir / "molecules.csv"
            usm.molecules.to_csv(molecules_path, index=False)

    manifest = {
        "version": VERSION,
        "units": DEFAULT_UNITS,
        "numeric_tolerances": DEFAULT_TOLERANCES,
        "tables": {
            "atoms": {
                "path": str(atoms_path.name),
                "dtypes": _dtype_map(usm.atoms),
                "row_count": int(len(usm.atoms)),
            },
            "bonds": {
                "path": str(bonds_path.name) if bonds_path else None,
                "dtypes": _dtype_map(getattr(usm, "bonds", None)),
                "row_count": int(len(usm.bonds)) if getattr(usm, "bonds", None) is not None else 0,
            },
            "molecules": {
                "path": str(molecules_path.name) if molecules_path else None,
                "dtypes": _dtype_map(getattr(usm, "molecules", None)),
                "row_count": int(len(usm.molecules)) if getattr(usm, "molecules", None) is not None else 0,
            },
        },
        "cell": dict(getattr(usm, "cell", {})),
        "provenance": dict(getattr(usm, "provenance", {}) or {}),
        "preserved_text": dict(getattr(usm, "preserved_text", {}) or {}),
        "extras": {},
    }

    manifest_path = out_dir / "manifest.json"
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    return str(out_dir)


def load_bundle(folder: str) -> USM:
    """
    Optional loader for completeness. Not used by current workflows.
    Provided as a minimal placeholder; implement as needed.
    """
    raise NotImplementedError("load_bundle is not required by current workspaces")