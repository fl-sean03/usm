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
        usm.atoms.to_csv(atoms_path, index=False, float_format="%.17g")

    # Bonds
    bonds_path = None
    if getattr(usm, "bonds", None) is not None and len(usm.bonds) > 0:
        bonds_path = out_dir / "bonds.parquet"
        try:
            usm.bonds.to_parquet(bonds_path, index=False)
        except Exception:
            bonds_path = out_dir / "bonds.csv"
            usm.bonds.to_csv(bonds_path, index=False, float_format="%.17g")

    # Molecules
    molecules_path = None
    if getattr(usm, "molecules", None) is not None and len(usm.molecules) > 0:
        molecules_path = out_dir / "molecules.parquet"
        try:
            usm.molecules.to_parquet(molecules_path, index=False)
        except Exception:
            molecules_path = out_dir / "molecules.csv"
            usm.molecules.to_csv(molecules_path, index=False, float_format="%.17g")

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
    Load a USM bundle written by save_bundle.
    Behavior:
      - Parquet preferred; CSV fallback (and when both exist, prefer Parquet).
      - Strict manifest handling; raises ValueError with actionable messages on issues.
      - Preserves stored row order; USM.__post_init__ enforces schema/dtypes/contiguous IDs.
    """
    base = Path(folder)
    manifest_path = base / "manifest.json"
    if not manifest_path.exists():
        raise ValueError(f"USM bundle missing manifest.json at {manifest_path}")
    try:
        with manifest_path.open("r", encoding="utf-8") as f:
            manifest = json.load(f)
    except Exception as e:
        raise ValueError(f"Failed to parse manifest.json: {e}")

    version = manifest.get("version")
    if version != VERSION:
        raise ValueError(f"Unsupported bundle version '{version}'; expected '{VERSION}'")

    tables = manifest.get("tables") or {}
    if "atoms" not in tables:
        raise ValueError("Manifest missing required 'tables.atoms' entry")

    def _get_table_entry(name: str) -> Dict[str, Any]:
        t = tables.get(name) or {}
        # Normalize None/"null"/"" to None for path
        path_val = t.get("path")
        t["path"] = None if path_val in (None, "", "null") else str(path_val)
        t["row_count"] = int(t.get("row_count") or 0)
        t["dtypes"] = t.get("dtypes") or {}
        return t

    atoms_meta = _get_table_entry("atoms")
    bonds_meta = _get_table_entry("bonds")
    mols_meta = _get_table_entry("molecules")

    def _read_df(rel_path: Optional[str], required: bool, label: str, expected_rows: int) -> Optional[pd.DataFrame]:
        if rel_path is None:
            if required:
                raise ValueError(f"Manifest marks {label} as required but no path provided")
            return None
        p = base / rel_path
        if not p.exists():
            # Try fallback when manifest references Parquet but file missing and CSV exists (or vice versa)
            alt = p.with_suffix(".csv") if p.suffix.lower() == ".parquet" else p.with_suffix(".parquet")
            if alt.exists():
                p = alt
            else:
                raise ValueError(f"Bundle incomplete: expected file for {label} at {p} (or {alt})")
        # Prefer parquet when both exist; if p is csv but parquet sibling exists, try parquet first
        if p.suffix.lower() == ".csv":
            parquet_sibling = p.with_suffix(".parquet")
            if parquet_sibling.exists():
                try:
                    df = pd.read_parquet(parquet_sibling)
                except Exception:
                    # Fall back to CSV
                    df = pd.read_csv(p)
            else:
                df = pd.read_csv(p)
        else:
            # p is parquet
            try:
                df = pd.read_parquet(p)
            except Exception as e:
                # Fall back to CSV counterpart if available
                csv_alt = p.with_suffix(".csv")
                if csv_alt.exists():
                    df = pd.read_csv(csv_alt)
                else:
                    raise ValueError(f"Failed to read Parquet for {label} at {p}: {e}. No CSV fallback at {csv_alt}.")
        # Basic row count validation if provided (>0 or explicitly 0)
        if expected_rows is not None:
            if int(len(df)) != int(expected_rows):
                raise ValueError(f"Row count mismatch for {label}: manifest={expected_rows} actual={len(df)}")
        return df

    atoms_df = _read_df(atoms_meta["path"], required=True, label="atoms", expected_rows=atoms_meta.get("row_count", None))
    bonds_df = _read_df(bonds_meta["path"], required=False, label="bonds", expected_rows=bonds_meta.get("row_count", None))
    mols_df = _read_df(mols_meta["path"], required=False, label="molecules", expected_rows=mols_meta.get("row_count", None))

    # Construct USM; __post_init__ will enforce schema/dtypes and contiguous IDs
    usm = USM(
        atoms=atoms_df,
        bonds=bonds_df,
        molecules=mols_df,
        cell=dict(manifest.get("cell") or {}),
        provenance=dict(manifest.get("provenance") or {}),
        preserved_text=dict(manifest.get("preserved_text") or {}),
    )
    # Validate basic invariants
    usm.validate_basic()
    return usm