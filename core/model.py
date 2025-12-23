from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List
import pandas as pd
import numpy as np

# Default atoms schema columns and dtypes
ATOMS_DTYPES = {
    "aid": "int32",
    "name": "string",
    "element": "string",
    "atom_type": "string",
    "charge": "float32",
    # Optional per-atom physical / forcefield parameters (nullable)
    # (Not written to CAR/MDF by default; used by downstream parametrization/export pipelines.)
    "mass_amu": "float32",
    "lj_epsilon_kcal_mol": "float32",
    "lj_sigma_angstrom": "float32",
    "x": "float64",
    "y": "float64",
    "z": "float64",
    "mol_label": "string",
    "mol_index": "int32",
    "mol_block_name": "string",
    # MDF preservation fields (nullable)
    "isotope": "string",
    "formal_charge": "string",
    "switching_atom": "Int8",
    "oop_flag": "Int8",
    "chirality_flag": "Int8",
    "occupancy": "float32",
    "xray_temp_factor": "float32",
    "charge_group": "string",
    "connections_raw": "string",
}

BONDS_DTYPES = {
    "bid": "Int32",
    "a1": "Int32",
    "a2": "Int32",
    "ix": "Int32",
    "iy": "Int32",
    "iz": "Int32",
    "order": "float32",
    "type": "string",
    "source": "string",
    "order_raw": "string",
    "mol_index": "Int32",
    "notes": "string",
}

MOLECULES_DTYPES = {
    "mid": "Int32",
    "mol_label": "string",
    "mol_index": "Int32",
    "mol_block_name": "string",
    "provenance": "string",
}

REQUIRED_ATOM_COLUMNS = [
    "aid", "name", "element", "atom_type", "charge", "x", "y", "z",
    "mol_label", "mol_index", "mol_block_name"
]

def _ensure_columns(df: pd.DataFrame, dtypes: Dict[str, str]) -> pd.DataFrame:
    # add missing columns with NA
    for col, dtype in dtypes.items():
        if col not in df.columns:
            if dtype.startswith("float"):
                df[col] = np.nan
            elif dtype.lower().endswith("int8") or dtype.lower().endswith("int16") or dtype.lower().endswith("int32") or dtype.lower().endswith("int64") or dtype.startswith("Int"):
                # Create as object with NA to avoid casting issues, cast later in _astype_with_nullable
                df[col] = pd.Series([pd.NA] * len(df), dtype="object")
                continue
            else:
                df[col] = pd.Series([pd.NA] * len(df), dtype="string")
    # order columns stable: present ones first in same order as dtypes mapping; keep extras at end
    ordered = [c for c in dtypes.keys() if c in df.columns]
    extras = [c for c in df.columns if c not in dtypes]
    return df[ordered + extras]

def _astype_with_nullable(df: pd.DataFrame, dtypes: Dict[str, str]) -> pd.DataFrame:
    for col, dtype in dtypes.items():
        if col in df.columns:
            try:
                df[col] = df[col].astype(dtype)
            except Exception:
                if dtype.startswith("float"):
                    df[col] = pd.to_numeric(df[col], errors="coerce").astype(dtype)
                elif dtype.startswith("Int"):
                    df[col] = pd.to_numeric(df[col], errors="coerce").astype(dtype)
                elif dtype in ("string", "category"):
                    df[col] = df[col].astype("string")
    return df

@dataclass
class USM:
    atoms: pd.DataFrame
    bonds: Optional[pd.DataFrame] = None
    molecules: Optional[pd.DataFrame] = None
    cell: Dict[str, Any] = field(default_factory=lambda: dict(pbc=False, a=np.nan, b=np.nan, c=np.nan, alpha=np.nan, beta=np.nan, gamma=np.nan, spacegroup=""))
    provenance: Dict[str, Any] = field(default_factory=dict)
    preserved_text: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.atoms is None:
            raise ValueError("USM requires an atoms table")
        self.atoms = self.atoms.copy()
        if "aid" not in self.atoms.columns:
            self.atoms.insert(0, "aid", np.arange(len(self.atoms), dtype=np.int32))
        self.atoms = _ensure_columns(self.atoms, ATOMS_DTYPES)
        self.atoms = _astype_with_nullable(self.atoms, ATOMS_DTYPES)
        self.atoms["aid"] = np.arange(len(self.atoms), dtype=np.int32)

        if self.bonds is not None:
            self.bonds = self.bonds.copy()
            self.bonds = _ensure_columns(self.bonds, BONDS_DTYPES)
            # Default missing image flags to 0
            for col in ["ix", "iy", "iz"]:
                if self.bonds[col].isna().any():
                    self.bonds.loc[self.bonds[col].isna(), col] = 0
            self.bonds = _astype_with_nullable(self.bonds, BONDS_DTYPES)
            a1 = self.bonds["a1"].to_numpy()
            a2 = self.bonds["a2"].to_numpy()
            swap = a1 > a2
            if swap.any():
                self.bonds.loc[swap, ["a1", "a2"]] = self.bonds.loc[swap, ["a2", "a1"]].to_numpy()
                # Negate image flags on swap
                for col in ["ix", "iy", "iz"]:
                    self.bonds.loc[swap, col] = -self.bonds.loc[swap, col]

            # Lexicographical normalization for self-bonds (a1 == a2)
            # Ensure (ix, iy, iz) is lexicographically positive
            a1_now = self.bonds["a1"].to_numpy()
            a2_now = self.bonds["a2"].to_numpy()
            self_bonds = (a1_now == a2_now)
            if self_bonds.any():
                ix = self.bonds.loc[self_bonds, "ix"].to_numpy()
                iy = self.bonds.loc[self_bonds, "iy"].to_numpy()
                iz = self.bonds.loc[self_bonds, "iz"].to_numpy()
                neg = (ix < 0) | ((ix == 0) & (iy < 0)) | ((ix == 0) & (iy == 0) & (iz < 0))
                if neg.any():
                    for col in ["ix", "iy", "iz"]:
                        self.bonds.loc[self_bonds & neg, col] = -self.bonds.loc[self_bonds & neg, col]

            self.bonds["bid"] = np.arange(len(self.bonds), dtype=np.int32)

        if self.molecules is not None:
            self.molecules = self.molecules.copy()
            self.molecules = _ensure_columns(self.molecules, MOLECULES_DTYPES)
            self.molecules = _astype_with_nullable(self.molecules, MOLECULES_DTYPES)
            self.molecules["mid"] = np.arange(len(self.molecules), dtype=np.int32)

    def copy(self) -> "USM":
        return USM(
            atoms=self.atoms.copy(),
            bonds=None if self.bonds is None else self.bonds.copy(),
            molecules=None if self.molecules is None else self.molecules.copy(),
            cell=dict(self.cell),
            provenance=dict(self.provenance),
            preserved_text=dict(self.preserved_text),
        )

    def validate_basic(self) -> None:
        missing = [c for c in REQUIRED_ATOM_COLUMNS if c not in self.atoms.columns]
        if missing:
            raise ValueError(f"Atoms table missing required columns: {missing}")
        coords = self.atoms[["x","y","z"]].to_numpy()
        # Allow NaN for formats that do not provide coordinates (e.g., MDF); disallow infinities
        if np.isinf(coords).any():
            raise ValueError("Infinite coordinate values found")
        if "charge" not in self.atoms.columns:
            raise ValueError("Atoms table missing 'charge'")

    @staticmethod
    def from_records(atoms_records: List[Dict[str, Any]],
                     bonds_records: Optional[List[Dict[str, Any]]] = None,
                     molecules_records: Optional[List[Dict[str, Any]]] = None,
                     cell: Optional[Dict[str, Any]] = None,
                     provenance: Optional[Dict[str, Any]] = None,
                     preserved_text: Optional[Dict[str, Any]] = None) -> "USM":
        atoms_df = pd.DataFrame(atoms_records)
        bonds_df = None if bonds_records is None else pd.DataFrame(bonds_records)
        molecules_df = None if molecules_records is None else pd.DataFrame(molecules_records)
        return USM(
            atoms=atoms_df,
            bonds=bonds_df,
            molecules=molecules_df,
            cell=cell or dict(pbc=False, a=np.nan, b=np.nan, c=np.nan, alpha=np.nan, beta=np.nan, gamma=np.nan, spacegroup=""),
            provenance=provenance or {},
            preserved_text=preserved_text or {},
        )