from __future__ import annotations

"""
USM v0.1 minimal core and CAR I/O for the WAT/DOP spike.

- Core: USM class (atoms table + metadata), dtype enforcement, validation.
- CAR I/O: load_car, save_car with exact header/footer preservation.
- Scope: PBC=OFF fully supported. PBC=ON header/footer preserved; PBC cell line parsing is best-effort if present.

Atom line format supported (as seen in examples):
  name  x  y  z  mol_label  mol_index  atom_type  element  charge

Example:
  H1  2.568056419  2.265759415  2.682995600  XXXX  1  H*  H  0.410
"""

import re
import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Required atom columns (names and dtypes)
REQUIRED_ATOM_COLUMNS: List[str] = [
    "aid", "name", "element", "atom_type", "charge",
    "x", "y", "z",
    "mol_label", "mol_index", "mol_block_name",
]

DTYPES = {
    "aid": np.int32,
    "name": "object",
    "element": "object",
    "atom_type": "object",
    "charge": np.float32,
    "x": np.float64,
    "y": np.float64,
    "z": np.float64,
    "mol_label": "object",
    "mol_index": np.int32,
    "mol_block_name": "object",
    # Optional MDF-preservation field (Car keeps None)
    "connections_raw": "object",
}


def _coerce_atom_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure atoms dataframe has the required columns and dtypes; fill missing optionals."""
    # Add any missing required columns
    for col in REQUIRED_ATOM_COLUMNS:
        if col not in df.columns:
            df[col] = None

    # Optional MDF-only column for future round-trip; absent for CAR inputs
    if "connections_raw" not in df.columns:
        df["connections_raw"] = None

    # Column order preference (keep any extra columns at end)
    order = [c for c in REQUIRED_ATOM_COLUMNS if c in df.columns]
    extras = [c for c in df.columns if c not in order]
    df = df[order + extras]

    # Dtype coercion
    for col, dt in DTYPES.items():
        if col in df.columns:
            try:
                if dt == "object":
                    # leave as-is (strings/nullable)
                    continue
                df[col] = df[col].astype(dt)
            except Exception:
                # Best-effort: for numeric targets, coerce with errors='ignore'
                if np.issubdtype(np.dtype(dt), np.number):
                    df[col] = pd.to_numeric(df[col], errors="coerce").astype(dt)
    return df


class USM:
    """
    Unified Structure Model v0.1 (minimal)
    - atoms: pandas DataFrame with fixed schema and dtypes (see REQUIRED_ATOM_COLUMNS, DTYPES)
    - metadata: dict carrying cell/PBC, preserved text blocks, provenance, parse notes
    """

    def __init__(self, atoms: pd.DataFrame, metadata: Dict):
        self.atoms = _coerce_atom_dtypes(atoms.copy())
        self.metadata = dict(metadata or {})
        self.validate()

    def validate(self) -> None:
        # Required columns present
        missing = [c for c in REQUIRED_ATOM_COLUMNS if c not in self.atoms.columns]
        if missing:
            raise ValueError(f"USM atoms missing required columns: {missing}")
        # Dtype checks (best-effort)
        for col, dt in DTYPES.items():
            if col in self.atoms.columns:
                # Skip strict checking for 'object' because pandas string/object variants vary
                if dt != "object":
                    if str(self.atoms[col].dtype) != str(np.dtype(dt)):
                        # Allow Int32 vs int32 equivalence
                        try:
                            self.atoms[col] = self.atoms[col].astype(dt)
                        except Exception as e:
                            raise TypeError(f"Column {col} cannot be coerced to {dt}: {e}") from e

        # Basic numeric validity
        for col in ("x", "y", "z", "charge"):
            if not np.isfinite(self.atoms[col].to_numpy(dtype=float)).all():
                raise ValueError(f"Non-finite values found in atoms.{col}")

        # Metadata minimums
        prov = self.metadata.setdefault("provenance", {})
        preserved = self.metadata.setdefault("preserved_text", {})
        cell = self.metadata.setdefault("cell", {"pbc": False})
        if "pbc" not in cell:
            cell["pbc"] = False

    @property
    def pbc(self) -> bool:
        return bool(self.metadata.get("cell", {}).get("pbc", False))


# ----------------------
# CAR Import/Export I/O
# ----------------------

_DATE_PREFIX = "!DATE"
_ARCHIVE_PREFIX = "!BIOSYM archive"
_MATSCI_LINE = "Materials Studio Generated CAR File"
_PBC_EQ = re.compile(r"^PBC=(ON|OFF)\s*$", re.IGNORECASE)
_PBC_CELL = re.compile(
    r"^PBC\s+([+\-0-9.]+)\s+([+\-0-9.]+)\s+([+\-0-9.]+)\s+([+\-0-9.]+)\s+([+\-0-9.]+)\s+([+\-0-9.]+)",
    re.IGNORECASE,
)


def _try_parse_car_atom_tokens(tokens: List[str]) -> Optional[Dict]:
    """
    Attempt to parse a CAR atom line given whitespace-split tokens.
    Expected pattern (from end): [... x y z] mol_label mol_index atom_type element charge
    """
    if len(tokens) < 9:
        return None
    try:
        # Name and coordinates from leading tokens
        name = tokens[0]
        x = float(tokens[1])
        y = float(tokens[2])
        z = float(tokens[3])

        # Trailing tokens (more robust across variants)
        charge = float(tokens[-1])
        element = tokens[-2]
        atom_type = tokens[-3]
        mol_index = int(tokens[-4])
        mol_label = tokens[-5]

        return {
            "name": name,
            "x": x,
            "y": y,
            "z": z,
            "mol_label": mol_label,
            "mol_index": mol_index,
            "atom_type": atom_type,
            "element": element,
            "charge": charge,
        }
    except Exception:
        return None


def _scan_header_footer(lines: List[str]) -> Tuple[List[str], List[Tuple[int, Dict]], List[str], Dict]:
    """
    Identify header lines, atom rows (index + parsed dict), and footer lines.
    Returns: (header_lines, [(line_idx, row_dict), ...], footer_lines, header_meta)
    """
    parsed_rows: List[Tuple[int, Dict]] = []
    for i, line in enumerate(lines):
        toks = line.strip().split()
        rec = _try_parse_car_atom_tokens(toks)
        if rec is not None:
            parsed_rows.append((i, rec))

    if not parsed_rows:
        # Entire file is header/footer; treat as empty structure
        header_lines = lines[:]
        footer_lines: List[str] = []
        header_meta: Dict = {}
        return header_lines, [], footer_lines, header_meta

    first_idx = parsed_rows[0][0]
    last_idx = parsed_rows[-1][0]
    header_lines = lines[:first_idx]
    footer_lines = lines[last_idx + 1 :]

    # Extract minimal header meta
    header_text = "".join(header_lines)
    date_line = next((ln for ln in header_lines if ln.startswith(_DATE_PREFIX)), None)
    pbc_eq_line = next((ln for ln in header_lines if _PBC_EQ.match(ln.strip())), None)
    pbc_cell_line = next((ln for ln in header_lines if _PBC_CELL.match(ln.strip())), None)

    pbc_flag = False
    if pbc_eq_line:
        m = _PBC_EQ.match(pbc_eq_line.strip())
        if m:
            pbc_flag = (m.group(1).upper() == "ON")

    cell_params = None
    if pbc_cell_line:
        m2 = _PBC_CELL.match(pbc_cell_line.strip())
        if m2:
            a, b, c, alpha, beta, gamma = map(float, m2.groups())
            cell_params = {
                "a": a, "b": b, "c": c,
                "alpha": alpha, "beta": beta, "gamma": gamma,
            }

    header_meta = {
        "date_line": date_line,
        "pbc": pbc_flag,
        "cell_params": cell_params,
    }
    return header_lines, parsed_rows, footer_lines, header_meta


def load_car(path: str) -> USM:
    """
    Parse a CAR file into a USM object. Preserves header/footer lines verbatim.
    """
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        lines = f.readlines()  # keepends=True behavior by default for readlines

    header_lines, parsed_rows, footer_lines, header_meta = _scan_header_footer(lines)

    atoms_records: List[Dict] = []
    for _, rec in parsed_rows:
        row = {
            **rec,
            "mol_block_name": None,        # not present in CAR
            "connections_raw": None,       # MDF-only
        }
        atoms_records.append(row)

    atoms_df = pd.DataFrame(atoms_records)
    if not atoms_df.empty:
        atoms_df.insert(0, "aid", np.arange(len(atoms_df), dtype=np.int32))
    else:
        atoms_df = pd.DataFrame(columns=list(DTYPES.keys()))
        atoms_df["aid"] = atoms_df["aid"].astype(np.int32)

    atoms_df = _coerce_atom_dtypes(atoms_df)

    metadata: Dict = {
        "provenance": {
            "source_format": "car",
            "source_path": str(p),
            "parse_notes": "",
        },
        "preserved_text": {
            "car_header_lines": header_lines[:],   # includes original line endings
            "car_footer_lines": footer_lines[:],
        },
        "cell": {
            "pbc": bool(header_meta.get("pbc", False)),
        },
        "date_line": header_meta.get("date_line"),
    }
    # If cell parameters present, store them
    if header_meta.get("cell_params"):
        metadata["cell"].update(header_meta["cell_params"])

    return USM(atoms=atoms_df, metadata=metadata)


def _synthesize_car_header(usm: USM) -> List[str]:
    """
    Create a canonical CAR header when preserved header is not available.
    """
    hdr: List[str] = []
    hdr.append(f"{_ARCHIVE_PREFIX} 3\n")
    pbc_flag = "ON" if usm.pbc else "OFF"
    hdr.append(f"PBC={pbc_flag}\n")
    hdr.append(f"{_MATSCI_LINE}\n")
    # Synthesize a date like: !DATE Fri Jun 06 09:02:18 2025
    now = datetime.datetime.now()
    hdr.append("!DATE " + now.strftime("%a %b %d %H:%M:%S %Y") + "\n")

    # If pbc cell parameters exist, write a PBC cell line (canonical spacing)
    cell = usm.metadata.get("cell", {})
    if usm.pbc and all(k in cell for k in ("a", "b", "c", "alpha", "beta", "gamma")):
        hdr.append(
            f"PBC   {cell['a']:.4f}   {cell['b']:.4f}   {cell['c']:.4f}   "
            f"{cell['alpha']:.4f}   {cell['beta']:.4f}   {cell['gamma']:.4f} (P1)\n"
        )
    return hdr


def save_car(usm: USM, path: str, preserve_headers: bool = True) -> str:
    """
    Write a CAR file from a USM object.

    - If preserve_headers and preserved_text.car_header_lines exist, they are written verbatim.
    - Atom lines are rendered in DataFrame order with fixed numeric formats.
    - If preserve_headers and preserved_text.car_footer_lines exist, they are written verbatim;
      otherwise, a canonical 'end\\nend\\n' footer is appended.
    """
    usm.validate()  # ensure schema

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)

    preserved = usm.metadata.get("preserved_text", {})
    header_lines: Optional[List[str]] = None
    footer_lines: Optional[List[str]] = None

    if preserve_headers:
        header_lines = preserved.get("car_header_lines")
        footer_lines = preserved.get("car_footer_lines")

    with p.open("w", encoding="utf-8", newline="") as f:
        # Header
        if header_lines:
            for ln in header_lines:
                f.write(ln)
        else:
            for ln in _synthesize_car_header(usm):
                f.write(ln)

        # Atom lines (simple whitespace-separated formatting)
        # Coordinates: 9 decimals; charge: 3 decimals to match examples
        for _, row in usm.atoms.sort_values(by=["aid"]).iterrows():
            name = str(row["name"])
            x = float(row["x"]); y = float(row["y"]); z = float(row["z"])
            mol_label = str(row["mol_label"]) if row["mol_label"] is not None else "XXXX"
            mol_index = int(row["mol_index"]) if not pd.isna(row["mol_index"]) else 1
            atom_type = str(row["atom_type"]) if row["atom_type"] is not None else "X"
            element = str(row["element"]) if row["element"] is not None else "X"
            charge = float(row["charge"]) if not pd.isna(row["charge"]) else 0.0

            line = (
                f"{name} "
                f"{x:.9f} {y:.9f} {z:.9f} "
                f"{mol_label} {mol_index} {atom_type} {element} {charge:.3f}\n"
            )
            f.write(line)

        # Footer
        if footer_lines:
            for ln in footer_lines:
                f.write(ln)
        else:
            f.write("end\n")
            f.write("end\n")

    return str(p)


__all__ = ["USM", "load_car", "save_car"]