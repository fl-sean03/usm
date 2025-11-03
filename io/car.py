from __future__ import annotations

import re
from typing import List, Dict, Any, Tuple, Optional
from datetime import datetime
import math

import pandas as pd
import numpy as np

from usm.core.model import USM


# Regex to parse CAR atom lines (Materials Studio)
# Example:
# H1  2.568056419  2.265759415  2.682995600 XXXX 1  H*  H  0.410
ATOM_LINE_RE = re.compile(
    r"""^\s*
    (?P<name>\S+)\s+
    (?P<x>[-+]?[\d\.Ee]+)\s+
    (?P<y>[-+]?[\d\.Ee]+)\s+
    (?P<z>[-+]?[\d\.Ee]+)\s+
    (?P<mol_label>\S+)\s+
    (?P<mol_index>\d+)\s+
    (?P<atom_type>\S+)\s+
    (?P<element>[A-Za-z]+)\s+
    (?P<charge>[-+]?[\d\.Ee]+)
    \s*$
    """,
    re.VERBOSE,
)


def _parse_pbc_switch(line: str) -> Optional[bool]:
    # PBC=ON / PBC=OFF
    s = line.strip()
    if s.startswith("PBC="):
        return "ON" in s.upper()
    return None


def _parse_pbc_line(line: str) -> Optional[Dict[str, Any]]:
    # Example:
    # PBC   24.7290   23.7945   40.9810   90.0000   90.0000   90.0000 (P1)
    s = line.strip()
    if not s.startswith("PBC "):
        return None
    parts = s.split()
    # Expect: ["PBC", a, b, c, alpha, beta, gamma, "(P1)"]
    if len(parts) < 7:
        return None
    try:
        a = float(parts[1])
        b = float(parts[2])
        c = float(parts[3])
        alpha = float(parts[4])
        beta = float(parts[5])
        gamma = float(parts[6])
        return dict(pbc=True, a=a, b=b, c=c, alpha=alpha, beta=beta, gamma=gamma, spacegroup="")
    except Exception:
        return None


def _detect_atom_line(line: str) -> bool:
    return ATOM_LINE_RE.match(line) is not None


def _split_header_atoms_footer(lines: List[str]) -> Tuple[List[str], List[str], List[str]]:
    header: List[str] = []
    atoms: List[str] = []
    footer: List[str] = []
    mode = "header"
    for ln in lines:
        if mode == "header":
            if _detect_atom_line(ln):
                mode = "atoms"
                atoms.append(ln.rstrip("\n"))
            else:
                header.append(ln.rstrip("\n"))
        elif mode == "atoms":
            if ln.strip().lower().startswith("end"):
                mode = "footer"
                footer.append(ln.rstrip("\n"))
            else:
                atoms.append(ln.rstrip("\n"))
        else:
            footer.append(ln.rstrip("\n"))
    return header, atoms, footer


def _parse_atoms_atom_lines(atom_lines: List[str]) -> List[Dict[str, Any]]:
    recs: List[Dict[str, Any]] = []
    for ln in atom_lines:
        m = ATOM_LINE_RE.match(ln)
        if not m:
            # tolerate blank or comment-like lines in atom block
            if ln.strip() == "" or ln.strip().startswith("!"):
                continue
            raise ValueError(f"Unrecognized CAR atom line: {ln}")
        gd = m.groupdict()
        recs.append(
            {
                "name": gd["name"],
                "element": gd["element"],
                "atom_type": gd["atom_type"],
                "charge": float(gd["charge"]),
                "x": float(gd["x"]),
                "y": float(gd["y"]),
                "z": float(gd["z"]),
                "mol_label": gd["mol_label"],
                "mol_index": int(gd["mol_index"]),
                "mol_block_name": "",
                # MDF carry-through (not applicable to CAR; left null)
                "isotope": pd.NA,
                "formal_charge": pd.NA,
                "switching_atom": pd.NA,
                "oop_flag": pd.NA,
                "chirality_flag": pd.NA,
                "occupancy": np.nan,
                "xray_temp_factor": np.nan,
                "charge_group": pd.NA,
                "connections_raw": pd.NA,
            }
        )
    return recs


def load_car(path: str) -> USM:
    """
    Load a Materials Studio CAR file into a USM instance, preserving header/footer text.

    Scope: PBC=OFF and simple PBC line parsing for v0.1 (sufficient for WAT/DOP).
    """
    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    header_lines, atom_lines, footer_lines = _split_header_atoms_footer(lines)

    # Parse PBC switch and optional 'PBC a b c alpha beta gamma ...' line from header
    pbc_flag: Optional[bool] = None
    cell = dict(pbc=False, a=np.nan, b=np.nan, c=np.nan, alpha=np.nan, beta=np.nan, gamma=np.nan, spacegroup="")
    date_line = ""
    for h in header_lines:
        if h.strip().startswith("!DATE"):
            date_line = h
        sw = _parse_pbc_switch(h)
        if sw is not None:
            pbc_flag = sw
        pc = _parse_pbc_line(h)
        if pc is not None:
            cell.update(pc)

    if pbc_flag is not None:
        cell["pbc"] = bool(pbc_flag)

    atom_recs = _parse_atoms_atom_lines(atom_lines)
    atoms_df = pd.DataFrame(atom_recs)

    preserved_text = {
        "car_header_lines": header_lines,
        "car_footer_lines": footer_lines,
    }
    provenance = {
        "source_format": "car",
        "source_path": path,
        "date_line": date_line,
        "parse_notes": "",
    }

    usm = USM(atoms=atoms_df, bonds=None, molecules=None, cell=cell, provenance=provenance, preserved_text=preserved_text)
    usm.validate_basic()
    return usm


def _format_float(val: float, width: int = 14, prec: int = 9) -> str:
    if val is None or (isinstance(val, float) and (math.isnan(val) or math.isinf(val))):
        val = 0.0
    return f"{val:>{width}.{prec}f}"


def _format_charge(val: float, width: int = 7, prec: int = 3) -> str:
    if val is None or (isinstance(val, float) and (math.isnan(val) or math.isinf(val))):
        val = 0.0
    return f"{val:>{width}.{prec}f}"


def save_car(usm: USM, path: str, preserve_headers: bool = True) -> str:
    """
    Save a USM instance to a CAR file.

    - If preserve_headers is True and preserved header/footer lines exist, write them byte-for-byte.
    - Otherwise synthesize a canonical header/footer (v0.1).
    - Atom rows are written in aid order with stable formatting.
    """
    lines: List[str] = []

    header_lines = (usm.preserved_text or {}).get("car_header_lines") if preserve_headers else None
    footer_lines = (usm.preserved_text or {}).get("car_footer_lines") if preserve_headers else None

    if header_lines:
        lines.extend(header_lines)
    else:
        lines.append("!BIOSYM archive 3")
        lines.append("PBC=ON" if bool(usm.cell.get("pbc", False)) else "PBC=OFF")
        lines.append("Materials Studio Generated CAR File")
        # Preserve original date if present, else synthesize
        date_line = (usm.provenance or {}).get("date_line", "")
        if date_line and date_line.strip().startswith("!DATE"):
            lines.append(date_line)
        else:
            lines.append("!DATE " + datetime.now().strftime("%a %b %d %H:%M:%S %Y"))
        # Optional explicit PBC parameters line
        if bool(usm.cell.get("pbc", False)):
            a = usm.cell.get("a", np.nan)
            b = usm.cell.get("b", np.nan)
            c = usm.cell.get("c", np.nan)
            alpha = usm.cell.get("alpha", np.nan)
            beta = usm.cell.get("beta", np.nan)
            gamma = usm.cell.get("gamma", np.nan)
            if all(np.isfinite([a, b, c, alpha, beta, gamma])):
                lines.append(
                    f"PBC{_format_float(a, 9, 4)}{_format_float(b, 9, 4)}{_format_float(c, 9, 4)}"
                    f"{_format_float(alpha, 9, 4)}{_format_float(beta, 9, 4)}{_format_float(gamma, 9, 4)} (P1)"
                )

    # Atom lines
    atoms = usm.atoms.sort_values(by=["aid"]).reset_index(drop=True)
    for _, row in atoms.iterrows():
        name = (row.get("name") or "").strip()
        x = float(row.get("x", 0.0))
        y = float(row.get("y", 0.0))
        z = float(row.get("z", 0.0))
        mol_label = (row.get("mol_label") or "XXXX").strip()
        mol_index = int(row.get("mol_index") or 1)
        atom_type = (row.get("atom_type") or "").strip()
        element = (row.get("element") or "").strip()
        charge = float(row.get("charge", 0.0))

        # Compose with spacing similar to examples
        line = (
            f"{name:<8s}"
            f"{_format_float(x)}{_format_float(y)}{_format_float(z)} "
            f"{mol_label:>4s} {mol_index:<6d}"
            f"{atom_type:>8s} {element:>2s}"
            f"{_format_charge(charge)}"
        )
        lines.append(line)

    # Footer
    if footer_lines:
        lines.extend(footer_lines)
    else:
        lines.append("end")
        lines.append("end")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines).rstrip("\n") + "\n")

    return path


__all__ = ["load_car", "save_car"]