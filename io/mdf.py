"""
MDF (Material Design File) format I/O for USM.

This module provides functions for reading and writing Materials Studio
MDF format files to/from USM (Universal Structure Model) instances.

Public API:
    load_mdf(path) -> USM: Load an MDF file into a USM instance
    save_mdf(usm, path, ...) -> str: Save a USM instance to an MDF file
"""
from __future__ import annotations

from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path

import numpy as np
import pandas as pd

from usm.core.model import USM
from ._mdf_parser import (
    MDF_LINE_RE,
    split_sections as _split_sections,
    current_molecule_name_from_header as _current_molecule_name_from_header,
    molecule_order as _molecule_order,
    parse_atom_line as _parse_atom_line,
    build_bonds_from_connections as _build_bonds_from_connections,
)


def load_mdf(path: str) -> USM:
    """
    Load a Materials Studio MDF file into a USM instance, parsing topology and connections.

    Notes:
    - Coordinates are not present in MDF; x,y,z remain NaN (to be populated from CAR or other sources).
    - connections_raw is preserved as text; bonds are inferred best-effort and deduplicated.
    """
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        lines = f.readlines()

    header, atom_lines, footer = _split_sections(lines)
    mol_block_name = _current_molecule_name_from_header(header)
    mol_order = _molecule_order(header)

    date_line = ""
    for h in header:
        if h.strip().lower().startswith("!date"):
            date_line = h

    atom_records: List[Dict[str, Any]] = []
    for ln in atom_lines:
        s = ln.strip()
        if not s:
            continue
        # Tolerate comments and non-atom lines within atom block (e.g., "!")
        if s.startswith("!"):
            continue
        if MDF_LINE_RE.match(ln) is None:
            continue
        atom_records.append(_parse_atom_line(ln, mol_block_name))

    atoms_df = pd.DataFrame(atom_records)
    # USM will add 'aid' and dtype-normalize
    usm = USM(
        atoms=atoms_df,
        bonds=None,
        molecules=None,
        cell=dict(pbc=False, a=np.nan, b=np.nan, c=np.nan, alpha=np.nan, beta=np.nan, gamma=np.nan, spacegroup=""),
        provenance={
            "source_format": "mdf",
            "source_path": str(p),
            "date_line": date_line,
            "parse_notes": "",
        },
        preserved_text={
            "mdf_header_lines": header,
            "mdf_molecule_order": mol_order,
            "mdf_footer_lines": footer,
        },
    )

    # Build bonds from connections_raw
    bonds_df = _build_bonds_from_connections(usm.atoms)
    if len(bonds_df) > 0:
        usm.bonds = bonds_df

    usm.validate_basic()
    return usm


def _format_float_mdf(val: Any, prec: int = 4) -> str:
    """Format a float value for MDF output with specified precision."""
    try:
        f = float(val)
        if np.isnan(f):
            f = 0.0
        return f"{f:.{prec}f}"
    except Exception:
        return f"{0.0:.{prec}f}"


def _order_token(order: Optional[float]) -> Optional[str]:
    """Convert bond order to a token string, omitting if order is 1.0."""
    if order is None or np.isclose(float(order), 1.0):
        return None
    # Minimal decimal representation (e.g., 1.5 instead of 1.5000)
    return f"{float(order):g}"


def _compose_connections_for_atom(
    row: pd.Series,
    aid_to_info: Dict[int, Tuple[str, int, str]],
    adj_list: Dict[int, List[Tuple[int, float]]],
    write_normalized_connections: bool
) -> str:
    """Compose the connections field string for a single atom."""
    # Prefer raw connections when available and lossless mode
    if not write_normalized_connections:
        raw = row.get("connections_raw")
        if isinstance(raw, str) and raw.strip():
            return raw.strip()

    # Current atom identity
    aid = int(row["aid"])
    neighbors = adj_list.get(aid)
    if not neighbors:
        return ""

    src_label = str(row.get("mol_label"))
    try:
        src_index = int(row.get("mol_index"))
    except Exception:
        src_index = 1

    tokens: list[str] = []
    for other_aid, order_val in neighbors:
        info = aid_to_info.get(other_aid)
        if not info:
            continue
        o_label, o_index, o_name = info
        
        same_mol = (o_label == src_label) and (int(o_index) == int(src_index))
        token_base = o_name if same_mol else f"{o_label}_{int(o_index)}:{o_name}"
        
        ord_token = _order_token(order_val)
        tokens.append(f"{token_base}/{ord_token}" if ord_token else token_base)

    return " ".join(tokens)


def save_mdf(usm: USM, path: str, preserve_headers: bool = True, write_normalized_connections: bool = False) -> str:
    """
    Save a USM instance to an MDF file.

    - If preserve_headers and preserved header/footer exist, write them byte-for-byte.
    - Otherwise, synthesize a canonical header with @column block and a single @molecule using the first mol_block_name.
    - Write atom lines with MDF's 12 columns and connections (raw or normalized).
    """
    from datetime import datetime

    header_lines = None
    footer_lines = None
    if preserve_headers:
        header_lines = (usm.preserved_text or {}).get("mdf_header_lines")
        footer_lines = (usm.preserved_text or {}).get("mdf_footer_lines")

    lines: list[str] = []

    if header_lines:
        lines.extend(header_lines)
    else:
        # Synthesize minimal header
        lines.append("!BIOSYM molecular_data 4")
        lines.append("")
        date_line = (usm.provenance or {}).get("date_line", "")
        if date_line and date_line.strip().lower().startswith("!date"):
            lines.append(date_line)
        else:
            lines.append("!Date: " + datetime.now().strftime("%a %b %d %H:%M:%S %Y") + "   USM Generated MDF file")
        lines.append("")
        lines.append("#topology")
        lines.append("")
        # Canonical @column block (1..12)
        lines.append("@column 1 element")
        lines.append("@column 2 atom_type")
        lines.append("@column 3 charge_group")
        lines.append("@column 4 isotope")
        lines.append("@column 5 formal_charge")
        lines.append("@column 6 charge")
        lines.append("@column 7 switching_atom")
        lines.append("@column 8 oop_flag")
        lines.append("@column 9 chirality_flag")
        lines.append("@column 10 occupancy")
        lines.append("@column 11 xray_temp_factor")
        lines.append("@column 12 connections")
        lines.append("")
        # @molecule: prefer preserved name from atoms if consistent
        mol_name = ""
        if "mol_block_name" in usm.atoms.columns:
            vals = usm.atoms["mol_block_name"].dropna().unique().tolist()
            if len(vals) == 1:
                mol_name = str(vals[0])
        if not mol_name:
            mol_name = "USM_Molecule"
        lines.append(f"@molecule {mol_name}")
        lines.append("")

    # Atom lines in aid order
    atoms = usm.atoms.sort_values(by=["aid"]).reset_index(drop=True)

    # Precompute maps for O(N) performance
    # aid -> (label, index, name)
    try:
        aid_vec = atoms["aid"].to_numpy().astype(int)
    except Exception:
        aid_vec = atoms["aid"].astype(int).to_numpy()
    
    label_list = atoms["mol_label"].astype("string").fillna("XXXX").astype(str).tolist()
    try:
        index_vec = atoms["mol_index"].to_numpy().astype(int)
    except Exception:
        index_vec = pd.to_numeric(atoms["mol_index"], errors="coerce").fillna(1).astype(int).to_numpy()
    name_list = atoms["name"].astype("string").fillna("X").astype(str).tolist()
    
    aid_to_info = {int(a): (label_list[i], int(index_vec[i]), name_list[i]) for i, a in enumerate(aid_vec)}

    # Precompute adjacency list
    adj_list: Dict[int, List[Tuple[int, float]]] = {}
    if usm.bonds is not None and len(usm.bonds) > 0:
        for _, br in usm.bonds.iterrows():
            a1 = int(br["a1"])
            a2 = int(br["a2"])
            order = float(br.get("order", 1.0))
            if a1 not in adj_list: adj_list[a1] = []
            if a2 not in adj_list: adj_list[a2] = []
            adj_list[a1].append((a2, order))
            adj_list[a2].append((a1, order))

    for _, row in atoms.iterrows():
        mol_label = "XXXX" if pd.isna(row.get("mol_label")) else str(row.get("mol_label"))
        _mol_index_val = row.get("mol_index")
        mol_index = 1 if pd.isna(_mol_index_val) else int(_mol_index_val)
        name = "X" if pd.isna(row.get("name")) else str(row.get("name"))
        prefix = f"{mol_label}_{mol_index}:{name}"

        element = "?" if pd.isna(row.get("element")) else str(row.get("element"))
        atom_type = "?" if pd.isna(row.get("atom_type")) else str(row.get("atom_type"))
        charge_group = "?" if pd.isna(row.get("charge_group")) else str(row.get("charge_group"))
        isotope = "?" if pd.isna(row.get("isotope")) else str(row.get("isotope"))
        formal_charge = "0" if pd.isna(row.get("formal_charge")) else str(row.get("formal_charge"))

        charge = _format_float_mdf(row.get("charge"), 4)

        # Integer-like flags defaulting to 0
        def _to_int(val: Any, default: int = 0) -> int:
            try:
                if pd.isna(val):
                    return default
                return int(val)
            except Exception:
                return default

        switching_atom = _to_int(row.get("switching_atom"), 0)
        oop_flag = _to_int(row.get("oop_flag"), 0)
        chirality_flag = _to_int(row.get("chirality_flag"), 0)

        occupancy = row.get("occupancy")
        xray_temp_factor = row.get("xray_temp_factor")
        occ_val = 1.0 if pd.isna(occupancy) else occupancy
        xrf_val = 0.0 if pd.isna(xray_temp_factor) else xray_temp_factor
        occ_str = _format_float_mdf(occ_val, 4)
        xrf_str = _format_float_mdf(xrf_val, 4)

        # Preserve formal_charge token without truncation; align to width 3 only for short tokens
        fc_field = formal_charge if len(formal_charge) > 3 else f"{formal_charge:>3s}"

        conns = _compose_connections_for_atom(row, aid_to_info, adj_list, write_normalized_connections)
        line = f"{prefix:<18s} {element:>2s} {atom_type:<6s} {charge_group:<6s} {isotope:>5s} {fc_field} {charge:>8s} {switching_atom:d} {oop_flag:d} {chirality_flag:d} {occ_str:>7s}  {xrf_str:>7s}"
        if conns:
            line = f"{line} {conns}"
        lines.append(line)

    lines.append("")
    if footer_lines:
        lines.extend(footer_lines)
    else:
        lines.append("#end")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines).rstrip("\n") + "\n")

    return path


__all__ = ["load_mdf", "save_mdf"]
