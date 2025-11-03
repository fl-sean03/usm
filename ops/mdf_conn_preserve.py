from __future__ import annotations

"""
MDF connections preservation helpers.

Centralizes common utilities used by MXN workspaces to preserve or cleanse the
'connections_raw' column from MDF when atoms are removed or replaced.

Exports:
- key_from_row(row) -> str
- parse_base_token(tok) -> Optional[str]
- cleanse_connections_raw(raw, remove_keys) -> str
- build_conn_map(atoms_df) -> Dict[str, str]
"""

from typing import Dict, Optional, Set
import re
import pandas as pd


# Token pattern examples:
#   "XXXX_23:Al1"
#   "XXXX_23:Al1/L1"            (with bond property suffix)
#   "XXXX_23:Al1%comment"       (with comment suffix)
#   "XXXX_23:Al1/L1%comment"    (with both)
TOKEN_RE = re.compile(r"^([A-Za-z0-9_]+_\d+:[^%\s/]+)(?:/[^%\s]+)?(?:%.*)?$")


def key_from_row(row: pd.Series) -> str:
    """
    Compose a stable key for a row: 'mol_label_mol_index:name'
    - mol_label: defaults to 'XXXX' when NaN/empty
    - mol_index: defaults to 1 when NaN/invalid
    - name: defaults to 'X' when NaN/empty
    """
    lbl = row.get("mol_label")
    try:
        if pd.isna(lbl) or lbl is None or str(lbl).strip() == "":
            lbl = "XXXX"
    except Exception:
        lbl = lbl or "XXXX"

    idx = row.get("mol_index")
    try:
        idx = 1 if pd.isna(idx) else int(idx)
    except Exception:
        idx = 1

    nm = row.get("name")
    try:
        if pd.isna(nm) or nm is None or str(nm).strip() == "":
            nm = "X"
    except Exception:
        nm = nm or "X"

    return f"{lbl}_{idx}:{nm}"


def parse_base_token(tok: str) -> Optional[str]:
    """
    Extract the base token (without '/...' or '%...') from a connections token.
    Returns None if the token does not match expected pattern.
    """
    if not isinstance(tok, str):
        return None
    m = TOKEN_RE.match(tok.strip())
    if not m:
        return None
    return m.group(1)


def cleanse_connections_raw(raw: str, remove_keys: Set[str]) -> str:
    """
    Remove references to any keys in 'remove_keys' from a raw connections token string.

    Parameters:
    - raw: original connections_raw string (space-separated tokens)
    - remove_keys: set of base keys like 'XXXX_23:Al1' that should be removed

    Returns:
    - A space-joined string with offending references removed; empty string if none.
    """
    if not isinstance(raw, str) or not raw.strip():
        return ""
    toks = raw.strip().split()
    kept: list[str] = []
    for t in toks:
        base = parse_base_token(t)
        if base and base in remove_keys:
            continue
        kept.append(t)
    return " ".join(kept)


def build_conn_map(atoms_df: pd.DataFrame) -> Dict[str, str]:
    """
    Build a mapping key -> connections_raw string for the atoms table.

    Expects columns: 'mol_label', 'mol_index', 'name', and 'connections_raw' (optional).
    If 'connections_raw' is missing, returns an empty map.
    """
    conn_map: Dict[str, str] = {}
    if atoms_df is None or "connections_raw" not in atoms_df.columns:
        return conn_map
    # Iterate in row order to preserve the last occurrence policy naturally if duplicates exist.
    for _, r in atoms_df.iterrows():
        k = key_from_row(r)
        raw = r.get("connections_raw")
        if isinstance(raw, str) and raw.strip():
            conn_map[k] = raw.strip()
        else:
            conn_map[k] = ""
    return conn_map


__all__ = [
    "key_from_row",
    "parse_base_token",
    "cleanse_connections_raw",
    "build_conn_map",
]