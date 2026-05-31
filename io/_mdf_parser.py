"""
MDF format parsing internals.

This module contains regex patterns and helper functions for parsing
Materials Studio MDF (Material Design File) format files.

Private module - not for direct import. Use `usm.io.mdf` instead.
"""
from __future__ import annotations

import re
from typing import List, Dict, Any, Optional, Tuple

import numpy as np
import pandas as pd


# Regex to parse an MDF atom/topology line into named fields.
# Example (WAT):
# XXXX_1:H1           H  h*      ?     0  0     0.4100 0 0 8 1.0000  0.0000 O1
# Example (DOP, with formal charge '1+'):
# MOL2_1:N1           N  NH1     ?     0  1+   -0.1000 0 0 8 1.0000  0.0000 H8 C8 H9 H12
MDF_LINE_RE = re.compile(
    r"""^\s*
    (?P<prefix>\S+)                                        # e.g., MOL2_1:C1
    \s+
    (?P<element>\S+)\s+
    (?P<atom_type>\S+)\s+
    (?P<charge_group>\S+)\s+
    (?P<isotope>\S+)\s+
    (?P<formal_charge>\S+)\s+
    (?P<charge>[-+]?\d+(?:\.\d+)?)\s+
    (?P<switching_atom>-?\d+)\s+
    (?P<oop_flag>-?\d+)\s+
    (?P<chirality_flag>-?\d+)\s+
    (?P<occupancy>[-+]?\d+(?:\.\d+)?)\s+
    (?P<xray_temp_factor>[-+]?\d+(?:\.\d+)?)
    (?:\s+(?P<connections>.*))?
    \s*$
    """,
    re.VERBOSE,
)

# Prefix pattern e.g. "MOL2_1:C1" or "XXXX_1:H1"
PREFIX_RE = re.compile(r"^(?P<mol_label>[A-Za-z0-9]+)_(?P<mol_index>\d+):(?P<name>\S+)$")


# Materials Studio MDF connection-token periodic-image suffix:
#
#   ``%abc#i``
#
# where ``a``, ``b``, ``c`` are each a single signed digit shift along the
# corresponding cell axis (a, b, c), and ``i`` is the Materials Studio image
# index (positive integer; internal bookkeeping with no geometric meaning, so
# we preserve it on ``connections_raw`` but not on the parsed Bond).
#
# Examples seen in Sean's real MDF corpus:
#   ``%100#3``    → ix=+1, iy=0,  iz=0
#   ``%010#1``    → ix=0,  iy=+1, iz=0
#   ``%0-10#1``   → ix=0,  iy=-1, iz=0
#   ``%110#3``    → ix=+1, iy=+1, iz=0
#   ``%-1-10#5``  → ix=-1, iy=-1, iz=0
#
# A complete corpus scan (~/Backups/Dropbox/**/*.mdf) found only six distinct
# patterns, all matching this regex. See docs/MDF_PARSER_AUDIT.md §1.
_MDF_PBC_SUFFIX_RE = re.compile(r"%(-?\d)(-?\d)(-?\d)#(\d+)")


def _parse_image_suffix(token: str) -> Tuple[str, Tuple[int, int, int]]:
    """Split a Materials Studio MDF connection token's PBC suffix.

    Wire format::

        NAME[%abc#i][/order]

    where ``a``, ``b``, ``c`` are each a single signed digit shift along the
    corresponding cell axis, and ``i`` is the Materials Studio image index
    (positive integer; internal bookkeeping with no geometric meaning).

    The image index ``i`` is **not** returned — it is preserved on
    ``connections_raw`` for any caller that needs the original text. The
    Bond model's ``(ix, iy, iz)`` carries the full geometric content of the
    suffix.

    Parameters
    ----------
    token : str
        A single whitespace-delimited connection token, e.g. ``"cb%100#1"``.
        May or may not include a ``%abc#i`` suffix and/or a ``/order`` part.

    Returns
    -------
    (bare_token, shift) : tuple[str, tuple[int, int, int]]
        ``bare_token`` is the input with the ``%abc#i`` suffix removed (the
        ``/order`` part, if any, is left intact). ``shift`` is ``(ix, iy, iz)``,
        or ``(0, 0, 0)`` if no recognizable suffix is present.

    Examples
    --------
    >>> _parse_image_suffix("cb%100#1")
    ('cb', (1, 0, 0))
    >>> _parse_image_suffix("H%0-10#1")
    ('H', (0, -1, 0))
    >>> _parse_image_suffix("XXXX_825:C%0-10#1")
    ('XXXX_825:C', (0, -1, 0))
    >>> _parse_image_suffix("H")
    ('H', (0, 0, 0))
    >>> _parse_image_suffix("Al1%-1-10#5/1.5")
    ('Al1/1.5', (-1, -1, 0))
    """
    if not isinstance(token, str) or not token:
        return token, (0, 0, 0)
    m = _MDF_PBC_SUFFIX_RE.search(token)
    if m is None:
        return token, (0, 0, 0)
    ix = int(m.group(1))
    iy = int(m.group(2))
    iz = int(m.group(3))
    bare = token[: m.start()] + token[m.end():]
    return bare, (ix, iy, iz)


def split_sections(lines: List[str]) -> Tuple[List[str], List[str], List[str]]:
    """
    Split MDF file into header lines, atom/topology lines, and footer lines.
    We consider the first line that matches MDF_LINE_RE as the start of atoms section,
    and "#end" (case-insensitive) as a footer marker thereafter.
    """
    header: List[str] = []
    atoms: List[str] = []
    footer: List[str] = []

    mode = "header"
    for ln in lines:
        s = ln.rstrip("\n")
        if mode == "header":
            if MDF_LINE_RE.match(s):
                mode = "atoms"
                atoms.append(s)
            else:
                header.append(s)
        elif mode == "atoms":
            if s.strip().lower().startswith("#end"):
                mode = "footer"
                footer.append(s)
            else:
                atoms.append(s)
        else:
            footer.append(s)
    return header, atoms, footer


def current_molecule_name_from_header(header_lines: List[str]) -> Optional[str]:
    """
    For simple single-@molecule files, capture the latest @molecule name.
    If multiple @molecule sections exist, we keep order in preserved_text and still
    assign the last seen name to atoms following it (v0.1).
    """
    mol_name: Optional[str] = None
    for h in header_lines:
        hs = h.strip()
        if hs.lower().startswith("@molecule"):
            # Format: @molecule Name (name may contain spaces but examples show single token)
            parts = hs.split(maxsplit=1)
            if len(parts) == 2:
                mol_name = parts[1].strip()
            else:
                mol_name = ""
    return mol_name


def molecule_order(header_lines: List[str]) -> List[str]:
    """Extract the order of @molecule declarations from header lines."""
    order: List[str] = []
    for h in header_lines:
        hs = h.strip()
        if hs.lower().startswith("@molecule"):
            parts = hs.split(maxsplit=1)
            if len(parts) == 2:
                order.append(parts[1].strip())
            else:
                order.append("")
    return order


def parse_atom_line(line: str, default_mol_block_name: Optional[str]) -> Dict[str, Any]:
    """Parse a single MDF atom line into a dictionary of fields."""
    m = MDF_LINE_RE.match(line)
    if not m:
        raise ValueError(f"Unrecognized MDF atom line: {line}")
    gd = m.groupdict()

    # Parse the prefix into molecule label/index and atom name
    pm = PREFIX_RE.match(gd["prefix"])
    if not pm:
        raise ValueError(f"Unrecognized MDF atom prefix: {gd['prefix']}")
    mol_label = pm.group("mol_label")
    mol_index = int(pm.group("mol_index"))
    name = pm.group("name")

    conn = gd.get("connections") or ""
    # normalize spacing minimally (preserve token text order)
    connections_raw = conn.rstrip()

    rec: Dict[str, Any] = {
        "name": name,
        "element": gd["element"],
        "atom_type": gd["atom_type"],
        "charge_group": gd["charge_group"],
        "isotope": gd["isotope"],
        "formal_charge": gd["formal_charge"],  # keep string (can be "1+")
        "charge": float(gd["charge"]),
        "switching_atom": int(gd["switching_atom"]),
        "oop_flag": int(gd["oop_flag"]),
        "chirality_flag": int(gd["chirality_flag"]),
        "occupancy": float(gd["occupancy"]),
        "xray_temp_factor": float(gd["xray_temp_factor"]),
        "connections_raw": connections_raw if connections_raw else pd.NA,
        # Common USM columns
        "x": np.nan,
        "y": np.nan,
        "z": np.nan,
        "mol_label": mol_label,
        "mol_index": mol_index,
        "mol_block_name": default_mol_block_name or "",
    }
    return rec


def build_bonds_from_connections(atoms_df: pd.DataFrame) -> pd.DataFrame:
    """Convert ``connections_raw`` into a normalized undirected bonds table.

    Token grammar supported (one space-delimited token per partner)::

        NAME[%abc#i][/order]
        LABEL_INDEX:NAME[%abc#i][/order]

    - ``NAME`` / ``LABEL_INDEX:NAME`` resolves the partner atom. A bare
      ``NAME`` is scoped to the source atom's ``(mol_label, mol_index)``.
    - ``%abc#i`` is the Materials Studio periodic-image suffix; ``a``, ``b``,
      ``c`` are signed single digits giving the partner image's shift along
      the ``(a, b, c)`` cell axes. ``i`` is the Materials Studio image index
      and is preserved only on ``connections_raw``. See
      ``_parse_image_suffix`` for the grammar and ``docs/MDF_PARSER_AUDIT.md``
      for the design discussion.
    - ``/order`` is the bond order; defaults to ``1.0``.

    Bond rows are deduplicated by ``(a1, a2, ix, iy, iz)`` *after*
    canonicalization (``a1 <= a2`` with sign-flipped shift on swap), so two
    bonds between the same pair into different periodic images are kept as
    distinct rows. A self-image bond (``A — A%abc#i``) is emitted with the
    recovered shift; the USM constructor renormalizes the shift to
    lexicographically positive (see ``USM.__post_init__``).

    Tokens whose partner can't be resolved (unknown ``NAME`` in the source
    scope, or unknown ``LABEL_INDEX:NAME``) are silently skipped — matching
    the historical permissive behavior.
    """
    # Map (mol_label, mol_index, name) -> aid
    key_to_aid: Dict[Tuple[str, int, str], int] = {}
    for _, r in atoms_df[["aid", "mol_label", "mol_index", "name"]].iterrows():
        key = (str(r["mol_label"]), int(r["mol_index"]), str(r["name"]))
        key_to_aid[key] = int(r["aid"])

    tgt_prefix_re = re.compile(r"^(?P<label>[A-Za-z0-9]+)_(?P<idx>\d+):(?P<name>\S+)$")

    def parse_target_token(
        tok: str, src_label: str, src_index: int
    ) -> Tuple[str, int, str, Optional[float], Optional[str], Tuple[int, int, int]]:
        # Peel off the PBC image suffix first. Materials Studio emits the
        # ``%abc#i`` suffix on the bare name, before the optional ``/order``,
        # but the regex is unambiguous either way so we don't depend on order.
        bare, shift = _parse_image_suffix(tok)

        # Split order part if present
        order_val: Optional[float] = None
        order_raw: Optional[str] = None
        base = bare
        if "/" in bare:
            base, order_raw = bare.split("/", 1)
            try:
                order_val = float(order_raw)
            except Exception:
                order_val = None
        # Defensive: strip any leftover ``%...`` we didn't recognize. The
        # known grammar is exhausted above, but Materials Studio occasionally
        # emits non-PBC ``%`` annotations that should not contaminate the
        # partner name. Treat unrecognized suffixes as same-cell.
        if "%" in base:
            base = base.split("%", 1)[0]

        label = src_label
        idx = src_index
        name = base
        # If fully qualified target is provided, use it
        m = tgt_prefix_re.match(base)
        if m:
            label = m.group("label")
            idx = int(m.group("idx"))
            name = m.group("name")
        return label, idx, name, order_val, order_raw, shift

    bonds: List[Dict[str, Any]] = []
    # Dedup key includes the canonical shift so that distinct PBC images of
    # the same pair are kept as distinct rows. See docs/MDF_PARSER_AUDIT.md §4.2.
    seen: set = set()

    for _, r in atoms_df.iterrows():
        src_aid = int(r["aid"])
        src_label = str(r["mol_label"])
        src_index = int(r["mol_index"])
        raw = r.get("connections_raw")
        if pd.isna(raw) or not str(raw).strip():
            continue
        tokens = str(raw).split()
        for tok in tokens:
            t_label, t_idx, t_name, order_val, order_raw, shift = parse_target_token(
                tok, src_label, src_index
            )
            key = (t_label, t_idx, t_name)
            tgt_aid = key_to_aid.get(key)
            if tgt_aid is None:
                # Could not resolve token; skip
                continue

            # Canonicalize (a1, a2) with a1 <= a2, flipping the shift sign on
            # swap so that ``shift`` always describes the offset of ``a2``'s
            # image relative to ``a1``'s canonical position. The USM
            # constructor performs the same renormalization, but we do it
            # here as well so the dedup key sees the canonical form.
            a1 = src_aid
            a2 = int(tgt_aid)
            ix, iy, iz = shift
            if a2 < a1:
                a1, a2 = a2, a1
                ix, iy, iz = -ix, -iy, -iz

            # Self-image bond (a1 == a2, non-zero shift): keep it; the USM
            # constructor will lex-positive-normalize the shift.
            # Same-cell self-bond (a1 == a2, zero shift): drop it (legacy
            # behavior; a same-cell self-bond is malformed input).
            if a1 == a2 and (ix, iy, iz) == (0, 0, 0):
                continue

            pair_key = (a1, a2, ix, iy, iz)
            if pair_key in seen:
                continue
            seen.add(pair_key)

            bonds.append(
                {
                    "a1": a1,
                    "a2": a2,
                    "ix": int(ix),
                    "iy": int(iy),
                    "iz": int(iz),
                    "order": float(order_val) if order_val is not None else float(1.0),
                    "order_raw": order_raw if order_raw is not None else pd.NA,
                    "type": pd.NA,
                    "source": "mdf.connections",
                    "mol_index": src_index,
                    "notes": pd.NA,
                }
            )

    if not bonds:
        return pd.DataFrame(
            columns=[
                "a1", "a2", "ix", "iy", "iz",
                "order", "order_raw", "type", "source", "mol_index", "notes",
            ]
        )
    bonds_df = pd.DataFrame(bonds)
    # bid and a1<a2 normalization handled by USM constructor
    return bonds_df
