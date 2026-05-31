import pytest
import pandas as pd
import numpy as np
from usm.core.model import USM
from usm.ops.topology import perceive_periodic_bonds

def test_perceive_orthorhombic_wrap():
    """Verify bond perception in a simple cubic cell."""
    # 10x10x10 cell
    cell = dict(pbc=True, a=10.0, b=10.0, c=10.0, alpha=90.0, beta=90.0, gamma=90.0)
    
    atoms_df = pd.DataFrame([
        {"aid": 0, "name": "C1", "element": "C", "atom_type": "c", "charge": 0.0, "x": 1.0, "y": 5.0, "z": 5.0, "mol_label": "M1", "mol_index": 0, "mol_block_name": "B1"},
        {"aid": 1, "name": "C2", "element": "C", "atom_type": "c", "charge": 0.0, "x": 9.0, "y": 5.0, "z": 5.0, "mol_label": "M1", "mol_index": 0, "mol_block_name": "B1"},
    ])
    # Bond 0-1. Without image flags, distance is 8.0.
    # With ix=1, C2 is at 9+10=19. Distance is 18.0.
    # With ix=-1, C2 is at 9-10=-1. Distance is |1.0 - (-1.0)| = 2.0.
    bonds_df = pd.DataFrame([
        {"a1": 0, "a2": 1, "order": 1.0}
    ])
    usm = USM(atoms=atoms_df, bonds=bonds_df, cell=cell)
    
    usm_perceived = perceive_periodic_bonds(usm)
    
    # Check flags for bond 0-1
    # a1=0, a2=1. f1=0.1, f2=0.9. f2-f1=0.8. round(0.8)=1. ix = -1.
    assert usm_perceived.bonds.loc[0, "ix"] == -1
    assert usm_perceived.bonds.loc[0, "iy"] == 0
    assert usm_perceived.bonds.loc[0, "iz"] == 0

def test_perceive_triclinic_wrap():
    """Verify bond perception in a triclinic cell."""
    # a=10, b=10, c=10, alpha=90, beta=90, gamma=60
    # cos(60) = 0.5, sin(60) = 0.866
    # a_vec = [10, 0, 0]
    # b_vec = [10*0.5, 10*0.866, 0] = [5, 8.66, 0]
    cell = dict(pbc=True, a=10.0, b=10.0, c=10.0, alpha=90.0, beta=90.0, gamma=60.0)
    
    # Atom 1 at (1, 0, 0) -> fractional [0.1, 0, 0]
    # Atom 2 at (9.5, 0, 0) -> fractional [0.95, 0, 0]
    # f2-f1 = 0.85. round = 1. ix = -1.
    atoms_df = pd.DataFrame([
        {"aid": 0, "name": "C1", "element": "C", "atom_type": "c", "charge": 0.0, "x": 1.0, "y": 0.0, "z": 0.0, "mol_label": "M1", "mol_index": 0, "mol_block_name": "B1"},
        {"aid": 1, "name": "C2", "element": "C", "atom_type": "c", "charge": 0.0, "x": 9.5, "y": 0.0, "z": 0.0, "mol_label": "M1", "mol_index": 0, "mol_block_name": "B1"},
    ])
    bonds_df = pd.DataFrame([{"a1": 0, "a2": 1}])
    usm = USM(atoms=atoms_df, bonds=bonds_df, cell=cell)
    
    usm_perceived = perceive_periodic_bonds(usm)
    assert usm_perceived.bonds.loc[0, "ix"] == -1

def test_stability_no_changes():
    """Verify that existing correct flags are preserved."""
    cell = dict(pbc=True, a=10.0, b=10.0, c=10.0, alpha=90.0, beta=90.0, gamma=90.0)
    atoms_df = pd.DataFrame([
        {"aid": 0, "name": "C1", "x": 1.0, "y": 0.0, "z": 0.0, "mol_label": "M1", "mol_index": 0, "mol_block_name": "B1", "element":"C", "atom_type":"c", "charge":0.0},
        {"aid": 1, "name": "C2", "x": 9.0, "y": 0.0, "z": 0.0, "mol_label": "M1", "mol_index": 0, "mol_block_name": "B1", "element":"C", "atom_type":"c", "charge":0.0},
    ])
    # Manually set correct flag
    bonds_df = pd.DataFrame([{"a1": 0, "a2": 1, "ix": -1, "iy": 0, "iz": 0}])
    usm = USM(atoms=atoms_df, bonds=bonds_df, cell=cell)
    
    usm_perceived = perceive_periodic_bonds(usm)
    assert usm_perceived.bonds.loc[0, "ix"] == -1

def test_no_pbc_case():
    """Verify that if PBC is false, flags are set to 0."""
    cell = dict(pbc=False, a=10.0, b=10.0, c=10.0, alpha=90.0, beta=90.0, gamma=90.0)
    atoms_df = pd.DataFrame([
        {"aid": 0, "name": "C1", "x": 1.0, "y": 0.0, "z": 0.0, "mol_label": "M1", "mol_index": 0, "mol_block_name": "B1", "element":"C", "atom_type":"c", "charge":0.0},
        {"aid": 1, "name": "C2", "x": 9.0, "y": 0.0, "z": 0.0, "mol_label": "M1", "mol_index": 0, "mol_block_name": "B1", "element":"C", "atom_type":"c", "charge":0.0},
    ])
    # Manually set incorrect flag
    bonds_df = pd.DataFrame([{"a1": 0, "a2": 1, "ix": 5}])
    usm = USM(atoms=atoms_df, bonds=bonds_df, cell=cell)

    usm_perceived = perceive_periodic_bonds(usm)
    assert usm_perceived.bonds.loc[0, "ix"] == 0


# -----------------------------------------------------------------------------
# PBC suffix parsing — load_mdf preserves the Materials Studio %abc#i suffix
# on bond rows via the (ix, iy, iz) columns. Regression coverage for the
# downstream "diagonal-cylinder" artifact discussed in
# docs/MDF_PARSER_AUDIT.md.
# -----------------------------------------------------------------------------

import tempfile
from pathlib import Path
from usm.io.mdf import load_mdf, save_mdf


_MINIMAL_PBC_MDF = """!BIOSYM molecular_data 4

!Date: Mon Jan 01 00:00:00 2026  USM test fixture

#topology

@column 1 element
@column 2 atom_type
@column 3 charge_group
@column 4 isotope
@column 5 formal_charge
@column 6 charge
@column 7 switching_atom
@column 8 oop_flag
@column 9 chirality_flag
@column 10 occupancy
@column 11 xray_temp_factor
@column 12 connections

@molecule TEST_MOL

XXXX_1:A             C  c       ?     0  0     0.0000 0 0 8 1.0000  0.0000 B B%100#1 C%0-10#1
XXXX_1:B             C  c       ?     0  0     0.0000 0 0 8 1.0000  0.0000 A A%-100#1
XXXX_1:C             C  c       ?     0  0     0.0000 0 0 8 1.0000  0.0000 A%010#1

#end
"""


def _write_tmp_mdf(text: str) -> str:
    """Write ``text`` to a tempfile and return its path."""
    f = tempfile.NamedTemporaryFile(mode="w", suffix=".mdf", delete=False, encoding="utf-8")
    f.write(text)
    f.close()
    return f.name


def test_load_mdf_recovers_pbc_image_offsets():
    """The Materials Studio %abc#i suffix becomes (ix, iy, iz) on the Bond row."""
    path = _write_tmp_mdf(_MINIMAL_PBC_MDF)
    try:
        usm = load_mdf(path)
    finally:
        Path(path).unlink(missing_ok=True)

    # Three atoms, three distinct bonds: A-B same-cell, A-B (image), A-C (image)
    aid_by_name = {row["name"]: int(row["aid"]) for _, row in usm.atoms.iterrows()}
    a_aid, b_aid, c_aid = aid_by_name["A"], aid_by_name["B"], aid_by_name["C"]

    # Build a lookup keyed by canonical (a1, a2, ix, iy, iz) so the test is
    # robust to row ordering.
    triples = {
        (int(r["a1"]), int(r["a2"]), int(r["ix"]), int(r["iy"]), int(r["iz"]))
        for _, r in usm.bonds.iterrows()
    }

    # A-B same cell. Token "B" on A (no suffix) → (0, 0, 0). Reciprocal "A" on B
    # is the same triple after canonicalization → deduped.
    ab1 = (min(a_aid, b_aid), max(a_aid, b_aid), 0, 0, 0)
    assert ab1 in triples, f"A-B same-cell bond missing; got {triples}"

    # A-B periodic-image bond. Token "B%100#1" on A: shift (+1, 0, 0) of B
    # relative to A. Reciprocal "A%-100#1" on B canonicalizes to the same row.
    # Canonical a1 is min(a_aid, b_aid); the shift sign flips on swap.
    if a_aid < b_aid:
        ab_img = (a_aid, b_aid, 1, 0, 0)
    else:
        ab_img = (b_aid, a_aid, -1, 0, 0)
    assert ab_img in triples, f"A-B PBC bond missing; got {triples}"

    # A-C periodic-image bond. Token "C%0-10#1" on A: shift (0, -1, 0).
    # Reciprocal "A%010#1" on C: shift (0, +1, 0) of A relative to C.
    # After canonicalization they collapse to one row.
    if a_aid < c_aid:
        ac_img = (a_aid, c_aid, 0, -1, 0)
    else:
        ac_img = (c_aid, a_aid, 0, 1, 0)
    assert ac_img in triples, f"A-C PBC bond missing; got {triples}"

    # The three distinct bonds and no more.
    assert len(triples) == 3, f"expected 3 unique bonds, got {len(triples)}: {triples}"


def test_load_mdf_distinct_image_shifts_are_distinct_bonds():
    """Two tokens between the same pair into different images survive as two rows."""
    text = """!BIOSYM molecular_data 4

#topology

@column 1 element
@column 12 connections

@molecule TEST_MOL

XXXX_1:A             C  c       ?     0  0     0.0000 0 0 8 1.0000  0.0000 B%100#1 B%-100#1
XXXX_1:B             C  c       ?     0  0     0.0000 0 0 8 1.0000  0.0000 A%-100#1 A%100#1

#end
"""
    path = _write_tmp_mdf(text)
    try:
        usm = load_mdf(path)
    finally:
        Path(path).unlink(missing_ok=True)

    triples = {
        (int(r["a1"]), int(r["a2"]), int(r["ix"]), int(r["iy"]), int(r["iz"]))
        for _, r in usm.bonds.iterrows()
    }
    # Two distinct PBC images of the A-B pair survive as distinct rows.
    aid_by_name = {row["name"]: int(row["aid"]) for _, row in usm.atoms.iterrows()}
    a, b = sorted([aid_by_name["A"], aid_by_name["B"]])
    assert (a, b, 1, 0, 0) in triples
    assert (a, b, -1, 0, 0) in triples
    assert len(triples) == 2, f"expected 2 distinct PBC bonds, got: {triples}"


def test_self_image_bond_preserved():
    """A self-image bond (A — A%100#1) is a legitimate PBC bond."""
    text = """!BIOSYM molecular_data 4

#topology

@column 1 element
@column 12 connections

@molecule TEST_MOL

XXXX_1:A             C  c       ?     0  0     0.0000 0 0 8 1.0000  0.0000 A%100#1

#end
"""
    path = _write_tmp_mdf(text)
    try:
        usm = load_mdf(path)
    finally:
        Path(path).unlink(missing_ok=True)

    assert usm.bonds is not None and len(usm.bonds) == 1
    row = usm.bonds.iloc[0]
    # Self-bond: a1 == a2 == 0
    assert int(row["a1"]) == int(row["a2"]) == 0
    # USM lex-positive normalization keeps (1, 0, 0) as-is.
    assert (int(row["ix"]), int(row["iy"]), int(row["iz"])) == (1, 0, 0)


def test_same_cell_self_bond_filtered():
    """A bare self-token (A — A, no suffix) is malformed and gets dropped."""
    text = """!BIOSYM molecular_data 4

#topology

@column 1 element
@column 12 connections

@molecule TEST_MOL

XXXX_1:A             C  c       ?     0  0     0.0000 0 0 8 1.0000  0.0000 A

#end
"""
    path = _write_tmp_mdf(text)
    try:
        usm = load_mdf(path)
    finally:
        Path(path).unlink(missing_ok=True)

    assert usm.bonds is None or len(usm.bonds) == 0


def test_preserve_mode_round_trip_keeps_pbc_in_text():
    """load → save(preserve_headers=True) → load preserves the PBC bonds."""
    path = _write_tmp_mdf(_MINIMAL_PBC_MDF)
    try:
        usm1 = load_mdf(path)
        out_path = path + ".out.mdf"
        save_mdf(usm1, out_path, preserve_headers=True, write_normalized_connections=False)
        usm2 = load_mdf(out_path)
        Path(out_path).unlink(missing_ok=True)
    finally:
        Path(path).unlink(missing_ok=True)

    triples1 = sorted(
        (int(r["a1"]), int(r["a2"]), int(r["ix"]), int(r["iy"]), int(r["iz"]))
        for _, r in usm1.bonds.iterrows()
    )
    triples2 = sorted(
        (int(r["a1"]), int(r["a2"]), int(r["ix"]), int(r["iy"]), int(r["iz"]))
        for _, r in usm2.bonds.iterrows()
    )
    assert triples1 == triples2


def test_normalized_mode_round_trip_keeps_pbc():
    """load → save(write_normalized_connections=True) → load preserves the PBC bonds.

    This is the stronger guarantee: the writer must emit %abc#1 suffixes on
    the synthesized connection tokens (not just copy connections_raw verbatim).
    """
    path = _write_tmp_mdf(_MINIMAL_PBC_MDF)
    try:
        usm1 = load_mdf(path)
        # Strip connections_raw so the writer is forced to synthesize the
        # tokens from the bonds table (otherwise preserve-mode would copy
        # connections_raw and mask any bug in the synthesizer).
        usm1.atoms = usm1.atoms.copy()
        usm1.atoms["connections_raw"] = pd.NA
        out_path = path + ".norm.mdf"
        save_mdf(usm1, out_path, preserve_headers=True, write_normalized_connections=True)
        usm2 = load_mdf(out_path)
        Path(out_path).unlink(missing_ok=True)
    finally:
        Path(path).unlink(missing_ok=True)

    triples1 = sorted(
        (int(r["a1"]), int(r["a2"]), int(r["ix"]), int(r["iy"]), int(r["iz"]))
        for _, r in usm1.bonds.iterrows()
    )
    triples2 = sorted(
        (int(r["a1"]), int(r["a2"]), int(r["ix"]), int(r["iy"]), int(r["iz"]))
        for _, r in usm2.bonds.iterrows()
    )
    assert triples1 == triples2, f"PBC bonds lost in normalized round-trip:\n  before: {triples1}\n  after:  {triples2}"
