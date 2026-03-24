"""Tests for PDB import/export roundtrip."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from usm.core.model import USM
from usm.io.pdb import load_pdb, save_pdb


def _make_simple_usm() -> USM:
    """Create a minimal USM with 3 atoms and 2 bonds."""
    atoms = pd.DataFrame([
        {"aid": 0, "name": "C1", "element": "C", "atom_type": "C", "charge": 0.0,
         "x": 1.0, "y": 2.0, "z": 3.0, "mol_label": "XXXX", "mol_index": 1, "mol_block_name": "RES"},
        {"aid": 1, "name": "O1", "element": "O", "atom_type": "O", "charge": -0.5,
         "x": 4.0, "y": 5.0, "z": 6.0, "mol_label": "XXXX", "mol_index": 1, "mol_block_name": "RES"},
        {"aid": 2, "name": "H1", "element": "H", "atom_type": "H", "charge": 0.25,
         "x": 7.0, "y": 8.0, "z": 9.0, "mol_label": "XXXX", "mol_index": 1, "mol_block_name": "RES"},
    ])
    bonds = pd.DataFrame([
        {"bid": 0, "a1": 0, "a2": 1, "ix": 0, "iy": 0, "iz": 0,
         "order": 1.0, "type": "single", "source": "test", "order_raw": None, "mol_index": None, "notes": None},
        {"bid": 1, "a1": 1, "a2": 2, "ix": 0, "iy": 0, "iz": 0,
         "order": 1.0, "type": "single", "source": "test", "order_raw": None, "mol_index": None, "notes": None},
    ])
    cell = {"pbc": True, "a": 10.0, "b": 10.0, "c": 10.0,
            "alpha": 90.0, "beta": 90.0, "gamma": 90.0, "spacegroup": "P 1"}
    return USM(atoms=atoms, bonds=bonds, cell=cell, provenance={}, preserved_text={})


def test_pdb_roundtrip_coordinates(tmp_path: Path) -> None:
    """save_pdb → load_pdb preserves coordinates to 3 decimal places."""
    usm1 = _make_simple_usm()
    pdb_path = tmp_path / "test.pdb"
    save_pdb(usm1, str(pdb_path), include_conect=True)
    usm2 = load_pdb(str(pdb_path))

    assert len(usm2.atoms) == 3
    for col in ("x", "y", "z"):
        np.testing.assert_allclose(
            usm2.atoms[col].astype(float).values,
            usm1.atoms[col].astype(float).values,
            atol=0.001,  # PDB format has 3 decimal places
        )


def test_pdb_roundtrip_cryst1(tmp_path: Path) -> None:
    """CRYST1 cell parameters survive roundtrip."""
    usm1 = _make_simple_usm()
    pdb_path = tmp_path / "test.pdb"
    save_pdb(usm1, str(pdb_path))
    usm2 = load_pdb(str(pdb_path))

    assert usm2.cell["pbc"] is True
    assert usm2.cell["a"] == pytest.approx(10.0, abs=0.01)
    assert usm2.cell["alpha"] == pytest.approx(90.0, abs=0.01)


def test_pdb_roundtrip_conect(tmp_path: Path) -> None:
    """CONECT bonds survive roundtrip."""
    usm1 = _make_simple_usm()
    pdb_path = tmp_path / "test.pdb"
    save_pdb(usm1, str(pdb_path), include_conect=True)
    usm2 = load_pdb(str(pdb_path))

    assert usm2.bonds is not None
    assert len(usm2.bonds) == 2


def test_pdb_no_conect(tmp_path: Path) -> None:
    """When no CONECT records, bonds should be None."""
    usm1 = _make_simple_usm()
    pdb_path = tmp_path / "test.pdb"
    save_pdb(usm1, str(pdb_path), include_conect=False)
    usm2 = load_pdb(str(pdb_path))

    assert usm2.bonds is None


def test_pdb_element_parsing(tmp_path: Path) -> None:
    """Element is correctly parsed from PDB columns 77-78."""
    usm1 = _make_simple_usm()
    pdb_path = tmp_path / "test.pdb"
    save_pdb(usm1, str(pdb_path))
    usm2 = load_pdb(str(pdb_path))

    elements = usm2.atoms["element"].tolist()
    assert "C" in elements
    assert "O" in elements
    assert "H" in elements


def test_pdb_no_atoms_raises() -> None:
    """Empty PDB file raises ValueError."""
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".pdb", mode="w", delete=False) as f:
        f.write("END\n")
        f.flush()
        with pytest.raises(ValueError, match="no ATOM"):
            load_pdb(f.name)
