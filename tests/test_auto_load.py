"""Tests for unified usm.load() auto-detection."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from usm.core.model import USM
from usm.io.loader import load
from usm.io.pdb import save_pdb


def _make_usm() -> USM:
    atoms = pd.DataFrame([{
        "aid": 0, "name": "C1", "element": "C", "atom_type": "C", "charge": 0.0,
        "x": 1.0, "y": 2.0, "z": 3.0, "mol_label": "XXXX", "mol_index": 1, "mol_block_name": "RES",
    }])
    return USM(atoms=atoms, cell={"pbc": False}, provenance={}, preserved_text={})


def test_load_pdb(tmp_path: Path) -> None:
    usm1 = _make_usm()
    pdb_path = tmp_path / "test.pdb"
    save_pdb(usm1, str(pdb_path))
    usm2 = load(str(pdb_path))
    assert len(usm2.atoms) == 1
    assert float(usm2.atoms.iloc[0]["x"]) == pytest.approx(1.0, abs=0.01)


def test_load_unknown_extension(tmp_path: Path) -> None:
    f = tmp_path / "test.xyz"
    f.write_text("dummy")
    with pytest.raises(ValueError, match="Unsupported"):
        load(str(f))


def test_load_missing_file() -> None:
    with pytest.raises(FileNotFoundError):
        load("/nonexistent/file.pdb")


def test_load_dispatches_by_extension(tmp_path: Path) -> None:
    """Verify that usm.load() dispatches correctly based on extension."""
    # Use a PDB file we know works
    usm1 = _make_usm()
    pdb_path = tmp_path / "structure.pdb"
    save_pdb(usm1, str(pdb_path))

    # load() should dispatch to load_pdb based on .pdb extension
    result = load(str(pdb_path))
    assert isinstance(result, USM)
    assert len(result.atoms) == 1
