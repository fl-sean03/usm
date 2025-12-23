import pytest
import pandas as pd
import numpy as np
from usm.core.model import USM
from usm.ops.renumber import renumber_atoms

def test_bond_flag_initialization():
    """Check that missing ix, iy, iz flags default to 0 in USM.__post_init__."""
    atoms_df = pd.DataFrame([
        {"aid": 0, "name": "C1", "element": "C", "atom_type": "c3", "charge": 0.0, "x": 0.0, "y": 0.0, "z": 0.0, "mol_label": "M1", "mol_index": 0, "mol_block_name": "B1"},
        {"aid": 1, "name": "C2", "element": "C", "atom_type": "c3", "charge": 0.0, "x": 1.0, "y": 1.0, "z": 1.0, "mol_label": "M1", "mol_index": 0, "mol_block_name": "B1"},
    ])
    # Bonds missing ix, iy, iz
    bonds_df = pd.DataFrame([
        {"a1": 0, "a2": 1, "order": 1.0}
    ])
    usm = USM(atoms=atoms_df, bonds=bonds_df)
    
    assert "ix" in usm.bonds.columns
    assert "iy" in usm.bonds.columns
    assert "iz" in usm.bonds.columns
    assert usm.bonds.loc[0, "ix"] == 0
    assert usm.bonds.loc[0, "iy"] == 0
    assert usm.bonds.loc[0, "iz"] == 0

def test_bond_normalization_with_flags():
    """Check that flags are negated when a1 > a2 triggers a swap."""
    atoms_df = pd.DataFrame([
        {"aid": 0, "name": "C1", "element": "C", "atom_type": "c3", "charge": 0.0, "x": 0.0, "y": 0.0, "z": 0.0, "mol_label": "M1", "mol_index": 0, "mol_block_name": "B1"},
        {"aid": 1, "name": "C2", "element": "C", "atom_type": "c3", "charge": 0.0, "x": 1.0, "y": 1.0, "z": 1.0, "mol_label": "M1", "mol_index": 0, "mol_block_name": "B1"},
    ])
    # Bond with a1 > a2 and non-zero flags
    bonds_df = pd.DataFrame([
        {"a1": 1, "a2": 0, "ix": 1, "iy": -1, "iz": 0, "order": 1.0}
    ])
    usm = USM(atoms=atoms_df, bonds=bonds_df)
    
    # After normalization: a1 should be 0, a2 should be 1
    assert usm.bonds.loc[0, "a1"] == 0
    assert usm.bonds.loc[0, "a2"] == 1
    # Flags should be negated: ix=-1, iy=1, iz=0
    assert usm.bonds.loc[0, "ix"] == -1
    assert usm.bonds.loc[0, "iy"] == 1
    assert usm.bonds.loc[0, "iz"] == 0

def test_renumbering_preserves_and_normalizes_flags():
    """Check that flags are correctly handled during atom renumbering."""
    # Atoms in reverse order
    atoms_df = pd.DataFrame([
        {"aid": 1, "name": "C2", "element": "C", "atom_type": "c3", "charge": 0.0, "x": 1.0, "y": 1.0, "z": 1.0, "mol_label": "M1", "mol_index": 0, "mol_block_name": "B1"},
        {"aid": 0, "name": "C1", "element": "C", "atom_type": "c3", "charge": 0.0, "x": 0.0, "y": 0.0, "z": 0.0, "mol_label": "M1", "mol_index": 0, "mol_block_name": "B1"},
    ])
    # Bond 0-1 with image flag
    bonds_df = pd.DataFrame([
        {"a1": 0, "a2": 1, "ix": 1, "iy": 0, "iz": 0, "order": 1.0}
    ])
    usm = USM(atoms=atoms_df, bonds=bonds_df)
    
    # Initial USM will normalize 0-1 (a1 < a2, so no swap)
    # Note: USM.__post_init__ actually reassigns 'aid' to 0, 1 based on row order.
    # So atoms_df[0] (originally aid 1) becomes aid 0.
    # atoms_df[1] (originally aid 0) becomes aid 1.
    # Original bond (a1=0, a2=1) now points to (new_aid 1, new_aid 0).
    # Wait, USM.__post_init__:
    # 111 |         self.atoms["aid"] = np.arange(len(self.atoms), dtype=np.int32)
    # It overwrites whatever aid was there.
    
    # Let's verify what happens in __post_init__ first.
    # In my setup above:
    # Row 0: name C2, original aid 1 -> new aid 0
    # Row 1: name C1, original aid 0 -> new aid 1
    # Original bond: a1=0, a2=1. 
    # Since aid 0 is now at index 1 and aid 1 is at index 0, 
    # the bond's a1, a2 values (which are meant to be IDs) are tricky.
    
    # Actually, USM assumes input 'aid' are valid or it assigns them.
    # Let's create a more predictable case.
    
    atoms_df = pd.DataFrame([
        {"aid": 0, "name": "A", "element": "H", "atom_type": "h", "charge": 0.0, "x": 0.0, "y": 0.0, "z": 0.0, "mol_label": "M1", "mol_index": 0, "mol_block_name": "B1"},
        {"aid": 1, "name": "B", "element": "H", "atom_type": "h", "charge": 0.0, "x": 1.0, "y": 0.0, "z": 0.0, "mol_label": "M1", "mol_index": 0, "mol_block_name": "B1"},
    ])
    bonds_df = pd.DataFrame([
        {"a1": 0, "a2": 1, "ix": 1, "iy": 0, "iz": 0}
    ])
    usm = USM(atoms=atoms_df, bonds=bonds_df)
    
    # Renumber such that A and B are swapped
    # renumber_atoms sorts by mol_index, then name by default.
    # Here A and B are already in order. Let's use name to swap them.
    usm.atoms.loc[0, "name"] = "Z" # A becomes Z
    # Now B (aid 1) < Z (aid 0)
    
    usm_new = renumber_atoms(usm)
    # After renumbering:
    # Atom B (originally aid 1) gets new aid 0
    # Atom Z (originally aid 0) gets new aid 1
    # Original bond (a1=0, a2=1) -> (new_a1=1, new_a2=0)
    # Normalization should swap them to (a1=0, a2=1) and negate flags.
    
    assert usm_new.bonds.loc[0, "a1"] == 0
    assert usm_new.bonds.loc[0, "a2"] == 1
    assert usm_new.bonds.loc[0, "ix"] == -1
    assert usm_new.bonds.loc[0, "iy"] == 0
    assert usm_new.bonds.loc[0, "iz"] == 0
