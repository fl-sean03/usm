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
