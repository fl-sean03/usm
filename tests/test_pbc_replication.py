import pytest
import pandas as pd
import numpy as np
from usm.core.model import USM
from usm.ops.replicate import replicate_supercell

def test_replicate_1d_chain_pbc():
    """
    Test 1D replication: 1x1x1 -> 2x1x1.
    Original cell has 2 atoms and 2 bonds:
    - Bond 0-1 (internal, ix=0)
    - Bond 0-1 (periodic, ix=1) - connects atom 0 to atom 1 in next image.
    In a 2x1x1 supercell, we expect:
    - 4 atoms total.
    - 4 bonds total.
    Image 0:
    - (0, 1, ix=0) -> (0, 1, nix=0) [Internal to Supercell]
    - (0, 1, ix=1) -> target_i = (0+1)%2 = 1. new_a2 = atom 1 in image 1 (aid 3). nix = (0+1)//2 = 0.
      Result: (0, 3, nix=0) [Internal to Supercell]
    Image 1:
    - (2, 3, ix=0) -> (2, 3, nix=0) [Internal to Supercell]
    - (2, 3, ix=1) -> target_i = (1+1)%2 = 0. new_a2 = atom 1 in image 0 (aid 1). nix = (1+1)//2 = 1.
      Result: (2, 1, nix=1) [Periodic to Supercell]
    """
    cell = dict(pbc=True, a=10.0, b=10.0, c=10.0, alpha=90.0, beta=90.0, gamma=90.0)
    atoms_df = pd.DataFrame([
        {"aid": 0, "name": "C1", "element": "C", "atom_type": "c", "charge": 0.0, "x": 0.0, "y": 5.0, "z": 5.0, "mol_label": "M1", "mol_index": 0, "mol_block_name": "B1"},
        {"aid": 1, "name": "C2", "element": "C", "atom_type": "c", "charge": 0.0, "x": 5.0, "y": 5.0, "z": 5.0, "mol_label": "M1", "mol_index": 0, "mol_block_name": "B1"},
    ])
    bonds_df = pd.DataFrame([
        {"a1": 0, "a2": 1, "ix": 0, "iy": 0, "iz": 0},
        {"a1": 0, "a2": 1, "ix": 1, "iy": 0, "iz": 0},
    ])
    usm = USM(atoms=atoms_df, bonds=bonds_df, cell=cell)
    
    # Replicate 2x1x1
    usm_super = replicate_supercell(usm, na=2, nb=1, nc=1)
    
    assert len(usm_super.atoms) == 4
    assert len(usm_super.bonds) == 4
    
    # Check bond connectivity
    # Sorted by a1, a2
    bonds = usm_super.bonds.sort_values(["a1", "a2"]).reset_index(drop=True)
    
    # Bond 0: (0, 1, 0, 0, 0)
    assert (bonds.loc[0, ["a1", "a2", "ix", "iy", "iz"]].to_numpy() == [0, 1, 0, 0, 0]).all()
    # Bond 1: (0, 3, 0, 0, 0) - This was the periodic bond from image 0
    assert (bonds.loc[1, ["a1", "a2", "ix", "iy", "iz"]].to_numpy() == [0, 3, 0, 0, 0]).all()
    # Bond 2: (1, 2, -1, 0, 0) - Normalized version of (2, 1, 1, 0, 0)
    # 2 is a1, 1 is a2. a1 > a2 -> swap -> a1=1, a2=2, ix=-1
    assert (bonds.loc[2, ["a1", "a2", "ix", "iy", "iz"]].to_numpy() == [1, 2, -1, 0, 0]).all()
    # Bond 3: (2, 3, 0, 0, 0)
    assert (bonds.loc[3, ["a1", "a2", "ix", "iy", "iz"]].to_numpy() == [2, 3, 0, 0, 0]).all()

def test_replicate_3d_lattice_counts():
    """Verify bond counts for 2x2x2 replication."""
    cell = dict(pbc=True, a=10.0, b=10.0, c=10.0, alpha=90.0, beta=90.0, gamma=90.0)
    # Simple cubic: 1 atom, 3 periodic bonds (one along each axis)
    atoms_df = pd.DataFrame([
        {"aid": 0, "name": "C1", "element": "C", "atom_type": "c", "charge": 0.0, "x": 5.0, "y": 5.0, "z": 5.0, "mol_label": "M1", "mol_index": 0, "mol_block_name": "B1"},
    ])
    bonds_df = pd.DataFrame([
        {"a1": 0, "a2": 0, "ix": 1, "iy": 0, "iz": 0},
        {"a1": 0, "a2": 0, "ix": 0, "iy": 1, "iz": 0},
        {"a1": 0, "a2": 0, "ix": 0, "iy": 0, "iz": 1},
    ])
    usm = USM(atoms=atoms_df, bonds=bonds_df, cell=cell)
    
    na, nb, nc = 2, 2, 2
    usm_super = replicate_supercell(usm, na=na, nb=nb, nc=nc)
    
    assert len(usm_super.atoms) == na * nb * nc * len(atoms_df)
    assert len(usm_super.bonds) == na * nb * nc * len(bonds_df)
    
    # Check that for each axis, we have half internal and half periodic bonds (since N=2)
    # For a=2: 
    # i=0 -> ix=1 -> ti=1, wi=1, nix=0 (Internal)
    # i=1 -> ix=1 -> ti=2, wi=0, nix=1 (Periodic)
    # Across j,k there are 4 such pairs. Total 4 internal, 4 periodic for 'a' direction.
    
    ix_vals = usm_super.bonds["ix"].to_numpy()
    iy_vals = usm_super.bonds["iy"].to_numpy()
    iz_vals = usm_super.bonds["iz"].to_numpy()
    
    # Total bonds = 8 * 3 = 24.
    # Bonds in x: 8. 4 internal (ix=0), 4 periodic (ix=1 or -1 if normalized)
    # Wait, normalization might negate flags.
    
    x_bonds = usm_super.bonds[(usm_super.bonds["ix"] != 0) | ((usm_super.bonds["iy"] == 0) & (usm_super.bonds["iz"] == 0))]
    # Actually it's easier to just check total counts of non-zero flags.
    assert np.count_nonzero(ix_vals) == 4
    assert np.count_nonzero(iy_vals) == 4
    assert np.count_nonzero(iz_vals) == 4

def test_replicate_non_periodic_bonds():
    """Ensure non-periodic bonds are still handled correctly (all nix=0)."""
    cell = dict(pbc=True, a=10.0, b=10.0, c=10.0, alpha=90.0, beta=90.0, gamma=90.0)
    atoms_df = pd.DataFrame([
        {"aid": 0, "name": "C1", "element": "C", "atom_type": "c", "charge": 0.0, "x": 1.0, "y": 1.0, "z": 1.0, "mol_label": "M1", "mol_index": 0, "mol_block_name": "B1"},
        {"aid": 1, "name": "C2", "element": "C", "atom_type": "c", "charge": 0.0, "x": 2.0, "y": 2.0, "z": 2.0, "mol_label": "M1", "mol_index": 0, "mol_block_name": "B1"},
    ])
    bonds_df = pd.DataFrame([
        {"a1": 0, "a2": 1, "ix": 0, "iy": 0, "iz": 0}
    ])
    usm = USM(atoms=atoms_df, bonds=bonds_df, cell=cell)
    
    usm_super = replicate_supercell(usm, 2, 2, 2)
    assert len(usm_super.bonds) == 8
    assert (usm_super.bonds["ix"] == 0).all()
    assert (usm_super.bonds["iy"] == 0).all()
    assert (usm_super.bonds["iz"] == 0).all()

def test_replicate_self_bond_pbc():
    """
    Test replication of a self-bond crossing boundary: 1x1x1 -> 2x1x1.
    Original: 1 atom (0), bond (0, 0, ix=1).
    Supercell 2x1x1: 2 atoms (0, 1).
    Image 0: (0, 0, ix=1) -> ti=1, wi=1, nix=0. Bond (0, 1, 0).
    Image 1: (1, 1, ix=1) -> ti=2, wi=0, nix=1. Bond (1, 0, 1) -> Normalized (0, 1, -1).
    """
    cell = dict(pbc=True, a=10.0, b=10.0, c=10.0, alpha=90.0, beta=90.0, gamma=90.0)
    atoms_df = pd.DataFrame([
        {"aid": 0, "name": "C1", "element": "C", "atom_type": "c", "charge": 0.0, "x": 0.0, "y": 0.0, "z": 0.0, "mol_label": "M1", "mol_index": 0, "mol_block_name": "B1"},
    ])
    bonds_df = pd.DataFrame([
        {"a1": 0, "a2": 0, "ix": 1, "iy": 0, "iz": 0},
    ])
    usm = USM(atoms=atoms_df, bonds=bonds_df, cell=cell)
    
    usm_super = replicate_supercell(usm, 2, 1, 1)
    
    assert len(usm_super.atoms) == 2
    assert len(usm_super.bonds) == 2
    
    # Sort bonds for comparison
    bonds = usm_super.bonds.sort_values(["a1", "a2", "ix"]).reset_index(drop=True)
    
    # Bond 0: (0, 1, ix=-1)
    assert (bonds.loc[0, ["a1", "a2", "ix", "iy", "iz"]].to_numpy() == [0, 1, -1, 0, 0]).all()
    # Bond 1: (0, 1, ix=0)
    assert (bonds.loc[1, ["a1", "a2", "ix", "iy", "iz"]].to_numpy() == [0, 1, 0, 0, 0]).all()
