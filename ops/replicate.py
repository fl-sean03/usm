from __future__ import annotations

from typing import Tuple, Optional, List, Dict
import numpy as np
import pandas as pd

from usm.core.model import USM
from usm.ops.lattice import lattice_matrix, lattice_inverse, xyz_to_frac, frac_to_xyz


def _orthorhombic_vectors(cell: Dict) -> Optional[np.ndarray]:
    """
    Return lattice vectors for an orthorhombic cell as a 3x3 matrix with rows a_vec, b_vec, c_vec.
    If PBC is False or parameters are not finite or angles not ~90 deg, return None.
    """
    if not bool(cell.get("pbc", False)):
        return None

    a = cell.get("a", np.nan)
    b = cell.get("b", np.nan)
    c = cell.get("c", np.nan)
    alpha = cell.get("alpha", 90.0)
    beta = cell.get("beta", 90.0)
    gamma = cell.get("gamma", 90.0)

    if not np.all(np.isfinite([a, b, c, alpha, beta, gamma])):
        return None

    if not (abs(alpha - 90.0) < 1e-6 and abs(beta - 90.0) < 1e-6 and abs(gamma - 90.0) < 1e-6):
        return None

    return np.array([[a, 0.0, 0.0], [0.0, b, 0.0], [0.0, 0.0, c]], dtype=float)


def replicate_supercell(usm: USM, na: int, nb: int, nc: int, add_image_indices: bool = True) -> USM:
    """
    Create a supercell by integer replication along lattice directions a,b,c for general triclinic cells.
    Uses robust mapping-based materialization for periodic bonds (Policy P1).

    Semantics:
      - Requires PBC cell with finite and valid (a,b,c,alpha,beta,gamma).
      - Replication performed in fractional space: frac_img = (frac_unit + shift) / [na, nb, nc].
      - Periodic bonds crossing unit cell boundaries are 'materialized' as internal bonds
        if they land within the supercell, or updated supercell-level periodic bonds if they don't.
      - Cell lengths a,b,c are scaled by na,nb,nc; angles are preserved.
    """
    if na <= 0 or nb <= 0 or nc <= 0:
        raise ValueError("na, nb, nc must be positive integers")

    cell = usm.cell or {}
    if not bool(cell.get("pbc", False)):
        raise ValueError("replicate_supercell requires pbc=True")

    a = float(cell.get("a", np.nan))
    b = float(cell.get("b", np.nan))
    c = float(cell.get("c", np.nan))
    alpha = float(cell.get("alpha", 90.0))
    beta = float(cell.get("beta", 90.0))
    gamma = float(cell.get("gamma", 90.0))
    if not np.all(np.isfinite([a, b, c, alpha, beta, gamma])):
        raise ValueError("replicate_supercell requires finite cell parameters")

    # Build lattice and inverse
    try:
        A = lattice_matrix(a, b, c, alpha, beta, gamma)
        A_inv = lattice_inverse(A)
    except Exception as e:
        raise ValueError(f"Invalid or singular lattice: {e}")

    # Prepare unit cell atoms
    unit_atoms = usm.atoms.copy()
    num_unit_atoms = len(unit_atoms)
    xyz_unit = unit_atoms[["x", "y", "z"]].to_numpy(dtype=np.float64)
    frac_unit = xyz_to_frac(A_inv, xyz_unit)

    # 1. Atom Replication
    # We will use a 4D array to store the mapping (unit_aid_idx, tx, ty, tz) -> super_aid
    # Since USM.atoms has 'aid' as 0..N-1 (guaranteed by post_init), we can use the index directly.
    aid_map = np.zeros((num_unit_atoms, na, nb, nc), dtype=np.int32)
    
    super_atoms_list = []
    current_super_aid = 0
    
    for i in range(na):
        for j in range(nb):
            for k in range(nc):
                shift = np.array([float(i), float(j), float(k)], dtype=np.float64)
                # Compute XYZ in the supercell
                # frac_super = (frac_unit + shift) -- this is fractional relative to UNIT cell
                # To get XYZ, we multiply by UNIT lattice matrix A
                xyz_img = frac_to_xyz(A, frac_unit + shift[None, :])
                
                img_atoms = unit_atoms.copy()
                img_atoms["x"] = xyz_img[:, 0]
                img_atoms["y"] = xyz_img[:, 1]
                img_atoms["z"] = xyz_img[:, 2]

                # Ensure atom names are unique across tiles.
                # This prevents downstream tools (e.g., msi2lmp) from incorrectly resolving
                # connectivity based on non-unique atom names in replicated structures.
                if "name" in img_atoms.columns:
                    suffix = f"_T_{i}_{j}_{k}"
                    img_atoms["name"] = img_atoms["name"].astype(str) + suffix
                
                if add_image_indices:
                    img_atoms["image_i"] = np.int32(i)
                    img_atoms["image_j"] = np.int32(j)
                    img_atoms["image_k"] = np.int32(k)
                
                # Assign super_aid
                aids = np.arange(current_super_aid, current_super_aid + num_unit_atoms, dtype=np.int32)
                img_atoms["aid"] = aids
                aid_map[:, i, j, k] = aids
                
                super_atoms_list.append(img_atoms)
                current_super_aid += num_unit_atoms

    all_atoms = pd.concat(super_atoms_list, ignore_index=True)

    # 2. Bond Materialization
    new_bonds = None
    if usm.bonds is not None and len(usm.bonds) > 0:
        unit_bonds = usm.bonds.copy()
        
        # We'll build the supercell bonds by iterating over tiles and unit bonds
        # For performance, we can vectorize over unit bonds within each tile
        super_bonds_list = []
        
        u1_arr = unit_bonds["a1"].to_numpy().astype(np.int32)
        u2_arr = unit_bonds["a2"].to_numpy().astype(np.int32)
        ix_arr = unit_bonds["ix"].to_numpy().astype(np.int32)
        iy_arr = unit_bonds["iy"].to_numpy().astype(np.int32)
        iz_arr = unit_bonds["iz"].to_numpy().astype(np.int32)
        
        # Check if 'order' or other columns exist to preserve them
        other_cols = [c for c in unit_bonds.columns if c not in ["bid", "a1", "a2", "ix", "iy", "iz"]]
        other_data = unit_bonds[other_cols]

        for i in range(na):
            for j in range(nb):
                for k in range(nc):
                    # For each unit bond (u1, u2, ix, iy, iz):
                    # source_tile = (i, j, k)
                    # target_tile = (i+ix, j+iy, k+iz)
                    ti = i + ix_arr
                    tj = j + iy_arr
                    tk = k + iz_arr
                    
                    # Wrapped tile within supercell
                    wi = ti % na
                    wj = tj % nb
                    wk = tk % nc
                    
                    # Supercell-level periodic shifts
                    nix = ti // na
                    niy = tj // nb
                    niz = tk // nc
                    
                    s1 = aid_map[u1_arr, i, j, k]
                    s2 = aid_map[u2_arr, wi, wj, wk]
                    
                    df_tile = pd.DataFrame({
                        "a1": s1,
                        "a2": s2,
                        "ix": nix,
                        "iy": niy,
                        "iz": niz
                    })
                    for col in other_cols:
                        df_tile[col] = other_data[col].values
                    
                    super_bonds_list.append(df_tile)
        
        bcat = pd.concat(super_bonds_list, ignore_index=True)
        
        # Normalize a1 < a2 is handled by USM post_init, but let's do it here for deduplication safety
        a1v = bcat["a1"].to_numpy()
        a2v = bcat["a2"].to_numpy()
        ixv = bcat["ix"].to_numpy()
        iyv = bcat["iy"].to_numpy()
        izv = bcat["iz"].to_numpy()
        
        swap = a1v > a2v
        if swap.any():
            a1v[swap], a2v[swap] = a2v[swap], a1v[swap].copy()
            ixv[swap], iyv[swap], izv[swap] = -ixv[swap], -iyv[swap], -izv[swap]
        
        bcat["a1"] = a1v
        bcat["a2"] = a2v
        bcat["ix"] = ixv
        bcat["iy"] = iyv
        bcat["iz"] = izv
        
        # Deduplicate
        subset = ["a1", "a2", "ix", "iy", "iz"]
        if "order" in bcat.columns:
            subset.append("order")
        new_bonds = bcat.drop_duplicates(subset=subset).reset_index(drop=True)

    # 3. Update Cell
    new_cell = dict(usm.cell)
    new_cell["a"] = a * na
    new_cell["b"] = b * nb
    new_cell["c"] = c * nc

    return USM(
        atoms=all_atoms,
        bonds=new_bonds,
        molecules=None if usm.molecules is None else usm.molecules.copy(),
        cell=new_cell,
        provenance=dict(usm.provenance or {}),
        preserved_text=dict(usm.preserved_text or {}),
    )
