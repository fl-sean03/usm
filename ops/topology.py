from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Optional, Dict, Any
import collections

from usm.core.model import USM
from usm.ops.lattice import lattice_matrix, lattice_inverse

def perceive_periodic_bonds(usm: USM, threshold_factor: float = 1.2) -> USM:
    """
    Update bond image flags (ix, iy, iz) based on the minimum image convention.
    
    This operator identifies the correct periodic image for each bond endpoint 
    by finding the image of a2 that is closest to a1 in Cartesian space.
    
    Parameters
    ----------
    usm : USM
        The structure model to update.
    threshold_factor : float, optional
        Reserved for future heuristic checks (default 1.2).
        Currently, minimum image convention is applied strictly if PBC is enabled.
        
    Returns
    -------
    USM
        A new USM instance with updated bond flags.
    """
    if usm.bonds is None or len(usm.bonds) == 0:
        return usm.copy()
    
    cell = usm.cell
    if not cell.get("pbc", False):
        # If no PBC, ensure all flags are 0
        out = usm.copy()
        for col in ["ix", "iy", "iz"]:
            out.bonds[col] = 0
        return out

    # Check for valid lattice parameters
    required_keys = ["a", "b", "c", "alpha", "beta", "gamma"]
    if not all(k in cell and np.isfinite(cell[k]) for k in required_keys):
        # Cannot perceive periodicity without valid lattice
        return usm.copy()

    # Build lattice matrix and its inverse
    try:
        A = lattice_matrix(
            cell["a"], cell["b"], cell["c"],
            cell["alpha"], cell["beta"], cell["gamma"]
        )
        A_inv = lattice_inverse(A)
    except ValueError:
        # Singular or invalid lattice
        return usm.copy()

    out = usm.copy()
    
    # Extract coordinates and bond atom IDs
    # USM ensures aid matches index 0..N-1
    coords = out.atoms[["x", "y", "z"]].to_numpy()
    a1_indices = out.bonds["a1"].to_numpy().astype(int)
    a2_indices = out.bonds["a2"].to_numpy().astype(int)
    
    r1 = coords[a1_indices]
    r2 = coords[a2_indices]
    
    # Cartesian difference: dr = r2 - r1
    dr = r2 - r1
    
    # Fractional difference: df = dr @ A_inv
    df = dr @ A_inv
    
    # Minimum image convention image flags
    # We want f2_actual = f2 + img_flags
    # s.t. f2_actual - f1 = (f2 - f1) - round(f2 - f1)
    # So img_flags = -round(df)
    img_flags = -np.round(df).astype(int)
    
    out.bonds["ix"] = img_flags[:, 0]
    out.bonds["iy"] = img_flags[:, 1]
    out.bonds["iz"] = img_flags[:, 2]
    
    return out

def validate_supercell(usm: USM) -> Dict[str, Any]:
    """
    Validate connectivity and coordination of the supercell structure.
    
    Produces a report with:
    - n_atoms: Total number of atoms
    - n_bonds: Total number of bonds
    - n_connected_components: Number of disjoint graphs (framework should be 1)
    - zn_coordination: Histogram of coordination numbers for Zn atoms {coord: count}
    
    Parameters
    ----------
    usm : USM
        The structure model to validate.
        
    Returns
    -------
    Dict[str, Any]
        Validation report.
    """
    n_atoms = len(usm.atoms)
    adj = collections.defaultdict(list)
    n_bonds = 0
    
    if usm.bonds is not None and len(usm.bonds) > 0:
        n_bonds = len(usm.bonds)
        # USM.atoms.aid matches index 0..N-1
        a1_indices = usm.bonds["a1"].to_numpy().astype(int)
        a2_indices = usm.bonds["a2"].to_numpy().astype(int)
        for u, v in zip(a1_indices, a2_indices):
            adj[u].append(v)
            adj[v].append(u)
            
    # Connected Components (BFS)
    visited = np.zeros(n_atoms, dtype=bool)
    n_components = 0
    for i in range(n_atoms):
        if not visited[i]:
            n_components += 1
            q = collections.deque([i])
            visited[i] = True
            while q:
                u = q.popleft()
                for v in adj[u]:
                    if not visited[v]:
                        visited[v] = True
                        q.append(v)
                        
    # Metal coordination (Zn)
    zn_coordination = {}
    if "element" in usm.atoms.columns:
        # Zn atoms filter
        is_zn = usm.atoms["element"] == "Zn"
        zn_indices = usm.atoms.index[is_zn].to_numpy()
        
        counts = []
        for idx in zn_indices:
            counts.append(len(adj[idx]))
            
        # Build histogram
        hist = collections.Counter(counts)
        # Sort by coordination number and convert keys to string for JSON
        zn_coordination = {str(k): int(v) for k, v in sorted(hist.items())}
        
    return {
        "n_atoms": int(n_atoms),
        "n_bonds": int(n_bonds),
        "n_connected_components": int(n_components),
        "metal_coordination": {
            "Zn": zn_coordination
        }
    }
