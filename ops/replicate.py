from __future__ import annotations

from typing import Tuple, Optional, List, Dict
import numpy as np
import pandas as pd

from usm.core.model import USM


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
    Create a supercell by integer replication along lattice directions a,b,c (orthorhombic assumption).
    - If an orthorhombic cell is not available, raises ValueError.
    - Bonds are replicated within each image, preserving connectivity.
    - Optionally annotates atoms with image_i, image_j, image_k (int32).
    """
    if na <= 0 or nb <= 0 or nc <= 0:
        raise ValueError("na, nb, nc must be positive integers")

    M = _orthorhombic_vectors(usm.cell)
    if M is None:
        raise ValueError("replicate_supercell requires orthorhombic PBC cell with finite parameters")

    atoms = usm.atoms.reset_index(drop=True).copy()
    xyz = atoms[["x", "y", "z"]].to_numpy(dtype=float)

    # Prepare output atoms by tiling coordinates
    images: List[pd.DataFrame] = []
    bonds_images: List[pd.DataFrame] = []

    n_atoms = len(atoms)
    base_aids = atoms["aid"].to_numpy().astype(int)

    # Compute translation vectors for each image
    a_vec, b_vec, c_vec = M
    idx = 0
    for i in range(na):
        for j in range(nb):
            for k in range(nc):
                t = i * a_vec + j * b_vec + k * c_vec
                img_atoms = atoms.copy()
                img_atoms["x"] = xyz[:, 0] + t[0]
                img_atoms["y"] = xyz[:, 1] + t[1]
                img_atoms["z"] = xyz[:, 2] + t[2]
                if add_image_indices:
                    img_atoms["image_i"] = np.int32(i)
                    img_atoms["image_j"] = np.int32(j)
                    img_atoms["image_k"] = np.int32(k)
                # Temporary keep old aid for bond remap
                img_atoms["_old_aid"] = base_aids
                img_atoms["_img_idx"] = idx
                images.append(img_atoms)

                if usm.bonds is not None and len(usm.bonds) > 0:
                    bdf = usm.bonds.copy()
                    # We'll remap after concatenation using a stable mapping
                    bdf["_img_idx"] = idx
                    bonds_images.append(bdf)
                idx += 1

    all_atoms = pd.concat(images, ignore_index=True)
    # Assign new aids sequentially (drop existing 'aid' if present)
    if "aid" in all_atoms.columns:
        all_atoms = all_atoms.drop(columns=["aid"])
    all_atoms.insert(0, "aid", np.arange(len(all_atoms), dtype=np.int32))

    # Build mapping (old_aid, img_idx) -> new_aid
    key = pd.MultiIndex.from_arrays([all_atoms["_old_aid"].to_numpy(), all_atoms["_img_idx"].to_numpy()])
    new_aid_series = pd.Series(all_atoms["aid"].to_numpy(), index=key)
    # Clean temp columns
    all_atoms.drop(columns=["_old_aid", "_img_idx"], inplace=True, errors="ignore")

    # Replicate bonds if any
    new_bonds = None
    if bonds_images:
        bcat = pd.concat(bonds_images, ignore_index=True)
        # Map endpoints for each image independently
        def remap_endpoints(row):
            img_idx = int(row["_img_idx"])
            a1_old = int(row["a1"])
            a2_old = int(row["a2"])
            try:
                a1_new = int(new_aid_series.loc[(a1_old, img_idx)])
                a2_new = int(new_aid_series.loc[(a2_old, img_idx)])
            except KeyError:
                return pd.NA, pd.NA
            return a1_new, a2_new

        a1_new_list = []
        a2_new_list = []
        for _, r in bcat.iterrows():
            a1n, a2n = remap_endpoints(r)
            a1_new_list.append(a1n)
            a2_new_list.append(a2n)
        bcat["a1"] = a1_new_list
        bcat["a2"] = a2_new_list
        bcat = bcat.dropna(subset=["a1", "a2"]).copy()
        # Normalize a1 < a2
        a1v = bcat["a1"].astype("int32").to_numpy()
        a2v = bcat["a2"].astype("int32").to_numpy()
        swap = a1v > a2v
        if swap.any():
            tmp = a1v[swap].copy()
            a1v[swap] = a2v[swap]
            a2v[swap] = tmp
        bcat["a1"] = a1v
        bcat["a2"] = a2v
        # Drop helper column
        bcat.drop(columns=["_img_idx"], inplace=True, errors="ignore")
        # Deduplicate bonds across images just in case (shouldn't be needed)
        bcat = bcat.drop_duplicates(subset=["a1", "a2", "order"], keep="first").reset_index(drop=True)
        new_bonds = bcat

    # Update cell dimensions by scaling
    new_cell = dict(usm.cell)
    new_cell["a"] = float(new_cell.get("a", np.nan)) * na
    new_cell["b"] = float(new_cell.get("b", np.nan)) * nb
    new_cell["c"] = float(new_cell.get("c", np.nan)) * nc

    return USM(
        atoms=all_atoms,
        bonds=new_bonds,
        molecules=None if usm.molecules is None else usm.molecules.copy(),
        cell=new_cell,
        provenance=dict(usm.provenance or {}),
        preserved_text=dict(usm.preserved_text or {}),
    )