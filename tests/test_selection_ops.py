"""Tests for selection and site operations."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from usm.core.model import USM
from usm.ops.selection import split_threshold, pair_oh_by_distance, count_by_side
from usm.ops.sites import (
    compute_surface_split_threshold,
    pair_by_distance,
    Site,
    pick_half_per_surface,
    pick_n_per_surface,
    count_sites_by_side,
)
from usm.ops.select import select_by_element, select_by_mask


def _make_usm_slab() -> USM:
    """Create a simple slab with atoms at different z heights."""
    atoms = pd.DataFrame([
        {"aid": i, "name": f"A{i}", "element": "O" if i < 4 else "H",
         "atom_type": "O" if i < 4 else "H", "charge": 0.0,
         "x": float(i), "y": 0.0, "z": float(i * 2),
         "mol_label": "SLAB", "mol_index": 1, "mol_block_name": "SLAB"}
        for i in range(8)
    ])
    return USM(atoms=atoms, cell={"pbc": False}, provenance={}, preserved_text={})


# --- split_threshold ---

def test_split_threshold_auto() -> None:
    z = [0.0, 1.0, 2.0, 8.0, 9.0, 10.0]
    thr = split_threshold(z, method="auto")
    assert 2.0 < thr < 8.0


def test_split_threshold_median() -> None:
    z = [1.0, 2.0, 3.0, 4.0]
    thr = split_threshold(z, method="median")
    assert thr == pytest.approx(2.5)


# --- pair_oh_by_distance ---

def test_pair_oh_by_distance_basic() -> None:
    o_df = pd.DataFrame({"aid": [0], "x": [0.0], "y": [0.0], "z": [0.0]})
    h_df = pd.DataFrame({"aid": [1], "x": [1.0], "y": [0.0], "z": [0.0]})
    pairs = pair_oh_by_distance(o_df, h_df, cutoff=2.0)
    assert len(pairs) == 1
    assert pairs[0][0] == 0  # O aid
    assert pairs[0][1] == 1  # H aid


def test_pair_oh_beyond_cutoff() -> None:
    o_df = pd.DataFrame({"aid": [0], "x": [0.0], "y": [0.0], "z": [0.0]})
    h_df = pd.DataFrame({"aid": [1], "x": [10.0], "y": [0.0], "z": [0.0]})
    pairs = pair_oh_by_distance(o_df, h_df, cutoff=2.0)
    assert len(pairs) == 0


# --- count_by_side ---

def test_count_by_side() -> None:
    atoms = pd.DataFrame({
        "atom_type": ["o", "o", "h", "h"],
        "z": [1.0, 2.0, 8.0, 9.0],
    })
    result = count_by_side(atoms, ["o", "h"], thr=5.0)
    assert result["o"]["bottom"] == 2
    assert result["h"]["top"] == 2


# --- compute_surface_split_threshold ---

def test_compute_surface_split() -> None:
    z = [0.0, 1.0, 2.0, 10.0, 11.0, 12.0]
    thr = compute_surface_split_threshold(z, method="auto")
    assert 2.0 < thr < 10.0


# --- select_by_element ---

def test_select_by_element() -> None:
    usm = _make_usm_slab()
    result = select_by_element(usm, ["O"])
    assert len(result.atoms) == 4
    assert all(result.atoms["element"] == "O")


# --- select_by_mask ---

def test_select_by_mask() -> None:
    usm = _make_usm_slab()
    mask = usm.atoms["z"].astype(float) > 5.0
    result = select_by_mask(usm, mask)
    assert len(result.atoms) < len(usm.atoms)
    assert all(result.atoms["z"].astype(float) > 5.0)


# --- pick_half_per_surface ---

def test_pick_half_per_surface_deterministic() -> None:
    sites = [
        Site(site_id=i, o_aid=i, h_aid=i + 100, side="top" if i < 5 else "bottom",
             o_pos=(float(i), 0.0, float(i)), distance=1.0, removal_aids=[i + 100], meta={})
        for i in range(10)
    ]
    picked1 = pick_half_per_surface(sites, seed=42)
    picked2 = pick_half_per_surface(sites, seed=42)
    # Same seed → same result
    assert [s.site_id for s in picked1] == [s.site_id for s in picked2]
    # Should pick ~half from each side
    counts = count_sites_by_side(picked1)
    assert counts.get("top", 0) <= 5
    assert counts.get("bottom", 0) <= 5
