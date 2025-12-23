"""Tests for atom name uniqueness in merge_structures().

This module tests the _ensure_unique_names() function and its integration
into merge_structures() to prevent atom name collisions when merging
structures from different sources.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from usm.core.model import USM
from usm.ops.merge import (
    merge_structures,
    merge_preserving_first,
    _extract_name_parts,
    _update_connections_raw,
    _ensure_unique_names,
)


class TestExtractNameParts:
    """Tests for _extract_name_parts() helper function."""
    
    def test_simple_element_number(self):
        """Test extraction of simple element + number names."""
        assert _extract_name_parts("C1") == ("C", 1)
        assert _extract_name_parts("O2") == ("O", 2)
        assert _extract_name_parts("N3") == ("N", 3)
        assert _extract_name_parts("H10") == ("H", 10)
    
    def test_multi_letter_element(self):
        """Test extraction of multi-letter element names."""
        assert _extract_name_parts("Zn1") == ("Zn", 1)
        assert _extract_name_parts("Ca2") == ("Ca", 2)
        assert _extract_name_parts("Fe3") == ("Fe", 3)
    
    def test_non_numeric_suffix(self):
        """Test names with non-numeric suffix (returns original name, 0)."""
        assert _extract_name_parts("H2A") == ("H2A", 0)
        assert _extract_name_parts("C1a") == ("C1a", 0)
        assert _extract_name_parts("HAB") == ("HAB", 0)
    
    def test_no_suffix(self):
        """Test names without numeric suffix."""
        assert _extract_name_parts("C") == ("C", 0)
        assert _extract_name_parts("Zn") == ("Zn", 0)


class TestUpdateConnectionsRaw:
    """Tests for _update_connections_raw() helper function."""
    
    def test_simple_rename(self):
        """Test simple atom renaming in connections_raw."""
        df = pd.DataFrame({
            "name": ["C4", "O3", "O4"],
            "connections_raw": ["O3 O4", "C4", "C4"],
        })
        # No mapping - should return unchanged
        result = _update_connections_raw(df, {})
        assert result["connections_raw"].tolist() == ["O3 O4", "C4", "C4"]
        
        # With mapping
        mapping = {"O1": "O3", "O2": "O4", "C1": "C4"}
        df2 = pd.DataFrame({
            "name": ["C1", "O1", "O2"],
            "connections_raw": ["O1 O2", "C1", "C1"],
        })
        result2 = _update_connections_raw(df2, mapping)
        assert result2["connections_raw"].tolist() == ["O3 O4", "C4", "C4"]
    
    def test_with_bond_order(self):
        """Test renaming with bond order suffix."""
        df = pd.DataFrame({
            "name": ["C1"],
            "connections_raw": ["O1/2.0 O2/2.0"],
        })
        mapping = {"O1": "O3", "O2": "O4"}
        result = _update_connections_raw(df, mapping)
        assert result["connections_raw"].tolist() == ["O3/2.0 O4/2.0"]
    
    def test_missing_column(self):
        """Test behavior when connections_raw column is missing."""
        df = pd.DataFrame({"name": ["C1", "O1"]})
        result = _update_connections_raw(df, {"C1": "C2"})
        assert "connections_raw" not in result.columns
    
    def test_empty_values(self):
        """Test handling of empty/NaN connection values."""
        df = pd.DataFrame({
            "name": ["C1", "O1"],
            "connections_raw": ["", np.nan],
        })
        mapping = {"C1": "C2"}
        result = _update_connections_raw(df, mapping)
        assert result["connections_raw"].tolist()[0] == ""


class TestEnsureUniqueNames:
    """Tests for _ensure_unique_names() function."""
    
    def test_no_collisions(self):
        """Test that non-colliding names are preserved."""
        df1 = pd.DataFrame({"name": ["C1", "O1", "O2"]})
        df2 = pd.DataFrame({"name": ["C3", "O3", "O4"]})
        result = _ensure_unique_names([df1, df2])
        
        names1 = result[0]["name"].tolist()
        names2 = result[1]["name"].tolist()
        
        assert names1 == ["C1", "O1", "O2"]
        assert names2 == ["C3", "O3", "O4"]
    
    def test_collision_renamed(self):
        """Test that colliding names in second structure are renamed."""
        df1 = pd.DataFrame({"name": ["C1", "O1", "O2"]})
        df2 = pd.DataFrame({"name": ["C1", "O1", "O2"]})  # Same names!
        result = _ensure_unique_names([df1, df2])
        
        names1 = result[0]["name"].tolist()
        names2 = result[1]["name"].tolist()
        
        # First structure preserved
        assert names1 == ["C1", "O1", "O2"]
        
        # Second structure renamed - all names should be unique
        all_names = names1 + names2
        assert len(all_names) == len(set(all_names)), f"Duplicate names: {all_names}"
        
        # Second structure should have renamed atoms
        assert "C1" not in names2  # C1 was taken
        assert "O1" not in names2  # O1 was taken
        assert "O2" not in names2  # O2 was taken
    
    def test_partial_collision(self):
        """Test with only some names colliding."""
        df1 = pd.DataFrame({"name": ["Zn1", "N1", "O1"]})
        df2 = pd.DataFrame({"name": ["C1", "O1", "O2"]})  # O1 collides
        result = _ensure_unique_names([df1, df2])
        
        names1 = result[0]["name"].tolist()
        names2 = result[1]["name"].tolist()
        
        # First structure preserved
        assert names1 == ["Zn1", "N1", "O1"]
        
        # C1 and O2 should be preserved (no collision)
        assert "C1" in names2
        assert "O2" in names2
        
        # O1 should be renamed
        assert "O1" not in names2
        
        # All names unique
        all_names = names1 + names2
        assert len(all_names) == len(set(all_names))
    
    def test_connections_raw_updated(self):
        """Test that connections_raw is updated with new names."""
        df1 = pd.DataFrame({
            "name": ["C1", "O1", "O2"],
            "connections_raw": ["O1 O2", "C1", "C1"],
        })
        df2 = pd.DataFrame({
            "name": ["C1", "O1", "O2"],
            "connections_raw": ["O1 O2", "C1", "C1"],
        })
        result = _ensure_unique_names([df1, df2])
        
        # Get the new names from second structure
        names2 = result[1]["name"].tolist()
        conn2 = result[1]["connections_raw"].tolist()
        
        # Connections should reference the new names
        # The carbon's connections should contain its new oxygen names
        c_idx = 0  # First atom is carbon
        c_conns = conn2[c_idx].split()
        for conn_name in c_conns:
            assert conn_name in names2, f"{conn_name} not in {names2}"
    
    def test_three_structures(self):
        """Test merging three structures with cascading collisions."""
        df1 = pd.DataFrame({"name": ["C1", "O1"]})
        df2 = pd.DataFrame({"name": ["C1", "O1"]})
        df3 = pd.DataFrame({"name": ["C1", "O1"]})
        result = _ensure_unique_names([df1, df2, df3])
        
        all_names = []
        for df in result:
            all_names.extend(df["name"].tolist())
        
        # All 6 names should be unique
        assert len(all_names) == 6
        assert len(set(all_names)) == 6, f"Duplicates in: {all_names}"


class TestMergeStructuresUniqueNames:
    """Integration tests for merge_structures() with unique names."""
    
    def _make_usm(self, atoms_data: dict, bonds_data: dict = None) -> USM:
        """Helper to create a USM instance."""
        atoms = pd.DataFrame(atoms_data)
        bonds = pd.DataFrame(bonds_data) if bonds_data else None
        return USM(
            atoms=atoms,
            bonds=bonds,
            molecules=None,
            cell={"pbc": False},
            provenance={},
            preserved_text={},
        )
    
    def test_merge_with_name_collision(self):
        """Test that merge_structures produces unique names."""
        # MOF-like structure
        usm1 = self._make_usm({
            "aid": [0, 1, 2, 3],
            "name": ["Zn1", "C1", "O1", "O2"],
            "element": ["Zn", "C", "O", "O"],
            "x": [0.0, 1.0, 2.0, 3.0],
            "y": [0.0, 0.0, 0.0, 0.0],
            "z": [0.0, 0.0, 0.0, 0.0],
        })
        
        # CO2-like structure with colliding names
        usm2 = self._make_usm({
            "aid": [0, 1, 2],
            "name": ["C1", "O1", "O2"],  # Collides with usm1!
            "element": ["C", "O", "O"],
            "x": [5.0, 6.0, 7.0],
            "y": [0.0, 0.0, 0.0],
            "z": [0.0, 0.0, 0.0],
        })
        
        merged = merge_structures([usm1, usm2], cell_policy="first")
        
        # Should have 7 atoms total
        assert len(merged.atoms) == 7
        
        # All names should be unique
        names = merged.atoms["name"].tolist()
        assert len(names) == len(set(names)), f"Duplicate names in merged: {names}"
        
        # First structure's names should be preserved
        assert "Zn1" in names
        assert "C1" in names  # First C1
        
        # Second structure's colliding names should be renamed
        # Count how many of each base element we have
        c_count = sum(1 for n in names if n.startswith("C"))
        o_count = sum(1 for n in names if n.startswith("O"))
        assert c_count == 2  # C1 from MOF, C2 from CO2
        assert o_count == 4  # O1, O2 from MOF; O3, O4 from CO2
    
    def test_merge_preserves_bonds(self):
        """Test that bonds are preserved correctly after name renaming."""
        # Structure 1 with bonds
        usm1 = self._make_usm(
            {
                "aid": [0, 1, 2],
                "name": ["C1", "O1", "O2"],
                "element": ["C", "O", "O"],
                "x": [0.0, 1.0, 2.0],
                "y": [0.0, 0.0, 0.0],
                "z": [0.0, 0.0, 0.0],
            },
            {
                "a1": [0, 0],
                "a2": [1, 2],
                "order": [2.0, 2.0],
            }
        )
        
        # Structure 2 with bonds (same names!)
        usm2 = self._make_usm(
            {
                "aid": [0, 1, 2],
                "name": ["C1", "O1", "O2"],
                "element": ["C", "O", "O"],
                "x": [5.0, 6.0, 7.0],
                "y": [0.0, 0.0, 0.0],
                "z": [0.0, 0.0, 0.0],
            },
            {
                "a1": [0, 0],
                "a2": [1, 2],
                "order": [2.0, 2.0],
            }
        )
        
        merged = merge_structures([usm1, usm2], cell_policy="first")
        
        # Should have 4 bonds total (2 from each structure)
        assert merged.bonds is not None
        assert len(merged.bonds) == 4
        
        # Check bond endpoints are valid aids
        all_aids = set(merged.atoms["aid"].tolist())
        for _, bond in merged.bonds.iterrows():
            assert bond["a1"] in all_aids
            assert bond["a2"] in all_aids


class TestMergePreservingFirstUniqueNames:
    """Integration tests for merge_preserving_first() with unique names."""
    
    def _make_usm(self, atoms_data: dict) -> USM:
        """Helper to create a USM instance."""
        atoms = pd.DataFrame(atoms_data)
        return USM(
            atoms=atoms,
            bonds=None,
            molecules=None,
            cell={"pbc": False},
            provenance={},
            preserved_text={},
        )
    
    def test_preserves_first_names(self):
        """Test that first structure's names are always preserved."""
        usm1 = self._make_usm({
            "aid": [0, 1, 2],
            "name": ["C1", "O1", "O2"],
            "element": ["C", "O", "O"],
            "x": [0.0, 1.0, 2.0],
            "y": [0.0, 0.0, 0.0],
            "z": [0.0, 0.0, 0.0],
        })
        
        usm2 = self._make_usm({
            "aid": [0, 1, 2],
            "name": ["C1", "O1", "O2"],  # Same names!
            "element": ["C", "O", "O"],
            "x": [5.0, 6.0, 7.0],
            "y": [0.0, 0.0, 0.0],
            "z": [0.0, 0.0, 0.0],
        })
        
        merged = merge_preserving_first(usm1, usm2, cell_policy="first")
        
        # First 3 atoms should have original names
        first_three_names = merged.atoms.iloc[:3]["name"].tolist()
        assert first_three_names == ["C1", "O1", "O2"]
        
        # Last 3 should have unique (renamed) names
        last_three_names = merged.atoms.iloc[3:]["name"].tolist()
        all_names = first_three_names + last_three_names
        assert len(set(all_names)) == 6, f"Duplicates: {all_names}"
