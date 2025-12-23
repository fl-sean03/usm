import os
import pytest
import numpy as np
from usm.io.cif import load_cif, _parse_symop_string, _parse_symmetry_code
from usm.ops.lattice import lattice_matrix

def test_parse_symop_string():
    # Identity
    rot, trans = _parse_symop_string("x, y, z")
    assert np.allclose(rot, np.eye(3))
    assert np.allclose(trans, 0)

    # Rotation and translation
    rot, trans = _parse_symop_string("-x, y+1/2, -z+1/2")
    expected_rot = np.array([[-1, 0, 0], [0, 1, 0], [0, 0, -1]])
    expected_trans = np.array([0, 0.5, 0.5])
    assert np.allclose(rot, expected_rot)
    assert np.allclose(trans, expected_trans)

def test_parse_symmetry_code():
    op_idx, trans = _parse_symmetry_code("3_556")
    assert op_idx == 2
    assert np.allclose(trans, [0, 0, 1])

    op_idx, trans = _parse_symmetry_code("2")
    assert op_idx == 1
    assert np.allclose(trans, [0, 0, 0])

    op_idx, trans = _parse_symmetry_code(".")
    assert op_idx == 0
    assert np.allclose(trans, [0, 0, 0])

def test_calf20_load_expanded(tmp_path):
    # Use the actual file if available, or skip
    cif_path = "assets/NIST/2084733.cif"
    import os
    if not os.path.exists(cif_path):
        pytest.skip("CALF-20 CIF not found")

    usm = load_cif(cif_path, expand_symmetry=True)
    
    # Check atom counts
    # 11 AU atoms * 4 ops = 44 atoms
    assert len(usm.atoms) == 44
    
    # Check Zn coordination
    zn_indices = usm.atoms[usm.atoms["element"] == "Zn"].index
    for idx in zn_indices:
        b_count = len(usm.bonds[(usm.bonds["a1"] == idx) | (usm.bonds["a2"] == idx)])
        assert b_count == 5

    # Check bond count
    assert len(usm.bonds) == 58

def test_occupancy_filtering():
    # Create a dummy CIF with partial occupancy
    cif_content = """
data_test
_cell_length_a 10
_cell_length_b 10
_cell_length_c 10
_cell_angle_alpha 90
_cell_angle_beta 90
_cell_angle_gamma 90
loop_
_space_group_symop_operation_xyz
'x, y, z'
loop_
_atom_site_label
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
_atom_site_occupancy
C1 0.1 0.1 0.1 1.0
O1 0.2 0.2 0.2 0.5
"""
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.cif', delete=False) as f:
        f.write(cif_content)
        fname = f.name
    
    try:
        # Without expansion, all atoms loaded (current behavior in AU)
        # Wait, I changed load_cif to always parse occupancy but Rule D1 only applies if expand_symmetry=True
        usm_no_exp = load_cif(fname, expand_symmetry=False)
        assert len(usm_no_exp.atoms) == 2
        
        usm_exp = load_cif(fname, expand_symmetry=True)
        assert len(usm_exp.atoms) == 1
        assert usm_exp.atoms.iloc[0]["element"] == "C"
    finally:
        os.remove(fname)
