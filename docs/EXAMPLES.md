USM v0.1 — Examples

This guide shows end-to-end flows using the WAT and DOP structures, demonstrating import → operate → export and composition of CAR (coordinates) with MDF (topology).

Prereqs
- Python 3.10+
- Pandas, NumPy
- Optional: pyarrow or fastparquet for Parquet. If unavailable, USM Bundle falls back to CSV automatically.

1) Load WAT (CAR), translate, and export CAR/MDF/PDB
Python
from usm.io.car import load_car, save_car
from usm.io.mdf import save_mdf
from usm.io.pdb import save_pdb
from usm.ops.transform import translate

# Load CAR (header/footer preserved)
usm = load_car("msiExamples/WAT.car")

# Translate by +1 Å along x
usm2 = translate(usm.copy(), (1.0, 0.0, 0.0))

# Save back to CAR with headers/footers preserved
save_car(usm2, "output/WAT_shifted.car", preserve_headers=True)

# Save to MDF (synthesizes a canonical header if none preserved)
save_mdf(usm2, "output/WAT_shifted.mdf", preserve_headers=False)

Note
- The MDF writer preserves formal_charge tokens without truncation. Tokens longer than width 3 (e.g., 1/2+) are emitted verbatim with single-space field delimiters, ensuring lossless round-trip via [python.load_mdf()](src/usm/io/mdf.py:243) and [python.save_mdf()](src/usm/io/mdf.py:378).
# Save to PDB (minimal writer)
save_pdb(usm2, "output/WAT_shifted.pdb")

2) Load DOP (MDF), inspect bonds, select, and export MDF
Python
from usm.io.mdf import load_mdf, save_mdf
from usm.ops.select import select_by_element, select_within_radius

dop = load_mdf("msiExamples/DOP.mdf")
print(dop.atoms.head())
print(dop.bonds.head())  # bonds parsed from connections_raw

# Select all carbons
carbons = select_by_element(dop, ["C"])
print(len(dop.atoms), len(carbons.atoms))

# Select within a radius of 2.0 Å around the first atom (example)
center = dop.atoms.loc[0, ["x","y","z"]].fillna(0.0).to_numpy(dtype=float)
subset = select_within_radius(dop, center, 2.0)

# Save MDF preserving headers/footer (and connections_raw)
save_mdf(subset, "output/DOP_subset.mdf", preserve_headers=True)

3) Compose CAR (coords) with MDF (bonds) via stable keys
Python
from usm.io.car import load_car
from usm.io.mdf import load_mdf
from usm.ops.compose import compose_on_keys

wat_car = load_car("msiExamples/WAT.car")
wat_mdf = load_mdf("msiExamples/WAT.mdf")

# Compose MDF topology into CAR coordinates
composed = compose_on_keys(wat_car, wat_mdf)  # keys = ["mol_label","mol_index","name"] by default

# Bonds now present; connections preserved inside atoms.connections_raw if needed for MDF round-trip
print(composed.bonds.head())

4) Merge multiple structures and renumber consistently
Python
from usm.ops.merge import merge_structures
from usm.ops.renumber import renumber_atoms, renumber_molecules

# Merge two copies (e.g., as a simple combine)
merged = merge_structures([wat_car, wat_car], cell_policy="first")

# Deterministically renumber aids
renum = renumber_atoms(merged, order_by=["mol_index","name"])

# Derive molecule ids mid (optional)
renum2 = renumber_molecules(renum)

5) Supercell replication (general triclinic)
Python
from usm.ops.replicate import replicate_supercell

# Example: monoclinic cell (beta=100°)
mono = wat_car.copy()
mono.cell.update({"pbc": True, "a": 10.0, "b": 12.0, "c": 8.0, "alpha": 90.0, "beta": 100.0, "gamma": 90.0})
mono_rep = replicate_supercell(mono, 2, 1, 3)

# Example: hexagonal cell (gamma=120°)
hexa = wat_car.copy()
hexa.cell.update({"pbc": True, "a": 10.0, "b": 10.0, "c": 15.0, "alpha": 90.0, "beta": 90.0, "gamma": 120.0})
hexa_rep = replicate_supercell(hexa, 3, 2, 1)

print(len(mono.atoms), len(mono_rep.atoms))
print(len(hexa.atoms), len(hexa_rep.atoms))

6) USM Bundle: Save and load (Parquet preferred, CSV fallback)
Python
from usm.bundle.io import save_bundle, load_bundle

# Compose WAT CAR + MDF to get coords + bonds, then serialize
bundle_dir = save_bundle(composed, "output/usm_bundle_wat")
loaded = load_bundle(bundle_dir)
print(len(loaded.atoms), "atoms")

7) Round-trip fidelity checks (snippets)
- CAR round-trip WAT: load → save_car(preserve_headers=True) → load; verify header/footer lines identical; xyz within tolerance
Python
from usm.io.car import load_car, save_car
import numpy as np

u1 = load_car("msiExamples/WAT.car")
save_car(u1, "output/WAT_rt.car", preserve_headers=True)
u2 = load_car("output/WAT_rt.car")

h1 = (u1.preserved_text or {}).get("car_header_lines", [])
h2 = (u2.preserved_text or {}).get("car_header_lines", [])
assert h1 == h2

xyz1 = u1.atoms.sort_values("aid")[["x","y","z"]].to_numpy()
xyz2 = u2.atoms.sort_values("aid")[["x","y","z"]].to_numpy()
assert np.allclose(xyz1, xyz2, atol=1e-5)

- MDF round-trip WAT: load → save_mdf(preserve_headers=True, write_normalized_connections=False) → load; verify headers equal and connections_raw identical per atom name
Python
from usm.io.mdf import load_mdf, save_mdf

m1 = load_mdf("msiExamples/WAT.mdf")
save_mdf(m1, "output/WAT_rt.mdf", preserve_headers=True, write_normalized_connections=False)
m2 = load_mdf("output/WAT_rt.mdf")

h1 = (m1.preserved_text or {}).get("mdf_header_lines", [])
h2 = (m2.preserved_text or {}).get("mdf_header_lines", [])
assert h1 == h2

name_conn1 = dict(zip(m1.atoms["name"].astype(str), m1.atoms["connections_raw"].fillna("").astype(str)))
name_conn2 = dict(zip(m2.atoms["name"].astype(str), m2.atoms["connections_raw"].fillna("").astype(str)))
assert name_conn1 == name_conn2

Tips
- Determinism: All operations produce stable ordering and contiguous ids (aid/bid/mid) according to documented policies.
- PBC: wrap_to_cell and replicate_supercell support general triclinic lattices via fractional coordinates; orthorhombic fast path preserved for performance.
- Bonds: Normalized from MDF connections; raw tokens are preserved for lossless MDF round-trip.