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

# Optional: include CONECT from bonds (deduplicated, 4 neighbors per line, deterministic)
save_pdb(usm2, "output/WAT_shifted_conect.pdb", include_conect=True)

# Optional: wrap in a single MODEL/ENDMDL block (MODEL index defaults to 1)
save_pdb(usm2, "output/WAT_shifted_model.pdb", include_model=True, model_index=1)

# Optional: combine MODEL and CONECT with the "full" policy (emit both directions)
save_pdb(
  usm2,
  "output/WAT_shifted_model_conect.pdb",
  include_model=True,
  model_index=1,
  include_conect=True,
  conect_policy="full",
)

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

# Policy-controlled coverage diagnostics (optional)
# Compute composition coverage over unique keys; attach metrics to provenance and optionally return a report dict
from usm.ops.compose import compose_on_keys

composed2, report = compose_on_keys(
    wat_car,
    wat_mdf,
    policy="warn",                 # "silent" (default), "warn", or "error_below_coverage"
    coverage_threshold=0.95,       # trigger warn/error when coverage_ratio < threshold
    return_report=True,            # also return structured coverage metrics
)
print("compose coverage:", report)  # e.g., {"coverage_ratio": 1.0, "matched_count": N, ...}

# You can also read the same metrics from provenance
print((composed2.provenance or {}).get("compose_coverage", {}))

# If below threshold and policy="warn", a deterministic message is appended to provenance.parse_notes
if report["coverage_ratio"] < report["coverage_threshold"]:
    print("Warning:", (composed2.provenance or {}).get("parse_notes", ""))

# If policy="error_below_coverage", compose_on_keys raises ValueError when coverage is insufficient
try:
    compose_on_keys(wat_car, wat_mdf, policy="error_below_coverage", coverage_threshold=0.95)
except ValueError as e:
    print("Composition failed:", e)

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
from usm.bundle.io import save_bundle, load_bundle  # [save_bundle()](src/usm/bundle/io.py:43), [load_bundle()](src/usm/bundle/io.py:118)

# Compose WAT CAR + MDF to get coords + bonds, then serialize and validate
bundle_dir = save_bundle(composed, "output/usm_bundle_wat")
loaded = load_bundle(bundle_dir)

# Basic validation of round-trip counts and metadata
assert len(loaded.atoms) == len(composed.atoms)
assert (0 if composed.bonds is None else len(composed.bonds)) == (0 if loaded.bonds is None else len(loaded.bonds))
assert (0 if composed.molecules is None else len(composed.molecules)) == (0 if loaded.molecules is None else len(loaded.molecules))
assert dict(loaded.cell) == dict(composed.cell)
print("bundle round-trip ok:", True)

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
 
 Runner numeric checks (LB_SF_carmdf)
 - After MDF round-trip, the workspace runner [run_scenario()](workspaces/other/usm_lb_sf_carmdf_v1/run.py:260) computes per-column metrics for MDF numeric fields.
 - The summary.json contains:
   {
     "validations": {
       "mdf_header_equal": true,
       "mdf_connections_equal": true,
       "mdf_numeric": {
         "charge": { "max_abs_diff": 0.0, "exact_equal": true, "atol_used": 1e-06, "rtol_used": 0.0 },
         "occupancy": { "max_abs_diff": 0.0, "exact_equal": true, "atol_used": 1e-06, "rtol_used": 0.0 },
         "xray_temp_factor": { "max_abs_diff": 0.0, "exact_equal": true, "atol_used": 1e-06, "rtol_used": 0.0 },
         "switching_atom": { "exact_equal": true, "mismatch_count": 0 },
         "oop_flag": { "exact_equal": true, "mismatch_count": 0 },
         "chirality_flag": { "exact_equal": true, "mismatch_count": 0 }
       },
       "mdf_numeric_ok": true,
       "mdf_roundtrip_ok": true
     }
   }
 - Configure tolerances per float column via config key mdf_numeric_tolerances. Value can be a number (atol) or an object { "atol": float, "rtol": float }.
   Example:
   {
     "outputs_dir": "workspaces/other/usm_lb_sf_carmdf_v1/outputs",
     "scenarios": { "A": { "car": "assets/LB_SF_carmdf/FAPbBr2I.car", "mdf": "assets/LB_SF_carmdf/FAPbBr2I.mdf" } },
     "mdf_numeric_tolerances": {
       "charge": 1e-6,
       "occupancy": { "atol": 1e-6, "rtol": 0.0 },
       "xray_temp_factor": 1e-6
     }
   }
 
 Tips
- Determinism: All operations produce stable ordering and contiguous ids (aid/bid/mid) according to documented policies.
- PBC: wrap_to_cell and replicate_supercell support general triclinic lattices via fractional coordinates; orthorhombic fast path preserved for performance.
- Bonds: Normalized from MDF connections; raw tokens are preserved for lossless MDF round-trip.

8) Multi-generation writer stability checks (optional)
- Purpose: Assert multi-generation stability of CAR/MDF writers by comparing only the deterministic core sections.
- Runner integration: enable in the workspace runner [run_scenario()](workspaces/other/usm_lb_sf_carmdf_v1/run.py:345) via config key "stability_checks".

Runner config snippet (JSON)
{
  "outputs_dir": "workspaces/other/usm_lb_sf_carmdf_v1/outputs",
  "stability_checks": true,
  "scenarios": {
    "A": {
      "car": "assets/LB_SF_carmdf/FAPbBr2I.car",
      "mdf": "assets/LB_SF_carmdf/FAPbBr2I.mdf"
    }
  }
}

Behavior
- CAR: load → save gen2 → load gen2 → save gen3, using [save_car()](src/usm/io/car.py:187) with preserve_headers=False for deterministic header synthesis. Compare only atom lines that match [ATOM_LINE_RE](src/usm/io/car.py:17).
- MDF (canonical): load → save gen2 → load gen2 → save gen3, using [save_mdf()](src/usm/io/mdf.py:378) with preserve_headers=False and write_normalized_connections=True. Compare only topology lines that match [MDF_LINE_RE](src/usm/io/mdf.py:18).

Outputs
- Files (for debugging on failure):
  - outputs/{scenario}/car_gen2.car, car_gen3.car
  - outputs/{scenario}/mdf_gen2.mdf, mdf_gen3.mdf
- Summary fields in outputs/{scenario}/summary.json:
  - validations.car_text_stable_across_generations: true|false
  - validations.mdf_text_stable_across_generations: true|false

Notes
- The CAR comparison excludes date and non-deterministic headers by comparing only lines matching [ATOM_LINE_RE](src/usm/io/car.py:17).
- The MDF comparison is performed under a canonical writer mode to avoid dependence on original headers or raw connections, using the normalized connection emission of [save_mdf()](src/usm/io/mdf.py:378).
8) Workspace autodiscover (LB_SF_carmdf)
JSON
{
  "outputs_dir": "workspaces/other/usm_lb_sf_carmdf_v1/outputs",
  "autodiscover": {
    "root": "assets/LB_SF_carmdf",
    "include_patterns": ["*.mdf", "*.car"],
    "pairing": { "mode": "stem_exact" }
  },
  "stability_checks": true,
  "ops": {
    "wrap_to_cell": true,
    "replicate": [2, 1, 1],
    "select_first_n": 100,
    "transform": { "translate": [0.1, 0.0, 0.0], "rotate_deg_z": 5 }
  }
}

- Purpose: Automatically discover .car/.mdf under assets/LB_SF_carmdf, pair by filename stem, and run scenarios via [workspaces/other/usm_lb_sf_carmdf_v1/run.py](workspaces/other/usm_lb_sf_carmdf_v1/run.py:1).
- Pairing: Lowercase stem grouping; when both stem.car and stem.mdf exist they form a paired scenario; otherwise single-input scenario is created.
- Run:
  - python workspaces/other/usm_lb_sf_carmdf_v1/run.py --config workspaces/other/usm_lb_sf_carmdf_v1/config.json
- Artefacts:
  - Top-level inventory: workspaces/other/usm_lb_sf_carmdf_v1/outputs/_inventory.json
  - Per-scenario folder: composed.car/.mdf/.pdb, round-trip files, stability files (car_gen2/3, mdf_gen2/3), optional ops outputs, bundle folder.
- Coverage: Paired scenarios compose CAR coords with MDF topology using [compose_on_keys()](src/usm/ops/compose.py:13) (policy "warn" by default) and record coverage metrics in summary.json.