USM v0.1 — Workflow Playbooks and Best Practices

Purpose
Operational notes to remember when building complex structures (e.g., grafting ligands onto slabs, merging multiple components) so outputs remain physically consistent and IO is lossless.

Key principles
- Always compose host CAR+MDF before edits
  - Load host CAR (coords) and host MDF (topology) and compose them so bonds are present during all edits. This ensures MDF connections are emitted for the slab on save.
  - See host compose pattern in workspaces (e.g., graft pipeline).
- Guest molecule should carry its bonds
  - Load guest MDF and compose (guest CAR+MDF) so intra-guest bonds are preserved. If dropping atoms (e.g., phenolic H), do so before placement.
- No cross-interface bonds when using nonbonded anchoring
  - If your physical model uses nonbonded terminations, do not create host–guest bonds at the interface. Export will include slab bonds and intra-guest bonds only.
- After growth, resize the periodic cell
  - After merging or grafting bulky ligands, resize the PBC cell to enclose all atoms plus padding and wrap coordinates into [0, L) to avoid visual wrap-around.
  - Orthorhombic helper: see [usm/ops/cell.py](usm/ops/cell.py).
- Save MDF with normalized connections after topology edits
  - When bonds have changed, prefer normalized connections on export. Use lossless raw-token mode only for identity round-trips.

Grafting playbook (general)
1) Load and compose host
   - host = compose_on_keys(load_car(host.car), load_mdf(host.mdf))
2) Detect sites and pick targets
   - Pick site anchors deterministically (e.g., by distance pairing and per-surface policy).
3) Load and prepare guest
   - guest = compose_on_keys(load_car(guest.car), load_mdf(guest.mdf))
   - Drop atoms (e.g., proton) if reaction semantics require it.
4) Orientation and clash control
   - Pin guest anchor atom at site anchor coordinate.
   - Align a guest axis (anchor → preferred neighbor) to a target axis (e.g., surface normal).
   - Scan torsion and apply small outward tilt; reject clashing orientations via min-distance screen.
5) Merge accepted placements
   - Remove host site atoms (e.g., original –OH O and H), merge guest clone, assign unique mol_index, renumber aids.
6) Resize cell and wrap
   - Compute bounds, set new orthorhombic a,b,c with padding, recenter, and wrap to [0, L).
   - See [usm/ops/cell.py](usm/ops/cell.py).
7) Save CAR and MDF
   - CAR: write with synthesized header if cell was changed, so updated PBC is serialized.
   - MDF: write_normalized_connections=True to regenerate connections from bonds consistently.

MXN + Dopamine (DOP) notes
- Reaction semantics
  - Remove the slab’s –OH pair (O and paired H) at each selected site.
  - Drop dopamine’s phenolic H (e.g., H10 on O1). Dopamine O1 becomes the anchor –O at the former –OH O coordinate.
  - Interface remains nonbonded.
- Orientation defaults (tunable)
  - Align O1→(preferred ring carbon) to the surface normal (+z top, −z bottom).
  - Scan torsion (15° steps) and apply a fixed 20° outward tilt; accept first non-clashing orientation.
- PBC cell growth
  - After grafting, resize the cell (commonly along c, the slab normal) and add vacuum padding to prevent wrap-around visuals.

Common pitfalls and remedies
- “Missing bonds in MDF”
  - Cause: Host loaded from CAR only (no bonds), or saved MDF without normalized connections after edits.
  - Fix: Compose host CAR+MDF at start; save MDF with normalized connections.
- “New ligand atoms wrap across boundaries”
  - Cause: Cell not resized after adding bulky ligands.
  - Fix: Apply auto resize along relevant axes with padding and wrap.
- “Non-deterministic selection or ordering”
  - Cause: Unseeded random picks or unstable sort keys.
  - Fix: Seed RNG and use stable ordering (e.g., sort by layer, mol_index, name), then renumber.

Recommended configs (slabs)
- Cell resize (post-merge):
  - adjust_axes: ["c"]
  - pad: c ≈ 10 Å (or larger if ligands extend far)
  - recenter: "min" or "center", wrap: true
- Orientation search:
  - torsion_step_deg: 12–15°
  - outward tilt: 15–25°
  - (Optional) outward lift along normal by ~1.5–2.5 Å for bulkier ligands

Where to look in code
- Host/guest compose and MDF emit: [usm/io/car.py](usm/io/car.py), [usm/io/mdf.py](usm/io/mdf.py), [usm/ops/compose.py](usm/ops/compose.py)
- Sites and pairing: [usm/ops/sites.py](usm/ops/sites.py)
- Orientation/frames: [usm/ops/frames.py](usm/ops/frames.py)
- Collision checks: [usm/ops/collide.py](usm/ops/collide.py)
- Unique molecule indexing: [usm/ops/label.py](usm/ops/label.py)
- Cell resize/wrap: [usm/ops/cell.py](usm/ops/cell.py)
- Grafting orchestrator: [usm/ops/graft_ops.py](usm/ops/graft_ops.py)