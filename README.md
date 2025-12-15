# USM (Unified Structus) — Local Docs Mirror

This directory vendors the USM library (data model + IO + ops) for MolSAIC to be self-contained. The authoritative plan is to split USM into a separate library repository and have MolSAIC depend on it. Until then, the docs below are mirrored locally for convenience.

Docs (mirrored under ./docs):
- [API.md](./docs/API.md)
- [DATA_MODEL.md](./docs/DATA_MODEL.md)
- [DESIGN.md](./docs/DESIGN.md)
- [EXAMPLES.md](./docs/EXAMPLES.md)
- [LIMITS.md](./docs/LIMITS.md)
- [PERFORMANCE.md](./docs/PERFORMANCE.md)
- [WORKFLOWS.md](./docs/WORKFLOWS.md)
- [MOLSAIC_V2_DESIGN.md](./docs/MOLSAIC_V2_DESIGN.md)

Notes:
- Paths in these documents may refer to `usm/...` modules. In this repository, those modules live under [`src/usm/`](src/usm:1).
- Some links to tests and examples may reference the repository root (e.g., [`tests/`](tests:1)).
- When the split is finalized, these mirrored docs will be replaced by links to the external USM library documentation.

Recent additions (v0.1+ in this repo):
- CIF I/O (minimal): [`load_cif()`](src/usm/io/cif.py:222) and [`save_cif()`](src/usm/io/cif.py:365)
  - Intended for cell + `atom_site` positions; advanced crystallography (symmetry expansion/disorder models) is out of scope.
- Optional per-atom parameter columns on [`USM`](src/usm/core/model.py:90) atoms table:
  - `mass_amu`, `lj_epsilon_kcal_mol`, `lj_sigma_angstrom` (nullable)
  - These are carried in the USM tables/bundles; CAR/MDF writers do not serialize them by default.
- Project-specific parameter assignment logic should live in workspaces (e.g., NIST demo), not in generic USM ops.