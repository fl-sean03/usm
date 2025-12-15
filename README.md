# USM (Unified Structure Model)

USM is a small Python library for representing atomistic structures (atoms + bonds + periodic cell), performing deterministic structure operations, and reading/writing common structure file formats.

Docs (under [`docs/`](docs:1)):
- [API.md](docs/API.md:1)
- [DATA_MODEL.md](docs/DATA_MODEL.md:1)
- [DESIGN.md](docs/DESIGN.md:1)
- [EXAMPLES.md](docs/EXAMPLES.md:1)
- [LIMITS.md](docs/LIMITS.md:1)
- [PERFORMANCE.md](docs/PERFORMANCE.md:1)
- [WORKFLOWS.md](docs/WORKFLOWS.md:1)

Highlights:
- Minimal CIF I/O: [`load_cif()`](io/cif.py:222) and [`save_cif()`](io/cif.py:365)
  - Intended for cell + `atom_site` positions; advanced crystallography (symmetry expansion/disorder models) is out of scope.
- Core data model: [`USM`](core/model.py:90) with a schema-checked atoms table (and bonds/cell metadata).
- Deterministic ops for selection/transform/replicate/merge/compose/renumber/etc. under [`ops/`](ops:1).

Notes:
- This package intentionally stays dependency-light (NumPy/Pandas). If you need full crystallography support, consider layering a specialized library (e.g., gemmi/ASE/pymatgen) outside of USM and converting to/from [`USM`](core/model.py:90).