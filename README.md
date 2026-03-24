# USM — Unified Structure Model

USM is a standalone Python library for atomistic structure I/O and deterministic manipulation. It supports CAR, MDF, CIF, and PDB formats with full read/write capability.

## Install

```bash
pip install -e ".[dev]"
```

## Quick Start

```python
import usm

# Auto-detect format by extension
structure = usm.load("structure.cif")

# Or use format-specific loaders
from usm.io import load_car, load_mdf, load_pdb, save_pdb
car = load_car("coords.car")
mdf = load_mdf("topology.mdf")
pdb = load_pdb("protein.pdb")

# Compose geometry + topology
from usm.ops.compose import compose_on_keys
combined = compose_on_keys(car, mdf)

# Transform
from usm.ops.transform import translate, wrap_to_cell
from usm.ops.replicate import replicate_supercell
wrapped = wrap_to_cell(combined)
supercell = replicate_supercell(wrapped, 2, 2, 1)

# Export
save_pdb(supercell, "output.pdb", include_conect=True)
```

## Supported Formats

| Format | Read | Write | Description |
|--------|------|-------|-------------|
| CAR | Yes | Yes | Materials Studio coordinates (header-preserving roundtrip) |
| MDF | Yes | Yes | Materials Studio topology (connections_raw preserved) |
| CIF | Yes | Yes | Crystallographic Information File (lattice + atom_site) |
| PDB | Yes | Yes | Protein Data Bank (ATOM/CRYST1/CONECT) |
| Bundle | Yes | Yes | USM native (Parquet/CSV + manifest.json) |

## Core Features

- **Deterministic operations**: All ops produce identical output for identical input
- **Lossless I/O**: CAR/MDF headers and footers preserved byte-for-byte
- **General triclinic PBC**: Full lattice math (not just orthorhombic)
- **Schema-enforced data model**: Atoms, bonds, molecules DataFrames with strict dtypes
- **Minimal dependencies**: NumPy + Pandas only (SciPy optional for performance)

## Operations

- `compose_on_keys` — Join CAR geometry with MDF topology
- `replicate_supercell` — PBC supercell with bond materialization
- `merge_structures` — Combine multiple structures with collision handling
- `translate`, `rotate`, `scale`, `wrap_to_cell` — Coordinate transforms
- `select_by_element`, `select_by_mask`, `select_box` — Atom selection
- `renumber_atoms` — Deterministic re-indexing
- `perceive_periodic_bonds` — Auto-detect PBC bonds from geometry
- Requirement/termset/parameterset derivation for UPM integration

## Documentation

- [API Reference](docs/API.md)
- [Data Model](docs/DATA_MODEL.md)
- [Design](docs/DESIGN.md)
- [Examples](docs/EXAMPLES.md)
- [Limits](docs/LIMITS.md)

## Design Principles

- **Standalone**: No dependency on UPM, orchestrators, or chemistry libraries
- **Deterministic**: Canonical ordering, stable sorts, reproducible output
- **Dependency-light**: NumPy + Pandas; layer specialized libs (ASE/pymatgen) externally
- **Schema-first**: DataFrame columns and dtypes are enforced at construction
