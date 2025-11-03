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
- Paths in these documents may refer to `usm/...` modules. In this repository, those modules live under `24-MOLSAIC-V3/src/usm/...`.
- Some links to tests and examples may reference the repository root (e.g., `tests/...`, `examples/...`).
- When the split is finalized, these mirrored docs will be replaced by links to the external USM library documentation.