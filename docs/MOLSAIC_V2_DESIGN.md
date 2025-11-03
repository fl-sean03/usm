USM-Based MolSAIC V2 — Design and Development Plan

Summary
MolSAIC V2 replaces plugin registries, YAML DAGs, and executor indirection with a code-first pattern that mirrors this repository’s Unified Structure Model library and workspaces. The core becomes a small, dependable Python library for structure IO and deterministic ops, and pipelines are simple, reproducible scripts under workspaces with minimal JSON configs.

Status
This document serves as the authoritative plan for V2 implementation. The USM reference in this repository is the baseline for data model, IO, and core operations.

References
- Data model: [docs/DATA_MODEL.md](docs/DATA_MODEL.md)
- API: [docs/API.md](docs/API.md)
- Design v0.1: [docs/DESIGN.md](docs/DESIGN.md)
- Examples: [docs/EXAMPLES.md](docs/EXAMPLES.md)
- Workflows: [docs/WORKFLOWS.md](docs/WORKFLOWS.md)
- Limits: [docs/LIMITS.md](docs/LIMITS.md)
- Performance: [docs/PERFORMANCE.md](docs/PERFORMANCE.md)
- IO modules: [usm/io/car.py](usm/io/car.py), [usm/io/mdf.py](usm/io/mdf.py), [usm/io/pdb.py](usm/io/pdb.py), [usm/bundle/io.py](usm/bundle/io.py)
- Selected workspaces: [workspaces/mxn_dop_graft_v1/run.py](workspaces/mxn_dop_graft_v1/run.py), [workspaces/mxn_f_sub_v2/run.py](workspaces/mxn_f_sub_v2/run.py), [workspaces/mxn_f_rand_csr_v2/run.py](workspaces/mxn_f_rand_csr_v2/run.py), [workspaces/mxn_orientation_v1/run.py](workspaces/mxn_orientation_v1/run.py), [workspaces/mxn_zbins_v1/run.py](workspaces/mxn_zbins_v1/run.py), [workspaces/replace_remove_dop_v1/run.py](workspaces/replace_remove_dop_v1/run.py), [workspaces/mxn_counts_v1/run.py](workspaces/mxn_counts_v1/run.py)

1. Goals
- Reduce complexity and ceremony while preserving fidelity and determinism
- Provide a cohesive library surface for structure IO, transforms, selection, renumbering, composition, cell helpers, and grafting utilities
- Make pipelines plain Python scripts, easy to debug and version-control, with consistent outputs
- Ensure reproducibility and portability via small JSON configs and stable outputs

2. Out of scope for V2
- Plugin registry, entry points, and dynamic plugin loading
- YAML pipeline specifications and multi-executor orchestration
- Event bus and cross-process observability streams
- Advanced chemical perception or force-field parameterization beyond pass-through preservation

3. Core principles
- USM as single source of truth for schema, ids, and IO fidelity
- Deterministic operations and stable renumbering
- Simplicity first: direct function calls, no hidden indirection
- Workspace conventions over frameworks

4. Architecture overview
V2 consists of a small Python library plus a set of workspace scripts that call it.

Mermaid
flowchart TD
  A[USM IO] --> B[USM Ops]
  B --> C[Workspace Scripts]
  A --> C
  C --> D[Outputs]
  D --> E[Summaries and Plots]
  style D fill:#eaffea,stroke:#93c893

Components
- USM IO
  - Direct readers and writers for CAR and MDF with preservation, and a best effort PDB writer
  - Bundle folder format with Parquet preferred and CSV fallback
  - See [usm/io/car.py](usm/io/car.py), [usm/io/mdf.py](usm/io/mdf.py), [usm/io/pdb.py](usm/io/pdb.py), [usm/bundle/io.py](usm/bundle/io.py)
- USM Ops
  - Selection, transforms, replication, merge, compose, renumber, cell helpers, site finding, graft placement and collision checks
  - See modules noted in [docs/WORKFLOWS.md](docs/WORKFLOWS.md) and tests under [tests/](tests)
- Workspace Scripts
  - Self contained run.py and minimal config.json, writing outputs under outputs
  - Examples in [workspaces/](workspaces)

5. Repository layout
- usm core and io as in this repo
- usm ops organized by concern
- workspaces as user facing pipelines
- docs and tests as contract

Proposed structure adjustments
- Centralize repeated helpers currently embedded in workspaces into shared modules under usm ops
- Provide a workspace template for new pipelines

6. Workspace contract
Each workspace directory contains:
- run.py with a main entry that logs step messages and writes a summary manifest
- config.json optional override of defaults embedded in the script
- outputs directory for artifacts

Required behavior
- Deterministic logic with seeded randomness where applicable
- All file system writes happen under the workspace outputs directory
- summary.json includes inputs, parameters, counts, metrics, and output file paths

Reference implementations
- Grafting: [workspaces/mxn_dop_graft_v1/run.py](workspaces/mxn_dop_graft_v1/run.py), bottom side variant [workspaces/mxn_dop_graft_bottom_v1/run.py](workspaces/mxn_dop_graft_bottom_v1/run.py)
- Substitution: [workspaces/mxn_f_sub_v2/run.py](workspaces/mxn_f_sub_v2/run.py), bonds based variant [workspaces/mxn_f_sub_v1/run.py](workspaces/mxn_f_sub_v1/run.py)
- CSR selection and visualization: [workspaces/mxn_f_rand_csr_v2/run.py](workspaces/mxn_f_rand_csr_v2/run.py)
- Orientation analysis: [workspaces/mxn_orientation_v1/run.py](workspaces/mxn_orientation_v1/run.py)
- Z bin counts and type counts: [workspaces/mxn_zbins_v1/run.py](workspaces/mxn_zbins_v1/run.py), [workspaces/mxn_counts_v1/run.py](workspaces/mxn_counts_v1/run.py)
- Targeted edit workflow: [workspaces/replace_remove_dop_v1/run.py](workspaces/replace_remove_dop_v1/run.py)

7. Library surface to stabilize
IO
- CAR: load and save with preserved header and footer
- MDF: load and save with preserved headers and lossless connections_raw, plus normalized connections mode
- PDB: minimal writer with CRYST1 when PBC available
- Bundle: save and load with Parquet preferred and CSV fallback
See [usm/io/car.py](usm/io/car.py), [usm/io/mdf.py](usm/io/mdf.py), [usm/io/pdb.py](usm/io/pdb.py), [usm/bundle/io.py](usm/bundle/io.py)

Core ops
- Selection utilities as documented in [docs/API.md](docs/API.md)
- Transforms including translate, rotate, scale, wrap to cell
- Replicate supercell for orthorhombic cells
- Merge and compose for joining tables and reconciling bonds
- Renumber atoms and molecules with deterministic policies
- Cell helpers for resize, recenter, wrap

Grafting related ops
- Site finding including O plus H pairing by cutoff distance or bonds
- Surface split threshold for top versus bottom classification
- Site selection policies half or exact counts with optional spacing
- Orientation frames and torsion tilt scanning
- Clash checks and acceptance policies
- Placement result manifest model with metrics
Reference scripts: [workspaces/mxn_dop_graft_v1/run.py](workspaces/mxn_dop_graft_v1/run.py), [workspaces/mxn_dop_graft_bottom_v1/run.py](workspaces/mxn_dop_graft_bottom_v1/run.py)

8. Helpers to centralize in usm ops
The following functions exist in multiple workspaces and should be moved into shared modules to reduce duplication and ensure uniform behavior.

Surface split and side counts
- Threshold computation with auto, median, and midrange strategies
- Side count helpers for reporting top versus bottom counts for given types
Sources: [workspaces/mxn_f_sub_v2/run.py](workspaces/mxn_f_sub_v2/run.py), [workspaces/mxn_f_all_v1/run.py](workspaces/mxn_f_all_v1/run.py), [workspaces/mxn_zbins_v1/run.py](workspaces/mxn_zbins_v1/run.py)

OH pairing
- Distance based greedy nearest neighbor matching between O and H
- Bonds based neighbor extraction for O to H
Sources: [workspaces/mxn_f_all_v1/run.py](workspaces/mxn_f_all_v1/run.py), [workspaces/mxn_f_sub_v1/run.py](workspaces/mxn_f_sub_v1/run.py), [workspaces/mxn_f_sub_v2/run.py](workspaces/mxn_f_sub_v2/run.py)

MDF connections preservation helpers
- Build map of key to raw connections
- Cleanse tokens when removing or replacing atoms
Sources: [workspaces/mxn_f_all_v1/run.py](workspaces/mxn_f_all_v1/run.py), [workspaces/mxn_f_sub_v2/run.py](workspaces/mxn_f_sub_v2/run.py)

CSR selection utilities
- Ripley metric based best of trials selection
- Gallery sampling and plotting integration hooks
Source: [workspaces/mxn_f_rand_csr_v2/run.py](workspaces/mxn_f_rand_csr_v2/run.py)

9. Workspace template
Add a template directory with the following minimal pieces to accelerate new workflows:
- run.py patterned after [workspaces/mxn_counts_v1/run.py](workspaces/mxn_counts_v1/run.py) with standardized logging and summary writing
- config.json defaults for inputs and outputs
- outputs directory created on first run

10. Testing strategy
Unit tests
- Keep and extend existing tests such as [tests/test_car_roundtrip.py](tests/test_car_roundtrip.py), [tests/test_mdf_import.py](tests/test_mdf_import.py), [tests/test_mdf_roundtrip.py](tests/test_mdf_roundtrip.py)
- Add tests for centralized helpers introduced in usm ops

Integration tests
- Add end to end tests that run selected workspaces with fixed seeds, capturing summary manifests and verifying key counts and determinism

Golden files policy
- Prefer assertions on structured outputs JSON rather than byte identical files except where header preservation is the guarantee under test

11. Performance and scale
- Maintain current O of N characteristics noted in [docs/PERFORMANCE.md](docs/PERFORMANCE.md)
- Keep Parquet as preferred bundle format with CSV fallback
- For very large jobs prefer transform operations before replicate operations to reduce cost

12. Minimal CLI launcher optional
- For batch runs add a tiny wrapper that dispatches to run dot py in a workspace with a config path
- Avoid introducing a pipeline DSL or executor

13. Migration guide from MolSAIC v1 to V2
- Replace YAML pipelines with direct Python scripts under workspaces
- Replace plugin references with explicit imports from usm modules
- For any third party command line tools call them from run dot py with explicit arguments and staged paths inside outputs
- Use the bundle folder for interchange when handing off to non Python tools

Mapping examples
- fileio.parse mdf becomes load from [usm/io/mdf.py](usm/io/mdf.py) inside the script
- fileio.parse car becomes load from [usm/io/car.py](usm/io/car.py) inside the script
- external tool invocations become subprocess calls from run dot py with explicit inputs and outputs

14. Deliverables and milestones
M1 Baseline
- Adopt this repository’s USM modules and docs as baseline
- Freeze plugin and YAML paths and remove references in docs
- Add workspace template and centralize surface split and counts helpers

M2 Grafting and substitution consolidation
- Move OH pairing and MDF connections helpers into usm ops
- Stabilize placement manifest structure used by grafting results
- Add integration tests for graft and substitution workspaces

M3 Visualization and CSR selection
- Harden CSR selection utilities and sampling outputs
- Harmonize plotting hooks behind optional dependencies
- Add integration test for CSR selection with fixed seed

M4 Documentation and quickstart
- Add V2 quickstart page under docs
- Document workspace template and recommended logging and summary content

15. Acceptance criteria
- IO round trips for CAR and MDF meet the preservation and numeric tolerance guarantees described in [docs/DATA_MODEL.md](docs/DATA_MODEL.md) and [docs/DESIGN.md](docs/DESIGN.md)
- Workspaces run from a clean checkout with Python and produce the documented outputs under outputs
- Deterministic behavior verified by tests with fixed seeds
- No dependency on plugin registries, YAML configs, or executor frameworks remains

16. Risks and mitigations
- Drift across duplicate helper implementations mitigated by centralization into usm ops
- Long running pipelines mitigated by explicit logging and summary metrics to aid debugging
- Optional plotting dependency handled gracefully with feature detection

17. Appendix A workspace patterns
Grafting pattern
- Compose host CAR and MDF and compute surface split threshold by oxygen z
- Build candidate sites by pairing O and H using distance or bonds
- Select sites half per surface or by explicit counts
- Prepare guest by composing CAR and MDF and dropping selected atoms such as phenolic hydrogen
- Place guest per site using torsion and tilt scanning with clash checks
- Optionally resize orthorhombic cell and wrap to avoid visual wrap around
- Save CAR and MDF with normalized connections and write placements manifest and summary
Reference: [workspaces/mxn_dop_graft_v1/run.py](workspaces/mxn_dop_graft_v1/run.py)

Substitution pattern
- Identify O and H pairs and select a subset per surface by seed
- Replace O with F and remove H and update names and types
- Preserve MDF connections raw tokens where needed or regenerate normalized connections
- Renumber aids to keep dense ids
- Save CAR and MDF and write selection manifest and summary
References: [workspaces/mxn_f_sub_v2/run.py](workspaces/mxn_f_sub_v2/run.py), [workspaces/mxn_f_sub_v1/run.py](workspaces/mxn_f_sub_v1/run.py)

CSR like selection pattern
- Partition candidate sites by surface using split threshold
- Run multiple random draw trials and score by Ripley L error
- Keep the best selection and optionally save a gallery of sampled draws
- Render 2D plots when matplotlib is available
Reference: [workspaces/mxn_f_rand_csr_v2/run.py](workspaces/mxn_f_rand_csr_v2/run.py)

Orientation analysis pattern
- Select basal sublayer by element and rounded z
- Compute nearest neighbor in plane distances and vectors
- Measure absolute angle to plus x and classify zigzag versus armchair
- Save JSON and TXT and optional plot
Reference: [workspaces/mxn_orientation_v1/run.py](workspaces/mxn_orientation_v1/run.py)

Z bins and counts pattern
- For requested atom types split by z into top and bottom using threshold method
- Produce JSON CSV and TXT summaries and aggregate totals
References: [workspaces/mxn_zbins_v1/run.py](workspaces/mxn_zbins_v1/run.py), [workspaces/mxn_counts_v1/run.py](workspaces/mxn_counts_v1/run.py)

18. Appendix B developer checklist
- Create workspace template and document usage
- Extract and centralize helpers noted in section 8
- Verify tests pass and extend with integration tests per section 10
- Update docs with quickstart and workspace conventions
- Remove or archive plugin and YAML related materials if present in other repos

19. Appendix C diagrams
Architecture
flowchart LR
  Lib[USM Library] --> IO[IO and Bundle]
  Lib --> Ops[Ops Selection Transforms Cell]
  Ops --> WS[Workspaces]
  IO --> WS
  WS --> Out[Outputs and Manifests]
  style Out fill:#eaffea,stroke:#93c893