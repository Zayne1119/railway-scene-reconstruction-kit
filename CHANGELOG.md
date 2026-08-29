# Changelog

## Unreleased

## 0.2.0 - 2026-08-29

- Added trajectory-relative rail-height windows and adaptive 5/10/50 m segmentation for long corridors with changing camera elevation and complex topology.
- Added hash-bound pair-level rail curation with explicit owner-override provenance and low-confidence rule-inferred gap observations.
- Split observed and inferred rail intervals into separate mesh objects, materials and registry assets; inferred rails render orange and can no longer inherit an observed label from the aggregate track.
- Added opt-in, length-bounded turnout display connectors that require a confirmed turnout boundary and remain `rule_inferred` with conservative confidence.
- Added a standalone generated-asset sidecar for every TrackGraph Mesh build, including builds that intentionally skip the project registry.
- Added geometry-derived scene relationship auditing for canopy columns, beams,
  roofs, platforms, stairs, guards and vertical-circulation enclosures.
- Added hash-bound, fail-closed relationship reports and a narrow column-boundary
  repair path that refuses unsupported failure types.
- Added post-repair requirements for Mesh audit, fixed-view visual review and
  retained TrackGraph interface regression.
- Added complete-scene TrackGraph integration with literal versioned output prefixes, registry preservation and GLB/FBX/OBJ delivery.
- Added rail winding repair, a 2 m endpoint taper, a 5 mm non-coplanar interface cap and retained sleeper-phase alignment.
- Added a fail-closed six-rail interface audit for longitudinal gap, lateral offset and rail-top offset.
- Added hash-bound project-owner overrides when independent rail review is explicitly waived.
- Switched continuous TrackGraph mesh control points to fitted rail world coordinates.
- Added centerline step, deflection and backtracking gates plus reusable six-view Blender review renders.

- Replaced occupancy-only rail detection with longitudinal height-prominence extraction, context-stabilized corridor frames and endpoint cross/elevation fits.
- Distinguished 1.435 m working-edge gauge from rail center spacing and made nominal projection corrections explicit and auditable.
- Added resolution-aware TrackGraph seam defaults for 5 cm derived clouds and retained exact endpoint residuals in the audit.
- Added hash-bound `track-review-create` / `track-review-apply` independent review packages that never mutate automatic candidate reports.
- Added global TrackGraph v1 with camera-chainage identities, explicit observation intervals, fail-closed seam/topology/coverage/inference/duplicate checks, CLI commands and historical rail-failure regressions.
- Added a TrackGraph-only full-corridor OBJ generator with continuous rails, globally phased non-duplicated sleepers, registered visible nodes, input/config hash binding and automatic Mesh audit.
- Added the P0 Quality Gate v2 runner with strict status propagation, immutable evidence directories, previous-Gate hash chaining, artifact hashing, approvals and explicit artifact-bound waivers.
- Changed baseline parametric track assets from automatically accepted to candidate status and added release-readiness blocking for all non-accepted registry assets.
- Added accepted-only release registry freezing and hash-bound Web release config generation.
- Added fail-closed Web acceptance mode for release/model/registry/asset-set hashes, registry state and Mesh-node mappings.
- Added production run-manifest output hashes and regression tests for local-pass/global-fail contradictions, candidate leakage, waiver expiry/binding and evidence tampering.
- Added Paper Benchmark v1 schemas, public templates and Chinese operating guidance.
- Added `benchmark-init`, `benchmark-validate` and `benchmark-freeze` CLI commands.
- Added a unified benchmark evaluator for instances, topology, confidence, bidirectional distance summaries and human-review effort.
- Added double-blind vertical-candidate annotation package generation with copied evidence sheets and label-leakage controls.
- Added pre-ground-truth locking of legacy HITL vertical predictions with task-set equality and SHA-256 provenance checks.
- Added completed-review agreement analysis with Cohen's kappa, consensus seeding and an explicit adjudication queue.
- Added Run Manifest v2 input/config hashing and private Site A freeze support.
- Tightened absolute-precision readiness to require independent check points in addition to CRS and vertical datum.

## 0.1.0 - 2026-08-25

- Extracted a reusable project/configuration model from the 200 m railway reconstruction case.
- Added project initialization, input audit, camera-based segment planning, LAZ cropping, registry validation, QA and repository safety checks.
- Added Chinese technical documentation, synthetic templates, CI and optional Web/Blender handoff scaffolding.
