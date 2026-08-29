# Blender interchange preparation

The script keeps asset objects separate, applies transforms, removes near-duplicate vertices within each object, removes degenerate geometry, recalculates face normals and exports an interchange file.

```powershell
blender --background --python blender/prepare_interchange.py -- `
  --input projects/my-project/workspace/exports/scene.glb `
  --output projects/my-project/workspace/exports/scene_ue.glb `
  --merge-distance 0.0001 `
  --triangulate
```

Supported input/output suffixes are GLB/GLTF, FBX and OBJ. The script deliberately does **not** merge asset objects because that would destroy stable identity and picking. It also cannot safely decide whether overlapping faces in different objects are redundant. Review the generated JSON report and run six-view visual acceptance with back-face culling before delivery.

## Integrate a reviewed TrackGraph into the complete scene

`integrate_track_graph.py` replaces a named legacy track layer with the continuous
TrackGraph mesh while preserving the rest of the scene and the asset registry. It
also repairs negative rail winding, tapers the new rail endpoints to the retained
next segment, removes truncated interface sleepers and aligns the sleeper phase.

Do not pass an output filename extension: `--output-prefix` is a literal prefix,
so version numbers such as `v9.1` are preserved.

```powershell
blender --background --python blender/integrate_track_graph.py -- `
  --source-glb projects/my-project/workspace/exports/scene_v9.0.glb `
  --track-obj projects/my-project/workspace/exports/track_graph.obj `
  --track-origin projects/my-project/workspace/exports/track_graph_origin.json `
  --full-origin projects/my-project/workspace/exports/full_scene_origin.json `
  --track-registry projects/my-project/workspace/exports/track_graph_registry.json `
  --replace-layer detail_s000_200m `
  --output-prefix projects/my-project/workspace/exports/scene_v9.1_trackgraph `
  --report projects/my-project/workspace/reports/trackgraph-integration.json
```

The command writes GLB, FBX, OBJ, MTL and a merged registry. Treat the generated
integration report as an automatic gate, not as a substitute for visual review.

## Fail-closed rail interface audit

Run the interface audit after integration. The interface chainage is expressed in
the imported Blender scene coordinate convention used by the complete-scene GLB.

```powershell
blender --background --python blender/audit_track_interface.py -- `
  --source projects/my-project/workspace/exports/scene_v9.1_trackgraph.glb `
  --output projects/my-project/workspace/reports/track-interface.json `
  --interface-chainage-m -100 `
  --new-prefix TRACK- `
  --retained-prefix s200_250m.Track
```

The gate checks every new/retained rail pair for longitudinal gap, lateral offset
and rail-top offset. A non-passing pair makes the Blender process fail.

## Audit and close scene-component relationships

`audit_scene_relationships.py` checks the complete model using imported mesh
geometry, BVH ray casts and configured asset selectors. It does not accept
registry bounds or relationship claims as proof. The default template covers
column-to-platform, column-to-cross-beam, beam-to-roof, stair-to-platform,
stair-to-guard and vertical-circulation enclosure contacts.

```powershell
blender --background --python blender/audit_scene_relationships.py -- `
  --source projects/my-project/workspace/exports/scene.glb `
  --registry projects/my-project/workspace/exports/scene_registry.json `
  --config configs/templates/scene_relationships.default.json `
  --output projects/my-project/workspace/reports/scene-relationships.json
```

The report is hash-bound to the source GLB, registry and configuration. Any
failed relationship returns a non-zero process status. Review the failures
before repair; a passing global count must not hide an individual unsupported
component.

The companion repair script is deliberately narrow. It only closes a positive
vertical gap between a canopy column and its platform or paired cross-beam by
moving the relevant boundary vertices. Unsupported failure kinds are rejected
instead of being guessed.

```powershell
blender --background --python blender/repair_scene_relationships.py -- `
  --source-glb projects/my-project/workspace/exports/scene.glb `
  --registry projects/my-project/workspace/exports/scene_registry.json `
  --audit projects/my-project/workspace/reports/scene-relationships.json `
  --output-prefix projects/my-project/workspace/exports/scene_relationship_closed `
  --report projects/my-project/workspace/reports/scene-relationship-repair.json
```

Always rerun the relationship audit, Mesh audit and fixed-view visual acceptance
against the repaired GLB. Keep the pre-repair report as immutable failure
evidence; do not overwrite it with the passing report.
