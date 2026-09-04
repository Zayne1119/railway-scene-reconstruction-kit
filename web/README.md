# Railway evidence / hypothesis QA viewer

This local Three.js dashboard loads one integrated GLB scene plus the pipeline's QA reports. OBJ/MTL remains available as a fallback, and production geometry is not copied into the source repository.

Features:

- evidence / bounded-hypothesis comparison without loading two copies of the model;
- a Meshopt-compressed Web LOD for fast review, while the full GLB remains unchanged;
- discipline layers for track, catenary, conductor and station geometry;
- automatic issue lists for conductor seams, withheld catenary candidates and inferred track gaps;
- asset ID/type search, click selection and fixed QA views;
- report-driven metrics and risk markers.

## Run locally

From the web directory, copy `public/project.example.json` to `public/project.json`, then run `npm ci` and `npm run dev`.

Open `http://localhost:3010`. The dev server listens on all interfaces for trusted LAN review. Never expose the project or its models publicly without the project owner's explicit approval.

An alternate local config can be selected without replacing the default project:

```text
http://localhost:3010/?project=project.track-boundary.json
```

Only a plain JSON filename is accepted. Paths and external URLs are rejected.

When Blender is unavailable, a named flat-material OBJ can be converted without external runtime dependencies:

```powershell
python scripts/export_obj_to_web_glb.py --input scene.obj --output scene.web.glb
```

The converter preserves OBJ object names as GLB nodes and validates node and triangle counts before writing its conversion report.

`public/project.json` is intentionally ignored by Git because it contains local paths. Use forward-slash Vite `/@fs/` URLs and add the data root to `vite.config.js` under `server.fs.allow`.

## Scene contract

- Integrated OBJ object names may start with `TRACK--`, `CATENARY--`, `CONDUCTOR--` or `STATION--`.
- Candidate scenes may also retain pipeline names such as `TRACKGRAPH--TRACK-`, `SEG*`, `SUPPLEMENTAL-*` and `ADJACENT-*`; the viewer classifies these without renaming the assets.
- Bounded track hypotheses must include `INFERRED` in their object name.
- Registry assets are indexed by geometry node, asset ID and the configured namespace-prefixed ID.
- QA reports remain the source of truth; a risk marker is not a released asset.
- This viewer is for local candidate review. Formal release acceptance and content-hash validation remain separate pipeline steps.
