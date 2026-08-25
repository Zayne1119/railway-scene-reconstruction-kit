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

