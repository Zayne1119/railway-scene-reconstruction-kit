# 证据门控后的站台候选网格

`build-platform-mesh` 只处理已经通过 `railway.platform-mesh-gate.v1` 的站台组件。命令会在生成几何前强制检查：项目段、站台组件、未处置缺口数量和 Gate 状态必须一致；任何未处置缺口都会让构建失败。

```powershell
railway-recon build-platform-mesh `
  --project <project.json> `
  --segment <segment-id> `
  --mesh-gate <platform_mesh_gate.json>
```

## 几何策略

- 相邻 5 m 拟合段不会分别导出独立薄片。
- 所有拟合段先转换成共享纵向截面；段界处采用相邻拟合结果的共同截面，因此不会产生段间裂缝或重叠面。
- 点云支持的顶面导出为 `platform_surface / observed` 资产。
- 侧壁、底面和厚度仅用于可视化，单独导出为 `platform_volume / rule_inferred` 资产。
- 被 Mesh Gate 判定为遮挡的点云空白不会切入顶面。
- 点云证据不足的另一侧站台不会镜像生成。

## 输出

- `workspace/exports/<segment>/platform_candidate/platform_candidate.obj`
- `platform_candidate.mtl`
- `model_origin.json`
- `<segment>_platform_mesh_build.json`
- `<segment>_platform_mesh_audit.json`
- `<segment>_platform_mesh.png`
- 两条候选资产及其 `supported_by` 关系

OBJ 仍属于候选几何。进入完整场景、GLB 或 UE 之前，需要继续检查与轨道、站台墙、雨棚柱、楼梯和电梯的空间接口。
