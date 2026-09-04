# 03｜五类资产端到端追踪

## A. Rail：当前最完整的权威链路

```text
segment LAZ
→ detect_rail_candidates()
→ candidate JSON/LAZ/PNG
→ rail review / owner override / pair curation
→ build_track_graph()
→ audit_track_graph(pass)
→ build_track_graph_mesh()
→ rail nodes + sleepers + bed + registry relations
→ mesh audit / registry QA / gate
→ OBJ，后续 Blender 导出 GLB/FBX
```

| 步骤 | 实际源码 | 输入 | 输出/side effect | 失败模式 |
|---|---|---|---|---|
| 检测 | `algorithms/rail_candidates.py::detect_rail_candidates` | segment LAZ、frame、config | candidate LAZ、diagnostic PNG、report | 空 z mask、无 peaks、参数非法；结果仍需人工 |
| 接受 | `rail_review.py::*` / `rail_pair_curation.py::curate_rail_pairs` | report + hash + decision | 新 accepted copy | hash 变化、reviewer/原因缺失、reject |
| 全局状态 | `track_graph.py::build_track_graph` | accepted reports、camera、settings | graph/audit | ID 交换、接缝、重复、source review 失败 |
| 参数 | `_track_centerline`、`_interpolated_track_value` | observations | chainage + XYZ centerline | 非有限、重复切线、绑定陈旧 |
| 几何 | `track_graph_mesh.py::build_track_graph_mesh` | pass graph | swept rails、sleepers、bed | 非 observed 边策略、接触不合、中心线偏折 |
| registry | `_asset`、`_replace_registry_assets` | component ids/evidence | asset + contains relations | 重复 ID/悬空 relation |
| QA/export | `audit_obj`、`quality_report`、Blender scripts | OBJ/registry | audit、GLB/FBX | 退化/重复对象、映射不全、视觉闪面 |

## B. Platform：当前是审核布局驱动

可确认路径是：

```text
raw evidence（点云/照片）
→ 人工审核并形成 railway.reviewed-scene-layout.v1 JSON
→ geometry.kind（box / extruded polygon / panel 等）
→ build_reviewed_scene()
→ OBJ + registry asset(type=platform)
→ audit_obj + 关系/DCC QA
→ export
```

- 示例契约：`examples/minimal_project/reviewed_scene.example.json`。
- 生成器：`algorithms/reviewed_scene.py::_mesh_for_geometry()` 与 `build_reviewed_scene()`。
- 当前仓库中**没有可确认的自动站台点云分割/分段高程拟合生产函数**。`project.quality.platform_fit_p90_max_m` 是配置字段，但并不等于算法已实现。
- 与柱/楼梯的接口可由 `relationships.py::contact_status()` 及 `blender/audit_scene_relationships.py` 一类外部 DCC 审计检查；这不是核心包自动拟合器。

## C. Canopy：几何构件存在，自动屋面理解不完整

```text
vertical candidates / photo evidence
→ 人工分类 canopy_column
→ reviewed layout 中 column/beam/roof
→ _box/_beam/_extruded_polygon_xy/_panel
→ build_reviewed_scene()
→ registry + OBJ
→ Blender interface/normal/overlap QA
```

- 候选：`algorithms/linear_candidates.py::_vertical_candidates()` 只输出 `unclassified_requires_photo_evidence`。
- 语义任务：`annotation_tasks.py` 支持 `canopy_column` 标签和双盲包，但没有自动分类器。
- 生成：`reviewed_scene.py`；示例含 `canopy_column`、`canopy_beam`、`canopy_roof`。
- 当前仓库中**没有可确认的多平面屋面分割、真实轮廓恢复、柱网优化生产算法**。历史工程可能做过，但不能据此声称当前工具包自动具备。

## D. Catenary：线/杆候选与规则资产，不是完整专业重建

```text
segment LAZ
→ detect_linear_candidates()
→ vertical/cable candidates
→ 照片/人工专业分类
→ reviewed layout（mast/beam/tube/cable）
→ build_reviewed_scene()
→ registry evidence + mesh
→ continuity/interface/visual QA
```

- `_vertical_candidates()` 使用 XY 栅格的 Z 跨度、点数、连通域和最大 footprint。
- `_cable_candidates()` 在 cross–Z 栅格统计纵向覆盖率并输出细线候选。
- `resources/rules.default.json::catenary` 声明连续导线和推断跨段证据规则。
- **缺口**：支柱型号分类、腕臂/定位器/绝缘子装配、悬链线物理拟合和遮挡恢复，在当前生产包中为 UNKNOWN/未实现。不能把一个 `tube` 或 beam layout 当完整接触网系统。

## E. Photo-interpreted sign/facade/window

```text
panorama + camera pose + RGB point cloud
→ calibrate_projection()
→ RGB MAE 最小的姿态/轴排列假设 + overlay
→ 多相机人工视觉接受
→ reviewed layout（panel/box/extruded polygon）
→ evidence_level=photo_interpreted + panorama source
→ build_reviewed_scene()
→ registry/mesh/gate
```

- 投影：`projection.py::signed_permutations()`、`project_equirectangular()`、`calibrate_projection()`。
- 算法会搜索 Euler order、pose direction 和 48 个 signed axis permutations，以点颜色与全景像素的 median RGB MAE 排序。
- 输出状态明确是 `automatic_rgb_hypothesis_visual_acceptance_required`；最低色差不是 pose 正确性的证明。
- 未见当前代码中 Grounding DINO/SAM2 自动检测 sign/window/facade；这些若曾在历史工程使用，也不属于当前依赖或权威路径。

## 某资产错位 30 cm 时，按这个顺序查五处

1. **原始坐标/pose**：检查 `input_audit.json`、camera 单位/轴/时序、EPSG/垂直基准；若所有资产同向偏移，先怀疑这里。
2. **局部 frame/chainage**：检查 candidate report 的 `frame`、frame camera range、segment bounds 和 `RouteSampler`；若只在弯线或段界偏，重点看方向/上下文。
3. **候选拟合/人工绑定**：对 rail 看 `rail_lines` 的 cross/z fit、review hash、curation 是否选错 pair；对竖直/照片资产看审核 layout 的 center/size/source。
4. **graph/参数插值**：检查 TrackGraph observation 的 world start/mid/end、lateral offset、seam reconciliation 和 evidence interval；不要先手改 mesh。
5. **导出原点/坐标转换**：检查 `model_origin.json`、OBJ 本地坐标、Blender/UE 的轴与单位、GLB node transform；若 registry 参数正确但画面错，才到这一层。

额外判断：若只有 mesh 外观偏而 graph/registry 参数正确，查 generator/DCC；若 candidate 就偏，后续平滑只会把错误变得连续。

