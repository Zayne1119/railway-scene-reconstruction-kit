# 01｜数据生命周期与 Source of Truth

## 总原则

本系统不是把“最新生成的模型文件”当真相，而是逐步提升证据资格：`raw → candidate → reviewed/accepted → graph pass → mesh audited → registry accepted/frozen → hash-bound release`。任何视觉上更漂亮的几何，如果缺少身份、证据、状态和哈希绑定，都不应覆盖更高资格的数据。

## 十五类数据的生命周期

| 数据 | SOURCE | TRANSFORM | INTERMEDIATE | VALIDATION | CONSUMER | FINAL ARTIFACT |
|---|---|---|---|---|---|---|
| LAS/LAZ | `project.inputs.point_cloud` | `audit_project`、`crop_segments` | 分段 LAZ | 格式/范围/点数/存在性 | rail/linear/projection | 不直接发布；作为 evidence source |
| camera.csv / pose | `project.inputs.camera_csv` | `load_camera_rows`、`camera_trajectory` | 含 `distance_m` 的行 | 必填列、顺序、重复、旋转完整性 | 分段、RouteSampler、投影 | hash 绑定到 TrackGraph/manifest |
| panorama/image | `panorama_root` + camera `file` | `_find_photo`、投影 overlay | RGB 假设/人工解释 | 文件引用、pose、跨相机视觉验收 | reviewed layout / registry source | `panorama` evidence reference |
| local coordinate | 世界 XY + corridor camera | `fit_corridor_frame*().project` | longitudinal/cross | 有限值、方向对齐、上下文 | detectors、candidate report | 只在报告/参数中；不是绝对坐标真相 |
| chainage/mileage | 相机 XYZ 序列 | `camera_trajectory` 累积三维欧氏距离 | `distance_m`、segment start/end | 单调、区间正长度 | segment、RouteSampler、graph | graph canonical axis；不是法定测量里程 |
| geometric candidate | segment points | `detect_rail_candidates` / `detect_linear_candidates` | candidate JSON/LAZ/PNG | detector 阈值与诊断 | 人工复核 | 不可直接权威发布 |
| TrackGraph | accepted candidate reports + camera | `build_track_graph` | observations/nodes/edges/tracks/seams | schema、绑定、`audit_track_graph` | graph mesh | audit `pass` 的 hash-bound graph |
| RailAssetGraph | **代码中无同名对象** | registry imports/builders | `assets + relations` | `validate_registry_value` | QA/release/Web | frozen asset registry；概念上承担资产图角色 |
| evidence | 点云/照片/规则/人工 | detector、layout、builder赋值 | asset/edge `evidence_level` + sources | enum、人工要求、graph audit | confidence/QA/release | frozen registry 中不可丢失 |
| confidence | detector/人工/默认配置 | 赋给 asset/candidate | 0..1 | registry schema 只校范围 | UI/筛选/风险判断 | registry 字段；不是误差概率的自动证明 |
| geometry parameters | config + fitted values + review | graph interpolation/layout geometry | gauge、offset、z、profile、box/panel/tube | 范围、拓扑、接口 audit | mesh generators | registry `parameters` + mesh |
| Mesh | graph/layout parameters | `sweep_mesh`、`oriented_box`、reviewed geometry | OBJ/MTL；后续 DCC GLB/FBX | `audit_obj`、接口/视觉检查 | Blender/UE/Web | hash-bound model artifact |
| QA result | source reports/model/registry | audit/quality/gate helpers | JSON checks | 显式 `pass`/`passed` 才算通过 | gate/owner | immutable gate supporting docs |
| authoritative registry | working registry | 人工接受 + `freeze_release_registry` | release registry | schema、全部 accepted、集合哈希 | gate/Web config | frozen JSON |
| release manifest | model + registry + checks | `evaluate_quality_gate` / `build_web_acceptance_config` | artifact manifest/gate result/config | SHA-256、前门、审批、豁免 | Web/交接 | gate directory + Web config |

## 关键变换细节

### 世界坐标到局部走廊坐标

`geometry.CorridorFrame.project()` 计算：

\[
s=(p-o)\cdot a,\qquad c=(p-o)\cdot n
\]

其中 `a` 为沿线单位向量，`n` 为横向单位向量。`fit_segment_corridor_frame()` 使用该段前后相机上下文，避免仅由短段拟合导致方向翻转。局部坐标服务检测，不应覆盖原始世界坐标。

### 相机路径到 chainage

`camera.camera_trajectory()` 对相邻相机三维位置距离累积：

\[
d_i=d_{i-1}+\|p_i-p_{i-1}\|_2
\]

这是“相机轨迹累计距离”，不是已校核铁路里程。`track_graph.RouteSampler` 以它为 canonical axis。若第二场景有正式里程桩/测量线，当前代码没有自动替换机制，属于迁移时需设计的扩展点。

## Source of Truth 分层

| 层级 | 可作为何种真相 | 不能声称什么 |
|---|---|---|
| 原始 LAS/LAZ、camera、panorama | 原始证据真相；不可被派生物静默改写 | 不能单独证明资产类别或完整背面 |
| candidate reports | 算法假设与诊断的真相 | 不是接受几何、不是资产清单 |
| reviewed/owner-override reports | 人工决策真相；来源 hash-bound | owner override 不是独立复核 |
| audit-pass TrackGraph | 轨道身份、拓扑、链程、证据区间的几何控制真相 | 不代表站台/雨棚/立面真相 |
| working asset registry | 当前资产状态账本 | 含 candidate/reviewed 时不可发布 |
| frozen accepted registry | 某 release 的资产身份/证据真相 | 不自动证明模型二进制未改变 |
| gate artifact manifest + Web config | 某 release 的精确交付绑定 | 不能替代现场绝对精度验证 |

## 覆盖与禁止覆盖规则

### 可以覆盖

- `import_registry_records(..., replace_existing_ids=True)` 可显式替换同 ID 资产；默认拒绝冲突。
- `build_track_graph_mesh(..., overwrite=True)` 可替换自己生成的 TrackGraph 资产和关系；冲突 ID 被显式清理再重建。
- review/owner override **不修改**原 candidate，而是在新目录写 hash-bound copy。
- 新 release 应使用新 `release_id` 和新 gate 目录，而不是原地修改旧证据。

### 绝对不能静默覆盖

- 不得用 candidate 覆盖 accepted/reviewed 的语义结论。
- 不得把 `rule_inferred` 区间聚合成全 `observed`；`_track_evidence_intervals()` 专门防止这一点。
- 不得修改已经审计后的 graph 而继续复用旧 audit；mesh builder 比较 graph SHA-256。
- 不得修改 camera/config 后继续复用旧 graph；`validate_track_graph_bindings()` 拒绝 stale binding。
- 不得在 frozen release registry 中保留 candidate/reviewed/rejected/superseded 资产。
- 不得改变模型/registry 后复用旧 Web acceptance config；浏览器重新计算哈希和 asset-set hash。
- 不得在无 CRS/垂直基准/独立检查点时把内部点模拟合误写成绝对测量精度。

## 两类 manifest 不要混淆

1. `manifest.py::write_run_manifest()` 写 `railway.run-manifest.v1` 风格的日常命令运行记录，包含配置快照、git commit、输入/输出 hash 与 runtime。
2. `resources/run-manifest-v2.schema.json` 定义 benchmark 的冻结实验运行契约，要求 protocol/dataset/split/method/variant、随机种子、ground truth、human review 等。

两者名称相似但用途和 schema 不同。代码尚未把所有日常运行自动升级为 benchmark v2 manifest。

