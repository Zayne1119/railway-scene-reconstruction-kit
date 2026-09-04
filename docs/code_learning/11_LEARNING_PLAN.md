# 11｜成为项目 Owner 的 7 天学习路线

目标不是背代码，而是能独立处理第二场景：知道入口、证据资格、失败回退、阈值标定和发布责任。每天 2–3 小时。

## 知识优先级

### MUST KNOW

- CLI 各阶段依赖和退出行为。
- project/camera/segment/TrackGraph/registry/gate 的数据合同。
- candidate、reviewed/accepted、observed/photo/rule/unsupported 的区别。
- TrackGraph 为什么防断裂、重叠、错连。
- 哪些 threshold 是项目默认而非行业标准。
- graph/mesh/registry/release 的 hash binding。
- 哪类失败回 detector、review、graph、mesh、DCC 或 release 修。
- 没有独立控制点时的精度表达边界。

### SHOULD KNOW

- CorridorFrame、RouteSampler、rail grid/prominence、robust fit、pairing的直觉。
- Hungarian matching、seam reconciliation 和推断区间。
- OBJ sweep、轨枕 phase、轨道装配接口。
- projection convention search 和 RGB score 限制。
- spatial voxel holdout、experiment lock 和 threshold leakage。
- Blender/UE 的 axis/unit/origin/normals/back-face/Z-fighting。

### CAN LOOK UP

- argparse boilerplate、JSON Schema 详细语法。
- NumPy/SciPy API 参数细节。
- LAS point format 全字段。
- GLB binary chunk、Three.js 具体 API。
- Blender `bpy` 的低层调用。
- 每个 benchmark report 的所有次级指标。

## Day 1｜入口、顺序和状态（2.5 h）

1. 读 `cli.py` 的 parser、`_project_command()`、主 dispatch（60 min）。
2. 读 `config.py`、`project.template.json`、`project.schema.json`（45 min）。
3. 做练习 1（45 min）。

验收：不看文档能从 raw input 说到 release；能指出没有总 orchestrator 的风险。

## Day 2｜坐标、相机和分段（2 h）

1. 读 `camera.py`、`geometry.py`、`segments.py`（75 min）。
2. 手算 4 个 camera 点的 `distance_m`，再读 adaptive zone test（30 min）。
3. 做练习 3 的预测部分（15 min）。

验收：能解释正式里程与 camera chainage 的差异；遇到弯道偏移知道先查 frame/RouteSampler。

## Day 3｜Rail perception 与 review（3 h）

1. 按函数索引读 `rail_candidates.py`，重点 z mask、grid、peak、robust fit、pair support（90 min）。
2. 读默认 rail config，写每个关键阈值“过高/过低”的症状（30 min）。
3. 读 `rail_review.py`、`rail_pair_curation.py`（30 min）。
4. 做练习 4 或 5（30 min）。

验收：能说明为什么候选不能进 mesh；能诚实选择 independent review、owner override 或 curation。

## Day 4｜TrackGraph 与连续网格（3 h）

1. 先读 `track-graph.schema.json`，再读 `RouteSampler`、observation/matching/build/audit（90 min）。
2. 读 `track_graph_mesh.py` 的 centerline、evidence intervals、sleeper phase、registry mapping（60 min）。
3. 做练习 7（30 min）。

验收：给出断轨/重叠截图时，能先查 graph checks，不直接手改 mesh。

## Day 5｜资产、照片与非轨道能力边界（2.5 h）

1. 读 `asset-registry.schema.json`、`registry.py`、`reviewed_scene.py`（60 min）。
2. 读 `linear_candidates.py`、`projection.py`（45 min）。
3. 对照 `examples/minimal_project/reviewed_scene.example.json` 做一条 sign/facade trace（30 min）。
4. 写下当前没有实现的自动能力（15 min）。

验收：不会再把 vertical candidate 叫“已识别接触网柱”；不会把 RGB 最优叫“照片已配准正确”。

## Day 6｜QA、DCC 与发布（3 h）

1. 读 `mesh_audit.py`、`qa.py`、`quality_gate.py`、`release.py`（90 min）。
2. 浏览 `blender/audit_*` 和 `prepare_interchange.py`，只理解输入/输出/失败状态（45 min）。
3. 做练习 8 或 9（30 min）。
4. 快答口试 C41–C60（15 min）。

验收：能区分 audit pass 与视觉 pass；能解释 frozen registry、gate、Web config 三层绑定。

## Day 7｜第二场景桌面演练（3 h）

1. 从头讲一遍 `10_SECOND_SCENE_PLAYBOOK.md`，不看答案（30 min）。
2. 做练习 12 的纸面/空项目演练（90 min）。
3. 回答迁移题 71–80，再核答案（30 min）。
4. 建立个人一页“故障→先查文件/报告→允许修复点”速查表（30 min）。

验收：面对新数据能在一小时内完成资格审查；能明确说“可复用/需重标定/需人工/UNKNOWN”；不会为了结果好看绕过 fail-closed。

## 一个月后的 owner 标准

- 能独立跑一个 50–100 m synthetic/new prototype。
- 能在不看 AI 建议时判断 rail 断裂属于 candidate、graph、mesh 还是 DCC 层。
- 能审查同事提交的 config diff，指出哪些更改使旧 graph/review 失效。
- 能拒绝没有 evidence/registry/hash 的“final model”。
- 能把外部算法接到 candidate contract 并做冻结 holdout 比较，而不污染生产 source of truth。

