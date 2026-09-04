# Railway Reconstruction Kit：代码学习课程

本目录以当前工作树中的**实际代码**为准，而不是以 README、历史汇报或文件名推测实现。当前工作树包含大量尚未提交的新模块；因此学习时应记录当前文件哈希/提交状态，不能只引用仓库唯一提交 `9d541b7`。

## 十行架构摘要

1. `railway-recon` 由 `railway_recon.cli:main` 提供 50 多个显式子命令，没有一个自动执行全流程的总编排器。
2. `project.json`、JSON Schema 和项目内相对路径组成运行契约；`ProjectConfig.workspace_path()` 阻止输出逃逸项目目录。
3. LAS/LAZ 与 `camera.csv` 先经输入审计，再按相机累计路径里程规划、裁剪区段。
4. 钢轨检测器输出局部几何候选与诊断图，明确标记为候选，不能直接当权威模型。
5. 候选经过独立复核、项目负责人显式豁免或成对整理后，才可进入 TrackGraph。
6. TrackGraph 用全局轨道身份、节点、边、观测、接缝和证据等级解决分段断裂与重叠问题。
7. 只有哈希绑定且审计为 `pass` 的 TrackGraph 才可生成连续钢轨/轨枕/道床网格。
8. 站台、雨棚、接触网和照片解释资产当前主要从审核布局生成；自动语义重建在本仓库中并不完整。
9. 资产注册表区分 `observed`、`photo_interpreted`、`rule_inferred`、`unsupported`，候选资产会阻断发布。
10. 发布配置与质量门用 SHA-256 绑定模型、注册表、证据和前一门结果；Web 验收模式再次核验这些绑定。

## 推荐阅读顺序

1. [`00_SYSTEM_MAP.md`](00_SYSTEM_MAP.md)：先看真实入口与阶段边界。
2. [`01_DATA_LIFECYCLE.md`](01_DATA_LIFECYCLE.md)：理解各类数据何时只是候选、何时才可发布。
3. [`02_CORE_ABSTRACTIONS.md`](02_CORE_ABSTRACTIONS.md)：理解 TrackGraph、证据模型与 fail-closed。
4. [`03_ASSET_TRACES.md`](03_ASSET_TRACES.md)：沿五类资产追到最终导出。
5. [`04_QA_AND_GATES.md`](04_QA_AND_GATES.md)：掌握什么会阻断权威交付。
6. [`05_ALGORITHMS.md`](05_ALGORITHMS.md)：再读真正影响能力的算法。
7. [`06_DEPENDENCY_BOUNDARIES.md`](06_DEPENDENCY_BOUNDARIES.md)：分清自研与第三方。
8. [`07_FAILURE_HISTORY.md`](07_FAILURE_HISTORY.md)：从失败理解制度为何存在。
9. [`09_HANDS_ON_EXERCISES.md`](09_HANDS_ON_EXERCISES.md) 与 [`08_ORAL_EXAM.md`](08_ORAL_EXAM.md)：练习与自测。
10. [`10_SECOND_SCENE_PLAYBOOK.md`](10_SECOND_SCENE_PLAYBOOK.md) 与 [`11_LEARNING_PLAN.md`](11_LEARNING_PLAN.md)：迁移到第二场景。

## 最重要的十个源码文件

| 顺序 | 文件 | 为什么先读 |
|---:|---|---|
| 1 | `src/railway_recon/cli.py` | 所有实际入口和退出码 |
| 2 | `src/railway_recon/config.py` | 项目契约、路径安全、初始化 |
| 3 | `src/railway_recon/segments.py` | 里程分段与点云裁剪 |
| 4 | `src/railway_recon/algorithms/rail_candidates.py` | 钢轨候选的主算法 |
| 5 | `src/railway_recon/rail_review.py` | 候选到人工接受的哈希绑定 |
| 6 | `src/railway_recon/track_graph.py` | 全局拓扑和轨道身份 |
| 7 | `src/railway_recon/algorithms/track_graph_mesh.py` | 权威轨道网格入口 |
| 8 | `src/railway_recon/registry.py` | 资产状态、证据和冻结 |
| 9 | `src/railway_recon/quality_gate.py` | 发布门禁、豁免和不可变证据 |
| 10 | `src/railway_recon/release.py` | 模型/注册表/Web 发布绑定 |

## 最重要的十个符号/契约

1. `cli.main()`：命令分派和退出码。
2. `config.ProjectConfig`：路径与项目值的统一访问器。
3. `segments.plan_segments()` / `crop_segments()`：里程区段生命周期。
4. `rail_candidates.detect_rail_candidates()`：只产生候选。
5. `rail_review.apply_rail_review_package()`：独立复核的接受入口。
6. `track_graph.RouteSampler`：以相机累计链程为统一轴。
7. `track_graph.build_track_graph()` / `audit_track_graph()`：全局状态和拓扑审计。
8. `track_graph_mesh.build_track_graph_mesh()`：连续轨道网格及注册表映射。
9. `asset-registry.schema.json`：资产身份、证据与关系契约。
10. `quality_gate.evaluate_quality_gate()`：发布证据、哈希、审批和豁免契约。

## 最重要的十个系统不变量

1. 项目输出必须留在项目根目录内。
2. 局部候选不是权威几何；未复核来源不得令 TrackGraph 为 `pass`。
3. TrackGraph 必须与当前 `camera.csv` 和 TrackGraph 配置哈希一致。
4. 普通轨道节点不得无理由产生异常度数；内部终止必须有确认原因。
5. 不得为了“看起来连续”强制连接错误轨道家族。
6. 推断区间必须保留 `rule_inferred`，不能被汇总资产伪装成全观测。
7. 网格可见对象名必须唯一，并能映射到注册表资产。
8. 发布注册表只能包含 `accepted` 资产，且资产/关系集合必须冻结。
9. Web 发布配置必须绑定当前模型、注册表和资产集合的精确哈希。
10. 没有 EPSG、垂直基准和独立检查点时，只能声明内部拟合精度。

## 第一天的第一个动手练习

完成 [`09_HANDS_ON_EXERCISES.md`](09_HANDS_ON_EXERCISES.md) 的练习 1：不运行重建，只从 `cli.py` 追踪 `detect-rails`、`track-review-apply`、`track-graph-build`、`build-track-graph-mesh`、`registry-freeze` 和 `gate-evaluate` 六个命令，写出每个命令的输入、输出与失败退出码。完成标准是路径和函数名全部能在当前代码中定位。

