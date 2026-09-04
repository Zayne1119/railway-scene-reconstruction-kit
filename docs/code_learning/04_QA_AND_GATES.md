# 04｜QA、Guard 与 Release Gate

## QA 不是一件事

本项目至少有六层 QA：数据、证据、几何、拓扑、Mesh/装配、发布。`quality_report()` 是汇总器，不会替代专业检查；`evaluate_quality_gate()` 是证据封装器，不会自己重新跑检测算法。Gate 只相信你显式传入并能哈希的报告。

## 系统表

| 类别 / Gate | WHAT IT CHECKS | WHY | INPUT | 默认阈值/契约 | FAIL RESULT | Auto-fix? | 阻断权威? | 测试 |
|---|---|---|---|---|---|---|---|---|
| Project schema | 必填键、单位、Z-up、路径字段 | 防错误工程配置进入算法 | `project.json` | `project.schema.json` | `load_project` 抛异常 | 否 | 是 | `test_config.py` |
| Path guard | workspace 输出不逃逸 root | 防覆盖项目外文件 | path | `root in parents` | ValueError | 否 | 是 | `test_workspace_cannot_escape_project` |
| Input audit | 点云、相机列、顺序、照片引用、精度条件 | 早发现坏输入 | raw inputs | 必需列 index/timestamp/file/x/y/z | 报告 warning/error | 否 | 流程上应是 | camera/config/segments tests |
| Segment validation | 自适应区不重叠、正长度、合法 ID | 防重复/空裁剪 | camera trajectory/config | `length_m>0` | ValueError | 否 | 检测前是 | `test_segments.py` |
| Rail candidate guards | z、峰值、配对、轨距、跨高、支持 | 防噪声直接成轨 | LAZ/config | gauge pair 1.35–1.65 m；crosslevel ≤0.20 m；位置修正 ≤0.06 m | 无候选/异常，或仅 candidate | 部分约束，不权威修复 | 候选不能发布 | `test_rail_top_fit.py` |
| Source review | 决策、reviewer、hash、segment | 防审错版本/无人负责 | candidate package | accepted/rejected；hash exact | blocked | 人工 | 是 | `test_rail_review.py` |
| Graph binding | camera/config/source hashes | 防审计后漂移 | graph + project | 完全一致 | mesh 构建 ValueError | 必须重建/重审 | 是 | graph mesh drift tests |
| Topology structure | ID 唯一、引用、obs-edge 一致、node degree | 防断轨、错连、孤儿 | TrackGraph | 普通节点 degree 1/2 等 | graph fail | 否；应改上游 | 是 | `test_track_graph.py` |
| Direction/frame | 沿线方向、局部 frame 正交性 | 防段方向翻转 | observations | direction dot ≥0.99；det error ≤0.001 | fail | 小 seam 可显式 reconcile | 是 | reversed segment test |
| Gauge/crosslevel | 轨距、轨顶差 | 防两轨压叠/高程错 | observations | gauge error ≤0.02 m；crosslevel ≤0.20 m | fail | nominal constraint 有限度 | 是 | crosslevel/gauge tests |
| Observation support | 观测栅格覆盖 | 防细碎噪点成轨 | candidate metrics | coverage ≥0.20 | fail | targeted recovery 只重查证据 | 是 | targeted recovery tests |
| Pair continuity | 双轨共同支持/非对称/空洞 | 防一根轨凭空延伸 | candidate/graph | 默认 **diagnostic-only**；启用时 joint ≥0.10、asym ≤0.25、gap ≤5 m | 诊断或 fail | 可固定中心重查原始支持，不移动轨 | 启用后是 | top-fit/graph tests |
| Segment seams | gap/横向/竖向/3D/切线 | 防 50 m 段界断裂 | graph seams | 0.02/0.075/0.075/0.10 m；tangent dot ≥0.9999985；auto correction ≤0.10 m | fail | 仅阈值内显式 reconciliation | 是 | seam tests |
| Coverage/inference | 覆盖率、连续无观测、推断比例 | 防长距离补画 | tracks/edges | coverage ≥0.80；连续未观测/推断 ≤5 m；inferred ratio ≤0.10 | fail | 否 | 是 | long inferred edge test |
| Duplicate tracks | 同空间重叠轨道 | 防两套轨压在一起 | tracks | center ≤0.25 m 且 overlap ≥2 m 判重复 | fail | 否 | 是 | duplicate tracks test |
| Track mesh preflight | graph audit/hash/bindings、只观测策略、中心线步长/转角 | 防旧图/坏线成网格 | graph + audit/config | observed edges required；step ratio≤2；deflection≤15° | ValueError | 否 | 是 | `test_track_graph_mesh.py` |
| Rail–sleeper contact | 轨底与轨枕顶接口 | 防悬空/穿插 | generated parameters | error ≤0.001 m | ValueError | 调整参数后重建 | 是 | contact mismatch test |
| OBJ audit | 空模型、非有限、索引、退化/重复面、重复对象名 | 防导入破面和资产身份冲突 | OBJ | degenerate area ≤1e-12 算失败 | `passed=false` | 只可通过 DCC 清理后再审 | 是 | `test_mesh_audit.py` |
| Track assembly DCC audit | 轨枕相位、道床端点、材料/接缝 | 防整体模型集成错位 | Blender scene/config | 脚本中显式 thresholds | 非零退出/JSON fail | 有 repair/bridge，但修后必须重审 | 是 | pure `test_track_assembly.py` + DCC |
| Scene relationships | 柱—站台、柱—梁、梁—屋面、楼梯—站台/栏板等 | 防浮柱、穿插、断接口 | Blender scene + config | `scene_relationships.default.json` | 非零退出/JSON fail | repair 脚本可修部分，仍需重审 | 是 | `test_relationships.py`；DCC 未在 pytest 全跑 |
| Registry QA | schema、ID、关系端点、status | 防模型对象与资产账不一致 | registry | confidence 0..1；enum；unique IDs | invalid/release false | 显式 import replacement | 是 | registry/qa tests |
| Precision QA | EPSG、垂直基准、独立点 | 防把内部拟合冒充测量精度 | project CRS | 三项都具备 | `absolute_precision_ready=false` | 只能补测量资料 | 阻断绝对精度声明 | `test_qa.py` |
| Frozen registry | 非空、全部 accepted、集合 hash | 固定 release 资产范围 | working registry | status=accepted only | ValueError | 否，需完成审核 | 是 | `test_release.py` |
| Quality Gate v2 | 只认显式 pass、artifact/registry/previous gate/approval/waiver | 防局部通过掩盖全局失败 | reports/artifacts | QG7–9 必有 hash artifacts | FAIL/CLI exit 5 | 仅合规 waiver | 是 | 7 quality-gate tests |
| Web acceptance | 模型/registry/asset-set SHA-256 | 防上线文件被替换 | release config + downloads | 64 hex、release exact | `failClosed` | 否 | 是 | `web/test/acceptance.test.mjs` |
| Safety | 私有数据、模型、大文件、路径/IP/secret | 防公开仓污染 | repo tree | banned extensions/names/patterns | `passed=false` | 人工移除/隔离 | GitHub 发布是 | `test_safety.py` |

## Quality Gate 的关键行为

路径：`src/railway_recon/quality_gate.py`。

1. `_source_report_passed()` 只接受 `PASS`、`PASS_WITH_WAIVER`、大小写归一的 `pass/passed` 或布尔 `passed=true`；`candidate`、`conditional`、`pending`、`review`、未知状态全失败。
2. `_registry_checks()` 对发布 gate 要求 registry 存在、schema 有效、project/release 匹配、所有资产 accepted，并重新计算 asset set hash。
3. `_release_artifact_checks()` 对 QG7–QG9 验证 registry/model/Web config 的精确互相绑定。
4. `_valid_waiver()` 要求当前 artifact hash、至少两名不同审批人、未来到期时间、reason/risk/mitigation。
5. `evaluate_quality_gate()` 创建不可变 gate 目录；同 release/gate 再运行会 `FileExistsError`。
6. `validate_gate_result()` 重新检查 supporting docs、artifacts、previous gate 的大小与 SHA-256。

注意：代码只有在调用时传入 `previous_gate` 才建立链。没有一个固定 stage registry 自动强迫所有 QG0…QG9 按序执行。

## 自动修复的边界

- 允许：小接缝在 `maximum_automatic_correction_m` 内显式记录 reconciliation；DCC repair/bridge 脚本对确定的接口修复；targeted recovery 在固定中心位置重新检查原始支持。
- 不允许：为连续而连接错误轨道；移动轨道去迎合噪声；把推断改成 observed；自动接受照片姿态；自动接受任何资产；在 gate 内把 fail 改 pass。
- 修复后的共同规则：产生新 artifact/报告，再重新 audit/gate；不能复用旧 hash。

## 十个最重要的不变量

1. Candidate 永远不是 authoritative。
2. Review 必须绑定候选内容 hash。
3. Graph、camera、settings 形成不可分的绑定三元组。
4. 分段 ID 不得在全局匹配中交叉换序。
5. 错误轨道宁可断开也不能 force-connect。
6. 推断证据不能被汇总隐藏。
7. 每个可点击 Mesh 对象必须有唯一资产身份。
8. 所有发布资产必须 accepted，且 release 集合冻结。
9. Gate 只认显式通过；模糊状态一律阻断。
10. 无独立测量控制，禁止绝对精度声明。

