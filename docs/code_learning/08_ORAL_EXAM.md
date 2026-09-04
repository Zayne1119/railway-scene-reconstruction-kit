# 08｜项目 Owner 口试题

先独立回答。答案集中在文末，不要边看边答。题目只考当前项目，不考 Python 语法。

## A. 基础题（20）

1. 当前项目真正的 Python CLI entrypoint 是什么？
2. 为什么说本仓库没有“一键 pipeline orchestrator”？
3. `ProjectConfig.workspace_path()` 保护什么不变量？
4. `camera_trajectory()` 中的 `distance_m` 是怎样得到的？它是否等于正式铁路里程？
5. segment manifest 由哪个函数产生，谁消费？
6. `detect_rail_candidates()` 的三个主要输出是什么？
7. 为什么 rail detector report 不能直接进入权威 mesh？
8. 当前有哪三种把 rail source 变成可接受来源的路径？
9. TrackGraph 的 `observation`、`edge`、`track` 分别表示什么？
10. TrackGraph 为什么要绑定 camera CSV 和 settings hash？
11. `observed` 与 `rule_inferred` 在当前代码中造成什么实际差异？
12. 代码中是否存在名为 `RailAssetGraph` 的 class 或 schema？
13. working registry 与 frozen release registry 有何区别？
14. 为什么 registry relation 的 `from`/`to` 必须都存在？
15. `audit_obj()` 能检测什么，明确不能检测什么？
16. `quality_report.release_ready` 至少依赖哪些条件？
17. QG7–QG9 为什么要求 hashed artifacts？
18. Web acceptance config 绑定哪三类 hash？
19. 没有 independent check points 时可以声明哪种精度？
20. Open3D 在当前项目中是生产依赖还是比较基线？

## B. 系统设计题（20）

21. 为什么局部 rail detector 的输出不应直接交给 mesh generator？
22. 为什么错误轨道宁可断开，也不能自动 force-connect？
23. TrackGraph 相比“把每段 OBJ 拼起来”多解决了哪些状态问题？
24. 为什么 review 生成新文件而不直接修改 candidate report？
25. owner override 为什么必须保留 `independent_review_completed=false`？
26. 为什么推断 rail interval 要拆成单独可见 node/asset？
27. 为什么 aggregate track asset 还要保存 `evidence_intervals`？
28. 为什么 sleeper 使用全局 phase，而不是每段从 0 开始？
29. 为什么 TrackGraph mesh 构建前要重新运行 current audit，而不仅看旧 audit status？
30. 为什么 `candidate`、`conditional`、`review_required` 都不能被 gate 当 pass？
31. 为什么 waiver 要绑定 artifact hash 且至少两人批准？
32. 为什么 gate 目录不可覆盖，而应使用新 release id？
33. 为什么模型文件名不能表示 authoritative？
34. 为什么 confidence 不能替代 evidence level？
35. 为什么最小 RGB MAE 不能自动接受照片 pose？
36. 为什么 core OBJ audit 不能替代 UE/Blender 固定视角验收？
37. 如果以后接入 SAM2，为什么应把它放在 candidate producer 而不是 registry accepter？
38. 为什么第二场景可以复用 schema，却不能照搬 rail thresholds？
39. 为什么 pipeline 没总 DAG 编排器是一个实际运营风险？
40. 你会如何定义真正的 authoritative model 状态，而不新增一个容易被伪造的布尔值？

## C. Debugging 题（20）

41. 所有资产在 UE 中整体偏移 0.5 m，你先查哪三层？
42. 只有弯道附近 rail 横向偏移，直线正常，你优先查什么？
43. 两个相邻 segment 的轨道方向反了，哪个 graph check 应失败？
44. segment 边界有 8 cm 横向 gap，默认会自动修吗？为什么？
45. graph audit 文件显示 pass，但 graph JSON 后来被编辑，mesh builder 会怎样？
46. camera.csv 修改了一个 pose，但 graph/audit 未重建，会怎样？
47. rail pair 的共同支持很差，但默认 graph 仍可能通过，为什么？
48. 网页里出现两条几乎完全重叠的轨道，应依次检查哪四处？
49. rail 与 sleeper 之间悬空 2 mm，哪个阈值会拦截？
50. OBJ audit pass，但 UE 中仍闪面，为什么不矛盾？
51. 屋面下看出现大片消失，依次查 winding、normals 和什么导出设置？
52. 柱底浮在站台上 4 cm，应读哪个 DCC audit 的哪类关系？
53. 楼梯顶层与站台错台，应检查哪些 relationship 项？
54. projection overlay 左右镜像，但 RGB score 很低，为什么仍不能接受？
55. registry-check 通过，但 `registry-freeze` 失败，最常见原因是什么？
56. frozen registry 成功，但 Web 页面 fail closed，应该比较哪些 hash？
57. gate 的 report 文件是 `conditional`，视觉上没有问题，为什么 gate 仍 FAIL？
58. 更换 TrackGraph threshold 后复用旧 graph，哪个绑定检查会抓住？
59. 新 detector 在开发段更好、holdout 也异常好，如何排除 threshold leakage？
60. 安全检查报告包含 private IPv4，为什么这会阻断公开仓发布但不等于模型错误？

## D. “为什么不这样做”题（10）

61. 为什么不把 gauge 永远强制成 1.435 m 并忽略原始 pair？
62. 为什么不把所有缺失 rail 用 spline 自动补满？
63. 为什么不把 200 m 雨棚生成为一个连续大平面？
64. 为什么不让 Open3D DBSCAN 的 cluster 直接成为 accepted rail？
65. 为什么不只保存 GLB，删除 graph、registry 和 reports？
66. 为什么不在 gate 失败时自动把状态改为 PASS_WITH_WAIVER？
67. 为什么不把低 confidence 资产直接删除，让模型看起来更干净？
68. 为什么不使用随机点划分 train/holdout？
69. 为什么不只用双面材质掩盖 normal/winding 问题？
70. 为什么不把第一项目调好的全部阈值当铁路行业标准？

## E. 第二场景迁移题（10）

71. 收到 `new.laz + camera.csv + panoramas` 后第一小时要确认哪些元数据？
72. 哪些 schema/config 可以直接复制，哪些值必须改？
73. 第一段 prototype 为什么应选 50–100 m 而不是立即全线？
74. 新站场是强弯线时，首先验证哪两个坐标算法？
75. 新点云密度只有原项目一半，哪些 rail 参数必须重新标定？
76. 新站有道岔，TrackGraph 哪些默认假设必须重点复审？
77. 新 camera.csv 没有旋转字段，哪些流程仍可跑，哪些不能可靠跑？
78. 新场景只有一名 reviewer，如何诚实使用 owner override 而不伪装独立复核？
79. 什么时候可以从 prototype 扩展到全 corridor？
80. 什么时候可以对外使用“authoritative”一词？

---

# 答案区

## A 答案

1. `pyproject.toml` 的 `railway-recon = railway_recon.cli:main`，最终为 `cli.py::main()`。
2. 没有统一 DAG/Stage 类；`main()` 只是子命令分派，操作者显式串联命令。
3. 所有 workspace 输出必须位于项目根目录内。
4. 相邻 camera XYZ 的三维欧氏距离累计；它只是相机轨迹链程，不是正式铁路里程。
5. `segments.plan_segments()` 产生；`crop_segments()` 和 `fit_segment_corridor_frame()` 等消费。
6. candidate LAZ、诊断 PNG、rail candidate JSON report。
7. 状态是 geometry candidate，需要 hash-bound 人工接受和全局 topology audit。
8. independent review apply、project-owner override、rail-pair curation。
9. observation 是分段测量；edge 是链程区间几何/证据；track 是跨段的全局身份汇总。
10. 防 camera 或阈值改变后旧 graph 仍被误用。
11. 影响颜色/节点、asset evidence/confidence、mesh policy、release 解释；推断不能伪装观测。
12. 没有；实际是 asset registry 的 assets + relations。
13. working 可含多状态并持续更新；frozen 必须非空、全 accepted、带 release/set hashes 且不可覆盖。
14. 防悬空关系，保证资产图可遍历、Web 查询不指向不存在对象。
15. 检查空/非有限/索引/退化/重复 face/重复对象名；不能发现跨对象共面重叠、运行时闪面或完整法线问题。
16. registry 非空有效且全 accepted，TrackGraph/mesh（若配置）通过，输入/manifest 等 checks 通过。
17. 确保 release/runtime/final gate 针对具体二进制，不是口头或路径名。
18. model SHA、registry SHA、asset-set SHA。
19. 只能声明相对当前点云/模型的内部拟合，不能声明绝对世界精度。
20. optional benchmark baseline，默认非生产权威路径。

## B 答案

21. 局部候选没有全局身份、人工接受、接缝和推断资格。
22. 错连会制造不存在的资产/拓扑，显式断点更可诊断、更安全。
23. global ID、canonical chainage/direction、seams、nodes/edges、source review、evidence intervals、duplicate/termination checks。
24. 保留原始算法输出和 review provenance，避免审后内容被静默改写。
25. 让下游知道责任与证据等级，不能把单人决定冒充独立共识。
26. 让视觉、registry 和 QA 都能定位推断区，防聚合隐藏。
27. aggregate asset 自身是查询入口，必须说明哪些局部不是 observed。
28. 每段重置会在边界产生重复/异常间距；全局 phase 保持连续。
29. 配置/项目状态可能漂移；旧 audit 只证明旧输入。
30. fail-closed：模糊状态不能证明通过。
31. 防 waiver 套到新文件，并建立双人风险责任。
32. 保留完整审计历史，避免证据被重写。
33. 名称不绑定内容、证据和资产集合；hash-bound gates 才绑定。
34. confidence 是数值信心，evidence 是来源性质；高信心规则推断仍不是观测。
35. 颜色相似、曝光和轴镜像会产生假最优；必须看结构边和多视图。
36. audit 明确不查跨对象共面、运行时 shader/culling/normal 行为。
37. 第三方模型可换；接受决策、拓扑、证据和发布合同必须由本系统控制。
38. 数据密度、断面、轨迹和噪声不同；schema 是接口，threshold 是场景标定值。
39. 人可能跳过 audit/review/previous gate；应靠 playbook/CI 或未来编排器降低风险。
40. 用 graph pass + mesh/关系 pass + accepted frozen registry + hash-bound gate/Web config 的组合资格。

## C 答案

41. raw CRS/pose；model origin/axis/unit；Blender/UE import transform。
42. segment corridor frame 上下文、dominant regression 与 RouteSampler 曲线切向。
43. canonical direction / minimum direction dot，通常同时影响 seam/identity。
44. 默认 lateral threshold 7.5 cm，8 cm 超阈值；即使 auto correction 上限 10 cm，也必须先满足各 seam contract，不能盲修。
45. graph SHA 与 audit 不匹配，`build_track_graph_mesh()` 抛 ValueError。
46. binding stale，拒绝构建。
47. paired continuity gate 默认 disabled/diagnostic-only；需要项目标定后显式启用。
48. graph duplicate check；registry ID/node；Blender 是否保留旧轨；Web 是否加载错误 GLB/release config。
49. `maximum_rail_sleeper_contact_error_m=0.001`，2 mm 超限。
50. OBJ audit limitations 明确不含跨对象共面 Z-fighting 和 runtime material flicker。
51. back-face culling、负缩放/未应用 transform、导入 normal/tangent recompute 策略。
52. `blender/audit_scene_relationships.py` 的 column–platform signed contact。
53. stair–platform surface、stair–guard distance，并核对 enclosure 接口。
54. score 只衡量颜色误差，不证明几何 convention；镜像可在重复纹理中得低分。
55. 存在非 accepted 资产或 registry 为空。
56. config 记录的 model SHA、registry SHA、asset-set SHA 与实际下载内容。
57. `_source_report_passed()` 只认显式 pass，conditional 是阻断态。
58. `validate_track_graph_bindings()` 的 settings hash。
59. 确认 whole-voxel split、experiment/prediction lock、参数在外 holdout 前冻结；必要时 nested holdout。
60. 它是 confidentiality/release hygiene gate，不是几何 QA。

## D 答案

61. 原始 gauge/crosslevel 是诊断证据；无限修正会把错误 pair 拉成“标准轨”。
62. 无观测长补线制造虚假拓扑；推断长度/比例有 gate。
63. 会覆盖真实断段和无证据区；应按实际屋面段和证据建资产。
64. cluster 没有人审、轨道配对、全局身份和 release binding。
65. 无法追溯、定位、重建、查询或证明 GLB 属于哪个 release。
66. waiver 是治理决定，需要显式允许、当前 hash、风险说明和双人审批。
67. “不存在”与“证据不足”不同；删除会隐藏 coverage 风险。
68. 相邻/同 voxel 点泄漏会虚高泛化指标。
69. 双面材质只掩盖症状，光照、阴影、性能和 UE 导入仍可能错。
70. 配置文件自己声明这些是项目保守默认，不是行业法规；必须新场景校准冻结。

## E 答案

71. 单位/轴、CRS/垂直基准、点格式/RGB、camera 列/时间/旋转约定、照片引用、数据许可和范围。
72. schema、目录结构、规则枚举可复制；project identity/input paths/CRS、segmentation bounds、detector/graph/quality thresholds 必须重填或重标定。
73. 先验证完整证据链和 failure modes，避免全线放大错误/成本。
74. corridor frame（尤其 dominant regression/context）与 RouteSampler chainage/tangent。
75. bin size、minimum points/coverage/prominence、quantile/minimum bin points、support thresholds。
76. one-to-one identity、普通 node degree、interior boundary reasons、inferred spans、turnout connector 审批。
77. audit/segment/点云 detectors 可运行；projection calibration 不能可靠运行，照片解释 pose 缺关键输入。
78. 使用 `track-review-owner-override`，填写真实 approver/reason，保留 false 和 warning；不要改成 reviewed_accepted。
79. prototype 的 input QA、review、graph、mesh/关系、registry/gates 全通过，阈值冻结且失败处置流程可重复后。
80. 针对明确 release，具备通过的 geometry/topology/mesh QA、全 accepted frozen registry、hash-bound gate/交付；绝对精度另需 CRS/基准/独立点。

