# 09｜12 个动手练习

共同规则：使用复制出来的 synthetic/minimal project 或测试临时目录；不要直接改生产项目。每次练习先写“我预计会发生什么”，再执行。除非题目明确允许，不修改 production source。

## 练习 1｜追踪六个命令（Level 1）

- 目标：从 CLI 追出 rail authoritative path。
- 允许修改：只写个人笔记。
- 禁止修改：全部源码、配置、项目数据。
- 任务：定位 `detect-rails`、`track-review-apply`、`track-graph-build`、`build-track-graph-mesh`、`registry-freeze`、`gate-evaluate` 的 parser、dispatch、实际函数。
- 预期现象：每个命令都能写出 input/output/side effect/exit behavior；不存在一个总 pipeline 函数。
- 验证：用 `railway-recon <command> --help` 对照参数。
- 完成标准：12 个函数/路径引用无误，能解释哪个阶段把 candidate 提升为 accepted。

## 练习 2｜画一条数据血缘（Level 1）

- 目标：追踪一个 rail pair 从 JSON 到 OBJ node 和 registry asset。
- 允许修改：笔记/临时图。
- 禁止修改：源码、真实 registry。
- 任务：沿 `rail_candidates → rail_review → TrackGraph observation/edge → _track_evidence_intervals → _asset` 记录 ID、evidence 和 hash。
- 预期现象：一个 aggregate track 对多个 rail interval、sleeper group、bed 有 contains relations。
- 验证：对照 `tests/test_track_graph_mesh.py::test_builds_one_continuous_mesh_set_per_global_track`。
- 完成标准：解释为什么可见 node 数和 registry asset 数可能不同。

## 练习 3｜预测分段变化（Level 2）

- 目标：理解 adaptive segment zone。
- 允许修改：临时项目 `project.json` 的 segmentation。
- 禁止修改：detector/graph 源码。
- 任务：把默认 50 m 改为 25 m，并只在一段 adaptive zone 用 10 m；运行 `plan-segments` 前先预测区间。
- 预期现象：zone 外按默认长度，zone 内细分；不产生重叠。
- 验证：运行 `tests/test_segments.py::test_adaptive_segment_zone_refines_only_requested_chainage` 或检查 manifest。
- 完成标准：预测区间与 manifest 完全一致，能解释边界 remainder。

## 练习 4｜Rail detector 参数敏感性（Level 2）

- 目标：建立 bin/prominence/coverage 的直觉。
- 允许修改：临时复制的 rail settings；只跑 synthetic/非保密 segment。
- 禁止修改：已冻结 settings、reviewed report、graph。
- 任务：分别增大 `minimum_height_prominence_m`、`minimum_line_coverage`、`cross_bin_m`，逐次只改一项。
- 预期现象：候选数、peak 位置或支持率变化；结果仍为 candidate。
- 验证：比较 report fields 与 diagnostic PNG，记录 config hash。
- 完成标准：每项给出“过低/过高”的失败模式，且不把开发结果写进 release。

## 练习 5｜打开 paired continuity gate（Level 2）

- 目标：理解 diagnostic-only 与 enforcement。
- 允许修改：临时 TrackGraph settings。
- 禁止修改：candidate metrics 伪造 pass。
- 任务：对同一 synthetic graph 分别关闭/开启 `paired_rail_continuity_gate_enabled`。
- 预期现象：同一差支持 observation 在关闭时只产生 diagnostics，在开启时可令 graph fail。
- 验证：运行 `test_paired_support_is_diagnostic_only...` 与 `test_asymmetric_paired_support_fails...`。
- 完成标准：能指出何时新场景可启用，以及启用前需哪些标定数据。

## 练习 6｜给已有 stage 加 diagnostic（Level 3）

- 目标：练习“加可观测性而不改接口”。
- 允许修改：新分支中 `detect_linear_candidates()` report，增加一个纯统计字段；相应测试。
- 禁止修改：schema version、候选选择逻辑、asset acceptance。
- 任务：添加 vertical candidate height P50/P90（无候选时为 null），不改变候选集合。
- 预期现象：原输出仍兼容，新增统计可用于调参。
- 验证：existing tests 全过；新增 synthetic test 比较确切统计。
- 完成标准：输出确定、无随机性、无 candidate 数变化，并记录为何不是质量 gate。

## 练习 7｜注入 graph seam 缺陷（Level 4）

- 目标：亲手验证 topology gate fail-closed。
- 允许修改：复制的 synthetic graph fixture。
- 禁止修改：质量阈值来“放过”缺陷。
- 任务：给 seam 注入 0.5 m lateral/chainage error。
- 预期现象：`audit_track_graph()` fail；mesh builder 拒绝 non-pass audit。
- 验证：参考 `test_half_metre_seam_is_detected` 和 `defect_injection.py`。
- 完成标准：指出失败 check ID、数值、阈值和上游正确修复点。

## 练习 8｜篡改证据后验证 hash（Level 4）

- 目标：理解 review/audit/gate 的内容绑定。
- 允许修改：临时复制的 candidate/graph/artifact。
- 禁止修改：真实 review package 和 release gate。
- 任务：分别在 review 后改 candidate、audit 后改 graph、gate 后改 artifact 一个字节。
- 预期现象：source hash mismatch、graph hash mismatch、gate validate invalid。
- 验证：对应 rail review、track graph mesh、quality gate tests。
- 完成标准：三层都能描述“谁保存 hash、谁重新计算、谁失败”。

## 练习 9｜制造并捕获 Mesh 身份错误（Level 4）

- 目标：区分几何退化与资产身份错误。
- 允许修改：临时 OBJ fixture。
- 禁止修改：生产 OBJ。
- 任务：创建两个同名 `o TRACK-001`，几何本身有效。
- 预期现象：duplicate object name 令 `audit_obj().passed=false`。
- 验证：`tests/test_mesh_audit.py`。
- 完成标准：解释为什么“画面看起来没问题”仍必须失败。

## 练习 10｜替换局部 peak detector（Level 5）

- 目标：学习接口稳定的算法替换。
- 允许修改：实验分支新增 rail peak variant 与 benchmark command/config；不可成为默认。
- 禁止修改：candidate report contract、review/TrackGraph/registry/gate。
- 任务：用另一种 1D peak/cluster 方法生成相同 report 必需字段。
- 预期现象：下游 review package 能无差别读取；指标可与 in-house/Open3D/GISLab 比较。
- 验证：contract tests + spatial holdout comparison。
- 完成标准：生产默认不变，variant/config/hash 可复现，失败不被记为 0 成功。

## 练习 11｜新增最小 asset type（Level 6）

- 目标：把 `kilometer_post` 从审核布局带到 Web 查询。
- 允许修改：临时 reviewed layout、registry fixture、必要的 UI 类型映射/测试。
- 禁止修改：证据枚举、绕过 accepted/freeze、假造自动 detector。
- 任务：用 box/panel 生成里程标，sources 指向 point cloud 或 panorama，写 limitations。
- 预期现象：`build_reviewed_scene` 生成唯一 node；registry valid；接受后可冻结；Web picking 对应 ID。
- 验证：reviewed-scene、registry、release、Web tests。
- 完成标准：几何、ID、type、evidence、source、confidence、limitations 六者一致。

## 练习 12｜第二项目配置迁移演练（Level 7）

- 目标：证明你能独立启动新场景而不复制旧阈值结论。
- 允许修改：全新 synthetic project 与独立 config copies。
- 禁止修改：第一项目 frozen artifacts；不可把旧 EPSG/threshold 当事实。
- 任务：init → 填 inputs/CRS unknown → validate/audit → 50 m plan/crop → rail/linear candidates → review path 设计 → graph/mesh/gate dry run。
- 预期现象：未知 CRS 会被诚实记录；候选不自动 accepted；只有通过全链才 release-ready。
- 验证：保存每步 run manifest、失败清单、参数校准表。
- 完成标准：能明确列出“直接复用、需重标定、需人工批准、当前 UNKNOWN”四类项目项。

