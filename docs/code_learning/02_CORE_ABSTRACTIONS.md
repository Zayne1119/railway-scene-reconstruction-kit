# 02｜核心抽象：为什么存在

## 1. TrackGraph

**真实失败案例**：每个 50 m 区段独立检测时会出现轨道 ID 交换、段界断裂、错误轨道被强行连接、两条轨道重叠和内部无理由终止。

- 路径/符号：`src/railway_recon/track_graph.py::RouteSampler`、`build_track_graph()`、`audit_track_graph()`；schema 为 `resources/track-graph.schema.json`。
- 创建者：CLI `track-graph-build`。
- 消费者：`track-graph-validate`、`algorithms.track_graph_mesh.build_track_graph_mesh()`、拓扑/缺陷实验。
- 状态：source reports、observations、global track IDs、nodes、edges、seams、tracks、identity events、canonical axis 和 binding hashes。
- 输入/输出：接受过的 segment rail reports + manifest + camera/config → graph JSON + audit JSON。
- side effect：写 derived graph 与 reports audit；run wrapper 另写 manifest。
- failure mode：来源未复核时 `review_required`；方向、轨距、跨高、支持、接缝、重复或内部终止异常时 `fail`；不 force-connect。
- 删除后的后果：mesh 又会退化为“每段各画一组轨”，重现断裂/重叠。
- invariants：canonical direction；普通节点度数合理；同一 global track 跨段身份稳定；seam 在阈值内或明确失败；inferred span 有限；source review 完整。
- 测试：`tests/test_track_graph.py` 的 14 个用例，包括 reversed segment、wrong family、0.5 m seam、duplicate tracks、unreviewed geometry 和 crosslevel。

## 2. RailAssetGraph（实际为资产注册表关系图）

当前代码中不存在 `RailAssetGraph` class/schema。这是概念名称；实际载体是 `asset-registry.schema.json` 的 `assets` 与 `relations`，由 `registry.py` 管理。

- 解决：模型对象名、资产 ID、专业类型、证据和父子关系分离后无法查询/验收的问题。
- 创建者：`new_registry()`、TrackGraph mesh、reviewed scene、CSV/JSON import。
- 消费者：`quality_report()`、`freeze_release_registry()`、quality gate、Web viewer。
- 状态：asset status、evidence level、confidence、sources、parameters、geometry node/file、limitations；relations 的 from/to。
- side effect：更新 working `asset_registry` 或写 frozen registry。
- failure mode：重复 ID、关系悬空、confidence 越界、枚举非法；发布时任一非 accepted 即拒绝。
- 删除后的后果：几何仍可能显示，但不能证明“点的是哪个资产、由什么证据生成、是否可发布”。
- 测试：`tests/test_registry.py`、`test_release.py`、`test_qa.py`。

## 3. Evidence model

枚举来自 `asset-registry.schema.json`：`observed`、`photo_interpreted`、`rule_inferred`、`unsupported`。TrackGraph edge 允许前三类。

- 解决：不可见背面、规则补线、照片解释与点云实测若使用同一颜色/状态，会造成错误确定性。
- 创建者：detectors、reviewed layouts、TrackGraph/mesh builder、manual import。
- 消费者：registry summary、UI、review/gate、失败分析。
- 不变量：证据等级必须伴随 sources；推断不能升级为 observed；unsupported 不能伪装可验收；规则文件要求 photo/rule/unknown class 人工审核。
- 删除后的后果：低置信背面、推断接触线和观测钢轨无法区分。
- 测试：`test_track_graph_mesh.test_evidence_intervals_are_merged_without_hiding_inference`、benchmark evidence metrics。

## 4. Reconstruction mode

代码中没有一个统一 `ReconstructionMode` enum。可确认的模式分散在：

- `project.template.json::reconstruction`：pilot length、gauge、unseen facade policy、manual review required。
- graph edge `evidence_level`：observed/photo_interpreted/rule_inferred。
- rail review `review_mode`：independent review、project owner override、curation。
- benchmark `execution_mode`：auto/HITL。

因此它是**协议组合而非单类抽象**。迁移时若要加入新模式，应优先保持现有 evidence/status/schema，而不是再用自由文本状态绕过 gate。

## 5. Confidence

- 存储：candidate scores/metrics；asset registry 的 0..1 `confidence`；TrackGraph mesh 从 `track-build.default.json` 取 observed/inferred 默认值。
- 解决：表达模型者对资产判断的相对确定程度。
- 重要限制：当前 schema 只校 0..1，不证明 calibration；它不是 90% 正确率。
- failure mode：把默认 0.85 当统计概率，或用高 confidence 覆盖弱 evidence。
- 测试：registry schema/benchmark confidence calibration metrics。

## 6. Topology contracts

`audit_track_graph()` 将结构和几何合理性拆为命名检查：structure、node degree、direction、gauge、crosslevel、correction、support、seams、identity order、coverage、termination、inferred spans、duplicates、source review。

- 解决：视觉 QA 很晚才看见的断轨、错连、重叠。
- 不变量：拓扑失败不能靠平滑 mesh 掩盖；未匹配观测保留为断点；只有确认道岔/止挡可解释内部边界。
- 测试：`test_wrong_track_family_becomes_explicit_break_not_connector` 等。

## 7. Quality gates

`quality_gate.evaluate_quality_gate()` 把报告、artifact、registry、审批、waiver 和 previous gate 写入不可变目录。

- 解决：局部报告 pass 掩盖全局 fail、模型换了但验收表没换、口头豁免无责任人。
- 状态：PASS、FAIL、PASS_WITH_WAIVER、NOT_APPLICABLE。
- 不变量：只有显式 pass；candidate/conditional/pending/review/unknown 全阻断；waiver 绑定当前 artifact hash、至少两名不同审批人且未过期；QG7–QG9 必有 artifacts。
- side effect：新建 `gates/<release>/<QGx>`，存在即拒绝覆盖。
- 测试：`tests/test_quality_gate.py` 7 个用例。

## 8. Authoritative model

代码没有 `authoritative=True` 的模型字段。权威资格由多条件组合：

1. 几何控制图 audit pass；
2. mesh audit 和对象—registry 映射通过；
3. registry 所有资产 accepted 并冻结；
4. release/gate hash 精确绑定；
5. Web acceptance 模式再次验证。

这是一种“由证据链产生的资格”，不是某个文件名。删除这种组合约束后，`final.glb` 之类文件名可轻易冒充权威。

## 9. Manifest/hash

`io.sha256_file()`、`sha256_json()`、`manifest.write_run_manifest()`、TrackGraph bindings、registry freeze、gate artifacts 和 Web acceptance 共同使用 SHA-256。

- 解决：review 后改源、audit 后改 graph、交付后换 model/registry。
- failure mode：无 hash 或仅记录路径时，同名文件被替换无法发现。
- 测试：graph edit、camera drift、threshold drift、gate evidence changed、release bindings。

## 10. Fail-closed pipeline

Fail-closed 不是单个函数，而是多个决策点共享的原则：

- 未识别状态不当 pass：`_source_report_passed()`。
- 未复核 graph 不当 pass：`_source_review_status()` + graph audit。
- 旧 audit 不接受新 graph：`build_track_graph_mesh()` hash check。
- 缺 registry/artifact 不发布：QG7–QG9 checks。
- Web 验收绑定错则 `web/src/main.js::failClosed()`。

副作用是系统有时会“麻烦、需要人工”，但这正是防止错误模型被静默升级的成本。

