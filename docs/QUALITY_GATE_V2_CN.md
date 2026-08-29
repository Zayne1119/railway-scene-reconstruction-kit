# Railway Scene Reconstruction Kit：Quality Gate v2

本文定义铁路场景结构化重建项目从立项、数据接收、Pilot、全线生产到发布和复用的统一质量门禁。目标是让每个阶段都具备明确输入、可验证产物、硬阻断条件、独立审批和可追踪哈希，避免配置漂移、人工决定丢失、发布包混版以及页面“看似可用但实际加载错误”等问题。

Quality Gate v2 是项目治理协议，不代替铁路专业判断、测量规范、现场安全制度或数据授权流程。自动检查通过只证明已实现的检查项通过，不能自动证明所有现场资产正确。

## 1. 适用范围

本规范适用于：

- 点云、相机轨迹和全景影像的接收与审计；
- 20–50 m Pilot 的选择、建模和冻结；
- 轨道、站台、雨棚、接触网、立面和线路附属资产的结构化重建；
- 人工复核、资产状态转换和证据登记；
- OBJ、GLB、FBX、Web、Blender 和 UE 交付；
- 生产发布包、研究实验包和后续项目复用。

公开代码仓库只保存通用代码、Schema、空模板、合成示例和脱敏文档。生产点云、全景、控制点、精确坐标、设备台账、客户信息、生产模型及内部审批记录应保存在获批的受控存储中。

## 2. 基本原则

1. **原始输入不可覆盖**：新批次使用新的 batch ID，不在原路径替换旧文件。
2. **配置即基线**：算法阈值、规则、依赖和代码 commit 共同决定 baseline。
3. **候选不等于资产**：自动候选默认不能直接进入 clean 主模型。
4. **人工决定可追踪**：人工删除、补全、移动、分类和风险接受必须留下追加式记录。
5. **建模与验收分离**：关键 Gate 采用 maker/checker 分离，建模人不能独自完成最终签字。
6. **产物按内容识别**：文件名不是版本依据，SHA-256、release ID 和来源 run 才是依据。
7. **发布包由清单组装**：不得手工从多个历史目录拖放文件拼成 final。
8. **失败必须沉淀为回归**：已修复的 major/critical 问题必须进入自动或固定视角回归集。
9. **页面验收必须失败闭合**：配置、模型或注册表加载失败时，验收模式不得显示占位模型冒充成功。
10. **复用流程而非旧几何**：第二项目重新执行数据审计和 Pilot，不复制旧项目坐标、资产 ID、照片编号或最终模型。

## 3. 唯一 Gate 状态枚举

所有 Gate 结果只能使用以下四个值，不得自行增加 `accepted`、`conditional_pass`、`warning` 等同义状态：

| 状态 | 含义 |
|---|---|
| `PASS` | 所有适用检查均通过，不存在有效 waiver |
| `FAIL` | 至少一个硬阻断或不可豁免检查失败 |
| `PASS_WITH_WAIVER` | 仅存在允许豁免的问题，且 waiver 已完成审批并绑定当前产物哈希 |
| `NOT_APPLICABLE` | 本项目确实不适用该 Gate 或检查，已记录理由和批准人 |

规则：

- 任一不可豁免项失败，Gate 必须为 `FAIL`。
- `FAIL` 时不得进入下一生产阶段。
- `PASS_WITH_WAIVER` 不是“口头同意继续”，必须有完整 waiver 记录。
- `NOT_APPLICABLE` 不能由执行人自行填写，必须由 Gate owner 和 QA 批准。
- 命令运行状态如 `completed/failed` 与 Gate 状态是不同概念，不得混用。

## 4. Gate 证据包与哈希链

每道 Gate 建议生成独立证据包：

```text
workspace/gates/<release-id>/<gate-id>/
├─ gate-result.json
├─ artifact-manifest.json
├─ checks.json
├─ approvals.json
├─ waivers.json
├─ logs/
└─ evidence/
```

### 4.1 `gate-result.json` 最低字段

```json
{
  "schema_version": "railway.quality-gate.v2",
  "project_id": "example-project",
  "release_id": "example-release",
  "baseline_id": "<sha256>",
  "gate_id": "QG0",
  "status": "PASS",
  "previous_gate_sha256": null,
  "run_ids": [],
  "input_artifacts": [],
  "output_artifacts": [],
  "checks": [],
  "approvals": [],
  "waivers": [],
  "generated_at": "<ISO-8601>"
}
```

从 QG1 开始，`previous_gate_sha256` 必须指向上一道 Gate 的 `gate-result.json` 内容哈希。任何上游证据被替换后，下游 Gate 链立即失效。

### 4.2 产物记录

每个产物至少记录：

- 逻辑 ID 和角色，如 `clean_model`、`asset_registry`、`quality_report`；
- 脱敏相对路径；
- 文件类型和 Schema 版本；
- 字节数和 SHA-256；
- 生成它的 run ID；
- 上游输入哈希；
- 是否为正式发布必需项。

正式冻结时必须重新计算内容哈希，不能仅信任人工填写的 declared hash。

### 4.3 基线身份

建议按以下内容计算 `baseline_id`：

```text
SHA256(
  Git commit
  + project.json
  + rules.json
  + 全部算法配置
  + dependency lock
  + 相关 Schema 版本
)
```

每个生产 run、人工复核包和 release manifest 都必须引用同一个 `baseline_id`。配置变化后必须生成新 baseline，不能继续复用旧报告或旧截图。

## 5. QG0：项目立项与数据权限

### 目标

在接收或处理数据前冻结项目范围、交付目标、权限边界和角色责任。

### 必需产物

- 项目范围及明确排除项；
- 交付格式、目标设备和使用场景；
- 数据授权、模型外发和公开边界；
- 精度声明边界；
- RACI 角色表；
- Gate 计划和缺陷严重度定义。

### 硬阻断

- 数据来源或使用权限不清；
- 未定义建模范围或交付目标；
- 未指定项目负责人、数据负责人和独立 QA；
- 需要现场补采但安全授权或作业边界不明确；
- 被要求输出超出当前真值和坐标条件所能支持的精度等级。

### Waiver

数据授权、现场安全和保密问题不可豁免。缺少 CRS 或独立检查点可以继续进入内部拟合项目，但必须在本 Gate 将成果等级和声明范围降级。

### 角色

- Accountable：项目负责人；
- Responsible：数据负责人；
- Verifier：QA/合规角色。

## 6. QG1：数据接收与不可变输入基线

### 目标

证明接收的数据完整、可解释、不可静默替换，并明确其限制。

### 必需产物

- 唯一 batch ID；
- 数据接收清单；
- 点云、相机、全景和可选测量资料的完整 SHA-256；
- 点数、字段、范围、scale/offset、单位和轴向说明；
- 相机姿态约定和照片缺失统计；
- CRS、垂直基准、控制点和独立检查点状态；
- 输入审计报告；
- 数据接收签字记录。

### 硬阻断

- 单位、轴向或坐标框架未知且无法确认；
- 点云与相机轨迹明显不对应；
- 必需输入缺失；
- 实际文件哈希或字节数与移交清单不一致；
- 原始输入被覆盖；
- 数据权限不清楚；
- 多批数据使用不同坐标基准但没有转换记录。

### 允许 Waiver 的情形

- 部分全景缺失；
- CRS、垂直基准或独立检查点缺失；
- 不影响当前目标的可选字段缺失。

Waiver 必须同步写明受影响区间、不可评价指标和成果降级方式。

### 角色

- Responsible：数据负责人；
- Consulted：Pipeline 工程师；
- Approver：项目负责人和 QA。

## 7. QG2：代码、配置与环境冻结

### 目标

建立可重跑的唯一生产 baseline，避免“同名版本、不同参数”。

### 必需产物

- Git commit/tag 和 dirty 状态；
- 项目配置、规则和全部算法配置快照及哈希；
- Python、Node、Blender 等工具版本；
- dependency lock 或等效依赖摘要；
- baseline ID；
- 配置变更记录。

### 硬阻断

- 代码来源不明或 dirty 工作树直接作为正式 baseline；
- 配置校验失败；
- 生产配置包含本机绝对路径或未脱敏现场标识；
- 现场参数写入核心源码；
- 依赖版本不可恢复；
- 修改配置后仍引用旧 QA、旧模型或旧截图。

### 变更影响规则

| 变化 | 必须重跑 |
|---|---|
| 新数据批次 | QG1 及之后 |
| 坐标或局部原点变换 | 所有几何和交付 Gate |
| 算法、阈值或铁路规则 | QG4 及之后 |
| 资产状态或人工决定 | 受影响模型生成阶段及之后 |
| 导出参数或材质策略 | QG6 及之后 |
| Web 代码或部署配置 | QG8 及之后 |
| 纯文字文档 | 重新生成发布清单，无需重算模型 |

### Waiver

正式发布不能豁免未知代码状态、不可恢复依赖或配置校验失败。紧急视觉修复也必须生成新 baseline 和 run。

### 角色

- Responsible：Pipeline 工程师；
- Verifier：QA；
- Approver：技术负责人。

## 8. QG3：Pilot 选区冻结

### 目标

选择具有代表性且包含真实困难点的 20–50 m Pilot，避免只挑最容易成功的区段。

### 选区记录

候选 Pilot 至少比较：

- 点密度和照片覆盖；
- 轨道数量和复杂度；
- 竖直/线性资产数量；
- 是否包含站台、雨棚、接触网或其他核心交付对象；
- 遮挡、玻璃、植被和结构交界；
- 已知数据盲区。

如果一个 Pilot 无法同时覆盖代表性和困难性，可以冻结一个主 Pilot 和一个 challenge micro-pilot。

### 必需产物

- 候选区段比较表；
- 选区理由和未选理由；
- 分段清单与人工确认的包围盒；
- Pilot 输入哈希；
- Pilot 负责人和 QA 签字。

### 硬阻断

- 只选择最干净的简单直线区；
- Pilot 不包含轨道或项目核心资产；
- 相机或点云覆盖不足；
- 自动包围盒未经人工检查；
- 在观察完整算法结果后才反向挑选 Pilot；
- 选区无法代表计划扩展区域。

### Waiver

项目范围确实不包含某类资产时可标为 `NOT_APPLICABLE`。不能用 waiver 合理化明显选择偏差。

### 角色

- Responsible：Pipeline 工程师；
- Consulted：铁路专业复核；
- Approver：QA。

## 9. QG4：Pilot 技术验收与参数冻结

### 目标

在扩展生产前证明坐标、投影、候选提取、人工复核、资产注册和导出链路能够闭环。

### 必需产物

- 多相机投影验证；
- 轨道、竖直和线性候选报告；
- 人工复核决定；
- Pilot 模型与资产注册表；
- 固定六视角；
- Mesh 审计；
- 问题关闭清单；
- 冻结配置和 baseline 哈希。

### 硬阻断

- 相机姿态只由单帧确认；
- 轨道断裂、重复、错误跨轨或严重轨距异常；
- 未处置候选进入 clean；
- 自动候选未经复核即标为 `accepted`；
- 模型、注册表和人工决定不同步；
- 存在未解释的 critical/major 问题；
- QA 完成后又修改配置或几何；
- 建模人独自完成专业和 QA 签字。

### Waiver

数据不足或明确不在范围内的对象，可以保留在独立 candidate/evidence 图层，并记录 `unsupported` 或 `rule_inferred`。错误轨道拓扑、候选混入 clean 和伪造精度声明不可豁免。

### 角色

- Responsible：Pipeline 工程师；
- Domain Approver：铁路专业复核；
- Gate Approver：独立 QA。

## 10. QG5：全线生产与人工决定闭环

### 目标

保证所有区段使用可追踪 baseline，所有人工编辑和候选处置都有结构化记录。

### 人工状态流

```text
candidate → reviewed → accepted / rejected
accepted → superseded
```

不得静默删除历史资产，不得用覆盖旧记录的方式修改人工决定。

### 人工决定最低字段

```text
decision_id
candidate_id 或 asset_id
previous_state
decision
evidence references and hashes
reviewer role
reason
affected asset IDs
timestamp
supersedes_decision_id
```

### 必需产物

- 每段 run manifest；
- 配置一致性报告；
- 追加式人工决定日志；
- 合并后的资产和关系注册表；
- candidate closure 报告；
- 跨段连续性与重复资产报告。

### 硬阻断

- 区段使用不同 baseline 而无变更记录；
- 存在未处置候选；
- 人工移动、删除、补全或分类无日志；
- `rejected/superseded` 资产进入 clean；
- 重复资产 ID 或未知关系端点；
- 人工决定更新后没有重新生成和哈希模型；
- 新旧几何版本同时进入场景；
- 分段拼接产生断裂、重复或错误连接。

### Waiver

仅允许对明确、局部、低严重度且不影响安全和核心拓扑的问题签署 waiver。Waiver 必须列出精确资产 ID、区段、风险、补偿检查和失效日期。

### 角色

- Responsible：Pipeline 工程师；
- Decision Owner：铁路专业复核；
- Verifier：QA。

## 11. QG6：技术冻结与失败案例回归

### 目标

在发布组装前冻结几何、注册表、Mesh、格式转换和历史失败回归结果。

### 必需产物

- 几何和拓扑 QA；
- 模型—注册表资产集合检查；
- OBJ/GLB/FBX 统计和边界检查；
- Mesh 自动审计；
- 固定视角人工验收；
- 历史失败回归报告；
- 当前候选 release 的产物哈希。

### 失败严重度

| 严重度 | 示例 | 发布处理 |
|---|---|---|
| `critical` | 坐标错误、轨道错误连接、敏感数据泄露、发布包混版 | 必须阻断，不可豁免 |
| `major` | 核心资产漏建、严重悬空、屋面破坏、模型—注册表不一致 | 默认阻断，修复或正式降级范围 |
| `minor` | 不影响资产身份和核心几何的局部显示问题 | 可修复或签署限期 waiver |

### 失败案例回归

每个已修复的 major/critical 问题应产生：

- 稳定 `failure_id` 和 `regression_id`；
- 首次发现和修复 release；
- 合成 fixture 或受控证据哈希；
- 预期结果与容差；
- 自动检查脚本，或固定镜头人工检查；
- 当前负责人和关闭依据。

建议覆盖：

- 双轨重叠与轨道段界断裂；
- 错误跨轨连接；
- 屋面破片和共面闪烁；
- 柱—梁—屋面悬空；
- 楼梯顶层平台和栏板接口；
- 接触网错误连线；
- candidate 残留在 clean；
- 模型节点与注册表 ID 不一致；
- 旋转后 Web 拾取错误。

### 硬阻断

- 任一 critical 回归失败；
- fixed 的 major 问题重新出现；
- 核心指标退化超出冻结容差；
- 模型和注册表资产集合不一致；
- 非有限坐标、退化面或不允许的重复面；
- 固定视角显示严重断裂、悬空、破面或错误拓扑。

### Waiver

critical 问题不可豁免。minor 视觉问题必须绑定当前产物哈希，并写入 limitations。

### 角色

- Responsible：Pipeline/导出工程师；
- Verifier：QA；
- Domain Approver：铁路专业复核。

## 12. QG7：发布包组装与防混版

### 目标

确保 clean、Web LOD、HQ、collision 和 evidence 等多个交付 profile 均来自同一正式基线，并防止模型、注册表和 manifest 混版。

### Release 身份

每个产物必须同时记录：

```text
base_release_id
profile_id
source_run_id
baseline_id
data_batch_id
git_commit
```

多个 profile 可以拥有不同几何哈希，但必须引用同一个 `base_release_id`，并记录父模型哈希和派生转换 run。

### 必需校验

- 发布目录符合明确 allowlist，无历史版本和额外文件；
- 所有正式文件有字节数和 SHA-256；
- 模型、注册表、manifest 和 Web config 的 release ID 一致；
- 模型节点 ID 集合与注册表发布资产集合一致；
- 计算排序后 `asset_set_sha256` 和 `relation_set_sha256`；
- clean 不包含 candidate、rejected 或 superseded；
- candidate/evidence 使用独立 profile；
- GLB/FBX/OBJ 的单位、局部原点、边界和发布资产集合一致；
- release manifest 记录 registry hash；
- 发布在 staging 中完成校验，再提升为 final；
- final 目录不可覆盖，修复时创建新 release ID。

### 硬阻断

- 任一文件哈希不一致；
- model、registry、manifest 的 release ID 不同；
- 新模型与旧注册表组合；
- 人工替换已签字文件；
- clean 含未接受资产；
- 存在未登记的额外文件；
- 发布包包含敏感数据、绝对路径或历史工作文件。

上述完整性问题不可豁免。

### 角色

- Responsible：Release 工程师；
- Verifier：独立 QA；
- Approver：项目负责人。

## 13. QG8：Web、Blender 与 UE 验收

### 目标

在实际目标环境中验证 QG7 已锁定的精确产物，而不是重新复制另一份模型。

### Web 自动检查

- 配置、注册表和模型均成功加载；
- release ID、期望哈希和资产数一致；
- 注册表通过 Schema；
- 所有可见节点都有资产映射；
- 旋转后拾取仍正确；
- 搜索、状态和证据筛选正确；
- candidate/evidence 默认显示策略正确；
- 加载错误进入明确失败状态；
- 桌面和移动视口通过；
- 加载时间、峰值内存、帧率和拾取延迟满足目标。

### Blender/UE 检查

- 在全新空场景导入；
- 单位、轴向和局部原点正确；
- 材质、法线、透明件和细线正确；
- GLB/FBX/OBJ 边界一致；
- 资产身份在格式转换后保留；
- 无严重闪面、黑面、背面消失或碰撞异常；
- 固定六视角由独立 QA 复核。

### 硬阻断

- 页面显示占位模型；
- registry 加载失败但仍显示验收成功；
- 未注册节点可见；
- 高亮对象与信息面板 ID 不一致；
- Web config 指向其他 release；
- 模型加载失败；
- 目标设备无法完成核心交互；
- 格式转换丢失资产身份或改变坐标。

### Waiver

仅允许对已确认不影响资产身份、几何可信性和核心交互的性能或显示 minor 问题签署 waiver。加载失败、资产映射错误、hash/release 不一致不可豁免。

### 角色

- Responsible：Web/引擎负责人；
- Verifier：独立 QA；
- Approver：项目负责人。

## 14. QG9：最终批准、归档与发布

### 目标

确认 Gate 链、发布包、风险接受和复现信息完整后，才把 candidate 提升为 final。

### 必需归档

- QG0–QG8 的 gate-result 哈希链；
- release manifest 和全部正式文件哈希；
- 自动与人工 QA；
- waiver register；
- limitations；
- failure regression 结果；
- Git tag、commit、baseline ID 和配置快照；
- 最终审批记录。

### 硬阻断

- Gate 哈希链断裂；
- 任一道适用 Gate 为 `FAIL`；
- waiver 已过期或不适用于当前产物哈希；
- final 与已签字 staging 内容不同；
- 存在未解释 FAIL；
- 安全检查或人工发布清单检查失败；
- 无法从 release manifest 确认模型来源。

### Waiver

QG9 不能创建新的技术 waiver，只能确认上游已批准且仍有效的 waiver。发现新问题必须返回相应 Gate。

### 角色

- Responsible：Release 工程师；
- Verifier：QA/安全角色；
- Accountable：项目负责人。

## 15. Waiver 统一规则

每份 waiver 至少包含：

```text
waiver_id
gate_id
check_id
severity
rationale
affected artifact IDs and SHA-256
affected asset IDs/segments
risk and user impact
compensating controls
approver roles
approved_at
expires_at or review_release
closure_issue
```

统一规则：

1. Waiver 只适用于列出的精确产物哈希；相关文件重建后自动失效。
2. Waiver 必须有期限或下一 release 的强制复核点。
3. 建模人不能批准自己的 waiver。
4. 专业几何/拓扑问题由铁路专业负责人和 QA 联合批准。
5. 数据/隐私问题还需要数据负责人和项目负责人批准。
6. Web/性能问题由 Web/引擎负责人、QA 和项目负责人批准。
7. Waiver 必须进入 release limitations，不能只留在聊天或会议纪要中。

不可豁免项：

- 数据授权、现场安全和敏感信息泄露；
- 未知单位或冲突坐标框架；
- 输入、Gate 或发布产物哈希不一致；
- 发布包混版；
- critical 轨道/拓扑错误；
- candidate 冒充 accepted；
- clean 中存在未注册几何；
- 模型加载失败却显示占位场景；
- 伪造绝对精度或超出证据范围的声明。

## 16. 角色与 maker/checker 分离

| 角色 | 主要责任 | 不应独自批准 |
|---|---|---|
| 项目负责人 | 范围、资源、风险接受、最终发布 | 专业几何和数据授权事实 |
| 数据负责人 | 接收、权限、原始哈希、批次管理 | 自己发现的数据授权例外 |
| Pipeline 工程师 | 配置、运行、模型和注册表生成 | QG4、QG6 最终结果 |
| 铁路专业复核 | 资产存在性、类型、拓扑和规则 | 文件完整性与安全扫描 |
| QA/Release Manager | Gate 证据、回归、防混版和签字链 | 自己主导建模的结果 |
| Web/引擎负责人 | Web、Blender、UE 和性能验收 | 模型—注册表不一致 waiver |
| 独立复现人员 | clean-clone 和第二项目演练 | 首项目实现细节的自证 |

小团队可以由一人承担多个执行角色，但 QG4、QG6、QG7 和 QG8 必须保留至少一名未参与相应产物制作的 verifier。

## 17. 第二项目启动与复用标准

第二项目必须从正式 tag 和空项目目录启动：

1. 由未参与首项目实现的人从 clean clone 检出指定 tag；
2. 安装锁定依赖并运行 smoke test、单元测试、静态检查和安全检查；
3. 跑通无现场信息的合成最小样例；
4. 使用 `railway-recon init` 创建新项目；
5. 建立新的 data batch，不复制旧项目的 `input/` 或 `workspace/`；
6. 对新输入执行 full-hash 审计；
7. 所有现场差异只进入项目配置、规则或有测试的通用模块；
8. 建立 Pilot 候选矩阵并冻结选区；
9. 依次通过 QG0–QG9；
10. 把缺失能力登记为产品化任务，不复制带历史版本号和现场常量的脚本。

### 复用成功标准

- 新同事仅依赖正式文档和 tag 即可完成初始化、审计、分段和 Pilot；
- 核心源码不出现新项目坐标、照片编号、资产 ID 或路径；
- 不复制旧项目最终模型、注册表或人工结论；
- 所有配置变化都能通过 baseline hash 识别；
- 一个完整 release candidate 可由 run manifest 和 Gate 链追溯；
- 历史失败回归在新项目上仍执行；
- 数据安全和公开边界保持不变。

如果第二项目必须复制旧项目专用脚本、旧坐标、旧注册表或旧模型才能继续，应判定为工具包能力缺口，而不是复用成功。

## 18. 当前已有代码与缺口边界

下表用于避免把文档要求误称为已自动实现的能力。

| 能力 | 当前边界 |
|---|---|
| 项目初始化、目录隔离和配置校验 | 已有代码 |
| 点云、相机和全景基础审计 | 已有代码；正式冻结仍需显式 full hash |
| 分段规划和 LAS/LAZ 裁切 | 已有代码；包围盒仍需人工检查 |
| 轨道、竖直和线性候选 | 已有基础代码；候选仍需照片和专业复核 |
| 点—全景姿态搜索和叠加图 | 已有代码；自动最优假设仍需多视角验收 |
| 资产注册表 Schema、导入和引用检查 | 已有代码；不能证明资产现实正确性 |
| reviewed layout 转参数化 OBJ | 已有代码；不会替代专业判读 |
| 基础 QA | 已有代码；主要检查输入、审计、分段和注册表条件 |
| OBJ Mesh 基础审计 | 已有代码；不覆盖跨对象共面重叠、完整法线和运行时闪烁 |
| Blender 清理和格式导出报告 | 已有代码；仍需固定视角和资产身份复核 |
| Web 模型浏览、资产查询和证据筛选 | 已有代码；acceptance mode 已校验 release/model/registry/asset-set 哈希并 fail closed，尚缺真实浏览器和性能回归 |
| 安全扫描和 CI | 已有代码；Web CI 主要验证可构建，不等于交互验收 |
| Benchmark Schema、冻结、统一指标和双盲复核支持 | 已有基础能力；生产 Gate 与 benchmark Gate 仍需统一 |
| 生产 Run Manifest | 现有生产 manifest 已记录输出文件存在性、大小和 SHA-256；完整依赖 DAG 和统一 v2 仍待接入 |
| Pilot 代表性评分和电子签字 | 主要是文档要求，尚无通用 Gate runner |
| 配置 baseline、漂移检测和影响分析 | 尚未形成完整自动化 |
| 通用人工决定追加日志和 candidate closure | QG7–QG9 已阻断非 accepted 资产；追加式人工决定日志仍待实现 |
| 全场景几何、拓扑和跨格式 QA | 全局 TrackGraph v1 已实现方向、身份、接缝、覆盖、推断、内部终止和重复轨道门禁；连续轨道 OBJ 已只消费通过的图并完成基础 Mesh 审计，其他资产拓扑及跨格式 QA 尚未完全通用化 |
| 失败案例与自动回归绑定 | 有失败记录结构，尚缺统一 regression runner |
| 通用 release builder/validator | 已能冻结 accepted-only 注册表、生成哈希绑定 Web 配置并验证 Gate 产物；完整 allowlist 打包器仍待实现 |
| 防混版、profile 血缘和资产集合哈希 | release ID、模型/注册表 SHA-256 和 asset-set SHA-256 已接入；多 profile 血缘仍待实现 |
| Web 浏览器、移动端、拾取和性能自动验收 | 尚无完整实现 |
| Gate 状态、waiver、审批和哈希链 | P0 runner 已实现：四状态、失败传播、不可覆盖目录、前序 Gate 哈希、显式 waiver 和绑定当前产物的审批；Gate 策略配置化仍待完善 |
| 第二项目端到端 orchestrator | 尚无完整实现 |

## 19. 已实现的 P0 命令

### 19.1 冻结 accepted-only 发布注册表

```powershell
railway-recon registry-freeze `
  --project projects/sample_line/project.json `
  --release-id release-001 `
  --output workspace/exports/release-001/asset_registry.json
```

只要存在 `candidate/reviewed/rejected/superseded`，命令即失败。冻结文件写入 `release_id`、`asset_set_sha256` 和 `relation_set_sha256`，并且拒绝覆盖。

### 19.2 生成 fail-closed Web 配置

```powershell
railway-recon web-release-config `
  --project projects/sample_line/project.json `
  --release-id release-001 `
  --title "Railway release-001" `
  --model workspace/exports/release-001/scene.glb `
  --registry workspace/exports/release-001/asset_registry.json `
  --output workspace/exports/release-001/project.json `
  --model-url /models/scene.glb `
  --registry-url /data/asset_registry.json
```

生成的配置绑定模型、注册表和资产集合哈希。Web 使用 `acceptance_mode: true` 或 URL 参数 `?acceptance=1` 后，缺配置、缺模型、缺注册表、哈希不一致、release 不一致、非 accepted 资产或未注册 Mesh 都会停止验收，不再展示占位模型冒充成功。

### 19.3 评估不可覆盖的 Gate

```powershell
railway-recon gate-evaluate `
  --project projects/sample_line/project.json `
  --release-id release-001 `
  --gate-id QG7 `
  --previous-gate workspace/gates/release-001/QG6/gate-result.json `
  --check track.full_length=workspace/reports/track-full-length.json `
  --check mesh.scene=workspace/reports/mesh-scene.json `
  --artifact model=workspace/exports/release-001/scene.glb `
  --artifact registry=workspace/exports/release-001/asset_registry.json `
  --registry workspace/exports/release-001/asset_registry.json
```

只有明确的 `pass/PASS/PASS_WITH_WAIVER` 或布尔 `passed: true` 才算通过；`conditional_pass`、`pass_candidate`、`review_required`、`pending`、`completed` 和未知状态全部阻断。QG1–QG9 必须引用同 release 的前一道 Gate，且前一道状态必须通过。

```powershell
railway-recon gate-validate `
  --path workspace/gates/release-001/QG7/gate-result.json
```

验证命令会重新计算支持文件、输入报告、发布产物和前序 Gate 的哈希，同时重新核对 Gate 状态是否与失败项一致。

Waiver 默认禁用。只有同时传入 `--allow-waiver <check-id>` 和完整 waiver 文件才可能得到 `PASS_WITH_WAIVER`；waiver 必须绑定当前产物 SHA-256、至少两个批准人和未过期日期。注册表状态、哈希、Gate 链和发布身份检查不可 waiver。

## 20. 后续推荐实现顺序

1. 实现 baseline ID、配置漂移和影响 DAG；
2. 把全部生产 CLI 阶段接入统一 run manifest v2；
3. 实现追加式人工决定日志和状态转换审批；
4. 实现模型节点 ID—注册表—关系集合的跨格式一致性；
5. 为 TrackGraph 轨道增加独立留出点残差、轨顶纵断面和超高拟合；
6. 实现接口图、跨对象 BVH、自交、穿插和近平行面检测；
7. 完成 release allowlist、profile 血缘和原子 staging→final；
8. 增加真实浏览器、移动端、拾取、性能和 UE 时序回归；
9. 把 Gate 必需审批角色和阈值改为项目策略配置；
10. 由独立人员执行第二项目 clean-clone 演练。

自动 Gate 只能阻断它已经知道的失败。全局 TrackGraph v1 已覆盖首批轨道方向、身份、接缝、覆盖、推断和重复问题，但不能替代道岔专业确认、空间 Mesh 接口和 UE 时序检查。在这些能力完全代码化前，仍应执行本文的人工 Gate，并把每次签字绑定到准确的 baseline、release ID 和产物 SHA-256。
