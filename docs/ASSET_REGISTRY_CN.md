# 资产注册表与证据等级规范

资产注册表是模型、照片证据、人工结论、Web 查询和交付版本之间的主索引。模型中“看得见的物体”不等于已登记资产；正式资产必须有稳定 ID 和可追踪来源。

## 1. 注册表用途

- 保证 GLB、FBX、OBJ 与 Web 页面引用同一资产身份；
- 区分观测、照片解释、规则推断和无支持体块；
- 保留候选、接受、拒绝和替代历史；
- 支持按类型、里程、置信度和状态筛选；
- 支持缺口闭环和人工复核；
- 防止删掉几何后注册表仍显示“已完成”。

## 2. 顶层结构

```json
{
  "schema_version": "railway.asset-registry.v1",
  "project_id": "sample_line",
  "updated_at": "<ISO-8601 time>",
  "assets": [],
  "relations": [],
  "summary": {}
}
```

注册表使用 UTF-8 JSON。生产注册表不存绝对本机路径，证据引用采用项目内逻辑 ID 或相对路径。

## 3. 资产 ID

Schema 要求 ID 以大写字母开头，只含大写字母、数字、下划线和连字符。建议：

```text
<专业>_<类型>_<四位序号>
```

示例：

```text
TRK_RAIL_0001
OCS_MAST_0001
PLT_PLATFORM_0001
CNP_COLUMN_0001
BLD_FACADE_0001
SIG_SIGN_0001
```

规则：

- ID 创建后永久稳定；
- 几何调整不改 ID；
- 被拒绝或替代的 ID 不复用；
- 一个资产跨段时保留同一资产 ID，并记录相关段；
- 纯渲染拆分节点可以加组件 ID，但必须回指主资产；
- 不在 ID 中写站名、客户名或真实坐标。

## 4. 必填字段

| 字段 | 说明 |
|---|---|
| `id` | 稳定唯一标识 |
| `type` | 资产主类型 |
| `status` | 生命周期状态 |
| `evidence_level` | 主证据等级 |
| `confidence` | 0–1 的工程置信度 |
| `sources` | 至少一条来源；`unsupported` 可记录人工占位说明 |

推荐补充：`subtype`、`chainage_m`、`parameters`、`geometry`、`limitations`、`segment_ids`、`review`。

## 5. 生命周期状态

| 状态 | 含义 | 是否进入 clean 主模型 |
|---|---|---|
| `candidate` | 自动或人工发现，尚未完成定性 | 默认否 |
| `reviewed` | 已有人复核，但仍可能待补证据或精修 | 由发布策略决定 |
| `accepted` | 满足当前项目验收条件 | 是 |
| `rejected` | 误检或证据否定 | 否 |
| `superseded` | 被新资产或新几何替代 | 否，保留追踪 |

发布前不得保留“无处置说明”的 candidate。确实无法判断时，将其保留在 evidence 图层，并记录负责人、原因和后续动作。

## 6. 证据等级

### `observed`

适用于点云直接支持资产关键形体的情况。要求：

- 点支持覆盖足以确认存在和主要尺寸；
- 不是单个噪点或跨段重复；
- 资产与附近轨道/站台的关系合理；
- 证据引用至少包含点云来源或派生候选。

### `photo_interpreted`

适用于照片可确认、点支持不足的资产，例如可见门窗、标识或局部立面。要求：

- 相机与点云配准已经验证；
- 记录照片逻辑引用和可见范围；
- 不把透视尺寸直接当测量尺寸；
- 被遮挡部分不得自动升级为 observed。

### `rule_inferred`

适用于依铁路规则、重复柱网、拓扑连续或相邻构件推断的资产。要求：

- 记录规则 ID；
- 说明哪些参数来自邻接/模板；
- confidence 通常低于同类 observed 资产；
- Web 中能够按证据等级关闭或着色；
- 不进入实测精度统计。

### `unsupported`

适用于证据不足但业务需要保留的占位体块。要求：

- 明确限制；
- 不赋予看似精确的尺寸；
- 默认以低置信样式显示；
- 待补采或外部资料到达后重新评审。

## 7. 置信度使用规则

`confidence` 表示团队在当前证据下对“资产类型和建模结论”的工程判断。它不是：

- 点—模型距离；
- 测量精度；
- 未经校准的真实正确概率；
- 用来掩盖人工不确定性的数字。

推荐初期按统一规则赋值区间，而不是个人随意填写：

| 情况 | 建议处理 |
|---|---|
| 多源一致且人工通过 | 较高 |
| 点支持清晰但照片不可用 | 中高 |
| 照片确认、几何部分缺失 | 中等 |
| 主要依赖规则推断 | 较低 |
| 证据冲突或仅占位 | 很低并强制复核 |

项目若要把置信度解释成统计概率，必须使用独立真值进行校准并报告校准方法。

## 8. 来源记录

允许的 `kind`：

- `point_cloud`
- `panorama`
- `survey`
- `rule`
- `manual_review`
- `asset_ledger`

示例：

```json
{
  "kind": "rule",
  "reference": "RULE_OCS_CONTINUITY_001",
  "note": "由相邻已观测节点的连续关系提出，已进入人工复核队列"
}
```

照片来源使用内部逻辑引用，例如 `PANO_SET_A/view-logical-id`。通用仓库和脱敏示例中不得出现真实照片文件名、帧号或位置坐标。

## 9. 完整资产示例

```json
{
  "id": "CNP_COLUMN_0001",
  "type": "CanopyColumn",
  "subtype": "RectangularColumn",
  "status": "accepted",
  "chainage_m": 25.0,
  "evidence_level": "observed",
  "confidence": 0.88,
  "sources": [
    {
      "kind": "point_cloud",
      "reference": "SEGMENT_A/VERTICAL_CANDIDATE_01"
    },
    {
      "kind": "manual_review",
      "reference": "REVIEW_BATCH_A",
      "note": "类型和柱顶连接已复核"
    }
  ],
  "parameters": {
    "profile": "rectangular",
    "fit_method": "section_fit"
  },
  "geometry": {
    "model_node": "CNP_COLUMN_0001"
  },
  "limitations": []
}
```

示例只表达结构，不作为实际尺寸模板。

## 10. 资产关系

关系用于表达拓扑而非仅做页面连线。建议关系类型：

- `supports`：柱支撑梁、支柱支撑腕臂；
- `connects_to`：轨道段、线索段连续；
- `belongs_to`：子构件属于系统；
- `adjacent_to`：空间相邻；
- `derived_from`：新资产替代旧候选；
- `evidenced_by`：资产与证据对象关联。

示例：

```json
{
  "id": "REL_0001",
  "type": "supports",
  "from": "CNP_COLUMN_0001",
  "to": "CNP_BEAM_0001"
}
```

关系两端必须存在，不能引用 rejected 后已删除且未保留的未知 ID。

## 11. 人工复核记录

建议在资产中加入：

```json
{
  "review": {
    "required": false,
    "state": "approved",
    "reviewer_role": "railway-domain-reviewer",
    "reviewed_at": "<ISO-8601 time>",
    "decision": "accept",
    "note": "分类与连接关系通过"
  }
}
```

不得写入个人手机号、账号或无关身份信息。需要实名审签时应在受控交接系统保存，注册表只记录角色或内部审签号。

## 12. 模型同步规则

- 模型节点的 `assetId` 必须匹配注册表 `id`；
- 一个 clean 模型节点不得对应两个主资产；
- 可视化组件可共享主资产 ID，但需要组件索引；
- 删除几何时同步更新资产状态；
- 资产类型改变时保留变更记录；
- GLB、FBX、OBJ 导出后分别校验资产数、对象名和边界；
- candidate/evidence 资产与 clean 主资产使用不同状态或不同导出层。

## 13. 日常操作

初始化和验证：

```powershell
railway-recon registry-init --project projects/sample_line/project.json
railway-recon registry-check --project projects/sample_line/project.json
```

每次合并代码前应再次运行 `registry-check`。检查通过只代表 Schema 和基本引用正确，资产现实正确性仍需专业复核。

## 14. 发布门禁

- [ ] ID 唯一且稳定；
- [ ] 所有资产有 status、evidence_level、confidence 和 sources；
- [ ] candidate 均有处置结论；
- [ ] rejected/superseded 不进入 clean 主模型；
- [ ] rule_inferred/unsupported 可单独筛选；
- [ ] 注册表与各导出格式资产集合一致；
- [ ] 人工结论、规则和照片证据可追踪；
- [ ] 注册表不含绝对本机路径、真实照片帧或敏感坐标。
