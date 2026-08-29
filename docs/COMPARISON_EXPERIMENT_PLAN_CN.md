# 方法对比与消融实验执行计划

## 1. 研究问题

本项目不把“做出一个可看的 200 米模型”直接当作算法精度。正式实验回答四个问题：

1. 几何候选提取是否优于通用点云基线；
2. TrackGraph 是否真正减少跨分段断裂、重复和错接；
3. 全景照片与铁路规则分别给语义分类带来多少收益；
4. 证据门控是否减少无依据补建，而不是简单隐藏困难样本。

现有 v9.x 工程模型经过多轮人工诊断和修复，只能作为 `HITL production upper bound`，不得改名为纯自动结果。

## 2. 固定实验矩阵

| 编号 | 对比 | 本项目侧 | 外部/删减侧 | 主要指标 |
|---|---|---|---|---|
| E1 | 轨道候选检测 | raw rail candidate detector | GISLab RailTrack | 轨头检测 P/R/F1、横向/高程误差、轨距原始误差 |
| E2 | 跨段连续性 | TrackGraph | 局部分段拼接、Full-no-topology | 断裂数/km、重复轨道、错接、端点间距与切向角 |
| E3 | 竖直资产语义 | Full-Auto | G、G+P、G+R | 实例 macro-F1、混淆矩阵、分类准确率 |
| E4 | 证据门控 | Evidence-Aware | Publish-All、Observed-Only | 支持率、确认错误率、无依据率、覆盖率、AURC |
| E5 | 通用几何提议 | 本项目几何提议器 | Open3D RANSAC+DBSCAN | 候选召回、误检、拟合残差、运行时间 |
| E6 | 自动质量门 | Quality Gate | 人工注入断裂/重叠/反法线/Z-fighting | 缺陷检测 P/R/F1、定位误差、误报 |

Grounded-SAM-2 在点云—全景投影冻结后再加入；Pointcept 在至少两个场景形成可用标签后再加入。当前阶段不把它们写成已完成基线。

## 3. 公平性约束

- 所有方法使用相同的冻结输入、25 米原子块、50 米评估组和相同 context；context 不计分。
- 候选检测、拓扑拼接、参数化建模分阶段比较，不拿一个完整系统与对方的单个中间模块比较。
- v2 评测将真值资产与预测资产分开存储，使用同块内中心距离进行匈牙利一对一匹配；匹配不读取类别标签。
- 外部方法的资产 ID 先经空间匹配映射到真值 ID，再计算拓扑关系；不要求外部方法复用本项目命名。
- 证据正确性由独立复核填写 `independent_evidence_status`，不得使用算法自己声明的证据等级作为答案。
- 风险分数只用于排序与 risk–coverage；没有独立场景校准前，不称为 correctness probability。
- 轨距同时报告 `raw_detection` 与 `constrained_output`。标准轨距约束后的 1.435 米不能冒充原始检测精度。

## 4. 真值与冻结流程

### 4.1 竖直资产

已生成双人随机顺序盲标包。两位标注者只查看未分类几何、全景证据与中性俯视索引，不查看最终模型或既有分类。两份 CSV 完成后计算一致率和 Cohen's kappa，分歧进入第三方裁决。

### 4.2 轨道几何

轨道真值必须来自无模型、无候选叠加的原始点云横断面和局部俯视图。中性证据清单必须声明：

```json
{
  "evidence_source": "raw_point_cloud",
  "model_overlay": false,
  "candidate_overlay": false
}
```

由此生成双人盲标任务：

```powershell
railway-recon benchmark-make-rail-truth-tasks `
  --root benchmarks/local/site-a `
  --evidence-manifest D:/private/rail-neutral-evidence.json

railway-recon benchmark-check-rail-truth-tasks `
  --path benchmarks/local/site-a/annotations/rail_geometry_blind_v1
```

现有“距离模型附近轨头点 40 mm 内再计算残差”的结果属于工程一致性 QA，不进入独立精度主表。

### 4.3 自动实验冻结

每个 AUTO/HITL 变体均需独立实验文件、非空命令、随机种子、参数和方法配置哈希：

```powershell
railway-recon benchmark-freeze --root benchmarks/local/site-a --full-hash

railway-recon benchmark-lock-experiment `
  --root benchmarks/local/site-a `
  --experiment benchmarks/local/site-a/experiments/g-only.json `
  --binding method_config=configs/g-only.json `
  --binding source_snapshot=dist/railway-recon-source.zip
```

锁定后不允许根据测试真值改阈值。需要改动时新建 experiment ID，并将旧版本保留为独立结果。

## 5. v2 统一评测

`railway.benchmark.evaluation-input.v2` 分开保存真值资产、预测资产、两侧关系图、双向几何距离、原始与约束后轨距、独立证据状态和人工复核时间。

```powershell
railway-recon benchmark-evaluate `
  --input benchmarks/local/site-a/experiments/evaluation-input-v2.json `
  --output benchmarks/local/site-a/reports/metrics-v2.json
```

## 6. 当前状态

| 项目 | 状态 |
|---|---|
| 数据、协议、空间切分冻结 | 已完成 0.5 m 空间体素级 80/20 冻结；四段校验通过 |
| v2 中性实例匹配 | 已实现并测试 |
| 跨方法拓扑 ID 映射 | 已实现并测试 |
| 独立证据指标与风险覆盖 | 已实现并测试 |
| 原始/约束后轨距分开报告 | 已实现并测试 |
| 可执行实验哈希冻结 | 已实现并测试 |
| 竖直候选盲标任务 | 已生成；当前无人标注，保留为未来可选增强，不阻塞无标注实验 |
| 轨道中性证据 | 已从 Site A 原始 LAZ 生成 100 个断面，未叠加模型或候选 |
| 轨道盲标任务 | 已生成并通过校验；当前无人标注，保留但不写入当前主结果 |
| GISLab RailTrack | 已用未修改上游提交在 Ubuntu 22.04 容器完成 8 个冻结 train/holdout 用例；7 个零检出、1 个超时、0 个几何输出，几何指标记为 N/A |
| Open3D 基线 | 已实现并运行 RANSAC 平面去除 + DBSCAN；与本方法使用同一冻结留出集 |
| 无 TrackGraph 消融 | 已运行；暴露 11 个跨段身份连接、11 个接缝和 1 个无依据内部新生 |
| 已知缺陷注入 | 已运行 7 类，质量门命中 7/7，无串扰失败检查 |
| 轨顶垂向提纯 | 已在 outer-train 内建立嵌套空间留出，固定规则选择 q50；配对与高度估计已解耦 |
| 第二独立场景 | 尚无，因此当前不声明跨场景泛化和统计显著性 |

## 7. 执行优先级

1. 完成无标注主实验的空间块 bootstrap 置信区间；
2. 对 Open3D 与本方法保存典型成功/失败断面，形成论文图；
3. 固定并运行 G、G+P、G+R、Full-Auto（语义部分当前只报告无真值指标）；
4. GISLab RailTrack 已完成固定提交、容器构建和 8 用例验证；下一步仅在预先登记的新协议下复验，不对本轮结果反向调参；
5. 新场景到来时保持参数冻结，直接执行 prospective blind test；
6. 有标注资源后再补资产 P/R/F1，不反向修改本轮无标注结果。

## 8. Site A 无标注留出实验（已执行）

评测单位不是随机点，而是完整 0.5 m 空间体素。四段合计训练点
94,683,812，留出点 23,156,391；同一体素不会同时进入训练与留出。
完整数据人工修复模型不能参加这组评测，候选必须仅由 train LAZ 重新生成。

| 指标 | Open3D RANSAC+DBSCAN | 本项目轨头高度显著性 | 较优 |
|---|---:|---:|---|
| train 候选线匹配召回 | 57.69% | 66.67% | 本项目 |
| holdout 候选线匹配精度 | 39.47% | 100.00% | 本项目 |
| 横向重复性 P90 | 68.65 mm | 24.62 mm | 本项目 |
| 垂向重复性 P90 | **30.12 mm** | 56.32 mm | Open3D |
| 原始轨距重复性 P90 | 122.47 mm | 23.65 mm | 本项目 |
| 模型到留出点支撑 P90 | 13.628 m | 69.33 mm | 本项目 |
| 50 mm 内留出点支撑率 | 63.22% | 83.47% | 本项目 |
| 100 mm 内留出点支撑率 | 66.48% | 95.03% | 本项目 |

机器可读结果位于私有基准目录：

- `reports/rail-holdout-v2.json`：本项目方法；
- `reports/rail-holdout-open3d-v1.json`：通用 Open3D 基线；
- `reports/rail-holdout-comparison-v1.json`：同一留出集配对比较；
- `reports/rail-holdout-bootstrap-v1.json`：按完整 50 米段重采样的探索性区间；
- `reports/topology-ablation-v1.json`：无 TrackGraph 消融；
- `reports/defect-injection-v1.json`：质量门缺陷注入。
- `reports/figures/rail-comparison-v2/`：四段原始点云横断面预测对比图，图内明确标注“非真值”。

这些数字只证明分割重复性和原始点支撑，不是真值精度、绝对测量精度或跨场景泛化。
当前配对比较 8 项中本项目胜 7 项、Open3D 胜 1 项；垂向稳定性是下一轮明确优化项。
空间块 bootstrap 只包含 4 个独立段，系统将其强制标记为
`exploratory_insufficient_independent_blocks`；有至少 10 个独立段前不作为确认性统计结论。

GISLab RailTrack 外部基线也已在同一冻结切分上执行。8 个 train/holdout 用例中，
7 个由日志证明为零检出，1 个超过资源门槛仍无输出，因此没有几何结果可进入上述数值表。
该方法的误差和覆盖率均记为 N/A，不以 0 冒充可比较数值。完整协议、容器提交、失败证据
和论文表述边界见 [GISLab RailTrack 外部基线对比](GISLAB_RAILTRACK_COMPARISON_CN.md)。

## 9. 轨顶垂向提纯、核心区隔离与连续 TrackGraph（开发后复盘）

### 9.1 已作废的早期数字

早期分段 LAZ 是带上下文的 AABB 裁剪，局部纵向范围可能超出正式 50 m 核心区。
旧检测器错误地让上下文点同时参与候选评分和轨顶拟合，形成了 core/context 泄漏。
因此旧的 q50 外层复评数字（包括 35.98 mm 垂向 P90）只保留为问题发现记录，**不得用于论文主表或精度声明**。

修复后，上下文只用于估计走廊方向；占用率、峰值、配对、轨顶拟合和候选输出都严格限制在分段核心里程。报告新增
`core_longitudinal_range_m`、`core_chainage_range_m`、`source_search_point_count` 和
`search_point_count`，用于审计上下文是否越权。

### 9.2 无标注嵌套选择

修复 core 泄漏后，仅在 outer-train 内的 0.5 m 空间体素嵌套留出上比较高度估计器。
候选配对固定使用 `rail_pair_top_method=height_grid_max`，高度估计器不得改变轨道数量、ID、横向位置和轨距。

| 高度估计器 | 垂向重复性 P90 |
|---|---:|
| 原方法 | 73.71 mm |
| anchored q50 | 72.92 mm |
| local q50 | **20.94 mm** |
| local q70 | 45.62 mm |
| local q75 | 56.52 mm |
| local q80 | 71.98 mm |
| local q85 | 81.60 mm |
| local q90 | 99.63 mm |

固定选择规则在嵌套验证中选择 `local q50`。但生产工程门禁是独立的否决条件：
`local q50/q70/q75/q80` 都在首个 50 m 的点支撑或覆盖检查中失败，不能因为验证集指标较好就直接替换生产估计器。
当前 200 m 生产候选因此保留通过同输入工程回归的 `anchored q50`：总体点支撑 P90
由 72.73 mm 降至 48.70 mm，100 mm 覆盖率由 94.36% 提升至 98.66%。
这些是同输入工程 QA，不是独立真值精度。

机器记录：

- `experiments/vertical_fit_core_v2/selection-v2.json`：修复 core 泄漏后的嵌套选择；
- `experiments/vertical_fit_core_v2/*-evaluation.json`：各分位数的机器可读指标；
- `workspace/reports/rail_vertical_q50_core_v4/coverage-expansion-audit-v3.json`：生产覆盖扩展审计；
- q50/q70/q75/q80 的失败候选均保留，禁止删除或覆盖，作为后续回归样本。

### 9.3 可审计的跨段共识

独立 50 m 直线拟合会在边界产生小幅横向和高程差。TrackGraph 只在以下条件同时满足时做共识校准：

1. 两段已经匹配为同一全局股道；
2. 边界里程差在冻结容差内；
3. 把差值平均分给两侧后，每侧修正量不超过 `maximum_automatic_correction_m`；
4. 原始横向差、高程差、三维差、共识值和每侧修正量全部写入图，不生成隐藏连接器。

0.5 m 断裂仍由回归测试保证必须失败。当前 v9 TrackGraph 包含 4 股道、16 个观测和
12 个接缝；最大原始端点差 155.5 mm，最大单侧共识修正 77.8 mm，低于本项目冻结的
100 mm 自动修正上限。最终 15/15 个 TrackGraph 检查通过，连续 Mesh 的 6 个固定视角未见断轨或压轨。
项目负责人豁免仍标记为 `project_owner_override`，不能写成完成了独立人工复核。

第二场景到来时，应冻结 core 隔离、候选配对、生产门禁和共识规则，直接执行 prospective blind test；
不得根据第二场景结果再次调参后仍称为盲测。
