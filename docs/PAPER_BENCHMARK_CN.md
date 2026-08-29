# 论文 Benchmark v1 操作手册

## 1. 目的

生产 QA 证明模型能否交付；论文 Benchmark 证明结果是否在冻结输入、冻结切分和统一指标下可复现。两者不能互相替代。

本协议重点评价五件事：

1. 几何与独立参考数据的一致性；
2. 资产实例识别是否正确；
3. 支承、连接和归属关系是否正确；
4. 证据等级与置信度是否可信；
5. 自动流程和人工复核分别消耗多少时间。

## 2. 数据角色

### Site A：开发案例

已经被反复观察、调参和人工修复的现场必须标记为 `development_case_study`。它可以用于方法开发、受控消融、场景内稳健性和失败案例分析，但不能作为独立盲测。

### Site B0：前瞻盲测

新数据到场后先登记哈希，冻结代码、阈值、资产字典、匹配规则和主指标。B0 仅允许格式、单位、CRS 与传感器标定适配，不允许根据测试结果改算法。

### Site B1：少量适配

B0 锁定后，可以使用声明过的 20–50 m Pilot 调参，剩余区段保持测试状态。B0 与 B1 必须分别报告。

## 3. 初始化与冻结

```powershell
railway-recon benchmark-init benchmarks/local/site-a `
  --dataset-id railway-site-a `
  --scene-id station-000-200m

railway-recon benchmark-validate `
  --path benchmarks/local/site-a/manifests/dataset-manifest.json

railway-recon benchmark-check --root benchmarks/local/site-a

railway-recon benchmark-freeze --root benchmarks/local/site-a

railway-recon benchmark-evaluate `
  --input benchmarks/local/site-a/experiments/evaluation-input.json `
  --output benchmarks/local/site-a/reports/metrics.json
```

默认冻结会复用数据清单中已声明的 SHA-256，并检查文件是否存在及字节数是否一致。发布论文或接收新盲测数据时使用 `--full-hash` 重新计算内容哈希。

## 4. 空间切分

- 按沿线里程唯一归属，不随机拆点或拆照片；
- 25 m 为原子块，50 m 为评估核心区；
- 核心区两端允许 10 m context，但 context 不计分；
- 同一物理资产、重复采集和对应照片不得跨 split；
- 跨段资产的实例归属按质心，边界拓扑单独登记。

## 5. Ground truth

真值等级：

- `GT-L0_survey`：独立测量或独立检查点；
- `GT-L1_observed_consensus`：两名标注者依据原始点云和照片形成的一致结果；
- `GT-L2_expert_ledger`：专业人员或设备台账确认；
- `GT-L3_pseudo`：由现有模型派生，只能开发评测器，不能当测试真值。

至少 20% 分层样本需要双人独立标注，所有疑难项双标。标注者不能查看待评方法的最终模型。分歧必须进入裁决记录，不能静默覆盖。

参与配准的 `control_points` 与用于最终评价的 `independent_check_points` 必须分开。没有独立检查点时，只能报告相对于当前观测的内部拟合误差。

### 竖直候选双人盲标包

使用未分类的 `continuous_candidate_features.json` 生成两个随机顺序不同、标签栏为空的 CSV，并复制对应全景裁剪、几何剖面和俯视索引图：

```powershell
railway-recon benchmark-make-vertical-tasks `
  --root benchmarks/local/site-a `
  --source s000_50m=D:/private/s000_50m/continuous_candidate_features.json `
  --source s050_100m=D:/private/s050_100m/continuous_candidate_features.json

railway-recon benchmark-check-vertical-tasks `
  --path benchmarks/local/site-a/annotations/vertical_candidates_blind_v1
```

不得把 `classification.json`、最终资产注册表或最终模型提供给标注者。两位标注者完成后再按 `task_id` 合并、计算一致率并进入裁决。

## 6. 固定实验组

| 变体 | 含义 |
|---|---|
| `G` | 仅几何候选与参数拟合 |
| `G+P` | 几何与全景，不使用铁路规则 |
| `G+R` | 几何与铁路规则，不使用全景 |
| `Full` | 几何、全景、规则、拓扑和证据门控 |
| `Full-no-topology` | 去掉资产关系约束 |
| `Full-no-evidence-gating` | 去掉证据分级与低置信控制 |
| `Full-no-gap-closure` | 去掉残差缺口闭环 |

历史工程版本如果同时改过点集、几何和阈值，只能用于失败演化展示，不能直接当受控消融。

## 7. 指标

### 几何

双向距离、MAE、RMSE、P50、P90、P95、F-score@固定阈值；轨道还报告轨距、横向、高程、接缝距离和切向角。

### 实例与拓扑

实例 Precision、Recall、macro-F1、混淆矩阵；关系边 Precision、Recall、F1；断裂数/km、错误跨轨连接和悬空连接。

### 证据与风险

无依据新增率、证据引用可解析率、ECE、Brier、reliability diagram、risk–coverage/AURC。原始启发式分数必须称为 risk score，经过独立标签校准后才能称为 correctness probability。

### 效率与交付

wall time、CPU/GPU、峰值内存、人工作业分钟/100 m、操作次数、三角面、文件大小、加载时间、退化面、重复面、法线和非流形问题。

## 8. 统计规则

- 主统计单位是独立场景，空间块是场景内次级单位；
- 不把数百万个点当成数百万个独立样本；
- 同一空间块进行配对比较；
- 使用空间块 cluster bootstrap 报告 95% CI 和效应量；
- 场景过少时逐场景报告，不强行声明普遍泛化或显著性。

## 9. 人工复核日志

每次复核记录开始/结束时间、候选、证据、决策和编辑动作。纯自动结果与 HITL 结果分开保存。人工模型或人工上界也必须采用固定时间预算。

## 10. 发布边界

公开仓库只提交通用代码、Schema、空模板与授权过的合成示例。点云、全景、精确坐标、客户名称、设备台账、生产模型、完整报告和内部 Benchmark 清单默认不进入 Git。

## 11. 在真值产生前锁定旧预测

双人盲标开始后、任何标注结果返回前，将既有 `classification.json` 单独锁定。该文件只能用于后续预测—真值对比，不得发给标注人员：

```powershell
railway-recon benchmark-lock-vertical-predictions `
  --root benchmarks/local/site-a `
  --annotation-package benchmarks/local/site-a/annotations/vertical_candidates_blind_v1 `
  --source s000_50m=D:/private/s000_50m/classification.json `
  --source s050_100m=D:/private/s050_100m/classification.json
```

命令会校验预测与盲标任务的 `task_id` 完全一致，写入输入、标注包和预测 CSV 的 SHA-256，并将执行模式明确记为 `HITL`。由于旧生产分类包含照片解释、铁路规则和既往人工决策，不得将其宣称为纯自动算法基线。

两位标注者分别完成文件后，执行：

```powershell
railway-recon benchmark-compare-vertical-reviews `
  --annotation-package benchmarks/local/site-a/annotations/vertical_candidates_blind_v1
```

工具会拒绝未填写完整的标签，按字段计算原始一致率与 Cohen's kappa。完全一致的任务进入 `consensus_seed.csv`，存在分歧的任务进入 `disagreements.csv`，且裁决字段保持为空，必须由第三方复核后填写。
