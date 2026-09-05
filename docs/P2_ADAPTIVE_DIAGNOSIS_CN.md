# P2 v2：低观测自动降级与归因来源

v1 的观测策略在证据缺失时保留几何告警但拒绝归因，导致整体诊断召回下降。v2 保留前三组对照，新增 `adaptive_evidence`；不修改 v1 代码、协议、标签或历史结果，不改变生产门禁，不建立额外人工签核队列。

## 决策规则

| 当前证据状况 | 自适应行为 | 输出含义 |
| --- | --- | --- |
| 当前端点邻域缺失或过稀 | 保留仅拓扑的原因推断 | `geometry_fallback`，未获观测支持 |
| 当前路径有足够观测支持 | 保留当前连接的断缝解释，不采纳邻近目标误导 | `evidence_supported` |
| 当前路径缺少支持，某替代路径及其邻域有支持 | 推断错接 | `evidence_supported` |
| 替代路径邻域不可观察 | 保留拓扑推断，不把缺测当反证 | `geometry_fallback` |
| 存在替代候选，当前及全部替代路径均不受支持，但端点邻域覆盖足够 | 保留告警，原因暂不确定 | `evidence_conflict`，仍计诊断 FN |

重复片段继续由几何检查处理。四组共用告警实体与位置，没有自动修复连接，也不是新的故障检出算法。

`cause_status=assigned` 表示提供了一个可评价预测，不等于预测正确。`confidence_level` 是 `inferred_geometry`、`observation_supported` 或待定原因等**定性来源标签**，不是校准概率或实测精度。

`evidence_verified=true` 只表示内部覆盖/路径支持条件满足，不表示已核实真实故障原因。假点、杂波或未建模的遮挡仍可能误导判断，不能把此字段当作安全验收认证。

## 可重复的四组对照

v2 内嵌原 v1 协议，保持 40/20/60 基础布局、全部种子、12 条件及数值阈值不变；前三组原始分类保持不变。新增方法已经参考过 v1 开发/验证结果，必须标为开发修订，不能冒充新的独立验证集成绩。正式测试仍未生成或查看。

主指标继续为包含拒判的整体诊断召回。报告同时列出精确率/F1、诊断覆盖、按决策来源分层的错误和拒判。降级预测照常计入正确/错误，不能因“低置信”而将错误移出分母；重复告警仍按一对一规则计分。

使用 `scripts/run_synthetic_track_adaptive_study.py` 的 `init`、`run`、`freeze` 子命令，具体参数以 `--help` 为准。每次输出必须是新目录，正式 test 需要与实际执行代码匹配的新冻结。旧冻结的绑定字节可复查，但新增源码后不能拿 v1 冻结授权 v2 执行。冻结不是对论文发表或性能结论的批准。

```powershell
python scripts/run_synthetic_track_adaptive_study.py init --base-protocol benchmarks/protocols/synthetic-track-study-v1.json --output benchmarks/protocols/synthetic-track-adaptive-v2.json
python scripts/run_synthetic_track_adaptive_study.py run --protocol benchmarks/protocols/synthetic-track-adaptive-v2.json --split development --output benchmarks/local/paper-topology-p2/development-v2
python scripts/run_synthetic_track_adaptive_study.py run --protocol benchmarks/protocols/synthetic-track-adaptive-v2.json --split validation --output benchmarks/local/paper-topology-p2/validation-v2
python scripts/run_synthetic_track_adaptive_stress.py --output benchmarks/local/paper-topology-p2/stress-v2
python scripts/run_synthetic_track_adaptive_study.py freeze --protocol benchmarks/protocols/synthetic-track-adaptive-v2.json --output benchmarks/local/paper-topology-p2/source-freeze-v2.json
```

这些路径是本次已实际执行的输出；复用协议时跳过 init，重新执行请选择新的输出目录。

## 对抗性开发探针

额外提供 6 个独立参数布局、每组 3 条件，共 18 个开发压力例：

- 真实缺口旁有替代候选但完全无观测：检查降级会不会给出错误且未经观测支持的归因。
- 错接路径加入人工假点桥：检查正向观测支持能否被污染。
- 错接仅保留当前源/目标端点邻域，正确替代路径上下文被遮挡：检查缺测是否被错误解释为反证。

探针使用与注册集合不相交的负数派生种子，既不是保留测试集，也不是新的现场数据。标签继承实际几何/连接修改，不调用检测器决定标签；失败必须保留。探针不与原开发/验证主表混算，不能当作跨现场泛化证据。

## 使用边界

2026-09-05 的实际回放中，v2 在开发集为 280/280 正确归因、验证集为 140/140，原三组结果与 v1 完全复现。验证集新增的 60 个缺测赋因均标为 `geometry_fallback`，没有伪装成观测支持。然而单列的 18 个压力探针中 v2 仅 11/18 正确，低于仅拓扑的 17/18；6 个假点桥全部误导了观测策略。必须将这个负结果与主表一起解释，不能只报告原验证集高分。

本轮只改研究模块，没有使用客户数据、改动客户模型或生产网页。自动降级减少不必要的拒判，但不会把来源不明的推断升级为真实观测。下一阶段应在冻结方法后使用保留测试，并继续推进公开点云中标签能支持的评价和外部基线比较。
