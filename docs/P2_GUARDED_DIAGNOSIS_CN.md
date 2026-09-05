# P2 v3：方向基线与有限观测污染防护

本轮是查看 v1/v2 开发、验证与旧压力样例之后的开发修订，不是盲测。原 v1/v2 源码、协议、标签和历史产物保持不变；不读取客户模型，不更改生产门禁，不新增人工签核。

## 方法与对照

保留 `local_geometry`、`topology_only`、`topology_evidence`、`adaptive_evidence` 四个原始对照，新增 `direction_geometry` 和 `guarded_evidence`。六组的几何告警实体、数量及位置必须完全相同；比较的是故障原因判断，不是检出能力提升。

| 新规则 | 输入与决策 | 输出边界 |
| --- | --- | --- |
| 方向几何基线 | 计算当前连接弦与两侧前向端点切线的余弦；两侧均相容时保留缺口解释，否则沿用旧拓扑判断 | 不读取观测来决策；更近的其他线路片段不自动胜过方向连续的当前路径 |
| 缺测降级 | v2 判定当前或替代上下文不足时，改用方向几何基线 | `geometry_fallback`，不是观测验证 |
| 方向矛盾防护 | v2 判当前点桥被支持，但该弦与端点方向不相容且存在旧规则认可的近处替代时，拒绝该支持、回退几何原因 | 不删除告警；不将“识别方向矛盾”宣传成识别所有污染 |
| 其余观测决策 | 保留 v2：当前无支持而替代有支持仍可改变几何原因；真正支持冲突仍拒判 | 观测不是无条件采信，也不是全部被几何覆盖 |

实现见 `src/railway_recon/synthetic_track_guarded_audit.py`。余弦检查直接复用既有 `minimum_direction_cosine`，其他阈值同样不变。没有新增调参得到的置信分数。`evidence_verified` 仅表示内部规则通过，不证明原因正确、点的真实性或测量精度。

方向检查假设连接局部接近直线、片段已按前向排序。真实曲线缺口可能违反该假设；平滑错接可能通过它。没有可辨识线路身份或关系真值时，坐标和无标签点本身不能总是唯一确定正确线路。

## 两套压力样例必须分开报告

1. `legacy`：原 v2 的 18 个样例，默认 seed `20260907`，直接复用旧生成器，逐字保留相同输入与真值。用于检查已知失败是否改善，不作为新独立证据。
2. `challenge`：新的独立解析开发探针，默认 seed `20260908`，覆盖曲线、平滑错接及扰动假支撑，并包含正常控制。生成器不调用审计器来决定标签，不使用客户几何或正式测试布局；这是反例搜索，不是独立现场测试。

不将两套压力样例混合成一个高分掩盖子条件失败。所有条件、误判、拒判与正常误报都保留；图按预先确定的首个样例选取，不按结果挑图。

## 本轮实际结果

以下均为开发材料，不是封存测试或现场精度。新增挑战中 24 项包含 6 个正常控制，正确归因分母为另外 18 个故障。

| 策略 | 开发正确归因 / 280 | 验证正确归因 / 140 | 旧压力 / 18 | 新挑战 / 18 |
| --- | ---: | ---: | ---: | ---: |
| local_geometry | 160 | 80 | 6 | 6 |
| topology_only | 264 | 134 | 17 | 18 |
| topology_evidence | 160 | 80 | 0 | 11 |
| adaptive_evidence | 280 | 140 | 11 | 11 |
| direction_geometry | 280 | 140 | 18 | 6 |
| guarded_evidence | 280 | 140 | 18 | 11 |

开发 200、验证 100、新挑战 6 个正常控制均无误报。旧严格观测在开发/验证分别拒判 120/60 个原因，分母仍包含它们。新防护在这些运行中无原因拒判，但新增挑战有 7 个错误赋因，不能把高覆盖率理解为高正确率。

旧压力下新防护的 18 个结果全部是几何降级推断；其中 6 个是点桥方向矛盾被拦截，不能称为观测已验证。新增挑战下真实圆弧缺口 6/6 正确（没有近端替代，因此方向规则未误伤），平滑错接 5/6，平滑扰动假点桥 0/6。12 个结果满足内部观测支持规则，其中 7 个原因仍错，保留原预测和全部负结果。

这支持有限方向矛盾防护的开发作用，但不支持其全面优于仅拓扑，也未证明观测模块有稳定额外收益。方向基线在原验证集表现更好，但在新解析布局上更差。路线身份是构造真值中的潜变量，并未作为输入泄漏给检测器；部分歧义不能仅由几何和无标签点唯一消除。

核验：624 项 pytest、343 项 unittest、Ruff、smoke 与安全扫描通过。四个新运行目录 6,157 份产物及各 192 份来源文件哈希通过；v3 冻结 234 份文件通过。旧四策略在同数据重放中的逐例预测与指标完全保持，原开发/验证指纹及旧压力指纹均一致。

## 可复现入口

在工具包根目录运行；每次 `--output` 必须是新路径，已存在的结果不会覆盖。

```powershell
.\.venv\Scripts\python.exe scripts/run_synthetic_track_guarded_study.py init --base-protocol benchmarks/protocols/synthetic-track-study-v1.json --output benchmarks/protocols/synthetic-track-guarded-v3.json
.\.venv\Scripts\python.exe scripts/run_synthetic_track_guarded_study.py run --protocol benchmarks/protocols/synthetic-track-guarded-v3.json --split development --output benchmarks/local/paper-topology-p2/development-v3
.\.venv\Scripts\python.exe scripts/run_synthetic_track_guarded_study.py run --protocol benchmarks/protocols/synthetic-track-guarded-v3.json --split validation --output benchmarks/local/paper-topology-p2/validation-v3
.\.venv\Scripts\python.exe scripts/run_synthetic_track_guarded_study.py stress --suite legacy --output benchmarks/local/paper-topology-p2/stress-legacy-v3
.\.venv\Scripts\python.exe scripts/run_synthetic_track_guarded_study.py stress --suite challenge --output benchmarks/local/paper-topology-p2/stress-challenge-v3
.\.venv\Scripts\python.exe scripts/run_synthetic_track_guarded_study.py freeze --protocol benchmarks/protocols/synthetic-track-guarded-v3.json --output benchmarks/local/paper-topology-p2/source-freeze-v3.json
```

完整结果见各目录的 `report.json` / `report.md`、逐例表、全部失败、隔离输入/真值/预测、定位图和哈希清单。开发/验证必须复用 v1 的布局顺序和数据指纹。旧冻结只证明历史绑定文件未变，不能授权新增 v3 方法；新冻结也不等于外部预注册或证明未看过测试。

本轮不执行 `--split test`。只有先固定比较方法及统计口径，再通过新源码绑定校验，才进入后续正式测试。
