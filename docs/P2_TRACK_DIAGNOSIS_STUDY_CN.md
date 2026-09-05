# P2 三组轨道归因实验与协议冻结

本入口分离局部几何、拓扑候选、观测支持，加入困难正常例和观测缺失。它研究的是**同一几何告警的原因判断**，不把相同故障检出率写成观测模块增益。P1 的代码与历史结果未覆盖，生产模型和网页没有因此变更。

## 自动运行

在工具包根目录，以已安装项目依赖的 Python 执行；所有输出必须是新路径。已有协议可直接使用，不必再次 init。

```powershell
python scripts/run_synthetic_track_study.py init --output benchmarks/protocols/synthetic-track-study-v1.json
python scripts/run_synthetic_track_study.py run --protocol benchmarks/protocols/synthetic-track-study-v1.json --split development --output benchmarks/local/paper-topology-p2/development-v1
python scripts/run_synthetic_track_study.py run --protocol benchmarks/protocols/synthetic-track-study-v1.json --split validation --output benchmarks/local/paper-topology-p2/validation-v1
python scripts/run_synthetic_track_study.py freeze --protocol benchmarks/protocols/synthetic-track-study-v1.json --output benchmarks/local/paper-topology-p2/source-freeze-v1.json
```

重新试验须换输出路径，不能覆盖旧结果。默认登记 120 个基础布局：40 development、20 validation、60 test。每个布局有 12 种派生条件，同一布局的所有条件都在同一集合。test 本阶段仅登记独立种子，不生成数据或预览。

`run` 必须显式指定 split；无有效冻结的 test 在生成数据和建立输出目录前拒绝。冻结绑定整个源码包及资源、CLI、依赖锁和协议；验证仓库必须与实际执行工具包一致，防止校验副本 A 却执行副本 B。冻结是内容一致性检查，不是外部预注册，也不能证明有人从未查看保留种子。

## 方法与拒判

| 模式 | 归因行为 |
| --- | --- |
| local_geometry | 根据重测几何报告端点分离或重叠 |
| topology_only | 存在方向与距离合理的替代目标时推断错接 |
| topology_evidence | 核查当前/替代路径与端点邻域的观测支持，决定是否采纳错接归因 |

三组共用 P1 几何触警结果，不增删告警实体。观测不足时 `cause_status=abstain`，几何告警仍保留。此时 `kind=gap` 只代表粗粒度端点分离，评价器不能将其算作正确断轨判断。邻域支持也不能排除所有中部遮挡或错误回波，这不是完整传感器模型。

本版本预先沿用试跑工作点，不按验证分数反复改阈值。参数写入协议与每份预测。允许负结果，尤其是拒判带来的整体诊断召回下降；不要求观测方法一定胜出。

## 12 种条件

- 基础正常、断缝、错股道连接、重复片段。
- 正常浅角交叉、同 XY 但不同高度的立交、小幅端点抖动。
- 正常无观测、断缝无观测、错接无观测、错接稀疏观测。
- 真实断缝旁存在独立浅角穿越线路：主线缺口保留，旁线片段可能成为几何替代候选。

旁线不是距离仅十厘米的两股平行轨道。条件按通用几何分布生成，不根据检测结果反推标签或选种子；并非每个混淆例都会触发替代候选规则。这不是严格几何匹配的因果对照，不能单凭该条件优势证明任意场景中的观测贡献。

正常条件的范围写在每例真值参数中；并未覆盖所有现场负例或道岔。观测均为带噪数学中心线采样，不是钢轨表面或真实 LiDAR。

## 评价与产物

主指标为包含拒判的整体诊断召回；同时报告诊断精确率/F1、覆盖率、已归因部分正确率和原因混淆矩阵。Detection 沿用 P1 的精确实体集合、一对一匹配；Diagnosis 还要求原因正确。错归因产生一个诊断 FP 和一个 FN，不等于额外几何误报。

拒判仍计诊断 FN，不能只摘取选择性正确率。重复告警不增加真值覆盖；诊断匹配索引回映射到原始完整预测。按基础布局配对汇总，派生病例不是独立现场；开发/验证结果不作确认性显著性声明。

| 产物 | 内容 |
| --- | --- |
| `inputs/`、`ground_truth/`、`predictions/` | 隔离的纯检测输入、评价真值与三组预测 |
| `case_index.json`、`case_results.csv` | 全部病例/布局/条件索引与统计 |
| `report.json`、`report.md` | 主表、分层、布局配对差、混淆与拒判 |
| `failure_cases.json` | 全部误报、漏报、误归因和拒判记录 |
| `figures/` | 预先排列的第一组中全部条件示例，不按结果挑图 |
| `timings.json`、`manifest.json` | 实际耗时、源码/产物哈希、依赖版本与冻结状态 |

图中红叉包括未可靠归因的几何告警；绿圈是真值，仅在检测后用于评价和绘图。完整输出默认放在 Git 忽略目录，不直接公开整个研究工作区。

## 公开数据与下一步

[公开 LAS 字段适配](PUBLIC_LAS_INTAKE_CN.md) 将真实头部、分类数值及 source ID 转为字段报告，输入前后均校验 SHA256。一个训练样本的字段适配不等于公开数据方法实验；语义/实例字段也不直接充当连接真值。

下一阶段运行冻结后的合成保留测试，并补公开点云候选提取及标签能支持的评价。外部方法基线、阈值曲线和论文主张仍需后续实验；内部三组消融本身不足以证明创新性。这里的科研完整性检查不恢复生产端逐图人工签核。
