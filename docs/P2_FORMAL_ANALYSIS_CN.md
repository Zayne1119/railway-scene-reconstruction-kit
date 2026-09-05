# P2 固定布局配对统计

本阶段只分析已经完成的 v3 六策略运行，不调用场景生成器，不重新执行诊断器，
不修改旧协议、旧算法或旧结果。开发集、验证集、两套压力案例已用于开发；这件事
保留在计划中，不把重跑称为盲测。60 个保留测试布局应在方法和本计划都锁定后只作
首次正式运行，结果无论正负都保留，不按测试结果再调参。

## 固定终点

主比较为 `guarded_evidence - direction_geometry`。先对每个布局计算全部缺陷真值中
正确归因的比例，再对布局差值等权求均值；相关的 12 个变体不能当成独立样本。
弃权仍计入缺陷分母，属于归因 FN。其他四个比较均为探索性次要比较，不能挑最高
增益代替主比较。所有正常案例报警、所有弃权、全部条件、全部布局均保留。

重采样单位是完整布局，六策略及布局内全部条件一起抽取；NumPy PCG64 固定种子
`947320260909`，10,000 次有放回重采样，2.5% / 97.5% 线性分位数区间。
不输出 p 值，不自动宣布优势或等效。若所有观察差值都相同，退化区间只意味着当前
经验分布不变，不意味着总体必然正确、零风险或已经证明方法等效。该区间描述注册
合成设计混合的布局变异，不是实际铁路总体抽样，不支持现场准确率外推。

六个策略都是内部对照，不冒充外部 SOTA。公共点云分割和这里的缺陷归因不是同一
终点，不混为一项优胜比较。

## 注册与运行

先完成代码验证，再创建一次计划（记录 UTC 时间、v3 协议原始字节 SHA、分析 CLI
与方法/评分源文件 SHA）。计划自检会拒绝之后改变分析规则或代码。时间戳与本地
哈希不是外部预注册，也不能证明历史上从未看过测试集。

```powershell
.\.venv\Scripts\python.exe scripts/analyze_synthetic_track_formal.py register --protocol benchmarks/protocols/synthetic-track-guarded-v3.json --output benchmarks/protocols/synthetic-track-formal-analysis-v2.json
.\.venv\Scripts\python.exe scripts/analyze_synthetic_track_formal.py analyze --run-directory benchmarks/local/paper-topology-p2/development-v3 --plan benchmarks/protocols/synthetic-track-formal-analysis-v2.json --output benchmarks/local/paper-topology-p2/analysis-development-v2
```

正式 test 之前还需建立新的 v3 source freeze，保留计划自身 SHA。原有 freeze 会绑定
完整 `src`（包括新增分析实现），但不会自动包含这份新增计划或新增 CLI；因此计划
自身同时绑定 CLI，运行后的 analysis manifest 再绑定计划，不声称旧 freeze 已覆盖
所有新增文件。测试运行缺少本分析模块的源文件绑定，会拒绝正式分析。

分析入口先检查完整 manifest 的文件集合、每个字节数与 SHA、无失败标记；再按
case index 逐一读取所有 prediction/truth 重算评分，核对数据和预测总指纹、协议与
策略、全部组/条件及六策略数量，拒绝删组、漏病例、改 headline 或多余文件。输出
位于旧 run 目录外且拒绝覆盖；报告包含布局率、主次比较、全条件结果与局限说明。
这种完整性检查不等于第三方见证或防止蓄意同时伪造全套文件。

## 正式测试前的绑定修订

v1 计划与开发/验证统计输出保留原样，标记为已被替代的完整性验收记录。独立复核
在任何保留测试生成之前发现：直接模块列表漏了 `relationships.py` 等真实依赖。
v2 改为绑定 v3 协议声明的完整 `REQUIRED_EXECUTION_FILES` 与分析源文件的并集；
统计规则、所有方法、评分器、布局分配均未改变。辅助关系代码或 guarded CLI 的
模拟字节变更均有拒绝测试。较早两版 runner CLI 虽被计划完整绑定，但不在 guarded
运行的自身执行清单中；不会错误声称这两个未执行 CLI 被该运行 manifest 单独记录。
