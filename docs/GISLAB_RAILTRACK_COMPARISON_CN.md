# GISLab RailTrack 外部基线对比

## 结论

在 Site A 的四个 50 m 空间块、固定 80/20 体素级 train/holdout 协议下：

- 本项目方法与 Open3D 可以完成全部四段几何评测；8 项聚合指标中，本项目方法 7 项较优，Open3D 的垂向重复性 P90 较优。
- 未修改的 GISLab RailTrack 在 8 个 train/holdout 用例中产生 **7 个零检出、1 个超时、0 个轨道几何输出**。
- 因为 GISLab 没有输出可匹配几何，它的几何误差、重复性和覆盖率必须记为 **N/A**，不能用 0 代替，也不能参加“7 胜 1 负”的数值排名。

这说明 GISLab RailTrack 在当前密集站场数据和固定运行协议下不适配；它不证明该方法在所有铁路点云上都无效。

## 冻结协议

| 项目 | 固定设置 |
|---|---|
| 数据 | 同一 Site A 点云；四个 50 m 空间块 |
| 切分 | 0.5 m 空间体素级 80/20；train 94,683,812 点，holdout 23,156,391 点 |
| 评测 | train/holdout 独立检测；分割重复性与原始留出点支撑 |
| GISLab 仓库 | `GISLab-ELTE/railroad` |
| GISLab 提交 | `3557ecff1bd284108c7a833cce2b50ec9fcd5189` |
| LAStools 提交 | `9bdc92c73047b46be25e5c2ed4abda2521e30fba` |
| 环境 | Ubuntu 22.04；PCL/OpenCV/Boost/PROJ；未修改上游源码 |
| 容器 | `railway/gislab-railtrack:3557ecf` |
| 镜像摘要 | `sha256:35c0b646dc668d61cd606e11e1d815522fff3ffeb301db0672175ee61a39dbbf` |
| 单用例上限 | 600 s |

共享预处理只做三件事：读取冻结走廊坐标框架、裁剪同一纵向范围与轨头高程包络、通过刚体旋转把走廊纵向对齐为局部 X 轴。它没有向 GISLab 提供轨道位置、轨对、语义标签、最终模型、标准轨距修正或人工种子。

原始世界坐标运行也被保留。该运行中上游方向启发式选择错误轴；局部坐标复验消除了这一坐标约定差异，但最终仍未输出轨道。因此报告的是更公平的局部坐标结果。

## 三方法结果

下表中的“召回/精度”是 train/holdout 候选线匹配指标，不是有标注真值下的语义 Precision/Recall。

| 指标 | Open3D RANSAC + DBSCAN | 本项目轨头高度显著性 | GISLab RailTrack |
|---|---:|---:|---:|
| train 候选线匹配召回 | 57.69% | **66.67%** | N/A |
| holdout 候选线匹配精度 | 39.47% | **100.00%** | N/A |
| 横向重复性 P90 | 68.65 mm | **24.62 mm** | N/A |
| 垂向重复性 P90 | **30.12 mm** | 56.32 mm | N/A |
| 原始轨距重复性 P90 | 122.47 mm | **23.65 mm** | N/A |
| 模型到留出点支撑 P90 | 13.628 m | **69.33 mm** | N/A |
| 50 mm 内留出点支撑率 | 63.22% | **83.47%** | N/A |
| 100 mm 内留出点支撑率 | 66.48% | **95.03%** | N/A |

本项目与 Open3D 的成对聚合比较为 7:1。四个独立空间块仍不足以形成确认性统计结论；现有 block bootstrap 被明确标为 `exploratory_insufficient_independent_blocks`。

## GISLab 执行结果

| 结果 | 用例数 |
|---|---:|
| 正常输出轨道几何 | 0 |
| 零检出后上游空结果异常退出 | 7 |
| 达到资源上限、无输出 | 1 |
| 合计 | 8 |

上游可执行程序累计运行 833.77 s，中位数 5.19 s；该时间不包含共享预处理。一个 train 块运行 789.07 s 后被停止并记录为超时，其他失败大多很快进入零候选或零轨对分支。

日志中保留了两种零检出证据：

1. `railH95 size: 0` 且 `cvHough size: 0`，说明轨高候选和 Hough 线均为空；
2. `Pairs: 0` 且过滤结果为 `-> 0`，说明没有通过上游轨距配对的轨对。

上游随后在处理空诊断图时以 134 或 139 退出。适配器只把已由日志证明的零检出转换为显式空报告，没有生成替代轨线。超时同样记录为无几何非结果。

## 可复现运行

构建环境后执行：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass `
  -File scripts/run_gislab_holdout_experiment.ps1 `
  -OutputName gislab_railtrack_v1_frozen_local

.\.venv\Scripts\python.exe scripts/summarize_external_rail_baselines.py `
  --ours benchmarks/local/site-a/reports/rail-holdout-v2.json `
  --open3d benchmarks/local/site-a/reports/rail-holdout-open3d-v1.json `
  --gislab benchmarks/local/site-a/reports/rail-holdout-gislab-v1.json `
  --gislab-baseline-root benchmarks/local/site-a/baselines/gislab_railtrack_v1_frozen_local `
  --output benchmarks/local/site-a/reports/rail-holdout-three-method-v1.json `
  --csv benchmarks/local/site-a/reports/rail-holdout-three-method-v1.csv
```

关键产物：

- `reports/rail-holdout-gislab-v1.json`：统一评测器生成的 GISLab 空几何报告；
- `reports/rail-holdout-three-method-v1.json`：三方法机器可读汇总；
- `reports/rail-holdout-three-method-v1.csv`：论文表格输入；
- `baselines/gislab_railtrack_v1_frozen_local/raw/`：每个用例的命令、输入哈希、退出码、耗时和完整日志；
- `benchmarks/external/gislab-railtrack/`：冻结 Docker 构建定义和上游来源说明。

`benchmarks/local/` 默认不进入公开仓库，因为其中包含项目数据和本地绝对路径；公开仓库只提交复现实验代码、容器定义和不泄露数据的汇总文档。

## 论文使用边界

可以写：在固定 Site A 无标注空间留出协议下，本方法相对 Open3D 获得更好的横向、轨距和原始点支撑重复性；未修改 GISLab RailTrack 未能在该协议中产生几何输出。

不能写：本方法已经达到绝对测量精度、语义检测 Precision/Recall、跨场景泛化，或普遍优于 GISLab。第二场景和独立真值到来前，本结果只能作为单场景探索性证据。
