# 多相机投影共识与多 LAZ 接缝验收

## 1. 目的

这两个门禁解决两类容易在最终模型中表现为“错位、双轨、断轨和闪面”的上游问题：

1. 单张全景可能因为重复颜色、遮挡或采集设备入镜，选出错误的 Euler 顺序或轴约定；
2. 相邻 LAZ 的包围盒相交，不代表两块数据在钢轨表面上存在足够共同观测。

生产规则是：一个走廊段只能使用一个共享投影约定；一个生产分段只能有一个几何 owner。相邻数据源只提供接缝证据，不能重复生成 Mesh。

## 2. 多相机投影共识

至少选择同一场景中 3–5 张全景。每张相机仍使用自己的位置和姿态数值，但 Euler 顺序、姿态方向和局部轴排列必须全段一致。

```powershell
railway-recon calibrate-projection-consensus `
  --project projects/site/project.json `
  --sample s0100_0150m=101 `
  --sample s0100_0150m=103 `
  --sample s0100_0150m=105 `
  --output-name station-core
```

聚合首先最小化各相机候选排名的中位数，再比较平均排名、RGB 中位误差和最大误差。输出包括：

- 唯一共享约定；
- 每张相机的局部最优与共享最优；
- 局部最优是否与共识一致；
- 每张相机在共享约定下的点云—全景叠加图；
- 参数文件哈希与运行清单。

数值最优仍不是姿态真值。必须检查钢轨、站台边缘、柱、檐口和接触网等结构边缘是否同时对齐。

## 3. 相同窗口的多源裁剪

先由 `prepare-inputs` 和 `plan-segments` 生成唯一 owner 及 `context_source_ids`。只对接缝窗口运行：

```powershell
railway-recon segment-context `
  --project projects/site/project.json `
  --segment s1500_1525m
```

命令会用完全相同的三维包围盒，从该段声明的每个 context LAZ 各裁一份证据文件。输出状态固定为 `evidence_only_no_duplicate_mesh_authority`，这些文件不能进入生产 Mesh 合并。

如果 50 m 窗口内共同钢轨支撑太短，应在项目配置中增加自适应短分段，使窗口对应真实共同表面长度，再重新执行裁剪和轨头检测。

## 4. 轨头直接对照

用相同 segment ID、相同冻结参数分别处理左右 context 文件：

```powershell
railway-recon detect-rails --project projects/site/project.json `
  --segment s1500_1525m --source left-context.laz `
  --settings rail_detection.frozen.json --report left-rails.json

railway-recon detect-rails --project projects/site/project.json `
  --segment s1500_1525m --source right-context.laz `
  --settings rail_detection.frozen.json --report right-rails.json

railway-recon seam-rail-compare --project projects/site/project.json `
  --left-report left-rails.json --right-report right-rails.json `
  --output seam-comparison.json
```

比较在同一个走廊局部坐标系内完成，检查轨道中心横向差、轨顶差和实测轨距差。阈值来自冻结的 TrackGraph 质量配置。

状态解释：

- `pass`：两侧轨道对数一致、身份一一匹配且所有误差通过；
- `fail`：两侧都有轨道证据，但数量、身份或误差不通过；
- `insufficient_shared_rail_support`：至少一侧没有足够轨头观测。该状态不是物理断轨证据，必须阻止自动补线并转入相邻 owner 段拓扑复核。

## 5. 与 TrackGraph 的关系

直接共同窗口对照回答“两个数据源是否观测到同一轨头”；相邻 owner 段 TrackGraph 回答“生产轨道身份能否跨所有权边界连续”。二者不可互相替代：

1. 直接对照通过，仍需 TrackGraph 检查轨道数量、顺序、端点和拓扑；
2. 直接对照证据不足时，TrackGraph 可以保留候选身份，但不得把推断冒充共同观测；
3. TrackGraph 只有在 source review、观测支撑、内部终止和 segment seam 等门禁全部通过后，才能授权跨段 Mesh；
4. 自动端点调和必须记录原始差值、修正量和阈值，不能只输出修正后的零接缝。

## 6. 精度声明

上述毫米或厘米结果只表示当前点云坐标内的跨源内部一致性。没有 EPSG、垂直基准和独立控制点时，不得称为绝对测量精度、运维精度或施工图精度。
