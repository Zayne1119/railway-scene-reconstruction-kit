# 多块点云输入、接缝与唯一所有权

## 1. 为什么不能直接把多块 LAZ 叠在一起

移动采集项目常把一条走廊拆成多块 LAS/LAZ，并在块边界保留重叠数据。直接合并会让同一钢轨、轨枕、立柱或地面出现两套略有偏差的观测，后续可能形成双轨、重复柱、Z-fighting、闪面或错误拓扑。

本工具包采用 **core owner + overlap context**：

- 每个生产分段只能有一个 `primary_source_id`，只有主源生成该段几何；
- 重叠源写入 `context_source_ids`，只用于接缝检查和证据对照；
- 所有权边界取相邻数据覆盖重叠区的里程中点；
- 没有覆盖重叠、相机覆盖不连续或清单过期时阻断自动分段。

这解决数据源重复消费问题，但不自动证明两块点云已经完成刚性配准。接缝处仍要复核钢轨横向、高程和三维端点差。

## 2. 项目配置

单块点云继续使用：

```json
"inputs": {
  "point_cloud": "input/pointcloud/site.laz",
  "camera_csv": "input/cameras.csv",
  "panorama_root": "input/panoramas"
}
```

多块点云使用：

```json
"inputs": {
  "point_clouds": [
    {"id": "tile-001", "path": "input/pointcloud/tile-001.laz", "priority": 0},
    {"id": "tile-002", "path": "input/pointcloud/tile-002.laz", "priority": 1}
  ],
  "camera_csv": "input/cameras.csv",
  "panorama_root": "input/panoramas"
}
```

`point_cloud` 与 `point_clouds` 必须二选一。`id` 在项目内唯一；`priority` 只用于覆盖起点相同情况下的稳定排序，不代表可信度。

## 3. 标准命令顺序

```powershell
railway-recon validate --project projects/sample_line/project.json
railway-recon audit --project projects/sample_line/project.json
railway-recon prepare-inputs --project projects/sample_line/project.json
railway-recon plan-segments --project projects/sample_line/project.json
railway-recon segment --project projects/sample_line/project.json --ids <pilot-id>
```

生产冻结前重新执行完整哈希：

```powershell
railway-recon audit --project projects/sample_line/project.json --full-hash
railway-recon prepare-inputs --project projects/sample_line/project.json --full-hash
```

`prepare-inputs` 默认用各点云 LAS/LAZ header 的 XY 范围筛选相机。只有明确知道 header 范围与相机轨迹存在小量边界偏差时，才可使用 `--camera-bbox-margin-m`；参数和值必须写入项目记录。

## 4. 生成文件

- `workspace/manifests/input_sources.generated.json`：数据源范围、文件信息、相机覆盖、重叠接缝、core 所有权和状态；
- `workspace/manifests/cameras.filtered.csv`：只保留当前点云集合覆盖的连续相机；
- `workspace/manifests/segments.generated.json`：每段新增 `primary_source_id`、`context_source_ids` 和分配方法；
- `workspace/reports/segment_crop.json`：实际读取的数据源、输出点数和文件大小。

如果原相机 CSV、过滤相机、点云路径、点云文件大小、项目 ID 或清单内容发生变化，后续命令会拒绝旧清单，必须重新运行 `prepare-inputs`。

## 5. 必查验收项

1. `filtered_row_count` 与现场覆盖相符，`coverage_gap_count` 为 0；
2. 每个 source 的 `camera_coverage_gap_count` 为 0；
3. 每处 seam 为 `overlap_context_available`，并有足够重叠长度；
4. 每个 segment 恰好有一个主源；
5. 接缝两侧各裁一个 Pilot，确认输出非空且来源正确；
6. 在钢轨、站台边缘和架空线三个高度层分别检查两源对齐；
7. 未通过接缝几何复核前，不生成跨源 authoritative mesh。

## 6. 当前边界

- header 包围盒只证明空间范围相交，不证明实际表面重叠或配准精度；
- 当前所有权按相机里程分配，适合沿线连续采集，不适合相互交叉或多次折返路线；
- context 数据不会自动混入生产几何；未来接缝优化也应输出显式变换和审计结果，不能静默移动原始点云；
- 没有 CRS、垂直基准和独立控制点时，只能报告内部拟合与接缝一致性，不能声明绝对测量精度。
