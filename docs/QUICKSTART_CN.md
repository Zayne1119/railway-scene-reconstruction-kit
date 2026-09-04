# 快速上手：从空目录到第一个 Pilot

本页面向第一次接手项目的同事。完成后，你应得到一个通过配置校验的数据安全项目目录、一份输入审计报告、一份走廊分段计划和一个 20–50 m Pilot 点云。

## 1. 准备环境

推荐环境：

- Windows 10/11；
- Python 3.11；
- PowerShell 7 或 Windows PowerShell；
- 后续需要模型导出时安装团队指定版本的 Blender；
- 后续需要 Web 页面时安装仓库规定版本的 Node.js。

在仓库根目录执行：

```powershell
.\scripts\bootstrap.ps1
.\.venv\Scripts\Activate.ps1
railway-recon --version
```

如果 PowerShell 禁止激活脚本，可直接使用虚拟环境中的 Python 调用：

```powershell
.\.venv\Scripts\python.exe -m railway_recon --help
```

## 2. 跑自动测试

```powershell
python scripts\run_tests.py
```

安装了开发依赖时，也可以运行 `python -m pytest` 获得同一组测试结果。

测试失败时先停止，不要开始生产项目。把完整错误、Python 版本和当前 Git commit 记录到问题单。

## 3. 建立项目骨架

项目 ID 只能包含小写字母、数字、连字符或下划线，并以小写字母开头。

```powershell
railway-recon init projects/sample_line --name "Sample Railway"
```

生成结构：

```text
projects/sample_line/
├─ project.json
├─ rules.json
├─ rail_detection.json
├─ linear_detection.json
├─ track_build.json
├─ projection_calibration.json
├─ input/
│  ├─ pointcloud/
│  └─ panoramas/
└─ workspace/
   ├─ manifests/
   ├─ segments/
   ├─ derived/
   ├─ registry/
   ├─ reports/
   └─ exports/
```

`input/` 和 `workspace/` 都不应提交到 GitHub。

## 4. 放置输入

按默认模板放置：

```text
input/pointcloud/site.laz
input/cameras.csv
input/panoramas/<image-files>
```

相机 CSV 至少包含：

```csv
index,timestamp,file,x,y,z,rot_x,rot_y,rot_z
```

其中 `rot_x/rot_y/rot_z` 在基础审计中可为空，但如果要做照片投影则必须补齐，并明确角度单位、欧拉顺序和旋转方向。不要用真实数据覆盖 `configs/templates/camera.example.csv`。

如果现场交付的是连续的多块 LAS/LAZ，不要先手工合并。将单文件配置：

```json
"point_cloud": "input/pointcloud/site.laz"
```

替换为：

```json
"point_clouds": [
  {"id": "tile-001", "path": "input/pointcloud/tile-001.laz", "priority": 0},
  {"id": "tile-002", "path": "input/pointcloud/tile-002.laz", "priority": 1}
]
```

多块点云项目在输入审计后、规划分段前必须运行：

```powershell
railway-recon prepare-inputs --project projects/sample_line/project.json
```

只有生成清单的 `status` 为 `ready_for_segment_planning` 才能继续。详细规则参见
[多块点云输入、接缝与唯一所有权](MULTI_SOURCE_INPUT_CN.md)。

## 5. 检查项目配置

打开 `project.json`，先核对：

- 输入相对路径；
- `project.id`；
- 保密等级；
- 单位和 Z-up 约定；
- CRS、垂直基准、控制点是否已确认；
- 分段长度、走廊半宽和高程边界；
- Pilot 长度；
- 未观测立面策略；
- QA 阈值。

然后运行：

```powershell
railway-recon validate --project projects/sample_line/project.json
```

只有输出 `"valid": true` 才能继续。

## 6. 审计输入

```powershell
railway-recon audit --project projects/sample_line/project.json
```

查看：

```text
projects/sample_line/workspace/reports/input_audit.json
```

至少确认：

- 点云文件是 LAS/LAZ；
- 点数和空间范围合理；
- 相机索引、时间戳和文件引用无重复；
- 全景缺失数量已知；
- 轨迹长度符合现场认知；
- 报告未把点式观测误写成原始高斯属性。

冻结生产基线前再运行完整哈希：

```powershell
railway-recon audit --project projects/sample_line/project.json --full-hash
```

大文件完整哈希会耗时，日常调试无需每次执行。

## 7. 规划走廊分段

```powershell
railway-recon plan-segments --project projects/sample_line/project.json
```

打开 `workspace/manifests/segments.generated.json`。逐段检查：

- 长度是否约为配置值；
- 相机数量是否足够；
- 包围盒是否覆盖线路、站台和需要建模的线路外侧资产；
- 是否误包含大面积无关区域；
- 端部是否需要接缝冗余。

自动规划采用相机轨迹外包络，不是最终铁路限界。未经人工检查不得批量裁切全线。

## 8. 裁切 20–50 m Pilot

从清单中选择一个有代表性的段：

```powershell
railway-recon segment --project projects/sample_line/project.json --ids <segment-id>
```

输出位于 `workspace/segments/`。Pilot 选择优先级：

1. 钢轨清晰且有完整轨迹覆盖；
2. 同时包含至少一种竖直资产；
3. 最好包含一个遮挡或结构交界难点；
4. 不要只选最干净、最简单的直线段。

已存在输出时命令默认拒绝覆盖。确需重跑：先归档旧 manifest 和报告，再显式使用 `--overwrite`。

## 9. 初始化资产注册表

```powershell
railway-recon registry-init --project projects/sample_line/project.json
railway-recon registry-check --project projects/sample_line/project.json
```

后续每生成一个资产，都要写入唯一 ID、类型、状态、证据等级、置信度和来源。参见 [资产注册表规范](ASSET_REGISTRY_CN.md)。

人工复核结果可按 `examples/minimal_project/manual_review.example.csv` 填写并导入：

```powershell
railway-recon registry-import --project projects/sample_line/project.json --source <review.csv>
```

已存在的资产 ID 默认拒绝覆盖；只有确认这是同一资产的新复核结论时，才能显式使用 `--replace-existing-ids`。

## 10. 运行基础 QA

```powershell
railway-recon qa --project projects/sample_line/project.json
```

基础 QA 只确认输入、审计、分段清单和注册表等工程条件。它不代表轨道、站台或接触网已经建模正确。几何、Mesh、UE 和 Web 验收需要继续执行 [质量验收规范](QA_ACCEPTANCE_CN.md)。

## 11. 运行 Pilot 基线建模

先用点云生成几何候选。候选不是最终资产，必须结合全景和人工复核：

```powershell
railway-recon detect-rails --project projects/sample_line/project.json --segment <segment-id>
railway-recon detect-linear --project projects/sample_line/project.json --segment <segment-id>
railway-recon classify-vertical --project projects/sample_line/project.json --segment <segment-id>
railway-recon analyze-canopy --project projects/sample_line/project.json --segment <segment-id>
railway-recon canopy-photo-evidence --project projects/sample_line/project.json --segment <segment-id> --projection-consensus <consensus.json>
railway-recon analyze-platform --project projects/sample_line/project.json --segment <segment-id>
railway-recon platform-photo-evidence --project projects/sample_line/project.json --segment <segment-id> --projection-consensus <consensus.json>
railway-recon build-platform-mesh --project projects/sample_line/project.json --segment <segment-id> --mesh-gate <platform_mesh_gate.json>
railway-recon audit-platform-interfaces --project projects/sample_line/project.json --segment <segment-id>
railway-recon vertical-conflict-photo-evidence --project projects/sample_line/project.json --segment <segment-id> --projection-consensus <consensus.json>
```

若相机 CSV 包含旋转字段，可先对代表性相机搜索姿态约定并生成点—全景叠加图：

```powershell
railway-recon calibrate-projection --project projects/sample_line/project.json --segment <segment-id> --camera-index <camera-index>
```

RGB 误差最小只代表候选假设。必须至少检查多个相机、钢轨边缘、柱线和屋檐等结构线一致后，才能冻结投影约定。

检查 `workspace/reports/` 中的候选 JSON 和诊断图，确认钢轨配对、接触网/雨棚柱候选与高空线性候选。若高度分位区间或覆盖阈值不合适，应修改项目目录中的算法配置并记录原因，不得把现场阈值写回公共源码。

单段 `build-track` 只用于 Pilot 观察，不得直接拼成全线模型。推荐先生成哈希绑定的独立复核包：

```powershell
railway-recon track-review-create `
  --project projects/sample_line/project.json `
  --source <segment-id>=<rail-candidate-report.json>

railway-recon track-review-apply `
  --project projects/sample_line/project.json `
  --path projects/sample_line/workspace/reports/track_graph_rail_review_v1
```

复核通过后，用 reviewed 报告构建全局 TrackGraph；只有审计为 `pass` 才能生成连续钢轨、全局相位轨枕和道床：

```powershell
railway-recon track-graph-build `
  --project projects/sample_line/project.json `
  --source <segment-id>=<reviewed-rail-report.json>

railway-recon build-track-graph-mesh --project projects/sample_line/project.json
railway-recon registry-check --project projects/sample_line/project.json
```

输出的 OBJ 仍是候选模型，不代表轨道已完成。仍需复核轨距、钢轨中心距、轨顶高程、段界、道岔、轨枕方向、道床肩部和点—模型误差。之后的推荐顺序是：相机投影与照片证据 → 竖直构件语义分类 → 接触网 → 站台/雨棚/可见立面 → 缺口闭环 → Mesh 清理和 Web/Blender/UE 验收。

对于已经由点云/照片和人工复核确定的站台、柱、梁、屋面段、立面板或线缆，可以按照 `examples/minimal_project/reviewed_scene.example.json` 建立布局并生成统一 OBJ：

```powershell
railway-recon build-reviewed-scene --project projects/sample_line/project.json --layout <reviewed_scene.json>
railway-recon mesh-audit --path projects/sample_line/workspace/exports/reviewed_scene/reviewed_scene.obj
```

该命令只负责把**已复核布局**稳定地转为模型和注册资产，不会替代照片判读或专业确认。`extruded_polygon_xy` 适合站台体块，`box` 适合柱/箱柜，`beam` 适合梁，`panel` 适合分段屋面/立面，`tube` 适合接触网线索。复杂凹多边形、道岔和真实型材仍需专用模块。

当前版本未内置的专业模块，不要从历史项目直接复制带现场坐标和 `_vNN` 的脚本。先参数化、补测试，再通过 Pull Request 合入通用代码。

## 12. 上传前检查

```powershell
railway-recon safety-check --root .
```

只有 `passed` 为真才允许推送。仍需人工确认没有：原始点云、全景、精确坐标、设备台账、密钥、本机路径、客户名称和未授权模型。

## 13. 常见问题

### `validate` 失败

按输出字段路径修改 `project.json`。不要删除必填字段来绕过 Schema。

### `audit` 找不到照片

检查 CSV 中 `file` 使用的是相对于 `panorama_root` 的路径，或文件名是否与实际大小写一致。

### 分段边界太宽或太窄

调整 `corridor_half_width_m`、上下高程缓冲后重新 `plan-segments`，再人工查看新清单。

### 可以报告厘米级精度吗

只有内部拟合指标不能。缺少 CRS、垂直基准和独立控制时，只报告“模型相对当前点式观测的内部拟合误差”。

### 页面能打开是否代表验收通过

不能。页面加载只是交付链路的一项检查；仍需验证资产存在性、几何、拓扑、Mesh、拾取和证据等级。
