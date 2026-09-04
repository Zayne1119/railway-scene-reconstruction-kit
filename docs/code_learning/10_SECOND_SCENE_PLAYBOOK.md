# 10｜第二铁路场景作战手册

假设收到：`new.laz`、`new camera.csv`、`new panoramas/`。命令示例使用 PowerShell，`$Project` 只是说明变量；执行前必须把它设为**明确的新项目绝对路径**，不要指向旧项目。

## 第一小时：只做输入资格，不调模型

### 0–15 分钟：登记不可猜的事实

确认并记录：

- 点云单位、轴、LAS version/point format、scale/offset、是否有 RGB。
- `camera.csv` 列、坐标单位、时间顺序、旋转单位与 Euler/pose convention。
- panorama 与 camera `file` 的一一对应、分辨率和许可。
- EPSG、垂直基准、控制点、独立检查点；不知道就填 `null`，不要复制旧项目。
- 线路范围、是否有道岔/曲线/多层/站台雨棚、数据缺侧。

对应代码：`pointcloud.audit_point_cloud`、`camera.audit_camera_rows`、`audit.audit_project`。

### 15–30 分钟：创建隔离项目

```powershell
& .\.venv\Scripts\python.exe -m railway_recon init $Project --name "New Site" --project-id new-site-001
```

然后只修改新项目 `project.json` 的 identity、inputs、CRS 和路径。将文件复制/链接到新项目是外部数据管理动作；本工具不会替你安全搬运原始资料。

```powershell
& .\.venv\Scripts\python.exe -m railway_recon validate --project $Project\project.json
& .\.venv\Scripts\python.exe -m railway_recon audit --project $Project\project.json --full-hash
```

### 30–60 分钟：停下来读 audit

阻断条件：camera 必填列缺失、文件引用大面积不存在、单位不明、路径错、点云为空/损坏。若只有 CRS/独立点未知，可继续做内部拟合 prototype，但书面限制绝对精度声明。

## 第一天：建立 50–100 m prototype

### 1. 选择代表性而非“最漂亮”区段

至少包含：一处普通直轨、一处分段边界、一个高竖直构件/雨棚、一张可核照片。若有曲线/道岔，prototype 必须覆盖一个，不要只选最容易直线。

### 2. 规划和裁剪

```powershell
& .\.venv\Scripts\python.exe -m railway_recon plan-segments --project $Project\project.json
& .\.venv\Scripts\python.exe -m railway_recon segment --project $Project\project.json --ids s0000_0050m s0050_0100m
```

检查 segment manifest 的 bounds 和 camera ranges。`plan_segments()` 使用相机附近的轴对齐 envelope，不是精确曲线 corridor；弯线要特别看是否漏点/带入邻线。

### 3. 校准 rail detector，禁止复制结论

先复制默认 config 到新项目（init 已完成），一次只改一项：

- 必重标定：z mode/percentiles，cross/long bins，minimum prominence/coverage，peak spacing，candidate width/height，top fit method/quantile/min bin points。
- 几何常量需现场确认：nominal gauge、rail head width、允许 correction/crosslevel。
- frame：弯道验证 dominant regression 上下文；必要时比较 PCA，但不要用最终指标反向挑阈值。

```powershell
& .\.venv\Scripts\python.exe -m railway_recon detect-rails --project $Project\project.json --segment s0000_0050m
& .\.venv\Scripts\python.exe -m railway_recon detect-linear --project $Project\project.json --segment s0000_0050m
```

输出仍是 candidate。用 diagnostic PNG、原始点云和照片核对，不凭模型“看着顺”。

### 4. 照片 pose hypothesis

仅在 camera 有 rotation 且 segment 点云有 RGB 时：

```powershell
& .\.venv\Scripts\python.exe -m railway_recon calibrate-projection --project $Project\project.json --segment s0000_0050m --camera-index 10
```

至少复核多张相邻相机结构边。自动最低 RGB error 不能批准 pose。

### 5. 轨道人工接受

```powershell
& .\.venv\Scripts\python.exe -m railway_recon track-review-create --project $Project\project.json --source s0000_0050m=workspace/reports/s0000_0050m_rail_candidates.json --source s0050_0100m=workspace/reports/s0050_0100m_rail_candidates.json --output-name pilot-review
```

有独立 reviewer：填写 CSV 后 `track-review-apply`。只有项目 owner：使用 `track-review-owner-override --approved-by <真实姓名> --reason <具体原因>`，保留其非独立性质。需要手工增删 pair/确认道岔边界：使用明确 schema 的 `track-pair-curate`，不要改原 report。

### 6. TrackGraph 和 mesh

```powershell
& .\.venv\Scripts\python.exe -m railway_recon track-graph-build --project $Project\project.json --source s0000_0050m=<accepted-report-1.json> --source s0050_0100m=<accepted-report-2.json>
& .\.venv\Scripts\python.exe -m railway_recon track-graph-validate --project $Project\project.json --graph workspace/derived/track_graph.json
& .\.venv\Scripts\python.exe -m railway_recon build-track-graph-mesh --project $Project\project.json
```

若 graph 为 fail/review_required，不能通过改 audit JSON 或增大阈值跳过。回到 candidate/review/identity。

### 7. 非轨道资产

当前代码不会自动把 vertical/cable candidate 完整变成站台、雨棚、接触网专业模型。必须形成审核的 `railway.reviewed-scene-layout.v1`，每个 asset 写 type、evidence、confidence、sources、limitations、geometry，再执行：

```powershell
& .\.venv\Scripts\python.exe -m railway_recon build-reviewed-scene --project $Project\project.json --layout <reviewed-layout.json>
```

自动站台拟合/雨棚多平面/完整接触网 fitter 当前为缺口；不要把手工 layout 描述成全自动。

## 哪些可直接复用

### Schema/制度可复用

- `project.schema.json`、`asset-registry.schema.json`、`track-graph.schema.json`。
- evidence/status/source 枚举。
- review hash binding、TrackGraph 不强连、registry freeze、Quality Gate v2、Web hash acceptance。
- OBJ 基础 audit、关系的 signed gap 语义、公开仓 safety rules。

### 算法结构可复用但需验证

- camera chainage、CorridorFrame、RouteSampler。
- rail grid/prominence、robust fit、pairing。
- Hungarian track identity、seam/duplicate/coverage gates。
- rail sweep、registry mapping、projection hypothesis search。

## 哪些绝不能沿用旧值

- EPSG/vertical datum/control points。
- z percentile/height window、点数/coverage/prominence。
- bin size 与 candidate width（受密度和扫描分辨率影响）。
- segment length/context（受曲率、站场复杂度影响）。
- graph identity/seam/support thresholds（需新场景误差分布）。
- platform/canopy P90（当前算法尚未接通，更不能照抄）。
- photo pose convention、曝光/颜色阈值。
- UE scale/origin/import normals 策略。

## 什么时候扩展全 corridor

必须同时满足：

1. prototype 包含至少一个困难段，不只是直线；
2. 候选诊断和人工 review 流程可重复；
3. graph pass，且段界/方向/重复/推断检查都有实际样例；
4. mesh audit、轨道装配、站台/雨棚关系和六视角验收通过；
5. threshold/config 已冻结，未看外部 test 后反复调参；
6. asset ID/evidence/registry 能从点云/照片追溯；
7. 失败有明确返工入口，不靠手改 final mesh。

## 什么时候可以宣称 authoritative

- 明确 release id；
- graph 与当前 camera/config/source hash 绑定且 pass；
- mesh/装配/关系/视觉 QA 通过；
- 所有发布资产 accepted，registry frozen；
- model/registry/asset set 被 gate 和 Web config 精确绑定；
- 所有低证据/不可见区域保持标识；
- 若声称绝对精度，还必须有 EPSG、垂直基准和独立检查点。

缺任一项，只能称 candidate、reviewed model 或 internal-fit model。
