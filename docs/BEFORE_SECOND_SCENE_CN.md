# 第二铁路数据到来前：基线冻结与盲测操作手册

## 1. 这套准备不会降低下一项目的最终效果

研究层与生产重建层分开：

- 本手册新增的命令只读取并哈希现有代码、配置、协议和报告；
- 第二场景建立在独立的 `benchmarks/local/<site>` 目录；
- 不改生产算法，不改默认阈值，不覆盖 Site A 模型；
- B0 首次盲测永久保留，B1 少量标定和最终生产模型写入不同目录；
- B0 效果一般只说明“零场景调参泛化有限”，不禁止随后优化最终交付模型。

研究层回答“未经现场调参能做到什么”，生产层回答“利用合法现场信息后最终能交付什么”。两者必须同时保留，不能互相覆盖。

## 2. 新数据到来前只做一次：冻结 Site A

在仓库根目录运行：

```powershell
.\.venv\Scripts\python.exe -m railway_recon.research_protocol freeze-baseline `
  --repo-root . `
  --output benchmarks/local/site-a/research_freezes/site-a-predata-v2/manifest.json `
  --freeze-id site-a-predata-v2 `
  --benchmark-root benchmarks/local/site-a `
  --include pyproject.toml `
  --include uv.lock `
  --include src/railway_recon `
  --include configs/templates `
  --include tests `
  --include benchmarks/protocols `
  --include docs/PAPER_BENCHMARK_CN.md `
  --include docs/BEFORE_SECOND_SCENE_CN.md `
  --include benchmarks/local/site-a/protocols `
  --include benchmarks/local/site-a/manifests `
  --include benchmarks/local/site-a/reports
```

输出清单记录每个文件的字节数和 SHA-256，但不复制、不改写这些文件。即使当前工作树有未提交修改，也能由逐文件哈希精确绑定。

随时可以验证冻结状态是否仍与磁盘一致：

```powershell
.\.venv\Scripts\python.exe -m railway_recon.research_protocol verify-baseline `
  --repo-root . `
  --manifest benchmarks/local/site-a/research_freezes/site-a-predata-v2/manifest.json
```

后续代码继续开发后，旧基线验证失败是正常现象；不能重写旧 manifest，应建立 `baseline_v2`。

## 3. 新数据到场当天：三个命令

### 3.1 初始化独立盲测目录

`chainage-end-m` 是论文盲测评价走廊，不是客户全部数据长度。选择 25 m 的整数倍，例如 200、500 或 1000 m。

```powershell
.\.venv\Scripts\python.exe -m railway_recon.research_protocol init-blind-site `
  --target benchmarks/local/site-b `
  --dataset-id railway-site-b `
  --scene-id site-b-eval-corridor `
  --chainage-end-m 200 `
  --baseline-manifest benchmarks/local/site-a/research_freezes/site-a-predata-v2/manifest.json
```

生成的数据角色是 `prospective_blind_test`，所有空间块初始为 `test`。目录中同时写入允许适配项、禁止动作和 B0/B1 分离规则。

### 3.2 自动登记输入

本地坐标示例：

```powershell
.\.venv\Scripts\python.exe -m railway_recon.research_protocol register-inputs `
  --root benchmarks/local/site-b `
  --point-cloud D:\Private\site-b\point-cloud.laz `
  --camera-poses D:\Private\site-b\camera.csv `
  --panoramas D:\Private\site-b\panoramas `
  --coordinate-mode local
```

已知 EPSG 示例：

```powershell
.\.venv\Scripts\python.exe -m railway_recon.research_protocol register-inputs `
  --root benchmarks/local/site-b `
  --point-cloud D:\Private\site-b\point-cloud.laz `
  --coordinate-mode epsg `
  --epsg 4547 `
  --vertical-datum "项目提供的正式高程基准名称"
```

命令计算文件或全景目录的 SHA-256。输入已经登记后拒绝静默替换；拿错数据时应新建 benchmark，而不是覆盖旧证据。

### 3.3 全哈希封存输入

```powershell
.\.venv\Scripts\python.exe -m railway_recon.research_protocol seal-intake `
  --root benchmarks/local/site-b
```

只有必要输入存在、字节数一致且全哈希成功时，状态才变为 `intake_frozen_ready_for_experiment_lock`。

## 4. B0 与 B1 的边界

### B0：冻结泛化结果

允许：

- 文件格式转换；
- 单位和坐标轴声明；
- CRS 转换；
- 传感器标定和相机字段映射。

禁止：

- 根据模型效果修改检测阈值；
- 看见断轨后新增只适用于 Site B 的拓扑规则；
- 在第一次指标计算前人工补 Mesh；
- 删除或覆盖失败的第一次运行。

### B1：少量标定和最终生产优化

B0 保存以后，可以声明 20–50 m Pilot 进行参数标定。B1 和最终交付模型不受 B0 外观约束，但必须使用新运行目录和新配置哈希。

因此盲测不会“锁死”下一项目，只会保留一份未调参证据。

## 5. 单人项目怎样减少标注工作

当前只有一名 Owner，不强制双人全量标注。执行规则是：

1. 先自动保存 B0 候选、Graph、Mesh 和指标；
2. 只抽查预先固定的空间块与失败类型；
3. 记录抽查分钟数和修改次数；
4. 论文中明确写“single-owner sampled review”；
5. 没有独立测量点时，只报告相对观测的内部拟合误差。

对应真值等级使用 `GT-L1S_observed_single_reviewer`，不能填写为
`GT-L1_observed_consensus`。

这不会把个人标注伪装成独立真值，也不会给日常开发增加全量人工标注任务。

## 6. 到场前完成标准

- [x] Site A 代码、配置、协议和结果可生成内容哈希清单；
- [x] Site B 可在独立目录初始化为前瞻盲测；
- [x] 点云、Pose 和全景可一键登记；
- [x] 文件与目录均支持全哈希封存；
- [x] B0 与 B1 输出禁止互相覆盖；
- [x] 单人复核边界已写入协议；
- [ ] 新数据真实路径、坐标和里程范围——只能等数据到场后填写；
- [ ] B0 实验命令和参数绑定——只能在输入封存后锁定。

## 7. 发生错误时

- 输入登记错误：不要覆盖，删除尚未使用的整个 Site B benchmark 后重新初始化；
- B0 已运行：不得删除，创建新的 B1 或工程运行；
- 坐标信息不清楚：使用显式 `local`，不得猜 EPSG；
- 垂直基准未知：保留 `null`，不得声明绝对高程精度；
- 哈希不匹配：回到数据交付方确认文件版本，不能手改 manifest 中的哈希。
