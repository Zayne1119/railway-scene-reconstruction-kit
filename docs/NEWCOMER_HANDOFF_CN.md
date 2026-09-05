# 新人交接：安装、出第一个模型、打开网页

这份文档是交接的第一入口。你不需要先学建模或逐页签核。先运行独立合成演示，确认自己的电脑能出模和查看，再让 AI 按 [接手指令](AI_HANDOFF_CN.md) 帮你处理真实项目。

**当前能力边界：**一条命令可以安装环境、从合成点云生成演示模型并启动网页；不是把任意客户文件拖进来就自动完成整站精细重建。真实项目仍需要明确数据位置、单位、范围与交付目标；不确定的几何会保留为候选，最后可以手工微调。

## 1. 拿到什么，放在哪里

拿到完整的 `railway-scene-reconstruction-kit` 源代码目录，里面至少应有 `Railway.ps1`、`pyproject.toml`、`uv.lock`、`scripts/`、`src/` 和 `web/package-lock.json`。只拿 `paper/`、单个 README 或网页源文件不够。

解压到自己有写入权限的普通目录，例如 `C:\Work\RailwayKit`。不要求 D 盘，也支持含空格目录。不要放在系统目录、他人的生产目录或需要管理员权限的位置。不要复制旧电脑的 `.venv`、`node_modules` 和 `.runtime`；它们可能绑定原电脑路径。

默认一键入口面向 Windows 10/11 x64、PowerShell 5.1 或更新版本。基础演示无需 Blender、CUDA、显卡计算环境、Git 或 AI API 密钥。网页需要支持 WebGL 的现代浏览器。安装需要网络及可用磁盘空间，时间取决于网络；安装失败会明确停止，不会把失败显示为“已就绪”。

## 2. 第一次：只运行这一条

在工具包目录空白处打开终端，运行：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\Railway.ps1 demo
```

该执行策略只作用于这次 PowerShell 进程，不修改整台电脑的策略。脚本不会要求管理员权限，也不会更改全局 PATH、注册表或自动停止其他应用。

它依次执行：

1. 使用已有 uv，或下载固定 uv 0.12.1；缺少 Python 时用 uv 安装项目内 Python 3.11.15。
2. 依据 `uv.lock` 安装 Python 依赖，依据 `web/package-lock.json` 安装网页依赖；不安装大型可选研究基线，也不自动移除已有 Python 环境中额外安装的工具。
3. 使用已有 Node，或下载项目内 Node 22.23.2。便携 uv/Node 下载核对官方 HTTPS 校验清单的 SHA-256；Python 由 uv 管理安装。
4. 自动检查运行环境，生成合成点云、轨道候选、具名模型、资产表与检查报告。
5. 启动仅本机可访问的网页，终端保持运行。打开 **http://127.0.0.1:3010/?demo=1**。

网页应显示“合成演示 / 非真实站场 / 非测量成果”、模型对象数及 3D 模型。可以旋转缩放、选择对象、开关专业图层；关闭服务按终端里的 `Ctrl+C`。脚本不自动打开新窗口。

演示包含 2,726 个合成点、1 对检测轨道、80 个具名对象、6,096 个三角面。轨道来自合成点云检测；站台、柱、梁、雨棚采用明确标记的人工编写合成布局。它证明的是安装与产物链路，不是现场精度、自动站房识别或完整工程交付。

## 3. 结果在哪里

| 路径（相对工具包根目录） | 内容 |
| --- | --- |
| `projects/onboarding-demo/` | 独立演示项目、合成输入、配置和生成产物 |
| `projects/onboarding-demo/viewer/scene.glb` | 可直接用于支持 GLB 的查看工具的演示模型 |
| `projects/onboarding-demo/demo_manifest.json` | 输入、源码、依赖和产物的校验记录 |
| `web/public/demo/` | 网页使用的演示副本，不覆盖正式 `public/project.json` |
| `.venv/`、`.runtime/`、`web/node_modules/` | 本机环境与缓存，不是模型，也不应随源代码交接 |

重复运行会核对既存演示的内容和版本，一致时复用；不会直接覆盖未知文件。若你修改了生成源码或演示产物，它会停止并说明原因，不会以旧结果冒充新结果。更新代码后的旧演示，请让 AI 保留原目录并用新版本目录重新构建，不需要人工逐文件比对。

## 4. 日常只用这些命令

以下命令均在工具包根目录运行，前缀相同：

```powershell
# 已安装：生成/校验演示并启动；不重新安装
powershell -ExecutionPolicy Bypass -File .\Railway.ps1 demo -SkipSetup

# 只生成和自动检查，不启动网页
powershell -ExecutionPolicy Bypass -File .\Railway.ps1 demo -SkipSetup -NoStart

# 只启动现有演示网页
powershell -ExecutionPolicy Bypass -File .\Railway.ps1 start

# 只读环境诊断，不安装东西
powershell -ExecutionPolicy Bypass -File .\Railway.ps1 doctor

# 维护者回归：Python 测试 + 网页测试 + 网页构建；不用每次出模都跑
powershell -ExecutionPolicy Bypass -File .\Railway.ps1 check
```

只需要模型、暂时不看网页：首次用 `demo -CoreOnly`，之后用 `demo -CoreOnly -SkipSetup`。基础 OBJ/GLB 演示不要求安装 Blender。需要 Blender 专用导出、FBX 或渲染时，再按具体任务配置。

## 5. 常见问题：只处理异常，不增加审核表

| 现象 | 处理 |
| --- | --- |
| 找不到 `Railway.ps1` | 切换到包含该文件的工具包根目录，不是 `paper` 或 `web` |
| 下载超时、代理或证书错误 | 检查公司网络是否允许官方依赖源；不要禁用证书验证。可使用下方已有环境参数 |
| 3010 端口被占用 | `Railway.ps1 start -Port 3037`，打开 `http://127.0.0.1:3037/?demo=1`；不会停止占用端口的其他程序 |
| 缺少 Node 或 Python 依赖 | 运行 `Railway.ps1 setup`；体检失败不是建模失败 |
| `--locked` 报依赖锁不一致 | 交给 AI/维护者核对代码与锁文件版本，不能随意去掉 `--locked` |
| 网页显示尚未配置 | 确认地址有 `?demo=1`，运行 `demo -SkipSetup -NoStart`，然后重试 |
| 网页明确报 403、配置或模型加载失败 | 保留错误交给 AI；不开放整个磁盘，不隐藏错误，不自动换成另一项目 |
| 只有模型不可见，检查已通过 | 检查浏览器 WebGL/硬件加速与控制台错误；环境通过不等于完成可见验收 |
| 演示文件已被修改或源码版本变化 | 让 AI 新建版本目录，保留旧产物；不要改 manifest 假装通过 |

首次安装不能离线凭空获得依赖。`-Offline` 仅供已经准备好运行时和依赖缓存的电脑使用：

```powershell
powershell -ExecutionPolicy Bypass -File .\Railway.ps1 setup `
  -UvPath 'C:\ApprovedTools\uv.exe' `
  -PythonPath 'C:\ApprovedTools\Python311\python.exe' `
  -NodeDirectory 'C:\ApprovedTools\node-v22.23.2-win-x64'
```

这些参数是你明确指定的现成工具，不会被复制进项目；更换电脑需要重新指定或走默认安装。Node 至少满足锁内 Vite 的版本要求（当前建议直接使用测试版本 22.23.2）；不是任何“Node 22”小版本都能保证可用。

离线模式还需要项目 `.runtime/cache` 中可用的 uv 缓存和本机 npm 缓存。无缓存会明确失败。正常首次安装不要加 `-Offline`。

运行时来源：[uv 官方安装说明](https://docs.astral.sh/uv/getting-started/installation/)、[uv 的 Python 安装说明](https://docs.astral.sh/uv/guides/install-python/)、[Node 22.23.2 官方下载](https://nodejs.org/en/download/archive/v22.23.2)。uv 使用其维护的 Python 独立发行构建，不是声称从 Python.org 下载官方 Windows 安装程序。

## 6. 接手真实项目

把 [AI_HANDOFF_CN.md](AI_HANDOFF_CN.md) 交给协助你的 AI。提供一次最少信息：**允许读取的数据目录、单位/坐标已知情况、要做哪一段、需要什么输出**。之后由 AI 检查配置、执行小样段、生成候选、报告异常；正常步骤不要反复要求你批准。

```powershell
powershell -ExecutionPolicy Bypass -File .\Railway.ps1 new -ProjectId site_c -Name 'Site C'
```

这只创建空项目，**不是出模命令**。真实输入接收和 Pilot 操作见 [QUICKSTART_CN.md](QUICKSTART_CN.md)。现有构件级流水线可以复用，但真实站场的轨道拓扑、站台/雨棚资产配置与语义不能从合成演示参数直接套用。自动候选可先用于看效果和微调；未验证部分不能被标成实测真值或正式验收。

默认工作方式是单人 + AI：机器做格式、路径、依赖、产物和结构检查；你只处理确实阻塞的输入问题及最后希望调整的外观。旧版完整检查清单用于特定工程合同的正式验收参考，不是每次运行必须额外完成的多人签字任务。

## 7. 安全交接与验证边界

源代码和生产资料分开交接。不要直接打包整个工作区：其中可能有客户点云、全景、精确坐标、模型、网页静态资源和本机凭据。`.gitignore` 不是数据泄露保证，网页 `public/` 中的内容还会进入网页构建产物，不能直接公开。

维护者可用 `scripts/prepare_handoff_source.py --output <全新目录>` 创建明确白名单的源码副本；该操作不上传、不发布、不复制现成模型或旧环境。它保留许可证，并记录所复制文件的哈希，但不能证明任意人工修改过的源码文本里绝无秘密。许可证与客户数据授权仍然有效。

自动体检通过只代表所选运行环境满足要求；模型检查通过只代表相应机器可检查的条件通过；浏览器实际看到模型、现场精度和合同验收是不同层次，不能互相冒充。本轮验证结果见 [交接验证记录](HANDOFF_VERIFICATION_CN.md)。

其他系统可按 `web/README.md` 手动使用锁文件安装，但本轮一键脚本仅验证 Windows；不宣称 macOS/Linux 一键兼容。
