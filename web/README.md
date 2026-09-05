# Railway evidence / hypothesis QA viewer

This local Three.js dashboard loads one integrated GLB scene plus the pipeline's QA reports. OBJ/MTL remains available as a fallback, and production geometry is not copied into the source repository.

Features:

- evidence / bounded-hypothesis comparison without loading two copies of the model;
- a Meshopt-compressed Web LOD for fast review, while the full GLB remains unchanged;
- discipline layers for track, catenary, conductor and station geometry;
- automatic issue lists for conductor seams, withheld catenary candidates and inferred track gaps;
- asset ID/type search, click selection and fixed QA views;
- report-driven metrics and risk markers.

## 快速安装与合成演示

需要 Node.js 22+；使用现有 Node 安装即可，不要求全局安装 npm 包。Python 环境使用工具包已有的锁文件。

Windows 首次使用推荐在工具包根目录运行统一入口（安装、环境检查、生成演示、启动网页）：

```powershell
powershell -ExecutionPolicy Bypass -File .\Railway.ps1 demo
```

已经安装时可用 `demo -SkipSetup`；只生成并检查、不启动服务用 `demo -SkipSetup -NoStart`；已有演示只启动网页用 `start`。端口冲突可显式指定 `-Port 3037`；已有便携 Node 可传 `-NodeDirectory '<Node 22 所在目录>'`。这与根目录 `Railway.ps1` 和 [新人交接说明](../docs/NEWCOMER_HANDOFF_CN.md) 使用同一入口，不需要改写网页配置。

以下是跨平台/手动运行的等效步骤。在工具包根目录生成独立合成演示：

```text
uv sync --frozen --extra dev
uv run python scripts/build_onboarding_demo.py
```

然后进入 `web`：

```text
npm ci
npm test
npm run dev
```

打开 `http://127.0.0.1:3010/?demo=1`。演示从 `/demo/project.json` 加载；页面始终显示“合成演示 / 非真实站场 / 非测量成果”横幅。构件选择、专业图层、检索和 QA 视角仍可使用。演示结果只证明安装与查看链路可用，不证明实测重建精度。

默认地址 `http://127.0.0.1:3010/` 仍只读取 `public/project.json`，不改写它，也不会自动换成演示。没有默认配置或尚未生成演示时，页面显示中文“生成演示 / 安装入口”。显式项目出错、401/403、网络失败或无效 JSON 会显示保留错误详情的重试面板，不会暗中切换到其他项目。只有未提供的可选报告或其 404 可被省略，界面显示“未提供”，不伪装成 QA 通过。

开发和预览均只监听本机 `127.0.0.1:3010`，端口占用直接失败，不会悄悄切换端口。不要为解决 403 而开放整个工作区，也不要未经授权对外公开模型。`npm run build` 后可用 `npm run preview` 本地检查；预览不提供 Vite 的开发期 `/@fs/` 资源入口。

## 本地项目与文件访问边界

保留已有 `public/project.json`。首次配置可复制 `public/project.example.json` 并填写自己的资源地址；示例使用本地 `/models/` 和 `/data/` 路径，对应 `web/public/models/` 与 `web/public/data/`。其中任何文件都会被本机开发服务器提供，而且构建时会复制进 `dist`，因此不要把私密凭据放进 `public`，也不要直接发布包含生产数据的 `dist`。

默认只放行网页目录，不依赖机器盘符。如果确需开发期加载外部模型，在启动进程的环境中明确设置 `RECON_VIEWER_DATA_ROOT` 为一个已批准的专用模型目录，再使用对应的 Vite `/@fs/` 绝对路径 URL。该变量不会作为前端 `VITE_*` 配置暴露。

PowerShell 示例从用户明确指定的目录解析绝对路径：

```powershell
$env:RECON_VIEWER_DATA_ROOT = (Resolve-Path -LiteralPath '../approved-models').Path
npm run dev
```

路径必须存在且为目录。磁盘根目录、用户主目录、网页根及其上级工作区会被拒绝；不会扫描或自动加入其他模型目录。符号链接按真实路径检查，Vite 的敏感文件拒绝规则继续生效。设置目录意味着授权本机网页服务器读取该目录中的资源，请只指定待查看的模型资料。此选项不开放局域网；本轮没有加入公网/LAN 发布功能。文件白名单与严格端口行为依据 [Vite 官方服务器选项](https://vite.dev/config/server-options)。

An alternate local config can be selected without replacing the default project:

```text
http://127.0.0.1:3010/?project=project.track-boundary.json
```

Only a plain JSON filename is accepted. Paths and external URLs are rejected. An explicit `project` takes precedence over `demo=1`; an invalid explicit filename raises an error instead of selecting the default project.

When Blender is unavailable, a named flat-material OBJ can be converted without external runtime dependencies:

```powershell
python scripts/export_obj_to_web_glb.py --input scene.obj --output scene.web.glb
```

The converter preserves OBJ object names as GLB nodes and validates node and triangle counts before writing its conversion report.

`public/project.json` is intentionally ignored by Git because it contains local paths. Generated `public/demo/` payloads are also ignored and can be regenerated; do not hand-edit the generated demo or replace production configuration to enable it.

## 自动检查入口

`npm test` 检查默认选择、显式项目优先级、演示标识、401/403 不降级、端口与本机监听、目录许可和无写死盘符。`npm run build` 检查实际打包；现有 Three.js 大包提示不是构建失败。

自动取景同时考虑横向与纵向视场，避免竖屏裁边。已生成 `public/demo/scene.glb` 时，测试还使用真实 Three.js GLTFLoader 在 Node 中核对对象、注册表、三角面和包围盒；干净仓库尚未生成演示时仅跳过这一项。该检查是 CPU 数据/几何验证，不是 WebGL 视觉验收。

浏览器自动验证可读取：

- `[data-testid="viewer-status"]`：`data-state` 为 `loading` / `ready` / `onboarding` / `error`；仅成功载入非空网格后 `data-model-loaded="true"`。
- `data-object-count` 与 `[data-testid="model-object-count"]`：实际载入的网格对象数量，不使用旧案例默认值。
- `[data-testid="synthetic-banner"]`：合成演示时可见。
- `[data-testid="onboarding"]`、`[data-testid="load-error"]`：分别表示未配置入口与明确加载错误；均有重新检查/重试操作。

## Scene contract

- Integrated OBJ object names may start with `TRACK--`, `CATENARY--`, `CONDUCTOR--` or `STATION--`.
- Candidate scenes may also retain pipeline names such as `TRACKGRAPH--TRACK-`, `SEG*`, `SUPPLEMENTAL-*` and `ADJACENT-*`; the viewer classifies these without renaming the assets.
- Bounded track hypotheses must include `INFERRED` in their object name.
- Registry assets are indexed by geometry node, asset ID and the configured namespace-prefixed ID.
- QA reports remain the source of truth; a risk marker is not a released asset.
- This viewer is for local candidate review. Formal release acceptance and content-hash validation remain separate pipeline steps.
