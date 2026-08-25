# GitHub 私有仓库部署与协作

本指南用于把通用工具包交给同事协作。默认策略是私有仓库；生产数据和成果模型是否进入任何远程仓库，必须由项目负责人另行授权。

## 1. 什么可以提交

允许进入通用仓库：

- 通用源码；
- JSON Schema 和脱敏配置模板；
- 合成或完全脱敏的小样例；
- 自动测试和 CI；
- 技术文档；
- 不包含现场信息的流程图；
- 授权明确的第三方依赖说明。

禁止进入通用仓库：

- 生产 LAS/LAZ；
- 现场全景和相机轨迹；
- 精确坐标、控制点和垂直基准记录；
- 站名、线路名、客户名和设备台账；
- 生产 GLB/FBX/OBJ；
- 局域网地址、访问令牌、账号和 `.env`；
- 本机绝对路径；
- 日志、缓存、`runs/`、`workspace/`；
- 来源许可不清楚的纹理、HDRI、字体和图片。

## 2. 首次建私有仓库

在仓库根目录初始化：

```powershell
git init
git branch -M main
git add .
git status
git commit -m "chore: initialize reusable railway reconstruction kit"
```

在 GitHub 创建 **Private** 空仓库，不勾选自动生成 README 或 License，然后关联远程：

```powershell
git remote add origin <private-repository-url>
git push -u origin main
```

不要在聊天、截图或文档中保存带令牌的远程地址。优先使用 Git Credential Manager 或 SSH。

## 3. 推送前安全检查

每次首次推送、发布 tag 和合并 release PR 前运行：

```powershell
railway-recon safety-check --root .
git status --short
git diff --cached --stat
```

再人工搜索常见敏感项：

```powershell
git grep -n -I -E "(token|secret|password|api[_-]?key)"
git ls-files
```

安全检查通过不代表数据已获授权。它只能发现已实现的模式和文件大小问题。

## 4. `.gitignore` 最低要求

至少忽略：

```gitignore
.venv/
__pycache__/
*.pyc
.pytest_cache/
.ruff_cache/
projects/*/input/
projects/*/workspace/
runs/
*.las
*.laz
*.glb
*.fbx
*.obj
*.mtl
*.hdr
*.exr
*.log
.env
.env.*
!.env.example
node_modules/
dist/
.next/
```

若需要提交合成小模型，应为指定文件建立白名单，而不是放宽全部模型后缀。

## 5. 大文件与 Git LFS

通用仓库建议不跟踪超过 10–50 MB 的文件。Git LFS 不是生产点云的默认解决方案：

- LFS 仍会把文件上传到远程；
- 删除工作区文件不会清除 Git 历史；
- 敏感模型进入 LFS 同样属于外发；
- 大量点云会迅速消耗额度并拖慢克隆。

只有经授权的合成/脱敏演示资产才考虑 LFS。生产数据使用受控对象存储或内部数据平台，并在项目配置中使用本地相对路径。

## 6. 分支和 Pull Request

推荐分支：

- `main`：始终可安装、可测试；
- `feature/<topic>`：通用功能；
- `fix/<topic>`：缺陷修复；
- `docs/<topic>`：文档；
- 不为每个现场建长期代码分支，现场差异应尽量进入受控配置。

PR 至少说明：

- 修改原因；
- 影响的流水线阶段；
- 配置是否兼容；
- 新增/修改测试；
- 是否接触生产数据；
- 是否改变证据等级或人工复核边界；
- 回退方法。

禁止直接向 `main` 强推。建议开启：

- Require pull request；
- Require CI checks；
- 至少一名 reviewer；
- 禁止 force push；
- 删除已合并分支；
- Secret scanning 与 Dependabot（若组织允许）。

## 7. CI 建议

最小 CI：

1. Python 3.11 安装；
2. `ruff check`；
3. `python scripts/run_tests.py`（或开发环境中的 `pytest`）；
4. 配置模板 Schema 校验；
5. 合成样例的 audit/segment/registry/qa；
6. `safety-check`；
7. 若有 Web：`npm ci`、测试和 production build。

Blender 体积较大，可用专用 runner 或发布前本地强制验收；不能因为 CI 未安装 Blender 就省略引擎交付检查。

## 8. 版本发布

软件使用语义化版本：

- `0.x`：内部试用，接口可能变化；
- `1.0.0`：至少由第二位同事在新项目上独立跑通；
- patch：兼容修复；
- minor：向后兼容功能；
- major：破坏性配置或 Schema 变化。

发布步骤：

```powershell
python scripts\run_tests.py
railway-recon safety-check --root .
git status
git tag -a v0.1.0 -m "Internal reusable toolkit v0.1.0"
git push origin main --tags
```

发布说明必须包含：新增内容、配置迁移、已知限制、依赖版本和回退 tag。

## 9. 同事克隆与上手

同事只需要获得：

- 私有仓库读写权限；
- 通用代码；
- 单独授权的项目数据访问；
- 项目负责人提供的 `project.json`/`rules.json` 基线；
- 人工复核人和发布负责人名单。

克隆后依次执行：

```powershell
git clone <private-repository-url>
cd <repository-folder>
.\scripts\bootstrap.ps1
python scripts\run_tests.py
railway-recon --help
```

之后按 [快速上手](QUICKSTART_CN.md) 建项目，严禁把生产输入复制到通用仓库目录中再提交。

## 10. 误提交处理

发现密钥或敏感数据被提交时：

1. 立即停止继续推送和转发；
2. 撤销/轮换密钥；
3. 通知仓库管理员和项目负责人；
4. 按组织流程从 Git 历史清除；
5. 检查 fork、缓存、CI artifact 和 release；
6. 形成事件记录；
7. 更新 `.gitignore` 和安全检查。

只删除最新提交中的文件是不够的，旧 Git 历史仍可能包含内容。

## 11. 公开仓库前的额外门禁

私有改公开前必须重新批准：

- 代码著作权和开源许可证；
- 第三方依赖、图片和模型许可；
- 脱敏样例是否可反推现场；
- Git 历史是否曾包含生产数据；
- 文档截图是否包含站名、坐标、用户信息或内部界面；
- 安全负责人和项目负责人的书面确认。

没有明确授权时，仓库持续保持 Private。
