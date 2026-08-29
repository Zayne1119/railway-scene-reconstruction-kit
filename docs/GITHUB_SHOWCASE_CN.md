# GitHub 宣传页与演示视频发布指南

本指南用于把仓库首页做成可对外说明价值的展示页，同时避免将客户数据、现场影像和生产模型误传到 Git 历史。

## 1. 推荐的展示层级

建议按三层组织：

1. **README 首屏**：项目名称、一句话价值、匿名封面、能力标签；
2. **README 正文**：Pipeline、证据等级、脱敏案例、快速开始和产品边界；
3. **Release / GitHub Pages**：完整演示视频、版本说明和经授权的交互式页面。

README 负责让访客在一分钟内理解项目。详细手册继续保存在 `docs/`，不要把开发历史全部堆到首页。

## 2. 媒体目录

仓库内只保留小体积、已脱敏、许可明确的宣传素材：

```text
docs/media/
├─ cover-pointcloud-to-cim.png   # README 首屏示意封面
├─ hero-preview.gif              # 可选：6–10 秒无敏感信息预览
├─ pipeline-overview.png         # 可选：不含真实数量的技术路线图
├─ component-track.png           # 可选：匿名轨道局部
└─ component-canopy.png          # 可选：匿名雨棚节点局部
```

当前封面是概念示意图，不作为测量成果或模型精度证据。

`hero-preview.gif` 和 `showcase-input-output.png` 由 `scripts/build_showcase_media.py` 从该概念封面生成。脚本不读取 `projects/`、`data/`、`deliverables/` 或任何生产模型。完整 MP4 输出到仓库外并作为 Release 资产上传。

## 3. 视频推荐方案

### README 内

使用短 GIF 或动态图作为可点击封面：

```html
<a href="https://example.invalid/approved-demo.mp4">
  <img src="docs/media/hero-preview.gif" alt="铁路场景重建演示">
</a>
```

发布时请把示例地址替换为实际 GitHub Release 资产地址。

推荐参数：

- 时长 6–10 秒；
- 960×540 或 1280×720；
- 10–15 fps；
- 无声音或无敏感声音；
- 文件尽量小于 10 MB；
- 第一帧本身就能说明“点式观测 → 结构化模型”。

### 完整视频

完整 MP4 建议放入私有 GitHub Release 或获批的内部对象存储：

1. 建立与代码版本一致的 Release；
2. 上传 H.264 MP4；
3. 在 Release 说明中写明视频版本、模型版本和授权范围；
4. 把 README 动图链接更新为 Release 资产 URL；
5. 不把完整 MP4 提交到普通 Git 历史。

当前 `v0.2.0` 使用以下命令生成 36 秒合成演示视频：

```powershell
python scripts/build_showcase_media.py `
  --mp4 ..\release-assets\railway-scene-reconstruction-kit-v0.2.0-showcase.mp4
```

该视频只展示方法概念，不是现场成果或精度证据。

## 4. 私有与公开仓库的素材边界

### 私有仓库在明确授权下可使用

- 纯模型飞行视频；
- 不带现场标识的模型总览；
- 不含真实照片的构件近景；
- 脱敏后的点—模型对比结果。

### 公开仓库必须再次审批

纯模型渲染仍可能暴露真实站场布局。公开前必须确认客户或数据所有方允许发布衍生模型画面。

### 不允许作为宣传素材

- 原始全景和可识别站名、标牌、设备铭牌；
- 原始点云全局空间布局和相机轨迹；
- 精确坐标、控制点、内部网络地址；
- 带聊天头像、浏览器地址栏或 Windows 水印的截图；
- 未经许可的 HDRI、纹理、字体和第三方图片；
- 候选帧号、设备台账和客户内部编号。

## 5. 发布前媒体验收

每张图片和每段视频至少完成：

- 检查站名、坐标、候选编号、账号和水印；
- 清除 EXIF、PNG 文本块和视频 metadata；
- 在 200% 缩放下人工检查标牌和铭牌；
- 确认画面没有暴露未授权的真实空间布局；
- 确认素材版权和再分发许可；
- 记录审批人、审批日期和可见范围；
- 运行仓库安全扫描。

## 6. GitHub 首页设置

仓库建立后建议补充：

- About：一句话价值描述；
- Topics：`railway`、`point-cloud`、`cim`、`3d-reconstruction`、`digital-twin`；
- Social preview：使用匿名封面，避免现场截图；
- Releases：保存正式演示视频和变更说明；
- Pages：只有在需要公开交互演示且完成授权后再启用。

## 7. 推荐首页文案

中文：

> 将铁路点式几何观测与全景影像转换为证据可追踪、资产可查询、可交付至 Web / Blender / UE 的结构化 CIM 场景。

英文：

> An evidence-aware pipeline for turning railway point observations and panoramas into traceable CIM assets.
