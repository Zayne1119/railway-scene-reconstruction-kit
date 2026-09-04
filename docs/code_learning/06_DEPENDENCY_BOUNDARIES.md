# 06｜自研核心与第三方边界

## 依赖矩阵

| 技术 | 当前状态 | 实际责任 | Adapter/入口 | Fallback / default-off | 能否成为权威路径 |
|---|---|---|---|---|---|
| NumPy | 核心依赖 | 数组、栅格、拟合、插值、mesh vertices | 多数算法模块 | 无；基础数值层 | 是，但权威性来自本项目契约 |
| SciPy | 核心依赖 | find_peaks、滤波、连通域、Hungarian、Rotation、KDTree | rail/linear/graph/projection/benchmark | 部分简单逻辑可重写；当前无 runtime fallback | 是，算法原语而非决策 owner |
| laspy/lazrs | 核心依赖 | LAS/LAZ 流式读写和压缩 | pointcloud/segments/detectors | 无 PDAL fallback | 是，I/O adapter |
| Pillow | 核心依赖 | 读全景、写 overlay/诊断 | rail/projection/figures | 无 | 是，图像 I/O，不决定资产接受 |
| jsonschema | 核心依赖但有 fallback | 项目/registry/benchmark schema | config/registry/benchmark | import 失败时手写最小校验；能力更弱 | 是；发布环境应安装完整依赖 |
| Open3D | optional `baselines` | plane RANSAC、DBSCAN 通用 baseline | `open3d_baseline.py`，CLI `benchmark-detect-open3d-rails` | 默认不安装；ImportError 明确报错 | **否，默认仅比较实验** |
| GISLab RailTrack | 外部程序/归一化适配 | 外部 rail point output | `gislab_baseline.py`、PowerShell runner | timeout/no-detection 被记录为非结果/0 detection | 否，benchmark only |
| Blender | 外部 DCC | 集成、接口 audit/repair、GLB/FBX export、固定视角渲染 | `blender/*.py` | 核心包可输出 OBJ；DCC QA 不可由 OBJ audit完全替代 | 可作为交付转换器，不是证据 source |
| Three.js | Web runtime | GLB 显示、picking、交互 | `web/src/main.js` | synthetic placeholder 非验收模型 | 展示层；不能决定资产真伪 |
| Vite | Web build/dev | 打包与本地服务 | `web/package.json` | 无 | 否，构建工具 |
| Web Crypto | Browser API | SHA-256 验收 | `web/src/acceptance.js` | 缺失会 fail closed | 是，交付完整性校验 |
| PDAL | 未依赖/未引用 | UNKNOWN | 无 | laspy 是当前 I/O | 当前不是项目能力 |
| COLMAP | 未依赖/未引用 | UNKNOWN | 无 | 使用现成 camera.csv | 当前不是项目能力 |
| HLoc | 未依赖/未引用 | UNKNOWN | 无 | 无图像定位 fallback | 当前不是项目能力 |
| LightGlue | 未依赖/未引用 | UNKNOWN | 无 | 无 | 当前不是项目能力 |
| Grounding DINO | 未依赖/未引用 | UNKNOWN | 无 | 人工 reviewed layout | 当前不是项目能力 |
| SAM2 | 未依赖/未引用 | UNKNOWN | 无 | 人工 reviewed layout | 当前不是项目能力 |
| Meshoptimizer | 未依赖/未引用 | UNKNOWN | 无 | Blender/GLB export；无自动 meshopt | 当前不是项目能力 |

## 我们自己真正负责什么

1. **项目契约**：schema、项目内路径、配置快照和初始化。
2. **铁路特定候选逻辑**：走廊 frame、轨头峰值、双轨配对、纵向共同支持。
3. **证据资格**：candidate/reviewed/accepted、observed/photo/rule/unsupported。
4. **全局拓扑**：TrackGraph、轨道身份、段界、不强连、推断边界。
5. **参数化建模接口**：轨道 sweep、轨枕/道床、审核 layout primitives。
6. **资产身份与关系**：registry、relations、geometry node 映射。
7. **fail-closed QA/release**：graph/mesh/registry/gate/Web hash binding。
8. **实验协议**：空间 holdout、blind task、experiment lock、baseline normalization。

这些是即使替换数值库或外部模型仍属于项目的核心 IP/工程能力。

## 第三方真正负责什么

- NumPy/SciPy 提供通用数值原语，不知道“什么是轨道权威资产”。
- laspy/lazrs 处理格式和压缩，不决定走廊/证据/拓扑。
- Open3D/GISLab 提供比较候选，当前不能绕过 review/TrackGraph/gates。
- Blender 将参数化/资产对象转换为 DCC/GLB/FBX，并做部分空间关系审计。
- Three.js 显示和拾取由 registry 标识的模型；浏览器不重建资产。

## Adapter、fallback 与默认关闭

- Open3D adapter：`open3d_baseline.detect_open3d_rail_baseline()`，函数内部延迟 import；没装 optional dependency 时明确失败。
- GISLab adapter：准备输入、规范化输出、记录 timeout/zero detection，保证外部失败不会被当本方法成功。
- Blender adapter：脚本目录独立于 Python package；通过 OBJ/GLB、origin、registry 和 JSON reports 交换。
- jsonschema fallback：`config.py`、`registry.py` 仅做最小手工校验，不能视为与完整 Draft 2020-12 等价。
- paired continuity：默认 diagnostic-only；要在新场景标定后显式启用。
- Open3D baseline：optional/default-off。

## 为什么外部方法被拒绝成为 authoritative path

不是因为它们一定“不准”，而是当前 adapter 输出缺少完整资格链：

1. 它们产生候选，不拥有本项目的人工复核状态。
2. 它们不自动维持跨 segment global track identity。
3. 它们不自动编码 observed/inferred evidence intervals。
4. 它们没有绑定本项目 camera/config/release hashes。
5. 它们没有通过本项目 registry/gate/Web acceptance 合同。

如果未来某外部方法更准，正确接入方式是让它替换**候选 producer**，同时保持 candidate schema、review、TrackGraph、registry 和 gates，而不是让它直接导出 final mesh。

## 明天全部替换 Open3D/COLMAP/SAM2 会怎样

- Open3D 当前本来只是 benchmark；删除它不影响默认生产轨道 detector。
- COLMAP/HLoc/LightGlue 当前不在代码中；项目依赖输入 camera.csv。未来换 pose producer，只要保持 camera contract 并重新 audit/bind，核心仍在。
- Grounding DINO/SAM2 当前不在公开生产代码；照片资产本来靠投影假设 + 人工审核 layout。替换语义模型只改变 candidate producer。
- 不变的核心：project schema、evidence/status、TrackGraph、asset registry、QA/gates、hash-bound release。

