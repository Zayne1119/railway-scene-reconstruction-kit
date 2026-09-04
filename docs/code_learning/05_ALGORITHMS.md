# 05｜真正影响能力的算法

## 1. 坐标转换与走廊 frame

**Problem**：铁路长而窄，直接在世界 XY 栅格检测会把线路朝向变化混入横向峰值。

- Input：相机 XY、点云 XY。
- 数学：PCA/SVD 取主轴，或 dominant-axis regression 拟合次坐标对主坐标；方向按首末相机对齐。投影为 `s=(p-o)·a`、`c=(p-o)·n`。
- Implementation：`geometry.py::fit_corridor_frame()`、`fit_corridor_frame_dominant_axis_regression()`、`fit_segment_corridor_frame()`。
- Parameters：rail config 的 frame method、前后上下文 100 m。
- Assumption：相机沿铁路前进，短范围内可由单一局部轴表示。
- Failure：短段横摆、回头、相机离轨、弯道过强；方向可能翻转或横向峰被抹平。
- Why：dominant regression 专门减少短段 lateral oscillation 对 rail peak 的污染。
- Alternative：PCA/SVD 已在同文件；RouteSampler 在 graph 阶段提供曲线 canonical route。
- Downstream risk：改 frame 算法会改变候选 cross position、ID matching 和所有 binding hash，必须重跑 review/graph。

## 2. Chainage 构造

- Problem：给不同 segment 一个共同有序坐标轴。
- Input：camera XYZ。
- 数学：累计相邻 3D 距离；`RouteSampler.sample()` 对位置、切向、法向插值。
- Implementation：`camera.camera_trajectory()`；`track_graph.RouteSampler`。
- Assumption：相机顺序正确，路径可代表铁路沿线。
- Failure：重复/乱序 pose、绕行站房、整体偏移；chainage 仍可单调但不等于正式里程。
- Alternative：当前 repo 没有 survey alignment/正式里程映射实现，记为 UNKNOWN。
- Downstream：改变 chainage 定义会改变 segment、track IDs、sleeper phase、gate evidence，不可局部替换。

## 3. Rail candidate extraction

- Problem：从站场大量水平/线性结构中找细长高起钢轨。
- Input：segment LAZ、corridor frame、z settings。
- Idea：在 longitudinal–cross 栅格统计占用和高度；以横向中值基线去趋势；平滑覆盖/高度 prominence；`scipy.signal.find_peaks` 找横向峰；再从候选窗口取 rail-top samples。
- Implementation：`algorithms/rail_candidates.py::detect_rail_candidates()`。
- Parameters：cross bin 0.05 m、long bin 0.25 m、height prominence 0.07 m、line coverage 0.12、peak spacing 0.35 m、candidate half width 0.10 m。
- Assumption：轨头在局部横断面是连续高起峰，点云高度/密度足够。
- Failure：站台边缘/电缆槽伪峰、遮挡、道岔、多层结构、z percentile 截掉轨头。
- Why：利用铁路几何先验，成本低且可解释，优于无约束聚类直接输出资产。
- Alternatives：`open3d_baseline.detect_open3d_rail_baseline()`；`gislab_baseline.normalize_gislab_railtrack_points()`，均位于 benchmark path。
- Downstream：峰位置变动会影响 pair、graph identity 和 geometry；必须重新 review。

## 4. Robust rail line / rail-top fitting

`_robust_linear_fit()` 先 `polyfit`，以残差 MAD 建阈值：

\[
\tau=\max(0.02,3\times1.4826\times MAD)
\]

保留 `|r|≤τ` 后重新拟合并报告 P90。轨顶 sampling 可用 anchored/local quantile 或 height-grid max，且 core mask 排除上下文点。

- Why：轨头点少且有高噪/异物，普通最小二乘易被少数点拉偏。
- Failure：有效点太少、窗口混入邻轨、量化分辨率低。
- Tests：`test_rail_top_fit.py` 覆盖高离群点、quantile、core-only percentile、grade-following。
- Change impact：rail z、cross fit 是 TrackGraph observation 的控制值；不可只重导 mesh 而不重审 graph。

## 5. Rail pairing

- Problem：把单个横向峰组成物理双轨。
- Idea：枚举中心间距在 1.35–1.65 m 的 peak pair；考虑 nominal gauge + rail-head width、跨高、位置修正、共同纵向支持；按 policy 排序，避免 peak 被多次使用。
- Implementation：rail candidate helpers 与 `_paired_longitudinal_support()`。
- Formula：中心距近似 `gauge + head_width`；配置默认 `1.435+0.073=1.508 m`。
- Guard：nominal constraint 的每轨修正不得超过 0.06 m；crosslevel ≤0.20 m。
- Important：pair continuity 默认 diagnostic-only，不生成缺失几何。
- Failure：道岔、交叉渡线、相邻伪峰；错误 nominal 约束可把两根轨向错误位置拉。
- Alternative：Open3D DBSCAN/GISLab peak grouping；只能比较，不能自动升级 authoritative。

## 6. Track identity matching 与 seam reconciliation

- Problem：分段观测如何属于同一全局 track。
- Idea：以 lateral/vertical/gauge/chainage cost 构造矩阵，用 `scipy.optimize.linear_sum_assignment` 做一对一匹配；超阈值者不匹配。小 seam 允许带 provenance 的修正。
- Implementation：`track_graph.py::_match_observations()`、`_reconcile_observation_seams()`、`audit_track_graph()`。
- Parameters：identity gap 10 m、lateral 1 m、vertical 0.5 m、gauge 0.15 m；seam thresholds 见 04。
- Failure：线路分合、交叉、segment frame 翻转；系统选择显式断点而不是补线。
- Downstream：graph ID 是 registry 和 Web picking 的身份基础，算法变化是破坏性 schema-level change。

## 7. Parametric rail sweep

- Problem：从中心线和钢轨截面生成可导出的实体表面。
- Idea：对中心线计算局部切线与水平法向，在每个 ring 放置二维 profile，连接相邻 rings 成 quads，并封端。
- Implementation：`algorithms/mesh.py::rail_profile()`、`sweep_mesh()`；权威调用在 `track_graph_mesh.py`。
- Parameters：60 kg dimensional envelope、0.5 m centerline step、gauge/rail head width。
- Assumption：Z-up，中心线局部切线非零，断面沿水平法向放置。
- Failure：重复点导致 undefined tangent；急弯导致截面翻折；段界 caps 重叠导致闪面。
- Alternative：`algorithms/track.py::build_parametric_track()` 是旧直线 baseline。
- Downstream：profile 修改会影响 rail–sleeper contact 与 UE 尺寸，必须重跑 assembly audit。

## 8. Platform segmentation/fitting

当前代码能确认的只有 layout geometry 到 mesh：`reviewed_scene.py::_extruded_polygon_xy()` 等。自动点云分割、分段高程 RANSAC/真实边界拟合在当前仓库为 **UNKNOWN/未实现**。

- Existing input：人工审核的 polygon/box/panel 参数。
- Validation：polygon 非退化、winding 修正、registry evidence；DCC scene relationships。
- Alternative：没有可确认生产实现。不能把 `platform_fit_p90_max_m=0.08` 误认为算法。
- Downstream：未来实现必须输出保持 `reviewed-scene-layout.v1` 或新增 schema adapter，不能直接写 mesh 绕过 registry。

## 9. Vertical candidates 与分类

- Problem：找接触网支柱/雨棚柱/建筑边缘等高而窄对象。
- Idea：世界 XY 以 0.15 m 栅格统计 z span/point count，膨胀后连通域；限制 footprint≤1.5 m、height≥4 m、points≥30。
- Implementation：`linear_candidates.py::_vertical_candidates()`。
- Output：`unclassified_requires_photo_evidence`，不是类别。
- Classification：`annotation_tasks.py` 定义人工标签；`review_agreement.py` 汇总双人一致性。自动 classifier 为 UNKNOWN。
- Failure：相邻柱合并、建筑边缘/树干误检、稀疏柱漏检。

## 10. Canopy plane/support reconstruction

当前生产实现只有审核 layout 的 column/beam/roof primitives 和 Blender 关系审计。自动多平面屋面提取、柱网规则化为 **UNKNOWN/未实现**。

- Existing QA：`blender/audit_scene_relationships.py` 用 BVH/ray/nearest distance 检查 column–platform、column–beam、beam–roof、stair/enclosure interfaces。
- Existing repair：`repair_scene_relationships.py`，状态仍是 `repaired_pending_post_audit_and_visual_review`。
- Design rule：任何未来自动屋面算法不能用一个贯穿 200 m 的平面替代多个证据段；应输出独立资产/边界和证据。

## 11. Catenary reconstruction

- Candidate algorithm：`_cable_candidates()` 在 cross–Z 栅格按纵向 bin 覆盖率找细线。
- Parameters：Z 70–99.9 percentile、cross/z bin 0.10 m、long bin 0.50 m、coverage 0.14、cross-section span≤0.8 m。
- Output：未分类 cable candidates。
- 完整悬链线/腕臂/吊弦拓扑和物理方程实现：**UNKNOWN/未实现**。
- Rules：`rules.default.json` 要求导线连续，推断跨段标 `rule_inferred`。
- Downstream：若接入物理 fitter，应仍保留观测/推断区间，而不是把拟合曲线全标 observed。

## 12. Image projection、visibility 与 multi-view

- Projection：equirectangular `u=(atan2(x,z)/(2π)+0.5)W`，`v=(0.5-asin(y/r)/π)H`。
- Calibration：穷举 Euler order、大小写内/外旋表达、pose direction、48 个有符号轴排列，以 median RGB MAE 排序。
- Implementation：`projection.py`。
- Failure：相似颜色产生假极小；动态物体/曝光；无 RGB；pose 整体偏移。
- Guard：输出仅是 hypothesis，必须多相机和结构边视觉接受。
- Visibility/occlusion：当前 `_save_overlay()` 仅按距离从远到近画 marker，不是严格深度缓冲/遮挡求解；完整 visibility/occlusion 和 multi-view fusion 为 UNKNOWN。

## 13. Point-to-model QA

- Current rail evaluation：`rail_holdout_metrics.py` 以空间 voxel hash 分 train/holdout，采样 model，cKDTree/assignment 计算支持和重复性；`rail_production_regression.py` 比 baseline/candidate。
- Generic full-scene point-to-model：当前核心发布 QA 中没有统一实现；项目 `quality` 的 platform/canopy P90 尚未接通自动 fitter。标记 **部分实现**。
- Important：holdout voxel 必须整体属于一侧，防同一局部点泄漏到训练和测试。

## 14. Mesh duplicate/degenerate/normal checks

- `mesh_audit.audit_obj()`：非有限、非法 index、退化三角形、重复 face、重复 object name；duplicate vertices 仅报告。
- `blender/prepare_interchange.py::clean_mesh_object()`：DCC 清理/三角化/导出。
- Limitations：核心 OBJ audit 不检测跨对象共面重叠、运行时 Z-fighting、法线/背面视觉问题；代码明确要求 back-face culling + 六固定视角。
- Normals/winding：reviewed polygon 会纠正 XY winding；一般 OBJ 未写显式 normals，最终仍依赖 DCC 计算/检查。

