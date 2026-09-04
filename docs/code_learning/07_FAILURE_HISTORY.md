# 07｜失败 → 修复 → 制度

## 证据等级说明

当前 Git 历史只有一个提交 `9d541b7`，而大量新代码仍在工作树中；因此无法从 commit 序列可靠恢复每次修改的准确日期。下表使用：

- **L1**：当前测试明确注入/复现该失败。
- **L2**：当前代码/配置明确存在对应 guard/repair。
- **L3**：仓库历史报告或脚本名称支持，但根因时间线不能从 Git 证明。

“早期修复”若无代码证据，明确写 UNKNOWN，不编造历史。

## 1. Rail discontinuity（L1/L2）

- SYMPTOM：50 m 段界钢轨断开、切线突变或缺失一段。
- ROOT CAUSE：局部 segment 独立坐标和候选，缺少全局身份/接缝合同；具体项目首次触发条件 UNKNOWN。
- EARLY FIX：直接延长/拼接 mesh 的确切实现历史 UNKNOWN；`algorithms/track.py` 的直线 baseline 是可见的早期简化路径。
- FINAL SYSTEMIC FIX：TrackGraph observations/tracks/seams；小 seam 带 provenance reconcile；错误 family 不 force-connect；graph mesh 按全局中心线一次生成。
- PREVENTION：`test_half_metre_seam_is_detected`、`test_small_seam_is_reconciled...`、mesh graph hash/binding checks。

## 2. Track overlap / two tracks on top of each other（L1/L2）

- SYMPTOM：两套轨道中心接近且长距离重叠，网页看到双层钢轨/轨枕。
- ROOT CAUSE：segment 内/跨段身份重复，或旧 mesh 与新 mesh 同时保留。
- EARLY FIX：人工隐藏/删除的历史 UNKNOWN。
- FINAL SYSTEMIC FIX：Hungarian one-to-one matching；identity order；duplicate track audit；TrackGraph asset replacement 清理冲突 ID；Blender integration 按 asset ID 管理对象。
- PREVENTION：`test_overlapping_duplicate_tracks_fail`；duplicate thresholds 0.25 m/2 m；registry unique IDs。

## 3. Canopy gap / broken roof-support interface（L1 fixture/L2）

- SYMPTOM：柱帽不接梁、梁不接屋面、吊顶/屋面出现洞或悬空。
- ROOT CAUSE：各构件独立生成，存在性检查无法证明接触关系。
- EARLY FIX：单纯检查对象数量；fixture `canopy_interfaces_fail.json` 专门说明“对象都在但接口失败”。
- FINAL SYSTEMIC FIX：`blender/audit_scene_relationships.py` 用 signed gap/BVH/coverage 检查 column–platform、column–beam、beam–roof；repair 后状态仍 pending re-audit。
- PREVENTION：quality gate fixtures、`relationships.contact_status()` unit tests、DCC 非零退出。

## 4. False roof filling / one roof through 200 m（L2/L3）

- SYMPTOM：无证据区也被大平面填满，形成贯穿式假屋面。
- ROOT CAUSE：把“屋面大致存在”误编码为单一长 polygon；具体早期代码历史 UNKNOWN。
- EARLY FIX：人工切面历史 UNKNOWN。
- FINAL SYSTEMIC FIX：current reviewed layout 以显式独立资产/geometry 生成，不自动补未知空间；evidence/limitations 随资产进入 registry。
- PREVENTION：当前**没有自动屋面分段 gate**，仍是能力缺口。第二场景必须新增 asset interval/overlap/coverage QA，不能宣称已完全防住。

## 5. Floating column（L2/L3）

- SYMPTOM：柱底离开站台或柱顶没接到梁/屋面。
- ROOT CAUSE：柱中心/高度单独拟合或导出轴/原点错误，缺少关系 QA。
- EARLY FIX：手动移动对象的历史 UNKNOWN。
- FINAL SYSTEMIC FIX：scene relationship audit；`contact_status()` 正 gap=空气、负 gap=穿透；repair 只修确定边界并要求再审。
- PREVENTION：relationship tests 和 DCC report；自动分类/柱网 fitter 仍不完整。

## 6. Stair/platform interface（L2/L3）

- SYMPTOM：楼梯顶步与站台错台、栏板开口不接、围护悬空。
- ROOT CAUSE：楼梯、站台、guard/enclosure 被当独立 mesh。
- EARLY FIX：具体历史 UNKNOWN。
- FINAL SYSTEMIC FIX：`audit_scene_relationships.py` 检查 stair–platform surface、stair–guard distance、enclosure–platform/roof interval。
- PREVENTION：关系配置阈值 + DCC fail；核心 pytest 只保护纯关系分类函数，不覆盖 Blender 几何全流程。

## 7. Z-fighting / flashing faces（L2/L3）

- SYMPTOM：UE/Web 视角移动时表面闪烁。
- ROOT CAUSE：跨对象共面重叠、段界 cap 重合、旧/新 mesh 并存；OBJ 基础 audit 无法检测全部情况。
- EARLY FIX：`merge by distance`/人工清面可能存在于 DCC；准确时间线 UNKNOWN。
- FINAL SYSTEMIC FIX：连续 global mesh 减少段界 cap；Blender integration/prepare interchange 清理；对象 ID replacement；固定视角 + back-face culling 视觉验收。
- PREVENTION：`audit_obj` 仅防单文件重复/退化；其 limitations 明确说不能检测 cross-object coplanar overlap/runtime flicker。因此最终视觉 gate 仍不可省。

## 8. Normal / winding（L2/L3）

- SYMPTOM：背面消失、屋底大片破面、UE 光照异常。
- ROOT CAUSE：polygon winding 反向、负缩放对象、OBJ 无显式 normals、DCC 变换未应用。
- EARLY FIX：手工双面材质历史 UNKNOWN。
- FINAL SYSTEMIC FIX：`reviewed_scene._extruded_polygon_xy()` 根据 signed area 纠正 winding；Blender integration 有 negative closed asset repair；interchange 清理/三角化；固定视角背面剔除验收。
- PREVENTION：当前核心 `audit_obj` 不验证法线，是已知剩余风险。

## 9. Photo evidence mismatch（L2）

- SYMPTOM：点云 overlay 在全景中镜像/旋转/整体错位，门窗贴到错误位置。
- ROOT CAUSE：pose direction、Euler order、轴排列约定不明确，颜色相似造成假极小。
- EARLY FIX：猜一个固定轴约定的历史 UNKNOWN。
- FINAL SYSTEMIC FIX：`calibrate_projection()` 搜索多个 convention，输出 top results 和 overlay，并把结果明确标成 hypothesis requiring visual acceptance。
- PREVENTION：`test_projection.py` 保护 equirectangular 方向与 48 permutation；多相机验收仍是人工制度。

## 10. Global trajectory collapse / local frame smear（L1/L2）

- SYMPTOM：某段横向峰压成一团、轨道中心横摆、段方向不稳定。
- ROOT CAUSE：短段 PCA 把相机横向摆动/站台行驶路径当线路方向。
- EARLY FIX：普通 PCA `fit_corridor_frame()`。
- FINAL SYSTEMIC FIX：dominant-axis regression + 100 m before/after context；graph 使用曲线 RouteSampler。
- PREVENTION：`test_vertical_route_uses_y_as_predictor_and_preserves_direction`；graph canonical direction gate。

## 11. Threshold leakage（L1/L2）

- SYMPTOM：用测试区调阈值后再在同一区报告提升，结果虚高；同一 voxel 出现在 train/holdout。
- ROOT CAUSE：空间相邻点泄漏、实验命令/参数未冻结、外 holdout 已被观察。
- EARLY FIX：普通随机点划分历史 UNKNOWN。
- FINAL SYSTEMIC FIX：稳定空间 voxel hash、完整 voxel 单边归属、nested holdout、experiment/prediction lock、vertical follow-up 明示 post-holdout limitation。
- PREVENTION：`test_same_voxel_never_crosses...`、experiment lock、prediction lock、vertical follow-up tests。

## 12. Release contamination（L1/L2）

- SYMPTOM：GitHub 带入原始点云/模型/私有 IP；发布模型与 registry 不同版；candidate 混入 release。
- ROOT CAUSE：工作区、公开源码和交付物边界不清；仅凭文件名交接。
- EARLY FIX：`.gitignore`，但单靠 ignore 不够。
- FINAL SYSTEMIC FIX：`safety_check()` 扫 banned extension/name/path/IP/secret；frozen accepted registry；quality gate artifact hashes；Web acceptance hash verification。
- PREVENTION：safety/release/quality-gate/Web tests。

## 为什么系统变成现在这样

它不是“算法越来越复杂”，而是每次失败都暴露一个缺少的合同：局部候选缺身份，于是有 TrackGraph；几何存在不等于接口正确，于是有关系审计；漂亮 mesh 不等于有证据，于是有 registry；报告 pass 不等于交付未被替换，于是有 hash-bound gate。第二场景应优先复用这些合同，再重新标定 detector，而不是复制第一场景的最终 mesh 或阈值。

