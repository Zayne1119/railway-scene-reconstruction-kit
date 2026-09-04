# 站台缺口与遮挡 Mesh Gate

点云中的空白不能直接解释为楼梯、电梯或站台洞口。柱子、标牌、栏杆、采集车和视线遮挡都会在站台水平面上留下空白；如果直接按空白切网格，就会产生破面。

当前 pipeline 使用两级门控：

1. `analyze-platform` 提取被合格水平面完整包围的缺口，并检查其空间规律。
2. 同一横向位置、固定纵向节距重复的缺口标记为 `periodic_occlusion_pattern_not_opening_candidate`，`mesh_action` 为 `preserve_platform_surface_do_not_create_opening`。
3. 非周期缺口继续由 `platform-photo-evidence` 投影到已标定全景。红框为独立待复核缺口，橙框为周期性遮挡。
4. 只有独立照片或点云证据明确支持真实开口时，后续网格阶段才允许切孔。

输出中的框线只是证据窗口，不是资产确认。该规则用于防止新项目中出现由柱网或采集遮挡造成的站台破面。
