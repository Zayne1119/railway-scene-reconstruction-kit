# 柱网播种的雨棚屋面与接触候选

`analyze-canopy` 接在 `classify-vertical` 之后运行。它只使用已经形成稳定周期的雨棚柱网播种屋面搜索，普通区段或没有稳定柱网的区段会闭锁，不会凭空生成屋面。

```powershell
railway-recon analyze-canopy `
  --project D:\path\to\project.json `
  --segment s2100_2150m
```

## 方法

1. 在柱顶附近建立线路坐标系中的高程搜索带；
2. 以 0.5 m 网格统计屋面点的二维连通性；
3. 只保留与周期柱网相接的连通组件；
4. 按沿线分箱记录真实观测区间和横向轮廓；
5. 输出平面残差，但曲面/多层屋面不会被强制压成平面；
6. 计算每根柱顶到屋面候选的最近三维距离；
7. 对柱位横向条带与相邻条带做点密度对比，只在存在明显增量时提出横梁候选。

## 输出与边界

- `*_canopy_structure.json`：屋面组件、轮廓、曲面诊断、柱顶接触和横梁证据检查；
- `*_canopy_structure.png`：俯视轮廓、横断面和逐柱关系图；
- `*_canopy_candidate_graph.json`：柱—屋面—横梁证据检查图。

这些输出仍是候选证据，不会进入正式资产注册表。柱顶接触距离来自同一份点云，只能说明观测几何相接，不能替代独立 Mesh 接口验收。`no_distinct_transverse_member_in_point_cloud` 表示数据中没有足够独立证据，不能解释为现场一定没有横梁。
