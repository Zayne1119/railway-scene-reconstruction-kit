# 站区竖直候选合并、四分类与候选资产图

`detect-linear` 输出的是几何候选，不是已经确认的铁路资产。曲面雨棚、建筑立面和屋面肋线都可能在二维栅格内产生较大高差，因此不能只按“高度大于阈值”识别柱子。

## 新增处理

```powershell
railway-recon classify-vertical `
  --project D:\path\to\project.json `
  --segment s2100_2150m
```

该命令读取原始分段点云、`*_linear_candidates.json` 和同段钢轨候选（若存在），输出：

- `*_vertical_hypotheses.json`：合并后的候选、四分类分数、置信度和解释原因；
- `*_vertical_hypotheses.png`：线路坐标系中的候选诊断图；
- `*_vertical_candidate_asset_graph.json`：柱、候选线缆、钢轨和候选柱网之间的证据关系图。

四类假设为 `catenary_support`、`canopy_column`、`building_edge` 和 `false_positive`。分类使用主轴竖直度、沿高度横向漂移、竖向连续占据率、沿线密度/周期、钢轨横向位置及候选线缆邻近关系。

## 安全边界

该阶段不会写入正式资产注册表。输出状态为 `automatic_hypotheses_only_no_asset_registry_write`；没有照片复核、屋面提取或人工确认时，不得把候选图当作资产真值。尤其是雨棚肋线也可能进入线缆候选，因此“靠近线缆”只是一条弱证据。
