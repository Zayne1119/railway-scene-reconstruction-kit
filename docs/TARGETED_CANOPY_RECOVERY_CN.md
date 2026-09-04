# 复核证据驱动的雨棚定向恢复

当自动语义分类将真实雨棚柱误判为误检时，不能直接修改原始预测，也不能据此生成贯穿全站的规则雨棚。本阶段读取人工/照片复核覆盖层，以确认柱为种子，在对应站台范围内重新搜索屋面点和柱网。

```powershell
railway-recon targeted-canopy-recover `
  --project projects/site-b/project.json `
  --segment s2100_2150m `
  --semantic-review projects/site-b/workspace/reports/s2100_2150m_vertical_conflict_semantic_review.json
```

规则：

- 原始语义预测保持不可变，复核结果作为覆盖层读取。
- 柱网间距由多个复核种子的长基线整数细分并最小二乘校正。
- 复核柱可生成候选几何；推断柱必须同时获得局部垂直点云支持。
- 无点云支持的规则网格位置只写入报告，不生成几何。
- 屋面只在目标站台的空间包络中提取，并按真实连续观测区段出候选网格。
- 结果不自动写入正式资产注册表；审查通过后另行合并。

对尚未得到局部点云支持的规则柱位，应继续投影到已标定全景，而不是直接补柱：

```powershell
railway-recon targeted-canopy-photo-evidence `
  --project <project.json> `
  --segment <segment-id> `
  --recovery-report <targeted-canopy-recovery.json> `
  --projection-consensus <projection-consensus.json>
```

## 审查通过后的版本化合并

固定视角视觉复审通过后，使用显式授权参数生成新的候选场景。该命令会统一不同 OBJ 的全局原点、审计柱脚—站台接触、检查模型对象与资产 ID 的一一映射，并在修改工作资产库前保存历史副本：

```powershell
railway-recon targeted-canopy-integrate `
  --project <project.json> `
  --segment <segment-id> `
  --recovery-report <targeted-canopy-recovery.json> `
  --visual-review <targeted-canopy-node-visual-review.json> `
  --release-id <versioned-candidate-id> `
  --approve-candidate-merge
```

安全门禁：

- 缺少 `--approve-candidate-merge` 时拒绝写入；
- 视觉门禁和局部节点链门禁必须同时通过；
- 只允许 3 个明确复核的柱位进入当前合并；
- 被否决的规则柱位一旦出现在 OBJ 中，映射审计立即失败；
- 输出仍为 `candidate`，不会伪装成冻结发布或工程验收模型。
