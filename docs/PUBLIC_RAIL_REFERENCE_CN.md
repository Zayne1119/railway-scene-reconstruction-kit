# 公开点云数字类别评价契约

本轮核查未找到 **Rail 名称与数字标签的显式对应关系**。不能将类别列表的第二项推断成 `classification == 1`。因此当前目标固定为 `numeric_class_1` 探针，不把它的 F1/IoU 写成钢轨精度。所有数值类的点数、被选中点数均保留。

[数据论文](https://www.nature.com/articles/s41597-025-06392-9)的标签说明给出 Rail 的资产含义；实例覆盖段明确说明 class 0 是未分类背景。[Table 3](https://www.nature.com/articles/s41597-025-06392-9/tables/3)和[固定版本类别图](https://raw.githubusercontent.com/Arshia-Gha/SemanticRail3D_Dataset/ff7d48c7518646f5093b1de57ce89f8969f9d25b/Data_Info/class_distribution_log_labeled.png)只有类名与计数，没有数字 ID。

来源、定位和已读取文件的哈希写入 [事实记录](../src/railway_recon/resources/public-rail-reference-facts-v1.json)。`PUBLIC_RAIL_REFERENCE_CONTRACT` 同时包含该文件原始字节 SHA256 和完整记录，供实验协议绑定。事实记录是我们的核查记录，不冒充上游映射文件。

## 接口与两个口径

```python
evaluate_public_rail_mask(
    classification, predicted_mask,
    target_class_ids=(1,), ignored_class_ids=(0,),
)
```

数组必须一维、同长度；标签必须为非负整数，预测必须为布尔值。拒绝目标/忽略类重叠、重复 ID、浮点概率及隐式标签重编码。输入不改写。

- `all_points`：完整点云。所有非目标数字类（包括 0）按非目标参考标签计数；FP 表示与发布标签不一致，不意味着未分类点已被独立确认不是钢轨。
- `annotated_only`：只排除调用者明确指定的忽略类。本轮为 0；这是本项目披露的辅助口径，不宣称官方统一评分规则，也不证明剩余标注完整无误。

两者都输出 `tp/fp/fn/tn`、`precision/recall/f1/iou`、评价/参考正例/预测正例点数。分母为 0 返回 JSON `null`，不以 1 填充。顶层保留原始总点数、忽略点数、忽略类中被选中点数和 `per_numeric_class`。未指定忽略类时两个口径相同，**不会自动删除 class 0**。

只在候选完成后读取分类标签进行评价；标签、实例 ID 和由它们生成的子集不得进入候选提取。该模块不做配准、距离容差匹配、实例关联或连接关系评价。单个公开 train 片段是工程开发样本，不是独立现场性能结果；点数不作为独立样本数做显著性推断。
