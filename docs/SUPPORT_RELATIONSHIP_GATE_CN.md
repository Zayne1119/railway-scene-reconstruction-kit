# 承托关系门禁

`audit-support-relationships` 用于验证柱脚、杆件、基础、平台和屋面之间的候选承托关系。它解决的核心问题是：几何距离很近，并不等于已经识别出真实承力对象。

门禁要求每条关系明确提供：

- 已注册且带几何的承托对象；
- 有符号接触残差；
- 承托对象语义已解析；
- 承托对象的识别方法；
- 至少一种允许的直接证据；
- 候选状态，不能包含 `accepted`、`formal` 或 `final` 等状态污染。

最近距离、包围盒相交和邻近对象只能用于发现候选，不能单独建立承托关系。输出始终保留 `formal_acceptance=false`。

```powershell
railway-recon audit-support-relationships `
  --registry asset_registry.json `
  --patch candidate_support_patch.json `
  --output support_relationship_audit.json `
  --maximum-contact-residual-m 0.05 `
  --minimum-direct-evidence-kinds 2
```

通过时命令返回成功；任何关系失败时写出完整拒绝原因并以非零状态结束。输出文件已存在时拒绝覆盖。

输入补丁至少需要：

```json
{
  "formal_release": false,
  "delivery_allowed": false,
  "status": "review_candidate",
  "relations": [
    {
      "id": "REL-COLUMN-RESTS-ON-DECK",
      "type": "rests_on",
      "source": "COLUMN",
      "target": "DECK",
      "status": "review_candidate",
      "evidence_status": "direct_interface_candidate",
      "support_evidence": {
        "direct_contact_observed": true,
        "contact_residual_m": 0.008,
        "owner_semantics_resolved": true,
        "owner_identification_method": "observed_boundary",
        "evidence_kinds": [
          "full_density_point_cloud",
          "manual_panorama_review"
        ],
        "proximity_only": false
      }
    }
  ]
}
```

该门禁不生成几何，也不把候选提升为正式资产。它应位于接口候选生成之后、Registry 合并和模型发布之前。
