# 站台边界与安全线照片证据

`platform-photo-evidence` 将站台每个 5 m 拟合段的轨侧边、外侧边及安全线探测位置投影到已标定全景图，并计算轨侧定向边缘与暖色线性响应。

```powershell
railway-recon platform-photo-evidence `
  --project <project.json> `
  --segment <segment-id> `
  --projection-consensus <consensus.json>
```

青色为轨侧边投影，紫色为观测外侧边，橙色为向站台内部偏移的安全线探测位置。强边缘不等于物理边界，橙黄响应也可能来自机械设备、标牌、道砟或立面材料，因此所有结果只写候选证据图，不生成 Mesh 或正式资产。楼梯、电梯等内部开口不由本阶段推断。
