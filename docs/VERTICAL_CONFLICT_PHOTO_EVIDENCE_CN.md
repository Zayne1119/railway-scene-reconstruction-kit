# 站台接触竖直候选的照片复核

`vertical-conflict-photo-evidence` 专门处理“自动分类为误检，但几何底部实际接触站台”的冲突候选。工具从站台接口审计读取冲突 ID，将候选完整高度、底部足迹和柱脚投影到已标定全景，并生成逐候选五视角证据页。

```powershell
railway-recon vertical-conflict-photo-evidence `
  --project <project.json> `
  --segment <segment-id> `
  --projection-consensus <consensus.json>
```

证据页中：橙线表示点云候选完整高度，洋红框表示底部足迹，绿色十字表示底部，青色十字表示顶部。

投影窗口只帮助人找到同一个现场构件，不会自动修改原始预测、生成模型或写资产注册表。复核结论应进入独立 override/review 文件，从而保留原始模型输出和实验可追溯性。
