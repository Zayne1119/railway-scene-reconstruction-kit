# 雨棚柱顶多视角照片证据

`canopy-photo-evidence` 将已经通过点云周期性筛选的雨棚柱顶投影到多张已标定全景图，生成逐柱证据页，并沿场景真实横向计算定向边缘响应。

```powershell
railway-recon canopy-photo-evidence `
  --project <project.json> `
  --segment <segment-id> `
  --projection-consensus <multi-camera-consensus.json>
```

输出包括逐柱逐相机指标、候选证据图、九柱总览和逐柱证据页。“跨视角边缘支持”只表示照片中存在与横向结构方向一致的边缘。屋檐、阴影、标牌和横梁都可能产生该响应，因此不能单独确认梁，也不能直接生成模型几何。局部最优投影约定与共享约定不一致的相机只作为参考，不参与跨视角表决。
