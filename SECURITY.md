# Security and data disclosure

Do not commit production point clouds, panoramas, camera trajectories, precise coordinates, station names, asset registers, customer models, credentials or deployment IDs.

Report a suspected disclosure privately to the repository owner. Do not open a public issue containing sample data or paths. Before every push run:

```powershell
railway-recon safety-check --root .
```

The first GitHub repository should be private. A public release requires a separate review of data authorization, image privacy, site security and third-party licenses.

