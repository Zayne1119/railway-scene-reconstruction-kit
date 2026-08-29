# GISLab RailTrack baseline adapter

This directory records the reproducible environment used for the external
`GISLab-ELTE/railroad` RailTrack baseline. It does not vendor or modify the
upstream source.

Frozen provenance:

- repository: `https://github.com/GISLab-ELTE/railroad`
- upstream commit: `3557ecff1bd284108c7a833cce2b50ec9fcd5189`
- LAStools submodule commit: `9bdc92c73047b46be25e5c2ed4abda2521e30fba`
- license: BSD-3-Clause
- build image: Ubuntu 22.04, an operating system covered by upstream CI
- executable: `railroad_benchmark`
- algorithm selector: `RailTrack`

The strict run is invoked with the original upstream executable and no source
patches. The `--shift` option is mandatory for this project's projected map
coordinates. Raw outputs, logs, elapsed time, exit code and input hashes belong
under the ignored
`benchmarks/local/site-a/baselines/gislab_railtrack_v1_frozen_local/` tree.

The shared preprocessing applies the already-frozen longitudinal range and rail
height envelope, then uses a rigid XY rotation so corridor longitudinal is the
local X axis. It does not supply a rail position, rail pair, semantic label,
model geometry, nominal-gauge correction or manual seed. The original
world-coordinate attempts remain preserved under the sibling
`gislab_railtrack_v1_frozen/` experiment directory.

The upstream executable returns detected rail points in `RailTrack.laz`. A
separate, explicitly reported normalization step converts those points into the
common `railway.rail-candidates.v1` schema used by the frozen holdout evaluator.
That conversion is evaluation plumbing, not part of the upstream method.

By default, the runner expects the project's own frozen rail-candidate reports
under `benchmarks/local/site-a/reference/rail_candidates/`. Both the benchmark
root and reference-report root can be overridden with command parameters; no
machine-specific absolute path is embedded in the public runner.

Run the frozen 4 x 50 m train/holdout experiment from the repository root:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass `
  -File scripts/run_gislab_holdout_experiment.ps1 `
  -OutputName gislab_railtrack_v1_frozen_local
```

The runner caps each upstream invocation at 600 seconds. Zero detections and
timeouts are written as explicit non-results without adding synthetic geometry.
