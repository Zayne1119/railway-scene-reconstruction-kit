# Paper Benchmark v1

This directory contains the public, data-safe protocol and templates used to turn a
production railway reconstruction into a reproducible research experiment.

The benchmark layer is deliberately separated from production QA:

- production QA answers whether a deliverable can be opened and handed over;
- the benchmark answers whether geometry, instances, topology, evidence confidence,
  and human effort were evaluated with frozen and reproducible rules.

## Start a private benchmark

```powershell
railway-recon benchmark-init benchmarks/local/site-a `
  --dataset-id railway-site-a `
  --scene-id station-000-200m

railway-recon benchmark-validate `
  --path benchmarks/local/site-a/manifests/dataset-manifest.json

railway-recon benchmark-check --root benchmarks/local/site-a

railway-recon benchmark-freeze --root benchmarks/local/site-a

railway-recon benchmark-evaluate `
  --input benchmarks/local/site-a/experiments/evaluation-input.json `
  --output benchmarks/local/site-a/reports/metrics.json

railway-recon benchmark-make-vertical-tasks `
  --root benchmarks/local/site-a `
  --source s000_50m=D:/private/s000_50m/continuous_candidate_features.json

railway-recon benchmark-lock-vertical-predictions `
  --root benchmarks/local/site-a `
  --annotation-package benchmarks/local/site-a/annotations/vertical_candidates_blind_v1 `
  --source s000_50m=D:/private/s000_50m/classification.json

railway-recon benchmark-make-rail-truth-tasks `
  --root benchmarks/local/site-a `
  --evidence-manifest D:/private/rail-neutral-evidence.json

railway-recon benchmark-lock-experiment `
  --root benchmarks/local/site-a `
  --experiment benchmarks/local/site-a/experiments/g-only.json `
  --binding method_config=D:/private/configs/g-only.json

railway-recon benchmark-detect-open3d-rails `
  --source D:/private/holdout/s0000_0050m.train.laz `
  --reference-report D:/private/reports/s0000_0050m_rail_candidates.json `
  --segment s0000_0050m `
  --output D:/private/baselines/open3d/s0000_0050m.json

railway-recon benchmark-compare-rail-holdout `
  --baseline D:/private/reports/rail-holdout-open3d-v1.json `
  --method D:/private/reports/rail-holdout-v2.json `
  --output D:/private/reports/rail-holdout-comparison-v1.json
```

When no manual labels are available, tune a rail-height estimator only inside the
outer-train clouds. First create a nested whole-voxel split, evaluate every frozen
variant on that nested split, then select with fixed guardrails:

```powershell
railway-recon benchmark-split-point-sources `
  --source s0000_0050m=D:/private/holdout/s0000_0050m.train.laz `
  --output-dir D:/private/holdout/nested_vertical_v1

railway-recon benchmark-select-rail-vertical-fit `
  --variant legacy=D:/private/legacy-evaluation.json=D:/private/legacy.json `
  --variant anchored_q50=D:/private/q50-evaluation.json=D:/private/q50.json `
  --baseline legacy `
  --output D:/private/vertical-selection.json
```

Do not repeatedly tune against the outer holdout. If one outer follow-up exposes an
engineering interaction, preserve the failed report and generate a post-development
audit. Such a result is explicitly exploratory and cannot replace a prospective new-site
test.

Lock legacy predictions before either reviewer returns labels. The locked set is
explicitly recorded as `HITL` and is not eligible to be reported as a fully automatic
baseline. Never distribute the locked prediction directory to blind reviewers.

Rail truth tasks accept only neutral evidence manifests derived from raw point-cloud
slices with both `model_overlay` and `candidate_overlay` set to false. Evaluation input
v2 keeps ground truth and predictions separate, performs class-neutral one-to-one
spatial matching, maps topology IDs after matching, and reports raw rail measurements
separately from rule-constrained outputs.

After both independent CSVs are complete, measure agreement and create the adjudication
queue:

```powershell
railway-recon benchmark-compare-vertical-reviews `
  --annotation-package benchmarks/local/site-a/annotations/vertical_candidates_blind_v1
```

The command rejects incomplete labels, calculates per-field raw agreement and Cohen's
kappa, writes directly agreed items to `consensus_seed.csv`, and writes conflicts to
`disagreements.csv` with empty adjudication fields.

`benchmarks/local/` is ignored by Git. Raw point clouds, panoramas, coordinates,
customer names, survey data, and production models must remain there or in another
approved private location.

## Scientific roles

- Existing repeatedly inspected data: `development_case_study`.
- A new site before tuning: `prospective_blind_test` and `frozen_generalization`.
- A new site after a declared 20–50 m pilot: `calibration_assisted`.
- Absolute world accuracy requires independent check points that were not used for
  registration or fitting.

## Canonical schemas

Runtime JSON Schemas are packaged in `src/railway_recon/resources/`:

- benchmark protocol;
- dataset and spatial split manifests;
- ground truth and experiment definitions;
- metric report and failure case;
- run manifest v2.

See `docs/PAPER_BENCHMARK_CN.md` for the complete Chinese operating procedure.
