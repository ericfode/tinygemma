# Untracked Benchmark Artifact Inventory

## Objective

Resolve `untracked-benchmark-artifact-inventory-071`: inventory untracked benchmark/profile artifacts without deleting, pushing, or committing generated outputs.

## Summary

- Timestamp: `2026-04-27T03:13:17-0700`
- Untracked paths: `56`
- Total untracked bytes: `1048622` (~1.00 MiB)
- Referenced by committed docs/state: `18`
- Tracked benchmark artifacts already in repo: `153` observed before this inventory

## Counts

By category:

- `benchmark-progress-log`: `14`
- `benchmark-result-artifact`: `11`
- `decode-profiler-artifact`: `28`
- `dependency-lockfile`: `1`
- `paired-helper-smoke`: `2`

By suffix:

- `.csv`: `23`
- `.json`: `18`
- `.jsonl`: `14`
- `.lock`: `1`

## Recommendation

- Do not blanket-ignore `benchmarks/*.csv` / `*.json`: this repo intentionally tracks selected benchmark evidence.
- Do not delete generated artifacts until referenced evidence has either been committed or archived outside the repo.
- Treat docs-referenced artifacts as selective-commit candidates only if the project wants the raw evidence to travel with the plan notes.
- Treat uncited progress logs as archive/delete candidates after review.
- Hold `uv.lock` local unless the project adopts a lockfile policy; current project metadata and AGENTS.md use the setuptools/pip editable workflow.

## Inventory

| Path | Category | Size bytes | Referenced by | Suggested disposition |
|---|---:|---:|---|---|
| `benchmarks/beam4-test.csv` | `benchmark-result-artifact` | `285` | — | archive or delete after review unless it becomes cited by a plan/evolution entry |
| `benchmarks/cache-loop-existing-knob-sweep.json` | `benchmark-result-artifact` | `2064` | — | archive or delete after review unless it becomes cited by a plan/evolution entry |
| `benchmarks/cache-loop-raw-bridge-smoke.json` | `benchmark-result-artifact` | `1635` | — | archive or delete after review unless it becomes cited by a plan/evolution entry |
| `benchmarks/e2b-floor-corrected.csv` | `benchmark-result-artifact` | `9267` | — | archive or delete after review unless it becomes cited by a plan/evolution entry |
| `benchmarks/e2b-floor-corrected.csv.progress.jsonl` | `benchmark-progress-log` | `6214` | — | archive or delete after review; progress logs are usually reproducible run telemetry, not primary evidence |
| `benchmarks/e2b-floor-final.csv` | `benchmark-result-artifact` | `285` | — | archive or delete after review unless it becomes cited by a plan/evolution entry |
| `benchmarks/e2b-floor-final.csv.progress.jsonl` | `benchmark-progress-log` | `5583` | — | archive or delete after review; progress logs are usually reproducible run telemetry, not primary evidence |
| `benchmarks/e2b-floor.csv` | `benchmark-result-artifact` | `1764` | — | archive or delete after review unless it becomes cited by a plan/evolution entry |
| `benchmarks/e2b-floor.csv.progress.jsonl` | `benchmark-progress-log` | `4971` | — | archive or delete after review; progress logs are usually reproducible run telemetry, not primary evidence |
| `benchmarks/gemma4-metal-decode-graph-all-local-phase-profile-700.csv` | `decode-profiler-artifact` | `22405` | docs/plans/2026-04-27-profile-phase-target-generalization.md | selective-commit candidate if evidence should travel with docs; otherwise archive externally |
| `benchmarks/gemma4-metal-decode-graph-all-local-phase-profile-700.json` | `decode-profiler-artifact` | `81034` | configs/repo-loop-state.json<br>docs/plans/2026-04-27-profile-phase-target-generalization.md<br>state/evolution-log.md | selective-commit candidate if evidence should travel with docs; otherwise archive externally |
| `benchmarks/gemma4-metal-decode-graph-local-layer11-phase-overlap-compare-700.csv` | `decode-profiler-artifact` | `9527` | — | archive or delete after review unless it becomes cited by a plan/evolution entry |
| `benchmarks/gemma4-metal-decode-graph-local-layer11-phase-overlap-compare-700.json` | `decode-profiler-artifact` | `58255` | configs/repo-loop-state.json<br>docs/plans/2026-04-27-compare-neighbor-exact-phase-targets.md | selective-commit candidate if evidence should travel with docs; otherwise archive externally |
| `benchmarks/gemma4-metal-decode-graph-local-layer12-phase-overlap-700.csv` | `decode-profiler-artifact` | `9524` | configs/repo-loop-state.json<br>docs/plans/2026-04-27-phase-overlap-reporting-and-next-runtime-surface.md | selective-commit candidate if evidence should travel with docs; otherwise archive externally |
| `benchmarks/gemma4-metal-decode-graph-local-layer12-phase-overlap-700.json` | `decode-profiler-artifact` | `58240` | configs/repo-loop-state.json<br>docs/plans/2026-04-27-phase-overlap-reporting-and-next-runtime-surface.md<br>state/evolution-log.md | selective-commit candidate if evidence should travel with docs; otherwise archive externally |
| `benchmarks/gemma4-metal-decode-graph-local-layer12-phase-overlap-compare-700.csv` | `decode-profiler-artifact` | `9525` | — | archive or delete after review unless it becomes cited by a plan/evolution entry |
| `benchmarks/gemma4-metal-decode-graph-local-layer12-phase-overlap-compare-700.json` | `decode-profiler-artifact` | `58234` | configs/repo-loop-state.json<br>docs/plans/2026-04-27-compare-neighbor-exact-phase-targets.md | selective-commit candidate if evidence should travel with docs; otherwise archive externally |
| `benchmarks/gemma4-metal-decode-graph-local-layer12-phase-profile-700-scope-fixed.csv` | `decode-profiler-artifact` | `9527` | configs/repo-loop-state.json<br>docs/plans/2026-04-27-local-layer12-phase-attribution-scope-fix.md | selective-commit candidate if evidence should travel with docs; otherwise archive externally |
| `benchmarks/gemma4-metal-decode-graph-local-layer12-phase-profile-700-scope-fixed.json` | `decode-profiler-artifact` | `52264` | configs/repo-loop-state.json<br>docs/plans/2026-04-27-local-layer12-phase-attribution-scope-fix.md<br>state/evolution-log.md | selective-commit candidate if evidence should travel with docs; otherwise archive externally |
| `benchmarks/gemma4-metal-decode-graph-local-layer12-phase-profile-700.csv` | `decode-profiler-artifact` | `9331` | docs/plans/2026-04-27-profile-phase-target-generalization.md | selective-commit candidate if evidence should travel with docs; otherwise archive externally |
| `benchmarks/gemma4-metal-decode-graph-local-layer12-phase-profile-700.json` | `decode-profiler-artifact` | `50206` | configs/repo-loop-state.json<br>docs/plans/2026-04-27-profile-phase-target-generalization.md<br>state/evolution-log.md | selective-commit candidate if evidence should travel with docs; otherwise archive externally |
| `benchmarks/gemma4-metal-decode-graph-merged-parity-guard-attribution-profile-512.csv` | `decode-profiler-artifact` | `7955` | — | archive or delete after review unless it becomes cited by a plan/evolution entry |
| `benchmarks/gemma4-metal-decode-graph-merged-parity-guard-attribution-profile-512.json` | `decode-profiler-artifact` | `42720` | — | archive or delete after review unless it becomes cited by a plan/evolution entry |
| `benchmarks/gemma4-metal-decode-graph-merged-parity-guard-attribution-profile.csv` | `decode-profiler-artifact` | `7955` | — | archive or delete after review unless it becomes cited by a plan/evolution entry |
| `benchmarks/gemma4-metal-decode-graph-merged-parity-guard-attribution-profile.json` | `decode-profiler-artifact` | `42724` | — | archive or delete after review unless it becomes cited by a plan/evolution entry |
| `benchmarks/gemma4-metal-decode-graph-packed-kv-repeat.csv` | `decode-profiler-artifact` | `4456` | — | archive or delete after review unless it becomes cited by a plan/evolution entry |
| `benchmarks/gemma4-metal-decode-graph-packed-kv-repeat.json` | `decode-profiler-artifact` | `29684` | — | archive or delete after review unless it becomes cited by a plan/evolution entry |
| `benchmarks/gemma4-metal-decode-graph-rolled-cache-700-jit1-experiment.csv` | `decode-profiler-artifact` | `4457` | — | archive or delete after review unless it becomes cited by a plan/evolution entry |
| `benchmarks/gemma4-metal-decode-graph-rolled-cache-700-jit1-experiment.json` | `decode-profiler-artifact` | `29711` | — | archive or delete after review unless it becomes cited by a plan/evolution entry |
| `benchmarks/gemma4-metal-decode-graph-shared-source-layer13-phase-overlap-compare-700.csv` | `decode-profiler-artifact` | `9736` | — | archive or delete after review unless it becomes cited by a plan/evolution entry |
| `benchmarks/gemma4-metal-decode-graph-shared-source-layer13-phase-overlap-compare-700.json` | `decode-profiler-artifact` | `59993` | configs/repo-loop-state.json<br>docs/plans/2026-04-27-compare-neighbor-exact-phase-targets.md | selective-commit candidate if evidence should travel with docs; otherwise archive externally |
| `benchmarks/gemma4-metal-decode-graph-shared-source-layer14-phase-overlap-compare-700.csv` | `decode-profiler-artifact` | `9967` | — | archive or delete after review unless it becomes cited by a plan/evolution entry |
| `benchmarks/gemma4-metal-decode-graph-shared-source-layer14-phase-overlap-compare-700.json` | `decode-profiler-artifact` | `60356` | configs/repo-loop-state.json<br>docs/plans/2026-04-27-compare-neighbor-exact-phase-targets.md | selective-commit candidate if evidence should travel with docs; otherwise archive externally |
| `benchmarks/gemma4-metal-decode-graph-shared-source-windowed-profile-512.csv` | `decode-profiler-artifact` | `7955` | docs/plans/2026-04-25-packed-cache-next-steps.md<br>state/evolution-log.md | selective-commit candidate if evidence should travel with docs; otherwise archive externally |
| `benchmarks/gemma4-metal-decode-graph-shared-source-windowed-profile-512.json` | `decode-profiler-artifact` | `42721` | docs/plans/2026-04-25-packed-cache-next-steps.md<br>state/evolution-log.md | selective-commit candidate if evidence should travel with docs; otherwise archive externally |
| `benchmarks/gemma4-metal-decode-graph-shared-source-windowed-reverted-profile-512.csv` | `decode-profiler-artifact` | `7956` | — | archive or delete after review unless it becomes cited by a plan/evolution entry |
| `benchmarks/gemma4-metal-decode-graph-shared-source-windowed-reverted-profile-512.json` | `decode-profiler-artifact` | `42730` | docs/plans/2026-04-25-packed-cache-next-steps.md<br>state/evolution-log.md | selective-commit candidate if evidence should travel with docs; otherwise archive externally |
| `benchmarks/gemma4-metal-e2b-int8-1000-greedy-softcap-elision-current.csv.progress.jsonl` | `benchmark-progress-log` | `3137` | — | archive or delete after review; progress logs are usually reproducible run telemetry, not primary evidence |
| `benchmarks/gemma4-metal-e2b-int8-1000-greedy-softcap-elision-repeat.csv.progress.jsonl` | `benchmark-progress-log` | `3139` | — | archive or delete after review; progress logs are usually reproducible run telemetry, not primary evidence |
| `benchmarks/gemma4-metal-e2b-int8-1000-prepacked-kv-current.csv.progress.jsonl` | `benchmark-progress-log` | `1258` | — | archive or delete after review; progress logs are usually reproducible run telemetry, not primary evidence |
| `benchmarks/gemma4-metal-e2b-int8-1000-prepacked-kv-repeat.csv.progress.jsonl` | `benchmark-progress-log` | `1258` | — | archive or delete after review; progress logs are usually reproducible run telemetry, not primary evidence |
| `benchmarks/gemma4-metal-e2b-int8-1000-prepacked-kv-repeat2.csv.progress.jsonl` | `benchmark-progress-log` | `1258` | — | archive or delete after review; progress logs are usually reproducible run telemetry, not primary evidence |
| `benchmarks/gemma4-metal-e2b-int8-1000-rolled-cache-warmup600-experiment.csv` | `benchmark-result-artifact` | `677` | — | archive or delete after review unless it becomes cited by a plan/evolution entry |
| `benchmarks/gemma4-metal-e2b-int8-1000-rolled-cache-warmup600-experiment.csv.progress.jsonl` | `benchmark-progress-log` | `3066` | — | archive or delete after review; progress logs are usually reproducible run telemetry, not primary evidence |
| `benchmarks/gemma4-metal-e2b-int8-128-greedy-softcap-elision-current.csv.progress.jsonl` | `benchmark-progress-log` | `1241` | — | archive or delete after review; progress logs are usually reproducible run telemetry, not primary evidence |
| `benchmarks/gemma4-metal-e2b-int8-128-merged-parity-guard-attribution-current.csv` | `benchmark-result-artifact` | `672` | — | archive or delete after review unless it becomes cited by a plan/evolution entry |
| `benchmarks/gemma4-metal-e2b-int8-128-merged-parity-guard-attribution-current.csv.progress.jsonl` | `benchmark-progress-log` | `622` | — | archive or delete after review; progress logs are usually reproducible run telemetry, not primary evidence |
| `benchmarks/gemma4-metal-e2b-int8-128-nosplit-current.csv` | `benchmark-result-artifact` | `672` | — | archive or delete after review unless it becomes cited by a plan/evolution entry |
| `benchmarks/gemma4-metal-e2b-int8-128-nosplit-current.progress.jsonl` | `benchmark-progress-log` | `620` | — | archive or delete after review; progress logs are usually reproducible run telemetry, not primary evidence |
| `benchmarks/gemma4-metal-e2b-int8-128-prepacked-kv-current.csv` | `benchmark-result-artifact` | `672` | — | archive or delete after review unless it becomes cited by a plan/evolution entry |
| `benchmarks/gemma4-metal-e2b-int8-128-prepacked-kv-current.csv.progress.jsonl` | `benchmark-progress-log` | `619` | — | archive or delete after review; progress logs are usually reproducible run telemetry, not primary evidence |
| `benchmarks/gemma4-metal-e2b-int8-128-split-producer-current.csv` | `benchmark-result-artifact` | `672` | — | archive or delete after review unless it becomes cited by a plan/evolution entry |
| `benchmarks/gemma4-metal-e2b-int8-128-split-producer-current.progress.jsonl` | `benchmark-progress-log` | `619` | — | archive or delete after review; progress logs are usually reproducible run telemetry, not primary evidence |
| `benchmarks/paired-exp0005-self-default-hash4.json` | `paired-helper-smoke` | `2976` | configs/repo-loop-state.json<br>docs/plans/2026-04-27-paired-helper-smoke-on-evo-frontier.md<br>state/evolution-log.md | selective-commit candidate if evidence should travel with docs; otherwise archive externally |
| `benchmarks/paired-exp0005-self-hash16.json` | `paired-helper-smoke` | `3283` | configs/repo-loop-state.json<br>docs/plans/2026-04-27-paired-helper-smoke-on-evo-frontier.md<br>state/evolution-log.md | selective-commit candidate if evidence should travel with docs; otherwise archive externally |
| `uv.lock` | `dependency-lockfile` | `150945` | docs/plans/2026-04-27-final-working-tree-and-gate-summary.md<br>state/evolution-log.md | hold local; commit only after explicit lockfile policy because repo uses setuptools/pip workflow today |

## Verification

- Inventory source: `git status --porcelain -z` plus filesystem stat calls.
- No untracked benchmark artifacts were deleted, staged, or ignored by this increment.
