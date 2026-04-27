# Compare Neighbor Exact Phase Targets

## Objective

Resolve `compare-neighbor-exact-phase-targets-057`: use the corrected phase-target scope and overlap reporting to compare nearby exact cache-write phase targets before attempting another runtime experiment.

## Method

Ran sequential METAL graph profiles with `--phase-cutpoints` and exact `--phase-target` selectors at context length 700:

- `local-layer11`
- `local-layer12`
- `shared-source-layer13`
- `shared-source-layer14`

Artifacts are local/untracked under `benchmarks/` with names:

- `gemma4-metal-decode-graph-local-layer11-phase-overlap-compare-700.json`
- `gemma4-metal-decode-graph-local-layer12-phase-overlap-compare-700.json`
- `gemma4-metal-decode-graph-shared-source-layer13-phase-overlap-compare-700.json`
- `gemma4-metal-decode-graph-shared-source-layer14-phase-overlap-compare-700.json`

## Results

All profiles preserved 7 MetalGraph batches and 0 raw gate/up runners.

| Target | Phase sources | Phase ms | KV projection ms | Conflicts | Interpretation |
|---|---:|---:|---:|---:|---|
| local-layer11 | 236 | 3.222 | 3.140 | 6 | real but small |
| local-layer12 | 256 | 3.142 | 3.069 | 6 | real but small |
| shared-source-layer13 | 888 | 14.503 | 14.422 | 6 | dominant exact target |
| shared-source-layer14 | 296 | 5.011 | 4.905 | 6 | medium exact target |

Overlap reporting confirmed that each phase bucket maps back to its claimed exact parent category.

## Decision

The next runtime work, if any, should focus on `shared-source-layer13` rather than local-layer12. However, the hot subphase is still `kv_projection`, and prior rejected experiments already covered the obvious K/V projection reshapes, producer materialization, prepacked K/V repeats, and raw Metal runners.

Do not start another evo runtime child until a new transformation is identified that is not merely a repeat of those rejected projection variants.

## Next Candidate

Inspect shared-source layer13’s special role in full-length K/V sharing and cache production. Look for a graphable transformation that changes dependency shape or avoids redundant work without splitting graph batches. If no non-rejected transformation appears, mark this branch exhausted and return to higher-level benchmark strategy.
