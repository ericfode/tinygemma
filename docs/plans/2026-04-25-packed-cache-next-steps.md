# Packed Cache Next-Step Research

Date: 2026-04-25

## Objective

Identify the next credible optimization probes for `tinygrad-gemma` with tokens per second as the hard metric. The near-term target is not aesthetic graph grooming; it is a repeated E2B int8 `METAL` `1000/20` improvement over the refreshed observed `19.424036 tok/s` floor.

## 2026-04-25 research refresh

A read-only refresh originally revalidated `conditional-sliding-roll-parity-036` as the immediate next implementation target. The later parallel probe sweep below supersedes that recommendation: parity was tested and rejected by profile/long-floor evidence.

Additional verified facts:

- Current unpatched `GemmaForConditionalGeneration.generate` has no roll transition: a forced small conditional generation with sliding JIT enabled produced `roll_calls=[]` and `decode_fallback=False` when `tinygrad_gemma.model.roll_sliding_cache_entries` was wrapped.
- The hard E2B benchmark still reaches the conditional path: `scripts/benchmark_gemma4_matrix.py` calls `model.generate(...)`, and `tinygrad_gemma/loader.py` selects `GemmaForConditionalGeneration` for checkpoints with multimodal towers.
- Patch surface is narrow: import `roll_sliding_cache_entries` into `tinygrad_gemma/multimodal.py`, add a one-time `rolled_sliding_cache` guard in `GemmaForConditionalGeneration.generate`, and call `roll_sliding_cache_entries(cache, self.model.language_model.layers)` immediately before the first sliding-window rollout JIT call.
- Use `self.model.language_model.layers`, not `self.model.layers`; the conditional wrapper shape is different from the causal model.
- Profile comparisons must record the tinygrad runtime path. The accepted profile artifact was produced with `/Users/ericfode/src/.tinygrad_research/tinygrad/__init__.py`, while the repo venv imports `.venv/lib/python3.14/site-packages/tinygrad/__init__.py` unless `PYTHONPATH=/Users/ericfode/src/.tinygrad_research:.` is set. Mixing these runtimes makes source-count deltas slippery in the usual way performance archaeology gets slippery: quietly, and with excellent posture.

New exact focused tests recommended for the parity patch:

1. Conditional rolled-cache forward parity: prefill a conditional model with a sliding layer, manually call `roll_sliding_cache_entries(cache, model.model.language_model.layers)`, set `cache.decode_sliding_window=True`, then assert one-token cached logits match full forward.
2. Conditional generate wiring: monkeypatch `tinygrad_gemma.multimodal.roll_sliding_cache_entries`, force `_sliding_decode_start()` below the prompt length, generate enough tokens to cross the switch, and assert exactly one roll call with the sliding layer physically window-sized while the full layer remains absolute.

## 2026-04-25 parallel probe sweep results

The next-step probes were run in isolated worktrees, with METAL profile/throughput gates serialized. Mainline accepted only profiler instrumentation; runtime candidates were rejected unless measured.

Accepted profiler-only result:

- `benchmarks/gemma4-metal-decode-graph-attribution-only-restored-profile-512.json`
- `PYTHONPATH=/Users/ericfode/src/.tinygrad_research:.`
- `2994` source items, `34.319667` ms/token, `7` MetalGraph batches, complete attribution.
- Exact largest categories:
  - `attention_packed_cache_write__role_shared_source__layer_13__type_sliding_attention`: `11.798664` ms, `885` source items, `34.38%`.
  - `attention_packed_cache_write__role_shared_source__layer_14__type_full_attention`: `4.174449` ms, `293` source items, `12.16%`.
  - Local packed writes across layers remain `16.763` ms rollup; MLP remains only `1.584` ms.

Rejected candidates:

| Candidate | Profile result | Throughput evidence | Decision |
| --- | ---: | ---: | --- |
| Conditional sliding-roll parity | `34.896750` ms/token, `2994` source items | merged parity+guard long rows `18.320320`, `18.352303` tok/s | reject |
| Active-cache no-op slice guard | `34.658542` ms/token, `2994` source items | included in merged regression above | reject |
| Flat-slot packed layout `(B,H,L,2*D)` | `35.801833` ms/token, `2994` source items | no long run earned | reject |
| Deferred current-token readback/store ordering | `36.976458` ms/token, `3594` source items | no long run earned | reject |
| Research-tinygrad/runtime comparison lane | `34.109500` ms/token, `2994` source items | not a repo-local runtime patch | research only |

Restored-mainline repeated hard rows:

- `benchmarks/gemma4-metal-e2b-int8-1000-attribution-only-restored-current.csv`: `19.424036` tok/s.
- `benchmarks/gemma4-metal-e2b-int8-1000-attribution-only-restored-repeat.csv`: `19.613212` tok/s.
- Both rows: `generated_tokens=1000`, `measured_decode_tokens=980`, `rollout_jit_count=999`, `decode_fallback=false`, stable hash `aa4944455473e236807f6c711a77a969caf73680ea5ea9725648294e5fd23cfa`.
- Treat the lower row, `19.424036`, as the refreshed observed floor for the unchanged runtime checkpoint. Do not credit profiler-only instrumentation as a runtime speedup.

Next target supersedes the earlier parity-first recommendation: `shared-source-layer13-cache-write-specialization-037`. The layer-13 shared-source sliding attention producer/store path is now the largest exact target. Broad parity/layout/deferred-store hooks should remain rejected unless new tinygrad semantics or a narrower layer-13 hypothesis changes the evidence.

## 2026-04-26 shared-source windowed-allocation result

`shared-source-layer13-cache-write-specialization-037` was implemented as a temporary runtime probe and rejected by profile evidence.

- Candidate artifact: `benchmarks/gemma4-metal-decode-graph-shared-source-windowed-profile-512.json` plus CSV.
- Command: `PYTHONPATH=/Users/ericfode/src/.tinygrad_research:. .venv/bin/python scripts/profile_decode_jit.py --model-dir checkpoints/gemma-4-E2B-int8 --device METAL --context-length 512 --jit-mode 1 --out benchmarks/gemma4-metal-decode-graph-shared-source-windowed-profile-512.json --csv-out benchmarks/gemma4-metal-decode-graph-shared-source-windowed-profile-512.csv`.
- Result: `2994` source items, `7` MetalGraph batches, `34.557625` ms/token.
- Control: attribution-only restored profile remains `2994` source items, `7` MetalGraph batches, `34.319667` ms/token.
- Decision: reject and skip the hard E2B `1000/20` tok/s run; the profile did not earn it.
- Cleanup: candidate runtime/test surface was removed. Post-revert structural profile `benchmarks/gemma4-metal-decode-graph-shared-source-windowed-reverted-profile-512.json` returned to `2994` source items and `7` graph batches, with noisy timing at `34.865500` ms/token.
- Implication: physical cache length was not the source of the layer-13 cost. The profile is measuring the dependency-closed producer/store graph under the cache-write realization scope.

Next target: `layer13-shared-source-producer-store-attribution-038`, a profiler-only split of the layer-13 shared-source scope into producer/norm/RoPE/pack/store components before another runtime patch.

## 2026-04-26 donor optimization transfer audit

Read-only donor: `/Users/ericfode/src/gemma4-tinygrad-opt`. Recipient: `/Users/ericfode/Downloads/tinygrad-gemma`. Metric: real-checkpoint E2B int8 Apple `METAL` autoregressive decode throughput, not full-forward synthetic throughput.

| Donor optimization/evidence | Recipient status | Decision |
| --- | --- | --- |
| Donor `evolution.jsonl` rows: `26.971663` and `528.732270` `tokens_per_second`; donor `benchmark.py` full-forwards 1000 random tokens on Modal CUDA/T4. | Not comparable to recipient `scripts/benchmark_gemma4_matrix.py` autoregressive `generate` on Apple `METAL`. | Research hint only. |
| Final-lane packed K/V cache backing and packed assignment. | Already present and accepted in recipient via `packed-kv-cache-assignment-033` and `packed-kv-lastdim-cache-assignment-034`; recipient uses `.assign(...).realize()`. | Already present / recipient stronger. |
| Fused int8 gate/up and graphable fused int8 K/V projection. | Already present; fused K/V accepted as `graphable-fused-int8-kv-031`. | Already present. |
| Sliding cache roll/window helper surface. | Already present in the recipient's accepted/rejected cache-history surface; broad parity/windowing variants were recently rejected by profile/long-floor evidence. | Do not re-port broadly. |
| Donor packed-cache writes without `.realize()`. | Conflicts with recorded local tinygrad assignment semantics. | Reject as unsafe/obsolete. |
| Donor-style `Tensor.scaled_dot_product_attention` branch. | Tested as temporary env-gated recipient patch; removed after measurements below. | Reject. |

SDPA probe artifacts:

- Manual attention control: `benchmarks/gemma4-metal-e2b-int8-128-sdpa-off.csv`, `measured_decode_tokens_per_second=30.808261`, `output_sha256=1c39dd289bb7363f0f600ae1087ad39926e9733447df72bf03955c92d06066b0`, `decode_fallback=false`.
- SDPA candidate: `benchmarks/gemma4-metal-e2b-int8-128-sdpa-on.csv`, `measured_decode_tokens_per_second=30.769433`, same output hash, `decode_fallback=false`.
- Manual profile: `benchmarks/gemma4-metal-decode-graph-sdpa-off-profile-512.json`, `2994` source items/kernels, `44.466920` ms/token at the context-700 profile point.
- SDPA profile: `benchmarks/gemma4-metal-decode-graph-sdpa-on-profile-512.json`, same `2994` source-item structure, `70.510040` ms/token; source-attributed shared-source attention increased from `19.162958` ms to `38.811997` ms and MLP from `2.963834` ms to `9.669416` ms.

Conclusion: no donor runtime patch currently improves the accepted metric. Keep the accepted floor at `19.424036 tok/s` for repeated E2B int8 `METAL` `1000/20` rows; a future patch must beat that floor with stable hash, `rollout_jit_count=999`, and `decode_fallback=false`, or first reduce the profiled source count below `2994`.

## Historical accepted baseline before the parallel sweep

Runtime checkpoint: `packed-kv-lastdim-cache-assignment-034`.

Hard floor artifact:

- `benchmarks/gemma4-metal-e2b-int8-1000-packed-kv-lastdim-current.csv`
- E2B int8 `METAL`, beam `0`, `1000/20`
- `generated_tokens=1000`
- `measured_decode_tokens=980`
- `measured_decode_tokens_per_second=19.053995`
- `rollout_jit_count=999`
- `decode_fallback=false`
- output hash `aa4944455473e236807f6c711a77a969caf73680ea5ea9725648294e5fd23cfa`

Clean repeat:

- `benchmarks/gemma4-metal-e2b-int8-1000-packed-kv-lastdim-repeat2.csv`
- `19.081371 tok/s`, same hash, `decode_fallback=false`

Profile artifact:

- `benchmarks/gemma4-metal-decode-graph-packed-kv-lastdim-profiled.json`
- `2994` captured source items
- `34.414750 ms/token`
- `7` MetalGraph batches
- complete source attribution
- `attention_packed_cache_write_local=1709`, `16.782780 ms`, `48.77%`
- `attention_packed_cache_write_shared_source=1178`, `16.043710 ms`, `46.62%`
- `mlp=107`, `1.588261 ms`, `4.62%`

Rejected route:

- Producer-side prepacking / RHS materialization increased profile source items to `3009`, worsened profile time to `34.718833 ms/token`, and failed repeated long-floor acceptance (`19.010608` and `18.867781 tok/s`).
- Current high-level tinygrad `assign(...).realize()` is dependency-closed over lazy RHS producers; pre-realizing RHS reproduces the rejected materialization path.

## Source anchors

Current packed cache geometry and update path:

- `tinygrad_gemma/model.py:201-213` — `make_packed_cache_entry` allocates `(batch, heads, length, head_dim, 2)` and exposes `key=packed[..., 0]`, `value=packed[..., 1]`.
- `tinygrad_gemma/model.py:280-302` — `realize_cache_update`, including the single-token packed write:
  `key.squeeze(2).unsqueeze(-1).cat(value.squeeze(2).unsqueeze(-1), dim=-1)` then `packed_cache[:, :, start, :, :].assign(...).realize()`.
- `tinygrad_gemma/model.py:561-620` — K/V projection, norm/RoPE, cache write, and active-cache readback.
- `tinygrad_gemma/model.py:1015-1049` — causal LM generate path rolls sliding cache entries when switching to the sliding-window decode JIT.
- `tinygrad_gemma/multimodal.py:813-839` — conditional generate path creates the sliding-window decode JIT but does not currently roll sliding cache entries before using it.
- `scripts/benchmark_gemma4_matrix.py:137-139` — benchmark invokes `model.generate(...)`.
- `tinygrad_gemma/loader.py:203` plus `checkpoints/gemma-4-E2B-int8/config.json:1-4` — E2B benchmark loads `GemmaForConditionalGeneration`, so the conditional generate path is the hard benchmark path.

E2B config facts:

- `checkpoints/gemma-4-E2B-int8/config.json:72-107` — 35-layer mixed sliding/full pattern.
- `checkpoints/gemma-4-E2B-int8/config.json:113-116` — `num_key_value_heads=1`, `num_kv_shared_layers=20`.
- `checkpoints/gemma-4-E2B-int8/config.json:130` — `sliding_window=512`.

## Original immediate probes, ranked (historical; superseded where noted)

### 1. Conditional sliding-cache roll parity

Hypothesis: the accepted hard benchmark uses `GemmaForConditionalGeneration.generate`, whose decode loop has a sliding-window JIT but lacks the causal LM path's one-time `roll_sliding_cache_entries(...)` transition. Porting the causal roll hook may reduce post-window symbolic slice/store work on the actual E2B benchmark path.

Status after the parallel sweep: rejected. The parity-only profile worsened to `34.896750` ms/token, and the merged parity+guard long rows regressed to `18.320320` and `18.352303` tok/s.

Implementation surface:

- Mirror `tinygrad_gemma/model.py:1015-1039` into `tinygrad_gemma/multimodal.py:821-839`.
- Set `cache.decode_sliding_window = True` exactly when the conditional path first switches to `sliding_rollout_jit`.
- Add conditional-generate coverage so the parity does not silently regress.

Why first:

- It is source-grounded and specific to the benchmark's loaded architecture.
- It does not require new tinygrad internals.
- It may let the existing rolled-cache optimization actually apply to the conditional path.

Risk:

- Medium. Must preserve multimodal/text-only prefill semantics and shared-KV behavior.

Acceptance:

- Focused conditional generate/cache tests pass.
- Full pytest, CLI help, and Metal smoke gate pass.
- Profile source items drop below `2994` or profile time beats `34.414750 ms/token` without graph-batch fragmentation.
- Historical acceptance threshold was the then-current `19.053995 tok/s` floor; future runtime patches now need to beat the refreshed `19.424036 tok/s` floor.

### 2. Packed slot-layout sweep

Hypothesis: after the large packed-cache win, storage geometry still matters. A layout that improves write/read locality may reduce packed cache-write time even if source count does not immediately fall.

Status after the parallel sweep: flat-slot `(B,H,L,2*D)` was rejected at `35.801833` ms/token with no source-count reduction. Lane-before-head-dim remains untested but should not outrank the layer-13 shared-source target.

Candidate variants:

1. Flat slot storage: `(B, H, L, 2 * D)`, with `key=packed[..., :D]`, `value=packed[..., D:]`, and single-position RHS `key.squeeze(2).cat(value.squeeze(2), dim=-1)`.
2. Lane-before-head-dim storage: `(B, H, L, 2, D)`, with `key=packed[:, :, :, 0, :]`, `value=packed[:, :, :, 1, :]`.
3. Current final-lane storage `(B, H, L, D, 2)` as control.

Implementation surface:

- `tinygrad_gemma/model.py:201-213`
- `tinygrad_gemma/model.py:280-302`
- `scripts/profile_decode_jit.py:347-353`
- cache shape/view tests in `tests/test_tinygrad_gemma.py`

Why second:

- Leading-lane to final-lane packing already improved the clean long floor from `18.537605` to `19.053995 tok/s` even with the same `2994` source count.
- A controlled slot-layout sweep is cheaper and safer than custom lowerings.

Risk:

- Low to medium. Reshape/cat variants can add UOps; accept only measured wins.

Acceptance:

- Source count `<2994`, or profile time `<34.414750 ms/token` plus repeated long-floor improvement.
- Graph batches `<=7`.
- No fallback and stable output hash.

### 3. Packed cache write layer/type attribution

Hypothesis: current profile splits only local versus shared-source writes. It does not identify which layer types or exact source layers dominate. Shared-source writes are fewer by count but nearly equal in time, so a concentrated layer/type target may exist.

Implementation surface:

- Profiler-only sidecar categories in `scripts/profile_decode_jit.py:327-353`.
- Include `layer_idx`, `layer_type`, role, and possibly physical layout/rolled status in category strings or a parallel summary map.

Why third:

- If attribution is concentrated, the next runtime patch can specialize precisely.
- If diffuse, it prevents a blind patch. Negative information is still information; one does not scold a thermometer for being boring.

Acceptance as instrumentation:

- Complete attribution over all `2994` source items.
- No unparsed graph batches or category laundering.
- A runtime follow-up is justified only if a role/layer cluster accounts for enough elapsed time to plausibly move the hard metric.

### 4. Active-cache no-op slice guard

Hypothesis: once entries are physically window-sized, `active_cache_tensors` may still emit no-op `:entry.window` slices that can add view/source work.

Implementation surface:

- `tinygrad_gemma/model.py:216-223`
- Existing no-op guard pattern in `sliding_decode_kv` at `tinygrad_gemma/model.py:122-129`

Risk:

- Low.

Acceptance:

- Profile source count or time improves; reject if it is invisible on profile and long-floor gates.

### 5. Current-token readback/deferred-store ordering probe

Hypothesis: current decode writes K/V into cache and then reads active K/V through cache views for the same token. A graph-equivalent ordering that attends over prior cache plus the just-computed one-token K/V may reduce read-after-write coupling.

Risk:

- Medium to high. It can easily duplicate producer work or perturb graph capture.

Acceptance:

- Must beat either source count or profile time before any long run.
- Same hash, no fallback, graph batches `<=7`.

## Tinygrad-internal research route

A separate read-only tinygrad inspection found that the installed tinygrad scheduler forces RHS realization for non-trivial `Tensor.assign` RHS tensors before lowering to cache stores. A microbench monkeypatch of the assign realization policy reduced a variable-index packed-cache write from `2` ExecItems to `1` ExecItem on CPU while preserving correctness. The local research checkout at `/Users/ericfode/src/.tinygrad_research` already uses a `STORE+AFTER`-style assignment design and scheduled the same microbench as `1` ExecItem.

Recommended research-only probe:

```bash
PYTHONPATH=/Users/ericfode/src/.tinygrad_research:. \
  .venv/bin/python scripts/profile_decode_jit.py ...
```

Goal: test whether the real E2B packed-cache profile drops below `2994` source items under the research tinygrad assignment semantics without model changes.

Caveat: this is a framework/runtime route, not a repo-local model patch. It should not supersede the accepted runtime checkpoint unless the full profile and repeated long benchmark gates pass on the same acceptance contract.

## Dead ends to avoid

- Producer-side prepacking / explicit RHS materialization: rejected by repeated long-floor evidence.
- Length-1 range assignment as a replacement for single-position assignment: kept `2994` source items and worsened profile time to `38.416875 ms/token`.
- JIT batch-size knob sweeping: short sweep was flat; `JIT=2` was much worse.
- Python custom raw Metal Runner wiring: fragmented graph batching badly in previous artifacts.
- MLP-only work as the main 100 tok/s route: current profile attributes only `4.62%` to MLP.

## What must change to approach 100 tok/s

The refreshed observed long floor is `19.424036 tok/s`, about `51.48 ms/token` over the measured decode window. A `100 tok/s` target is `10 ms/token`, requiring about a `5.15x` speedup.

The latest attribution profile assigns `32.736` ms/token of `34.319667` ms/token to packed cache-write realization scopes. Therefore:

- MLP-only wins cannot close the gap.
- Eliminating only local packed cache-write time leaves roughly `17.63 ms/token` in the profile.
- Eliminating only shared-source packed cache-write time leaves roughly `18.37 ms/token`.
- Reaching a `10 ms/token` profile while keeping MLP unchanged requires reducing cache-attributed time by roughly three quarters.

The plausible route is therefore:

1. Start with `shared-source-layer13-cache-write-specialization-037`, because layer 13 shared-source sliding attention is now the largest exact category.
2. Avoid retrying broad parity/layout/deferred-store hooks unless a narrower layer-13 hypothesis changes the evidence.
3. If model-side layer specialization saturates, test tinygrad assignment/store semantics from the research checkout or an equivalent graph-preserving scheduler branch.
4. Only then consider lower-level fused producer/store lowering, and only if it stays inside MetalGraph batching.

## Minimal next loop

Next implementation loop should be `shared-source-layer13-cache-write-specialization-037`.

Cheap gate before long benchmark:

```bash
.venv/bin/python -m pytest -q tests/test_tinygrad_gemma.py tests/test_profile_decode_jit.py
.venv/bin/python scripts/profile_decode_jit.py --help
```

Acceptance gate for any runtime patch:

```bash
.venv/bin/python -m pytest -q
.venv/bin/tinygrad-gemma --help
.venv/bin/python scripts/smoke_metal.py
.venv/bin/python scripts/benchmark_gemma4_matrix.py \
  --root checkpoints --sizes E2B --formats int8 --devices METAL --beams 0 \
  --max-new-tokens 1000 --decode-warmup-tokens 20 \
  --out benchmarks/<candidate>-1000.csv
```

Accept only with repeated clean long-floor improvement over `19.424036 tok/s`, stable output hash, `rollout_jit_count=999`, and `decode_fallback=false`.
