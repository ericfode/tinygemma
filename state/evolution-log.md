# Evolution Log

## 2026-04-27 062 - Paired helper frontier smoke accepted

- Status: accepted infrastructure.
- Real smoke 1: explicit-script exp_0005 self-pair hash16 wrote `benchmarks/paired-exp0005-self-hash16.json`; baseline `30.4576`, candidate `30.4196`, delta `-0.0380` on the same target.
- Real smoke 2: default-resolution exp_0005 self-pair hash4 wrote `benchmarks/paired-exp0005-self-default-hash4.json`; baseline `0.2349`, candidate `0.2359`, delta `+0.0010`. Hash4 is only a helper smoke, not a throughput metric.
- Fix: `scripts/paired_e2b_decode_benchmark.py` now resolves the default benchmark script from the baseline evo worktree when `benchmarks/evo_e2b_int8_metal_decode.py` is absent on main.
- Tests: added unit coverage for worktree-default benchmark resolution.
- Verification: full suite passed (`86 passed, 2 warnings`); CLI help, research METAL smoke, helper py_compile, and `git diff --check` passed.

## 2026-04-27 061 - Paired baseline/candidate benchmark helper accepted

- Status: accepted infrastructure; no runtime code changed.
- Change: added `scripts/paired_e2b_decode_benchmark.py`, which runs a benchmark script against `--baseline-target` and `--candidate-target` in one session, forwards extra args after `--`, parses JSON scores, and writes score deltas plus command/stdout/stderr provenance.
- Test: added `tests/test_paired_decode_benchmark.py` with a fake benchmark to verify argument forwarding, output file/stdout equivalence, delta/relative-delta computation, and provenance capture.
- Verification: full suite passed (`85 passed, 2 warnings`); CLI help, research METAL smoke, helper py_compile/help, and `git diff --check` passed.
- Decision: infrastructure only; no throughput claim.

## 2026-04-27 060 - Frontier saturation and RMSNorm cache review accepted

- Status: accepted negative review; no runtime code changed.
- Frontier: best remains `exp_0005` at `28.6153`; run has 13 experiments, 2 committed, 11 discarded.
- RMSNorm finding: inference scale caching was already tested as `exp_0009` and rejected (`26.1647` default score, long row around `17.1434`, failed floor). Corrected phase attribution does not reopen it: exact shared-source-layer13 `rmsnorm_rope` is only 4 sources / ~0.054 ms.
- Saturation finding: K/V projection variants, cache layout/write variants, attention/logit variants, RMSNorm cache, raw Metal runner, and `JIT_BATCH_SIZE=0` defaulting are all rejected/exhausted under current evidence.
- Decision: pause runtime children until benchmark/infrastructure improves; next target is paired same-session baseline-vs-candidate benchmarking and saturation ledger support.

## 2026-04-27 059 - Decode-only attention output view elision rejected

- Status: rejected evo experiment; no runtime code merged.
- Evo child: `exp_0012` from `exp_0005`, hypothesis `probe: decode-only attention output view elision`.
- Implementation tested only in the child worktree: skip final attention-output transpose/permute before reshape when `query_len == 1` in both grouped and ungrouped attention paths.
- Pre-evo verification: focused tests passed (`73 passed, 1 skipped, 2 warnings`), research METAL smoke passed (`rollout_jit_count=3`, `decode_fallback=False`), hash16 real-checkpoint gate passed at `30.415459 tok/s` with expected hash.
- Evo result: default score `28.4939 tok/s`, regressing from parent `exp_0005` at `28.6153`; inherited `e2b_int8_metal_hash1000_current_floor` failed.
- Action: discarded `exp_0012` with reason not to skip transpose/permute before decode attention-output reshape under this benchmark.
- Next target: `frontier-saturation-and-rmsnorm-cache-review-060`, a review of remaining non-rejected surfaces before spending another evo child.

## 2026-04-27 058 - Shared-source layer13 runtime hypothesis exhausted

- Status: accepted negative decision; no runtime code changed.
- Evidence: corrected exact-target comparison shows `shared-source-layer13` is the dominant cache-write phase bucket at 888 source items / ~14.503 ms, with `kv_projection` at 882 / ~14.422 ms.
- Rejection ledger: raw Metal runners, K/V projection reshape/split variants, QKV fusion, prepacked/repeated K/V, producer materialization, windowed shared-source allocation, store tweaks, and RHS-pack tweaks are already rejected or structurally too small.
- Decision: do not start a new shared-source-layer13 runtime evo child without a genuinely new graphable transformation; the hot bucket is real, but the known levers are exhausted.
- Next target: orthogonal `decode-only-attention-output-view-elision-evo-probe-059`, tested as an evo child against `exp_0005`, not committed directly to main.

## 2026-04-27 057 - Neighbor exact phase targets compared

- Status: accepted profiling decision; no runtime code changed.
- Method: sequential METAL graph profiles with exact `--phase-target` selectors: `local-layer11`, `local-layer12`, `shared-source-layer13`, `shared-source-layer14`, all with `--phase-cutpoints` at context length 700.
- Result: all four profiles preserved 7 MetalGraph batches and 0 raw gate/up runners.
- Exact target comparison: local-layer11 = 236 sources / ~3.222 ms; local-layer12 = 256 / ~3.142 ms; shared-source-layer13 = 888 / ~14.503 ms; shared-source-layer14 = 296 / ~5.011 ms.
- Hot exact target: `shared-source-layer13`, with `kv_projection` at 882 sources / ~14.422 ms (~29.8% elapsed).
- Decision: if another runtime experiment is attempted, it should target shared-source-layer13, not local-layer12; however obvious K/V projection reshape/fusion/materialization paths have already been rejected, so 058 must first find a non-repeated graphable transformation.

## 2026-04-27 056 - Phase overlap reporting accepted

- Status: accepted profiler instrumentation.
- Change: added `cache_write_phase_parent_category_counts` per source row and original capture, plus `source_attributed_cache_write_phase_summary.by_phase_parent_category`.
- Test: `test_profile_phase_summary_reports_parent_category_overlap` failed before overlap reporting existed, then passed.
- Artifact: `benchmarks/gemma4-metal-decode-graph-local-layer12-phase-overlap-700.json` shows 7 MetalGraph batches, 0 raw gate/up runners, phase source count 256, and exact local-layer12 parent overlap for every local-layer12 phase bucket.
- Corrected local-layer12 bucket: `kv_projection` is 250 sources / ~2.954 ms (~5.96% elapsed); total local-layer12 phase bucket is 256 sources.
- Verification: full suite passed (`84 passed, 2 warnings`); METAL smoke passed with `rollout_jit_count=3`, `decode_fallback=False`.
- Decision: attribution is now reliable, but the exclusive surface is small and needs comparison against neighboring exact targets before another runtime experiment.
- Next target: compare exact neighboring phase targets (`local-layer11`, `local-layer12`, `shared-source-layer13`, etc.) and select only if a non-rejected transformation appears.

## 2026-04-27 055 - Local layer12 phase attribution scope fix accepted

- Status: accepted profiler fix; no runtime probe launched.
- Decision: the previous local-layer12 phase artifact was not safe enough to justify a runtime experiment because source attribution/summarization ran after the phase-target scope reset.
- Change: added `attribute_profile_execution_sources(execution_items, rows, phase_targets)` and routed source attribution plus original-capture phase summary through the explicit requested target scope.
- Test: `test_profile_source_attribution_uses_explicit_phase_targets_after_scope_reset` failed before the helper existed, then passed.
- Corrected artifact: `benchmarks/gemma4-metal-decode-graph-local-layer12-phase-profile-700-scope-fixed.json` preserved 7 MetalGraph batches and 0 raw gate/up runners.
- Corrected signal: local-layer12 parent is 256 source items / ~3.255 ms (~6.45% elapsed); `kv_projection` is 250 source items / ~3.178 ms (~6.30%), not the previous 885-source / ~14 ms transitive closure.
- Verification: profiler test module passed (`34 passed, 2 warnings`); full suite passed (`83 passed, 2 warnings`); METAL smoke passed with `rollout_jit_count=3`, `decode_fallback=False`.
- Next target: add explicit overlap reporting / compare exact neighboring targets before considering another runtime model.py experiment.

## 2026-04-27 054 - Profile phase target generalization accepted

- Status: accepted.
- Change: generalized `scripts/profile_decode_jit.py` cache-write phase target selection from a hardcoded layer-13 shared-source selector to scoped presets and exact role/layer selectors.
- New CLI: `--phase-cutpoints` plus repeatable `--phase-target TARGET`; supported targets include `layer13`, `all-shared-source`, `all-local`, `all-packed`, `local-layer<N>`, `shared-source-layer<N>`, and `shared-consumer-layer<N>`. `--layer13-phase-cutpoints` remains a compatibility alias.
- TDD: new parser/scope tests failed before implementation, then passed.
- Verification: `tests/test_profile_decode_jit.py` passed (`32 passed, 2 warnings`); full suite passed (`82 passed, 2 warnings`); METAL smoke passed with `rollout_jit_count=3` and `decode_fallback=False`.
- Real artifacts: generated all-local and local-layer12 phase-cutpoint profiles at `benchmarks/gemma4-metal-decode-graph-all-local-phase-profile-700.json` and `benchmarks/gemma4-metal-decode-graph-local-layer12-phase-profile-700.json` (left untracked). Both preserved 7 `MetalGraph` batches.
- Observation: broad `all-local` targeting is useful for scanning but can smear phase metadata across graph batches; exact selectors are safer for runtime probes.
- Next target: use exact `local-layer12` phase attribution to choose a narrow graphable runtime experiment.

## 2026-04-27 053 - Graphable rowwise-int8 QKV fusion probe rejected

- Status: rejected.
- Evo child: `exp_0011` from `exp_0005`, hypothesis `probe: graphable rowwise-int8 qkv decode fusion`.
- Implementation tested inside the evo worktree only: decode-only Tensor-level rowwise-int8 Q/K/V projection fusion plus focused parity coverage.
- Verification before evo: focused QKV parity passed; `tests/test_tinygrad_gemma.py tests/test_profile_decode_jit.py` passed (`74 passed, 1 skipped`); research METAL smoke passed with `rollout_jit_count=3` and `decode_fallback=False`; hash16 E2B int8 METAL gate passed at `29.464043 tok/s` with expected hash.
- `evo run exp_0011` result: default score `27.7816 tok/s` versus parent `exp_0005` at `28.6153`; inherited `e2b_int8_metal_hash1000_current_floor` failed.
- Action: discarded `exp_0011` with reason: graphable rowwise-int8 QKV decode fusion regressed default score and failed the long floor.
- Decision: do not retry Q/K/V rowwise-int8 projection fusion under the current benchmark without new profile evidence.
- Next target: profiler-first phase target generalization for local/shared-source cache-write attribution before another runtime patch.

## 2026-04-27 - Metal Int8 First-Class Raw Bridge Graphability Rejected

- Objective: execute `metal-int8-first-class-raw-bridge-graphability-052` after 051 proved that a process-local research tinygrad bridge can replay the dormant raw rowwise-int8 METAL runner only when UOps are preserved through `resolve_params(ctx, call)`. This increment decides whether that bridge can honestly become a production, graphable decode optimization.
- Graphability finding: the research tinygrad `GraphRunner.supports_exec_item` accepts only `Ops.SINK` or `Ops.PROGRAM` single-device calls, and `MetalGraph.supports_exec_item` delegates to it. `MetalGraph.__init__` then casts each scheduled item to `CompiledRunner` and reads `prg._prg.pipeline_state` plus `prg.p.launch_dims(...)`. A project-local `RowwiseInt8DecodeLinearRunner` is a host-call runner around raw Metal library bytes, not a first-class compiled tinygrad program.
- Decision: reject production integration of the raw custom Runner. The 051 `pm_exec` / `_call_outs_ins` bridge remains a useful prototype and compatibility canary, but it depends on process-local monkeypatching and would not preserve MetalGraph batching without a real tinygrad runtime extension. The raw-runner branch is therefore pruned rather than promoted.
- Cleanup: removed the dead `GemmaMLP._can_use_metal_fused_int8_gate_up(...)` method and its test; removed the `scripts/profile_decode_jit.py --metal-int8-gate-up` selector so the profiler no longer advertises a raw path; kept new profile payloads schema-compatible by recording `metal_int8_gate_up="default"`; marked `tinygrad_gemma/metal_int8.py` as prototype-only in the module docstring.
- Plan artifact: `docs/plans/2026-04-27-metal-int8-first-class-raw-bridge-graphability.md` records the rejection evidence, invalidation criteria for revisiting, and the next graphable target.
- Verification: focused tests passed with `74 passed, 2 warnings`; `.venv/bin/python -m py_compile tinygrad_gemma/metal_int8.py scripts/profile_decode_jit.py` passed; profile help no longer contains `--metal-int8-gate-up`; full `.venv/bin/python -m pytest -q` passed with `79 passed, 2 warnings`; `.venv/bin/tinygrad-gemma --help >/dev/null` passed; `.venv/bin/python scripts/smoke_metal.py` reported `default_device=METAL`, `rollout_jit_count=3`, and `decode_fallback=False`; `git diff --check` and JSON validation passed.
- Throughput status: no real-checkpoint throughput floor is superseded. The evo frontier remains `exp_0005` at `28.6153` tok/s. The next target is `evo-frontier-profiler-backed-graphable-surface-053`: refresh frontier/profile evidence and choose the next narrow graphable Tensor/tinygrad-program surface.

## 2026-04-27 - Metal Int8 Research Capture Bridge Spike Accepted

- Objective: execute `metal-int8-research-capture-bridge-spike-051` after 050 proved that the dormant raw rowwise-int8 METAL runner must fail closed under add-linear-only research tinygrad capture. This increment is a prototype-only feasibility result; it does not enable the bridge in `GemmaMLP`, `RowwiseInt8Linear`, or the active E2B benchmark path.
- Research-runtime finding: `/Users/ericfode/src/.tinygrad_research` can replay a raw runner if it is represented as a real `Ops.LINEAR` containing an `Ops.CALL` over `Ops.CUSTOM_FUNCTION(arg="raw_rowwise_int8_decode_linear")`, and if a process-local `tinygrad.engine.realize.pm_exec` handler resolves operands through `resolve_params(ctx, call)` at replay time. Unknown custom calls are otherwise silently ignored by `pm_exec`, so a handler is mandatory.
- Crucial safety finding: buffer-only capture is unsound. A bridge that turns tensors into concrete buffers during capture loses `PARAM` identity and can read capture-time input buffers during replay. The accepted prototype preserves `x`, `qweight`, and `scale` UOps in the custom call until research TinyJit substitutes current replay inputs.
- Artifact: `scripts/prototype_research_raw_metal_capture_bridge.py` is an explicit spike script. It first reproduces the production helper's add-linear-only rejection, then temporarily patches `realize.pm_exec` and `jit._call_outs_ins`, builds the raw op as a UOp-preserving custom-function linear, verifies three distinct METAL replay inputs against a NumPy reference, proves a later tinygrad op can consume the raw output, and restores patched globals before exit.
- Plan artifact: `docs/plans/2026-04-27-metal-int8-research-capture-bridge-spike.md` records the UOp shape, non-goals, acceptance criteria, and decision rule.
- Verification: `.venv/bin/python -m py_compile scripts/prototype_research_raw_metal_capture_bridge.py` passed; `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/Users/ericfode/src/.tinygrad_research:. .venv/bin/python scripts/prototype_research_raw_metal_capture_bridge.py` passed and printed `ok research raw metal capture bridge prototype`; `.venv/bin/python -m pytest -q` passed with `82 passed, 2 warnings`; `.venv/bin/tinygrad-gemma --help >/dev/null && .venv/bin/python scripts/smoke_metal.py` passed with `default_device=METAL`, `rollout_jit_count=3`, and `decode_fallback=False`.
- Decision: accept the spike as evidence that a research TinyJit bridge is possible, but keep production code fail-closed. Project-local global monkeypatching is acceptable for this inspectable script, not as model hot-path integration. The next credible increment is to decide whether a first-class, graphable-enough bridge can avoid the known MetalGraph fragmentation regression; otherwise prune the raw-runner branch and return to profiler-backed graphable surfaces.

## 2026-04-27 - Metal Int8 Raw Runner Capture Safety Accepted

- Objective: execute `metal-int8-raw-runner-capture-safety-050` after 049 made the dormant raw rowwise-int8 METAL runner constructible under the research tinygrad runtime. This is a safety/compatibility increment only; it does not wire the raw runner into decode and does not supersede the `exp_0005` frontier score `28.6153`.
- Capture-surface finding: stock tinygrad TinyJit capture records `ExecItem`s through `capturing[0].add(item)`, while `/Users/ericfode/src/.tinygrad_research` records linear UOps through `capturing[0].add_linear(linear, var_vals)` and has no generic `add(ExecItem)` hook. A raw runner hidden inside research TinyJit capture therefore cannot be replayed honestly without a dedicated bridge.
- Change: `_run_or_capture_rowwise_int8_decode_linear()` now checks that an active capture object supports `add(ExecItem)` before registering the raw runner. Stock-style capture still records the `ExecItem` and runs the capture pass; add-linear-only capture now raises a clear `RuntimeError` before executing the kernel, avoiding a silent capture/replay bypass.
- Focused coverage: `test_raw_metal_rowwise_int8_stock_capture_adds_exec_item_and_runs` proves stock-style capture registration plus immediate capture-pass execution with a fake runner; `test_raw_metal_rowwise_int8_rejects_capture_without_exec_item_add` first failed with `AttributeError: 'LinearOnlyCapture' object has no attribute 'add'`, then passed after the explicit guard. The existing 049 MTLB construction test also passed.
- Real canaries: a stock tinygrad `TinyJit` canary over three different METAL inputs printed `ok stock tinygrad raw metal TinyJit replay canary [[-5.0, 0.0], [-3.25, 0.0], [3.0, -4.5]]`, proving standalone raw-runner replay changes with inputs. A research tinygrad canary printed `ok research tinygrad raw metal capture rejected explicitly ... add(ExecItem)`, proving the active benchmark runtime fails closed rather than pretending to replay.
- Verification: focused raw-metal tests passed (`5 passed, 40 deselected`); `.venv/bin/python -m pytest -q tests/test_tinygrad_gemma.py tests/test_profile_decode_jit.py` passed (`77 passed`); `.venv/bin/python -m pytest -q` passed (`82 passed`); `.venv/bin/tinygrad-gemma --help >/dev/null` passed; `.venv/bin/python scripts/smoke_metal.py` reported `default_device=METAL`, `rollout_jit_count=3`, and `decode_fallback=False`; `git diff --check` and JSON validation passed.
- Plan artifact: `docs/plans/2026-04-27-metal-int8-capture-safety.md` records the scope, non-goals, and acceptance criteria.
- Decision: accept capture safety but do not integrate the raw runner into `RowwiseInt8Linear`, `GemmaMLP`, or the active E2B decode path yet. The next credible increment is `metal-int8-research-capture-bridge-spike-051`: inspect whether a dedicated research tinygrad linear/custom-function bridge can represent the raw runner in capture/replay, or reject that path and return to graphable `model.py`/profiler-backed surfaces.

## 2026-04-27 - Metal Int8 MTLB Compatibility Accepted

- Objective: execute `metal-int8-research-mtlb-compatibility-049` as a separate compatibility increment after the evo frontier evidence refresh found no remaining evidence-backed `model.py` micro-surface. The active evo frontier remains `exp_0005` at default score `28.6153`; this increment does not claim a decode throughput improvement.
- Root cause addressed: `tinygrad_gemma/metal_int8.py` passed raw Metal source bytes directly to `Device["METAL"].runtime(...)`. Stock tinygrad accepts source bytes in its Metal program path, but the research tinygrad runtime used by the E2B harness expects compiled library bytes beginning with `MTLB` and ending with `ENDT`, causing the 047 canary failure `RuntimeError: Invalid library file`.
- Change: `_rowwise_int8_decode_linear_program()` now resolves the METAL device once, compiles `ROWWISE_INT8_DECODE_LINEAR_SOURCE` through `metal_device.compiler.compile_cached(...)`, and passes the compiled library bytes to `metal_device.runtime(...)`. The active graphable `RowwiseInt8Linear` path and disabled raw Metal gate/up integration remain unchanged.
- Focused TDD receipt: `tests/test_tinygrad_gemma.py::test_metal_rowwise_int8_program_compiles_source_before_runtime` first failed with `KeyError: 'compiled_source'`, proving the compiler was not called; after the patch it passed.
- Research-runtime canary: `PYTHONPATH=/Users/ericfode/src/.tinygrad_research:. PYTHONDONTWRITEBYTECODE=1 .venv/bin/python - <<'PY' ...` passed and printed `ok metal_int8 research MTLB canary`, including an assertion that `compile_cached(...)` returned `MTLB...ENDT` bytes and a tiny raw rowwise-int8 decode-linear call produced shape `(1, 1, 32)` with value `16.0`.
- Verification: `.venv/bin/python -m pytest -q tests/test_tinygrad_gemma.py -k 'metal_rowwise_int8 or RowwiseInt8'` passed (`2 passed, 41 deselected`); `.venv/bin/python -m pytest -q tests/test_tinygrad_gemma.py tests/test_profile_decode_jit.py` passed (`75 passed`); `.venv/bin/python -m pytest -q` passed (`80 passed`); `.venv/bin/tinygrad-gemma --help >/dev/null` passed; `PYTHONPATH=/Users/ericfode/src/.tinygrad_research:. .venv/bin/python scripts/smoke_metal.py` reported `default_device=METAL`, `loaded_model_device=METAL`, `logits_device=METAL`, `generated_tokens=4`, `rollout_jit_count=3`, and `decode_fallback=False`; `git diff --check` passed.
- Plan artifact: `docs/plans/2026-04-27-metal-int8-mtlb-compatibility.md` records the scope, non-goals, and acceptance criteria.
- Decision: accept the compatibility patch only. Do not wire `metal_rowwise_int8_decode_linear` into `RowwiseInt8Linear` or `GemmaMLP` yet, and do not tune `local_size` or report E2B throughput from this change. The next credible increment is `metal-int8-raw-runner-capture-safety-050`: prove raw-runner replay/capture behavior under stock and research tinygrad before any model hot-path integration.

## 2026-04-27 - Evo Active Shared-KV View Reuse Rejected

- Objective: execute `evo-frontier-graphable-model-probe-048` from frontier `exp_0005` (`28.6153` default `128/20` score) while avoiding already-rejected K/V rank/view, RMSNorm cache, scaling-guard, and raw-runner surfaces.
- `exp_0010` tested a narrow `model.py` change: for full-length shared-KV producer layers, store the active `k`/`v` tensors in `shared_kv_states` rather than handing shared consumers the bounded cache entry that was sliced again.
- Focused behavior coverage: `tests/test_tinygrad_gemma.py::test_shared_kv_consumers_reuse_active_cache_views` first failed on repeated shared-consumer reslicing, then passed after the implementation. The focused safety set also passed `test_preallocated_cache_matches_full_forward_for_gemma4` and `test_rolled_shared_sliding_source_matches_full_forward_after_window`.
- Verification before evo: `/Users/ericfode/Downloads/tinygrad-gemma/.venv/bin/python -m pytest -q tests/test_tinygrad_gemma.py tests/test_profile_decode_jit.py` passed inside the `exp_0010` worktree.
- Rejection evidence: `evo run exp_0010` measured default score `28.2824` (`measured_decode_tokens_per_second=28.282413`, `generated_tokens=128`, `rollout_jit_count=127`, `decode_fallback=false`) versus parent `exp_0005` at `28.6153`. All inherited gates passed, including `_init_gate`, `metal_smoke`, `cli_help`, `e2b_int8_metal_hash16`, and `e2b_int8_metal_hash1000_current_floor` (`18.003002 tok/s`), but the calibrated default frontier regressed. `evo discard exp_0010` recorded the rejection.
- Decision: do not pursue active shared-KV view reuse as a performance optimization under the current benchmark. It is semantically tidy and locally graphable, but the measured hot path is slower; the compiler has, with admirable terseness, declined the paperwork.
- Next target: `evo-frontier-evidence-refresh-049`. Before another code child, refresh `evo scratchpad`/frontier evidence and avoid the now-rejected graphable micro-edit family: scaling guard, one-token K/V rank flattening, attention-output view elision, RMSNorm scale cache, and active shared-KV reuse. Either find a genuinely new `model.py` surface backed by profile evidence or pivot to the separate `metal_int8.py` research-tinygrad MTLB compatibility target.

## 2026-04-26 - Metal Int8 Local-Size Survey Rejected

- Objective: execute `metal-int8-kernel-local-size-survey-047` before touching the runtime path. The candidate surface was `tinygrad_gemma/metal_int8.py`, especially the dormant raw rowwise-int8 METAL decode-linear runner and its `local_size` parameter.
- Baseline receipts: `.venv/bin/python -m pytest -q tests/test_tinygrad_gemma.py -k 'metal_rowwise_int8 or RowwiseInt8'` passed (`1 passed, 41 deselected`). Stock-tinygrad prototype sweeps for shape `1536 -> 12288` passed correctness and measured threadgroup-x medians of `0.300875 ms` (`local_size=64`), `0.124333 ms` (`128`), and `0.104708 ms` (`256`) in this local run.
- Compatibility blocker: the actual E2B evo harness prefers `/Users/ericfode/src/.tinygrad_research`; under that path, a direct `metal_rowwise_int8_decode_linear` canary failed with `RuntimeError: Invalid library file`. Root cause: `metal_int8.py` passes Metal source bytes to `Device.runtime`, while the research `MetalProgram` path expects compiled `MTLB` bytes.
- Real E2B sanity gate on unchanged source: `PYTHONPATH=/Users/ericfode/src/.tinygrad_research:. .venv/bin/python scripts/benchmark_gemma4_matrix.py --root checkpoints --sizes E2B --formats int8 --devices METAL --beams 0 --prompt hello --max-new-tokens 16 --decode-warmup-tokens 4 --progress-every 0 --out /tmp/metal-int8-047-e2b-hash16.csv` passed with `measured_decode_tokens_per_second=30.327774`, output hash `12c16f74b486dc759fa8064b549cac92c2aef68ced13b8202b4e55e9171a8c62`, `rollout_jit_count=15`, and `decode_fallback=false`.
- Decision: adopt no code change. A `local_size` tweak would affect only a dormant prototype path, and integrating the raw runner into `RowwiseInt8Linear` would contradict the current raw-runner-disabled/graph-breaking policy while also failing the actual research tinygrad benchmark path. Artifact: `benchmarks/metal-int8-kernel-local-size-survey-047.json`.
- Next target: `evo-frontier-graphable-model-probe-048`. Return to `exp_0005` and graphable `model.py`-local probes tied to layer-13 K/V projection source-mass evidence, unless a separate compatibility increment first makes `metal_int8.py` source compilation research-tinygrad-compatible. A tidy raw kernel is still raw; tinygrad is not obliged to admire it.

## 2026-04-26 - Evo RMSNorm Scale Cache Rejected

- Objective: execute `rmsnorm-inference-scale-cache-046` from the current evo frontier `exp_0005` (`28.6153` default `128/20` score, current long `1000/20` hash floor at `--min-score 18.0`).
- `exp_0009` tested caching immutable inference RMSNorm scale tensors for both ordinary `weight.float()` and Gemma plus-one `(1 + weight.float())` paths. The implementation avoided putting cached tensors on module instances after an initial state-dict/optimizer-load failure showed that instance Tensor caches are visible to tinygrad state traversal; the final probe used a module-level cache keyed by the `Tensor` weight object.
- Verification before evo: new focused RMSNorm tests first failed on the missing cache, then passed; `.venv/bin/python -m pytest -q tests/test_tinygrad_gemma.py tests/test_profile_decode_jit.py` passed with `75 passed, 1 skipped`; CLI help passed; `scripts/smoke_metal.py` reported `default_device=METAL`, `rollout_jit_count=3`, and `decode_fallback=False`.
- Rejection evidence: `evo run exp_0009` measured default score `26.1647` (`measured_decode_tokens_per_second=26.164706`, stable short hash, `rollout_jit_count=127`, `decode_fallback=false`), below parent `28.6153`, and failed the inherited `e2b_int8_metal_hash1000_current_floor` gate. `evo discard exp_0009` recorded the rejection.
- Decision: do not cache realized RMSNorm scale tensors in `model.py` under the current decode benchmark. The transformation is behaviorally tidy but performance-hostile; tinygrad has declined the offering with its usual tact.
- Next target: `metal-int8-kernel-local-size-survey-047`. Shift away from model.py-local view/cache tweaks and inspect `tinygrad_gemma/metal_int8.py` kernel/local-size or shape-specialization surfaces, likely requiring a new evo target/run or a narrow manual probe with real E2B int8 METAL gates.

## 2026-04-26 - Evo Layer-13 Rank/View Probes Rejected

- Objective: finish the two follow-up `model.py` evo probes against the calibrated current frontier. Parent remains `exp_0005`, the no-code calibration point at default `128/20` score `28.6153`, with inherited current long gate `e2b_int8_metal_hash1000_current_floor` at `--min-score 18.0` and stable hash `aa4944455473e236807f6c711a77a969caf73680ea5ea9725648294e5fd23cfa`.
- `exp_0007` tested one-token fused-int8 K/V decode-rank flattening inside `GemmaAttention._project_kv`: flatten `(batch, 1, hidden)` to 2D for the fused rowwise-int8 matmul, then reshape back before the existing split path. Focused behavior coverage and current hash/long gates passed, but `evo run exp_0007` measured default score `28.4375`, below parent `28.6153`, so the experiment was discarded.
- `exp_0008` tested decode-only attention-output view elision for `query_len == 1`: helper functions flatten/restore the one-token attention output so the output projection can consume the merged head dimension without the legacy transpose/reshape path. Focused helper coverage, cache/full-forward smoke, `tests/test_profile_decode_jit.py`, `_init_gate`, `metal_smoke`, `cli_help`, and the short/long hash gates all passed.
- Rejection evidence for `exp_0008`: `evo run exp_0008` measured default score `28.4258`, below parent `28.6153`; the inherited long `1000/20` gate measured `18.078904 tok/s`, also below the `exp_0005` long calibration row `18.090845`. Correct but slower remains slower; the compiler is an exacting critic, as ever.
- Decision: discard both probes and keep `exp_0005` as the honest evo frontier. The layer-13 K/V rank/view surface has now rejected dequantized/transposed K/V caching, separate K/V, fused-output lane reshape, unit-scaling guard, decode-rank flattening, and one-token attention-output view elision under the current benchmark/gate regime. No runtime throughput floor was superseded.
- Verification receipts: `evo status` reports `experiments=9`, `committed=2`, `discarded=7`, `failed=0`, active `0`, and best `exp_0005` score `28.6153`. `evo get exp_0007` and `evo get exp_0008` both report `gate_result=true` and status `discarded` with the reasons above.
- Next target: `rmsnorm-inference-scale-cache-046`. Try the broader but still `model.py`-local RMSNorm scale-cache probe only if continuing this evo run: cache immutable inference scale tensors in `RMSNorm.__call__` so repeated `(1 + weight.float())` / `weight.float()` construction is avoided, with focused tests for plus-one and ordinary scale parity plus training/mutable-weight safety. Acceptance remains a default score above `28.6153` and the current long hash gate passing with `rollout_jit_count=999` and `decode_fallback=false`.

## 2026-04-26 - Evo Layer-13 K/V Probe Sweep / Unit-Scaling Rejected

- Objective: verify the `exp_0006` model.py probe under the evo harness after recalibrating the benchmark floor. The current evo frontier is `exp_0005`, a no-code calibration child of `exp_0000`, with default `128/20` score `28.6153`; the inherited long gate is now `e2b_int8_metal_hash1000_current_floor` at `--min-score 18.0` plus stable long output hash `aa4944455473e236807f6c711a77a969caf73680ea5ea9725648294e5fd23cfa`.
- Calibration correction: unchanged `exp_0000` measured `18.181815 tok/s` on the hard `1000/20` row and `28.705971 tok/s` on the default `128/20` row in the current machine state, so the older absolute `19.424036` hard floor was stale. `exp_0005` committed as the current no-code comparison point and passed the recalibrated long gate with `18.090845 tok/s`.
- Rejected model.py probes: `exp_0003` separate K/V on sliding shared-source producers improved the stale short score to `28.6264` but measured only `18.177621 tok/s` on the long row; `exp_0004` fused K/V lane-reshape split measured `28.4425` short and `18.147725 tok/s` long; `exp_0006` skipped the attention scaling multiply when `self.scaling == 1.0`, passed behavior gates and the current long floor (`18.133788 tok/s`), but regressed the calibrated default score to `28.4567` versus parent `28.6153`.
- Decision: reject `exp_0006` and keep `exp_0005` as the honest current frontier. Do not treat sub-`28.6153` short rows as improvements, and treat the older `19.424036` long floor as environment-stale unless a fresh paired baseline returns to it.
- Next candidate shortlist from the read-only council: first try a one-token fused-int8 K/V decode-rank flattening probe if continuing the layer-13 K/V line; otherwise the safer orthogonal model.py probe is decode-only attention-output view elision for `query_len == 1`. RMSNorm inference scale caching remains a broader non-K/V candidate, but should earn a short/hash gate before spending the long row.
- Verification receipts: `evo get exp_0006` reports status `discarded`, score `28.4567`, `gate_result=true`, and discard reason `passed behavior and current long floor, but default score regressed to 28.4567 below calibrated parent exp_0005 score 28.6153; reject unit-scaling guard as no throughput win`. `evo status` reports `experiments=7`, `committed=2`, `discarded=5`, `failed=0`, and best `exp_0005` score `28.6153`.

## 2026-04-26 - Layer 13 reduce_r Source Attribution / Phase Propagation

- Objective: finish `layer13-reduce-source-attribution-043` by mapping the layer-13 `reduce_r`-heavy parent cache-write source rows to child producer operations before spending another runtime patch. The hard metric remains repeated real E2B int8 `METAL`, `beam=0`, `1000/20`, stable hash, `rollout_jit_count=999`, and `decode_fallback=false`.
- Profiler overhead fix: `scripts/profile_decode_jit.py` now caches cache-write category parsing and avoids recursive source-edge phase scans in the UOp creation/replace and `add_linear` sidecar paths. Phase propagation is kept direct: source UOps carry phase metadata forward during construction/replacement, and summary code reads direct toposort metadata rather than recursively rescanning every child edge. Focused tests in `tests/test_profile_decode_jit.py` pin that direct-toposort contract.
- Real METAL phase-propagation artifact: `PYTHONPATH=/Users/ericfode/src/.tinygrad_research:. .venv/bin/python scripts/profile_decode_jit.py --model-dir checkpoints/gemma-4-E2B-int8 --device METAL --context-length 512 --jit-mode 1 --out benchmarks/gemma4-metal-decode-graph-layer13-phase-propagation-profile-512.json --csv-out benchmarks/gemma4-metal-decode-graph-layer13-phase-propagation-profile-512.csv` completed with `source_attribution.status=complete`, `original_exec_count=2994`, `unattributed_tail_count=0`, `kernel_count=7`, and `elapsed_ms=48.08624996803701` on the profiling run.
- Child phase evidence: the refreshed artifact recovered child phase metadata for `872/885` layer-13 shared-source rows: `kv_projection=863`, `store=9`, with `13` unclassified rows and `37` phase-conflict rows. The source-attributed phase summary assigns `13.082764 ms` / `863` rows to K/V projection and only `0.139467 ms` / `9` rows to store. RHS packing and RMSNorm/RoPE did not appear as child-phase buckets in this profile.
- Analysis artifact: `scripts/analyze_layer13_reduce_sources.py` now defaults to the phase-propagation profile and writes `benchmarks/gemma4-metal-layer13-reduce-source-attribution-043.json` plus `docs/plans/2026-04-26-layer13-reduce-source-attribution.md`. The structural CSV still maps `680` layer-13 target rows to `r_*`/`reduce_r` display groups; the new phase profile says that mass is overwhelmingly K/V projection, not a store-tail or RHS-pack problem. A pleasing result, in the way a locked door is pleasing: at least now we know which lock it is.
- Decision: accept this as profiler/analysis infrastructure only. No runtime patch was landed and no throughput floor is superseded. The next runtime candidate must reduce layer-13 shared-source K/V projection source mass or prove an equivalent fused path before spending the hard `1000/20` throughput gate.
- Verification: `.venv/bin/python -m pytest -q` passed with `79 passed, 2 warnings in 46.98s`; `.venv/bin/tinygrad-gemma --help` exited 0; `PYTHONPATH=/Users/ericfode/src/.tinygrad_research:. .venv/bin/python scripts/smoke_metal.py` reported `default_device=METAL`, `loaded_model_device=METAL`, `logits_device=METAL`, `generated_tokens=4`, `rollout_jit_count=3`, and `decode_fallback=False`; `.venv/bin/python -m json.tool configs/repo-loop-state.json` and `git diff --check` passed.
- Next target: `layer13-kv-projection-source-reduction-044`. Inspect `GemmaAttention._project_kv` and the rowwise-int8 fused K/V path for a layer-13/shared-source-specific source-count or reduce-mass reduction. Do not pursue RHS packing or store-only tweaks unless new evidence contradicts the phase-propagation profile.

## 2026-04-26 - Layer 13 Phase Attribution / Greedy Softcap Elision Rejected

- Objective: finish `layer13-shared-source-producer-store-subattribution-042` and only keep a runtime patch if it moved the hard token-throughput metric. The active hard gate remains real E2B int8 `METAL`, `beam=0`, `1000/20`, stable hash, `rollout_jit_count=999`, and `decode_fallback=false`.
- Profiler work: `scripts/profile_decode_jit.py` now carries profiler-only layer-13 phase/sub-attribution plumbing and degrades cleanly on stock tinygrad builds where `TinyJit.add_linear` is absent. Focused regression coverage in `tests/test_profile_decode_jit.py` verifies the no-hook compatibility path and preservation of an existing hook.
- Real attribution evidence: `benchmarks/gemma4-metal-decode-layer13-structural-profile-512.json` kept the accepted packed-cache structure (`2994` compiled runners/source items, `7` graph batches in the corresponding source-coherent path) and again identified `attention_packed_cache_write__role_shared_source__layer_13__type_sliding_attention` as the dominant target. The structural analysis found that `reduce_r` kernels dominate the layer-13 target mass, but the child phase-attribution artifact `benchmarks/gemma4-metal-decode-graph-layer13-phase-profile-512.json` still recorded `cache_write_phase_unclassified_count=885` and an empty `source_attributed_cache_write_phase_summary.by_phase`.
- Diagnostic failure mode: a forced phase-cutpoint profile path reached the real METAL workload but was too expensive to use as a default profiler route; a tracked attempt was killed after producing no usable artifact. Future phase attribution should preserve metadata through captured/lowered source items rather than perturbing realization boundaries.
- Runtime probe: a temporary greedy softcap-elision patch bypassed final-logit softcapping for greedy argmax by sampling from raw logits. The semantic argument was monotonicity, but the hard metric rejected it. `benchmarks/gemma4-metal-e2b-int8-128-greedy-softcap-elision-current.csv` measured `28.378910` warmup-excluded tok/s, below the recent short-control `30.808261` row. Repeated long gates measured `17.572001` and `17.121386` tok/s in `benchmarks/gemma4-metal-e2b-int8-1000-greedy-softcap-elision-current.csv` and `benchmarks/gemma4-metal-e2b-int8-1000-greedy-softcap-elision-repeat.csv`, both with `generated_tokens=1000`, `measured_decode_tokens=980`, `rollout_jit_count=999`, `decode_fallback=false`, and stable output hash `aa4944455473e236807f6c711a77a969caf73680ea5ea9725648294e5fd23cfa`.
- Decision: reject greedy softcap elision and keep no runtime changes from that probe. No accepted throughput floor is superseded; the current campaign still targets the `100 tok/s` goal but must move through a source-attributed graph-mass reduction, not a logits-only shortcut.
- Verification after reverting the runtime probe: `.venv/bin/python -m pytest -q` passed with `78 passed, 2 warnings in 45.34s`; `.venv/bin/tinygrad-gemma --help` exited 0; `.venv/bin/python scripts/smoke_metal.py` reported `default_device=METAL`, `loaded_model_device=METAL`, `logits_device=METAL`, `generated_tokens=4`, `rollout_jit_count=3`, and `decode_fallback=False`; `.venv/bin/python -m json.tool configs/repo-loop-state.json` and `.venv/bin/python -m py_compile scripts/profile_decode_jit.py` passed; `git diff --check` passed.
- Next target: `layer13-reduce-source-attribution-043`. Inspect the layer-13 reduce-heavy source rows and map them back to exact producer operations before writing another runtime patch. Acceptance remains repeated `1000/20` improvement over the accepted floor or complete-attribution source-count reduction below `2994`; pretty graph rearrangements without tok/s evidence remain merely decorative.

## 2026-04-26 - Source-Coherent Packed Cache Restore / JIT_BATCH_SIZE=0 Rejected

- Objective: finish `source-coherent-packed-cache-restore-and-jitbatch-probe-040` by restoring the live runtime/source surface to the final-lane packed-cache state required by accepted artifacts, then testing whether `JIT_BATCH_SIZE=0` improves the hard `tok/s` metric.
- Source coherence restored: `tinygrad_gemma/model.py` again exposes the packed-cache runtime APIs expected by tests and artifacts, including `GemmaCacheEntry.packed`, `make_packed_cache_entry`, packed cache update support in `realize_cache_update`, rolled sliding-cache helpers, and graphable fused int8 K/V projection. `scripts/profile_decode_jit.py` now imports `make_packed_cache_entry` directly instead of carrying a split-cache fallback that could hide future source/artifact drift.
- Coherence gates: the previously red packed-cache focused tests passed (`7 passed`), `tests/test_tinygrad_gemma.py tests/test_profile_decode_jit.py` passed with `63 passed`, and the full cheap gate set passed: `.venv/bin/python -m pytest -q` reported `68 passed, 2 warnings`; `.venv/bin/tinygrad-gemma --help` exited 0; `PYTHONPATH=/Users/ericfode/src/.tinygrad_research:. .venv/bin/python scripts/smoke_metal.py` reported `default_device=METAL`, `loaded_model_device=METAL`, `logits_device=METAL`, `generated_tokens=4`, `rollout_jit_count=3`, and `decode_fallback=False`.
- Same-source profile control: `benchmarks/gemma4-metal-decode-graph-source-coherent-control-profile-512.json` reported complete attribution over `2994` source items, `7` `MetalGraph` batches, and `37.935 ms/token`. This control is structurally coherent but slower than the older accepted `34.319667 ms/token` artifact, so profile timing noise remains visible.
- `JIT_BATCH_SIZE=0` profile: `benchmarks/gemma4-metal-decode-graph-jitbatch0-profile-512.json` preserved complete attribution over `2994` source items and collapsed post-graph execution from `7` graph batches to `1`, with elapsed profile time `36.396 ms/token`. This is a `4.06%` same-state profile improvement, so the hard throughput gate was earned.
- Hard throughput gate: same-session default control `benchmarks/gemma4-metal-e2b-int8-1000-source-coherent-control-current.csv` measured `18.954322` warmup-excluded tok/s; `JIT_BATCH_SIZE=0` row `benchmarks/gemma4-metal-e2b-int8-1000-jitbatch0-current.csv` measured `18.685010` tok/s. Both rows generated `1000` tokens, measured `980` decode tokens after `20` warmup tokens, reported `rollout_jit_count=999`, `decode_fallback=false`, and stable output hash `aa4944455473e236807f6c711a77a969caf73680ea5ea9725648294e5fd23cfa`.
- Decision: reject `JIT_BATCH_SIZE=0` as a default runtime optimization. It improved graph shape and same-state profile elapsed, but worsened the hard metric by `0.269312 tok/s` (`-1.42%`) versus same-session control and landed `0.739026 tok/s` (`-3.80%`) below the accepted artifact floor of `19.424036 tok/s`. No throughput floor is superseded.
- Next target: `layer13-shared-source-producer-store-subattribution-042`. Add profiler-only sub-attribution inside `attention_packed_cache_write__role_shared_source__layer_13__type_sliding_attention` to separate K/V projection, RMSNorm/RoPE, RHS packing, and actual packed cache store. Do not write another runtime patch until this identifies removable or fusible work.

## 2026-04-26 - Graph-Batch Fusion Boundary Research / Source-Coherence Gate

- Objective: continue the `100 tok/s` optimization campaign by parsing the latest `METAL` graph profile for graph batches, source ranges, and fusion boundaries, then ranking the next measured probe by token throughput credibility.
- Profile finding: `benchmarks/gemma4-metal-decode-graph-attribution-only-restored-profile-512.json` contains `2994` original source items with complete attribution and `7` post-graph `MetalGraph` executions over source ranges `0..31`, `32..95`, `96..223`, `224..479`, `480..991`, `992..2015`, and `2016..2993`. Elapsed profile time is `34.319667 ms/token`.
- Boundary interpretation: the `MetalGraph` rows are tinygrad `JIT_BATCH_SIZE` grouping artifacts (`32`, `64`, `128`, `256`, `512`, `1024`, remainder `978`), not semantic model fusion boundaries. A synthetic local probe recorded at `benchmarks/tinygrad-metal-graph-jitbatch-policy-probe.json` confirmed the same grouping behavior and showed `JIT_BATCH_SIZE=0` can collapse 100 graphable kernels into one graph on the research tinygrad runtime.
- Bottleneck remains semantic cache-write work: source-attributed rollup is local packed cache writes `16.763029 ms` / `1709` source items, shared-source packed cache writes `15.973113 ms` / `1178` source items, and MLP `1.583524 ms` / `107` source items. The largest detailed bucket remains `attention_packed_cache_write__role_shared_source__layer_13__type_sliding_attention` at `11.798664 ms`, `885` source items, and `34.38%`.
- Source-coherence blocker: the checked-out `tinygrad_gemma/model.py` is currently split-cache (`GemmaCacheEntry` has only `key`, `value`, `length`, and `realize_cache_update` writes key/value separately) while the accepted profile artifacts, tests, and logs refer to final-lane packed cache APIs such as `make_packed_cache_entry`, `entry.packed`, packed cache update arguments, and rolled-cache helpers. `scripts/profile_decode_jit.py` was made import-compatible with both states, but that compatibility patch is profiler-only and does not make the current runtime equivalent to the accepted packed-cache artifact state.
- Council synthesis: three independent reviewers agreed that graph batches are not fusion boundaries, the layer-13 shared-source packed cache-write scope remains the dominant semantic target, and no new runtime/tok-s claim should be made until source and artifact state are coherent.
- Next target: `source-coherent-packed-cache-restore-and-jitbatch-probe-040`. First restore or identify the final-lane packed-cache source snapshot that produced the accepted artifacts, then run focused packed-cache tests. After coherence, run a same-state `JIT_BATCH_SIZE=0` graph profile and only proceed to a hard E2B int8 `METAL` `1000/20` row if profile elapsed improves without attribution/source-count regression. A runtime claim still requires stable hash, `rollout_jit_count=999`, `decode_fallback=false`, and repeated lower-row throughput above `19.424036 tok/s`.
- Verification in this pass: `.venv/bin/python -m pytest tests/test_profile_decode_jit.py -q` passed after the profiler compatibility patch. Full repo gates were intentionally not claimed because the current dirty source/test state is materially incoherent.

## 2026-04-26 - Donor Optimization Transfer Audit / SDPA Rejected

- Objective: audit `/Users/ericfode/src/gemma4-tinygrad-opt` for transferable optimization work into this repository, with token throughput as the objective metric and the current E2B int8 `METAL` `1000/20` floor target still `19.424036 tok/s` from `packed-cache-layer-attribution-probe-sweep-036`.
- Donor evidence quality: donor `evolution.jsonl` reports `26.971663` and `528.732270` `tokens_per_second`, but donor `benchmark.py` measures a 1000-token full forward on Modal CUDA/T4 with random tokens and bypasses autoregressive `generate`; it is not comparable to this repo's real-checkpoint E2B int8 Apple `METAL` decode metric. Treat it as a hint source only.
- Classification of donor optimizations:
  - Already present / stronger in recipient: final-lane packed K/V cache backing and packed assignment (`packed-kv-cache-assignment-033`, `packed-kv-lastdim-cache-assignment-034`); graphable fused int8 K/V projection (`graphable-fused-int8-kv-031`); fused gate/up paths; sliding-cache roll/window helper surface. The recipient versions include `.assign(...).realize()` and profile/test integration that the donor snapshot lacks in places.
  - Obsolete or unsafe as a direct port: donor packed-cache update lines that call `.assign(...)` without `.realize()` are incompatible with the local tinygrad 0.12 assignment semantics recorded in the optimization skill and repo history.
  - Transfer candidate tested: donor-style `Tensor.scaled_dot_product_attention` attention branch.
- SDPA probe result: a temporary recipient patch added an env-gated SDPA path and was then removed after measurement. Correctness-focused tests passed in both old/manual and SDPA modes, and the short E2B int8 `METAL` `128/32` rows produced the same output hash `1c39dd289bb7363f0f600ae1087ad39926e9733447df72bf03955c92d06066b0` with `decode_fallback=false`. However the measured decode suffix was slightly worse with SDPA (`30.769433 tok/s`) than the manual path (`30.808261 tok/s`).
- Profile rejection evidence: `benchmarks/gemma4-metal-decode-graph-sdpa-off-profile-512.json` reported `2994` kernels/source items and `44.466920` ms/token at the context-700 profile point. `benchmarks/gemma4-metal-decode-graph-sdpa-on-profile-512.json` kept the same `2994` source-item structure but worsened to `70.510040` ms/token. Source-attributed shared-source attention grew from `19.162958` ms to `38.811997` ms and MLP attribution from `2.963834` ms to `9.669416` ms.
- Decision: reject SDPA as a runtime patch and do not run the hard `1000/20` gate; the profile and short decode did not earn it. No donor optimization was accepted in this increment, and no throughput floor is superseded. The next useful target remains profiler-only layer-13 shared-source producer/store sub-attribution before another runtime patch.
- Verification after removing the temporary SDPA code: `.venv/bin/python -m py_compile tinygrad_gemma/model.py` passed; focused cache/full-forward tests passed with `7 passed, 2 warnings`; repo search found no `scaled_dot_product_attention`, `use_sdpa_attention`, or `TINYGRAD_GEMMA_SDPA` remnants in `tinygrad_gemma/model.py`; full `.venv/bin/python -m pytest -q` passed with `68 passed, 2 warnings`; `.venv/bin/tinygrad-gemma --help` exited 0; `PYTHONPATH=/Users/ericfode/src/.tinygrad_research:. .venv/bin/python scripts/smoke_metal.py` reported `default_device=METAL`, `loaded_model_device=METAL`, `logits_device=METAL`, `generated_tokens=4`, `rollout_jit_count=3`, and `decode_fallback=False`.

## 2026-04-26 - Shared-Source Layer 13 Windowed Allocation Rejected

- Objective: test `shared-source-layer13-cache-write-specialization-037`, a narrow runtime hypothesis that physically preallocating sliding shared-source cache storage at `sliding_window` length would reduce the dominant layer-13 shared-source packed write cost without changing the hard E2B semantics.
- Candidate behavior: the temporary patch made sliding shared-source cache producers allocate packed physical storage of `min(max_length, sliding_window)` and use windowed/ring writes for range and single-token updates. Focused red/green coverage passed, and the candidate full test run passed with `70 passed, 2 warnings` before rejection.
- Profile result: `PYTHONPATH=/Users/ericfode/src/.tinygrad_research:. .venv/bin/python scripts/profile_decode_jit.py --model-dir checkpoints/gemma-4-E2B-int8 --device METAL --context-length 512 --jit-mode 1 --out benchmarks/gemma4-metal-decode-graph-shared-source-windowed-profile-512.json --csv-out benchmarks/gemma4-metal-decode-graph-shared-source-windowed-profile-512.csv` produced `2994` source items, `7` MetalGraph batches, and `34.557625` ms/token.
- Decision: reject. The accepted control remains `benchmarks/gemma4-metal-decode-graph-attribution-only-restored-profile-512.json` at `2994` source items, `7` MetalGraph batches, and `34.319667` ms/token. The candidate did not lower source count and was slower than control; the hard E2B int8 `METAL` `1000/20` tok/s gate was therefore not earned.
- Bottleneck status after rejection: layer 13 shared-source sliding attention remained dominant in the candidate profile at `11.548308` ms, `885` source items, and `33.42%`; shared-source rollup remained `15.657384` ms over `1178` source items. The attempted physical-storage specialization did not attack the dependency-closed producer/store graph that attribution is actually measuring.
- Cleanup: removed the candidate windowed preallocation/runtime test surface and restored packed full-length profiler cache construction. A post-revert structural check, `benchmarks/gemma4-metal-decode-graph-shared-source-windowed-reverted-profile-512.json`, returned to `2994` source items and `7` MetalGraph batches, with a noisy `34.865500` ms/token timing.
- Verification after cleanup: `.venv/bin/python -m pytest -q` passed with `68 passed, 2 warnings`; `.venv/bin/tinygrad-gemma --help` exited 0; `PYTHONPATH=/Users/ericfode/src/.tinygrad_research:. .venv/bin/python scripts/smoke_metal.py` reported `default_device=METAL`, `loaded_model_device=METAL`, `generated_tokens=4`, `rollout_jit_count=3`, and `decode_fallback=False`.
- Next target: `layer13-shared-source-producer-store-attribution-038`. Add profiler-only sub-attribution inside the layer-13 shared-source packed write scope to separate K/V projection, RMSNorm/RoPE, RHS packing, and actual cache store before spending another runtime patch. Acceptance remains source count below `2994` or repeated long `1000/20` tok/s above `19.424036` with stable hash, `rollout_jit_count=999`, and `decode_fallback=false`.

## 2026-04-25 - Parallel Packed-Cache Probe Sweep

- Objective: answer the user request to run all credible next probes in parallel where safe, while keeping `tok/s` as the hard metric and serializing METAL throughput/profile gates to avoid contaminated measurements.
- Parallel worktrees created for `conditional-sliding-roll-parity-036`, packed cache layer/type attribution, flat-slot layout, active-cache no-op slice guard, deferred current-token readback/store ordering, and research-tinygrad runtime comparison. Only profiler-only attribution was merged to mainline; runtime candidates remained isolated until measured.
- Accepted mainline change: `scripts/profile_decode_jit.py` now preserves layer/type/role categories for packed cache writes. The restored mainline profile artifact `benchmarks/gemma4-metal-decode-graph-attribution-only-restored-profile-512.json`, generated with `PYTHONPATH=/Users/ericfode/src/.tinygrad_research:.`, reports complete attribution over `2994` source items, `7` MetalGraph batches, and `34.319667` ms/token.
- New exact bottleneck evidence: layer 13 shared-source sliding attention dominates the profiled token with `11.798664` ms, `885` source items, and `34.38%`; layer 14 shared-source full attention contributes `4.174449` ms, `293` source items, and `12.16%`. Rollup remains cache-write dominated: local `16.763` ms, shared source `15.973` ms, MLP `1.584` ms.
- Rejected runtime probes by profile evidence: parity-only profile worsened to `34.896750` ms/token; active-cache no-op guard worsened to `34.658542` ms/token; flat-slot packed layout worsened to `35.801833` ms/token; deferred readback/store ordering increased source count to `3594` and worsened to `36.976458` ms/token. All retained `7` graph batches except the source-count regression; none beat the accepted `2994` source-item structure.
- Additional rejection evidence: the merged parity+guard runtime candidate was tested before revert and produced repeated E2B int8 `METAL` `1000/20` rows of `18.320320` and `18.352303` tok/s, both below the accepted `19.053995` floor despite stable hash and `decode_fallback=false`.
- Restored-mainline hard rows after rejecting runtime hooks: `benchmarks/gemma4-metal-e2b-int8-1000-attribution-only-restored-current.csv` measured `19.424036` warmup-excluded tok/s and `benchmarks/gemma4-metal-e2b-int8-1000-attribution-only-restored-repeat.csv` measured `19.613212`, both with `generated_tokens=1000`, `measured_decode_tokens=980`, `rollout_jit_count=999`, stable hash `aa4944455473e236807f6c711a77a969caf73680ea5ea9725648294e5fd23cfa`, and `decode_fallback=false`. Treat `19.424036` as the refreshed observed floor for the unchanged runtime checkpoint, not as evidence that profiler instrumentation itself accelerated decode.
- Verification: `PYTHONPATH=. .venv/bin/python -m pytest -q` passed with `68 passed, 2 warnings`; `.venv/bin/tinygrad-gemma --help` exited 0; `PYTHONPATH=. .venv/bin/python scripts/profile_decode_jit.py --help` exited 0; `PYTHONPATH=. .venv/bin/python scripts/smoke_metal.py` reported `default_device=METAL`, `loaded_model_device=METAL`, `logits_device=METAL`, `generated_tokens=4`, `rollout_jit_count=3`, and `decode_fallback=False`.
- Next target: `shared-source-layer13-cache-write-specialization-037`. Start from the layer-13 shared-source sliding attention producer/store path rather than retrying broad parity/layout/deferred-store hooks. Acceptance remains repeated long `1000/20` improvement over `19.424036` tok/s or a source-count reduction below `2994`, with stable hash, `rollout_jit_count=999`, and `decode_fallback=false`.

## 2026-04-25 - Conditional Sliding-Roll Parity Research Refresh

- Objective: refreshed next-step research for the `100 tok/s` optimization campaign with tok/s as the hard metric and `conditional-sliding-roll-parity-036` as the active next target.
- Verified current state: accepted hard floor remains `packed-kv-lastdim-cache-assignment-034` at E2B int8 `METAL` `1000/20` floor `19.053995 tok/s`, clean repeat `19.081371 tok/s`, stable hash `aa4944455473e236807f6c711a77a969caf73680ea5ea9725648294e5fd23cfa`, `rollout_jit_count=999`, and `decode_fallback=false`. The accepted profile remains `2994` source items, `34.414750` ms/token, `7` MetalGraph batches, with cache-attributed time about `95.38%` of the profile.
- Verified benchmark-path asymmetry: `scripts/benchmark_gemma4_matrix.py` calls `model.generate(...)`; `tinygrad_gemma/loader.py` selects `GemmaForConditionalGeneration` for multimodal E2B; causal `GemmaForCausalLM.generate` rolls sliding cache entries before first sliding rollout JIT, but conditional `GemmaForConditionalGeneration.generate` still does not.
- Runtime proof on a forced small conditional generation: wrapping `tinygrad_gemma.model.roll_sliding_cache_entries` produced `roll_calls=[]` and `decode_fallback=False`, confirming the unpatched conditional path performs no roll transition even when `_sliding_decode_start()` is forced below the prompt length.
- Implementation guidance: patch `tinygrad_gemma/multimodal.py` only; import `roll_sliding_cache_entries`, add a one-time `rolled_sliding_cache` guard after JIT setup, and call `roll_sliding_cache_entries(cache, self.model.language_model.layers)` immediately before the first sliding-window rollout JIT call. Do not use `self.model.layers` in the conditional wrapper.
- Profile caveat: `benchmarks/gemma4-metal-decode-graph-packed-kv-lastdim-profiled.json` was generated with `/Users/ericfode/src/.tinygrad_research/tinygrad/__init__.py`; the repo venv imports `.venv/lib/python3.14/site-packages/tinygrad/__init__.py` unless `PYTHONPATH=/Users/ericfode/src/.tinygrad_research:.` is set. Future profile/source-count comparisons must record the tinygrad path.
- Next probes after parity remain, in order: layer/type attribution for packed cache writes, flat-slot packed cache layout `(B,H,L,2*D)`, active-cache no-op slice guard, then higher-risk current-token/deferred-store ordering. No runtime patch was landed in this research refresh, and no throughput row was superseded.

## 2026-04-25 - Packed Cache Next-Step Research

- Objective: researched the next credible optimization probes after `assign-realize-producer-decoupling-rejected-035`, keeping tok/s as the hard metric and preserving the accepted runtime baseline: E2B int8 `METAL` `1000/20` floor `19.053995` tok/s, `2994` source items, `34.414750` ms/token, `7` MetalGraph batches, stable hash, and `decode_fallback=false`.
- Evidence reviewed: current packed cache hot path (`tinygrad_gemma/model.py:201-213`, `280-302`, `561-620`), conditional/causal generation loops (`tinygrad_gemma/multimodal.py:813-839`, `tinygrad_gemma/model.py:1015-1049`), benchmark invocation (`scripts/benchmark_gemma4_matrix.py:137-139`), loader/config architecture selection (`tinygrad_gemma/loader.py:203`, `checkpoints/gemma-4-E2B-int8/config.json:1-4`), and profile artifacts including `benchmarks/gemma4-metal-decode-graph-packed-kv-lastdim-profiled.json`.
- Key new finding: the hard E2B checkpoint loads as `GemmaForConditionalGeneration`, and the benchmark calls `model.generate(...)`; the conditional generate path creates a sliding-window decode JIT but lacks the causal LM path's one-time `roll_sliding_cache_entries(...)` transition. This makes `conditional-sliding-roll-parity-036` the first runtime probe to test before more exotic cache layouts.
- Additional research: independent tinygrad-internal inspection found installed tinygrad's current `assign` scheduling forces non-trivial RHS realization before store lowering, while the local research checkout at `/Users/ericfode/src/.tinygrad_research` uses a `STORE+AFTER`-style assignment design. A read-only microbench reportedly reduced a variable-index packed-cache write from two ExecItems to one under the newer semantics, suggesting a framework-side research profile is worthwhile, but not yet an accepted repo-local runtime change.
- Roadmap artifact written: `docs/plans/2026-04-25-packed-cache-next-steps.md`. Ranked immediate probes are: conditional sliding-cache roll parity; packed slot-layout sweep (`(B,H,L,2*D)` and `(B,H,L,2,D)` against current `(B,H,L,D,2)`); profiler-only layer/type attribution for packed cache writes; active-cache no-op slice guard; and current-token readback/deferred-store ordering.
- Result: no runtime patch and no throughput row superseded. Next implementation loop should start with conditional generation sliding-roll parity unless a cheap research-tinygrad profile first proves a larger scheduler/store opportunity. Acceptance remains: source count below `2994` or repeated E2B int8 `METAL` `1000/20` floor above `19.053995` tok/s with stable hash, `rollout_jit_count=999`, and `decode_fallback=false`.

## 2026-04-25 - Assign/Realize Producer-Decoupling Rejected

- Hypothesis: option 2 from the packed-cache optimization loop was that tinygrad might offer a graphable way to assign cache storage without letting the cache `assign(...).realize()` pull the full lazy K/V producer graph into the cache-write realization scope. Invalidation criterion: source-level tinygrad mechanics prove dependency-closed realization is unavoidable for lazy RHS tensors, or a narrow assignment idiom probe fails to reduce the accepted `2994` captured source-item count and long-floor tok/s.
- Recovery: first restored the interrupted rejected prepacked-K/V cleanup to the accepted last-dimension packed-cache shape. Removed experimental prepacked helper tests, removed the profiler-only `packed_kv` sidecar argument, and restored cache-update call sites to inline final-lane packing via `realize_cache_update(..., single_position=query_len == 1, packed_cache=entry.packed)`.
- Research: inspected current repo hot path and tinygrad assignment mechanics. The installed tinygrad path still models `Tensor.assign` with assign/store lowering that schedules the RHS producer as part of realizing the assignment. The local research tinygrad checkout uses `STORE+AFTER`-style assignment that can fuse producer math into a store kernel, but the store remains dependency-closed over the producer graph. There is no high-level graphable tinygrad API that mutates cache storage from a lazy RHS while excluding the RHS producer work from the realized/captured graph. Pre-realizing the RHS is the only direct decoupling, and that is exactly the already-rejected producer materialization/prepacking shape.
- Existing artifact evidence remains decisive: accepted last-dimension packed K/V profile `benchmarks/gemma4-metal-decode-graph-packed-kv-lastdim-profiled.json` has `2994` source items, `34.414750` ms/token, and `7` MetalGraph batches, with `attention_packed_cache_write_local=1709`, `attention_packed_cache_write_shared_source=1178`, and `mlp=107`. Rejected prepacked K/V profile `benchmarks/gemma4-metal-decode-graph-prepacked-kv-current.json` worsened to `3009` source items and `34.718833` ms/token, then failed the repeated long-floor gate (`19.010608` and `18.867781` tok/s versus accepted `19.053995` tok/s), despite one attractive first row at `19.448882` tok/s.
- Narrow probe result: a read-only monkeypatch-style profile of a length-1 range assignment variant kept the same `2994` source items and `7` graph batches, but slowed the profiled token to `38.416875` ms/token versus the accepted `34.414750` ms/token. This does not justify a runtime patch.
- Result: rejected option 2 as an implementation increment for now. The accepted runtime checkpoint remains `packed-kv-lastdim-cache-assignment-034`; no throughput row is superseded, and the hard metric remains the clean E2B int8 `METAL` `1000/20` floor of `19.053995` tok/s with stable hash and `decode_fallback=false`.
- Verification after cleanup on 2026-04-25: focused gate `.venv/bin/python -m pytest -q tests/test_tinygrad_gemma.py tests/test_profile_decode_jit.py` passed with `59 passed, 2 warnings`; full `.venv/bin/python -m pytest -q` passed with `64 passed, 2 warnings in 44.50s`; `.venv/bin/tinygrad-gemma --help` exited 0; `.venv/bin/python scripts/smoke_metal.py` reported `default_device=METAL`, `loaded_model_device=METAL`, `logits_device=METAL`, `generated_tokens=4`, `rollout_jit_count=3`, and `decode_fallback=False`.
- Next target: stop trying producer-side prepacking unless a new tinygrad primitive changes the dependency boundary. The next plausible narrow surface is a cache storage/slot layout probe that might improve store locality or reduce packed cache-write source mass without explicit RHS materialization; any candidate must beat `2994` source items or the repeated `1000/20` long floor before acceptance.

## 2026-04-25 - Last-Dimension Packed K/V Cache Assignment

- Hypothesis: moving the packed K/V backing lane from a leading axis to the final dimension can make packed cache assignment more natural for tinygrad's slice/update graph: `packed.shape == (batch, heads, length, head_dim, 2)`, `key = packed[..., 0]`, and `value = packed[..., 1]`. Invalidation criterion: packed view assignment fails on `PYTHON` or `METAL`, focused cache/full-forward tests fail, profiler geometry diverges from runtime, decode falls back, output hashes change unexpectedly, or repeated long E2B int8 `METAL` `1000/20` rows fail to beat the prior `18.537605` tok/s packed-cache floor.
- Probe: a standalone tinygrad assignment check passed on both `PYTHON` and `METAL`, writing one position through `(B,H,L,D,2)` backing storage and observing the update through `key` / `value` views.
- Implemented the last-dimension layout in `tinygrad_gemma/model.py`: `make_packed_cache_entry` now allocates `(batch, heads, length, head_dim, 2)`, exposes `packed[..., 0]` / `packed[..., 1]`, and `realize_cache_update` writes packed single-position and range updates through final-dimension `cat(..., dim=-1)` slices. Updated `scripts/profile_decode_jit.py` to mirror the production geometry for profiler sidecar attribution.
- Updated focused coverage in `tests/test_tinygrad_gemma.py` so preallocated and rolled packed cache entries assert the new final-lane packed shape while preserving full-forward/cache equivalence.
- Artifacts: `benchmarks/gemma4-metal-e2b-int8-128-packed-kv-lastdim-current.csv`, `benchmarks/gemma4-metal-e2b-int8-128-packed-kv-lastdim-current.csv.progress.jsonl`, `benchmarks/gemma4-metal-decode-graph-packed-kv-lastdim-profiled.json`, `benchmarks/gemma4-metal-decode-graph-packed-kv-lastdim-profiled.csv`, `benchmarks/gemma4-metal-e2b-int8-1000-packed-kv-lastdim-current.csv`, `benchmarks/gemma4-metal-e2b-int8-1000-packed-kv-lastdim-current.csv.progress.jsonl`, `benchmarks/gemma4-metal-e2b-int8-1000-packed-kv-lastdim-repeat.csv`, `benchmarks/gemma4-metal-e2b-int8-1000-packed-kv-lastdim-repeat.csv.progress.jsonl`, `benchmarks/gemma4-metal-e2b-int8-1000-packed-kv-lastdim-repeat2.csv`, and `benchmarks/gemma4-metal-e2b-int8-1000-packed-kv-lastdim-repeat2.csv.progress.jsonl`.
- Result: accepted as a small tok/s improvement, with one explicitly recorded noisy outlier. The clean long E2B int8 `METAL`, beam=0, `1000/20` rows measured `19.053995` and `19.081371` warmup-excluded tok/s, both with `generated_tokens=1000`, `measured_decode_tokens=980`, `rollout_jit_count=999`, `decode_fallback=false`, and stable output hash `aa4944455473e236807f6c711a77a969caf73680ea5ea9725648294e5fd23cfa`. Use the lower clean row, `19.053995` tok/s, as the accepted floor; it supersedes `18.537605` tok/s by `0.516390` tok/s, or `2.785635%`. A middle repeat measured `15.557479` tok/s under visible background CPU load and is retained as an environment-contaminated outlier rather than used as an accepted floor.
- The short E2B int8 `METAL`, beam=0, `128/16` gate measured `30.838906` tok/s with stable output hash `1c39dd289bb7363f0f600ae1087ad39926e9733447df72bf03955c92d06066b0` and `decode_fallback=false`.
- The corrected post-window JIT=1 profile at `context_length=512`, using `PYTHONPATH=/Users/ericfode/src/.tinygrad_research`, reported `34.415` ms/token, `7` MetalGraph batches, and `source_attribution.status=complete`. Source count did not improve versus the leading-lane packed layout: still `2994` captured source items with `attention_packed_cache_write_local=1709`, `attention_packed_cache_write_shared_source=1178`, and `mlp=107`. Source-attributed elapsed shares were local `16.783` ms (`48.77%`), shared source `16.044` ms (`46.62%`), and MLP `1.588` ms (`4.62%`).
- Verification on 2026-04-25: focused cache/profile tests passed (`6 passed`); `tests/test_tinygrad_gemma.py tests/test_profile_decode_jit.py` passed (`59 passed, 2 warnings`); the short and long E2B int8 `METAL` gates above completed with `decode_fallback=false`; full `.venv/bin/python -m pytest -q` passed with `64 passed, 2 warnings in 44.45s`; `.venv/bin/tinygrad-gemma --help` exited 0; `.venv/bin/python scripts/smoke_metal.py` reported `default_device=METAL`, `loaded_model_device=METAL`, `logits_device=METAL`, `generated_tokens=4`, `rollout_jit_count=3`, and `decode_fallback=False`.
- Next target: last-dimension packing improves elapsed time but not source count. Continue attacking packed cache-write graph mass directly: produce already-packed K/V tensors before cache update, fuse or eliminate the remaining `cat`/view work, or find a graphable assignment idiom that reduces the `2994` captured source-item count while preserving the long `1000/20` floor.

## 2026-04-25 - Packed K/V Cache Assignment

- Hypothesis: storing key and value cache lanes in one packed backing tensor and updating them through one decode assignment can collapse the separate key-cache and value-cache write graphs while preserving the existing `entry.key` / `entry.value` view API. Invalidation criterion: cache/full-forward tests fail, profiler sidecar attribution cannot distinguish the new path, decode falls back, output hashes change unexpectedly, or the long E2B int8 `METAL` `1000/20` floor fails to improve.
- Implemented `GemmaCacheEntry.packed` plus `make_packed_cache_entry` in `tinygrad_gemma/model.py`. New preallocated cache entries now use a `(2, batch, heads, length, head_dim)` backing tensor with `key=packed[0]` and `value=packed[1]`; `realize_cache_update` writes one packed K/V tensor for both single-position and range updates. Rolled sliding cache entries are rebuilt through the same packed constructor.
- Updated `scripts/profile_decode_jit.py` so profiler sidecars understand `packed_cache`, track `attention_packed_cache_write_{local,shared_source,shared_consumer}` categories, and build its synthetic post-window cache with the production packed constructor rather than hand-allocating split key/value tensors.
- Added/updated focused coverage in `tests/test_tinygrad_gemma.py` and `tests/test_profile_decode_jit.py`: preallocated generation now asserts packed backing exists; rolled sliding entries retain packed storage; and the profiler cache-update sidecar records `attention_packed_cache_write_local` for the packed path.
- Artifacts: `benchmarks/gemma4-metal-e2b-int8-128-packed-kv-current.csv`, `benchmarks/gemma4-metal-e2b-int8-128-packed-kv-current.csv.progress.jsonl`, `benchmarks/gemma4-metal-e2b-int8-128-packed-kv-repeat.csv`, `benchmarks/gemma4-metal-e2b-int8-128-packed-kv-repeat.csv.progress.jsonl`, `benchmarks/gemma4-metal-e2b-int8-1000-packed-kv-current.csv`, `benchmarks/gemma4-metal-e2b-int8-1000-packed-kv-current.csv.progress.jsonl`, `benchmarks/gemma4-metal-e2b-int8-1000-packed-kv-repeat.csv`, `benchmarks/gemma4-metal-e2b-int8-1000-packed-kv-repeat.csv.progress.jsonl`, `benchmarks/gemma4-metal-decode-graph-packed-kv-profiled.json`, and `benchmarks/gemma4-metal-decode-graph-packed-kv-profiled.csv`.
- Result: accepted as a substantial tok/s improvement. The repeated E2B int8 `METAL`, beam=0, `1000/20` row measured `18.537605` warmup-excluded tok/s with `generated_tokens=1000`, `measured_decode_tokens=980`, `rollout_jit_count=999`, `decode_fallback=false`, and stable output hash `aa4944455473e236807f6c711a77a969caf73680ea5ea9725648294e5fd23cfa`. This supersedes the shared-source-roll repeated floor of `11.740118` tok/s by `6.797487` tok/s, or `57.899648%`.
- The short E2B int8 `METAL`, beam=0, `128/16` repeat measured `30.377797` tok/s with stable output hash `1c39dd289bb7363f0f600ae1087ad39926e9733447df72bf03955c92d06066b0` and `decode_fallback=false`.
- The corrected post-window JIT=1 profile at `context_length=512`, using `PYTHONPATH=/Users/ericfode/src/.tinygrad_research`, reported `36.865167` ms/token, `7` MetalGraph batches, and `source_attribution.status=complete`. Captured source items dropped from the split-cache `5239` shape to `2994`: `attention_packed_cache_write_local=1709`, `attention_packed_cache_write_shared_source=1178`, and `mlp=107`. Source-attributed elapsed shares remain dominated by packed cache writes: local `17.713` ms (`48.05%`), shared source `17.409` ms (`47.22%`), and MLP `1.744` ms (`4.73%`).
- Verification on 2026-04-25: focused cache tests passed (`7 passed, 35 deselected`); `tests/test_profile_decode_jit.py` passed (`17 passed`); the long and short E2B int8 `METAL` gates above both completed with `decode_fallback=false`; full `.venv/bin/python -m pytest -q` passed with `64 passed, 2 warnings in 44.15s`; `.venv/bin/tinygrad-gemma --help` exited 0; `.venv/bin/python scripts/smoke_metal.py` reported `default_device=METAL`, `loaded_model_device=METAL`, `logits_device=METAL`, `generated_tokens=4`, `rollout_jit_count=3`, and `decode_fallback=False`.
- Next target: packed cache writes are still the bottleneck and now the explicit surface. Investigate reducing packed cache-write source count and elapsed time, especially avoiding K/V packing graph inflation (`squeeze`/`unsqueeze`/`cat`) or producing already-packed K/V tensors directly, while preserving the long `1000/20` floor and MetalGraph batching.

## 2026-04-25 - Shared Sliding-Source Cache Roll

- Hypothesis: a sliding shared-KV source can use the same rolled physical storage as ordinary sliding producers during one-token decode, while full-attention shared sources must remain absolute. For E2B, layer 13 is the sliding shared source feeding sliding shared consumers, and layer 14 is the full-attention shared source feeding full shared consumers. Invalidation criterion: shared-source cache tests fail, full-forward/cache equivalence fails after the window, decode falls back, hashes change, or the long E2B int8 `METAL` `1000/20` floor does not improve.
- Implemented a narrower `roll_sliding_cache_entries` legality condition in `tinygrad_gemma/model.py`: shared full-attention sources still keep absolute-position storage, but shared sliding sources can roll because one-token causal sliding attention observes K/V as a key-value set, not by storage order, when no mask is needed and K/V are permuted together.
- Added `tests/test_tinygrad_gemma.py::test_roll_sliding_cache_entries_rolls_shared_sliding_sources_only` and `test_rolled_shared_sliding_source_matches_full_forward_after_window`. The tests verify that the sliding shared source rolls, the full shared source remains absolute, and a rolled shared sliding source still matches full forward after the sliding window.
- Artifacts: `benchmarks/gemma4-metal-e2b-int8-128-shared-source-roll-current.csv`, `benchmarks/gemma4-metal-e2b-int8-128-shared-source-roll-current.csv.progress.jsonl`, `benchmarks/gemma4-metal-e2b-int8-1000-shared-source-roll-current.csv`, `benchmarks/gemma4-metal-e2b-int8-1000-shared-source-roll-current.csv.progress.jsonl`, `benchmarks/gemma4-metal-e2b-int8-1000-shared-source-roll-repeat.csv`, `benchmarks/gemma4-metal-e2b-int8-1000-shared-source-roll-repeat.csv.progress.jsonl`, `benchmarks/gemma4-metal-decode-graph-shared-source-roll-current.json`, and `benchmarks/gemma4-metal-decode-graph-shared-source-roll-current.csv`.
- Result: accepted as a small repeated long-floor improvement. The first E2B int8 `METAL`, beam=0, `1000/20` run measured `11.762429` warmup-excluded tok/s with `generated_tokens=1000`, `measured_decode_tokens=980`, `rollout_jit_count=999`, `decode_fallback=false`, and stable output hash `aa4944455473e236807f6c711a77a969caf73680ea5ea9725648294e5fd23cfa`. The repeat measured `11.740118` tok/s with the same hash and no fallback. Use the repeat as the accepted floor: it supersedes `11.378305` tok/s by `0.361813` tok/s, or `3.179850%`.
- The short E2B int8 `METAL`, beam=0, `128/16` gate measured `19.303487` tok/s with stable output hash `1c39dd289bb7363f0f600ae1087ad39926e9733447df72bf03955c92d06066b0`, which sits inside the prior short-run noise band (`19.087715` first fused K/V run, `19.425499` repeat). Treat the long floor, not the short gate, as the acceptance evidence.
- The post-window JIT=1 profile at `context_length=512`, using `PYTHONPATH=/Users/ericfode/src/.tinygrad_research`, reported `68.873` ms/token, `8` MetalGraph batches, and `source_attribution.status=complete`. Source counts remained unchanged: `attention_key_cache_write_local=1696`, `attention_value_cache_write_local=1696`, `attention_key_cache_write_shared_source=1176`, `attention_value_cache_write_shared_source=564`, and `mlp=107`. Because the profiler row worsened from the previous `66.724416` ms/token while generation improved, treat this profile as noisy structural evidence rather than a complete explanation of the throughput win.
- Verification on 2026-04-25: focused sliding/shared cache tests passed (`5 passed, 37 deselected`); full `.venv/bin/python -m pytest -q` passed with `63 passed, 2 warnings in 45.24s`; `.venv/bin/tinygrad-gemma --help` exited 0; `.venv/bin/python scripts/smoke_metal.py` reported `default_device=METAL`, `generated_tokens=4`, `rollout_jit_count=3`, and `decode_fallback=False`.
- Next target: local producer cache writes remain the largest surface and source counts did not drop. Evaluate a packed decode-only local K/V assignment or another graphable way to collapse/sequencing-reduce the two local producer cache writes while preserving MetalGraph batching and the long `1000/20` floor.

## 2026-04-25 - Graphable Fused Int8 K/V Projection

- Hypothesis: for non-shared attention producer layers with runtime rowwise-int8 `k_proj` and `v_proj`, a graphable tinygrad-native fused K/V projection can reduce producer elapsed time without introducing raw custom Runners or breaking MetalGraph batching. Invalidation criterion: fused/separate projections differ, cache/full-forward tests fail, decode falls back, output hash changes unexpectedly, MetalGraph batching regresses, or the accepted E2B int8 `METAL` tok/s floor does not improve.
- Implemented `_FUSED_INT8_KV`, `GemmaAttention._can_use_fused_int8_kv`, `_fused_int8_kv_weight_scale`, and `_project_kv` in `tinygrad_gemma/model.py`. The fused path concatenates K/V int8 weights and row scales, runs one tinygrad `matmul(..., dtype="float")`, applies concatenated scales, chunks the result, and leaves K/V RMSNorm, RoPE, cache semantics, and shared-KV layers unchanged.
- Added `tests/test_tinygrad_gemma.py::test_runtime_int8_attention_fused_kv_matches_separate_path`, mirroring the existing fused MLP test and comparing fused vs separate projected K/V tensors on a quantized runtime checkpoint.
- Artifacts: `benchmarks/gemma4-metal-e2b-int8-128-fused-kv-current.csv`, `benchmarks/gemma4-metal-e2b-int8-128-fused-kv-current.csv.progress.jsonl`, `benchmarks/gemma4-metal-e2b-int8-128-fused-kv-repeat.csv`, `benchmarks/gemma4-metal-e2b-int8-128-fused-kv-repeat.csv.progress.jsonl`, `benchmarks/gemma4-metal-e2b-int8-1000-fused-kv-current.csv`, `benchmarks/gemma4-metal-e2b-int8-1000-fused-kv-current.csv.progress.jsonl`, `benchmarks/gemma4-metal-e2b-int8-1000-fused-kv-warmup600-current.csv`, `benchmarks/gemma4-metal-e2b-int8-1000-fused-kv-warmup600-current.csv.progress.jsonl`, `benchmarks/gemma4-metal-decode-graph-fused-kv-current.json`, and `benchmarks/gemma4-metal-decode-graph-fused-kv-current.csv`.
- Result: accepted as a small runtime improvement on the repo's current long-floor metric. The E2B int8 `METAL`, beam=0, `1000/20` row measured `11.378305` warmup-excluded tok/s with `generated_tokens=1000`, `measured_decode_tokens=980`, `rollout_jit_count=999`, `decode_fallback=false`, and stable output hash `aa4944455473e236807f6c711a77a969caf73680ea5ea9725648294e5fd23cfa`. This supersedes the prior accepted `11.076822` tok/s floor, but remains nowhere near the 100 tok/s target.
- The short E2B int8 `METAL`, beam=0, `128/16` gate was noisy: first fused run measured `19.087715` tok/s, while the repeat measured `19.425499` tok/s with the same stable output hash `1c39dd289bb7363f0f600ae1087ad39926e9733447df72bf03955c92d06066b0`. The prior accepted short row was `19.199621`, so use the repeat only as supportive evidence, not as the principal acceptance gate.
- The post-window fused K/V JIT=1 profile at `context_length=512` reports `source_attribution.status=complete`, `8` MetalGraph batches, and `66.724416` ms/token, improved from the prior rolled-cache `70.717625` ms/token artifact. Current category timings are `attention_key_cache_write_local=21.687557` ms, `attention_value_cache_write_local=20.624879` ms, `attention_key_cache_write_shared_source=15.659857` ms, `attention_value_cache_write_shared_source=7.175080` ms, and `mlp=1.577043` ms. Source counts remain unchanged, so the change reduces elapsed producer cost but does not solve cache-write graph fragmentation.
- The post-window suffix benchmark with `decode_warmup_tokens=600` measured `13.928612` tok/s, worse than the immediately preceding rolled-cache warmup600 experiment at `14.794015` tok/s. Treat steady-state suffix measurements as noisy until repeated on an idle device; do not claim a steady-state suffix win from this increment.
- Verification on 2026-04-25: fused K/V and fused MLP projection tests passed (`2 passed`); preallocated cache/full-forward/rolled-cache/generation focused tests passed (`4 passed`); `tests/test_profile_decode_jit.py` passed (`16 passed`); full `.venv/bin/python -m pytest -q` passed with `62 passed, 2 warnings in 44.55s`; `.venv/bin/python scripts/smoke_metal.py` reported `default_device=METAL`, `generated_tokens=4`, `rollout_jit_count=3`, and `decode_fallback=False`; `.venv/bin/tinygrad-gemma --help` exited 0.
- A context-length `700`, JIT=1 profile experiment produced `2029.568` ms/token despite still reporting `8` MetalGraph batches and the same `5239` source items. Given user guidance about possible device contention and the mismatch with ordinary generation throughput, treat this artifact as suspected device/pathology evidence, not an accepted bottleneck fact.
- Next target: source counts and local cache-write fragmentation remain unchanged. Evaluate one narrow structural cache layout change, preferably packed decode-only local K/V cache assignment, or separately prove whether rolling sliding shared-source storage is legal and beneficial.

## 2026-04-25 - Single-Position and Rolled Sliding Cache Writes

- Hypothesis: narrowing decode cache writes to single-position assignment, then rolling sliding-window producer cache entries into physical modulo storage after the sliding-window JIT switch, can reduce required cache-write graph cost without changing Gemma cache semantics. Invalidation criterion: focused cache tests fail, decode falls back, output hashes become unstable on the benchmark gate, graph attribution becomes incomplete, or the real tok/s row regresses.
- Implemented `GemmaCacheEntry.window`, `roll_sliding_cache_entry`, and `roll_sliding_cache_entries` in `tinygrad_gemma/model.py`. Sliding producers can now be compacted into a physical ring where decode writes use `absolute_position % window`; full-attention layers, shared-KV consumers, and full-length shared-KV sources remain absolute. The helper now skips `store_full_length_kv` only when `num_kv_shared_layers > 0`, so small configs without shared consumers can still roll legal sliding producers.
- Also kept the decode cache-write helper on a single-position path for `query_len == 1`, using `key_cache[:, :, start, :]` / `value_cache[:, :, start, :]` assignments instead of one-token range slices. `scripts/profile_decode_jit.py` mirrors this path for profiler sidecar attribution.
- Repaired profiler import compatibility with stock tinygrad by making `ExecContext`, `resolve_params`, `linear_to_schedule`, `pm_post_sched_cache`, and version-specific ignored `Ops` members optional at import time. Live captured-JIT profiling still requires the local tinygrad research checkout, but plain `pytest` and `profile_decode_jit.py --help` no longer fail during import.
- Artifacts: `benchmarks/gemma4-metal-e2b-int8-128-rolled-cache-current.csv`, `benchmarks/gemma4-metal-e2b-int8-128-rolled-cache-current.csv.progress.jsonl`, `benchmarks/gemma4-metal-e2b-int8-1000-rolled-cache-current.csv`, `benchmarks/gemma4-metal-e2b-int8-1000-rolled-cache-current.csv.progress.jsonl`, `benchmarks/gemma4-metal-decode-graph-rolled-cache-current.json`, and `benchmarks/gemma4-metal-decode-graph-rolled-cache-current.csv`.
- Result: accepted as a small runtime improvement. The short E2B int8 `METAL`, beam=0, `128/16` row measured `19.199621` warmup-excluded tok/s with `generated_tokens=128`, `measured_decode_tokens=112`, `rollout_jit_count=127`, `decode_fallback=false`, and output hash `1c39dd289bb7363f0f600ae1087ad39926e9733447df72bf03955c92d06066b0`. Because this gate starts before `sliding_start=511`, treat this gain as primarily single-position cache assignment rather than proof of the ring buffer.
- The accepted long E2B int8 `METAL`, beam=0, `1000/20` floor is now `11.076822` warmup-excluded tok/s with `generated_tokens=1000`, `measured_decode_tokens=980`, `rollout_jit_count=999`, `decode_fallback=false`, and output hash `aa4944455473e236807f6c711a77a969caf73680ea5ea9725648294e5fd23cfa`. This supersedes the prior accepted `10.863932` tok/s floor, but remains nowhere near the 100 tok/s target.
- The post-window rolled JIT=1 profile at `context_length=512` reports complete source attribution over `5239` source items and `8` MetalGraph batches. Source-attributed elapsed time improved from the prior `79.534916` ms/token artifact to `70.717625` ms/token. Current category timings are `attention_key_cache_write_local=23.406207` ms, `attention_key_cache_write_shared_source=15.811661` ms, `attention_value_cache_write_local=22.659419` ms, `attention_value_cache_write_shared_source=7.250811` ms, and `mlp=1.589527` ms.
- Verification on 2026-04-25: focused rolled-cache tests passed (`6 passed`); `tests/test_profile_decode_jit.py` passed both with stock tinygrad and with `PYTHONPATH=/Users/ericfode/src/.tinygrad_research`; full `.venv/bin/python -m pytest -q` passed with `61 passed, 2 warnings in 55.04s`; `.venv/bin/tinygrad-gemma --help` exited 0; and `.venv/bin/python scripts/smoke_metal.py` reported `default_device=METAL`, `generated_tokens=4`, `rollout_jit_count=3`, and `decode_fallback=False`.
- Next target: local producer cache writes remain essentially unchanged and still dominate the post-window profile. Attack cache-write graph fragmentation or required producer work while preserving MetalGraph batching; do not resurrect raw custom Runners unless they can stay inside graph capture.

## 2026-04-25 - UOp Sidecar Source-Attributed Timing

- Hypothesis: the previous 719-item final decode graph tail can be classified
  without changing model runtime by tagging UOp creation while profiler scopes
  are active and propagating that side table through `UOp.replace`.
  Invalidation criterion: focused profiler tests fail, the real E2B int8
  `METAL` artifact keeps an unattributed tail, or the profiler changes Gemma
  throughput behavior.
- Implemented profiler-only UOp sidecar attribution in
  `scripts/profile_decode_jit.py`, including `UOpMetaClass.__call__` tagging
  and `UOp.replace` side-table propagation. Added JSON
  `source_attributed_summary` output so benchmark artifacts report timing by
  best available source category, not only the coarse lowered row category.
- Artifact: `benchmarks/gemma4-metal-decode-graph-default-current.json` and
  `benchmarks/gemma4-metal-decode-graph-default-current.csv`.
- Result: accepted as profiler/benchmark instrumentation. The refreshed E2B
  int8 `METAL` JIT=2 profile has `source_attribution.status=complete`,
  `original_exec_count=5239`, `attributed_source_count=5239`,
  `unparsed_graph_batches=0`, and `unattributed_tail_count=0`.
- The former 719-item tail is now attributed by UOp sidecar metadata rather
  than left as `other`: the accepted artifact reports
  `category_basis_counts={"repo_sidecar_realize_scope_metadata": 4520,
  "repo_sidecar_uop_creation_metadata": 719}`.
- Source-attributed timing in the accepted artifact: total profiled token
  `79.534916` ms; `attention_key_cache_write_local=23.419337` ms,
  `attention_key_cache_write_shared_source=21.428166` ms,
  `attention_value_cache_write_local=23.801497` ms,
  `attention_value_cache_write_shared_source=7.927042` ms, and
  `mlp=2.958875` ms. Cache-write work is therefore `76.576041` ms,
  or about `96.28%` of the profiled decode-token time.
- Throughput gate: `.venv/bin/python scripts/benchmark_gemma4_matrix.py
  --root checkpoints --sizes E2B --formats int8 --devices METAL --beams 0
  --max-new-tokens 128 --decode-warmup-tokens 16 --progress-every 64
  --out benchmarks/gemma4-metal-e2b-int8-128-current.csv` produced an `ok`
  row with `generated_tokens=128`, `measured_decode_tokens=112`,
  `measured_decode_tokens_per_second=17.504447`, `rollout_jit_count=127`,
  and `decode_fallback=false`.
- No long-floor Gemma throughput row was superseded. Current accepted E2B int8
  `METAL`, `beam=0`, `1000/20` floor remains `10.863932` tok/s until rerun
  on the same gate.
- Verification on 2026-04-25: `.venv/bin/python -m pytest
  tests/test_profile_decode_jit.py -q` passed; the real E2B int8 `METAL`
  profiler command refreshed the JSON/CSV artifacts; full
  `.venv/bin/python -m pytest -q` passed with `57 passed, 2 warnings in
  58.43s`; `.venv/bin/tinygrad-gemma --help` exited 0; and
  `.venv/bin/python scripts/smoke_metal.py` reported `default_device=METAL`,
  `generated_tokens=4`, `rollout_jit_count=3`, and `decode_fallback=False`.
- Next target: stop treating MLP as the main 100 tok/s lever. The next runtime
  increment should attack attention cache-write kernel count/fragmentation or a
  legal cache layout/store idiom for required producer writes.

## 2026-04-25 - Unclassified Final-Tail Structural Attribution

- Hypothesis: the 719 unclassified final-tail source items are structurally
  identifiable from lowered source item display names, UOp roots, and operation
  signatures even when tinygrad metadata is empty. Invalidation criterion:
  source attribution becomes incomplete, focused profiler tests fail, or the
  unclassified summaries remain empty/non-distinguishing in the real METAL
  artifact.
- Implemented profiler-only structural summaries in
  `scripts/profile_decode_jit.py` for unclassified source items:
  `program_type_counts`, `display_name_counts`, `ast_root_counts`, and
  `op_signature_counts`. Added focused coverage in
  `tests/test_profile_decode_jit.py`.
- Artifact: `benchmarks/gemma4-metal-decode-graph-default-current.json` and
  `benchmarks/gemma4-metal-decode-graph-default-current.csv`.
- Result: accepted as profiler instrumentation. The refreshed default E2B int8
  `METAL` JIT=1 graph profile still has 8 `MetalGraph` rows and complete
  attribution over all 5239 source items:
  `source_attribution.status=complete`, `original_exec_count=5239`,
  `attributed_source_count=5239`, `unparsed_graph_batches=0`, and
  `unattributed_tail_count=0`.
- The remaining 719 unclassified source items are all `Ops.SINK` roots in the
  final `<batched 1175>` graph range. Top display signatures in the accepted
  artifact include `r_16_96=176`, `E_16_32_3=70`, `E_16_32_3n1=35`,
  `r_1536_16_16n1=35`, and several final-logits/norm-shaped reduce/index
  signatures.
- No Gemma throughput row was superseded. Current accepted E2B int8 `METAL`,
  `beam=0`, `1000/20` floor remains `10.863932` tok/s with
  `rollout_jit_count=999` and `decode_fallback=false`.
- Verification on 2026-04-25:
  `.venv/bin/python -m pytest -q tests/test_profile_decode_jit.py` passed with
  `11 passed, 2 warnings`; `.venv/bin/python -m py_compile
  scripts/profile_decode_jit.py` passed; and the real E2B int8 `METAL`
  profiler command refreshed the JSON/CSV artifacts.
- Next target: add profiler-only sidecar scopes around final language-model
  norm, logits/softcap, and `sample_next` realization boundaries to classify the
  final tail before attempting another runtime patch.

## 2026-04-25 - Shared-KV Cache-Write Elision Rejected

- Hypothesis: some cache-write source items come from shared-KV consumer
  layers, which would make them candidates for legal elision or narrowing.
  Invalidation criterion: the real graph profile shows zero
  `shared_consumer` cache-write source items, graph attribution becomes
  incomplete, or the role tagging changes runtime behavior.
- Research: the E2B config has 35 layers and `num_kv_shared_layers=20`.
  Layers 15-34 are shared-KV consumers; layers 13 and 14 are the
  `store_full_length_kv` shared sources; earlier layers are local producers.
- Extended the behavior-preserving `realize_cache_update` helper to accept
  layer-role metadata, then patched it only inside
  `scripts/profile_decode_jit.py` to tag key/value cache writes as
  `local`, `shared_source`, or unexpected `shared_consumer`.
- Artifact: `benchmarks/gemma4-metal-decode-graph-default-current.json` and
  `benchmarks/gemma4-metal-decode-graph-default-current.csv`.
- Result: rejected shared-consumer cache-write elision. The refreshed default
  E2B int8 `METAL` JIT=1 graph profile still has 8 `MetalGraph` rows and
  complete attribution over all 5239 source items:
  `source_attribution.status=complete`, `original_exec_count=5239`,
  `attributed_source_count=5239`, `unparsed_graph_batches=0`, and
  `unattributed_tail_count=0`.
- Cache-write role counts in the accepted artifact:
  `attention_key_cache_write_local=1696`,
  `attention_value_cache_write_local=1696`,
  `attention_key_cache_write_shared_source=564`,
  `attention_value_cache_write_shared_source=564`, and `other=719`.
  There are zero `attention_*_shared_consumer` source items.
- Conclusion: shared-KV consumer layers already avoid cache writes. The next
  useful target is not shared-consumer elision; it is either required producer
  cache-write optimization or classification of the remaining 719 unclassified
  source items in the final graph range.
- No Gemma throughput row was superseded. Current accepted E2B int8 `METAL`,
  `beam=0`, `1000/20` floor remains `10.863932` tok/s with
  `rollout_jit_count=999` and `decode_fallback=false`.
- Verification on 2026-04-25:
  `.venv/bin/python -m pytest -q tests/test_profile_decode_jit.py
  tests/test_tinygrad_gemma.py::test_preallocated_generate_matches_dynamic_cache_for_gemma4`
  passed with `12 passed, 2 warnings`; `.venv/bin/python -m py_compile
  scripts/profile_decode_jit.py tinygrad_gemma/model.py` passed; and the real
  E2B int8 `METAL` profiler command refreshed the JSON/CSV artifacts.
- Next target: classify the 719 unclassified source items in the final graph
  range before attempting another runtime patch.

## 2026-04-25 - Key/Value Cache-Write Split Diagnostic

- Hypothesis: splitting the cache-write profiler sidecar into key-cache and
  value-cache stores will show whether one side dominates the 4520
  attention-cache-write source items. Invalidation criterion: the split changes
  runtime behavior, reintroduces the paired-store assign-graph cycle, makes
  graph attribution incomplete, changes graph batching unexpectedly, or leaves
  the real METAL artifact at generic `attention_cache_write`.
- Research: the rejected paired-store prototype proved that building both
  assign tensors before the first realize triggers tinygrad's assign-graph
  cycle detector. This increment keeps the original sequential realization
  order and only factors it through `realize_cache_update`.
- Implemented a behavior-preserving `realize_cache_update` helper in
  `tinygrad_gemma/model.py`, then patched it only inside
  `scripts/profile_decode_jit.py` to wrap the key-cache and value-cache store
  realizes with separate sidecar scopes.
- Artifact: `benchmarks/gemma4-metal-decode-graph-default-current.json` and
  `benchmarks/gemma4-metal-decode-graph-default-current.csv`.
- Result: accepted as profiler instrumentation. The refreshed default E2B
  int8 `METAL` JIT=1 graph profile still has 8 `MetalGraph` rows and complete
  attribution over all 5239 source items: `source_attribution.status=complete`,
  `original_exec_count=5239`, `attributed_source_count=5239`,
  `unparsed_graph_batches=0`, and `unattributed_tail_count=0`.
- The cache-write attribution splits evenly:
  `original_capture.category_counts={"attention_key_cache_write": 2260,
  "attention_value_cache_write": 2260, "other": 719}`.
- The slowest graph ranges in the accepted artifact are headed by
  `<batched 2048>` at `25.112875` ms for source range `2016-4063`,
  `<batched 1175>` at `19.617708` ms for range `4064-5238`, and
  `<batched 1024>` at `11.937708` ms for range `992-2015`.
- No Gemma throughput row was superseded. Current accepted E2B int8 `METAL`,
  `beam=0`, `1000/20` floor remains `10.863932` tok/s with
  `rollout_jit_count=999` and `decode_fallback=false`.
- Verification on 2026-04-25:
  `.venv/bin/python -m pytest -q tests/test_profile_decode_jit.py
  tests/test_tinygrad_gemma.py::test_preallocated_generate_matches_dynamic_cache_for_gemma4`
  passed with `11 passed, 2 warnings`; `.venv/bin/python -m py_compile
  scripts/profile_decode_jit.py tinygrad_gemma/model.py` passed; the real E2B
  int8 `METAL` profiler command refreshed the JSON/CSV artifacts; and
  `.venv/bin/tinygrad-gemma --help` exited 0.
- Next target: inspect Gemma 4 shared-KV/full-attention layer behavior and test
  whether any cache write can be legally elided or narrowed for shared KV
  layers without changing logits.

## 2026-04-25 - Attention Cache-Write Realize Scope Diagnostic

- Hypothesis: the 4520 attention-scoped source items in the default JIT=1
  graph profile are mostly cache-store realization, not attention score math.
  Invalidation criterion: adding a profiler-only `Tensor.realize` refinement
  for attention-scoped `Ops.STORE` graphs leaves the source items labeled as
  generic `attention`, makes graph attribution incomplete, changes graph
  batching unexpectedly, or fails focused profiler tests.
- Implemented a profiler-only `Tensor.realize` wrapper in
  `scripts/profile_decode_jit.py`. While the sidecar stack is inside
  `GemmaAttention`, realized tensors whose UOp graph contains `Ops.STORE` are
  tagged as `attention_cache_write`; all patches are restored after profiling.
- Artifact: `benchmarks/gemma4-metal-decode-graph-default-current.json` and
  `benchmarks/gemma4-metal-decode-graph-default-current.csv`.
- Result: accepted as profiler instrumentation. The refreshed default E2B
  int8 `METAL` JIT=1 graph profile still has 8 `MetalGraph` rows and complete
  attribution over all 5239 source items: `source_attribution.status=complete`,
  `original_exec_count=5239`, `attributed_source_count=5239`,
  `unparsed_graph_batches=0`, and `unattributed_tail_count=0`.
- The prior generic attention attribution is now fully split:
  `original_capture.category_counts={"attention_cache_write": 4520,
  "other": 719}` and
  `original_capture.category_basis_counts={"repo_sidecar_realize_scope_metadata":
  4520, "unclassified_source_item_metadata": 719}`.
- The slowest graph ranges in the accepted artifact are headed by
  `<batched 2048>` at `24.880917` ms for source range `2016-4063`,
  `<batched 1175>` at `19.847833` ms for range `4064-5238`, and
  `<batched 1024>` at `11.365792` ms for range `992-2015`.
- No Gemma throughput row was superseded. Current accepted E2B int8 `METAL`,
  `beam=0`, `1000/20` floor remains `10.863932` tok/s with
  `rollout_jit_count=999` and `decode_fallback=false`.
- Verification on 2026-04-25:
  `.venv/bin/python -m pytest -q tests/test_profile_decode_jit.py` passed with
  `9 passed, 2 warnings`; `.venv/bin/python -m py_compile
  scripts/profile_decode_jit.py` passed; and the real E2B int8 `METAL`
  profiler command refreshed the JSON/CSV artifacts.
- Next target: test a paired key/value cache-store realization in
  `GemmaAttention`, then accept only if it preserves semantics and improves a
  real no-fallback benchmark row with nonzero `rollout_jit_count`.

## 2026-04-25 - Gemma Profiler Sidecar Category Attribution

- Hypothesis: integrating the repo-local `TinyJit.add_linear` sidecar into
  `scripts/profile_decode_jit.py` around Gemma decode methods will recover
  non-`other` category attribution for the default JIT=1 MetalGraph profile
  without changing model runtime behavior. Invalidation criterion: focused
  tests fail, the sidecar patch leaks after profiling, graph source attribution
  becomes incomplete, graph batching changes unexpectedly, or the refreshed
  real METAL profile remains entirely `other`.
- Implemented profiler-only sidecar scopes for embedding/per-layer projection,
  norm, attention, MLP, decoder residual, and logits/argmax methods. The patch
  temporarily wraps `TinyJit.add_linear` and restores all monkeypatched methods
  after capture.
- Tightened artifact truthfulness by separating native source-line metadata
  from `repo_sidecar_realize_scope_metadata`. The sidecar labels the Python
  realization scope that caused tinygrad to capture a `LINEAR` call; it is not
  a native per-operation source category.
- Artifact: `benchmarks/gemma4-metal-decode-graph-default-current.json` and
  `benchmarks/gemma4-metal-decode-graph-default-current.csv`.
- Result: accepted as profiler instrumentation. The default E2B int8 `METAL`
  JIT=1 graph profile still has 8 `MetalGraph` rows and complete attribution
  over all 5239 source items: `source_attribution.status=complete`,
  `original_exec_count=5239`, `attributed_source_count=5239`,
  `unparsed_graph_batches=0`, and `unattributed_tail_count=0`.
- The refreshed source attribution is no longer all `other`. The captured
  source items report `category_counts={"attention": 4520, "other": 719}` and
  `category_basis_counts={"repo_sidecar_realize_scope_metadata": 4520,
  "unclassified_source_item_metadata": 719}`. At the graph-row level,
  `source_attribution.category_basis_counts` is
  `{"repo_sidecar_realize_scope_metadata": 8}`.
- The five slowest graph ranges in the accepted artifact are headed by
  `<batched 2048>` at `24.921083` ms for source range `2016-4063`,
  `<batched 1175>` at `19.539667` ms for range `4064-5238`, and
  `<batched 1024>` at `11.205000` ms for range `992-2015`.
- No Gemma throughput row was superseded. Current accepted E2B int8 `METAL`,
  `beam=0`, `1000/20` floor remains `10.863932` tok/s with
  `rollout_jit_count=999` and `decode_fallback=false`.
- Verification on 2026-04-25:
  `.venv/bin/python -m pytest -q tests/test_profile_decode_jit.py` passed with
  `8 passed, 2 warnings`; `.venv/bin/python -m py_compile
  scripts/profile_decode_jit.py` passed; the real E2B int8 `METAL` profiler
  command refreshed the JSON/CSV artifacts; `.venv/bin/python -m json.tool`
  passed for the loop-state and graph-profile JSON artifacts; `git diff
  --check` passed; and `.venv/bin/tinygrad-gemma --help` exited 0.
- Next target: split the attention realization sidecar into cache-write,
  q/k/v projection, score/softmax/value, and output-projection regions, then
  rerun the default graph profile to choose the next real throughput patch from
  measured attention-scope evidence instead of guesswork.

## 2026-04-25 - Repo Sidecar Metadata Bypass Probe

- Hypothesis: tinygrad's native Tensor metadata remains lost before
  `CapturedJit.linear`, but a repo-local Python scope sidecar can tag captured
  `LINEAR` calls at `TinyJit.add_linear` time and preserve that metadata
  through `linear_to_schedule` and lowered `ExecItem.metadata`. Invalidation
  criterion: sidecar metadata fails to appear on captured calls or lowered
  `ExecItem`s in either JIT=1 or JIT=2.
- Extended `scripts/diagnose_tinygrad_metadata_preservation.py` with a scoped
  sidecar proof. The diagnostic temporarily patches `TinyJit.add_linear`,
  adds `Metadata(name="toy_sidecar", caller="repo_sidecar:1::toy_sidecar")`
  to empty captured call metadata when a Python sidecar scope is active, and
  restores the original method after the probe.
- Artifact: `benchmarks/tinygrad-metadata-preservation-current.json`.
- Result: native metadata is still lost
  (`captured_call_metadata_count=0`,
  `captured_ast_toposort_metadata_count=0`,
  `lowered_exec_item_metadata_count=0`), but the sidecar path succeeds:
  `sidecar_captured_call_metadata_count=2` and
  `sidecar_lowered_exec_item_metadata_count=2` across JIT=1 and JIT=2.
- This accepts the bypass approach for profiler attribution only. It does not
  change model runtime behavior and does not patch the tinygrad checkout.
- Verification on 2026-04-25:
  `.venv/bin/python -m json.tool` passed for the loop-state and diagnostic
  JSON artifacts; `git diff --check` passed; `.venv/bin/python -m py_compile
  scripts/diagnose_tinygrad_metadata_preservation.py` passed; rerunning
  `.venv/bin/python scripts/diagnose_tinygrad_metadata_preservation.py
  --device METAL --out benchmarks/tinygrad-metadata-preservation-current.json`
  reproduced `sidecar_captured_call_metadata_count=2` and
  `sidecar_lowered_exec_item_metadata_count=2`; `.venv/bin/tinygrad-gemma
  --help` exited 0; full `.venv/bin/python -m pytest -q` passed with
  `47 passed, 2 warnings in 59.37s`; and `.venv/bin/python
  scripts/smoke_metal.py` reported `default_device=METAL`,
  `generated_tokens=4`, `rollout_jit_count=3`, and
  `decode_fallback=False`.
- No Gemma throughput row was superseded. Current accepted E2B int8 `METAL`,
  `beam=0`, `1000/20` floor remains `10.863932` tok/s with
  `rollout_jit_count=999` and `decode_fallback=false`.
- Next target: integrate the sidecar into `scripts/profile_decode_jit.py`
  around Gemma attention, MLP, norm, embedding, decoder residual, and
  logits/argmax scopes, then rerun the default JIT=1 graph profile to verify
  non-`other` category attribution without claiming a throughput win.

## 2026-04-25 - Captured Linear Metadata Preservation Probe

- Hypothesis: current tinygrad creates Tensor caller metadata under
  `TRACEMETA=2`, but loses it before or during `TinyJit` capture/lowering into
  `CapturedJit.linear`. Invalidation criterion: a minimal captured call or
  lowered `ExecItem` still contains caller metadata, which would put the bug in
  the Gemma profiler rather than tinygrad capture metadata preservation.
- Implemented `scripts/diagnose_tinygrad_metadata_preservation.py`, a minimal
  METAL diagnostic that compares direct lazy Tensor metadata against the same
  operation through `TinyJit` under JIT=1 and JIT=2, then inspects
  `CapturedJit.linear` calls and lowered `ExecItem.metadata`.
- Artifact: `benchmarks/tinygrad-metadata-preservation-current.json`.
- Result: `summary.status=metadata_lost_before_captured_linear`.
  Direct lazy Tensor ops preserved 6 metadata entries with callers from
  `lazy_metadata_probe`, but JIT=1 and JIT=2 both reported
  `captured_call_metadata_count=0`,
  `captured_ast_toposort_metadata_count=0`, and
  `lowered_exec_item_metadata_count=0`.
- Conclusion: the Gemma graph/source profiler is not the layer losing source
  categories. The category data is gone before `CapturedJit.linear`, so JIT=2
  sidecars and graph-batch joins cannot recover categories without either a
  tinygrad metadata-retention fix or a repo-local pre-capture metadata sidecar.
- Verification on 2026-04-25:
  `.venv/bin/python -m py_compile
  scripts/diagnose_tinygrad_metadata_preservation.py` passed, and
  `.venv/bin/python scripts/diagnose_tinygrad_metadata_preservation.py
  --device METAL --out benchmarks/tinygrad-metadata-preservation-current.json`
  wrote the artifact above. `.venv/bin/python -m json.tool` passed for the
  loop-state and diagnostic JSON artifacts; `git diff --check` passed;
  `.venv/bin/tinygrad-gemma --help` exited 0; full
  `.venv/bin/python -m pytest -q` passed with `47 passed, 2 warnings in
  59.20s`; and `.venv/bin/python scripts/smoke_metal.py` reported
  `default_device=METAL`, `generated_tokens=4`, `rollout_jit_count=3`, and
  `decode_fallback=False`.
- No Gemma throughput row was superseded. Current accepted E2B int8 `METAL`,
  `beam=0`, `1000/20` floor remains `10.863932` tok/s with
  `rollout_jit_count=999` and `decode_fallback=false`.
- Next target: test a minimal metadata-retention fix in the local tinygrad
  capture path or a repo-local pre-capture metadata sidecar. Do not make
  another model-path optimization until profiler category attribution is
  trustworthy.

## 2026-04-25 - JIT2 Sidecar Category Recapture Rejected

- Hypothesis: a current JIT=2 ungraphed sidecar profile has the same source-row
  sequence as the JIT=1 `MetalGraph` source ranges and preserves enough
  metadata to recover category counts for each graph batch. Invalidation
  criterion: sidecar row-count mismatch, all categories remain `other`, or
  sidecar rows fail to cover the graph source range.
- Recovery baseline: current accepted E2B int8 `METAL`, `beam=0`, `1000/20`
  throughput remains `10.863932` warmed tok/s in
  `benchmarks/gemma4-metal-int8-fused-gate-up-1000-current.csv` with
  `decode_fallback=false` and `rollout_jit_count=999`.
- Research: the previous default graph artifact already proved complete
  structural coverage: 8 `MetalGraph` rows map to all 5239 lowered source
  items with no unparsed batch and no unattributed tail. The open question was
  whether JIT=2 could provide source categories for that same row sequence.
- Invalidation run 1:
  `env DEBUG=0 .venv/bin/python scripts/profile_decode_jit.py --jit-mode 2
  --metal-int8-gate-up default --out
  /tmp/tinygrad-gemma-decode-jit2-source-current.json --csv-out
  /tmp/tinygrad-gemma-decode-jit2-source-current.csv` produced 5239 rows and
  `source_attribution.status=complete`, but
  `summary.by_category={"other": ...}` and
  `category_basis_counts={"unclassified_source_item_metadata": 5239}`.
- Invalidation run 2 set `TRACEMETA=2` at process start:
  `env DEBUG=0 TRACEMETA=2 .venv/bin/python scripts/profile_decode_jit.py
  --jit-mode 2 --metal-int8-gate-up default --out
  /tmp/tinygrad-gemma-decode-jit2-source-tracemeta-current.json --csv-out
  /tmp/tinygrad-gemma-decode-jit2-source-tracemeta-current.csv`. It again
  produced 5239 rows, all category `other`, with
  `category_basis_counts={"unclassified_source_item_metadata": 5239}`.
- Preserved the second real METAL sidecar as
  `benchmarks/gemma4-metal-decode-source-jit2-tracemeta-current.json` and
  `benchmarks/gemma4-metal-decode-source-jit2-tracemeta-current.csv` so the
  rejected sidecar path is tied to repo artifacts.
- Result: reject the simple JIT=2 sidecar join. Count alignment is good, but
  current `CapturedJit.linear`/lowering metadata is already unclassified before
  graph batching, so joining it to graph ranges would only propagate false
  `other` categories.
- No Gemma throughput row was superseded.
- Next target: build a minimal current-tinygrad metadata-preservation probe
  around Tensor metadata, `CapturedJit.linear` calls, and
  `linear_to_schedule`/`ExecItem` metadata before making another model-path
  optimization.

## 2026-04-25 - Default Graph Batch Attribution

- Hypothesis: the default JIT=1 post-window `MetalGraph` profile can be made
  structurally attributable under current tinygrad by lowering
  `CapturedJit.linear` graph calls and mapping each `<batched N>` row to the
  graph runner's source `ExecItem` range. Invalidation criterion: any unparsed
  graph display, source-count mismatch, row/item count mismatch, or
  unattributed source tail.
- Research: current tinygrad no longer exposes the old
  `captured.jit_cache`/`captured._jit_cache` surface used by the profiler.
  The current path is `CapturedJit.linear`; graph calls lower through
  `exec_graph` using `resolve_params`, `MultiBuffer` flattening, and
  `Device[graph_device].graph(...)`.
- Implemented `scripts/profile_decode_jit.py` support for lowering current
  graph calls into executable profile rows, recording `source_start`,
  `source_end`, `source_count`, `source_category_counts`, and
  `source_category_basis` in the CSV/JSON artifacts.
- Added focused coverage in `tests/test_profile_decode_jit.py` for complete
  batch attribution, unparsed graph displays, batch-count mismatches, and the
  unclassified-source-metadata boundary.
- Real artifact:
  `benchmarks/gemma4-metal-decode-graph-default-current.json` now reports
  `source_attribution.status=complete`, `original_exec_count=5239`,
  `attributed_source_count=5239`, `unparsed_graph_batches=0`,
  `source_count_mismatches=0`, and `unattributed_tail_count=0`.
- The refreshed profile still has 8 `MetalGraph` launches and measured the
  profiled token at `78.078` ms. The slowest rows are `<batched 2048>` at
  `26.167417` ms for source range `2016-4063`, `<batched 1175>` at
  `20.695792` ms for range `4064-5238`, and `<batched 1024>` at
  `12.203750` ms for range `992-2015`.
- Invalidation attempt: category metadata did not survive the current lowered
  graph source items. The artifact records
  `category_basis_counts={"unclassified_source_item_metadata": 8}` so the
  graph range map is accepted while source-category bottleneck claims remain
  explicitly blocked.
- Verification on 2026-04-25: `.venv/bin/python -m pytest -q
  tests/test_profile_decode_jit.py` passed with `6 passed, 2 warnings`;
  `.venv/bin/python -m py_compile scripts/profile_decode_jit.py` passed; the
  real E2B int8 `METAL` profiler command refreshed the JSON/CSV artifacts;
  `.venv/bin/python -m json.tool configs/repo-loop-state.json` passed;
  `git diff --check` passed; `.venv/bin/tinygrad-gemma --help` exited 0;
  full `.venv/bin/python -m pytest -q` passed with `47 passed, 2 warnings in
  56.86s`; and `.venv/bin/python scripts/smoke_metal.py` reported
  `default_device=METAL`, `generated_tokens=4`, `rollout_jit_count=3`, and
  `decode_fallback=False`.
- No Gemma throughput row was superseded. Current accepted E2B int8 `METAL`,
  `beam=0`, `1000/20` floor remains `10.863932` tok/s with
  `rollout_jit_count=999` and `decode_fallback=false`.
- Next target: recover source-category attribution for the complete graph
  ranges from a current ungraphed sidecar profile or tinygrad source metadata,
  then choose the next real model-path bottleneck.

## 2026-04-25 - Raw Gate/Up Path Abandoned

- Hypothesis: raw Metal rowwise-int8 gate/up can only remain a useful 100 TPS
  route if it can be made graphable or at least kept as a reliable explicit
  diagnostic mode without risking production benchmark contamination.
- Recovery evidence already invalidated the graphability part: the default
  post-window decode capture condenses to 8 `MetalGraph` batches, while raw
  gate/up fragments into 37 `MetalGraph` batches plus 35 custom
  `RowwiseInt8DecodeLinearRunner` launches and regresses the profiled token
  from `67.716500` ms to `197.940124` ms.
- Fresh invalidation on 2026-04-25: running
  `env DEBUG=0 .venv/bin/python scripts/profile_decode_jit.py --jit-mode 1
  --metal-int8-gate-up raw --out /tmp/tinygrad-gemma-raw-gate-up-private.json
  --csv-out /tmp/tinygrad-gemma-raw-gate-up-private.csv` failed before
  producing a row with `RuntimeError: Invalid library file` from the raw Metal
  program compile path. Re-running outside the sandbox produced the same
  failure.
- Implemented the abandonment instead of another raw-kernel tuning pass:
  `GemmaMLP` no longer imports or dispatches
  `metal_rowwise_int8_decode_linear`; the fused runtime-int8 MLP path stays on
  the graphable tinygrad-native matmul/scale implementation.
- `scripts/profile_decode_jit.py --metal-int8-gate-up raw` now exits with an
  explicit abandonment message, and
  `scripts/diagnose_metal_mlp_runtime_gate_up.py` writes
  `summary.status=abandoned_graph_breaking_runner`.
- Refreshed
  `benchmarks/gemma4-metal-mlp-runtime-gate-up-diagnostic-current.json` with
  that abandoned status so the durable artifact no longer claims the raw path
  is a live optimization candidate.
- Verification on 2026-04-25: full tests passed with `43 passed, 2 warnings in
  60.42s`; `.venv/bin/tinygrad-gemma --help` exited 0;
  `.venv/bin/python -m py_compile scripts/profile_decode_jit.py
  scripts/diagnose_metal_mlp_runtime_gate_up.py tinygrad_gemma/model.py`
  passed; `git diff --check` passed; `.venv/bin/python scripts/smoke_metal.py`
  reported `default_device=METAL`, `generated_tokens=4`,
  `rollout_jit_count=3`, and `decode_fallback=False`.
- No Gemma throughput row was superseded. Current accepted E2B int8 `METAL`,
  `beam=0`, `1000/20` floor remains `10.863932` tok/s with
  `rollout_jit_count=999` and `decode_fallback=false`.
- Next target: attribute the default JIT=1 post-window `MetalGraph` batches
  back to JIT=2 source categories or runner ranges, then choose the next real
  bottleneck among norm, attention, logits/argmax, or cache growth.

## 2026-04-25 - Benchmark Acceptance Beam 0 Contract

- Recovery found pre-existing dirty benchmark state in
  `scripts/benchmark_gemma4_matrix.py` and untracked historical benchmark CSV
  artifacts. The dirty loader override was tested as the current hypothesis:
  forcing an int8-specific loader kwarg might preserve int8 benchmark
  truthfulness.
- The hypothesis was invalidated against the current loader contract:
  `load_pretrained` now accepts `runtime_quantization`, not
  `weight_only_quantize`; a stale kwarg would turn real benchmark attempts into
  `load_error` rows instead of producing acceptance evidence.
- Kept the useful part of the dirty benchmark change: the default beam matrix
  now includes `beam=0`, which is the E2B int8 `METAL` acceptance row used by
  this repo-loop lane.
- Added focused tests in `tests/test_benchmark_gemma4_matrix.py` for both the
  default beam list and the `benchmark_checkpoint` call boundary to
  `load_pretrained`.
- Verification on 2026-04-25: `.venv/bin/python -m pytest -q
  tests/test_benchmark_gemma4_matrix.py` passed with `5 passed, 2 warnings in
  0.14s`; `.venv/bin/python -m py_compile scripts/benchmark_gemma4_matrix.py`
  passed.
- No Gemma throughput row was superseded. Current accepted E2B int8 `METAL`,
  `beam=0`, `1000/20` floor remains `10.863932` tok/s with
  `rollout_jit_count=999` and `decode_fallback=false`.
- Next target remains `graphable-compiled-int8-gate-up-or-abandon`: determine
  whether a graphable compiled/tinygrad-native rowwise-int8 gate/up path can be
  built in-repo; if not, explicitly abandon raw gate/up and move to the next
  measured decode bottleneck.

## 2026-04-25 - Raw Gate/Up Graph Fragmentation Diagnostic

- Extended `scripts/profile_decode_jit.py` with `--jit-mode` and
  `--metal-int8-gate-up` so the same real E2B int8 `METAL` post-window decode
  token can be profiled on either the default fused rowwise-int8 path or the
  opt-in raw Metal gate/up path under tinygrad graph batching.
- Read the local tinygrad graph path: `apply_graph_to_jit` only batches
  graphable `CompiledRunner`/copy-style items, and `MetalGraph` rejects a
  batch unless every item is a `CompiledRunner`. The repo-local
  `RowwiseInt8DecodeLinearRunner` is therefore replay-safe but not graphable.
- Refreshed the default graph profile in
  `benchmarks/gemma4-metal-decode-graph-default-current.json`. The original
  capture has 4807 `CompiledRunner` items and no raw runners; tinygrad condenses
  it to 8 `MetalGraph` batches. The profiled token measured `67.716500` ms.
- Added the raw gate/up graph profile in
  `benchmarks/gemma4-metal-decode-graph-raw-gate-up-current.json`. The original
  capture grows to 13657 `CompiledRunner` items plus 35
  `RowwiseInt8DecodeLinearRunner` items; graph execution fragments into 37
  `MetalGraph` batches plus 35 raw runners. The profiled token measured
  `197.940124` ms.
- Conclusion: the raw gate/up path is slower because it breaks Metal graph
  batching and inflates the surrounding tinygrad capture, not because the raw
  rowwise-int8 kernel itself is slow. Keep
  `TINYGRAD_GEMMA_METAL_INT8_GATE_UP=1` as an experiment only.
- No Gemma throughput row was superseded. Current accepted E2B int8 `METAL`,
  `beam=0`, `1000/20` floor remains `10.863932` tok/s with
  `rollout_jit_count=999` and `decode_fallback=false`.
- Next target: pursue a graphable compiled/tinygrad-native rowwise-int8 gate/up
  route, or explicitly abandon raw Metal gate/up in favor of larger decode
  bottlenecks. Do not spend another loop tuning the custom Python `Runner`.

## 2026-04-24 - Guarded Gemma MLP Metal Runtime Gate-Up Rejected

- Wired the replay-safe raw Metal rowwise-int8 gate/up primitive into the
  existing fused `GemmaMLP` runtime-int8 path behind strict guards: METAL only,
  inference only through the existing fused-int8 predicate, exactly one decode
  row, and hidden size at most 4096.
- Added `scripts/diagnose_metal_mlp_runtime_gate_up.py`, a focused METAL
  diagnostic that compares the raw-Metal MLP gate/up path against the existing
  tinygrad fused rowwise-int8 path at the E2B gate/up shape
  `(hidden_size=1536, intermediate_size=6144)`.
- The focused diagnostic artifact is
  `benchmarks/gemma4-metal-mlp-runtime-gate-up-diagnostic-current.json`; it
  reports `safe_for_decode_tinyjit_replay` with `max_abs_error=2.288818359375e-05`
  for `bfloat16` under both `JIT=1` and `JIT=2`.
- The real model row rejected the default path. E2B int8 `METAL`, `beam=0`,
  `--max-new-tokens 200 --decode-warmup-tokens 20` measured only
  `6.609982` tok/s with `rollout_jit_count=199` and
  `decode_fallback=false` in
  `benchmarks/gemma4-metal-runtime-gate-up-200-current.csv`, down from the
  previous accepted fused-rowwise-int8 200/20 row of `16.025808` tok/s.
- The raw-Metal MLP path is now explicit opt-in only via
  `TINYGRAD_GEMMA_METAL_INT8_GATE_UP=1` or the diagnostic's private force flag.
  Default decode remains on the existing tinygrad fused rowwise-int8 matmul path
  to avoid a performance regression.
- No Gemma throughput row was superseded. Current accepted E2B int8 `METAL`,
  `beam=0`, `1000/20` floor remains `10.863932` tok/s with
  `rollout_jit_count=999` and `decode_fallback=false`.
- Next target: treat the Python custom `Runner` as graph-breaking until proven
  otherwise. Either make the raw gate/up path graph-compatible or abandon it in
  favor of a tinygrad-native lowering/fusion route.

## 2026-04-24 - Metal Runtime Captured Runner Bridge

- Updated `tinygrad_gemma/metal_int8.py` so
  `metal_rowwise_int8_decode_linear` appends a repo-local custom
  `RowwiseInt8DecodeLinearRunner` `ExecItem` when called during tinygrad
  `TinyJit` capture, then executes the same runner for the capture call.
- This keeps the raw Metal primitive narrow and repo-local. It does not patch
  tinygrad and does not wire the primitive into Gemma MLP yet.
- Refreshed
  `benchmarks/gemma4-metal-runtime-bridge-tinyjit-diagnostic-current.json`.
  The diagnostic now reports `summary.status=safe_for_decode_tinyjit_replay`.
- Replay proof: under both `JIT=1` and `JIT=2`, the `float32` probe captures
  one runner item and the `bfloat16` probe captures two items, the normal
  tinygrad cast plus the custom runner. All probes change output with changed
  inputs and match the NumPy reference across replay calls. The previous stale
  replay failure is gone.
- No Gemma throughput row was superseded. Current accepted E2B int8 `METAL`,
  `beam=0`, `1000/20` floor remains `10.863932` tok/s with
  `rollout_jit_count=999` and `decode_fallback=false`.
- Verification: focused non-METAL boundary test passed; the real `METAL`
  TinyJit bridge diagnostic passed with all four JIT/dtype probes
  `safe_for_tinyjit_replay`.
- Next target: wire the replay-safe primitive into the existing fused
  rowwise-int8 MLP gate/up path behind strict METAL/decode guards, then measure
  a real E2B row only if synthetic correctness and replay accounting hold.

## 2026-04-24 - Metal Runtime TinyJit Bridge Diagnostic

- Added `scripts/diagnose_metal_runtime_tinyjit_bridge.py` to check the
  reusable raw Metal rowwise-int8 tensor wrapper under `TinyJit` capture and
  replay before wiring it into Gemma decode.
- The diagnostic artifact is
  `benchmarks/gemma4-metal-runtime-bridge-tinyjit-diagnostic-current.json`.
- Result: direct decode wiring is unsafe. The `float32` probe fails to capture
  with `JitError: didn't JIT anything!`. The `bfloat16` probe captures one
  tinygrad cast kernel, but the raw `MetalProgram` launch is not a captured
  `ExecItem`; replay returns the capture-time raw-Metal output for changed
  inputs.
- The stale replay is large enough to be decisive:
  `call_index=2` has `max_abs_error=279.96636962890625`, and `call_index=3`
  has `max_abs_error=261.729248046875`, both while
  `matches_capture_output=true`.
- No Gemma throughput row was superseded. Current accepted E2B int8 `METAL`,
  `beam=0`, `1000/20` floor remains `10.863932` tok/s with
  `rollout_jit_count=999` and `decode_fallback=false`.
- Verification: the diagnostic ran on real `METAL` and wrote the artifact. The
  next target is to make the primitive replay-safe by adding a repo-local custom
  TinyJit `ExecItem`/`Runner` bridge during capture, or patch tinygrad only if
  that cannot be made correct in-repo.

## 2026-04-24 - Metal Rowwise Int8 Runtime Bridge

- Added `tinygrad_gemma/metal_int8.py`, a reusable METAL-only runtime
  primitive for the threadgroup-staged rowwise-int8 decode linear. It compiles
  the raw Metal kernel through tinygrad's runtime, accepts tinygrad tensors,
  writes a tinygrad output buffer, and returns a tensor shaped like
  `(*x.shape[:-1], out_features)`.
- Kept the boundary intentionally narrow: the primitive currently supports one
  decode row, int8 rank-2 weights, float32 row scales, and at most 4096 input
  features for the threadgroup-staged hidden vector. Non-METAL tensors raise
  instead of silently falling back.
- Updated `scripts/prototype_metal_rowwise_int8_linear.py` to import the shared
  Metal source and call `metal_rowwise_int8_decode_linear` as a module-level
  tensor wrapper check. The current artifact records
  `module_check.output_shape=[1, 1, 12288]`,
  `module_check.max_abs_error=0.00023651123046875`, and
  `module_check.passed=true`.
- Refreshed `benchmarks/gemma4-metal-rowwise-int8-linear-prototype-current.json`.
  The reusable threadgroup-x path measured `0.123437` ms median /
  `0.101625` ms min over 100 replays at local size 128 for the E2B gate/up
  shape.
- Added `test_metal_rowwise_int8_decode_linear_rejects_non_metal` so the new
  primitive's execution boundary is pinned without requiring METAL in the unit
  suite.
- This is still not a model throughput row. It moves the raw kernel from a
  standalone script into a repo runtime primitive that can be wired into decode
  next.
- Current accepted E2B int8 `METAL`, `beam=0`, `1000/20` floor remains
  `10.863932` tok/s with `rollout_jit_count=999` and
  `decode_fallback=false`.
- Verification: focused non-METAL boundary test passed; full pytest reported
  `37 passed, 1 skipped, 2 warnings in 42.62s`;
  `.venv/bin/tinygrad-gemma --help` exited 0; `scripts/smoke_metal.py`
  reported `default_device=METAL`, `generated_tokens=4`,
  `rollout_jit_count=3`, and `decode_fallback=False`; the prototype reran
  with `--repeats 100`; `configs/repo-loop-state.json` validated with
  `python -m json.tool`; `git diff --check` passed.
- Next target: use the runtime primitive in a guarded decode-only
  fused-gate/up experiment and measure whether the real E2B row improves
  without losing TinyJit replay accounting.

## 2026-04-24 - Raw Metal Rowwise Int8 Linear Prototype

- Added `scripts/prototype_metal_rowwise_int8_linear.py`, a lower-level
  prototype that compiles raw Metal source through tinygrad's `MetalProgram`,
  launches over tinygrad `Buffer`s, and checks the output against a NumPy
  rowwise-int8 reference.
- The prototype covers the E2B fused gate/up decode shape:
  `(1536) @ (12288, 1536).T`, with float input/output for this first
  lower-level kernel proof.
- The global-x scalar variant is correct but does not provide a strong win:
  `0.208875` ms median over 100 replays in
  `benchmarks/gemma4-metal-rowwise-int8-linear-prototype-current.json`.
- The threadgroup-x variant stages the single decode hidden vector in Metal
  threadgroup memory. It is correct with `max_abs_error=0.00023651123046875`
  and measured `0.123437` ms median / `0.101625` ms min over 100 replays at
  local size 128.
- A small local-size sweep found local size 128 as the best durable current
  setting for this prototype.
- This is not a model throughput row. It proves a repo-local lower-level Metal
  path is viable and faster than the previous one-token rowwise-int8 lowering
  diagnostic (`0.185354` ms median), but it is not yet wired into TinyJit
  replay or Gemma MLP execution.
- Current accepted E2B int8 `METAL`, `beam=0`, `1000/20` floor remains
  `10.863932` tok/s with `rollout_jit_count=999` and
  `decode_fallback=false`.
- Verification: `36 passed, 1 skipped, 2 warnings in 44.63s`;
  `.venv/bin/tinygrad-gemma --help` exited 0; `scripts/smoke_metal.py`
  reported `default_device=METAL`, `generated_tokens=4`,
  `rollout_jit_count=3`, and `decode_fallback=False`; `git diff --check`
  passed; `configs/repo-loop-state.json` and the prototype artifact validated
  with `python -m json.tool`.
- Next target: bridge the threadgroup-x prototype into a reusable decode-only
  runtime path without breaking TinyJit replay accounting, then measure a real
  E2B int8 row.

## 2026-04-24 - Metal Int8 Matmul Lowering Diagnostic

- Added `scripts/diagnose_metal_int8_matmul.py` to benchmark and inspect
  tinygrad METAL lowering for the Gemma 4 E2B decode-shape rowwise-int8 gate/up
  linear: `(1, token_count, 1536) @ (12288, 1536).T`.
- Diagnostic artifact:
  `benchmarks/gemma4-metal-int8-matmul-lowering-diagnostic-current.json`.
- The tinygrad Metal tensor-core registry visible to this repo has entries for
  float, half, and bfloat16, but not int8.
- For the exact one-token decode shape, `rowwise_int8_linear` captured one
  kernel, did not emit `simdgroup_multiply_accumulate`, and measured
  `0.185354` ms median over 20 diagnostic replays. The bf16 dequantized-weight
  control also captured one non-simdgroup kernel and measured `0.146021` ms.
- For the token-count 8 control, tinygrad can emit a simdgroup MMA kernel, but
  that does not change the actual decode conclusion: the current one-token
  rowwise-int8 path is not a hardware int8 tensor-core GEMV/GEMM path.
- No throughput row was superseded by this diagnostic. The current accepted
  E2B int8 `METAL`, `beam=0`, `1000/20` floor remains `10.863932` tok/s with
  `rollout_jit_count=999` and `decode_fallback=false`.
- Verification: `36 passed, 1 skipped, 2 warnings in 40.99s`;
  `.venv/bin/tinygrad-gemma --help` exited 0; `scripts/smoke_metal.py`
  reported `default_device=METAL`, `generated_tokens=4`,
  `rollout_jit_count=3`, and `decode_fallback=False`; the diagnostic script
  reran with `--repeats 20`; `git diff --check` passed; JSON artifacts
  validated with `python -m json.tool`.
- Next target: stop expecting another `model.py` reshape to produce a 10x win.
  Prototype or modify a lower-level Metal rowwise-int8 decode linear path, or
  choose an explicit semantic-breaking attention approximation as a separate
  diagnostic.

## 2026-04-24 - Fused Rowwise Int8 Gate-Up

- Added an inference-only fused rowwise-int8 MLP gate/up path in
  `GemmaMLP`. When both projections are runtime `RowwiseInt8Linear`, decode
  caches a concatenated int8 gate/up weight plus a concatenated scale vector,
  performs one float-accumulating matmul, applies the row scales, casts back to
  the input dtype, and then splits gate/up. The fused cache is module-external
  and not part of `nn.state.get_state_dict`.
- Added `test_runtime_int8_mlp_fused_gate_up_matches_separate_path` to compare
  the fused path against the original separate rowwise-int8 projections on a
  quantized synthetic checkpoint.
- Synthetic context-700 profile did not predict the real row: total time moved
  from `96.651` ms to `97.738` ms and the MLP bucket increased from
  `22.190` ms to `29.967` ms, while `other` dropped from `31.712` ms to
  `25.728` ms.
- Real E2B int8 `METAL`, `beam=0`, `--max-new-tokens 200
  --decode-warmup-tokens 20` measured `16.025808` tok/s with
  `rollout_jit_count=199` and `decode_fallback=false`, up from the previous
  runtime-int8 `15.751897` tok/s row.
- Real E2B int8 `METAL`, `beam=0`, `--max-new-tokens 1000
  --decode-warmup-tokens 20` measured `10.863932` tok/s with
  `rollout_jit_count=999` and `decode_fallback=false`, up from `10.721864`.
- This is a small accepted win, not a step change toward `50` tok/s. Next
  target: reduce rowwise-int8 scale/other kernel overhead or revisit the
  attention/norm kernel count. Do not move conditional token embedding outside
  replay; that was already measured as a real-row regression.

## 2026-04-23 - Post-Int8 Decode Profile And Rejected Pre-Embed

- Refreshed the real E2B int8 `METAL` post-window decode TinyJit profile after
  runtime int8 matmul. The synthetic context-700 profile measured `96.651` ms
  across `4807` kernels: `other` `31.712` ms, MLP `22.190` ms, norm
  `16.797` ms, attention `15.262` ms, logits/argmax `3.727` ms, and
  per-layer embedding `3.016` ms.
- The profile artifact is
  `benchmarks/gemma4-metal-postwindow-jit-profile-int8matmul-current.json`
  with CSV detail in
  `benchmarks/gemma4-metal-postwindow-jit-profile-int8matmul-current.csv`.
- Tried the obvious conditional-path parity change from the causal generator:
  move token embedding, per-layer inputs, and position id construction outside
  the conditional decode JIT. The synthetic profile improved to `89.465` ms,
  but the real E2B int8 `METAL`, `beam=0`,
  `--max-new-tokens 200 --decode-warmup-tokens 20` row regressed to
  `11.272605` tok/s with `rollout_jit_count=199` and
  `decode_fallback=false`.
- Rejected and reverted that source change. The current accepted real measured
  floor remains the runtime-int8 `1000/20` row: `10.721864` tok/s with
  `rollout_jit_count=999` and `decode_fallback=false`.
- Next target: keep work inside replay and reduce RowwiseInt8Linear MLP kernel
  volume, likely by fusing rowwise-int8 gate/up or folding the row-scale path.

## 2026-04-23 - Runtime Int8 Matmul

- Fixed rowwise int8 quantization for real Gemma 4 bfloat16 checkpoints. The
  old dtype predicate checked the tinygrad dtype name and missed bf16 because
  tinygrad reports it as `__bf16`; it produced an `int8` manifest with zero
  quantized tensors.
- Reworked `scripts/quantize_gemma4_matrix.py` to stream safetensors directly
  for checkpoint quantization. This avoids the tinygrad disk-bf16
  `tensor.numpy()` rangeify assertion and does not hold both the 9.5 GB source
  and 4.8 GB output in memory.
- Added `RowwiseInt8Linear` and a runtime quantized load path. Quantized
  `nn.Linear` weights stay int8 on the target device with one row scale vector;
  non-linear quantized tensors still dequantize to their manifest dtype. Use
  `load_pretrained(..., runtime_quantization=False)` for trainable dequantized
  loads.
- Regenerated the local E2B int8 checkpoint under ignored
  `/Users/ericfode/Downloads/tinygrad-gemma/checkpoints/gemma-4-E2B-int8`.
  It now has `582` int8 tensors, `582` scale tensors, a `134K`
  `quantization.json`, and a `4.8G` `model.safetensors`.
- Real load verification on 2026-04-23:
  `load_pretrained(..., device="METAL", verbose=True)` installed `525`
  runtime int8 linear weights; the first language-model MLP gate projection is
  `RowwiseInt8Linear` on `METAL`.
- Real E2B int8 `METAL`, `beam=0` benchmark rows after the patch:
  `--max-new-tokens 50 --decode-warmup-tokens 4` measured `20.063766`
  warmup-excluded tok/s with `rollout_jit_count=49` and
  `decode_fallback=false`; `--max-new-tokens 200 --decode-warmup-tokens 20`
  measured `15.751897` tok/s with `rollout_jit_count=199` and
  `decode_fallback=false`; `--max-new-tokens 1000 --decode-warmup-tokens 20`
  measured `10.721864` tok/s with `rollout_jit_count=999` and
  `decode_fallback=false`.
- Full repo gates passed after the patch: `35 passed, 1 skipped, 2 warnings in
  39.60s`; CLI help exit 0; `scripts/smoke_metal.py` reported
  `generated_tokens=4`, `rollout_jit_count=3`, and `decode_fallback=False`;
  `git diff --check` passed.
- Runtime int8 matmul is real and legal on Metal, but it is not the final
  throughput fix. The 1000-token row improved from `9.303522` to `10.721864`
  measured tok/s, still far below the `50` tok/s target. Next target: isolate
  the post-int8 long-context floor, likely full-attention context growth plus
  norm/attention kernel volume.

## 2026-04-23 - Fused MLP Gate-Up Decode Profile

- Added `scripts/profile_decode_jit.py`, a repo-owned synthetic post-window
  TinyJit profiler. It builds a zeroed KV cache at context length 700, forces
  `JIT=2` and `TRACEMETA=2`, then times the captured decode kernels by source
  category.
- The profile made the next bottleneck concrete: after the fused-MLP patch,
  synthetic post-window decode measured `110.302` ms total with MLP at
  `38.938` ms, attention at `26.236` ms, norm at `13.447` ms, and
  logits/argmax at `1.793` ms. Logits are not the current primary limiter.
- The same artifact records that the local `gemma-4-E2B-int8` checkpoint has an
  `int8` manifest with `0` quantized tensors and `2011` raw bfloat16 tensors.
  Treat the directory name as storage labeling, not runtime int8 proof.
- Added an inference-only fused gate/up path in `GemmaMLP`. When the gate and
  up weights are not trainable, decode caches a module-external concatenated
  gate/up weight and uses one wide linear before splitting gate and up. The
  fused cache is not part of `nn.state.get_state_dict`, and training still uses
  the original separate projections.
- Verification on 2026-04-23: full tests passed with `34 passed, 1 skipped,
  2 warnings in 37.79s`; CLI help exited 0; `scripts/smoke_metal.py`
  reported `generated_tokens=4`, `rollout_jit_count=3`, and
  `decode_fallback=False`; real E2B int8 `METAL`, `beam=0`,
  `max_new_tokens=700`, `decode_warmup_tokens=520` measured `11.891932`
  post-window tok/s with `rollout_jit_count=699` and
  `decode_fallback=false`; real E2B int8 `METAL`, `beam=0`,
  `max_new_tokens=1000`, `decode_warmup_tokens=20` measured `9.303522`
  tok/s with `rollout_jit_count=999` and `decode_fallback=false`.
- This improves the prior current 1000-token row from `8.300076` to
  `9.303522` measured tok/s, still far below the `50` tok/s target.
- Next target: reduce the remaining post-window floor around MLP/norm/attention
  kernel volume or introduce real runtime int8 matmul. Another logits-only
  optimization is unlikely to move the target materially.

## 2026-04-23 - Conditional Sliding Decode JIT Switch

- Diagnosed the post-window slowdown to a path split: `GemmaForCausalLM`
  already had a second Metal TinyJit for symbolic sliding-window decode, but
  the `GemmaForConditionalGeneration` path used by the official E2B int8
  checkpoint kept one symbolic decode JIT and never enabled
  `cache.decode_sliding_window`.
- Ported the Metal-only two-JIT sliding decode switch into
  `tinygrad_gemma/multimodal.py`. Conditional generation now uses the normal
  symbolic decode JIT before `sliding_window - 1`, then a bounded
  `gemma_start_pos_window` JIT with `decode_sliding_window=True` after the
  window.
- Added a unit test for the conditional Metal-only sliding decode start helper.
- Verification on 2026-04-23: focused tests passed; `scripts/smoke_metal.py`
  reported `generated_tokens=4`, `rollout_jit_count=3`, and
  `decode_fallback=False`; real E2B int8 `METAL`, `beam=0`,
  `max_new_tokens=700`, `decode_warmup_tokens=520` measured `10.295446`
  post-window tok/s with `rollout_jit_count=699` and `decode_fallback=false`;
  real E2B int8 `METAL`, `beam=0`, `max_new_tokens=1000`,
  `decode_warmup_tokens=20` measured `8.300076` tok/s with
  `rollout_jit_count=999` and `decode_fallback=false`.
- Full repo gates passed after the patch: `34 passed, 1 skipped, 2 warnings in
  38.65s`, CLI help exit 0, and the strengthened Metal smoke gate exit 0.
- This improves the prior current 1000-token row from `6.245549` measured tok/s
  to `8.300076` measured tok/s with the same output hash. The remaining gap is
  the second JIT capture around the window plus the full-attention layers that
  continue growing with context.
- Additional diagnostics on the same post-window slice: forcing the 7
  full-attention layers to use the local window measured `12.496987` tok/s, and
  a sync-free tensor loop measured `11.052106` tok/s. The full-attention layers
  and per-token CPU synchronization are real but insufficient to explain the
  remaining gap to `50` tok/s by themselves.
- Next target: produce a post-window kernel-level profile that separates
  attention/full-layer work, MLP work, and the tied logits/argmax path before
  making a more invasive optimization.

## 2026-04-23 - Long Metal Decode Floor Refresh

- Ran the queued real long-floor benchmark on the current grouped-query decode
  path and appended it to `benchmarks/gemma4-metal-1000-current.csv` instead of
  overwriting the existing interrupted beam=4 artifact.
- Verified E2B int8 `METAL`, `beam=0`, `max_new_tokens=1000`,
  `decode_warmup_tokens=20`: `status=ok`, `generated_tokens=1000`,
  `190.223269` seconds total, `5.256980` end-to-end tok/s,
  `156.911756` measured decode seconds, `6.245549` measured tok/s,
  `rollout_jit_count=999`, and `decode_fallback=false`.
- Progress sidecar evidence shows the long-context floor decays with generated
  length: `12.181298` measured tok/s at 100 generated tokens, `8.240744` at
  600, and `6.245961` at 1000.
- The existing `beam=4` row in the same artifact remains an interrupted run:
  `986` generated tokens, `438.910876` seconds, `status=interrupted`,
  `decode_fallback=false`. It is not the beam=0 acceptance row.
- Full repo gates passed after recording the artifact: `33 passed, 1 skipped,
  2 warnings in 37.08s`, CLI help exit 0, and the strengthened Metal smoke gate
  exit 0.
- Conclusion: the older 22 tok/s 1000-token note is stale for current code. The
  next code increment should target the long-context floor after separating
  sliding-layer cache-write/graph cost from the 7 full-attention layers.

## 2026-04-23 - Grouped-Query Decode Without KV Repeat

- Replaced the text attention decode math in `tinygrad_gemma/model.py` so
  grouped-query attention reshapes `q` into `(batch, kv_heads, groups, tokens,
  head_dim)` and broadcasts K/V with an extra group axis instead of physically
  expanding K/V through `repeat_kv()`.
- Left the shared `repeat_kv()` helper in place because multimodal vision/audio
  attention still imports and uses it outside the measured E2B text decode path.
- Verification on 2026-04-23: focused Gemma 4 forward/cache tests passed;
  `scripts/smoke_metal.py` reported `generated_tokens=4`,
  `rollout_jit_count=3`, and `decode_fallback=False`; real E2B int8 `METAL`,
  `beam=0`, `max_new_tokens=200`, `decode_warmup_tokens=20` measured
  `10.659780` warmup-excluded tok/s with `rollout_jit_count=199` and
  `decode_fallback=false`; full repo gates passed with `33 passed, 1 skipped,
  2 warnings in 43.23s`, CLI help exit 0, and the strengthened Metal smoke gate
  exit 0.
- Comparison row on the same target before this change measured `8.918411`
  warmup-excluded tok/s. The 50-token row was effectively neutral, so this is a
  modest longer-decode improvement, not the final throughput fix.
- Next target: refresh the 1000-token Metal row and then attack the long-context
  floor, likely around sliding-window cache behavior after the 512-token window.

## 2026-04-23 - Metal Decode JIT Replay Restored

- Read the tinygrad scheduler/JIT/cache-write path and compared this repo's
  symbolic cache mutation with tinygrad's own LLM examples in
  `.venv/lib/python3.11/site-packages/tinygrad/apps/llm.py` and
  `/Users/ericfode/src/.tinygrad_research/extra/models/llama.py`.
- Conclusion: the observed `RuntimeError('input to kernel must be AFTER or
  BUFFER, not Ops.INDEX')` was not proven to be a tinygrad bug. The repo was
  constructing symbolic cache writes with raw `UOp.after(...store...)`, while
  tinygrad's supported model pattern uses slice `.assign(...).realize()` inside
  the JIT path.
- Replaced the symbolic preallocated KV-cache update in
  `tinygrad_gemma/model.py` with the public slice-assign idiom. The integer
  prefill path already used this form.
- Strengthened `scripts/smoke_metal.py` so the Metal smoke gate now fails if
  decode generation falls back to eager mode or never runs a rollout TinyJit.
- Verification on 2026-04-23: focused cache/generate tests passed;
  `scripts/smoke_metal.py` reported `generated_tokens=4`,
  `rollout_jit_count=3`, and `decode_fallback=False`; real E2B int8 `METAL`,
  `beam=0`, `max_new_tokens=4` completed with `decode_fallback=false`,
  `rollout_jit_count=3`, `31.753536` seconds, and `0.125970` tok/s; real E2B
  int8 `METAL`, `beam=0`, `max_new_tokens=50`, `decode_warmup_tokens=4`
  measured `14.284291` warmup-excluded tok/s with `rollout_jit_count=49` and
  `decode_fallback=false`; full repo gates passed with `33 passed, 1 skipped,
  2 warnings in 39.12s`, CLI help exit 0, and the strengthened Metal smoke gate
  exit 0.
- Next target: improve the warmup-excluded Metal decode floor now that JIT
  replay is legal again. The most likely repo-local targets are sliding-window
  cache work and avoiding physical grouped-query KV repetition.

## 2026-04-23 - Metal Decode JIT Legality Fallback

- Found that the current tinygrad scheduler rejects the Metal decode JIT cache
  graph with `RuntimeError('input to kernel must be AFTER or BUFFER, not
  Ops.INDEX')`.
- Added explicit decode fallback tracking for both text and conditional Gemma 4
  generation paths. If the symbolic decode JIT raises that scheduler error,
  generation switches to eager token-id decode for the rest of the run instead
  of failing after the first token.
- Kept non-Metal generation on eager decode; the reusable decode JIT is now
  treated as a Metal-only optimization path.
- Extended `scripts/smoke_metal.py` so the cheap Metal gate verifies generation,
  prints `generated_tokens`, `rollout_jit_count`, and `decode_fallback`, and no
  longer stops at a forward/cache pass.
- Added `decode_fallback` to `scripts/benchmark_gemma4_matrix.py` rows so
  throughput artifacts do not hide whether a row used JIT replay.
- Added focused tests for non-Metal generate fallback behavior, preallocated
  sliding-cache correctness after the sliding window, and eliding the mask for
  already-cropped single-token sliding attention.
- Verification on 2026-04-23: focused tests passed; `scripts/smoke_metal.py`
  reported `generated_tokens=4`, `rollout_jit_count=0`, and
  `decode_fallback=True`; real E2B int8 `METAL`, `beam=0`,
  `max_new_tokens=4` completed with `status=ok`, `decode_fallback=true`,
  `rollout_jit_count=0`, `59.696246` seconds, and `0.067006` tok/s.
- This is not a throughput win. It converts a hidden Metal decode failure into a
  complete fallback path and names the real next target: restore legal TinyJit
  replay for symbolic cache stores.

## 2026-04-23 - Codex Local Environment Bootstrap

- Added root `AGENTS.md` so repo behavior is explicit for Codex sessions.
- Added repo-loop state, a bootstrap plan, procedure-candidate tracking, and a
  Codex environment action file.
- Copied Codex workflow skills/scripts into `.codex/skills/` and
  `.codex/scripts/` while preserving `.codex/skills/george-hotz-reviewer`.
- Created `.venv` with Python 3.11 and installed
  `.[dev,tokenizer,multimodal]`.
- Replaced removed tinygrad `Context(DEV=...)` usage with
  `temporary_default_device(...)` in loader/tests.
- Added `scripts/smoke_metal.py` as the cheap Metal gate. It requires tinygrad
  METAL, saves a tiny checkpoint, reloads it with
  `load_pretrained(..., device="METAL")`, runs a tiny Gemma forward/cache pass,
  realizes logits, and verifies the loaded model, logits, and cache tensors are
  on `METAL`.
- Cheap gates for this environment are `.venv/bin/python -m pytest -q`,
  `.venv/bin/tinygrad-gemma --help`, and
  `.venv/bin/python scripts/smoke_metal.py`.
- Verification on 2026-04-23: `pytest` reported 25 passed, 1 skipped, 2
  warnings; CLI help exited 0; `scripts/smoke_metal.py` reported
  `default_device=METAL`, `loaded_model_device=METAL`,
  `logits_device=METAL`, `logits_shape=(1, 3, 48)`, and
  `cache0_key_device=METAL`.
- Real-checkpoint Metal coverage remains outside this bootstrap until local
  checkpoints are present and a command loads them on `METAL`.
