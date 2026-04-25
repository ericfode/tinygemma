# Evolution Log

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
