# Layer 13 reduce_r Source Attribution

## Objective

Map the layer-13 `reduce_r`-heavy source rows back to producer operations before another runtime patch. The hard metric remains repeated real E2B int8 `METAL` token throughput, not profile aesthetics.

## Exact artifact mapping

- Target parent category: `attention_packed_cache_write__role_shared_source__layer_13__type_sliding_attention`.
- Total source rows in structural artifact: `2994`.
- Target parent source rows: `885`.
- Target parent `r_*` / `reduce_r` rows: `680`.
- Target parent non-reduce rows: `205`.
- `category` field alone maps only `216` reduce rows; use `source_category_counts` to map all `680` reduce rows.
- Child phase rows recovered in the phase-propagation profile: `872` classified, `13` unclassified, `37` conflicted.
- Child phase split: `kv_projection=863, store=9`.

Graph-batch split:

| graph batch | source range | target rows | phase counts | phase-unclassified rows |
| --- | ---: | ---: | --- | ---: |
| `5 <batched 1024>` | `992-2015` | `273` | `kv_projection=258, store=2` | `13` |
| `6 <batched 978>` | `2016-2993` | `612` | `kv_projection=605, store=7` | `0` |

## Top reduce display groups

| display name | rows | elapsed ms | example ordinals |
| --- | ---: | ---: | --- |
| `r_2048_32_4_384_4` | `1` | `3.405500` | `2989` |
| `r_256_32_3_384_4` | `20` | `2.085625` | `2596, 2616, 2636, 2656, 2677, 2697, 2717, 2737` |
| `r_128_32_3_384_4` | `28` | `1.676667` | `1727, 1747, 1767, 1787, 1808, 1828, 1848, 1868` |
| `r_16_96` | `242` | `1.478126` | `1712, 1724, 1726, 1729, 1733, 1735, 1744, 1746` |
| `r_8_2_16_4_4_(gemma_start_pos_window+1)` | `9` | `1.105750` | `1803, 1904, 2369, 2470, 2571, 2672, 2773, 2874` |
| `r_2048_16_96` | `39` | `0.963041` | `1714, 1736, 1756, 1776, 1817, 1837, 1857, 1877` |
| `r_1536_16_384` | `13` | `0.707792` | `1728, 1748, 1768, 1788, 1809, 1829, 1849, 1869` |
| `r_2_4_(gemma_start_pos_window+1)n1` | `9` | `0.666000` | `1801, 1902, 2367, 2468, 2569, 2670, 2771, 2872` |
| `r_32_32_4_384_4` | `9` | `0.644333` | `1796, 1897, 2362, 2463, 2564, 2665, 2766, 2867` |
| `r_2_4_(gemma_start_pos_window+1)` | `9` | `0.614667` | `1800, 1901, 2366, 2467, 2568, 2669, 2770, 2871` |

## Producer operation map

The parent realization scope is exact, and the phase-propagation profile now identifies the dominant child phase. The relevant source operations feeding that parent scope are:

- `kv_projection` — fused rowwise-int8 K/V projection or separate K/V linears (`tinygrad_gemma/model.py:509-520`, `tinygrad_gemma/model.py:512`, `tinygrad_gemma/model.py:516-517`, `tinygrad_gemma/model.py:561`).
  - Status: `dominant_child_phase_proven_by_phase_propagation_profile:863_source_rows`.
  - Reduction relevance: hidden_states.matmul(weight.transpose(), dtype='float') and linear projections lower to reduce kernels.
- `rmsnorm_rope` — K RMSNorm, K RoPE, and V RMSNorm after K/V projection (`tinygrad_gemma/model.py:91-105`, `tinygrad_gemma/model.py:327-333`, `tinygrad_gemma/model.py:562-564`).
  - Status: `not_observed_in_phase_propagation_profile`.
  - Reduction relevance: RMSNorm uses mean(-1, keepdim=True), which is a reduction; RoPE is mostly elementwise/slice/cat and is not expected to dominate reduce_r mass by itself.
- `rhs_pack` — pack K and V into the final cache lane before assignment (`tinygrad_gemma/model.py:298`, `tinygrad_gemma/model.py:301`, `scripts/profile_decode_jit.py:649-663`).
  - Status: `not_observed_in_phase_propagation_profile`.
  - Reduction relevance: Packing is squeeze/unsqueeze/cat/view work, so it should not be the primary origin of large reduce_r kernels unless fused with upstream producers.
- `store` — packed cache slice assignment and dependency-closed realize (`tinygrad_gemma/model.py:280-301`, `tinygrad_gemma/model.py:586-599`, `tinygrad_gemma/model.py:605-618`, `scripts/profile_decode_jit.py:653-667`).
  - Status: `minor_child_phase_proven_by_phase_propagation_profile:9_source_rows`.
  - Reduction relevance: The store realization itself is not a mathematical reduction, but assign(...).realize() pulls the lazy K/V producer graph into the cache-write parent scope.

## Recoverability conclusion

- Parent scope: exact: source_category_counts maps all 885 layer-13 shared-source rows, including 680 reduce_r rows, to the target category.
- Child phase: partly_recovered: 872/885 target parent rows carry child phase metadata (kv_projection=863, store=9); 13 remain unclassified and 37 rows report phase conflicts.
- Root cause: direct cache-write phase propagation through UOp creation/replace preserves most child phase tags without forced realization cutpoints; the remaining conflicts/unclassified rows mark places where lowered source items merge multiple phase-tagged paths or drop phase metadata.
- Next profiler target: Treat layer-13 shared-source K/V projection as the dominant reduce_r producer before writing runtime code; a candidate should reduce that projection/source mass or prove an equivalent fused path, not optimize RHS packing or the store tail.

No runtime patch is justified by this artifact yet. The profiler now points at layer-13 shared-source K/V projection as the dominant child phase, so the next runtime candidate must reduce that projection/source mass or prove an equivalent fused path before spending the hard `1000/20` throughput gate.
