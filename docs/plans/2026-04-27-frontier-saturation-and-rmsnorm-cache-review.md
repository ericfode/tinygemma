# Frontier Saturation and RMSNorm Cache Review

## Objective

Resolve `frontier-saturation-and-rmsnorm-cache-review-060`: decide whether another runtime evo child is justified after the recent QKV fusion and attention-output view-elision rejections, with special attention to the previously named RMSNorm inference scale-cache surface.

## Findings

The current evo frontier remains:

- Best: `exp_0005`
- Score: `28.6153`
- Recent rejected children: `exp_0011` QKV fusion, `exp_0012` attention-output view elision
- Run state: 13 experiments total, 2 committed, 11 discarded

RMSNorm inference scale caching is not a fresh candidate. It was already tested as `exp_0009`:

- Patch cached immutable inference scale tensors for `weight.float()` / `(1.0 + weight.float())`.
- Default score regressed to `26.1647` versus parent `28.6153`.
- Long 1000/20 gate measured about `17.1434`, below the current `18.0` floor and below `exp_0005`'s fresh long row around `18.090845`.
- The experiment was discarded.

Corrected phase attribution does not reopen this surface:

- `shared-source-layer13` hot phase is K/V projection: `882` sources / ~`14.422` ms.
- RMSNorm/RoPE within the exact target is only `4` sources / ~`0.054` ms.
- Neighbor exact targets show similarly tiny RMSNorm/RoPE buckets.

## Saturated Runtime Families

The following runtime families have already been rejected or exhausted:

- K/V projection variants: separate K/V, lane reshape split, rank flattening, QKV fusion, prepacked/repeated K/V, active shared-KV view reuse.
- Cache/write/layout variants: assign-realize decoupling, flat-slot layout, windowed shared-source allocation, deferred readback/parity/guard variants.
- Attention/logit variants: SDPA, greedy softcap elision, attention-output view elision.
- RMSNorm inference scale cache.
- Raw Metal custom Runner integration and local-size tuning as a model hot path.
- `JIT_BATCH_SIZE=0` as a default runtime knob.

A weak remaining packed-cache layout idea `(B,H,L,2,D)` is not recommended: prior layout work was negative, and corrected phase attribution says store/RHS layout is not the hot problem.

## Decision

Do not spend another evo runtime child until benchmark/infrastructure improves. The immediate value is not a new `model.py` patch; it is reducing measurement ambiguity and preventing repeated saturated hypotheses.

## Next Direction

Build a paired same-session baseline-vs-candidate benchmark helper so future candidates carry:

- same-session control score,
- candidate score,
- repeated rows or at least paired metadata,
- stable hash checks,
- gate/floor provenance,
- and an explicit saturation-ledger link.

This addresses the recurring problem that stale long-floor references and noisy single rows can mislead optimization decisions.
