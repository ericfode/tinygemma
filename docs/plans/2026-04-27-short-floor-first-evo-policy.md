# Short-Floor-First Evo Policy

## Objective

Resolve `short-floor-first-evo-policy-089`: apply the user's updated optimization policy to the local evo campaign.

## User policy

- Optimize only the default short floor until it appears saturated.
- Defer the long `1000/20` floor until a short-floor candidate is a serious contender or the short frontier appears maxed.
- Weigh graph-size/source-count reductions more heavily when choosing candidates.

## Changes applied

- Removed inherited long gate `e2b_int8_metal_hash1000_current_floor` from `exp_0000` with:

```bash
evo gate remove exp_0000 --name e2b_int8_metal_hash1000_current_floor
```

- Verified effective gates on `exp_0005` are now:
  - `metal_smoke`
  - `cli_help`
  - `e2b_int8_metal_hash16`
- Updated `.evo/project.md` with the short-floor-first policy and graph-size weighting note.

## Scan evidence

A read-only evo scan over `exp_0006` through `exp_0012` found:

- all recent children failed to beat `exp_0005 = 28.6153` on the default short score;
- repeated micro-elisions/view tweaks regressed;
- K/V projection reuse/fusion variants regressed;
- hash16 throughput should remain a safety gate, not the short-floor score;
- graph/source-count reductions are the next candidate-selection bias.

## Verification

```bash
evo gate list exp_0005
date +%Y-%m-%dT%H:%M:%S%z
```

Observed effective gates omitted the long `1000/20` floor.

## Next increment

Run one short-floor-only evo child from `exp_0005`, choosing a candidate whose value proposition is graph/source-count reduction rather than another micro-elision.
