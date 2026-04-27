# Post-Frontier Handoff

## Current evo frontier

- Best experiment: `exp_0005`
- Score: `28.6153`
- Hypothesis: `calibration: current baseline rerun`
- Evo status at handoff: `metric=max epoch=1 experiments=13 committed=2 evaluated=0 discarded=11 failed=0 active=0 best=28.6153`

## Recent rejected runtime children

- `exp_0011`: graphable rowwise-int8 QKV fusion. Regressed to `27.7816` and failed the long floor.
- `exp_0012`: decode-only attention-output view elision. Regressed to `28.4939` and failed the long floor.

## Raw runner decision

The raw Metal rowwise-int8 runner line is pruned from live model/profiler surfaces. It remains only as prototype/canary code with explicit capture guards. The first-class graphability review rejected it because tinygrad MetalGraph currently expects compiled program runners, not arbitrary host-call runners.

## Profiling improvements landed on main

- Configurable cache-write phase targets: presets plus exact selectors such as `local-layer12` and `shared-source-layer13`.
- Explicit phase target scoping for source attribution.
- Phase/parent overlap reporting, proving whether phase labels correspond to their claimed source category.
- Paired baseline/candidate decode helper for same-session comparison and score provenance.
- README usage for the paired helper.

## Corrected phase evidence

- `shared-source-layer13`: about `14.54 ms`, `888` source items, exact parent overlap; this is the large remaining cache-write bucket.
- `local-layer12`: about `3.25 ms`, `256` source items, exact parent overlap; too small to justify a runtime probe by itself.

## Current decision

Runtime frontier is saturated for the explored K/V projection and cache-write surfaces. The next useful work is benchmark/infrastructure gating, not another ungrounded `model.py` child.
