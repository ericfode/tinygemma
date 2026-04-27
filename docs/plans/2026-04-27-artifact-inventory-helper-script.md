# Artifact Inventory Helper Script

## Objective

Resolve `artifact-inventory-helper-script-072`: turn the one-off untracked artifact inventory logic into a reusable, read-only repo helper.

## Change

Added `scripts/inventory_untracked_artifacts.py`, which:

- reads untracked paths from `git status --porcelain -z`;
- reads tracked reference text from `git ls-files` over `.md`, `.json`, `.toml`, `.txt`, and `.rst` files;
- classifies generated artifacts into benchmark result, progress log, decode profiler, paired-helper smoke, dependency lockfile, or other buckets;
- emits a Markdown inventory with counts, references, sizes, and suggested dispositions;
- never deletes, ignores, stages, or mutates untracked artifacts.

## Test

Added `tests/test_artifact_inventory.py` using a temporary git repository. The test stages tracked reference text, creates untracked artifacts, runs the helper, verifies reference detection and category output, and then verifies the untracked artifacts remain untracked.

## Verification

- RED: missing script failed the new test.
- GREEN: helper implementation passed the focused test.
- Broader gates recorded in the repo-loop state for this increment.
