---
name: autonomous-repo-loop
description: "Run a deterministic repo work loop: recover state, cohere one narrow increment, implement it, review it, and close with explicit gates. Use when the user wants autonomous continuation, resumable repo work, or one-increment-per-run discipline."
---

# Autonomous Repo Loop

Use this skill when the user wants the repo to keep moving in auditable increments instead
of drifting across vague checkpoints.

Read [references/repo-loop-contract.md](references/repo-loop-contract.md) before using the
loop on a new repo.

For automation wording, start from
[references/codex-automation-prompt-template.md](references/codex-automation-prompt-template.md)
or render one with `../../scripts/render_autoloop_prompt.py`.

## Trigger Conditions

- The user says some version of "keep going", "continue until blocked", or "work one
  increment at a time".
- The repo already has state files such as `configs/repo-loop-state.json`,
  `state/`, `docs/plans/`, or similar handoff artifacts.
- You need a repeatable pattern for automation or recurring Codex runs.

## Operating Rule

Run exactly one narrow increment per loop.

Do not quietly sprawl into a second increment because the first one went well.

## Workflow

1. Recover
   - Inspect `git status --short --branch`
   - Inspect recent commits
   - Read repo plans, handoff docs, and machine-readable state if present
   - If there is no loop-state file yet, initialize one with
     `scripts/init_loop_state.py`
2. Cohere
   - Compress the next slice into one testable increment
   - Name explicit success criteria, non-goals, and risks
3. Implement
   - Make the smallest coherent change that satisfies the increment
   - Run focused checks as you go
4. Review
   - Review against the plan, not just against the diff
   - Prefer a second pass for bugs, regressions, and missing tests
5. Close
   - Run the repo gates
   - Update loop-state and acceptance artifacts
   - Name the next increment explicitly

For unattended runs, use `../../scripts/codex_autoloop.py`. That gives Codex a concrete
execution loop instead of relying only on prompt discipline.

## Stop Contract

If the surrounding workflow expects autonomous continuation, only stop with one of these
markers in the response:

- `WHOLE PROJECT COMPLETE`
- `ACTUAL QUESTION:`
- `BLOCKED:`
- `UNRESOLVED BLOCKER:`

This keeps stopping conditions explicit instead of polite.

## Gates

Prefer repo-declared gates from `README`, `AGENTS.md`, scripts, or plan docs.

When the repo has no stronger convention yet, the minimum sober baseline is:

- status check
- direct tests for the changed surface
- one broader repo verification command

## Pitfalls

- Do not skip the cohere step.
- Do not treat "continue" as permission to change the repo without a narrow slice.
- Do not close the loop without updating the handoff artifact.
- Do not claim completion because you reached a convenient pause.
