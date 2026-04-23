---
name: specialist-fanout
description: Split a repo task into parallel specialist subtasks with clear ownership and minimal overlap. Use when the user explicitly wants delegation, subagents, or parallel expert work, or when standing repo instructions already establish that delegation is normal.
---

# Specialist Fanout

This skill is about decomposition quality, not just "use more agents."

Read [references/task-brief-template.md](references/task-brief-template.md) for a compact
briefing pattern.

## Before Delegating

1. Decide the immediate blocking task you should keep local.
2. Identify sidecar tasks that can run in parallel without blocking that next step.
3. Split work by file ownership or by concrete question, not by vague role labels.

## Good Fanout

- bounded task
- clear output
- disjoint write scope when code changes are involved
- no duplicated investigation

Examples:

- one worker owns `plugins/honcho-codex/**`
- one worker owns `plugins/codex-studio/**`
- the main agent keeps root install/docs work local

## Bad Fanout

- handing away the next blocking task and then waiting
- asking multiple agents the same question
- letting two workers edit the same files

## Worker Brief Rules

- say exactly which files or module boundaries they own
- remind them they are not alone in the codebase
- tell them not to revert unrelated changes
- ask for changed file paths in the final answer

## Integration

While workers run, do real non-overlapping work locally.

When a worker returns:

- review the result quickly
- integrate or refine it
- do not redo the same task from scratch
