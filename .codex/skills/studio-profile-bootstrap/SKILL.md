---
name: studio-profile-bootstrap
description: Bootstrap a repo so Codex can operate with durable repo-local harness state rather than as a generic chat. Use when setting up a new project, converting a repo to durable Codex workflows, or creating the baseline files that later automation and repo loops depend on.
---

# Repo Harness Bootstrap

`studio-profile-bootstrap` is kept as the compatibility-stable skill id, but its job is
to create a general repo harness rather than a studio-specific setup.

Read [references/bootstrap-layout.md](references/bootstrap-layout.md) first.

Use `scripts/bootstrap_repo_files.py` when the user wants a real starting layout instead of
another prose plan.

## What To Create

Prefer a baseline like:

- `AGENTS.md`
- `configs/repo-loop-state.json`
- `docs/plans/`
- `state/evolution-log.md`
- `state/procedure-candidates.json`

Then install the relevant plugins from `is-codex-better` into the target repo.

## Operating Model

The repo should answer these questions from files:

- what is this project trying to do?
- what is the next increment?
- what workflows should Codex follow here?
- where do repeated procedures get promoted into reusable skills?

## Repo-Scoped Discipline

- derive repo guidance from the repo itself
- keep project-specific heuristics in repo files
- avoid leaking one project's lore into another

## When To Use Home Install Too

Install into the home marketplace when the user wants the same plugins available across
multiple repos from one source checkout.
