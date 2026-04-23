---
name: hermes-profile-port
description: Port a Hermes profile into Codex-friendly repo artifacts. Use when a user already has a Hermes profile with PROFILE.md and SOUL.md and wants the same repo-scoped operating context in a Codex repo.
---

# Hermes Profile Port

Use this when the important source of truth already exists in a Hermes profile and you want
Codex to inherit it instead of re-inventing it.

Read [references/profile-mapping.md](references/profile-mapping.md) first.

Use `scripts/import_hermes_profile.py` for the actual import work.

## Inputs

- a Hermes profile directory such as `~/.hermes/profiles/game-studio`
- optionally a target repo that should receive the imported artifacts

## What To Preserve

- project identity
- scope boundary
- repo root
- attached repo-local skills path
- the persona or SOUL text that changes engineering behavior

## Safe Default

Do not overwrite an existing `AGENTS.md` unless the user explicitly wants that.

Write imported artifacts under `docs/hermes-import/<profile-name>/` and generate a Codex-ready
draft there first.

## Good Outcome

The repo ends up with a sober Codex-native draft of the Hermes profile, not a raw dump with no
translation.
