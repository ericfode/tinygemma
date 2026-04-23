# Codex Automation Prompt Template

This is the direct Codex-side port of the Hermes cron prompt pattern used on this machine.

```text
Continue autonomous repo development in <repo-root>.

Start by running:
- <recovery-command>

Then read:
- <loop-state-file>
- <workflow-doc>
- <current-plan-reference>

Execute exactly ONE next planned increment per run.

Requirements:
- follow the repo workflow: Research -> Cohere -> Plan -> implement -> review -> loop to fix -> repeat
- use as many deterministic gates as possible
- run at minimum:
  - <gate-command-1>
  - <gate-command-2>
- if the increment is accepted:
  - create or update the acceptance note
  - update <loop-state-file> to mark the increment accepted
  - create the next increment plan file and set it as the next target
  - commit all changes with a clear commit message
  - report the commit hash, tests passed count, and new next target
- if you hit a real blocker or ambiguity that cannot be resolved from the repo, do not guess; report only the blocker clearly
- do not create, modify, or schedule automations from inside the autonomous run

Stay within the existing repo conventions and keep changes narrow, explicit, and import-safe.
```
