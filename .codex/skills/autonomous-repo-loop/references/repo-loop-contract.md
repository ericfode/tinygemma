# Repo Loop Contract

The loop is intentionally simple:

1. recover current state
2. choose one narrow increment
3. implement it
4. review it
5. update state
6. stop or queue the next run

Useful loop-state fields:

- `current_checkpoint`
- `accepted_increments`
- `next_target`
- `next_target_plan`
- `updated_at`

The exact schema can vary by repo. The important property is that the next session can
answer "what do I do next?" from the repo, not from chat memory.
