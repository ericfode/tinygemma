# Recovery Checklist

Minimum commands worth considering:

- `git status --short --branch`
- `git log --oneline --decorate -n 10`
- `git diff --stat`
- `git ls-files --others --exclude-standard`
- `git status --short --ignored`
- repo-specific tests or gate scripts

Treat handoff files, plan docs, and loop-state files as first-class evidence.
