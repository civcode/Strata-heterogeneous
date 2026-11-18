# Hardening: withdrawn

**2026-11-18:** the engine hardening from `civcode/strata-dev-private` was removed from `feature/mixed-native` at the
Human User's request. The 17 November host crash that prompted it is now attributed to a launch config that mixed
components from different builds (`.ai/PROJECT.md`, rule 7), not to vanilla 0.2.15.

How it was removed: `feature/mixed-native` was rebuilt from `3f65b5de`, which is vanilla 0.2.15 plus the original mixed
commit, before the port. The mixed commits since then were replayed on top:

| Hardened line | Rebuilt line |
| --- | --- |
| `584e8e62` | `cdb6c881` (already in the base) |
| `33e5782c` | `66be0efb` |
| `0bb4225b` | `2cc03cb3` |
| `4455e9a2` | `c628d6d8` |
| `ec8bb9c6` | `93f8d422` (front-client part only) |
| `9655e74d` | `acc49e17` |
| `4785ba1b` | `4dc6f3ce` |

Two lines of `ec8bb9c6` (the hardening's primary weight arena record) and the reworded `ok` expression next to them
were dropped. The mixed engine changes are otherwise the same line for line. The session buffer allocation takes
vanilla's form.

The hardened line, with the port table, the reports 0067 to 0069 measured on it and this file's earlier version, is
on branch `archive/hardened-0069` (`b099f46f`). Reports 0067 to 0069 were measured with the hardening in. Handoff 0070
re-checks the rebuilt line.
