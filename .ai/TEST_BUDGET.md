# Oberon test budget

User direction after handoff 0045: stop long-duration tests for now. The
256K capacity/lifecycle and performance evidence is already recorded in reports
0044 and 0045. Do not repeat the long-context ladder, near-capacity 256K
requests, or full performance matrices unless the user explicitly requests
them.

For subsequent handoffs:
- Prefer focused CPU tests, existing proven binaries when applicable, and
  short representative GPU smoke/regression cases. Continue local builds
  with 28 jobs and the existing vanilla toolchain and venv.
- For routine GPU checks, use short prompts and bounded completions. Use
  a 4K-token prompt only when necessary to exercise prefill. Avoid repeated
  long requests. A 256K configured capacity can be used for a short HTTP
  smoke test without filling that context.
- Plan for a few minutes of GPU testing per batch (aim for five minutes);
  if more would be necessary, report the limitation and ask for explicit
  approval before starting an extended test. Do not start additional
  cases after the budget is exhausted; allow any in-flight operation to
  finish or cancel cooperatively at an existing safe checkpoint.
- Memory/residency, commit/cancel safety and normal shutdown requirements
  remain unchanged. Never force-kill a GPU process to meet a time budget.
- Keep historical numerical discrepancies low priority. Reopen only for
  concrete new defect evidence or explicit user direction.
- No GitHub Actions execution. Use the previously qualified data rather
  than re-measuring it for routine changes.

These are test scope limits, not a reduction of supported maximum context.
