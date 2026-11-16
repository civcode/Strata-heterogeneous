# AI-Assisted Rapid Prototyping Workflow

**Version:** 1.2  
**Purpose:** A practical operating protocol for a Human User, a Chat AI, and a local Coding Agent collaborating through a shared GitHub repository.

**Changes in 1.1:** The Coding Agent now runs in one of two performance tiers, chosen per handoff (Section 4, "Coding Agent tiers").

**Changes in 1.2:** Chat AI implements what it can adequately check (rule 1). New standing project file `.ai/PROJECT.md` and Chat AI status file `.ai/STATUS.md` (Section 3). Open handoffs can be cancelled only by the Human User (Section 5). Section 11 is now a general example split. Hardware test runtime is limited by project stage (Section 12).

## 1. Objective

Maximize **useful working code per unit of time, human effort, and model cost**. Favor precise implementation, autonomous debugging, and rapid iteration over ceremony, extensive documentation, and repetitive verification.

This protocol has three participants:

- **Human User:** Establishes objectives, makes consequential decisions, copies short messages between AI interfaces, and starts the Coding Agent with the model tier named in the handoff.
- **Chat AI:** Primary architect, **primary implementer**, debugger, and technical coordinator. Runs on a frontier model. Writes as much of the actual code as is practical with its available tools.
- **Coding Agent:** Autonomous developer operating in the local development environment. Performs work that requires, or is materially more efficient with, local tools, runtimes, hardware, or physical devices. It runs in one of two tiers:
  - **Tier 1** — a frontier model, for architecture and complex development that is more productive in the local environment than in the Chat AI environment.
  - **Tier 2** — a smaller, cheaper model, for building, running, testing, debugging, and benchmarking code that is mostly complete.

The shared **GitHub repository** is the source of truth for code, implementation state, handoffs, and reports. The Human User should not have to manually relay lengthy code, logs, or specifications.

## 2. Core Rules

1. **Chat AI implements what it can adequately check.** It must not default to being a planner or dispatcher. For work it can check in its own environment, it analyzes, edits code, debugs, runs available checks, commits, and pushes before considering delegation. For work it cannot adequately check, it chooses between writing unverified code for a Tier 2 agent to build and correct, and a Tier 1 handoff. Unverified code is always labelled as such in the handoff.
2. **Delegate by comparative advantage.** Use the Coding Agent only for work that meaningfully benefits from its local environment or would be substantially more efficient there. Do not delegate simply because work involves coding, several files, or challenging reasoning.
3. **One developer owns the turn.** Chat AI and Coding Agent may both modify code, but **never concurrently**. Only the active participant writes to the project. Ownership transfers via the Human User's short message relay.
4. **Handoffs are substantial.** A handoff should describe a coherent milestone or a batch of related tasks—not merely “build and test this function.” Accumulate reasonable hardware-dependent checks and integration work until a meaningful agent turn is warranted.
5. **The Coding Agent owns completion.** After receiving a handoff, it independently implements, builds, tests, troubleshoots, corrects, and retries until the **whole batch's completion criteria** are met, within the authority of its tier. It should not return partial results for ordinary recoverable difficulties.
6. **No mandatory handoff.** If Chat AI can complete a request itself, it should finish and report directly. Small changes can remain unverified on the local hardware until a later useful local-work batch; disclose that distinction accurately.
7. **Short messages; useful files.** Chat-to-agent and agent-to-chat messages are brief pointers. Detailed task requirements and material findings live in numbered Markdown files in the repository.
8. **Pragmatic validation.** Run targeted checks sufficient for useful prototyping. Do not introduce time-consuming cross-check cycles, large test suites, or approval gates by default. Tests on the agent's hardware are meaningful but short; long-duration hardware tests wait until the project reaches release-candidate state (Section 12). Do not imply checks occurred if they did not.
9. **Real repository state only.** Never claim a file, commit, push, test, or hardware result occurred unless it actually did. Disclose missing tool access or unverified hardware behavior.
10. **Use the cheapest tier that can do the work.** Every handoff names exactly one agent tier. Tier 2 is the default. Tier 1 is used only when the handoff states why the work is complex and why it cannot be done as productively by Chat AI.

## 3. Repository Convention

Use the existing project layout plus:

```text
repository/
├── ...                              # Project code
└── .ai/
    ├── PROJECT.md                   # Standing project rules; read at the start of every turn
    ├── STATUS.md                    # Current state; maintained by Chat AI
    ├── handoffs/
    │   ├── handoff-0001.md
    │   ├── handoff-0002.md
    │   └── ...
    └── reports/
        ├── report-0001.md
        ├── report-0002.md
        └── ...
```

### Project file: `.ai/PROJECT.md`

One short file holds the rules that apply to **every** turn, so handoffs do not repeat them and fresh sessions do not lose them.

- **Content:** what the project is in a few lines, the standing constraints (for example an implementation contract, the toolchain and environment to use, directories that are off limits), the project stage (early development, integration, or release candidate), and the default hardware test budget (Section 12).
- **Length:** about one page. It states rules, not history or design detail; link to other files for those.
- **Readers:** Chat AI and the Coding Agent, in both tiers, read it at the start of every turn before anything else.
- **Changes:** Chat AI edits it, and only on the Human User's direction or with the Human User's agreement. The Coding Agent does not change it; if a rule there blocks the task, it reports that.
- **Precedence:** a handoff may narrow these rules for one turn but may not override them. If a handoff and the project file conflict, the agent follows the project file and reports the conflict.

### Status file: `.ai/STATUS.md`

A new Chat AI session starts without context just as an agent turn does. Chat AI keeps one short status file so that any session can resume.

- **Content:** who holds the turn (Chat AI, or Coding Agent with the open handoff number), the current milestone, open decisions waiting for the Human User, and the next step. Ten to twenty lines.
- **Maintenance:** Chat AI updates it as the last step of each of its turns: before pushing a handoff, and when it reports a finished request. It replaces the content; history lives in Git and in the reports.
- **Readers:** Chat AI reads it first in every session. The Coding Agent does not need it and does not edit it.

### Numbering

- Each **Coding Agent delegation** receives the next monotonically increasing, zero-padded number: `0001`, `0002`, etc. Tier 1 and Tier 2 handoffs share this one sequence.
- Handoff `handoff-NNNN.md` corresponds to report `report-NNNN.md`.
- Determine the next number from the highest existing handoff on the agreed working branch.
- Preserve previous handoffs and reports; do not overwrite them for new iterations.
- **Chat AI-only work does not consume a handoff number.** Numbering tracks delegations, not user requests or commits.
- Both AIs may make multiple commits during their own turns. A turn is not a commit.

### Git discipline

- Use the agreed working branch unless there is a concrete reason to change it.
- Commit and push relevant changes before handing off ownership.
- A transfer prompt identifies **branch, exact handoff path, and commit ID**. The Coding Agent must find and use that exact handoff and starting revision, not guess which file is current.
- Do not force-push, delete unrelated work, or silently discard uncommitted changes.
- Work sequentially; avoid unnecessary branches, pull requests, rebases, and merge overhead.
- The next participant should verify that the specified commit is available on the identified branch before proceeding; if repository state differs unexpectedly, resolve it without destroying unrelated work or report a genuine blocker.

## 4. Responsibilities and Authority

### Human User

- Defines goals, constraints, and priorities.
- Makes decisions that cannot be reasonably inferred or require human judgment.
- Copies Chat AI's short agent prompt into the Coding Agent interface, **started with the model for the tier the prompt names**.
- Copies the Coding Agent's short completion message back into Chat AI.

The Human User does **not** routinely transport full reports, source code, detailed instructions, or logs.

### Chat AI — Primary architect **and implementer**

Within its available environment, Chat AI should:

1. Read `.ai/PROJECT.md` and `.ai/STATUS.md`, then inspect the current repository and relevant prior report.
2. Make architectural and implementation decisions.
3. **Write and modify as much working code as it can adequately check**—including drivers, modules, interfaces, algorithms, configuration, tests, utilities, refactoring, and debugging.
4. Run efficient checks available to it, without inventing validation results.
5. For work it cannot adequately check, decide between writing it unverified for a Tier 2 turn and delegating it as a Tier 1 handoff (see "Choosing the tier").
6. Commit and push its implementation when GitHub access allows.
7. Identify concrete remaining work that is more efficient on the Coding Agent's local machine.
8. **Batch related local tasks**, choose the agent tier, and create a numbered handoff only if delegation is worthwhile.
9. Update `.ai/STATUS.md` as the last step of its turn.
10. On return, read the matching report, inspect changes as needed, and resume implementation.

Chat AI should **not** delegate merely because a change is substantial or spans multiple files. Conversely, it should not force itself to work inefficiently when the local Coding Agent has a clear execution advantage.

Chat AI remains the project's architect in every case. A Tier 1 agent turn does not transfer that role.

### Coding Agent — Autonomous local developer

Within the handoff's objective and constraints, the Coding Agent should:

1. Synchronize the exact specified branch/revision, read `.ai/PROJECT.md`, then read the numbered handoff.
2. Inspect relevant project files and local environment.
3. Implement, integrate, build, flash, run, debug, and test as appropriate.
4. Make reasonable corrections and repeat local operations autonomously.
5. Continue until the **overall** completion criteria are met.
6. Write the **matching concise report**, commit and push its changes, and return a short completion message.

It should not ask Chat AI to solve ordinary implementation difficulties mid-turn or return partial progress merely after one failed attempt. It may report **Blocked** or **Partial** only when there is a genuine unresolved obstacle after reasonable effort, or it needs unavailable resources, authorization, or a consequential human decision.

### Coding Agent tiers

The tier is a property of the **handoff**, not of the project. Chat AI chooses it; the Human User starts the agent with the matching model.

| | Tier 1 | Tier 2 |
| --- | --- | --- |
| Model | Frontier | Smaller, cheaper |
| Kind of work | Architecture and complex development that depends on the local environment | Build, flash, run, test, debug, and benchmark code that is mostly complete |
| Typical signal | Correctness of each step can only be judged by a local build, run, or device, and there are many such steps | The design and most of the code exist; what remains is making it build and run and measuring it |
| Design authority | May make design and implementation decisions **inside the boundaries the handoff states** | May fix what is needed to meet the completion criteria **inside the files and interfaces the handoff names** |
| Handoff style | Goal, boundaries, constraints, completion criteria | More explicit: what to build, what to run, what result to expect |

**Choosing the tier**

- **Chat AI first.** If Chat AI can implement and adequately check the work in its own environment, it does so; any remaining local build-and-run work is Tier 2.
- **Work Chat AI cannot adequately check** goes one of two ways:
  - **Unverified code for Tier 2** when the design is settled and the likely errors are ordinary ones a build or a run will expose (compile and link errors, configuration, small integration defects). Chat AI writes the code, labels it unverified in the handoff, and the Tier 2 agent builds and corrects it.
  - **Tier 1 handoff** when the design itself depends on what the local build, run, or device shows, or when the amount of unverified code is large enough that correcting it would amount to redevelopment. The handoff states this reason in one or two sentences.
- **Tier 2** is the default for everything else that needs the local environment.
- **One tier per handoff.** Do not mix tiers. A Tier 1 turn naturally includes the build, run, and debug work for what it implements. If a large independent Tier 2 batch follows, give it its own handoff.

**Authority and escalation**

- A **Tier 1** agent works within the architecture Chat AI has set. It records any design decision or deviation that matters to later work in its report. A change to the handoff's stated boundaries is a consequential decision: it reports instead of proceeding.
- A **Tier 2** agent resolves ordinary build failures, integration bugs, configuration mistakes, and small code defects on its own. If meeting the completion criteria would require a design change, an interface change outside the named scope, or substantial new development, it stops that part, completes what it can, and reports **Partial** or **Blocked** with the line `Tier escalation recommended` and the reason. It does not attempt the redesign.
- On an escalation, Chat AI decides whether to do the work itself or issue a Tier 1 handoff. The agent does not change tier mid-turn.

## 5. Exclusive Turn Ownership

Development is sequential:

```text
Human objective
      ↓
CHAT AI TURN (exclusive repo write ownership)
  Design → implement → debug → available checks → commit/push
      │
      ├── No meaningful local work remains → report to Human; done
      │
      └── Meaningful local work remains
                 ↓
       Choose tier → create numbered handoff → commit/push
                 ↓
       Short agent prompt (names the tier) → Human copies,
       starts the agent with that tier's model
                 ↓
CODING AGENT TURN, Tier 1 or Tier 2 (exclusive repo write ownership)
  Read handoff → implement/integrate/build/flash/debug as needed
  → meet batch criteria → write report → commit/push
                 ↓
       Short completion message → Human copies
                 ↓
CHAT AI TURN resumes
  Read report → assess → implement further → delegate only if useful
```

**Ownership rule:** Chat AI stops modifying project files after transferring a handoff. Coding Agent stops modifying project files after its completion message. The Human User's relay of that completion message returns ownership to Chat AI. Do not run parallel code-modifying tasks against the same project, including two agent tiers at once.

**Who holds the turn:** The repository shows it. A handoff `handoff-NNNN.md` without a matching `report-NNNN.md` is **open**, and the Coding Agent holds the turn. `.ai/STATUS.md` states the same in its first line. Chat AI checks both at the start of a session and does not write to the project while a handoff is open.

**Cancelling a handoff:** Only the Human User can cancel or supersede an open handoff. Neither AI may decide that an open handoff is stale and proceed. When the Human User cancels one:

1. The Human User makes sure the Coding Agent has stopped, then tells Chat AI that handoff `NNNN` is cancelled.
2. Chat AI writes `report-NNNN.md` as a one-line record, for example: `Cancelled by the Human User on <date>; agent changes up to commit <id> kept` (or `no agent changes`).
3. Chat AI inspects any commits the agent pushed before the cancellation, updates `.ai/STATUS.md`, and resumes its turn.

A cancelled number is not reused. A replacement delegation takes the next number.

## 6. Standard Iteration

### Phase A — Chat AI work

1. Receive the Human User's goal or the Coding Agent's completion message.
2. Read `.ai/PROJECT.md` and `.ai/STATUS.md`; confirm no handoff is open.
3. Inspect code, relevant commits, and any matching prior report.
4. Implement the parts it can adequately check in its environment.
5. Perform proportionate host-side/static validation; record any remaining verification limits.
6. Commit and push work.
7. Decide whether a substantial local-work batch exists.

**If not:** Update `.ai/STATUS.md`, push, and report results to the Human User. No handoff, no agent turn, no report number consumed.

### Phase B — Create handoff only when justified

1. Select the next available number `NNNN`.
2. Choose the tier using Section 4. Default to Tier 2.
3. Create `.ai/handoffs/handoff-NNNN.md` using the format in Section 7.
4. Include the completed Chat AI changes as context; delegate only remaining local tasks. Name any code that Chat AI wrote but could not check.
5. Update `.ai/STATUS.md` (turn holder: Coding Agent, handoff `NNNN`), then commit and push the handoff and status (plus any final relevant changes).
6. Provide a **short copy-paste prompt** containing the tier, the agreed branch, exact handoff path, and handoff commit ID.
7. End Chat AI's development turn.

### Phase C — Coding Agent execution

1. Synchronize the branch, read `.ai/PROJECT.md`, and locate the exact handoff at the specified commit.
2. Work autonomously on the entire assigned batch, within the authority of the handoff's tier and the hardware test budget.
3. Troubleshoot recoverable issues rather than returning immediately.
4. Once completion criteria are satisfied—or genuinely blocked—write `.ai/reports/report-NNNN.md` with only the meaningful findings.
5. Commit and push all relevant code and report changes.
6. Return the short completion message in Section 9.
7. End the agent's development turn.

### Phase D — Chat AI resumes

1. Human User pastes the agent's completion message.
2. Chat AI retrieves the exact report and relevant code changes via branch and commit.
3. Assesses the task outcome, handles architectural implications, and makes further code changes directly when practical. After a Tier 1 turn, this includes checking the agent's design decisions against the project's architecture.
4. Starts another handoff **only if another meaningful local-work batch exists**.

## 7. Handoff Document Format

**Path:** `.ai/handoffs/handoff-NNNN.md`

```markdown
# Handoff NNNN

## Tier
1 or 2. For Tier 1, one or two sentences on why this work is complex
and why it is more productive locally than in the Chat AI environment.

## Objective
The concrete outcome for this agent turn.

## Context
Relevant repository state, work already implemented by Chat AI,
key design decisions, and environment assumptions.
Name any code Chat AI wrote but could not build or run.

## Tasks
A substantial coherent task or batch of related local tasks.
State expected results, not a needlessly prescriptive command list.

## Constraints
Important architectural boundaries, interfaces, and user requirements.
Tier 1: the boundaries within which the agent may make design decisions.
Tier 2: the files and interfaces the agent may change.

## Hardware Test Budget (optional)
Total runtime allowed for tests on the agent's hardware in this turn.
Omit to use the default in .ai/PROJECT.md.

## Completion Criteria
Conditions for the entire batch to be considered accomplished.

## Reporting
Specific findings, corrections, or results Chat AI will need next.
```

Guidance:

- Prefer a **self-contained actionable brief** over a long historical narrative.
- Do not repeat standing rules from `.ai/PROJECT.md`; the agent reads that file every turn.
- Bundle build, flashing, hardware bring-up, integration, debugging, and related checks when they serve one milestone.
- State scope and completion criteria clearly enough for autonomous work, while allowing ordinary implementation choices.
- Do not insert mandatory checkpoints or ask for intermediate reports unless genuinely necessary.
- Use targeted hardware checks appropriate to the feature; avoid exhaustive test procedures by default.
- **Match the detail to the tier.** A Tier 1 handoff states the goal and the boundaries and leaves the approach open. A Tier 2 handoff is more explicit about what to build, what to run, and what result counts as success, so a smaller model does not have to infer intent.

## 8. Handoff Granularity and Batching

**One substantial handoff is generally better than many tiny handoffs.** A handoff should justify the human copy/paste and ownership transfer.

**Poor pattern:**

```text
Chat AI writes sensor driver → Agent builds/tests it
Chat AI writes telemetry → Agent builds/tests it
Chat AI writes alarm → Agent builds/tests it
Chat AI writes configuration → Agent builds/tests it
```

**Preferred pattern:**

```text
Chat AI turn:
  implement driver + telemetry + alarm + configuration + error handling
  perform available host-side checks and commit
            ↓
Single agent handoff (Tier 2):
  build + flash + bring up sensor + verify telemetry/alarm
  + investigate and fix physical integration problems
            ↓
One concise report and completion message
```

Create a handoff when:

- A meaningful batch of local/hardware-dependent work has accumulated.
- Multiple implemented features are ready for integrated hardware bring-up.
- A substantial task is significantly more efficient on the Coding Agent's machine.
- A hardware- or environment-specific blocker prevents productive Chat AI progress.

Do **not** create a handoff solely because a small feature has not yet been flashed or locally verified. A single important blocker can still justify an immediate handoff. Avoid batching unrelated work merely to enlarge an agent turn; optimize for **coherent milestones**, not maximum task count.

**Batching and tiers:** Batch within a tier. Do not attach routine Tier 2 work to a Tier 1 handoff to save a relay unless it belongs to the same milestone, and do not raise a batch to Tier 1 because one small item in it is hard—Chat AI should do that item itself or isolate it.

## 9. Short Transfer Messages

### Chat AI → Coding Agent

After the handoff and Chat AI changes are pushed:

```text
Agent tier: <1 | 2>
Sync branch <branch> to commit <handoff-commit>.
Read and execute .ai/handoffs/handoff-NNNN.md at that commit.
Work autonomously until the batch completion criteria are met.
Write .ai/reports/report-NNNN.md, commit and push all changes.
Return iteration, tier, status, branch, final commit ID, and short summary.
```

The prompt is a **pointer and execution trigger**, not the technical specification. The handoff file is authoritative. The first line tells the Human User which model to start the agent with.

### Coding Agent → Chat AI

After the agent's work and report are pushed:

```text
Iteration: NNNN
Tier: <1 | 2>
Status: Completed
Branch: <branch>
Commit: <final-commit>
Summary: <one or two sentences>
Report: .ai/reports/report-NNNN.md
```

Use `Partial` or `Blocked` only for a genuine unresolved exception. A Tier 2 agent that stops because the work exceeds its tier adds `Tier escalation recommended` to the summary. The final commit ID must identify the pushed changes, including the report.

## 10. Report Format: Minimal, Adaptive, Useful

**Path:** `.ai/reports/report-NNNN.md`

Reports are **not fixed-form checklists**. Include only information that materially helps Chat AI understand results or continue the project. Leave out routine command transcripts, empty sections, redundant details, and exhaustive test logs.

A simple successful report can be as short as:

```markdown
# Report 0004

Implemented task editing and deletion.
Persistence works after refresh. Build passed.
No known issues.
```

A more complicated report should add only **relevant** details, such as:

- Important implementation deviations or interface changes.
- Significant hardware observations and root causes.
- Fixes applied, tests that establish completion, or limits of validation.
- Remaining blockers or decisions that genuinely need the next participant.
- After a Tier 1 turn: the design decisions made within the handoff's boundaries.
- After a Tier 2 escalation: what was completed, what was left, and why it exceeds the tier.

The report should communicate **what matters**, not everything the agent did.

## 11. Example Split

The split below is a starting point for any project in which part of the work depends on a local toolchain, runtime, or hardware. Adjust it in `.ai/PROJECT.md` where a project differs.

| Work | Owner |
| --- | --- |
| Architecture, interfaces, algorithms, application logic, configuration, scripts, documentation | Chat AI |
| Code that Chat AI can build or test in its own environment | Chat AI |
| Code whose design is settled but which only the local toolchain can build | Chat AI writes it unverified; Tier 2 builds and corrects it |
| Building with the local toolchain, deploying, running, hardware or device bring-up, runtime diagnosis, integration fixes, benchmarking | Coding Agent, Tier 2 |
| Complex development whose design depends on what the local build, run, or device shows | Coding Agent, Tier 1 |
| Review of returned work against the architecture | Chat AI |

Two illustrations:

- **Embedded firmware.** Chat AI writes drivers, state machines, and application logic. Tier 2 cross-compiles, flashes, brings up the board, and fixes what the hardware reveals. Tier 1 is for work such as a timing-critical driver that has to be tuned on the board.
- **GPU or accelerator software.** Chat AI writes host-side code, interfaces, and anything it can test on a CPU. Tier 2 builds with the vendor toolchain, runs on the devices, and benchmarks. Tier 1 is for work such as restructuring device code where each step needs a real build and run to judge.

**Important:** A change can be implementation-complete but **not yet verified on the local hardware**. Chat AI should state that limitation, not automatically generate a separate “build and test this function” handoff. Bundle such checks into the next meaningful local milestone.

The Coding Agent is authorized to fix issues discovered during bring-up rather than merely reporting them back, within the authority of its tier. It should preserve specified architecture and avoid unrelated redesigns.

## 12. Validation, Failure, and GitHub Actions

- Favor targeted checks that give useful confidence for the prototype. No routine exhaustive validation loops.
- The Coding Agent should independently resolve recoverable build failures, integration bugs, toolchain problems, and hardware configuration mistakes during its turn.
- Escalate genuine blockers only after reasonable autonomous efforts. State exactly what blocks completion and what evidence was gathered.
- A Tier 2 agent treats "this needs a redesign or substantial new development" as a genuine blocker for that part and recommends tier escalation rather than attempting it.
- Neither AI should claim success if essential completion criteria were not met.

### Hardware test runtime

Tests on the Coding Agent's hardware should be **meaningful but time-efficient**. Their length follows the project stage and the size of the change, not the capacity of the machine.

- **Early development and minor changes:** short, targeted runs that show the changed behaviour works, typically minutes in total for an agent turn. A long hardware run for a small code change or an early prototype is not justified.
- **Integration milestones:** a short representative run per configuration that matters, still bounded.
- **Long-duration tests** (soak, endurance, full benchmark matrices, capacity sweeps) are reserved for when the project has converged to a **release-candidate state**, or when the Human User explicitly asks for one.
- **Where the limit is set:** `.ai/PROJECT.md` records the project stage and the default hardware test budget per agent turn. A handoff may set a different budget for one turn in its optional *Hardware Test Budget* section. Chat AI does not set a long budget before release-candidate state without the Human User's agreement.
- **At the limit:** the agent does not start further hardware test cases once the budget is used. It lets a running case finish or stops it at a safe point, and reports which planned checks were not run. Reaching the budget is not a failure and does not by itself make the status `Partial`.
- Builds, static checks, and tests that do not occupy the scarce hardware are not counted against this budget.

### GitHub Actions

- Prefer local builds, tests, linters, and scripts over GitHub Actions.
- **Do not dispatch, run, enable, or consume GitHub Actions unless the Human User explicitly requests it.**
- Any newly created GitHub Actions workflow must be manual-only (`workflow_dispatch`) unless the Human User explicitly requests automatic triggers. Do not silently alter existing automatic triggers.

This is a rapid-prototyping workflow, not a production release assurance process. Practical safeguards and truthfully stated limitations remain appropriate, but elaborate verification and ceremony are not the default.

## 13. Completion and Exceptions

An **agent iteration** ends when the Coding Agent has:

1. Completed the delegated batch as far as possible under its criteria.
2. Committed and pushed the implementation and matching report.
3. Returned its concise final message and stopped modifying the repository.

An agent iteration also ends when the Human User **cancels** the open handoff and Chat AI records the cancellation as `report-NNNN.md` (Section 5).

A **Chat AI-only request** ends when Chat AI has implemented, updated `.ai/STATUS.md`, pushed (if it has repository access), and reported its result; no handoff is required.

The **overall prototype** is done when the Human User accepts the outcome or chooses to stop.

If Chat AI cannot write to the repository, it must say so and provide usable implementation/handoff content through available means; it must **not** pretend a GitHub file or commit exists. If the Coding Agent lacks required hardware, credentials, or authorization, it should seek a practical local alternative or accurately report a genuine blocker.

## 14. Instructions for an AI Receiving This Document

**Adopt this protocol immediately.** Infer ordinary implementation details and avoid unnecessary clarification. Ask for missing essentials (for example repository identity or access) only when they genuinely block progress.

**If you are Chat AI:**

- Start every session by reading `.ai/PROJECT.md` and `.ai/STATUS.md`. If a handoff is open, do not write to the project; only the Human User can cancel it.
- Be a **hands-on developer**, not just an architect or coordinator.
- Implement everything you can adequately check with your own tools. For the rest, choose between unverified code for Tier 2 and a Tier 1 handoff, and label unverified code.
- Maintain exclusive ownership during your turn.
- Update `.ai/STATUS.md` as the last step of every turn.
- Commit and push changes before delegating.
- Create a **numbered, substantial, batched handoff only when necessary**.
- **Name the tier in every handoff.** Default to Tier 2; justify Tier 1 in the handoff.
- Include tier, exact branch, handoff path, and commit ID in the agent prompt.
- On return, inspect the matching report, resume coding, and delegate again only when worthwhile.

**If you are Coding Agent:**

- Sync the exact specified revision, read `.ai/PROJECT.md`, then read the exact handoff, including its tier.
- Own the whole batch autonomously: implement, integrate, test, debug, and retry as needed.
- Keep hardware tests short and within the budget; report planned checks you did not run.
- Do not edit `.ai/PROJECT.md` or `.ai/STATUS.md`.
- **Tier 1:** make design decisions inside the handoff's boundaries and record them in the report.
- **Tier 2:** fix what is needed inside the named scope; if the work needs redesign or substantial new development, stop that part and recommend tier escalation.
- Do not return partial progress for ordinary recoverable issues.
- Produce a **minimal task-relevant report**, commit and push, and return the brief completion message.
- Stop modifying project files when your turn is handed back.

**If you are the Human User:**

- Provide direction and decisions.
- Start the Coding Agent with the model for the tier named on the first line of the agent prompt.
- Agree changes to `.ai/PROJECT.md`, including the project stage and the hardware test budget.
- Cancel an open handoff when needed; nobody else can.
- Relay only the short agent prompt and short agent completion message between interfaces.

---

**Fundamental operating model:** Chat AI **implements what it can adequately check** and delegates **substantial coherent batches** selectively, each to the **cheapest agent tier that can do the work**. Coding Agent executes **autonomously to completion** with local resources, within the authority of its tier and a short hardware test budget. **Only one AI modifies the repository at a time**, and an open handoff shows who. `.ai/PROJECT.md` carries the standing rules, `.ai/STATUS.md` the current state, and numbered handoffs and reports the context of each delegation. Human coordination stays minimal.


**Fresh-session efficiency:** Assume every turn, Coding Agent or Chat AI, may begin without previous conversational context. A Coding Agent turn starts from `.ai/PROJECT.md` and the numbered handoff; a Chat AI session starts from `.ai/PROJECT.md` and `.ai/STATUS.md`. The handoff should contain only the information necessary to execute the assigned task. Do not reproduce project history or general architecture already discoverable in the repository. The Coding Agent should inspect only the files and context relevant to its immediate task, retrieving additional information when a concrete need arises. Previous reports and historical handoffs should not be read by default.
