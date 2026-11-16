# Vanilla-first development — mandatory project-wide

Owner direction: vanilla Strata guides all development, including native engine
changes, heterogeneous adapters, CLI, serving, tools, and performance work.
Do not invent new speculative, attention, prefill, cache, synchronization,
scheduling, memory-management, or generation algorithms. An opt-in switch is
not permission to invent an unproven alternative.

1. Identify the exact pinned vanilla source/revision, actual native function,
   effective options/dispatch and ownership/ordering contract before selecting
   a change. Record them in the implementation notes. Compare source and
   measurements to the best comparable native configuration, not a crippled
   serial reference. Repo main is a useful ancestor, not a substitute for the
   pinned Oberon reference.
2. Reuse the native operation unchanged whenever possible. If native assumes
   a single runtime, extract the smallest shared native interface, and have
   both callers use it. Keep native behavior/defaults intact. Never clone
   a scheduler, model executor, adaptive controller or prefix-cache policy.
3. Heterogeneous-only logic is limited to mixed-runtime transport, boundary
   validation, owner identity, resource admission, necessary coordination,
   supervision and observability. Keep adjacent same-backend layers in one
   runtime domain. Never pass foreign device/stream/event handles. Preserve
   all-owner commit/finite/identity gates and cooperative drain; retain
   necessary cross-process fences. Do not change the precision or layer-split
   algorithm to chase a benchmark. No tensor-parallel migration.
4. If the native implementation/reference cannot be inspected, stop that
   algorithm change and document the blocker. A real upstream gap needs a
   source-grounded minimal adaptation and explicit owner approval before
   introducing a new algorithm. Prefer extracting the proven native path.
5. Audit inherited experimental flags, alternate implementations, repeated
   loader/dispatch code and legacy diagnostics before reuse. Keep existing
   qualified functionality until a native replacement is verified. Historical
   diagnostic arithmetic stays isolated; do not promote it to performance
   defaults. Avoid blanket flag enabling and routine historical parity hunts.
6. Preserve the documented M5/M6 160 accepted output tok/s and useful
   utilization goals, qualified resource limits and test budget. Measure
   actual dispatch and overlap, use untraced comparable workloads for speed.
   Use local builds/tests and the existing toolchain/venv (28 jobs). No GitHub
   Actions execution or new automatic workflows; no forced GPU termination.

See [performance roadmap](PERFORMANCE_ROADMAP.md),
[divergence audit](VANILLA_DIVERGENCE_AUDIT.md), and
[implementation contract](../experimental/heterogeneous/docs/IMPLEMENTATION_CONTRACT.md).
