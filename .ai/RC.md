# Release candidate 1: qualification

The mixed split's first release candidate, on vanilla 0.2.15. It is qualified with a **synthetic workload shaped like
the Human User's pi coding-agent sessions** (Human User, 2026-11-19): 256K-token context, compaction at 92% of the
context, and a tool-call pruning extension. Real sessions are not used.

## What RC1 is
- **The commit:** a commit of `feature/mixed-native` that passed the criteria below, tagged `mixed-split-rc1` by Chat AI
  after its review. Configurations: `mixed_split/configs/mixed.example.json` (mixed) and `hiponly.example.json`
  (production, the reference).
- **The documentation:** `mixed_split/README.md` says how to build, configure and run the mixed split, what it does not
  support (`.ai/DIVERGENCE.md`), and the measured numbers of the qualification. Chat AI writes it from the reports.

## The workload model (driver: `mixed_split/tools/agent_session.py`)
All lengths are in model tokens (the model's own tokenizer). Every random choice comes from a fixed seed, so both
configurations see exactly the same requests.

- **Context:** `--max-context 262144`, as production.
- **Fixed prefix:** a system prompt with tool definitions of about 10,000 tokens. It is the same in every session and
  every cycle, as pi's is.
- **One turn:**
  - The assistant makes one tool call: 128 generated tokens.
  - The tool's output is appended: 50% of outputs are 500 to 2,000 tokens, 35% are 2,000 to 8,000, and 15% are
    8,000 to 20,000. Outputs are real files and search results from this checkout (vanilla source and docs), so the
    content looks like code. An output that would take the request past 248,000 tokens is cut to fit.
  - Every fifth turn the assistant writes a longer answer: 512 generated tokens.
  - Requests use pi's form where the server supports it: tool definitions in the request, assistant `tool_calls`, and
    `tool` messages.
- **Identical contexts across configurations:** the model's real reply is generated and timed. But what goes into the
  history is a scripted assistant message of the planned length, not the model's output. Both configurations
  therefore build exactly the same contexts, and their cache behaviour can be compared. Greedy decoding with a fixed
  token budget per request (the end-of-text token is ignored where the server allows it, so every request generates
  the same number of tokens).
- **Tool-call pruning** (the extension's role): every 8 turns, every tool output longer than 1,000 tokens that is
  older than the 6 most recent turns is replaced by a stub of about 150 tokens (its first lines and a "pruned" note).
  Outputs already pruned stay as they are. The history then changes in the middle, at the oldest pruned output: the
  engine reuses the prefix before it (a checkpoint) and rereads the rest.
- **Compaction** (pi's role), when the context reaches 92% of 262,144 = 241,172 tokens:
  - one compaction request: the whole context plus a "summarise the session" message, with up to 2,048 generated
    tokens;
  - the next cycle starts from the fixed prefix, a scripted summary of about 8,000 tokens, and the last two turns.
- **One session** runs until the second compaction is done and the third cycle has started (one request after it).
  Against the driver's stub, one full session is 589 requests (about 290 turns per cycle) and reads about 7 million
  new tokens, most of them rereads after pruning. Estimated at about 1.5 hours per session on the cards. All the numbers above are estimates of the
  Human User's workload; the driver takes each of them as a flag, so they can be corrected without code changes.
- **A scaled session** for smoke tests uses the same model with compaction at 92% of 32,768 tokens and two cycles.

## Measurements
- **Per request:**
  - context length, new tokens, reused tokens (from the server or engine log);
  - time to first token, prefill tok/s on the new tokens, decode tok/s, generated tokens;
  - whether a checkpoint, a park or the live session was reused.
- **Per compaction:** time to first token, and decode tok/s at full context.
- **Per session:** wall time, number of requests, total new and reused tokens.
- **Every 30 s:** VRAM per card, available RAM, the engines' resident memory.
- **Always:** `ERR` lines, refused session operations, restarts, the safety checks of `.ai/PROJECT.md`.
- **Output:** a JSON line per request and per sample under `.ai/runs/<iteration>-*`, and a summary table made by a
  script (decode and prefill by context bucket: 0 to 32K, 32K to 64K, and so on up to 241K).

## Pass criteria
1. **Stability:**
   - no `ERR` line and no refused session operation in any session;
   - no crash, core dump, GPU fault line or reboot;
   - every server and engine ends cooperatively;
   - over the soak run as well (RC handoff 2).
2. **Correctness:** at least three probes per full session, at these points:
   - after a pruning step;
   - after a compaction;
   - at a context of 200K tokens or more.

   Each probe (max_tokens 64, greedy, `STRATA_IQ_MT_MIN=1`) is served with the cache as it stands, then replayed on a
   freshly started server of the same configuration with top-5 logprobs (`agent_session.py replay`). It passes when
   the replies are equal, or when they first differ at a **near-tie**: the live reply's token there is among the fresh
   server's top 5 and at most 1.0 nat below its choice. It fails when the live token is not among them, or is further
   below. "Unjudged" (the fresh logprobs incomplete) is reported and repeated once.

   Why not "equal": reports 0087 and 0088 found that a reply served from cached prefixes is not byte-identical to a
   fresh one, on HIP-only (vanilla plus the carried patch) as on the mixed split; with prompt and conversation caches
   off, the tested probe was equal, and two fresh mixed servers agreed with each other. Corruption (the 19 November
   crash) shows as non-finite logits and garbage, not as a near-tie.

   In addition, over the whole session: **no `non-finite logits` line** in either server's log.
3. **Resources:** after the first cycle, VRAM per card grows by less than 1% and available RAM falls by less than
   2 GiB across the remaining cycles and the soak. No swap use.
4. **Performance against production** on the same sessions: session wall time, and decode tok/s at 200K tokens or more.
   - Mixed is expected to be no slower. If it is slower, the RC is not blocked, but the report says by how much, and
     the Human User decides.

## Plan
- **RC handoff 1:**
  - fresh 0.2.15 builds;
  - the driver;
  - scaled smoke sessions in both configurations;
  - one full session in each configuration, with the correctness probes.
- **RC handoff 2:**
  - a soak run of the mixed configuration (consecutive full sessions for about three hours), with a cancelled request
    during a compaction and one server restart in it;
  - then Chat AI reviews, writes `mixed_split/README.md`, and tags `mixed-split-rc1`.
