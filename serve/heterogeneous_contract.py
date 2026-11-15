"""CPU-only limits and progress validation for the existing native coordinator."""

from serve.heterogeneous_long_context import MAX_LONG_CONTEXT, command_cap, validate_capacity

MAX_CONTEXT = 8192
MAX_COMMAND_BYTES = 65536
MAX_WORKLOAD_CONTEXT = 32768
MAX_WORKLOAD_NEW = 4096
MAX_WORKLOAD_COMMAND_BYTES = 384 * 1024
MAX_WORKLOAD_PROMPT_BYTES = 256 * 1024
PAYLOAD_FLOATS = 16 * 1024 * 1024 // 4


def generation_limits(cells, max_t, vocab, hc, n_embd, requested, max_new, chunk, workload=False, long_context=False):
    if any(type(x) is not int or x <= 0 for x in (cells, max_t, vocab, hc, n_embd, max_new)):
        raise ValueError("invalid model geometry or generation limits")
    if type(workload) is not bool:
        raise ValueError("invalid workload mode")
    if type(long_context) is not bool or (long_context and not workload):
        raise ValueError("long-context requires workload mode")
    if long_context:
        validate_capacity(cells, requested)
    ceiling = MAX_LONG_CONTEXT if long_context else MAX_WORKLOAD_CONTEXT if workload else MAX_CONTEXT
    if type(requested) is not int or not 2 <= requested <= ceiling:
        raise ValueError(f"max-context must be 2..{ceiling}")
    if max_new > (MAX_WORKLOAD_NEW if workload else 256):
        raise ValueError("completion budget exceeds selected mode")
    if workload and requested > cells:
        raise ValueError("workload context exceeds actual worker capacity")
    context = min(cells, requested)
    if max_new >= context:
        raise ValueError("completion budget leaves no prompt capacity")
    # Match live_smoke.cpp's activation and STFR float payload guards.
    safe = min(8, max_t, PAYLOAD_FLOATS // ((hc + 1) * n_embd + hc),
               (PAYLOAD_FLOATS - 1) // (vocab + 2))
    if type(chunk) is not int or not 0 <= chunk <= safe or not safe:
        raise ValueError("prompt chunk exceeds validated worker or payload limits")
    return context, chunk or safe


MTP_CONTEXT_CAPACITY = 2048
MAX_MTP_BULK_PROMPT = MTP_CONTEXT_CAPACITY - 1
MAX_MTP_BULK_CHUNK = 64


def validate_request(ids, max_new, vocab, context, long_context=False, *, mtp_bulk=False):
    if (not ids or type(max_new) is not int or max_new <= 0 or
            len(ids) + max_new > context or
            any(type(t) is not int or not 0 <= t < vocab for t in ids)):
        raise ValueError("prompt/generation budget exceeds validated context or vocabulary")
    if mtp_bulk and (context != MTP_CONTEXT_CAPACITY or len(ids) > MAX_MTP_BULK_PROMPT):
        raise ValueError("MTP bulk prompt requires the exact 2K service capacity")
    # Include the resident request's completion-count prefix. One-shot
    # input is shorter, even with its terminating newline.
    cap = command_cap(context, long_context)
    if len(" ".join(map(str, [max_new, *ids]))) > cap:
        raise ValueError("serialized token request exceeds configured byte cap")


class Progress:
    def __init__(self, request, prompt, max_new, chunk, vocab, stops, stream, lookup=False, native_prefill=False, *, mtp_spec=0):
        if lookup and (stream or chunk < 2):
            raise ValueError("lookup requires buffered mode and chunk >= 2")
        if type(mtp_spec) is not int or not 0 <= mtp_spec <= 4 or (mtp_spec and lookup):
            raise ValueError("MTP service progress requires independent mode and spec 1..4")
        if mtp_spec and native_prefill and (not 1 <= prompt <= MAX_MTP_BULK_PROMPT or
                                           not 1 <= max_new <= 256 or prompt + max_new > MTP_CONTEXT_CAPACITY):
            raise ValueError("MTP bulk prompt and completion exceed 2K service capacity")
        self.mtp_spec = mtp_spec
        self.lookup = lookup
        self.native_prefill = native_prefill
        self.request, self.prompt, self.max_new = request, prompt, max_new
        self.chunk, self.vocab, self.stops, self.stream = chunk, vocab, stops, stream
        self.pos = self.window = 0
        self.tokens = []
        self.terminal = False

    def accept(self, line):
        fields = line.split()
        batched = bool(self.stream and self.mtp_spec)
        if fields[0:1] != ["HETERO_PROGRESS"] or (len(fields) < 7 if batched else len(fields) != (5 if self.stream else 4)):
            raise ValueError("wrong progress record shape")
        request, window, end = map(int, fields[1:4])
        if self.native_prefill and self.window == 0 and self.prompt > 1:
            expected = self.prompt - 1
        else:
            expected = self.pos + (min(self.chunk, self.prompt - self.pos) if self.pos < self.prompt else 1)
        max_advance = (self.mtp_spec + 1 if self.mtp_spec else self.chunk if self.lookup else 1)
        valid_position = (expected <= end <= self.pos + max_advance
                          if self.pos >= self.prompt else end == expected)
        if (self.terminal or request != self.request or window != self.window + 1 or
                not valid_position or end > self.prompt + self.max_new - 1):
            raise ValueError("resident progress identity or position mismatch")
        token = None
        if batched:
            if fields[4] != "BATCH" or fields[5] not in ("0", "1"):
                raise ValueError("invalid MTP streaming batch marker")
            terminal = fields[5] == "1"
            count = int(fields[6])
            if not 0 <= count <= self.mtp_spec + 1 or len(fields) != 7 + count:
                raise ValueError("invalid MTP streaming batch extent")
            tokens = tuple(map(int, fields[7:]))
            # One target prediction after final prompt commit, then one
            # new prediction per accepted feedback row, minus terminal stop.
            expected_tokens = 0 if end < self.prompt else end - self.prompt + (not terminal)
            if ((end < self.prompt and terminal) or len(self.tokens) + count != expected_tokens or
                    expected_tokens > self.max_new or
                    any(not 0 <= t < self.vocab or t in self.stops for t in tokens)):
                raise ValueError("MTP streamed predictions disagree with committed positions")
            self.tokens.extend(tokens)
            self.terminal = terminal
            token = tokens
        elif self.stream:
            prediction = int(fields[4])
            if end < self.prompt:
                if prediction != -1:
                    raise ValueError("prediction before prompt commit")
            elif prediction == -1:
                self.terminal = True
            else:
                if not 0 <= prediction < self.vocab or prediction in self.stops:
                    raise ValueError("invalid streamed prediction")
                self.tokens.append(prediction)
                token = prediction
        self.window, self.pos = window, end
        return token


def timing_record(start, end, prefill_end, last_checkpoint, first_output, prompt, pos, completion):
    # Wall-clock host timings, including IPC/ACKs, reset and consumer pacing;
    # not GPU kernel profiling or HTTP client TTFT.
    prefill = None if prefill_end is None else prefill_end - start
    decode = None if prefill_end is None else last_checkpoint - prefill_end
    feedback = max(0, pos - prompt)
    def ms(value):
        return None if value is None else value * 1000
    return {"elapsed_ms": ms(end - start), "prefill_ms": ms(prefill),
            "decode_ms": ms(decode), "first_prediction_ms": ms(prefill),
            "first_output_ms": ms(None if first_output is None else first_output - start),
            "prompt_tokens": prompt, "committed_prompt_tokens": min(prompt, pos),
            "decode_feedback_tokens": feedback, "completion_tokens": completion,
            "prompt_tokens_per_second": prompt / prefill if prefill else None,
            "decode_tokens_per_second": feedback / decode if decode else None}
