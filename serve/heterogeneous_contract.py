"""CPU-only limits and progress validation for the existing native coordinator."""

MAX_CONTEXT = 8192
MAX_COMMAND_BYTES = 65536
PAYLOAD_FLOATS = 16 * 1024 * 1024 // 4


def generation_limits(cells, max_t, vocab, hc, n_embd, requested, max_new, chunk):
    if any(type(x) is not int or x <= 0 for x in (cells, max_t, vocab, hc, n_embd, max_new)):
        raise ValueError("invalid model geometry or generation limits")
    if type(requested) is not int or not 2 <= requested <= MAX_CONTEXT:
        raise ValueError("max-context must be 2..8192")
    context = min(cells, requested)
    if max_new >= context:
        raise ValueError("completion budget leaves no prompt capacity")
    # Match live_smoke.cpp's activation and STFR float payload guards.
    safe = min(8, max_t, PAYLOAD_FLOATS // ((hc + 1) * n_embd + hc),
               (PAYLOAD_FLOATS - 1) // (vocab + 2))
    if type(chunk) is not int or not 0 <= chunk <= safe or not safe:
        raise ValueError("prompt chunk exceeds validated worker or payload limits")
    return context, chunk or safe


def validate_request(ids, max_new, vocab, context):
    if (not ids or type(max_new) is not int or max_new <= 0 or
            len(ids) + max_new > context or
            any(type(t) is not int or not 0 <= t < vocab for t in ids)):
        raise ValueError("prompt/generation budget exceeds validated context or vocabulary")
    # Include the resident request's completion-count prefix. One-shot
    # input is shorter, even with its terminating newline.
    if len(" ".join(map(str, [max_new, *ids]))) > MAX_COMMAND_BYTES:
        raise ValueError("serialized token request exceeds 64 KiB")


class Progress:
    def __init__(self, request, prompt, max_new, chunk, vocab, stops, stream):
        self.request, self.prompt, self.max_new = request, prompt, max_new
        self.chunk, self.vocab, self.stops, self.stream = chunk, vocab, stops, stream
        self.pos = self.window = 0
        self.tokens = []
        self.terminal = False

    def accept(self, line):
        fields = line.split()
        if fields[0:1] != ["HETERO_PROGRESS"] or len(fields) != (5 if self.stream else 4):
            raise ValueError("wrong progress record shape")
        request, window, end = map(int, fields[1:4])
        expected = self.pos + (min(self.chunk, self.prompt - self.pos) if self.pos < self.prompt else 1)
        if (self.terminal or request != self.request or window != self.window + 1 or
                end != expected or end > self.prompt + self.max_new - 1):
            raise ValueError("resident progress identity or position mismatch")
        token = None
        if self.stream:
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
