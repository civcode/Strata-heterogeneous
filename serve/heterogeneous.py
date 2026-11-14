"""Single-request-at-a-time adapter to the native resident coordinator.

The coordinator owns model loading, request IDs, generation and GPU commits;
this module only transports commands and implements the existing Engine
generator interface. No alternative inference or HTTP implementation.
"""
from __future__ import annotations

import argparse
import json
import queue
import subprocess
import sys
import threading


def prepare_server(cfg):
    """Validate explicit local configuration and reuse CLI source binding."""
    from strata_cli.heterogeneous import prepare_native, model_template, _native
    opts = cfg.get("heterogeneous")
    if not isinstance(opts, dict):
        raise ValueError("heterogeneous engine needs a heterogeneous config object")
    required = {"coordinator", "cuda_worker", "hip_worker", "config", "plan", "fingerprint"}
    optional = {"layers", "hc", "n_embd", "max_new_tokens", "prompt_chunk", "timeout_ms", "shared"}
    if not required <= opts.keys() or opts.keys() - required - optional:
        raise ValueError("missing or unknown heterogeneous config keys")
    for key in ("coordinator", "cuda_worker", "hip_worker", "plan"):
        if not isinstance(opts[key], str) or not opts[key]:
            raise ValueError(f"heterogeneous.{key} must be a nonempty string")
    if not isinstance(opts["config"], list) or not opts["config"] or not all(
            isinstance(x, str) and x for x in opts["config"]):
        raise ValueError("heterogeneous.config must list ordered worker config paths")
    values = dict(layers=48, hc=4, n_embd=2560, max_new_tokens=32,
                  prompt_chunk=1, timeout_ms=120000, shared=False)
    values.update(opts)
    for key in ("fingerprint", "layers", "hc", "n_embd", "max_new_tokens", "prompt_chunk", "timeout_ms"):
        if type(values[key]) is not int or values[key] <= 0:
            raise ValueError(f"heterogeneous.{key} must be a positive integer")
    if type(values["shared"]) is not bool:
        raise ValueError("heterogeneous.shared must be boolean")
    args = argparse.Namespace(**values)
    argv, source, tokenizer, stops = prepare_native(args)
    template = model_template(source)
    argv += ["--resident"]
    _native(argv)
    return args, argv, source, tokenizer, stops, template


class ResidentEngine:
    # Service serializes requests through its existing FIFO. An independent
    # lock also prevents protocol overlap if invoked outside Service.
    batch = 0

    def __init__(self, argv, source, stop_ids, max_new, chunk, timeout_ms, died):
        self.source = source
        self.stop_ids = set(stop_ids)
        self.max_context = min(source["max_cells"], 4096)
        self.max_new = max_new
        self.chunk = chunk
        self.info = {"version": "heterogeneous-resident/experimental"}
        self.last_completion = None
        self._died = died
        self._lock = threading.Lock()
        self._records = queue.Queue()
        self._request = 0
        self._failed = False
        self._closed = False
        # One window can involve several stage/frame deadlines. Native
        # per-operation timeouts remain authoritative.
        stages = argv.count("--config")
        self._deadline = max(10, timeout_ms / 1000 * (4 * stages + 4))
        self.proc = subprocess.Popen(argv + ["--run"], stdin=subprocess.PIPE,
                                     stdout=subprocess.PIPE, stderr=None, text=True,
                                     encoding="utf-8", errors="replace", bufsize=1,
                                     start_new_session=True)
        self._reader = threading.Thread(target=self._read_records, daemon=True,
                                        name="heterogeneous-native-output")
        self._reader.start()
        try:
            if self._receive() != "HETERO_READY 1":
                raise self._died("native resident readiness not acknowledged")
            print(f"HETERO_READY coordinator_pid={self.proc.pid}", file=sys.stderr, flush=True)
        except BaseException:
            self._poison()
            self._wait_exit()
            raise

    def _read_records(self):
        try:
            for line in self.proc.stdout:
                line = line.rstrip("\r\n")
                if line.startswith("HETERO_"):
                    self._records.put(line)
                else:
                    print(line, file=sys.stderr, flush=True)
        finally:
            self._records.put(None)

    def _receive(self):
        try:
            record = self._records.get(timeout=self._deadline)
        except queue.Empty as exc:
            raise self._died("native resident response timed out; restart required") from exc
        if record is None:
            raise self._died("native resident coordinator exited; restart required")
        return record

    def _send(self, command):
        try:
            self.proc.stdin.write(command + "\n")
            self.proc.stdin.flush()
        except (OSError, ValueError) as exc:
            raise self._died("native resident input closed; restart required") from exc

    def _poison(self):
        self._failed = True
        # EOF at the next safe checkpoint lets the existing native supervisor
        # close sockets and reap workers. No new GPU fault/kill path here.
        try:
            self.proc.stdin.close()
        except (OSError, ValueError):
            pass

    def _wait_exit(self):
        try:
            return self.proc.wait(timeout=self._deadline)
        except subprocess.TimeoutExpired as exc:
            raise self._died("native coordinator did not exit; operator inspection required") from exc

    def alive(self):
        return not self._closed and not self._failed and self.proc.poll() is None

    def restart(self):
        # Never automatically reload or reuse an uncertain worker set.
        raise self._died("heterogeneous resident engine stopped; restart the server explicitly")

    def exit_code(self):
        return self.proc.poll()

    def _validate(self, ids, max_new, sampling, embeddings):
        if (embeddings is not None or type(max_new) is not int or not 1 <= max_new <= self.max_new or
                not ids or len(ids) > 4096 or len(ids) + max_new > self.max_context or
                any(type(t) is not int or not 0 <= t < self.source["vocab"] for t in ids)):
            raise ValueError("unsupported resident request, prompt tokens or completion budget")
        sampling = sampling or {}
        allowed = {"temperature": (None, 0), "top_p": (None, 1),
                   "top_k": (None, 0, 1), "min_p": (None, 0),
                   "presence_penalty": (None, 0), "frequency_penalty": (None, 0),
                   "repetition_penalty": (None, 1), "repeat_penalty": (None, 1),
                   "penalty_last_n": (None, 0), "repeat_last_n": (None, 0),
                   "seed": (None, 0)}
        for key, values in allowed.items():
            if sampling.get(key) not in values:
                raise ValueError(f"{key} is not supported by the experimental greedy resident engine")
        for key in ("logit_bias", "logprobs", "top_logprobs", "tools", "functions",
                    "tool_choice", "strata_img_keys", "best_of", "mirostat",
                    "typical_p", "response_format"):
            if sampling.get(key):
                raise ValueError(f"{key} is not supported by the experimental resident engine")
        if sampling.get("n") not in (None, 1):
            raise ValueError("resident engine supports one completion per request")

    def _result(self, line, ids, max_new):
        if not line.startswith("HETERO_GENERATION "):
            raise self._died("unexpected native resident completion")
        try:
            record = json.loads(line[len("HETERO_GENERATION "):])
            tokens = record["token_ids"]
            reason = record["finish_reason"]
            stop = record["stop_token"]
            if (record["request_id"] != self._request or record["prompt_tokens"] != len(ids) or
                    not isinstance(tokens, list) or len(tokens) > max_new or
                    any(type(t) is not int or not 0 <= t < self.source["vocab"] or t in self.stop_ids
                        for t in tokens) or type(stop) is not int or
                    not ((reason == "length" and len(tokens) == max_new and stop == -1) or
                         (reason == "stop" and len(tokens) < max_new and stop in self.stop_ids) or
                         (reason == "cancelled" and not tokens and stop == -1))):
                raise ValueError("inconsistent completion metadata")
        except (ValueError, KeyError, TypeError) as exc:
            raise self._died("invalid native resident completion") from exc
        return record

    def generate(self, ids, max_new, sampling, cancel, embeddings=None, prefill_only=False):
        if prefill_only:
            raise ValueError("resident prefill-only requests are unsupported")
        self._validate(ids, max_new, sampling, embeddings)
        with self._lock:
            if cancel.is_set():
                return
            if not self.alive():
                raise self._died("resident engine unavailable; restart required")
            self._request += 1
            self.last_completion = None
            waiting = False
            completed = False
            cancelled = False
            window = pos = 0
            try:
                self._send(" ".join(map(str, [max_new, *ids])))
                while True:
                    line = self._receive()
                    if line.startswith("HETERO_PROGRESS "):
                        # No output token is exposed before the native request
                        # completes. Heartbeats let Service notice disconnects.
                        try:
                            request, current, end = map(int, line.split()[1:])
                        except (ValueError, TypeError) as exc:
                            raise self._died("malformed resident progress") from exc
                        expected = pos + (min(self.chunk, len(ids) - pos) if pos < len(ids) else 1)
                        if (cancelled or request != self._request or current != window + 1 or
                                end != expected or end > len(ids) + max_new - 1):
                            raise self._died("resident progress identity or position mismatch")
                        window, pos = current, end
                        waiting = True
                        if not cancel.is_set():
                            yield None
                        cancelled = cancel.is_set()
                        self._send("cancel" if cancelled else "continue")
                        waiting = False
                    else:
                        record = self._result(line, ids, max_new)
                        reason = record["finish_reason"]
                        expected = len(ids) + len(record["token_ids"]) - (reason == "length")
                        if (cancelled != (reason == "cancelled") or
                                (not cancelled and pos != expected)):
                            raise self._died("resident completion disagrees with committed positions")
                        self.last_completion = record
                        print(f"HETERO_REQUEST_DONE request={self._request} finish={reason} next={pos}",
                              file=sys.stderr, flush=True)
                        completed = True
                        break
            except GeneratorExit:
                # The only suspension before completion is a committed
                # checkpoint. Drain its cancellation ACK before releasing FIFO.
                if waiting:
                    try:
                        self._send("cancel")
                        record = self._result(self._receive(), ids, max_new)
                        if record["finish_reason"] != "cancelled":
                            raise self._died("resident cancellation not acknowledged")
                        self.last_completion = record
                        print(f"HETERO_REQUEST_DONE request={self._request} finish=cancelled next={pos}",
                              file=sys.stderr, flush=True)
                        completed = True
                        waiting = False
                    except BaseException:
                        self._poison()
                        raise
                raise
            except BaseException:
                self._poison()
                raise
            finally:
                if not completed:
                    self._poison()
            if cancelled or cancel.is_set():
                return
            for token in record["token_ids"]:
                if cancel.is_set():
                    return
                yield token
            if record["finish_reason"] == "stop" and not cancel.is_set():
                yield record["stop_token"]

    def close(self):
        if self._closed:
            return
        if not self._lock.acquire(timeout=self._deadline):
            raise self._died("resident request still active; operator inspection required")
        try:
            if self.alive():
                try:
                    self._send("quit")
                    if self._receive() != "HETERO_CLOSED":
                        raise self._died("resident shutdown not acknowledged")
                    print("HETERO_CLOSED", file=sys.stderr, flush=True)
                except BaseException:
                    self._poison()
                    self._wait_exit()
                    raise
            self._poison()
            code = self._wait_exit()
            self._closed = True
            if code:
                raise self._died(f"native resident coordinator exited with status {code}")
        finally:
            self._lock.release()
