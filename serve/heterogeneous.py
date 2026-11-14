"""Single-request-at-a-time adapter to the native resident coordinator.

The coordinator owns model loading, request IDs, generation and GPU commits;
this module only transports commands and implements the existing Engine
generator interface. No alternative inference or HTTP implementation.
"""
from __future__ import annotations

import argparse
import json
import os
import time
import queue
import subprocess
import sys
import threading

from serve.heterogeneous_contract import validate_request, Progress, timing_record


def prepare_server(cfg):
    """Validate explicit local configuration and reuse CLI source binding."""
    from strata_cli.heterogeneous import prepare_native, model_template, _native
    opts = cfg.get("heterogeneous")
    if not isinstance(opts, dict):
        raise ValueError("heterogeneous engine needs a heterogeneous config object")
    required = {"coordinator", "cuda_worker", "hip_worker", "config", "plan", "fingerprint"}
    optional = {"layers", "hc", "n_embd", "max_new_tokens", "prompt_chunk", "timeout_ms", "shared",
                "max_context", "stream_tokens", "restart_clean", "lookup_draft"}
    if not required <= opts.keys() or opts.keys() - required - optional:
        raise ValueError("missing or unknown heterogeneous config keys")
    for key in ("coordinator", "cuda_worker", "hip_worker", "plan"):
        if not isinstance(opts[key], str) or not opts[key]:
            raise ValueError(f"heterogeneous.{key} must be a nonempty string")
    if not isinstance(opts["config"], list) or not opts["config"] or not all(
            isinstance(x, str) and x for x in opts["config"]):
        raise ValueError("heterogeneous.config must list ordered worker config paths")
    values = dict(layers=48, hc=4, n_embd=2560, max_new_tokens=32,
                  prompt_chunk=1, timeout_ms=120000, shared=False, max_context=4096,
                  stream_tokens=False, restart_clean=False, lookup_draft=False)
    values.update(opts)
    for key in ("fingerprint", "layers", "hc", "n_embd", "max_new_tokens", "timeout_ms", "max_context"):
        if type(values[key]) is not int or values[key] <= 0:
            raise ValueError(f"heterogeneous.{key} must be a positive integer")
    if type(values["prompt_chunk"]) is not int or not 0 <= values["prompt_chunk"] <= 8:
        raise ValueError("heterogeneous.prompt_chunk must be 0..8")
    for key in ("shared", "stream_tokens", "restart_clean", "lookup_draft"):
        if type(values[key]) is not bool:
            raise ValueError(f"heterogeneous.{key} must be boolean")
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
        self.max_context = min(source["max_cells"], source.get("context_limit", 4096))
        self.stream_tokens = source.get("stream_tokens", False)
        self.lookup_draft = source.get("lookup_draft", False)
        if self.lookup_draft and (self.stream_tokens or chunk < 2):
            raise ValueError("lookup-draft requires buffered generation and chunk >= 2")
        self._restart_clean = source.get("restart_clean", False)
        self._argv = tuple(argv)
        self.last_metrics = None
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
        self._launch()

    def _launch(self):
        self._records = queue.Queue()
        self._request = 0
        self._failed = False
        self.last_completion = self.last_metrics = None
        self.proc = subprocess.Popen([*self._argv, "--run"], stdin=subprocess.PIPE,
                                     stdout=subprocess.PIPE, stderr=None, text=True,
                                     encoding="utf-8", errors="replace", bufsize=1,
                                     start_new_session=True)
        self._reader = threading.Thread(target=self._read_records, daemon=True,
                                        name="heterogeneous-native-output")
        self._reader.start()
        try:
            expected = ("HETERO_READY 1 STREAM" if self.stream_tokens else
                        "HETERO_READY 1 LOOKUP" if self.lookup_draft else "HETERO_READY 1")
            if self._receive() != expected:
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
        # Opt-in only; never replay a failed request or overlap worker sets.
        # Native children inherit the coordinator's new process group.
        if not self._restart_clean or self._closed:
            raise self._died("heterogeneous resident engine stopped; restart the server explicitly")
        with self._lock:
            if self.alive():
                return
            self._poison()
            self._wait_exit()
            self._reader.join(timeout=self._deadline)
            if self._reader.is_alive():
                raise self._died("old coordinator reader still active; restart refused")
            try:
                os.killpg(self.proc.pid, 0)  # existence check only, no signal delivery
            except ProcessLookupError:
                pass
            except OSError as exc:
                raise self._died("cannot verify old process group exit; restart refused") from exc
            else:
                raise self._died("old process group still exists; restart refused")
            self.proc.stdout.close()
            self._launch()

    def exit_code(self):
        return self.proc.poll()

    def _validate(self, ids, max_new, sampling, embeddings):
        if embeddings is not None or type(max_new) is not int or not 1 <= max_new <= self.max_new:
            raise ValueError("unsupported resident request or completion budget")
        validate_request(ids, max_new, self.source["vocab"], self.max_context)
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
            self.last_completion = self.last_metrics = None
            waiting = completed = cancelled = False
            progress = Progress(self._request, len(ids), max_new, self.chunk,
                                self.source["vocab"], self.stop_ids, self.stream_tokens,
                                 self.lookup_draft)
            start = time.perf_counter()
            prefill_end = last_checkpoint = first_output = None
            emitted = 0
            try:
                self._send(" ".join(map(str, [max_new, *ids])))
                while True:
                    line = self._receive()
                    if line.startswith("HETERO_PROGRESS "):
                        try:
                            if cancelled:
                                raise ValueError("progress after cancellation")
                            token = progress.accept(line)
                        except ValueError as exc:
                            raise self._died("invalid resident progress") from exc
                        last_checkpoint = time.perf_counter()
                        if progress.pos >= len(ids) and prefill_end is None:
                            prefill_end = last_checkpoint
                        waiting = True
                        if not cancel.is_set():
                            # Service may close at the output cap: validate the
                            # final record before releasing the last capped token.
                            if token is not None and len(progress.tokens) < max_new:
                                if first_output is None:
                                    first_output = time.perf_counter()
                                emitted += 1
                                yield token
                            else:
                                yield None
                        cancelled = cancel.is_set()
                        self._send("cancel" if cancelled else "continue")
                        waiting = False
                    else:
                        record = self._result(line, ids, max_new)
                        reason = record["finish_reason"]
                        expected = len(ids) + len(record["token_ids"]) - (reason == "length")
                        if (cancelled != (reason == "cancelled") or
                                (not cancelled and progress.pos != expected) or
                                (self.stream_tokens and not cancelled and
                                 (progress.tokens != record["token_ids"] or
                                  progress.terminal != (reason == "stop")))):
                            raise self._died("resident completion disagrees with committed positions or streamed tokens")
                        self.last_completion = record
                        end = time.perf_counter()
                        if first_output is None and record["token_ids"]:
                            first_output = end
                        self.last_metrics = timing_record(start, end, prefill_end, last_checkpoint,
                                                          first_output, len(ids), progress.pos,
                                                          len(record["token_ids"]))
                        print(f"HETERO_REQUEST_DONE request={self._request} finish={reason} next={progress.pos}",
                              file=sys.stderr, flush=True)
                        completed = True
                        break
            except GeneratorExit:
                # Drain cancellation at a committed checkpoint before unlocking.
                if waiting:
                    try:
                        self._send("cancel")
                        record = self._result(self._receive(), ids, max_new)
                        if record["finish_reason"] != "cancelled":
                            raise self._died("resident cancellation not acknowledged")
                        self.last_completion = record
                        print(f"HETERO_REQUEST_DONE request={self._request} finish=cancelled next={progress.pos}",
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
            for token in record["token_ids"][emitted:]:
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
            self._reader.join(timeout=self._deadline)
            if self._reader.is_alive():
                raise self._died("native output reader did not exit; operator inspection required")
            self.proc.stdout.close()
            self._closed = True
            if code:
                raise self._died(f"native resident coordinator exited with status {code}")
        finally:
            self._lock.release()
