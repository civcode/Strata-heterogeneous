"""CPU-only HTTP disconnect tests with the actual resident adapter and Service.

The synthetic native transport uses explicit events, not token generation
speed or socket-buffer saturation, to put cancellation at a known checkpoint.
"""
import json
import socket
import threading
import time
import unittest
import urllib.request
from pathlib import Path
from unittest.mock import Mock

from serve.frontend import ChatTemplate
from serve.heterogeneous import ResidentEngine
from serve.server import ByteTokenizer, EngineDied, Service, openai_chunks, serve

ROOT = Path(__file__).resolve().parents[1]


class ScriptedResident(ResidentEngine):
    def __init__(self):
        # Real generate()/Progress()/cancellation logic; no worker or GPU.
        self.source = dict(vocab=262, max_cells=262144, context_limit=262144,
                           mtp="/fixture", spec=4, spec_min_p=0.5,
                           native_prefill_chunk=32, workload_mode=True, long_context=True)
        self.max_context, self.max_new, self.chunk = 262144, 16, 8
        self.stop_ids = {257, 258}
        self.stream_tokens, self.lookup_draft = True, False
        self.info = {}
        self.last = self.last_completion = self.last_metrics = None
        self._died = EngineDied
        self._lock = threading.Lock()
        self._request = 0
        self._closed = self._failed = self._restart_clean = False
        self._deadline = 10
        self.proc = Mock()
        self.proc.poll.return_value = None
        self.entered = threading.Event()
        self.release_checkpoint = threading.Event()
        self.cancel_sent = threading.Event()
        self.release_ack = threading.Event()
        self.commands = []
        self.phase = 0

    def _send(self, command):
        self.commands.append((self._request, command))
        if command not in ("continue", "cancel"):
            fields = command.split()
            self.prompt = len(fields) - 1
            self.budget = int(fields[0])
            self.phase = 0
        elif command == "cancel":
            self.cancel_sent.set()

    def _receive(self, timeout=None):
        request, n = self._request, self.prompt
        self.phase += 1
        if request == 1 and self.phase == 1:
            self.entered.set()
            if not self.release_checkpoint.wait(10):
                raise EngineDied("test checkpoint gate timed out")
        if self.phase == 1:
            return f"HETERO_PROGRESS {request} 1 {n-1} BATCH 0 0"
        if request == 1 and self.phase == 2:
            if not self.cancel_sent.is_set() or not self.release_ack.wait(10):
                raise EngineDied("test cancellation ACK gate timed out")
            record = dict(request_id=request, prompt_tokens=n, token_ids=[],
                          finish_reason="cancelled", stop_token=-1)
        elif request == 2 and self.phase == 2:
            return f"HETERO_PROGRESS {request} 2 {n} BATCH 0 1 89"
        elif request == 2 and self.phase == 3:
            record = dict(request_id=request, prompt_tokens=n, token_ids=[89],
                          finish_reason="length", stop_token=-1)
        else:
            raise EngineDied("unexpected fixture receive")
        return "HETERO_GENERATION " + json.dumps(record)


class TestResidentHTTPDisconnect(unittest.TestCase):
    def wait_for(self, predicate, reason):
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            if predicate():
                return
            time.sleep(0.01)
        self.fail(reason)

    def test_disconnect_before_bulk_checkpoint_then_queued_recovery(self):
        engine = ScriptedResident()
        tok = ByteTokenizer()
        svc = Service(engine, tok, ChatTemplate(ROOT / "serve/chat_template.jinja"))
        # The fixture needs real HTTP but no hardware telemetry.
        svc.start_telemetry = lambda: None
        httpd = serve(svc, port=0)
        port = httpd.server_address[1]
        base = f"http://127.0.0.1:{port}/v1/chat/completions"

        def body(budget):
            return json.dumps(dict(model="x", messages=[dict(role="user", content="hi")],
                                   stream=True, max_tokens=budget, temperature=0)).encode()

        sock = None
        waiter = None
        replies, errors = [], []
        try:
            sock = socket.create_connection(("127.0.0.1", port), timeout=5)
            payload = body(16)
            sock.sendall((f"POST /v1/chat/completions HTTP/1.1\r\nHost: 127.0.0.1:{port}\r\n"
                          f"Content-Type: application/json\r\nContent-Length: {len(payload)}\r\n\r\n").encode() + payload)
            self.assertTrue(engine.entered.wait(5), "native request never entered bulk receive")
            # Actual TCP disconnect while no checkpoint is available yet.
            sock.shutdown(socket.SHUT_RDWR)
            sock.close()
            sock = None
            self.wait_for(lambda: svc.client_disconnect_cancels == 1,
                          "HTTP watcher did not cancel active resident request")
            engine.release_checkpoint.set()
            self.assertTrue(engine.cancel_sent.wait(5), "resident adapter did not send cancel at checkpoint")
            self.assertIsNone(engine.last_completion, "cancel ACK has not arrived")

            def recover():
                try:
                    req = urllib.request.Request(base, data=body(1),
                                                  headers={"Content-Type": "application/json"})
                    with urllib.request.urlopen(req, timeout=15) as response:
                        replies.append((response.status, response.read()))
                except Exception as exc:
                    errors.append(exc)

            waiter = threading.Thread(target=recover, daemon=True)
            waiter.start()
            self.wait_for(lambda: svc.status.get("queued", 0) == 1,
                          "second HTTP request did not queue behind cancellation drain")
            self.assertEqual([r for r, _ in engine.commands], [1, 1])
            self.assertTrue(engine._lock.locked())
            engine.release_ack.set()
            waiter.join(10)
            self.assertFalse(waiter.is_alive())
            self.assertEqual(errors, [])
            self.assertEqual(len(replies), 1)
            status, wire = replies[0]
            self.assertEqual(status, 200)
            self.assertIn(b"data: [DONE]", wire)
            self.assertNotIn(b'"error":', wire)
            self.assertEqual(engine.last_completion["request_id"], 2)
            self.assertEqual(engine.last_completion["token_ids"], [89])
            self.assertTrue(engine.alive())
            self.wait_for(lambda: svc.admission_reserved == 0 and not svc.status.get("busy"),
                          "HTTP inference reservation was not released")
        finally:
            engine.release_checkpoint.set()
            engine.release_ack.set()
            if sock is not None:
                sock.close()
            if waiter is not None:
                waiter.join(15)
            httpd.shutdown()
            httpd.server_close()

    def test_openai_encoder_closes_retained_service_iterator(self):
        closed = threading.Event()
        cancel = threading.Event()
        svc = Mock()
        svc.model_for.return_value = "fixture"
        svc.engine.gen = 1

        def source():
            try:
                yield "ping", None
            finally:
                closed.set()

        retained = source()
        chunks = openai_chunks(svc, {}, [1], False, None, 8, cancel, run=retained)
        next(chunks)  # initial assistant role
        self.assertIsNone(next(chunks))
        chunks.close()
        self.assertTrue(closed.is_set(), "retaining source must not defer cancellation/drain")


if __name__ == "__main__":
    unittest.main()
