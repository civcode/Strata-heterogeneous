"""CPU-only resident protocol tests; no native GPU workers or model files."""
import threading
import unittest
from threading import Event
from unittest.mock import Mock, patch
from serve.heterogeneous_contract import generation_limits, validate_request, Progress, timing_record
from serve.heterogeneous import ResidentEngine


class TestLimits(unittest.TestCase):
    def test_auto_and_context(self):
        self.assertEqual(generation_limits(8192, 4, 250000, 4, 2560, 8192, 16, 0), (8192, 4))
        self.assertEqual(generation_limits(2048, 8, 250000, 4, 2560, 8192, 16, 0), (2048, 8))
        self.assertEqual(generation_limits(8192, 8, 1000000, 4, 2560, 8192, 16, 0), (8192, 4))
        for requested, chunk in [(8193, 1), (4096, 5), (4096, -1)]:
            with self.assertRaises(ValueError):
                generation_limits(8192, 4, 250000, 4, 2560, requested, 16, chunk)

    def test_request_bounds(self):
        validate_request([249999] * 8191, 1, 250000, 8192)
        for ids, vocab in [([1] * 8192, 250000), ([999999999] * 8191, 1000000000), ([True], 3)]:
            with self.assertRaises(ValueError):
                validate_request(ids, 1, vocab, 8192)

    def test_progress_and_timing(self):
        p = Progress(1, 3, 3, 2, 100, {99}, True)
        self.assertIsNone(p.accept("HETERO_PROGRESS 1 1 2 -1"))
        self.assertEqual(p.accept("HETERO_PROGRESS 1 2 3 10"), 10)
        self.assertEqual(p.accept("HETERO_PROGRESS 1 3 4 11"), 11)
        self.assertIsNone(p.accept("HETERO_PROGRESS 1 4 5 -1"))
        self.assertEqual(p.tokens, [10, 11])
        self.assertTrue(p.terminal)
        with self.assertRaises(ValueError):
            p.accept("HETERO_PROGRESS 1 5 6 12")
        m = timing_record(1, 5, 2, 4, 2, 3, 5, 2)
        self.assertEqual(m["prompt_tokens_per_second"], 3)
        self.assertEqual(m["decode_tokens_per_second"], 1)


class TestResident(unittest.TestCase):
    def engine(self, records, stream=True):
        e = object.__new__(ResidentEngine)
        e.source = {"vocab": 100}
        e.stop_ids = {99}
        e.max_context, e.max_new, e.chunk = 8192, 3, 2
        e.stream_tokens = stream
        e._died = RuntimeError
        e._lock = threading.Lock()
        e._request = 0
        e._closed = e._failed = False
        e.proc = Mock()
        e.proc.poll.return_value = None
        e._receive = Mock(side_effect=records)
        e._send = Mock()
        return e

    def test_stream_before_final(self):
        e = self.engine(["HETERO_PROGRESS 1 1 2 10", "HETERO_PROGRESS 1 2 3 11",
                         "HETERO_PROGRESS 1 3 4 -1",
                         'HETERO_GENERATION {"request_id":1,"prompt_tokens":2,"token_ids":[10,11],"finish_reason":"stop","stop_token":99}'])
        g = e.generate([1, 2], 3, {}, Event())
        self.assertEqual(next(g), 10)
        self.assertEqual(e._receive.call_count, 1)
        self.assertEqual(list(g), [11, None, 99])
        self.assertTrue(e.alive())
        self.assertEqual(e.last_completion["token_ids"], [10, 11])

    def test_buffered_compatibility(self):
        e = self.engine(["HETERO_PROGRESS 1 1 2", "HETERO_PROGRESS 1 2 3",
                         'HETERO_GENERATION {"request_id":1,"prompt_tokens":2,"token_ids":[10,11],"finish_reason":"length","stop_token":-1}'], False)
        self.assertEqual(list(e.generate([1, 2], 2, {}, Event())), [None, None, 10, 11])

    def test_length_last_token_after_validation(self):
        e = self.engine(["HETERO_PROGRESS 1 1 2 10", "HETERO_PROGRESS 1 2 3 11",
                         'HETERO_GENERATION {"request_id":1,"prompt_tokens":2,"token_ids":[10,11],"finish_reason":"length","stop_token":-1}'])
        self.assertEqual(list(e.generate([1, 2], 2, {}, Event())), [10, None, 11])
        self.assertEqual(e.last_completion["finish_reason"], "length")

    def test_cancel_after_streamed_prefix(self):
        e = self.engine(["HETERO_PROGRESS 1 1 2 10",
                         'HETERO_GENERATION {"request_id":1,"prompt_tokens":2,"token_ids":[],"finish_reason":"cancelled","stop_token":-1}'])
        g = e.generate([1, 2], 3, {}, Event())
        self.assertEqual(next(g), 10)
        g.close()
        e._send.assert_called_with("cancel")
        self.assertTrue(e.alive())
        self.assertEqual(e.last_completion["finish_reason"], "cancelled")

    def test_mismatch_poisons(self):
        e = self.engine(["HETERO_PROGRESS 1 1 2 10", "HETERO_PROGRESS 1 2 3 11",
                         'HETERO_GENERATION {"request_id":1,"prompt_tokens":2,"token_ids":[12,11],"finish_reason":"length","stop_token":-1}'])
        g = e.generate([1, 2], 2, {}, Event())
        self.assertEqual(next(g), 10)
        self.assertIsNone(next(g))
        with self.assertRaises(RuntimeError):
            next(g)
        self.assertFalse(e.alive())

    def test_restart_refuses_live_group(self):
        e = self.engine([])
        e._restart_clean = True
        e._failed = True
        e.proc.pid = 123
        e._reader = Mock()
        e._reader.is_alive.return_value = False
        e._wait_exit = Mock(return_value=1)
        e._deadline = 1
        e._launch = Mock()
        with patch("serve.heterogeneous.os.killpg", return_value=None):
            with self.assertRaises(RuntimeError):
                e.restart()
        e._launch.assert_not_called()
        with patch("serve.heterogeneous.os.killpg", side_effect=ProcessLookupError):
            e.restart()
        e._launch.assert_called_once()


if __name__ == "__main__":
    unittest.main()
