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

    def test_workload_limits(self):
        self.assertEqual(generation_limits(32768, 8, 248320, 4, 2560, 32768, 4096, 0, True),
                         (32768, 8))
        for cells, context, cap, mode in [(8192, 16384, 256, True),
                                          (32768, 32769, 256, True),
                                          (32768, 32768, 4097, True),
                                          (32768, 8192, 257, False),
                                          (32768, 16384, 256, False)]:
            with self.assertRaises(ValueError):
                generation_limits(cells, 8, 248320, 4, 2560, context, cap, 1, mode)
        validate_request([248319] * 28672, 4096, 248320, 32768)
        with self.assertRaises(ValueError):
            validate_request([248319] * 28673, 4096, 248320, 32768)

    def test_long_context_limits(self):
        self.assertEqual(generation_limits(262144, 8, 248320, 4, 2560, 262144, 1, 0, True, True),
                         (262144, 8))
        ids = [248319] * 262143
        validate_request(ids, 1, 248320, 262144, True)
        with self.assertRaises(ValueError):
            validate_request(ids, 2, 248320, 262144, True)
        with self.assertRaises(ValueError):
            validate_request(ids, 1, 248320, 262144)
        for cells, context, workload, extended in [(262144, 262144, True, False),
                                                   (262144, 262144, False, True),
                                                   (262144, 131072, True, True),
                                                   (262145, 262145, True, True)]:
            with self.assertRaises(ValueError):
                generation_limits(cells, 8, 248320, 4, 2560, context, 1, 1, workload, extended)
        p = Progress(1, 262143, 2, 8, 248320, {99}, True, native_prefill=True)
        self.assertIsNone(p.accept("HETERO_PROGRESS 1 1 262142 -1"))
        self.assertEqual(p.accept("HETERO_PROGRESS 1 2 262143 10"), 10)
        self.assertEqual(p.accept("HETERO_PROGRESS 1 3 262144 11"), 11)

    def test_request_bounds(self):
        validate_request([249999] * 8191, 1, 250000, 8192)
        for ids, vocab in [([1] * 8192, 250000), ([999999999] * 8191, 1000000000), ([True], 3)]:
            with self.assertRaises(ValueError):
                validate_request(ids, 1, vocab, 8192)

    def test_lookup_progress_bounds(self):
        p = Progress(1, 3, 8, 4, 100, {99}, False, True)
        self.assertIsNone(p.accept("HETERO_PROGRESS 1 1 3"))
        self.assertIsNone(p.accept("HETERO_PROGRESS 1 2 7"))
        with self.assertRaises(ValueError):
            p.accept("HETERO_PROGRESS 1 3 12")
        with self.assertRaises(ValueError):
            p.accept("HETERO_PROGRESS 1 3 7")
        self.assertIsNone(p.accept("HETERO_PROGRESS 1 3 10"))
        with self.assertRaises(ValueError):
            Progress(1, 3, 8, 4, 100, {99}, True, True)

    def test_mtp_buffered_progress_bounds(self):
        p = Progress(1, 3, 8, 2, 100, {99}, False, mtp_spec=4)
        self.assertIsNone(p.accept("HETERO_PROGRESS 1 1 2"))
        self.assertIsNone(p.accept("HETERO_PROGRESS 1 2 3"))
        for bad in ("HETERO_PROGRESS 1 3 9", "HETERO_PROGRESS 2 3 4", "HETERO_PROGRESS 1 3 3"):
            with self.assertRaises(ValueError):
                p.accept(bad)
        self.assertIsNone(p.accept("HETERO_PROGRESS 1 3 8"))
        self.assertIsNone(p.accept("HETERO_PROGRESS 1 4 10"))
        with self.assertRaises(ValueError):
            p.accept("HETERO_PROGRESS 1 5 11")
        with self.assertRaises(ValueError):
            Progress(1, 3, 8, 2, 100, {99}, False, lookup=True, mtp_spec=4)
        # Bulk prefill remains within the exact 2K resident capacity.
        Progress(1, 2040, 8, 2, 100, {99}, False, native_prefill=True, mtp_spec=4)
        Progress(1, 8184, 8, 2, 100, {99}, False, native_prefill=True, mtp_spec=4, mtp_context=8192)
        with self.assertRaises(ValueError):
            Progress(1, 2041, 8, 2, 100, {99}, False, native_prefill=True, mtp_spec=4)

    def test_mtp_stream_progress(self):
        p = Progress(1, 3, 8, 2, 100, {99}, True, mtp_spec=4)
        self.assertEqual(p.accept("HETERO_PROGRESS 1 1 2 BATCH 0 0"), ())
        self.assertEqual(p.accept("HETERO_PROGRESS 1 2 3 BATCH 0 1 10"), (10,))
        for bad in ("HETERO_PROGRESS 1 3 9 BATCH 0 5 11 12 13 14 15",
                    "HETERO_PROGRESS 1 3 7 BATCH 0 3 11 12 13",
                    "HETERO_PROGRESS 1 3 7 BATCH 0 4 11 12 13 99",
                    "HETERO_PROGRESS 1 3 7 BATCH 0 4 11 12 13 100",
                    "HETERO_PROGRESS 1 3 7 11"):
            with self.assertRaises(ValueError):
                p.accept(bad)
        self.assertEqual(p.accept("HETERO_PROGRESS 1 3 7 BATCH 0 4 11 12 13 14"), (11,12,13,14))
        self.assertEqual(p.accept("HETERO_PROGRESS 1 4 9 BATCH 1 1 15"), (15,))
        self.assertTrue(p.terminal)
        self.assertEqual(p.tokens, [10,11,12,13,14,15])
        with self.assertRaises(ValueError):
            p.accept("HETERO_PROGRESS 1 5 10 BATCH 0 2 16 17")

    def test_native_prefill_progress_exact_prefix(self):
        p = Progress(1, 17, 3, 8, 100, {99}, True, native_prefill=True)
        for bad in ("HETERO_PROGRESS 1 1 8 -1", "HETERO_PROGRESS 1 1 17 -1",
                    "HETERO_PROGRESS 2 1 16 -1"):
            with self.assertRaises(ValueError):
                p.accept(bad)
        self.assertIsNone(p.accept("HETERO_PROGRESS 1 1 16 -1"))
        self.assertEqual(p.accept("HETERO_PROGRESS 1 2 17 10"), 10)
        self.assertEqual(p.accept("HETERO_PROGRESS 1 3 18 11"), 11)
        p = Progress(1, 1, 3, 8, 100, {99}, False, native_prefill=True)
        self.assertIsNone(p.accept("HETERO_PROGRESS 1 1 1"))

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
        e.lookup_draft = False
        e._died = RuntimeError
        e._lock = threading.Lock()
        e._request = 0
        e._closed = e._failed = False
        e.proc = Mock()
        e.proc.poll.return_value = None
        e._receive = Mock(side_effect=records)
        e._send = Mock()
        return e

    def test_tool_declarations_are_service_metadata(self):
        e = self.engine([])
        e._validate([1], 1, {"tools": [{"type": "function"}], "tool_choice": "auto"}, None)
        e._validate([1], 1, {"tool_choice": "none"}, None)
        for choice in ("required", {"type": "function", "function": {"name": "f"}}):
            with self.assertRaises(ValueError):
                e._validate([1], 1, {"tool_choice": choice}, None)
        with self.assertRaises(ValueError):
            e._validate([1], 1, {"functions": [{"name": "f"}]}, None)

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

    def test_workload_server_timings(self):
        for stream in (True, False):
            progress = ["HETERO_PROGRESS 1 1 2", "HETERO_PROGRESS 1 2 3"]
            if stream:
                progress = [progress[0]+" 10", progress[1]+" 11"]
            e = self.engine(progress+[
                'HETERO_GENERATION {"request_id":1,"prompt_tokens":2,"token_ids":[10,11],"finish_reason":"length","stop_token":-1}'], stream)
            e.source["workload_mode"] = True
            clock = [1.0, 2.0, 2.1, 3.0, 4.0] if stream else [1.0, 2.0, 3.0, 4.0]
            with patch("serve.heterogeneous.time.monotonic", side_effect=clock):
                list(e.generate([1,2], 2, {}, Event()))
            self.assertEqual(e.last["prompt_read"], 2)
            self.assertEqual(e.last["generated"], 2)
            self.assertEqual(e.last["prompt_ms"], 1000)
            self.assertEqual(e.last["decode_ms"], 1000 if stream else None)
            self.assertEqual(e.last["timing_source"], "heterogeneous_host_commit_progress")

    def test_buffered_compatibility(self):
        e = self.engine(["HETERO_PROGRESS 1 1 2", "HETERO_PROGRESS 1 2 3",
                         'HETERO_GENERATION {"request_id":1,"prompt_tokens":2,"token_ids":[10,11],"finish_reason":"length","stop_token":-1}'], False)
        self.assertEqual(list(e.generate([1, 2], 2, {}, Event())), [None, None, 10, 11])

    def test_lookup_buffered_completion(self):
        e = self.engine(["HETERO_PROGRESS 1 1 2", "HETERO_PROGRESS 1 2 4",
                         'HETERO_GENERATION {"request_id":1,"prompt_tokens":2,"token_ids":[10,11,12],"finish_reason":"length","stop_token":-1}'], False)
        e.lookup_draft = True
        self.assertEqual(list(e.generate([1,2], 3, {}, Event())), [None,None,10,11,12])
        self.assertEqual(e.last_metrics["decode_feedback_tokens"], 2)
        bounds = e.last_metrics["phase_monotonic_s"]
        self.assertLessEqual(bounds["request_start"], bounds["prefill_end"])
        self.assertLessEqual(bounds["prefill_end"], bounds["decode_end"])
        self.assertLessEqual(bounds["decode_end"], bounds["request_end"])

    def test_mtp_buffered_completion(self):
        e = self.engine(["HETERO_PROGRESS 1 1 2", "HETERO_PROGRESS 1 2 4",
                         'HETERO_GENERATION {"request_id":1,"prompt_tokens":2,"token_ids":[10,11,12],"finish_reason":"length","stop_token":-1}'], False)
        e.max_context = 2048
        e.source.update(mtp="/local/rt", spec=4, spec_min_p=0.5)
        self.assertEqual(list(e.generate([1,2],3,{},Event())), [None,None,10,11,12])
        self.assertEqual(e.last_metrics["decode_feedback_tokens"],2)

    def test_mtp_streamed_batch_and_cap(self):
        e = self.engine(["HETERO_PROGRESS 1 1 2 BATCH 0 1 10",
                         "HETERO_PROGRESS 1 2 4 BATCH 0 2 11 12",
                         'HETERO_GENERATION {"request_id":1,"prompt_tokens":2,"token_ids":[10,11,12],"finish_reason":"length","stop_token":-1}'])
        e.max_context = 2048
        e.source.update(mtp="/local/rt", spec=4, spec_min_p=0.5)
        g = e.generate([1,2],3,{},Event())
        self.assertEqual(next(g),10)
        self.assertEqual(next(g),11)
        self.assertEqual(e._receive.call_count,2)
        self.assertEqual(next(g),12)
        self.assertEqual(e._receive.call_count,3)
        self.assertEqual(list(g),[])

    def test_mtp_stream_mid_batch_close(self):
        e = self.engine(["HETERO_PROGRESS 1 1 2 BATCH 0 1 10",
                         "HETERO_PROGRESS 1 2 4 BATCH 0 2 11 12",
                         'HETERO_GENERATION {"request_id":1,"prompt_tokens":2,"token_ids":[],"finish_reason":"cancelled","stop_token":-1}',
                         "HETERO_PROGRESS 2 1 1 BATCH 0 1 20",
                         'HETERO_GENERATION {"request_id":2,"prompt_tokens":1,"token_ids":[20],"finish_reason":"length","stop_token":-1}'])
        e.max_context = 2048
        e.source.update(mtp="/local/rt", spec=4, spec_min_p=0.5)
        e.max_new=4
        g=e.generate([1,2],4,{},Event())
        self.assertEqual(next(g),10)
        self.assertEqual(next(g),11)
        g.close()
        self.assertEqual(e.last_completion["finish_reason"],"cancelled")
        e._send.assert_called_with("cancel")
        self.assertEqual(list(e.generate([1],1,{},Event())),[None,20])

    def test_mtp_stream_stop(self):
        e = self.engine(["HETERO_PROGRESS 1 1 2 BATCH 0 1 10",
                         "HETERO_PROGRESS 1 2 4 BATCH 1 1 11",
                         'HETERO_GENERATION {"request_id":1,"prompt_tokens":2,"token_ids":[10,11],"finish_reason":"stop","stop_token":99}'])
        e.source.update(mtp="/local/rt",spec=4,spec_min_p=0.5)
        self.assertEqual(list(e.generate([1,2],3,{},Event())),[10,11,99])

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
