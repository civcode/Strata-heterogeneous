import unittest
from tools.heterogeneous_critical_path import summarize, union, verify_benchmark


def span(component, phase, stage, a, b, window=1, rows=2):
    return (f"HETERO_PERF component={component} phase={phase} stage={stage} request=1 "
            f"window={window} rows={rows} us=0 start_ns={a} end_ns={b}")


def trace(window=1, offset=0, phase="prefill", rows=2, pos=0):
    return [span("coordinator", name, stage, a+offset, b+offset, window, rows)
            for name, stage, a, b in [("stage_roundtrip", 0, 0, 100),
                ("stage_roundtrip", 1, 110, 250), ("commit_barrier", 2, 260, 300)]] + [
            span("worker", "enqueue_execute", 0, 10+offset, 60+offset, window, rows),
            span("worker", "wait_execute", 0, 50+offset, 80+offset, window, rows),
            span("worker", "final_result_send", 1, 240+offset, 255+offset, window, rows),
            f"HETERO_PERF_COMMITTED request=1 window={window} phase={phase} rows={rows} keep={rows} pos0={pos}"]


class CriticalPathTests(unittest.TestCase):
    def test_union_and_clipping(self):
        self.assertEqual(union([(0, 5), (3, 10), (12, 13)]), 11)
        r = summarize(reversed(trace()), 2)
        w = r["windows"][0]
        self.assertEqual(w["critical_ns"], 300)
        self.assertEqual(w["dispatch_gap_ns"], 20)
        self.assertEqual(w["stages"][0]["worker_covered_ns"], 70)
        self.assertEqual(w["stages"][1]["worker_clipped_spans"], 1)
        self.assertEqual(w["stages"][1]["worker_covered_ns"], 10)

    def test_phase_and_gap(self):
        r = summarize(trace()+trace(2, 350, "decode", 1, 2)+trace(3, 700, "decode", 1, 3), 2)
        self.assertIsNone(r["windows"][1]["inter_window_gap_ns"])
        self.assertEqual(r["windows"][2]["inter_window_gap_ns"], 50)
        self.assertEqual(r["summaries"]["decode"]["accepted_rows"], 2)

    def test_partial_and_uncommitted(self):
        data = trace()[1:]+trace(2, 350, "decode", 1, 2)[:-1]
        r = summarize(data, 2)
        self.assertEqual(r["complete_windows"], 0)
        self.assertEqual(len(r["incomplete"]), 1)
        self.assertEqual(r["uncommitted"], [[1, 2]])

    def test_reject_duplicate_and_no_commit(self):
        for data in (trace()+trace()[:1], trace()[:-1]):
            with self.assertRaises(ValueError):
                summarize(data, 2)

    def test_benchmark_coverage(self):
        r = summarize(trace()+trace(2, 350, "decode", 1, 2), 2)
        b = {"status": "complete", "completions": [{"request_id": 1,
             "finish_reason": "length", "prompt_tokens": 2,
             "timing": {"decode_feedback_tokens": 1, "phase_monotonic_s": {
                 "request_start": 0, "prefill_end": 330e-9, "decode_end": 700e-9}}}]}
        self.assertTrue(verify_benchmark(r, b)["matched"])
        self.assertEqual(r["summaries"]["prefill"]["stages"][0]["worker_phase_ns"]["enqueue_execute"]["n"], 1)
        b["completions"][0]["timing"]["decode_feedback_tokens"] = 2
        self.assertFalse(verify_benchmark(r, b)["matched"])
        b["completions"][0]["timing"]["decode_feedback_tokens"] = 1
        b["completions"][0]["timing"]["phase_monotonic_s"]["decode_end"] = 500e-9
        self.assertFalse(verify_benchmark(r, b)["matched"])

    def test_grouped_endpoints_and_untimed_native_prefix(self):
        data = [s.replace("stage=1 ", "stage=2 ") for s in trace(2, 0, "prefill", 1, 3)]
        data += ["HETERO_NATIVE_PREFILL_COMMITTED request=1 rows=3 chunk=8"]
        r = summarize(data, 2, [0, 2])
        self.assertEqual([s["stage"] for s in r["windows"][0]["stages"]], [0, 2])
        self.assertNotIn("commit_barrier", r["windows"][0]["stages"][1]["coordinator_phase_ns"])
        b = {"status": "complete", "completions": [{"request_id": 1, "finish_reason": "length",
             "prompt_tokens": 4, "timing": {"decode_feedback_tokens": 0, "phase_monotonic_s": {
                 "request_start": 0, "prefill_end": 400e-9, "decode_end": 400e-9}}}]}
        coverage = verify_benchmark(r, b)
        self.assertTrue(coverage["position_counts_match"])
        self.assertFalse(coverage["matched"])
        self.assertEqual(coverage["untimed_native_prefill_requests"], [1])
        with self.assertRaises(ValueError):
            summarize(data, 2, [2, 0])

    def test_timed_prefix_and_orphan(self):
        data = trace(2, 400, "prefill", 1, 3)
        data += ["HETERO_NATIVE_PREFILL_COMMITTED request=1 rows=3 chunk=8",
                 "HETERO_NATIVE_PREFILL_TIMING request=1 rows=3 start_ns=100 end_ns=350"]
        b = {"status": "complete", "completions": [{"request_id": 1, "finish_reason": "length",
             "prompt_tokens": 4, "timing": {"decode_feedback_tokens": 0, "phase_monotonic_s": {
                 "request_start": 0, "prefill_end": 800e-9, "decode_end": 800e-9}}}]}
        result = summarize(data, 2)
        self.assertTrue(verify_benchmark(result, b)["matched"])
        self.assertEqual(result["native_prefill_roundtrip"]["n"], 1)
        self.assertEqual(result["native_prefill_markers"][0]["roundtrip_ns"], 250)
        overlap = data[:-1] + [data[-1].replace("350", "450")]
        self.assertFalse(verify_benchmark(summarize(overlap, 2), b)["matched"])
        orphan = data + ["HETERO_NATIVE_PREFILL_COMMITTED request=2 rows=3 chunk=8"]
        self.assertFalse(verify_benchmark(summarize(orphan, 2), b)["matched"])
        with self.assertRaises(ValueError):
            summarize(data[:-2] + [data[-1]], 2)

    def test_predispatch_auxiliary_and_overlap(self):
        data = trace(offset=20)
        data += [span("coordinator", "stage_predispatch", 1, 10, 19),
                 span("worker", "final_window_admission", 1, 30, 90),
                 span("coordinator", "reset_barrier", 2, 0, 8, 0, 1),
                 span("prefill", "native_init", 1, 5, 7, 0, 3),
                 span("mtp_proposal", "roundtrip", 1, 321, 340, 1, 1)]
        result = summarize(data, 2)
        w = result["windows"][0]
        self.assertEqual(w["start_ns"], 10)
        self.assertEqual(w["predispatch_ns"], 9)
        self.assertEqual(w["stages"][1]["worker_overlap_prior_stage_ns"], 60)
        self.assertEqual(len(result["auxiliary_spans"]), 2)
        self.assertTrue(result["auxiliary_spans"][1]["anchor_committed"])
        self.assertFalse(result["auxiliary_spans"][1]["next_window_committed"])
        self.assertEqual(result["uncommitted"], [])
        with self.assertRaises(ValueError):
            summarize(data+[data[-1]], 2)

    def test_bad_sequence(self):
        with self.assertRaises(ValueError):
            summarize(trace()+trace(2, 350, "decode", 1, 9), 2)


if __name__ == "__main__":
    unittest.main()
