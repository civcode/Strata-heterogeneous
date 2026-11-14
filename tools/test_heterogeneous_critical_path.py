import unittest
from tools.heterogeneous_critical_path import summarize, union


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

    def test_bad_sequence(self):
        with self.assertRaises(ValueError):
            summarize(trace()+trace(2, 350, "decode", 1, 9), 2)


if __name__ == "__main__":
    unittest.main()
