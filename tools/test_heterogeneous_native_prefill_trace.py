import unittest

from tools.heterogeneous_native_prefill_trace import join_native_prefill, native_timing_issue


class NativePrefillTraceTests(unittest.TestCase):
    marker = {"request": 1, "rows": 194, "chunk": 128}
    timing = {"request": 1, "rows": 194, "start_ns": 100, "end_ns": 300}

    def test_legacy_and_timed(self):
        self.assertEqual(join_native_prefill([self.marker], []), [self.marker])
        result = join_native_prefill([self.marker], [self.timing])[0]
        self.assertEqual(result["roundtrip_ns"], 200)
        self.assertNotIn("start_ns", self.marker)

    def test_reject_duplicate_or_orphan(self):
        for markers, timings in (([self.marker]*2, []), ([self.marker], [self.timing]*2),
                                 ([], [self.timing])):
            with self.subTest(markers=markers, timings=timings), self.assertRaises(ValueError):
                join_native_prefill(markers, timings)

    def test_reject_invalid_timing(self):
        for changes in ({"rows": 193}, {"start_ns": -1}, {"end_ns": 99}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                join_native_prefill([self.marker], [dict(self.timing, **changes)])

    def test_phase_and_order(self):
        result = join_native_prefill([self.marker], [self.timing])[0]
        bounds = {"request_start": 50e-9, "prefill_end": 400e-9}
        self.assertIsNone(native_timing_issue(result, bounds, 350))
        self.assertIsNone(native_timing_issue(self.marker, bounds, 350))
        self.assertIsNotNone(native_timing_issue(result, bounds, 299))
        self.assertIsNotNone(native_timing_issue(result, dict(bounds, prefill_end=250e-9), 350))
        self.assertIsNotNone(native_timing_issue(result, dict(bounds, request_start=150e-9), 350))


if __name__ == "__main__":
    unittest.main()
