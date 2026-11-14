import unittest
from heterogeneous_utilization import summarize


def benchmark():
    def request(i, start):
        return {"request_id": i, "finish_reason": "length", "timing": {
            "decode_feedback_tokens": 20, "phase_monotonic_s": {
                "request_start": start, "prefill_end": start + 1,
                "decode_end": start + 3, "request_end": start + 3.1}}}
    return {"status": "complete", "phase_monotonic_s": {
        "startup_start": 1, "startup_end": 2, "shutdown_start": 10, "shutdown_end": 11},
            "completions": [request(1, 3), request(2, 6.2)]}


def sample(cycle, device, t, util):
    backend, index = device.split(":")
    return {"cycle": cycle, "backend": backend, "monitor_index": int(index),
            "read_start_s": t, "monotonic_s": t + .001, "interval_s": .25,
            "util_percent": util, "vram_used_bytes": 100, "vram_total_bytes": 200, "power_w": 50}


class UtilizationTests(unittest.TestCase):
    devices = ["cuda:0", "hip:0", "hip:1"]

    def data(self):
        return [sample(cycle, device, t + i * .002, u)
                for cycle, t, values in [(1, 4.2, [100,100,100]), (2, 7.3, [100,90,0]),
                                          (3, 7.6, [90,95,100]), (4, 7.9, [None,100,100]),
                                          (5, 10.2, [100,100,100])]
                for i, (device, u) in enumerate(zip(self.devices, values))]

    def test_warm_excludes_first_and_shutdown(self):
        s = summarize(benchmark(), self.data(), self.devices)["warm_decode"]
        self.assertAlmostEqual(s["feedback_tok_s"], 10)
        self.assertEqual(s["gpu"]["cuda:0"]["utilization_percent"],
                         {"samples": 2, "mean": 95, "median": 95, "peak": 100})
        self.assertEqual(s["gpu"]["hip:1"]["samples_below_10_percent"], 100/3)
        self.assertEqual(s["aligned_complete_cycles"], 2)
        self.assertEqual(s["all_gpus_at_least_90_percent"], 50)
        self.assertTrue(s["low_sample_warning"])

    def test_power_is_independent_of_util_and_aligned(self):
        data = self.data()
        # One cycle has a missing busy sensor but all power sensors valid.
        # Another cycle has a missing power sensor; do not treat it as zero.
        for row in data:
            if row["cycle"] == 2 and row["backend"] == "cuda":
                row["power_w"] = None
            if row["cycle"] == 3 and row["backend"] == "cuda":
                row["power_w"] = 80
                row["power_limit_w"] = 150
                row["temperature_c"] = 56
        s = summarize(benchmark(), data, self.devices)["warm_decode"]
        self.assertEqual(s["total_gpu_power_w"], {"samples": 2, "mean": 165, "median": 165, "peak": 180})
        self.assertEqual(s["gpu"]["cuda:0"]["power_w"]["samples"], 2)
        self.assertEqual(s["gpu"]["cuda:0"]["power_limit_w"]["mean"], 150)
        self.assertEqual(s["gpu"]["cuda:0"]["temperature_c"]["peak"], 56)
        self.assertTrue(s["power_low_sample_warning"])

    def test_power_skew_and_missing_cycle(self):
        data = [sample(1, "cuda:0", 7.4, 100), sample(1, "hip:0", 7.42, 100),
                sample(1, "hip:1", 7.6, 100), sample(2, "cuda:0", 8.0, 100)]
        s = summarize(benchmark(), data, self.devices)["warm_decode"]
        self.assertEqual(s["total_gpu_power_w"], {"samples": 0, "mean": None, "median": None, "peak": None})
        self.assertTrue(s["power_low_sample_warning"])

    def test_duplicate_refused(self):
        data = self.data()
        with self.assertRaises(ValueError):
            summarize(benchmark(), data + data[:1], self.devices)

    def test_misaligned_cycle_not_counted(self):
        data = self.data() + [sample(6, "cuda:0", 8.1, 100), sample(6, "hip:0", 8.12, 100),
                              sample(6, "hip:1", 8.3, 100)]
        s = summarize(benchmark(), data, self.devices)["warm_decode"]
        self.assertEqual(s["aligned_complete_cycles"], 2)

    def test_missing_not_zero(self):
        s = summarize(benchmark(), [sample(1, "cuda:0", 7.4, None)], self.devices)["warm_decode"]
        self.assertIsNone(s["gpu"]["cuda:0"]["utilization_percent"]["mean"])
        self.assertIsNone(s["all_gpus_at_least_90_percent"])
        self.assertTrue(s["low_sample_warning"])

    def test_bad_phase_refused(self):
        b = benchmark()
        b["completions"][1]["timing"]["phase_monotonic_s"]["decode_end"] = 6
        with self.assertRaises(ValueError):
            summarize(b, self.data(), self.devices)


if __name__ == "__main__":
    unittest.main()
