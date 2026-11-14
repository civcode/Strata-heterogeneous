import unittest
from tools.heterogeneous_replay_report import summarize


class ReplayReportTest(unittest.TestCase):
    def setUp(self):
        self.manifest = {"runs_per_turn": 3, "turns": [
            {"task": "A", "turn": 1, "prompt_token_ids": [1]},
            {"task": "A", "turn": 2, "prompt_token_ids": [2]},
            {"task": "B", "turn": 1, "prompt_token_ids": [3]}]}
        self.run = {"completions": [
            {"prompt_token_ids": [i + 1], "token_ids": [7], "finish_reason": "stop",
             "timing": {"decode_tokens_per_second": (i + 1) * 10 + r,
                        "prefill_ms": 2, "elapsed_ms": 4}}
            for r in range(3) for i in range(3)]}

    def test_equal_weight_tasks_and_warmup(self):
        result = summarize(self.manifest, self.run)
        self.assertEqual(result["task_medians"], {"A": 16.5, "B": 31.5})
        self.assertEqual(result["equal_weighted_task_median"], 24)

    def test_missing_rate_never_silently_excluded(self):
        self.run["completions"][3]["timing"]["decode_tokens_per_second"] = None
        result = summarize(self.manifest, self.run)
        self.assertIsNone(result["equal_weighted_task_median"])
        self.assertEqual(result["turns"][0]["valid_rates"], 1)
        self.run["completions"][3]["timing"]["decode_tokens_per_second"] = float("inf")
        self.assertIsNone(summarize(self.manifest, self.run)["equal_weighted_task_median"])

    def test_wrong_shape_or_prompt_rejected(self):
        self.run["completions"][0]["prompt_token_ids"] = [99]
        with self.assertRaises(ValueError):
            summarize(self.manifest, self.run)
        with self.assertRaises(ValueError):
            summarize(self.manifest, {"completions": []})


if __name__ == "__main__":
    unittest.main()
