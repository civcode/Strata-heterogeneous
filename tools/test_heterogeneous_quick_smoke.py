import unittest
from tools.heterogeneous_quick_smoke import run_suite


class QuickSmokeTest(unittest.TestCase):
    def setUp(self):
        self.now = 0
        self.rows = []
        self.cases = [{"id": "a", "max_tokens": 8}, {"id": "b", "max_tokens": 8}]
        self.calls = []

    def sender(self, url, body, key, timeout, cancel):
        self.calls.append(timeout)
        self.now += 2
        return {"status": "COMPLETE"}

    def runner(self, base, model, case, key, timeout, sender):
        return {"stages": [sender(base, {}, key, timeout)], "quality": "PASS"}

    def run_checks(self, budget=10, sender=None, runner=None):
        return run_suite(self.cases, "http://127.0.0.1:8081", "test", "", budget, 3,
                         self.rows.append, runner or self.runner, sender or self.sender, lambda: self.now)

    def test_one_pass(self):
        result = self.run_checks()
        self.assertEqual(result["attempted"], 2)
        self.assertEqual(result["quality_passed"], 2)
        self.assertEqual(self.calls, [3, 3])

    def test_deadline(self):
        result = self.run_checks(budget=1)
        self.assertEqual(result["not_attempted"], ["b"])
        self.assertEqual(self.calls, [1])

    def test_failure_stops_new_cases(self):
        result = self.run_checks(sender=lambda *a: {"status": "ERROR"})
        self.assertEqual(result["transport_failed"], 1)
        self.assertEqual(result["not_attempted"], ["b"])

    def test_budget_checked_between_tool_stages(self):
        def runner(base, model, case, key, timeout, sender):
            stages = [sender(base, {}, key, timeout), sender(base, {}, key, timeout)]
            return {"stages": stages, "quality": "FAIL"}
        result = self.run_checks(budget=1, runner=runner)
        self.assertEqual(result["transport_failed"], 1)
        self.assertEqual(len(self.calls), 1)
        self.assertEqual(self.rows[0]["stages"][1]["status"], "BUDGET_EXHAUSTED")

    def test_quality_not_transport(self):
        def runner(base, model, case, key, timeout, sender):
            return {"stages": [sender(base, {}, key, timeout)], "quality": "FAIL"}
        result = self.run_checks(runner=runner)
        self.assertEqual(result["quality_failed"], 2)
        self.assertEqual(result["transport_failed"], 0)

    def test_reject_long(self):
        self.cases[0]["max_tokens"] = 65
        with self.assertRaises(ValueError):
            self.run_checks()


if __name__ == "__main__":
    unittest.main()
