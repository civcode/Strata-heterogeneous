import unittest

from serve.heterogeneous_sampling import validate_greedy_sampling


class GreedySamplingTests(unittest.TestCase):
    def test_native_inert_filters(self):
        validate_greedy_sampling({"temperature": 0, "top_p": 0.95,
                                  "min_p": 0.05, "top_k": 20,
                                  "seed": (1 << 64) - 1,
                                  "penalty_last_n": 64,
                                  "repetition_penalty": 1})
        validate_greedy_sampling({"temperature": -1, "top_k": 100,
                                  "seed": -1, "repeat_last_n": -1})
        validate_greedy_sampling({"temperature": None, "seed": None})
        validate_greedy_sampling({})

    def test_reject_active_or_invalid_options(self):
        for key, value in (
            ("temperature", 0.8), ("temperature", True),
            ("temperature", float("nan")), ("temperature", -float("inf")),
            ("top_p", 1.1), ("top_p", -0.1), ("min_p", float("nan")),
            ("top_k", -1), ("top_k", 20.0), ("top_k", False),
            ("seed", 1 << 64), ("seed", 1.0), ("seed", True),
            ("repetition_penalty", 1.2), ("repeat_penalty", 0.9),
            ("frequency_penalty", 0.1), ("presence_penalty", -0.1),
            ("penalty_last_n", 1.0), ("repeat_last_n", True),
            ("strata_tune", {"spec_min_p": 0.2}),
        ):
            with self.subTest(key=key, value=value), self.assertRaisesRegex(ValueError, key):
                validate_greedy_sampling({key: value})

    def test_active_penalty_is_not_excused_by_greedy_filters(self):
        with self.assertRaisesRegex(ValueError, "presence_penalty"):
            validate_greedy_sampling({"temperature": 0, "top_k": 1,
                                      "presence_penalty": 0.1, "penalty_last_n": 0})


if __name__ == "__main__":
    unittest.main()
