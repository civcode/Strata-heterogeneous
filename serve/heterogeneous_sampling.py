"""Admission for the existing unpenalized greedy resident execution path."""

import math


def validate_greedy_sampling(sampling):
    """Allow valid native filters that cannot change greedy argmax.

    No sampling or penalty algorithm is implemented here. The ordinary HTTP
    server still owns request normalization and native config defaults.
    """
    sampling = sampling or {}

    def number(value):
        return type(value) in (int, float) and (type(value) is int or math.isfinite(value))

    def integer(value):
        return type(value) is int

    checks = {
        "temperature": lambda v: number(v) and v <= 0,
        "top_p": lambda v: number(v) and 0 <= v <= 1,
        "min_p": lambda v: number(v) and 0 <= v <= 1,
        "top_k": lambda v: integer(v) and v >= 0,
        # Native treats nonpositive seeds as unset and forwards positive seeds.
        "seed": lambda v: integer(v) and v <= (1 << 64) - 1,
        "presence_penalty": lambda v: number(v) and v == 0,
        "frequency_penalty": lambda v: number(v) and v == 0,
        "repetition_penalty": lambda v: number(v) and v == 1,
        "repeat_penalty": lambda v: number(v) and v == 1,
        # A window has no effect without an active penalty.
        "penalty_last_n": integer,
        "repeat_last_n": integer,
    }
    for key, check in checks.items():
        value = sampling.get(key)
        if value is not None and not check(value):
            raise ValueError(f"{key} is not supported by the experimental greedy resident engine")
    if sampling.get("strata_tune"):
        raise ValueError("strata_tune is not supported by the experimental resident engine")
