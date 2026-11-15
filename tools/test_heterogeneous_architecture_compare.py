import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from heterogeneous_architecture_compare import comparison, SCHEMA


class CompareTests(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.dir.cleanup)
        self.path = Path(self.dir.name)
        self.manifest = self.path / "manifest.json"
        self.manifest.write_text(json.dumps({"schema": SCHEMA, "cases": [{"id": "case"}]}))
        self.digest = hashlib.sha256(self.manifest.read_bytes()).hexdigest()

    def runfile(self, name, *, elapsed=4, finish="stop", prompt=100, completion=128,
                timing_source="host", cache=0, quality="PASS", fail=False, omit=False):
        path = self.path / (name + ".jsonl")
        header = dict(kind="header", schema=SCHEMA, manifest_sha256=self.digest, model="same", repeats=3)
        stage = dict(status="COMPLETE", elapsed_s=elapsed, ttft_s=elapsed/2,
                     finish_reason=finish, usage=dict(prompt_tokens=prompt, completion_tokens=completion),
                     server_timings=dict(timing_source=timing_source, prompt_n=prompt-cache, cache_n=cache,
                                         prompt_per_second=100/elapsed, predicted_per_second=128/elapsed))
        rows = [header] + [dict(kind="case", id="case", repeat=i, quality=quality,
                     stages=[dict(status="ERROR") if fail else stage]) for i in range(2 if omit else 3)] + [dict(kind="summary")]
        path.write_text("\n".join(map(json.dumps, rows)))
        return path

    def compare(self, **kwargs):
        base = self.runfile("base")
        trial = self.runfile("trial", **kwargs)
        return comparison(self.manifest, {"base": base, "trial": trial}, "base")["arms"]["trial"]["case"]

    def test_matched_speedup(self):
        r = self.compare(elapsed=2)
        self.assertEqual(r["attempts"], 2)
        self.assertEqual(r["matched_prompt_pairs"], 2)
        for metric in r["paired_speedup"].values():
            self.assertEqual(metric["median"], 2)

    def test_timing_sources_not_interchangeable(self):
        r = self.compare(timing_source="different", elapsed=2)
        self.assertEqual(r["paired_speedup"]["ttft_s"]["median"], 2)
        self.assertIsNone(r["paired_speedup"]["server_decode_tps"]["median"])
        self.assertIsNotNone(r["metrics"]["server_decode_tps"]["median"])

    def test_different_prompt_excludes_ratios(self):
        r = self.compare(prompt=101)
        self.assertEqual(r["matched_prompt_pairs"], 0)
        self.assertTrue(all(v["n"] == 0 for v in r["paired_speedup"].values()))

    def test_short_or_truncated_generation(self):
        r = self.compare(completion=32, finish="length", quality="FAIL")
        self.assertEqual(r["quality"]["FAIL"], 2)
        self.assertEqual(r["metrics"]["server_decode_tps"]["n"], 0)
        self.assertEqual(r["paired_speedup"]["elapsed_s"]["n"], 0)

    def test_cached_prefill_excluded(self):
        self.assertEqual(self.compare(cache=10)["metrics"]["server_prefill_tps"]["n"], 0)

    def test_failed_attempts_retained(self):
        r = self.compare(fail=True, quality="FAIL")
        self.assertEqual(r["first_statuses"], {"ERROR": 2})
        self.assertEqual(r["matched_prompt_pairs"], 0)

    def test_missing_attempts_rejected(self):
        with self.assertRaises(ValueError):
            self.compare(omit=True)


if __name__ == "__main__":
    unittest.main()
