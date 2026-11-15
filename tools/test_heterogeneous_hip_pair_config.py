import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from heterogeneous_hip_pair_config import derive, read_config


class HipPairTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.sources = [self.root / f"src{i}.cfg" for i in range(3)]
        for i, (lo, hi) in enumerate(((0, 12), (12, 30), (30, 48))):
            f = dict(version=2, pack_dir="/pack", profile_path="/profile", index_sha256="a"*64,
                     model_fingerprint=123, stage_index=i, device_ordinal=max(0, i-1), first_layer=lo,
                     end_layer=hi, n_layers=48, hc=4, n_embd=2560, n_vocab=248320,
                     max_cells=32768, max_t=8, cpu_workers=4, max_cache_slots=(hi-lo)*512,
                     io_timeout_ms=120000, free_reserve_bytes=536870912+i, arena_budget_bytes=0,
                     gguf_shards="/a.gguf;/b.gguf")
            if i == 0:
                f["ple_gguf_path"] = "/b.gguf"
            self.sources[i].write_text("".join(f"{k}={v}\n" for k,v in f.items()))
        self.out = self.root / "out"

    def test_default_and_provenance(self):
        old = [p.read_bytes() for p in self.sources]
        result = derive(self.sources, self.out)
        self.assertEqual(result["plan"], "hip:0:0-24,hip:1:24-48")
        self.assertEqual(result["qualification"], "NOT_RUN")
        self.assertEqual(result, json.loads((self.out/"manifest.json").read_text()))
        for i in range(2):
            f, h = read_config(self.out/f"hip{i}.cfg")
            self.assertEqual(f["stage_index"], str(i))
            self.assertEqual(f["device_ordinal"], str(i))
            self.assertEqual(f["max_cache_slots"], "12288")
            self.assertEqual(f["max_cells"], "2048")
            self.assertEqual(f["free_reserve_bytes"], str(536870913+i))
            self.assertEqual(h, result["generated"][f"hip{i}.cfg"])
        self.assertEqual(read_config(self.out/"hip0.cfg")[0]["ple_gguf_path"], "/b.gguf")
        self.assertNotIn("ple_gguf_path", read_config(self.out/"hip1.cfg")[0])
        self.assertEqual(old, [p.read_bytes() for p in self.sources])
        self.assertEqual([hashlib.sha256(b).hexdigest() for b in old], [x["sha256"] for x in result["sources"]])
        with self.assertRaises(ValueError):
            derive(self.sources, self.out)

    def test_alternative_split(self):
        derive(self.sources, self.out, 23, 32768)
        self.assertEqual(read_config(self.out/"hip0.cfg")[0]["max_cache_slots"], str(23*512))
        self.assertEqual(read_config(self.out/"hip1.cfg")[0]["max_cache_slots"], str(25*512))

    def test_mismatched_inputs_fail_before_writing(self):
        cases = [("model_fingerprint=123", "model_fingerprint=456"),
                 ("max_cache_slots=9216", "max_cache_slots=4"),
                 ("free_reserve_bytes=536870914", "free_reserve_bytes=1"),
                 ("first_layer=30", "first_layer=29"), ("max_cells=32768", "max_cells=1024"),
                 ("arena_budget_bytes=0", "arena_budget_bytes=1")]
        old = self.sources[2].read_text()
        for before, after in cases:
            with self.subTest(before=before):
                self.sources[2].write_text(old.replace(before, after))
                with self.assertRaises(ValueError):
                    derive(self.sources, self.out)
                self.assertFalse(self.out.exists())

    def test_invalid_context_and_order(self):
        with self.assertRaises(ValueError):
            derive(self.sources, self.out, context=32769)
        with self.assertRaises(ValueError):
            derive(list(reversed(self.sources)), self.out)
        self.assertFalse(self.out.exists())


if __name__ == "__main__":
    unittest.main()
