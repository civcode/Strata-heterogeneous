import tempfile
import unittest
from pathlib import Path
from tools.heterogeneous_long_context_config import derive, Q2_INDEX
from tools.heterogeneous_hip_pair_config import read_config


class ConfigTests(unittest.TestCase):
    def setUp(self):
        t = tempfile.TemporaryDirectory()
        self.addCleanup(t.cleanup)
        self.root = Path(t.name)
        self.sources = []
        for i, (lo, hi) in enumerate(((0, 15), (15, 33), (33, 48))):
            p = self.root / f"old{i}.cfg"
            values = dict(version=2, pack_dir="/pack", profile_path="/profile", index_sha256=Q2_INDEX,
                          model_fingerprint=2026111301, stage_index=i, device_ordinal=(0,0,1)[i],
                          first_layer=lo, end_layer=hi, n_layers=48, hc=4, n_embd=2560, n_vocab=248320,
                          max_cells=2048, max_t=8, cpu_workers=4, max_cache_slots=(hi-lo)*512,
                          io_timeout_ms=120000, free_reserve_bytes=1<<30, arena_budget_bytes=0,
                          gguf_shards="/a.gguf;/b.gguf")
            if i == 0:
                values["ple_gguf_path"] = "/b.gguf"
            p.write_text("".join(f"{k}={v}\n" for k,v in values.items()))
            self.sources.append(p)

    def test_mixed_and_hip(self):
        for split, plan in (((12,21,15), "cuda:0:0-12,hip:0:12-33,hip:1:33-48"),
                            ((24,24), "hip:0:0-24,hip:1:24-48")):
            output = self.root / str(len(split))
            evidence = derive(self.sources, output, split, 262144)
            self.assertEqual(evidence["plan"], plan)
            self.assertEqual(evidence["qualification"], "NOT_RUN")
            for i, n in enumerate(split):
                f, _ = read_config(output/f"stage{i}.cfg")
                self.assertEqual(int(f["max_cells"]), 262144)
                self.assertEqual(int(f["max_cache_slots"]), n*512)
                self.assertEqual("ple_gguf_path" in f, i == 0)
            with self.assertRaises(ValueError):
                derive(self.sources, output, split, 262144)
        self.assertIn("max_cells=2048\n", self.sources[0].read_text())

    def test_reject_unsafe(self):
        output = self.root / "unsafe"
        for split, context in (((15,18,15), 262145), ((1,23,24), 262144), ((24,23), 262144)):
            with self.assertRaises(ValueError):
                derive(self.sources, output, split, context)
        p = self.sources[0]
        p.write_text(p.read_text().replace("free_reserve_bytes=1073741824", "free_reserve_bytes=536870912"))
        with self.assertRaises(ValueError):
            derive(self.sources, output, (15,18,15), 262144)
        self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
