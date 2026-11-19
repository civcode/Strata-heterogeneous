#!/usr/bin/env python3
"""setup.py's mixed split (`--backend mixed_split`, or answer 3 at the first question): no GPU, no compiler, no download.

    python3 mixed_split/tests/test_setup_mixed_split.py
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import setup  # noqa: E402

NV = {"index": 0, "name": "NVIDIA GeForce RTX 4070 Ti SUPER", "vram_gb": 16.0, "arch": "89"}
AMD = [{"index": 0, "name": "AMD Radeon AI PRO R9700", "vram_gb": 32.0, "arch": "gfx1201"},
       {"index": 1, "name": "AMD Radeon AI PRO R9700", "vram_gb": 32.0, "arch": "gfx1201"}]


def args(**kw):
    a = argparse.Namespace(gpus=None, gpu=None, front_layers=None, yes=True, check=False)
    a.__dict__.update(kw)
    return a


def production_cfg():
    """The shape of the Human User's production config (strata-q2_0.json): two R9700, a layer split, hand edits."""
    return {"exe": "/x/engine/strata", "backend": "hip", "gpu": [0, 1], "gpus_asked": True, "layer_split": "auto",
            "model_name": "qwen3.8-flash-next-q2_0", "port": 8080,
            "args": ["--native", "/m/a.gguf", "--max-context", "262144", "--kv", "int8", "--pipeline-windows", "2",
                     "--split-skip-if-fits", "--batch", "2", "--prompt-cache", "6"], "parallel": 2}


class Base(unittest.TestCase):
    def setUp(self):
        self.saved = (setup.MIXED_CHOICE, setup.MIXED_FRONT, setup.MIXED_LAYERS)
        self.out = []
        for name in ("say", "ok", "warn"):
            p = mock.patch.object(setup, name, side_effect=lambda m="", n=name: self.out.append((n, m)))
            p.start()
            self.addCleanup(p.stop)

    def tearDown(self):
        setup.MIXED_CHOICE, setup.MIXED_FRONT, setup.MIXED_LAYERS = self.saved

    def pick(self, **kw):
        with mock.patch.object(setup, "mixed_split_available", return_value=True):
            a = args(**kw)
            setup.mixed_split_pick(a, [NV], AMD)
            return a


class Pick(Base):
    def test_every_amd_card_and_15_layers_on_a_16_gb_card(self):
        a = self.pick()
        self.assertTrue(setup.MIXED_CHOICE)
        self.assertEqual(a.gpus, "0,1")
        self.assertEqual(setup.MIXED_LAYERS, 15)
        self.assertEqual(setup.MIXED_FRONT["index"], 0)

    def test_front_layers_flag_and_named_cards_win(self):
        a = self.pick(front_layers=12, gpus="1,0")
        self.assertEqual((a.gpus, setup.MIXED_LAYERS), ("1,0", 12))

    def test_refusals(self):
        with mock.patch.object(setup, "mixed_split_available", return_value=True), \
                mock.patch.object(setup, "fail", side_effect=SystemExit) as fail:
            with self.assertRaises(SystemExit):
                setup.mixed_split_pick(args(), [NV], AMD[:1])          # one AMD card
            self.assertIn("two or more AMD cards", fail.call_args[0][0])
            with self.assertRaises(SystemExit):
                setup.mixed_split_pick(args(), [], AMD)                # no NVIDIA card
            with self.assertRaises(SystemExit):
                setup.mixed_split_pick(args(front_layers=1), [NV], AMD)
        with mock.patch.object(setup, "mixed_split_available", return_value=False), \
                mock.patch.object(setup, "fail", side_effect=SystemExit):
            with self.assertRaises(SystemExit):
                setup.mixed_split_pick(args(), [NV], AMD)


class Config(Base):
    def test_front_options_added_refused_ones_left_out(self):
        self.pick()
        cfg = production_cfg()
        setup.mixed_split_config(cfg)
        a = cfg["args"]
        self.assertEqual(a[-4:], ["--front-exe", str(setup.FRONT_DIR / setup.EXE), "--front-layers", "15"])
        for refused in ("--split-skip-if-fits", "--batch"):
            self.assertNotIn(refused, a)
        self.assertIn("--pipeline-windows", a)                         # the person's options stay
        self.assertEqual(cfg["parallel"], 1)
        self.assertEqual(cfg["mixed_split"]["front_layers"], 15)
        self.assertNotIn("--front-arg", a)                             # one NVIDIA card: no --gpu for the front stage

    def test_a_later_run_keeps_it_once(self):
        self.pick()
        cfg = production_cfg()
        setup.mixed_split_config(cfg)
        first = list(cfg["args"])
        setup.MIXED_CHOICE = None                                      # a later run that did not ask (the key was carried over)
        setup.mixed_split_config(cfg)
        self.assertEqual(cfg["args"], first)

    def test_one_make_of_card_turns_it_off(self):
        self.pick()
        cfg = production_cfg()
        setup.mixed_split_config(cfg)
        setup.MIXED_CHOICE = False
        setup.mixed_split_config(cfg)
        self.assertNotIn("mixed_split", cfg)
        self.assertFalse(any(x.startswith("--front-") for x in cfg["args"]))

    def test_one_amd_card_runs_without_it(self):
        self.pick()
        cfg = {**production_cfg(), "gpu": 0}
        setup.mixed_split_config(cfg)
        self.assertNotIn("mixed_split", cfg)
        self.assertNotIn("--front-exe", cfg["args"])
        self.assertTrue(any(n == "warn" and "one AMD card" in m for n, m in self.out))

    def test_vanilla_config_untouched(self):
        setup.MIXED_CHOICE = None
        cfg = production_cfg()
        setup.mixed_split_config(cfg)
        self.assertEqual(cfg, production_cfg())

    def test_two_nvidia_cards_name_the_front_card(self):
        with mock.patch.object(setup, "mixed_split_available", return_value=True):
            setup.mixed_split_pick(args(), [{**NV, "index": 0, "vram_gb": 8.0}, {**NV, "index": 1}], AMD)
        cfg = production_cfg()
        setup.mixed_split_config(cfg)
        self.assertEqual(cfg["args"][-4:], ["--front-arg", "--gpu", "--front-arg", "1"])

    def test_write_setup_config_carries_it_and_the_persons_options(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "strata-q2_0.json"
            old = production_cfg()
            old["args"] += ["--control-vector-scaled", "/v.gguf:1.0"]           # a hand edit (not in args_written)
            old["args_written"] = production_cfg()["args"]
            path.write_text(json.dumps(old))
            self.pick()
            setup.write_setup_config(path, production_cfg())
            cfg = json.loads(path.read_text())
            self.assertIn("--control-vector-scaled", cfg["args"])
            self.assertIn("--front-exe", cfg["args"])
            self.assertIn("--front-exe", cfg["args_written"])            # setup's own option: not a hand edit next time
            setup.MIXED_CHOICE = None                                    # setup run again for this model, not asked
            setup.write_setup_config(path, production_cfg())
            again = json.loads(path.read_text())
            self.assertEqual(again["args"].count("--front-exe"), 1)
            self.assertIn("mixed_split", again)
            self.assertEqual(setup.mixed_split_files(again), [str(setup.FRONT_DIR / setup.EXE)])


class HipBuild(Base):
    def build(self, meta=None):
        calls = {}
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / "engine").mkdir()
            if meta is not None:
                (root / "engine" / "BUILD.json").write_text(json.dumps(meta))
            def cmake(src, bdir, target, defs, vcvars, bat):
                calls["defs"] = defs
            with mock.patch.object(setup, "ROOT", root), \
                    mock.patch.object(setup.acquisition, "require"), \
                    mock.patch.object(setup, "rocm_root", return_value=(root / "rocm", [str(root / "rocm" / "lib")])), \
                    mock.patch.object(setup, "cmake_build", side_effect=cmake), \
                    mock.patch.object(setup, "stale_binary", return_value=False), \
                    mock.patch.object(setup, "source_hash", side_effect=lambda parts: str(len(parts))), \
                    mock.patch.object(setup, "cpu_floor", return_value=""), \
                    mock.patch.object(setup, "cpu_info", return_value=(None, True)), \
                    mock.patch.object(setup, "ggml_defs", return_value=[]), \
                    mock.patch.object(setup, "isa_floor_defs", return_value=[]), \
                    mock.patch.object(setup.shutil, "copy2"), \
                    mock.patch.object(setup, "build_front_engine") as front:
                setup.build_engine_hip({"arch": "gfx1201", "archs": ["gfx1201"]}, None)
                calls["front"] = front.called
                calls["meta"] = json.loads((root / "engine" / "BUILD.json").read_text())
        return calls

    def test_mixed_build_has_the_option_and_the_front_engine(self):
        self.pick()
        c = self.build()
        self.assertIn("-DSTRATA_MIXED_SPLIT=ON", c["defs"])
        self.assertTrue(c["front"])
        self.assertTrue(c["meta"]["mixed_split"])

    def test_hip_build_turns_the_option_off(self):
        setup.mixed_split_off()
        c = self.build({"backend": "hip", "mixed_split": True})
        self.assertIn("-DSTRATA_MIXED_SPLIT=OFF", c["defs"])
        self.assertFalse(c["front"])
        self.assertNotIn("mixed_split", c["meta"])

    def test_an_update_keeps_a_mixed_engine_mixed(self):
        setup.MIXED_CHOICE = None
        c = self.build({"backend": "hip", "mixed_split": True})
        self.assertIn("-DSTRATA_MIXED_SPLIT=ON", c["defs"])
        self.assertTrue(c["front"])


class Question(unittest.TestCase):
    def test_the_first_question_offers_three_and_the_flag_exists(self):
        src = (ROOT / "setup.py").read_text(encoding="utf-8")
        self.assertIn('3) Both together: mixed split', src)
        self.assertIn('["1", "2", "3"] if mixed_offer else ["1", "2"]', src)
        self.assertIn('choices=["cuda", "hip", "sycl"] + ["mixed_split"]', src)


if __name__ == "__main__":
    unittest.main()
