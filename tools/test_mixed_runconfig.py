import copy
from pathlib import Path
import tempfile
import unittest
from strata_control.mixed_runconfig import resolve, bind_standard_fields, verify_source

class NativeConfigTests(unittest.TestCase):
    def setUp(self):
        self.cfg = {"heterogeneous": {"plan": "fixture"}, "args": [
            "--max-context", "262144", "--kv", "int8", "--kv-resident", "0",
            "--mtp", "/model/mtp", "--spec", "4", "--spec-min-p", "0.5", "--prefill", "32"]}

    def test_native_options_and_nonmutation(self):
        before = copy.deepcopy(self.cfg)
        opts = resolve(self.cfg)
        self.assertEqual(opts["native_spec_window"], 4)
        self.assertEqual(opts["native_prefill_chunk"], 32)
        self.assertEqual(opts["max_context"], 262144)
        self.assertEqual(opts["kv"], "int8")
        self.assertEqual(self.cfg, before)

    def test_overrides_and_last_wins(self):
        self.cfg["args"] += ["--prefill", "16"]
        self.assertEqual(resolve(self.cfg)["native_prefill_chunk"], 16)
        self.cfg["user_args"] = ["--prefill", "64", "--spec", "5", "--kv", "fp16"]
        opts = resolve(self.cfg)
        self.assertEqual((opts["native_prefill_chunk"], opts["native_spec_window"], opts["kv"]), (64, 5, "f16"))

    def test_native_placement_preserves_existing_mtp_policy(self):
        self.cfg["heterogeneous"].pop("plan")
        self.cfg["args"] += ["--layer-split", "15,33", "--hetero-devices", "cuda:0,hip:0,hip:1"]
        before = copy.deepcopy(self.cfg)
        opts = resolve(self.cfg)
        self.assertEqual(opts["plan"], "cuda:0:0-15,hip:0:15-33,hip:1:33-48")
        self.assertEqual((opts["spec"], opts["native_spec_window"]), (4, 4))
        self.assertEqual(self.cfg, before)
        self.cfg["user_args"] = ["--layer-split", "16,32"]
        self.assertEqual(resolve(self.cfg)["plan"], "cuda:0:0-16,hip:0:16-32,hip:1:32-48")
        self.cfg["heterogeneous"]["plan"] = "fixture"
        with self.assertRaisesRegex(ValueError, "heterogeneous.plan"):
            resolve(self.cfg)
        self.cfg["heterogeneous"].pop("plan")
        self.cfg["args"] = self.cfg["args"][:-2]
        with self.assertRaisesRegex(ValueError, "together"):
            resolve(self.cfg)

    def test_unsupported_never_ignored(self):
        for args in (["--vision"], ["--prefill", "auto"], ["--kv-resident", "1"], ["--spec", "8"], ["--spec-min-p", "0.6"]):
            with self.subTest(args=args):
                with self.assertRaises(ValueError):
                    resolve({**self.cfg, "user_args": args})
        self.cfg["heterogeneous"]["spec"] = 4
        with self.assertRaisesRegex(ValueError, "duplicated"):
            resolve(self.cfg)

    def test_source_identity_and_stamp(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "pack" / "tokenizer").mkdir(parents=True)
            for name in ("model.gguf", "ple.gguf", "profile.json", "wrong.gguf"):
                (root / name).touch()
            source = {"pack_dir": str(root / "pack"), "gguf": str(root / "model.gguf"),
                      "ple_gguf": str(root / "ple.gguf"), "profile_path": str(root / "profile.json")}
            self.cfg["cwd"] = temp
            bind_standard_fields(self.cfg, source)
            self.assertEqual(self.cfg["args"], self.cfg["args_written"])
            self.assertEqual(self.cfg["tokenizer"], str(root / "pack" / "tokenizer"))
            verify_source(self.cfg, source)
            self.cfg["user_args"] = ["--native", "wrong.gguf"]
            with self.assertRaisesRegex(ValueError, "differs"):
                verify_source(self.cfg, source)

if __name__ == "__main__":
    unittest.main()
