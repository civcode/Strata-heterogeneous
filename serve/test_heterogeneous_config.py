import unittest
from unittest.mock import patch
from serve.heterogeneous_config import options, read_args, verify_model_args


class Tests(unittest.TestCase):
    def test_vanilla_options(self):
        cfg = {"exe":"/tmp/coordinator", "args":["--max-context","262144","--kv","int8","--prefill","128"], "heterogeneous":{"plan":"hip:0:0-24,hip:1:24-48"}}
        self.assertEqual(options(cfg), {"plan":"hip:0:0-24,hip:1:24-48", "coordinator":"/tmp/coordinator", "max_context":262144, "kv":"int8", "native_prefill_chunk":128})
        cfg["args"][3] = "fp16"
        self.assertEqual(options(cfg)["kv"], "f16")
        self.assertNotIn("kv", cfg["heterogeneous"])

    def test_reject(self):
        for args in [["--spec","4"], ["--kv","int8","--kv","fp16"], ["--kv"], ["--kv","--prefill"], ["--kv", 1]]:
            with self.assertRaises(ValueError): read_args(args)
        cfg = {"args":["--max-context","4","--kv","int8","--prefill","1"], "heterogeneous":{"kv":"int8"}}
        with self.assertRaises(ValueError): options(cfg)

    def test_legacy(self):
        cfg = {"heterogeneous":{"coordinator":"/x","kv":"f16","max_context":4096}}
        self.assertEqual(options(cfg), cfg["heterogeneous"])

    def test_binding(self):
        model = ["--pack","/tmp/pack","--native","/tmp/first.gguf","--ple-gguf","/tmp/second.gguf","--expert-profile","/tmp/profile"]
        with patch("serve.heterogeneous_config.stage_model_args", return_value=model):
            cfg = {"args":model, "tokenizer":"/tmp/pack/tokenizer"}
            verify_model_args(cfg, {"config":["/tmp/stage"]})
            cfg["args"] = [*model[:-1], "/tmp/wrong"]
            with self.assertRaises(ValueError): verify_model_args(cfg, {"config":["/tmp/stage"]})


if __name__ == "__main__":
    unittest.main()
