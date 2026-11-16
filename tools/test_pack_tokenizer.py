"""Pack tokenizer loading and GGUF identity checks without model weights."""
import json
import tempfile
import unittest
from pathlib import Path

from tools.strata_tokenizer import Tokenizer


class TestPackTokenizer(unittest.TestCase):
    def test_shared_loader(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)
            tokens = ["a", "b", "ab"]
            cfg = {"pre": "qwen35", "special_ids": {"eos": 2}}
            (p / "vocab.json").write_text(json.dumps({t: i for i, t in enumerate(tokens)}))
            (p / "merges.txt").write_text("a b")
            (p / "token_type.json").write_text("[1, 1, 1]")
            (p / "tokenizer.json").write_text(json.dumps(cfg))
            tok, actual = Tokenizer.from_pack(p)
            self.assertEqual(actual, cfg)
            self.assertEqual(tok.encode("ab"), [2])
            self.assertEqual(tok.decode([2]), "ab")
            tok.require_same_encoding(Tokenizer(tokens, ["a b"], [1, 1, 1], "qwen35", {"eos": 2}))
            for mismatch in (
                Tokenizer(tokens, ["a b"], [1, 1, 1], "qwen35", {"eos": 1}),
                Tokenizer(["b", "a", "ab"], ["a b"], [1, 1, 1], "qwen35", {"eos": 2}),
                Tokenizer(tokens, [], [1, 1, 1], "qwen35", {"eos": 2}),
            ):
                with self.assertRaises(ValueError):
                    tok.require_same_encoding(mismatch)


if __name__ == "__main__":
    unittest.main()
