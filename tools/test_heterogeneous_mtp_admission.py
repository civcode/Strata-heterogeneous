import hashlib
import json
from pathlib import Path
import struct
import tempfile
import unittest

import heterogeneous_mtp_admission as m


class AdmissionTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.idx = self.root / 'dense.txt'
        self.vocab = self.root / 'draft_vocab.bin'

    def test_layout(self):
        self.idx.write_text('a q8_0 1 32 0 34\nb f32 1 4 256 16\n')
        self.assertEqual(m.dense_index(self.idx, 272)['b'], ('f32', 1, 4, 256, 16))

    def test_bad_extent(self):
        for line, size in [('a q8_0 1 31 0 34', 34), ('a f32 1 4 256 16', 272),
                           ('a f32 1 4 0 16', 17), ('a f32 1 4 0 12', 12),
                           ('a f32 1 4 0 16\na bf16 1 4 256 8', 264)]:
            with self.subTest(line=line):
                self.idx.write_text(line)
                with self.assertRaises(ValueError):
                    m.dense_index(self.idx, size)

    def test_vocab(self):
        self.vocab.write_bytes(struct.pack('<3i', 2, 0, 8))
        self.assertEqual(m.check_vocab(self.vocab, 9), 3)

    def test_bad_vocab(self):
        for raw in (b'', b'x', struct.pack('<2i', 3, 3), struct.pack('<i', -1), struct.pack('<i', 9)):
            self.vocab.write_bytes(raw)
            with self.assertRaises(ValueError):
                m.check_vocab(self.vocab, 9)

    def test_pins_no_import(self):
        source = self.root / 'fetch.py'
        pins = {str(i): '0' * 64 for i in range(31)}
        source.write_text('raise RuntimeError("must not execute")\nPINNED_REVISION="rev"\nSHA256=' + repr(pins))
        self.assertEqual(m.pinned_source(source), ('rev', pins))

    def test_manifest_pin_refuses(self):
        (self.root / 'mtp-manifest.json').write_text(json.dumps([{'name': 'a', 'sha256': 'bad'}]))
        with self.assertRaisesRegex(ValueError, 'unpinned'):
            m.check_manifest(self.root, {'a': 'good'})

    def test_sha(self):
        self.vocab.write_bytes(b'abc')
        self.assertEqual(m.sha(self.vocab), hashlib.sha256(b'abc').hexdigest())


if __name__ == '__main__':
    unittest.main()
