import json
from pathlib import Path
import tempfile
import unittest

from heterogeneous_mtp_state import compare


class StateTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.a = self.trace('a')
        self.b = self.trace('b')

    def trace(self, name, **changes):
        prefix = self.root / name
        Path(str(prefix) + '.complete.json').write_text(json.dumps({
            'schema': 'strata-heterogeneous-parity/v1', 'model_fingerprint': 123, 'plan': 'cuda,hip,hip'}))
        p = Path(str(prefix) + '.mtp.1.json')
        meta = {'schema': 'strata-mtp-state/v1', 'event': 1, 'phase': 'draft_after',
                'cell': 1, 'upto': 2, 'tokens': [9], 'residual_stride': 40960,
                'format': 1, 'kv_strides': [4, 8, 2, 2]}
        meta.update(changes)
        p.write_text(json.dumps(meta))
        p.with_suffix('.r.bin').write_bytes(bytes(len(meta['tokens'])*40960))
        for field, stride in zip(('k', 'v', 'ks', 'vs'), meta['kv_strides']):
            p.with_suffix('.'+field+'.bin').write_bytes(bytes(meta['upto']*stride))
        return p

    def test_equal(self):
        self.assertEqual(compare(self.a, self.b)['status'], 'IDENTICAL')

    def test_different_cell(self):
        self.b.with_suffix('.k.bin').write_bytes(b'\0'*4 + b'x'*4)
        self.assertEqual(compare(self.a, self.b)['fields']['k']['different_cells'], [1])
        self.assertEqual(compare(self.a, self.b, upto=1)['status'], 'IDENTICAL')

    def test_residual_difference(self):
        self.b.with_suffix('.r.bin').write_bytes(b'x'+bytes(40959))
        self.assertEqual(compare(self.a, self.b)['fields']['r']['different_cells'], [1])
        self.assertEqual(compare(self.a, self.b, kv_only=True)['status'], 'IDENTICAL')

    def test_wrong_pair(self):
        self.b = self.trace('b', tokens=[8])
        with self.assertRaisesRegex(ValueError, 'different input pair'):
            compare(self.a, self.b)
        self.assertEqual(compare(self.a, self.b, kv_only=True)['status'], 'IDENTICAL')

    def test_bad_extent(self):
        self.b.with_suffix('.v.bin').write_bytes(b'x')
        with self.assertRaisesRegex(ValueError, 'extent'):
            compare(self.a, self.b)

    def test_incomplete_run(self):
        (self.root/'b.complete.json').unlink()
        with self.assertRaises(OSError):
            compare(self.a, self.b)

    def test_bad_layout_and_prefix(self):
        self.b = self.trace('b', format=2)
        with self.assertRaisesRegex(ValueError, 'layout'):
            compare(self.a, self.b)
        with self.assertRaisesRegex(ValueError, 'outside committed'):
            compare(self.a, self.a, upto=3)


if __name__ == '__main__':
    unittest.main()
