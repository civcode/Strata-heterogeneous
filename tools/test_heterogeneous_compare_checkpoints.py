import json
from pathlib import Path
import struct
import tempfile
import unittest

from heterogeneous_compare_checkpoints import compare


class Checkpoints(unittest.TestCase):
    def setUp(self):
        t = tempfile.TemporaryDirectory()
        self.addCleanup(t.cleanup)
        self.root = Path(t.name)
        self.a = self.capture('a')
        self.b = self.capture('b')

    def capture(self, name):
        folder = self.root / name
        folder.mkdir()
        points = []
        for pos in range(3):
            file = f'{pos}.bin'
            (folder / file).write_bytes(struct.pack('<2f', 1., 2.))
            points.append(dict(position=pos, layer=0, name='post_layer', dtype='f32', shape=[2], file=file))
        path = folder / 'capture.json'
        path.write_text(json.dumps(dict(complete=True, kv_mode='f16', plan='hip:0:0-24,hip:1:24-48',
                                        token_ids=[1, 2, 3], checkpoints=points)))
        return path

    def change(self, fn):
        m = json.loads(self.b.read_text())
        fn(m)
        self.b.write_text(json.dumps(m))

    def test_first_drift(self):
        self.assertIsNone(compare(self.a, self.b)['first_byte_difference'])
        (self.b.parent / '2.bin').write_bytes(struct.pack('<2f', 1., 3.))
        result = compare(self.a, self.b)
        self.assertEqual(result['status'], 'DIAGNOSTIC_ONLY')
        self.assertEqual(result['first_numeric_difference']['position'], 2)
        self.assertEqual(result['first_numeric_difference']['first_element'], 1)
        self.assertIsNone(compare(self.a, self.b, atol=1.)['first_numeric_difference'])

    def test_format_and_sequence(self):
        self.change(lambda m: m.update(kv_mode='int8'))
        with self.assertRaisesRegex(ValueError, 'kv_mode'):
            compare(self.a, self.b)
        self.change(lambda m: m.update(kv_mode='f16', token_ids=[1, 2, 4]))
        with self.assertRaisesRegex(ValueError, 'token_ids'):
            compare(self.a, self.b)

    def test_incomplete_and_duplicate(self):
        self.change(lambda m: m.update(complete=False))
        with self.assertRaisesRegex(ValueError, 'incomplete'):
            compare(self.a, self.b)
        self.change(lambda m: (m.update(complete=True), m['checkpoints'].append(m['checkpoints'][-1])))
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            compare(self.a, self.b)

    def test_nonfinite_and_truncated(self):
        file = self.b.parent / '1.bin'
        file.write_bytes(struct.pack('<2f', 1., float('nan')))
        with self.assertRaisesRegex(ValueError, 'nonfinite'):
            compare(self.a, self.b)
        file.write_bytes(b'0')
        with self.assertRaisesRegex(ValueError, 'byte size'):
            compare(self.a, self.b)

    def test_path_escape(self):
        self.change(lambda m: m['checkpoints'][0].update(file='../a/0.bin'))
        with self.assertRaisesRegex(ValueError, 'relative'):
            compare(self.a, self.b)


if __name__ == '__main__':
    unittest.main()
