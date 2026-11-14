import struct
import unittest

import heterogeneous_mtp_front as f


class FrontTests(unittest.TestCase):
    def test_layout(self):
        self.assertEqual(f.BYTES, 298044)
        raw = bytes(i % 251 for i in range(f.BYTES))
        fields = f.split(raw)
        self.assertEqual(fields['residual'], raw[236:41196])
        self.assertEqual(fields['v_append'], raw[214076:216124])
        self.assertEqual(fields['source_before'], raw[216124:257084])
        self.assertEqual(fields['source_after'], raw[257084:])
        self.assertEqual(b''.join(fields.values()), raw)

    def test_bad_extent(self):
        for n in (0, f.BYTES-1, f.BYTES+1):
            with self.assertRaises(ValueError):
                f.split(bytes(n))

    def test_staging(self):
        fields = f.split(bytes(f.BYTES))
        fields['token'] = struct.pack('<i', 2724)
        fields['step'] = fields['rest_step'] = struct.pack('<4i', 6,7,1,7)
        fields['position'] = fields['rest_position'] = struct.pack('<24i', *([6]*24))
        self.assertEqual(f.staging(fields,6,2724,bytes(40960)), [])
        self.assertEqual(f.staging(fields,6,264,bytes(40960)), ['token'])
        fields['rest_step'] = bytes(16)
        fields['source_before'] = b'x' + bytes(40959)
        self.assertEqual(f.staging(fields,6,2724,bytes(40960)), ['rest_step', 'source_before'])

    def test_first_difference(self):
        a = f.split(bytes(f.BYTES))
        b = a.copy()
        b['embedding_fc'] = b'x'+bytes(10239)
        b['k_append'] = bytes(2047)+b'x'
        d = f.differences(a,b)
        self.assertEqual([k for k,v in d.items() if v], ['embedding_fc','k_append'])
        self.assertEqual(d['embedding_fc'],1)
        self.assertEqual(d['k_append'],1)


if __name__ == '__main__':
    unittest.main()
