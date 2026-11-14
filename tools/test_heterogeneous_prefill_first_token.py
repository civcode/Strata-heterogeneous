import struct
import tempfile
from pathlib import Path
import unittest

import heterogeneous_prefill_first_token as p


def row(v):
    return struct.pack('<2f',v,v)


def fields(names,v):
    return {k:row(v) for k in names}


class FirstToken(unittest.TestCase):
    def fixture(self):
        z=fields(p.STATES,0)
        s=fields(p.STATES,1)
        o=fields(p.OPS,2)
        return [[z.copy() for _ in range(3)],z.copy(),o.copy(),o.copy(),o.copy(),
                s.copy(),s.copy(),s.copy(),s.copy(),s.copy()]

    def test_identical_is_not_admission(self):
        g=p.classify(*self.fixture())
        self.assertEqual(g['conclusion'],'OBSERVED_ENTRY_FIELDS_IDENTICAL')
        self.assertEqual(g['status'],'DIAGNOSTIC_ONLY')
        self.assertEqual(g['production_admission'],'NOT QUALIFIED')

    def test_reset_and_init_take_precedence(self):
        f=self.fixture();f[0][1]['conv']=row(3);f[4]['embedding']=row(4)
        self.assertEqual(p.classify(*f)['conclusion'],'RESET_STATE_DIFFERENCE')
        f=self.fixture();f[1]['ple']=row(3)
        self.assertEqual(p.classify(*f)['conclusion'],'NATIVE_INITIALIZATION_STATE_DIFFERENCE')

    def test_serial_repeat(self):
        f=self.fixture();f[3]['hc_mixed']=row(3)
        self.assertEqual(p.classify(*f)['conclusion'],'UNSTABLE_SERIAL_CONTROL')

    def test_first_operation_pre_bridge(self):
        for op in p.OPS:
            f=self.fixture();f[4][op]=row(3)
            g=p.classify(*f)
            self.assertEqual(g['conclusion'],'CUDA_LOCAL_OPERATION_DIFFERENCE_BEFORE_BRIDGE')
            self.assertEqual(g['first_nonidentical_common_operation'],op)
            self.assertEqual(g['root_cause'],'NOT ESTABLISHED')

    def test_state_and_ownership_boundaries(self):
        for i,expected in [(7,'CUDA_LOCAL_STATE_DIFFERENCE_BEFORE_BRIDGE'),
                           (8,'STATE_CHANGED_BETWEEN_BRIDGE_AND_DRAIN'),
                           (9,'STATE_CHANGED_BETWEEN_DRAIN_AND_ADOPTION_OBSERVATION')]:
            f=self.fixture();f[i]['recurrence']=row(3)
            self.assertEqual(p.classify(*f)['conclusion'],expected)

    def test_bad_extent_and_nonfinite(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'row';path.write_bytes(row(1))
            self.assertEqual(p.read_f32(path,8),row(1))
            with self.assertRaises(ValueError):p.read_f32(path,4)
            path.write_bytes(row(float('nan')))
            with self.assertRaises(ValueError):p.read_f32(path,8)
        with self.assertRaises(ValueError):p.metric(row(1),b'')


if __name__=='__main__':unittest.main()
