import unittest
from heterogeneous_vram import estimate, persistent, GIB, MIB


class VramTests(unittest.TestCase):
    def test_q8_geometry(self):
        r = persistent(0, 15, 262144)
        self.assertEqual((r['qsa'], r['gdn']), (3, 12))
        self.assertEqual(r['kv_bytes'], 792 * MIB)
        self.assertEqual(r['rope_bytes'], 64 * MIB)
        self.assertEqual(persistent(0, 16, 262144)['kv_bytes'], 1056 * MIB)

    def test_split_total(self):
        r = estimate((15, 18, 15))
        self.assertEqual(r['total_kv_bytes'], 3168 * MIB)
        self.assertEqual([s['expert_slots'] for s in r['stages']], [7680, 9216, 7680])
        self.assertEqual([s['qsa'] for s in r['stages']], [3, 5, 4])
        self.assertTrue(r['stages'][0]['estimated_reserve_fit'])
        self.assertFalse(estimate((16, 17, 15))['stages'][0]['estimated_reserve_fit'])

    def test_baseline_identity(self):
        r = estimate((12, 18, 18), 32768, 'f16')
        self.assertEqual([s['estimated_used_bytes'] for s in r['stages']], [11817*MIB, 17135*MIB, 17280*MIB])

    def test_rounding_and_primary(self):
        self.assertEqual(persistent(0, 1, 5)['kv_bytes'], 8*1056)
        self.assertEqual(persistent(0, 4, 262144, 'f16')['kv_bytes'], GIB//2)

    def test_bad_inputs(self):
        for split in ((15, 18, 14), (0, 24, 24), (15, 18, 15, 1)):
            with self.assertRaises(ValueError):
                estimate(split)
        with self.assertRaises(ValueError):
            persistent(0, 15, 0)
        with self.assertRaises(ValueError):
            estimate((15, 18, 15), reserve_mib=-1)


if __name__ == '__main__':
    unittest.main()
