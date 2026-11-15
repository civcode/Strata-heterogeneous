import unittest

from heterogeneous_q2_capacity import estimate


class CapacityTests(unittest.TestCase):
    def test_measured_2k_sessions(self):
        for mode, expected in [('int8', [56300032, 64429568, 55461888]),
                               ('f16', [62394624, 74587392, 63588096])]:
            self.assertEqual([x['session_bytes'] for x in estimate('mixed', 2048, mode)['owners']], expected)
        self.assertEqual([x['session_bytes'] for x in estimate('hip', 2048, 'int8')['owners']],
                         [83203072, 83203072])
        self.assertEqual([x['session_bytes'] for x in estimate('hip', 2048, 'f16')['owners']],
                         [95392512, 95392512])

    def test_native_increment_and_subtotal(self):
        a = estimate('hip', 2048, 'int8')['owners'][0]
        b = estimate('hip', 262144, 'int8')['owners'][0]
        self.assertEqual(b['session_bytes'] - a['session_bytes'], (262144-2048)*(6*1185+260))
        self.assertEqual(b['subtotal_not_peak_bytes'], sum(b[k] for k in
            ('session_bytes', 'pack_weight_bytes', 'expert_slot_bytes', 'reserve_bytes',
             'ple_device_bytes', 'verifier_device_floor_bytes', 'handoff_device_bytes')))
        self.assertEqual(b['ple_device_bytes'], 274352)
        self.assertEqual(estimate('hip', 2048, 'int8')['owners'][1]['handoff_device_bytes'], 409728)
        self.assertEqual(b['native_dense_estimate_bytes'], 766984192)
        self.assertIsNone(b['actual_free_vram_bytes'])
        self.assertEqual(b['admission'], 'UNKNOWN')

    def test_reject_unsupported(self):
        for args in [('hip', 2049, 'int8'), ('hip', 1024, 'int8'),
                     ('hip', 262400, 'int8'), ('cuda', 2048, 'f16'), ('hip', 2048, 'q4')]:
            with self.assertRaises(ValueError):
                estimate(*args)


if __name__ == '__main__':
    unittest.main()
