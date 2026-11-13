#!/usr/bin/env python3
"""CPU-only STP0/v1 integration and first-divergent-phase fixtures."""
import json
from array import array
from pathlib import Path
import struct
import tempfile
import unittest

import heterogeneous_stage0_phases as ph
from test_heterogeneous_compare_parity import packet as stfr_packet


def stp0(*, window=1, position=0, token=1, change=None,
         delta=0.0, model_width=2560):
    vals = array('f', (
        1.0 + 0.1 * layer + 0.02 * phase + 0.0001 * col +
        (delta if change == (layer, phase) else 0.0)
        for layer in range(12)
        for phase in range(4)
        for col in range(model_width)
    ))
    blob = bytearray(ph.HEADER.pack(
        ph.MAGIC, 1, 0, 12, 4, model_width, 1, 48,
        1, window, position, 123, 0, 0))
    blob.extend(struct.pack('<i', token))
    blob.extend(vals.tobytes())
    hashvalue = 2166136261
    for i, val in enumerate(blob):
        if 64 <= i < 68:
            continue
        hashvalue = ((hashvalue ^ val) * 16777619) & 0xffffffff
    struct.pack_into('<I', blob, 64, hashvalue)
    return bytes(blob)


class StageZeroTests(unittest.TestCase):
    def setUp(self):
        self.t = tempfile.TemporaryDirectory()
        self.addCleanup(self.t.cleanup)
        self.folder = Path(self.t.name)
        self.ref = self.folder / 'ref'
        self.het = self.folder / 'mixed'
        self.write(self.ref, ph.REFERENCE_PLAN)
        self.write(self.het, ph.HETERO_PLAN)

    def write(self, prefix, plan, *, change=None, delta=0.0,
              windows=1, model='a', input_token=1):
        Path(f'{prefix}.complete.json').write_text(json.dumps({
            'schema': 'strata-heterogeneous-parity/v1',
            'windows': windows, 'plan': plan, 'model_fingerprint': 2026111301
        }))
        for i in range(1, windows + 1):
            meta = {'schema': 'strata-heterogeneous-parity/v1',
                    'plan': plan, 'model_fingerprint': 2026111301,
                    'plan_fingerprint': 123, 'pack_index_sha256': model * 64,
                    'gguf_shards': '/models/a.gguf;/models/b.gguf',
                    'profile_path': '/pack/profile.strp',
                    'window_id': i, 'pos0': i-1, 'rows': 1,
                    'n_keep': 1, 'token_ids': [input_token]}
            Path(f'{prefix}.w{i}.json').write_text(json.dumps(meta))
            Path(f'{prefix}.w{i}.stfr').write_bytes(stfr_packet(
                stage=plan.count(','), window=i))
            Path(f'{prefix}.stage0.w{i}.stp0').write_bytes(stp0(
                window=i, position=i-1, token=input_token,
                change=change, delta=delta))

    def run_compare(self):
        return ph.compare(ph.read_run(self.ref), ph.read_run(self.het), 0.001)

    def test_header_is_72_bytes(self):
        self.assertEqual(ph.HEADER.size, 72)

    def test_matching_all_phases(self):
        result = self.run_compare()
        self.assertEqual(result['status'], 'DIAGNOSTIC_ONLY')
        self.assertIsNone(result['first_crossings'][0]['first_phase_above_threshold'])
        self.assertEqual(len(result['phase_metrics']), 48)

    def test_first_at_layer6_attention_branch(self):
        self.write(self.het, ph.HETERO_PLAN, change=(6, 1), delta=0.1)
        first = self.run_compare()['first_crossings'][0]['first_phase_above_threshold']
        self.assertEqual((first['layer'], first['phase']), (6, 'attention_gdn_bo'))

    def test_first_at_layer8_hc0(self):
        self.write(self.het, ph.HETERO_PLAN, change=(8, 0), delta=0.1)
        first = self.run_compare()['first_crossings'][0]['first_phase_above_threshold']
        self.assertEqual((first['layer'], first['phase']), (8, 'hc0_mixed'))

    def test_first_at_layer7_ffn_hc(self):
        self.write(self.het, ph.HETERO_PLAN, change=(7, 2), delta=0.1)
        first = self.run_compare()['first_crossings'][0]['first_phase_above_threshold']
        self.assertEqual((first['layer'], first['phase']), (7, 'hc1_mixed'))

    def test_first_at_layer4_moe(self):
        self.write(self.het, ph.HETERO_PLAN, change=(4, 3), delta=0.1)
        first = self.run_compare()['first_crossings'][0]['first_phase_above_threshold']
        self.assertEqual((first['layer'], first['phase']), (4, 'moe_bo'))

    def test_two_windows(self):
        self.write(self.ref, ph.REFERENCE_PLAN, windows=2)
        self.write(self.het, ph.HETERO_PLAN, windows=2)
        result = self.run_compare()
        self.assertEqual(len(result['first_crossings']), 2)
        self.assertEqual(len(result['phase_metrics']), 96)

    def test_corrupt_checksum(self):
        file = Path(f'{self.het}.stage0.w1.stp0')
        raw = bytearray(file.read_bytes())
        raw[-1] ^= 1
        file.write_bytes(raw)
        with self.assertRaisesRegex(ValueError, 'checksum'):
            self.run_compare()

    def test_truncated_record(self):
        file = Path(f'{self.het}.stage0.w1.stp0')
        file.write_bytes(file.read_bytes()[:-4])
        with self.assertRaisesRegex(ValueError, 'size mismatch'):
            self.run_compare()

    def test_wrong_model_source(self):
        self.write(self.het, ph.HETERO_PLAN, model='b')
        with self.assertRaisesRegex(ValueError, 'pack_index_sha256'):
            self.run_compare()

    def test_mismatched_input_token(self):
        self.write(self.het, ph.HETERO_PLAN, input_token=2)
        with self.assertRaisesRegex(ValueError, 'token_ids'):
            self.run_compare()

    def test_missing_commit_marker(self):
        Path(f'{self.het}.complete.json').unlink()
        with self.assertRaisesRegex(ValueError, 'completion marker'):
            self.run_compare()

    def test_swapped_reference_rejected(self):
        with self.assertRaisesRegex(ValueError, 'unsplit HIP reference'):
            ph.compare(ph.read_run(self.het), ph.read_run(self.ref), 0.001)

    def test_invalid_threshold_rejected(self):
        with self.assertRaisesRegex(ValueError, 'threshold'):
            ph.compare(ph.read_run(self.ref), ph.read_run(self.het), -1)


if __name__ == '__main__':
    unittest.main()
