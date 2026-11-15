"""Synthetic evidence helper tests, not native model execution."""
import copy
import unittest

import numpy as np
from heterogeneous_hc_experiment import check_log, compare_evidence, metric, OWNERS


class HcExperimentTest(unittest.TestCase):
    def test_owner_modes(self):
        check_log('ordinary run', None)
        for scope in ('all', 'cuda', 'hip'):
            for mode in ('fp32', 'bf16x2'):
                ident = {'mode': mode, 'backend': scope}
                lines = [f'STRATA_HC_EXPERIMENT backend={b} layers={lo}-{hi} requested={mode} scope={scope} '
                         f'effective={mode if scope in ("all", b) else "off"}' for b, lo, hi in OWNERS]
                check_log('\n'.join(lines), ident)
                with self.assertRaisesRegex(ValueError, 'missing'):
                    check_log('\n'.join(lines[:-1]), ident)
                with self.assertRaises(ValueError):
                    check_log('\n'.join(lines), None)
                with self.assertRaises(ValueError):
                    check_log('\n'.join(lines).replace('effective=', 'effective=typo'), ident)

    def evidence(self):
        return {'marker': {'plan': 'same', 'prefix_rows': 2, 'chunk': 1, 'token_ids': [1,2,3],
                           'transport': 'shared', 'stage_reference': ''}, 'source': 'same',
                'bridge': {'status': 'PASS'}, 'local_links_exact': True, 'serial_controls_exact': True,
                'logits': {'serial-a': 'x', 'serial-b': 'x', 'native': 'y'},
                'residuals': {'stage0.p0.in': 'in', 'stage0.p0.out': 'out', 'stage1.p0.out': 'hip'},
                'states': {'0': {'identity': 'layout', 'sha256': {'a': 'x'}}}}

    def test_cross_arm_controls(self):
        base = self.evidence()
        for scope in ('all', 'cuda', 'hip'):
            ident = {'mode': 'fp32', 'backend': scope}
            arm = copy.deepcopy(base)
            arm['marker']['hc_experiment'] = ident
            arm['logits']['native'] = 'changed'
            arm['residuals']['stage1.p0.out'] = 'changed'
            compare_evidence(base, arm, ident)
            for key in ('stage0.p0.in', 'stage0.p0.out'):
                changed = copy.deepcopy(arm)
                changed['residuals'][key] = 'changed'
                if key.endswith('.in') or scope == 'hip':
                    with self.assertRaisesRegex(ValueError, 'CUDA'):
                        compare_evidence(base, changed, ident)
                else:
                    compare_evidence(base, changed, ident)
            changed = copy.deepcopy(arm)
            changed['logits']['serial-a'] = 'changed'
            with self.assertRaisesRegex(ValueError, 'serial'):
                compare_evidence(base, changed, ident)
            if scope == 'hip':
                changed = copy.deepcopy(arm)
                changed['states']['0']['sha256']['a'] = 'changed'
                with self.assertRaisesRegex(ValueError, 'CUDA checkpoint'):
                    compare_evidence(base, changed, ident)

    def test_metrics(self):
        a = np.array([1, 2], np.float32)
        self.assertEqual(metric(a, a)['max_abs'], 0)
        self.assertAlmostEqual(metric(a, a+1)['rmse'], 1)
        self.assertIsNone(metric(a*0, a)['relative_l2_to_baseline'])
        with self.assertRaises(ValueError):
            metric(a, np.array([np.nan, 1]))


if __name__ == '__main__':
    unittest.main()
