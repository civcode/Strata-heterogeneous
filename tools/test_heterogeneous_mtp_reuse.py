import unittest
from unittest.mock import patch
import sys
from types import SimpleNamespace

import heterogeneous_mtp_reuse as reuse


def log(relay, cell=1, anchor=7, tokens='8,9,', count=2):
    return (f'HETERO_MTP_PROBE_MODE state_trace=0 front_trace=0 repeat_round=0 relay_rounds={relay}\n'
            f'HETERO_MTP_PROBE_DRAFT cell={cell} verified_next={anchor} count={count} tokens={tokens} (not submitted to target)\n')


def window(pos, tokens, keep, predictions):
    return ({'pos0': pos, 'rows': len(tokens), 'n_keep': keep, 'token_ids': tokens},
            {'rows': len(tokens), 'vocab': 10, 'predictions': predictions,
             'logits': list(range(len(tokens)*10))})


class ReuseTests(unittest.TestCase):
    def setUp(self):
        self.fresh = [window(0, [1], 1, [2]), window(1, [2], 1, [7])]
        self.reused = [self.fresh[0], window(1, [2, 8, 9], 1, [7, 1, 1])]

    def test_match(self):
        f, r, rows, count = reuse.validate(self.fresh, self.reused, log(0), log(1))
        self.assertEqual(f, r)
        self.assertEqual(rows, self.fresh)
        self.assertEqual(count, 0)

    def test_natural_acceptance(self):
        self.reused[-1] = window(1, [2, 7, 9], 2, [7, 6, 1])
        self.fresh.append(window(2, [7], 1, [6]))
        _, _, _, count = reuse.validate(self.fresh, self.reused, log(0, 2, 6), log(1, 2, 6))
        self.assertEqual(count, 1)

    def test_wrong_keep(self):
        self.reused[-1][0]['n_keep'] = 2
        with self.assertRaisesRegex(ValueError, 'greedy'):
            reuse.validate(self.fresh, self.reused, log(0), log(1))

    def test_wrong_pair(self):
        with self.assertRaisesRegex(ValueError, 'paired'):
            reuse.validate(self.fresh, self.reused, log(0), log(1, anchor=6))

    def test_wrong_prefix(self):
        self.fresh[-1][0]['token_ids'] = [3]
        with self.assertRaisesRegex(ValueError, 'accepted prefix'):
            reuse.validate(self.fresh, self.reused, log(0), log(1))

    def test_trace_or_duplicate_or_missing_refused(self):
        for raw in (log(1).replace('state_trace=0', 'state_trace=1'), log(1)*2, ''):
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                reuse.read_offer(raw, 1, 10)

    def test_bad_offer(self):
        for raw in (log(1, tokens='8,,9,'), log(1, tokens='10,9,'), log(1, count=3)):
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                reuse.read_offer(raw, 1, 10)

    def test_exact_gate(self):
        for candidate, delta, expected in (('8,9,', 0, 'PASS'), ('8,1,', 0, 'FAIL'), ('8,9,', .01, 'FAIL')):
            fake = SimpleNamespace(read_run=lambda prefix, **kw: self.fresh if prefix == 'f' else self.reused,
                                   compare=lambda *args, **kw: {'status': 'PASS', 'max_abs_logit_difference': delta})
            with patch.dict(sys.modules, {'heterogeneous_compare_parity': fake}):
                result = reuse.analyze('f', 'r', log(0), log(1, tokens=candidate))
            self.assertEqual(result['status'], expected)
            self.assertIn('UNMEASURED', result['committed_kv'])


if __name__ == '__main__':
    unittest.main()
