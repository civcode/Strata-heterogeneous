import unittest
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import sys

import heterogeneous_mtp_acceptance as a


def relay(accepted):
    run = [({'pos0': i, 'rows': 1, 'n_keep': 1, 'token_ids': [t]}, {'predictions': [2724], 'fingerprint': 1})
           for i, t in enumerate(a.PROMPT)]
    ids = [2724, 579, 23513, 1830, 364]
    preds = ids[1:accepted+1] + [3]*(5-accepted)
    return run + [({'pos0': 7, 'rows': 5, 'n_keep': accepted+1, 'token_ids': ids}, {'predictions': preds, 'fingerprint': 1})]


def history(prefix, anchor, groups, drafts):
    out, cell = [], 0
    pairs = prefix[1:] + [anchor]
    for n in groups:
        phase = 'draft' if cell in drafts else 'prime'
        residual = b''.join(bytes([j])*a.STRIDE for j in range(cell, cell+n))
        for suffix, upto in [('before', cell), ('after', cell+n)]:
            meta = dict(event=len(out)+1, phase=phase+'_'+suffix, cell=cell, upto=upto,
                        tokens=pairs[cell:cell+n], format=0, kv_strides=[1024,1024,0,0] if upto else [0]*4)
            kv = b''.join(bytes([j])*a.KV_STRIDE for j in range(upto))
            out.append((meta, dict(r=residual, k=kv, v=kv, ks=b'', vs=b'')))
        cell += n
    return out


class Acceptance(unittest.TestCase):
    def test_all_greedy_outcomes(self):
        for n in range(5):
            r = a.relay_prefix(relay(n))
            self.assertEqual(r['accepted_drafts'], n)
            self.assertEqual(r['rejected_drafts'], 4-n)
            self.assertEqual(len(r['prefix']), 8+n)

    def test_wrong_greedy_keep(self):
        run = relay(1)
        run[-1][0]['n_keep'] = 3
        with self.assertRaisesRegex(ValueError, 'greedy'):
            a.relay_prefix(run)

    def test_prompt_and_anchor(self):
        for window in (0, -1):
            run = relay(1)
            run[window][0]['token_ids'][0] = 0
            with self.assertRaises(ValueError):
                a.relay_prefix(run)

    def setUp(self):
        self.info = a.relay_prefix(relay(3))
        self.prefix, self.anchor = self.info['prefix'], self.info['next_anchor']
        self.end = len(self.prefix)-1
        self.fresh = history(self.prefix, self.anchor, [1]*len(self.prefix), (self.end,))
        self.reuse = history(self.prefix, self.anchor, [1]*7+[3,1], (6,self.end))

    def rows(self, events, reused=True):
        return a.trace_history(events, self.prefix, self.anchor, (6,self.end) if reused else (self.end,))

    def test_batched_accepted_prime(self):
        self.assertEqual(a.differences(self.rows(self.fresh, False), self.rows(self.reuse)), [])

    def test_missing_tail_or_wrong_successor(self):
        for events in (self.reuse[:-2], deepcopy(self.reuse)):
            events[-2][0]['tokens'] = [0]
            with self.assertRaises(ValueError):
                self.rows(events)

    def test_committed_cell_mutation(self):
        self.reuse[-1][1]['k'] = b'x'+self.reuse[-1][1]['k'][1:]
        with self.assertRaisesRegex(ValueError, 'committed k changed at cell 0'):
            self.rows(self.reuse)

    def test_residual_mutation(self):
        self.reuse[-1][1]['r'] = b'x'+self.reuse[-1][1]['r'][1:]
        with self.assertRaisesRegex(ValueError, 'residual changed'):
            self.rows(self.reuse)

    def test_final_cell_difference(self):
        self.reuse[-1][1]['v'] = self.reuse[-1][1]['v'][:-1]+b'x'
        self.assertEqual(a.differences(self.rows(self.fresh, False), self.rows(self.reuse)),
                         [{'cell': self.end, 'field': 'v'}])

    def test_difference_in_accepted_batched_residual(self):
        for i in (-4,-3):
            self.reuse[i][1]['r'] = b'x'+self.reuse[i][1]['r'][1:]
        self.assertEqual(a.differences(self.rows(self.fresh, False), self.rows(self.reuse)),
                         [{'cell': 7, 'field': 'r'}])

    def test_acceptance_gate(self):
        for n, same, delta, status in [(0,True,0,'FAIL'), (1,True,0,'PASS'), (1,False,0,'FAIL'), (1,True,.01,'FAIL')]:
            run = relay(n)
            fresh = run[:-1]
            fake_parity = SimpleNamespace(read_run=lambda p, **kw: run if p == Path('r') else fresh,
                                          compare=lambda *x, **kw: dict(status='PASS', max_abs_logit_difference=delta))
            fake_reuse = SimpleNamespace(validate=lambda *x, **kw: ({'tokens':[1]}, {'tokens':[1 if same else 2]}, fresh, n))
            with patch.dict(sys.modules, {'heterogeneous_compare_parity':fake_parity, 'heterogeneous_mtp_reuse':fake_reuse}):
                result = a.analyze(Path('f'), Path('r'), '', '')
            self.assertEqual(result['status'],status)
            self.assertEqual(result['committed_kv'],'UNMEASURED')


if __name__ == '__main__':
    unittest.main()
