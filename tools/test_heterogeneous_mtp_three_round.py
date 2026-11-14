import unittest
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import sys

import heterogeneous_mtp_three_round as t


def run(keeps=(2,1,2)):
    out = [({'pos0': i, 'rows': 1, 'n_keep': 1, 'token_ids': [x]}, {'predictions': [300], 'fingerprint': 1, 'vocab': 500})
           for i, x in enumerate(t.PROMPT)]
    pos, anchor = len(t.PROMPT), 300
    for i, keep in enumerate(keeps):
        ids = [anchor, 10+i, 20+i, 30+i, 40+i]
        preds = ids[1:keep] + [301+i]*(6-keep)
        out.append(({'pos0': pos, 'rows': 5, 'n_keep': keep, 'token_ids': ids},
                    {'predictions': preds, 'fingerprint': 1, 'vocab': 500}))
        pos += keep
        anchor = preds[keep-1]
    return out


class ThreeRound(unittest.TestCase):
    def test_all_four_prefixes(self):
        points = t.checkpoints(run())
        self.assertEqual([len(p['prefix']) for p in points], [7,9,10,12])
        self.assertEqual(t.outcome(points), (2,10))
        self.assertEqual(points[-1]['prefix'][-2:], [302,12])

    def test_exact_bound_no_truncation(self):
        for keeps in ((4,1,1), (1,4,1), (2,1,3)):
            with self.assertRaisesRegex(ValueError, 'unrepresentable'):
                t.checkpoints(run(keeps))
        self.assertEqual(len(t.checkpoints(run((3,1,1)))[-1]['prefix']),12)

    def test_bad_anchor_keep_or_position(self):
        for key, value in [('token_ids',[0,1,2,3,4]), ('n_keep',1), ('pos0',8)]:
            records=run()
            records[7][0][key]=value
            with self.assertRaises(ValueError):
                t.checkpoints(records)

    def test_incomplete_or_wrong_prompt(self):
        for records in (run()[:-1],run() + run()[:1]):
            with self.assertRaises(ValueError):
                t.checkpoints(records)
        records=run()
        records[0][0]['token_ids']=[1]
        with self.assertRaises(ValueError):
            t.checkpoints(records)

    def test_offer_order(self):
        point = t.checkpoints(run())[0]
        offer = dict(cell=point['cell'], anchor=point['anchor'], tokens=point['candidates'])
        self.assertTrue(t.offer_equal(point,offer))
        offer['tokens']=list(reversed(offer['tokens']))
        self.assertFalse(t.offer_equal(point,offer))

    def test_gate_and_mismatch(self):
        for keeps, bad, expected in [((2,1,2),False,'PASS'), ((2,1,2),True,'FAIL'), ((1,1,1),False,'FAIL')]:
            reused=run(keeps)
            points=t.checkpoints(reused)
            points[-1]['candidates']=[99]
            fresh_paths=[Path(f'f{i}') for i in range(4)]
            fresh={p:[({'token_ids':[token]}, {'predictions':[point['anchor']], 'fingerprint':1,'vocab':500})
                       for token in point['prefix']] for p,point in zip(fresh_paths,points)}
            def offer(log, rounds, vocab, traced=False):
                pt=points[-1] if log=='r' else points[int(log[1:])]
                return dict(cell=pt['cell'],anchor=pt['anchor'],tokens=[1] if bad and log=='f1' else pt['candidates'])
            parity=SimpleNamespace(read_run=lambda p,**kw: reused if p==Path('r') else fresh[p],
                                   compare=lambda *a,**kw: dict(status='PASS',max_abs_logit_difference=0))
            reuse=SimpleNamespace(read_offer=offer,accepted_rows=lambda r:[None]*12)
            with patch.dict(sys.modules,{'heterogeneous_compare_parity':parity,'heterogeneous_mtp_reuse':reuse}):
                result=t.analyze(Path('r'),'r',[(p,str(p)) for p in fresh_paths])
            self.assertEqual(result['status'],expected)
            self.assertEqual(len(result['checkpoints']),4)
            self.assertEqual(result['production_admission'],'NOT QUALIFIED')

    def test_duplicate_fresh_refused(self):
        with patch.dict(sys.modules,{'heterogeneous_compare_parity':SimpleNamespace(read_run=None,compare=None),
                                     'heterogeneous_mtp_reuse':SimpleNamespace(read_offer=None,accepted_rows=None)}):
            with self.assertRaisesRegex(ValueError,'distinct'):
                t.analyze(Path('r'),'r',[(Path('f'),'')]*4)


if __name__=='__main__':
    unittest.main()
