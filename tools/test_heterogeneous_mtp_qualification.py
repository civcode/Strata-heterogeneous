import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace

import heterogeneous_mtp_qualification as q
import heterogeneous_mtp_three_round as three
import heterogeneous_prefill_localize as prefill


class Matrix(unittest.TestCase):
    def setUp(self):
        d = tempfile.TemporaryDirectory()
        self.addCleanup(d.cleanup)
        self.path = Path(d.name) / 'matrix.json'
        self.obj = dict(schema=q.SCHEMA, cases=[])
        for case, prompt in enumerate((three.PROMPT, three.PROMPT[:-1])):
            runs = []
            for i,(transport,traced) in enumerate(sorted(q.MODES)):
                base = f'{case}-{i}'
                fresh = [dict(prefix=f'{base}-f{j}',log=f'{base}-f{j}.log') for j in range(4)]
                runs.append(dict(transport=transport,traced=traced,prefix=base,log=base+'.log',fresh=fresh))
                for name in [base+'.log']+[x['log'] for x in fresh]:
                    (self.path.parent/name).write_text(f'LIVE workers ready, mode={transport} processes=2\n')
            self.obj['cases'].append(dict(prompt_ids=prompt,runs=runs))

    def save(self):
        self.path.write_text(json.dumps(self.obj))
        return self.path

    def test_load_and_modes(self):
        self.assertEqual(len(q.load_manifest(self.save())),2)
        self.obj['cases'][0]['runs'][0]['traced'] = not self.obj['cases'][0]['runs'][0]['traced']
        with self.assertRaisesRegex(ValueError,'each case'):
            q.load_manifest(self.save())

    def test_reused_capture_refused(self):
        self.obj['cases'][1]['runs'][0]['fresh'][0]['prefix'] = self.obj['cases'][0]['runs'][0]['prefix']
        with self.assertRaisesRegex(ValueError,'reused'):
            q.load_manifest(self.save())

    def test_transport(self):
        log = 'LIVE workers ready, mode=shared processes=2\n'
        q.read_transport(log,'shared')
        for raw,mode in [(log,'socket'),(log*2,'shared'),('','shared')]:
            with self.assertRaises(ValueError):
                q.read_transport(raw,mode)

    def exercise(self, identical_patterns=False, delta=0, source_change=False, failed=False):
        def gate(p,*args,traced,prompt):
            pattern = [1,0,0] if prompt==three.PROMPT or identical_patterns else [0,1,0]
            return dict(status='FAIL' if failed else 'PASS',parity_status='FAIL' if failed else 'PASS',
                        checkpoints=[dict(cell=i,anchor=2,candidates=[3],accepted_drafts=n) for i,n in enumerate(pattern)]+[dict(cell=3,anchor=2,candidates=[3])],
                        accepted_drafts=1,rejected_drafts=11)
        def read(p,**kwargs):
            return [(dict(plan='plan',model_fingerprint=1,pack_index_sha256='pack',gguf_shards='shards',profile_path='profile'),
                     dict(fingerprint=2 if source_change and p.name.startswith('1') else 1,vocab=10))]
        backend = SimpleNamespace(read_run=read,compare=lambda *a,**kw:dict(status='PASS',max_abs_logit_difference=delta))
        with patch.object(q,'analyze',gate),patch.dict(sys.modules,{'heterogeneous_compare_parity':backend}):
            return q.compare_matrix(self.save())

    def test_matrix_coverage(self):
        self.assertEqual(self.exercise()['status'],'PASS')
        self.assertEqual(self.exercise(identical_patterns=True)['status'],'FAIL')
        self.assertEqual(self.exercise(delta=.01)['status'],'FAIL')
        self.assertEqual(self.exercise(failed=True)['status'],'FAIL')
        with self.assertRaisesRegex(ValueError,'source changed'):
            self.exercise(source_change=True)

    def test_variable_checkpoint(self):
        from test_heterogeneous_mtp_three_round import run
        records=run((2,1,2))[1:]
        for i,(meta,packet) in enumerate(records):
            meta['pos0']-=1
        points=three.checkpoints(records,three.PROMPT[1:])
        self.assertEqual([len(p['prefix']) for p in points],[6,8,9,11])
        with self.assertRaises(ValueError):
            three.checkpoints(records)


class Prefill(unittest.TestCase):
    def test_separate_gates(self):
        good=dict(status='PASS',max_abs_logit_difference=0)
        state=dict(status='DIFFERENT',expected_ple_prev=[-1,1],
                   observed_ple_prev={m:[-1,1] for m in ('serial-a','serial-b','native')},
                   comparisons={f'{a}_vs_{b}':{'gdn0':{'bytes_equal':b=='serial-b'}}
                                for a,b in (('serial-a','serial-b'),('serial-a','native'),('serial-b','native'))})
        comparisons={k:good for k in state['comparisons']}
        result=prefill.summarize(state,dict(status='PASS'),comparisons)
        self.assertTrue(result['serial_control_exact'])
        self.assertTrue(result['bridge_exact'])
        self.assertTrue(result['native_logits_gate'])
        self.assertFalse(result['cuda_state_exact'])
        comparisons['serial-a_vs_serial-b']=dict(status='PASS',max_abs_logit_difference=.1)
        self.assertFalse(prefill.summarize(state,dict(status='PASS'),comparisons)['serial_control_exact'])


if __name__=='__main__':
    unittest.main()
