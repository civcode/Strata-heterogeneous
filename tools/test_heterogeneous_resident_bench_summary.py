import copy
import unittest

from heterogeneous_resident_bench_summary import compare


def block(kv='int8', plan=1, prefill=2):
    source = dict(pack_index_sha256='a'*64, gguf='/model', pack_dir='/pack', vocab=100,
                  max_cells=2048, max_t=8, context_limit=2048, prompt_chunk=1,
                  native_prefill_chunk=128, group_native_hip=True, stream_tokens=True,
                  lookup_draft=False, restart_clean=False, workload_mode=True,
                  long_context=False, kv=kv, plan_fingerprint=plan)
    records = []
    for i in range(4):
        p = 200 if i == 0 else prefill
        records.append(dict(request_id=i+1, prompt_tokens=3, prompt_token_ids=[1,2,3],
                            token_ids=[4,5,6], finish_reason='length', stop_token=-1,
                            timing=dict(prompt_tokens=3, committed_prompt_tokens=3,
                                        completion_tokens=3, decode_feedback_tokens=2,
                                        prefill_ms=p, decode_ms=10, elapsed_ms=p+12, first_output_ms=p+1)))
    return dict(status='complete', source=source, completions=records, startup_ms=20, shutdown_ms=10)


class BenchmarkTests(unittest.TestCase):
    def setUp(self):
        self.a = block()
        self.b = block('f16', prefill=4)
        self.blocks = [self.a,self.b,copy.deepcopy(self.b),copy.deepcopy(self.a)]

    def test_warmup_and_rates(self):
        result = compare(self.blocks, 'kv')
        self.assertEqual([r['measured'] for r in result['arms']], [6,6])
        self.assertEqual(result['arms'][0]['metrics']['prefill_tokens_s']['median'], 1500)
        self.assertEqual(result['arms'][1]['metrics']['prefill_tokens_s']['median'], 750)
        self.assertEqual(result['arms'][0]['metrics']['decode_feedback_tokens_s']['median'], 200)

    def test_topology_and_output_drift(self):
        for i in (1,2):
            self.blocks[i]['source'].update(kv='int8', plan_fingerprint=2)
        self.blocks[2]['completions'][2]['token_ids'] = [6,5,4]
        result = compare(self.blocks, 'topology')
        self.assertEqual(result['distinct_output_hashes'], 2)

    def test_mismatch(self):
        for key, value in [('context_limit',4096),('native_prefill_chunk',256),('pack_index_sha256','b'*64)]:
            data = copy.deepcopy(self.blocks)
            data[1]['source'][key] = value
            with self.assertRaisesRegex(ValueError, 'unmatched'):
                compare(data,'kv')
        with self.assertRaisesRegex(ValueError, 'arm order'):
            compare([self.a,self.b,self.a,self.b],'kv')

    def test_reject_early_stop_and_wrong_counts(self):
        self.blocks[2]['completions'][2]['finish_reason'] = 'stop'
        with self.assertRaisesRegex(ValueError, 'incomplete'):
            compare(self.blocks,'kv')
        self.blocks[2]['completions'][2]['finish_reason'] = 'length'
        self.blocks[2]['completions'][2]['timing']['decode_feedback_tokens'] = 3
        with self.assertRaisesRegex(ValueError, 'committed'):
            compare(self.blocks,'kv')

    def test_reject_nonfinite(self):
        self.blocks[1]['completions'][2]['timing']['prefill_ms'] = float('nan')
        with self.assertRaisesRegex(ValueError, 'timings'):
            compare(self.blocks,'kv')


if __name__ == '__main__':
    unittest.main()
