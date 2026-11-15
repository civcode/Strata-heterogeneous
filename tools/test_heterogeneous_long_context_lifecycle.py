import unittest

from tools.heterogeneous_long_context_lifecycle import exercise


class FakeResident:
    max_context = 262144
    stream_tokens = True
    source = {'long_context': True, 'native_prefill_chunk': 128}
    stop_ids = {99}

    def __init__(self):
        self.request = 0
        self.corrupt = None

    def alive(self):
        return True

    def generate(self, ids, budget, sampling, cancel):
        self.request += 1
        pos = len(ids)-1
        yield None
        tokens = []
        if not cancel.is_set():
            for j in range(budget):
                pos += 1
                token = 10+j
                if self.corrupt == self.request:
                    token += 4
                tokens.append(token)
                yield token
                if cancel.is_set():
                    break
        self.last_completion = dict(request_id=self.request, prompt_tokens=len(ids),
                                    finish_reason='cancelled' if cancel.is_set() else 'length',
                                    token_ids=[] if cancel.is_set() else tokens)
        self.last_metrics = dict(committed_prompt_tokens=min(pos,len(ids)),
                                 decode_feedback_tokens=max(0,pos-len(ids)))


class LifecycleTests(unittest.TestCase):
    def setUp(self):
        self.short = [1]*7
        self.long = [1]*262136

    def test_full_sequence(self):
        cases = exercise(FakeResident(), self.short, self.long)
        self.assertEqual([c['request'] for c in cases], list(range(1,8)))
        self.assertEqual(cases[2]['committed_position'], len(self.long)-1)
        self.assertEqual(cases[4]['committed_position'], len(self.long))
        self.assertEqual(cases[5]['committed_position'], len(self.long)+3)
        self.assertEqual(cases[1]['output_sha256'], cases[5]['output_sha256'])
        self.assertEqual(cases[2]['finish'], 'cancelled')
        self.assertEqual(cases[4]['streamed_tokens'], 1)

    def test_detect_reset_drift(self):
        e = FakeResident()
        e.corrupt = 4
        with self.assertRaisesRegex(ValueError, 'short reset'):
            exercise(e, self.short, self.long)
        e = FakeResident()
        e.corrupt = 6
        with self.assertRaisesRegex(ValueError, 'repeated long'):
            exercise(e, self.short, self.long)

    def test_not_near_capacity(self):
        with self.assertRaisesRegex(ValueError, 'near-capacity'):
            exercise(FakeResident(), self.short, [1]*100)


if __name__ == '__main__':
    unittest.main()
