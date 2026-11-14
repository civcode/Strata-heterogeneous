"""Synthetic protocol records, not native inference outputs."""
import json
from pathlib import Path
import tempfile
import unittest

from test_heterogeneous_mtp_capture_integration import capture, trace, log
import test_heterogeneous_mtp_capture_integration as fixture
import heterogeneous_mtp_generation as generation

PROMPT=[1,2]
# Feed slots through cell 6 (five output inputs); the bonus 14 is final output.
ACTUAL=[1,2,3,4,8,9,10]
PRED=[2,3,4,8,9,10,14]
LIVE=[([1],1,[2]),([2],1,[3]),([3,4,5,6,7],2,[4,8,9,10,11]),
      ([8,12,13],1,[9,20,21]),([9,10],2,[10,14])]
POINTS=[(2,3,[4,5,6,7]),(4,8,[12,13,14,15]),(5,9,[10,11,12,13]),(7,14,[16,17,18,19])]


def make_capture(prefix,windows,mtp,stops=None,max_new=6,output=None,reason='length',next_id=14,final=None):
    capture(prefix,windows)
    marker=Path(str(prefix)+'.complete.json')
    data=json.loads(marker.read_text())
    data.update(mtp_fresh_replay=False,generation_capture=True,mtp_generation=mtp,prompt_ids=PROMPT,
                max_new=max_new,stop_ids=stops or [31],token_ids=output if output is not None else ACTUAL[2:]+[14],
                finish_reason=reason,stop_token=next_id if reason=='stop' else -1,next=next_id,
                final_offer=(final if final is not None else [14,16,17,18,19]) if mtp else [])
    marker.write_text(json.dumps(data))
    return data


def make_log(marker,offers=(),traced=False):
    t=int(traced)
    raw='LIVE workers ready, mode=shared processes=2\n'
    if marker['mtp_generation']:
        raw+=f'HETERO_MTP_PROBE_MODE state_trace={t} front_trace={t} repeat_round=0 relay_rounds=0\n'
        raw+=f'HETERO_MTP_GENERATION_MODE prompt={len(marker["prompt_ids"])} max_new={marker["max_new"]}\n'
        for pos,anchor,ids,submitted in offers:
            raw+=f'HETERO_MTP_GENERATION_OFFER cell={pos-1} anchor={anchor} tokens='+','.join(map(str,ids))+f', submitted={submitted}\n'
    return raw+'HETERO_GENERATION '+json.dumps(dict(prompt_tokens=len(marker['prompt_ids']),finish_reason=marker['finish_reason'],token_ids=marker['token_ids']))+'\n'


class GenerationTests(unittest.TestCase):
    def setUp(self):
        tmp=tempfile.TemporaryDirectory();self.addCleanup(tmp.cleanup)
        root=Path(tmp.name)
        self.r=root/'mtp';self.s=root/'serial'
        self.m=make_capture(self.r,LIVE,True)
        self.b=make_capture(self.s,[([x],1,[y]) for x,y in zip(ACTUAL,PRED)],False)
        self.offers=[(*p,n) for p,n in zip(POINTS,(4,2,1,4))]
        self.fresh=[]
        for i,point in enumerate(POINTS):
            p=root/f'f{i}';capture(p,[([x],1,[y]) for x,y in zip(ACTUAL[:point[0]],PRED[:point[0]])]);self.fresh.append(p)

    def gate(self,traced=False):
        return generation.analyze(self.s,make_log(self.b),self.r,make_log(self.m,self.offers,traced),
            [(p,'LIVE workers ready, mode=shared processes=2\n'+log(point,traced=traced)) for p,point in zip(self.fresh,POINTS)],traced)

    def test_actual_stfr_length(self):
        g=self.gate();self.assertEqual(g['status'],'PASS');self.assertEqual(g['output'],ACTUAL[2:]+[14]);self.assertEqual(g['accepted_drafts'],2);self.assertEqual(g['clipped_drafts'],5)

    def test_trace_and_changed_front(self):
        # Reuse the actual schema-2 trace fixture: same ACTUAL and PRED.
        self.assertEqual(fixture.ACTUAL,ACTUAL);self.assertEqual(fixture.PRED,PRED)
        drafts=trace(self.r,len(ACTUAL),tuple(p[0]-1 for p in POINTS))
        for p,point in zip(self.fresh,POINTS):trace(p,point[0],(point[0]-1,))
        self.assertEqual(self.gate(True)['status'],'PASS')
        raw=bytearray(drafts[-1].read_bytes());raw[-1]^=1;drafts[-1].write_bytes(raw)
        self.assertEqual(self.gate(True)['status'],'FAIL')

    def test_bad_offer_and_output(self):
        wrong=self.offers.copy();wrong[1]=(*POINTS[1],3)
        with self.assertRaisesRegex(ValueError,'clipping'):
            generation.load_generation(self.r,make_log(self.m,wrong),True)
        with self.assertRaisesRegex(ValueError,'log output'):
            generation.load_generation(self.r,make_log(dict(self.m,token_ids=[1]),self.offers),True)

    def test_stop_clips_to_anchor_only(self):
        m=make_capture(self.r,[*LIVE[:2],([3],1,[4])],True,stops=[4],max_new=6,output=[3],reason='stop',next_id=4,final=[4,5,6])
        _,info,_=generation.load_generation(self.r,make_log(m,[(2,3,[4,5,6],0),(3,4,[5,6],2)]),True)
        self.assertEqual(info['output'],[3]);self.assertEqual(info['reason'],'stop');self.assertEqual(info['points'][0]['submitted'],[])

    def test_immediate_stop_and_one_token_length(self):
        m=make_capture(self.r,LIVE[:2],True,stops=[3],output=[],reason='stop',next_id=3,final=[3,4])
        _,info,_=generation.load_generation(self.r,make_log(m,[(2,3,[4],1)]),True);self.assertEqual(len(info['points']),1)
        m=make_capture(self.r,LIVE[:2],True,max_new=1,output=[3],reason='length',next_id=3,final=[3,4])
        _,info,_=generation.load_generation(self.r,make_log(m,[(2,3,[4],1)]),True);self.assertEqual(info['output'],[3]);self.assertEqual(len(info['points']),1)


if __name__=='__main__':unittest.main()
