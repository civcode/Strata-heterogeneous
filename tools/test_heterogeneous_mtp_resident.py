"""Actual synthetic STFR and trace readers; no device inference."""
import json
from pathlib import Path
import struct
import tempfile
import unittest

import heterogeneous_compare_parity as parity
import heterogeneous_mtp_resident as resident
from test_heterogeneous_mtp_generation import make_capture, make_log, LIVE, POINTS, ACTUAL
from test_heterogeneous_mtp_capture_integration import PLAN, trace

OFFERS=[(*point,n) for point,n in zip(POINTS,(4,2,1,4))]


def rewrite_request(path,request):
    raw=bytearray(path.read_bytes())
    struct.pack_into('<Q',raw,24,request)
    checksum=2166136261
    for i,b in enumerate(raw):
        if not 52<=i<56:
            checksum=((checksum ^ b)*16777619)&0xffffffff
    struct.pack_into('<I',raw,52,checksum)
    path.write_bytes(raw)


class ResidentTests(unittest.TestCase):
    def setUp(self):
        tmp=tempfile.TemporaryDirectory();self.addCleanup(tmp.cleanup)
        self.root=Path(tmp.name);self.prefix=self.root/'resident'

    def fixture(self,traced=False,count=7):
        refs=[]
        for i in (0,1):
            prompt=[1+i*14,2]
            windows=[([prompt[0]],1,[2])]+LIVE[1:]
            path=self.root/f'fresh{i}'
            m=make_capture(path,windows,True);m['prompt_ids']=prompt
            Path(str(path)+'.complete.json').write_text(json.dumps(m))
            refs.append((path,make_log(m,OFFERS,traced)))
            if traced:trace(path,len(ACTUAL),tuple(p[0]-1 for p in POINTS))
        mode=int(traced)
        log=f'LIVE workers ready, mode=shared processes=2\nHETERO_MTP_PROBE_MODE state_trace={mode} front_trace={mode} repeat_round=0 relay_rounds=0\nHETERO_MTP_GENERATION_MODE prompt=2 max_new=6\n'
        for i in range(1,count+1):
            prompt=[15 if i==2 else 1,2]
            windows=[([prompt[0]],1,[2])]+LIVE[1:]
            cancelled=i in (4,6)
            if cancelled:windows=windows[:4 if i==4 else 3]
            fed=sum(k for _,k,_ in windows)
            next_id=windows[-1][2][windows[-1][1]-1]
            path=Path(str(self.prefix)+f'.r{i}')
            m=make_capture(path,windows,True,output=[] if cancelled else None,
                reason='cancelled' if cancelled else 'length',next_id=next_id,final=[] if cancelled else None)
            m.update(prompt_ids=prompt,mtp_resident=True,request_id=i,fed_rows=fed)
            Path(str(path)+'.complete.json').write_text(json.dumps(m))
            log+=f'HETERO_MTP_RESIDENT_RESET request={i}\nHETERO_MTP_RESIDENT_REQUEST request={i}\n'
            pos=0
            for w,(_,keep,_) in enumerate(windows,1):
                rewrite_request(Path(f'{path}.w{w}.stfr'),i)
                if w>2:
                    a,anchor,tokens,submitted=OFFERS[w-3]
                    log+=f'HETERO_MTP_GENERATION_OFFER cell={a-1} anchor={anchor} tokens='+','.join(map(str,tokens))+f', submitted={submitted}\n'
                pos+=keep
                log+=f'HETERO_PROGRESS {i} {w} {pos}\n'
            if not cancelled:
                a,anchor,tokens,submitted=OFFERS[-1]
                log+=f'HETERO_MTP_GENERATION_OFFER cell={a-1} anchor={anchor} tokens='+','.join(map(str,tokens))+f', submitted={submitted}\n'
            log+='HETERO_GENERATION '+json.dumps(dict(request_id=i,prompt_tokens=2,stop_token=-1,finish_reason=m['finish_reason'],token_ids=m['token_ids']))+'\n'
            if traced:
                n=fed-1 if cancelled else fed
                trace(path,n,tuple(p[0]-1 for p in POINTS if p[0]-1<n))
        Path(str(self.prefix)+'.resident.json').write_text(json.dumps(dict(schema='strata-mtp-resident/v1',
            requests=count,prompt_rows=2,max_new=6,transport='shared',traced=traced,plan=PLAN,model_fingerprint=1)))
        return log+'HETERO_CLOSED\n',refs

    def test_seven_request_cancel_recovery(self):
        log,refs=self.fixture()
        g=resident.analyze(self.prefix,log,refs)
        self.assertEqual(g['status'],'PASS')
        self.assertTrue(g['cancellation_coverage'])
        self.assertEqual([r['request'] for r in g['requests'] if r['cancelled']],[4,6])

    def test_aba_traced_and_front_mismatch(self):
        log,refs=self.fixture(True,3)
        self.assertEqual(resident.analyze(self.prefix,log,refs)['status'],'PASS')
        path=sorted(self.root.glob('resident.r3.mtp.*.front.bin'))[-1]
        raw=bytearray(path.read_bytes());raw[-1]^=1;path.write_bytes(raw)
        self.assertEqual(resident.analyze(self.prefix,log,refs)['status'],'FAIL')

    def test_cancel_trace_prefix(self):
        log,refs=self.fixture(True)
        self.assertEqual(resident.analyze(self.prefix,log,refs)['status'],'PASS')

    def test_request_identity_and_closed_required(self):
        log,refs=self.fixture()
        with self.assertRaisesRegex(ValueError,'lifecycle'):resident.analyze(self.prefix,log.replace('HETERO_CLOSED\n',''),refs)
        with self.assertRaisesRegex(ValueError,'ordering'):resident.analyze(self.prefix,log.replace(
            'HETERO_MTP_RESIDENT_RESET request=2\nHETERO_MTP_RESIDENT_REQUEST request=2\n',
            'HETERO_MTP_RESIDENT_REQUEST request=2\nHETERO_MTP_RESIDENT_RESET request=2\n'),refs)
        packet=Path(str(self.prefix)+'.r3.w1.stfr');rewrite_request(packet,1)
        with self.assertRaises(parity.ParityError):resident.analyze(self.prefix,log,refs)

    def test_old_reader_refuses_resident_capture(self):
        self.fixture(count=3)
        with self.assertRaises(parity.ParityError):
            parity.read_run(Path(str(self.prefix)+'.r1'),generation_capture=True,allow_partial=True)


if __name__=='__main__':unittest.main()
