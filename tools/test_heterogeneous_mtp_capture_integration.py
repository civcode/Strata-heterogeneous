"""Synthetic protocol bytes only: real capture readers, no GPU or inference math."""
import json
from pathlib import Path
import struct
import tempfile
import unittest

import heterogeneous_compare_parity as parity
import heterogeneous_mtp_front as front
import heterogeneous_mtp_three_round as three

PLAN = 'cuda:0:0-12,hip:0:12-30,hip:1:30-48'
PROMPT = [1,2]
# Inputs, greedy keep, native target predictions (fixture values only).
WINDOWS = [([1],1,[2]),([2],1,[3]),([3,4,5,6,7],2,[4,8,9,10,11]),
           ([8,12,13,14,15],1,[9,20,21,22,23]),([9,10,11,12,13],2,[10,14,15,16,17])]
ACTUAL = [1,2,3,4,8,9,10]
PRED = [2,3,4,8,9,10,14]
POINTS = [(2,3,[4,5,6,7]),(4,8,[12,13,14,15]),(5,9,[10,11,12,13]),(7,14,[16,17,18,19])]


def put(path, value):
    Path(path).write_text(json.dumps(value))


def capture(prefix, windows, live=False):
    put(f'{prefix}.complete.json',dict(schema=parity.SCHEMA,windows=len(windows),plan=PLAN,
        model_fingerprint=1,mtp_fresh_replay=not live,mtp_three_round=live,
        mtp_three_round_variable=live,mtp_prompt_windows=len(PROMPT) if live else 0))
    pos = 0
    for w,(ids,keep,preds) in enumerate(windows,1):
        put(f'{prefix}.w{w}.json',dict(schema=parity.SCHEMA,plan=PLAN,model_fingerprint=1,plan_fingerprint=123,
            pack_index_sha256='a'*64,gguf_shards='/model/a;/model/b',profile_path='/profile',window_id=w,
            pos0=pos,rows=len(ids),n_keep=keep,token_ids=ids))
        logits = [float(v==pred) for pred in preds for v in range(32)]
        payload = struct.pack(f'<{len(ids)}iI{len(logits)}f',*preds,0,*logits)
        blob = bytearray(parity.HEADER.pack(parity.MAGIC,1,1,2,len(ids),32,0,1,w,123,len(payload),0,0)+payload)
        checksum = 2166136261
        for i,b in enumerate(blob):
            if not 52 <= i < 56:
                checksum = ((checksum ^ b)*16777619)&0xffffffff
        struct.pack_into('<I',blob,52,checksum)
        Path(f'{prefix}.w{w}.stfr').write_bytes(blob)
        pos += keep


def trace(prefix, count, draft_cells):
    drafts = []
    pairs = ACTUAL[1:count]+[PRED[count-1]]
    event = 0
    for cell,token in enumerate(pairs):
        r = bytes([cell+1])*40960
        phase = 'draft' if cell in draft_cells else 'prime'
        for suffix,upto in [('before',cell),('after',cell+1)]:
            event += 1
            stem = f'{prefix}.mtp.{event}'
            has_front = phase=='draft' and suffix=='after'
            put(stem+'.json',dict(schema='strata-mtp-state/v2',event=event,phase=phase+'_'+suffix,
                cell=cell,upto=upto,tokens=[token],residual_stride=40960,format=0,
                kv_strides=[1024,1024,0,0] if upto else [0]*4,
                front_schema=2 if has_front else 0,front_bytes=front.BYTES if has_front else 0))
            kv = b''.join(bytes([i+1])*1024 for i in range(upto))
            for field,raw in [('r',r),('k',kv),('v',kv),('ks',b''),('vs',b'')]:
                Path(stem+'.'+field+'.bin').write_bytes(raw)
            if has_front:
                step=struct.pack('<4i',cell,cell+1,(cell+1)//4,cell+1)
                pos=struct.pack('<24i',*([cell]*24))
                actual=dict(token=struct.pack('<i',token),step=step,position=pos,rest_step=step,
                            rest_position=pos,residual=r,source_before=r,source_after=r)
                raw=b''.join(actual.get(name,bytes(size)) for name,size in front.FIELDS)
                dest=Path(stem+'.front.bin')
                dest.write_bytes(raw)
                drafts.append(dest)
    return drafts


def log(point, live=False, traced=False):
    n,anchor,tokens=point
    t=int(traced)
    return (f'HETERO_MTP_PROBE_MODE state_trace={t} front_trace={t} repeat_round=0 relay_rounds={3 if live else 0}\n'
            f'HETERO_MTP_PROBE_DRAFT cell={n-1} verified_next={anchor} count={len(tokens)} tokens='+
            ','.join(map(str,tokens))+', (not submitted to target)\n')


class CaptureIntegration(unittest.TestCase):
    def setUp(self):
        tmp=tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        root=Path(tmp.name)
        self.reused=root/'reused'
        capture(self.reused,WINDOWS,True)
        self.fresh=[]
        for i,point in enumerate(POINTS):
            prefix=root/f'fresh{i}'
            capture(prefix,[([x],1,[p]) for x,p in zip(ACTUAL[:point[0]],PRED[:point[0]])])
            self.fresh.append(prefix)

    def gate(self,traced=False):
        return three.analyze(self.reused,log(POINTS[-1],True,traced),
                             [(p,log(x,traced=traced)) for p,x in zip(self.fresh,POINTS)],traced,PROMPT)

    def test_actual_stfr_and_mode_markers(self):
        result=self.gate()
        self.assertEqual(result['status'],'PASS')
        self.assertEqual([p['prefix'] for p in result['checkpoints']],[ACTUAL[:p[0]] for p in POINTS])
        self.assertEqual(result['accepted_drafts'],2)
        self.assertEqual(result['rejected_drafts'],10)
        with self.assertRaises(parity.ParityError):
            parity.read_run(self.reused,allow_partial=True)
        marker=Path(str(self.reused)+'.complete.json')
        d=json.loads(marker.read_text())
        d['mtp_prompt_windows']=8
        put(marker,d)
        with self.assertRaises(parity.ParityError):
            self.gate()

    def test_corrupt_packet_refused(self):
        packet=Path(f'{self.reused}.w3.stfr')
        raw=bytearray(packet.read_bytes())
        raw[-1]^=1
        packet.write_bytes(raw)
        with self.assertRaisesRegex(ValueError,'checksum'):
            self.gate()

    def test_traced_four_checkpoints_and_bad_front(self):
        drafts=trace(self.reused,len(ACTUAL),tuple(p[0]-1 for p in POINTS))
        for prefix,point in zip(self.fresh,POINTS):
            trace(prefix,point[0],(point[0]-1,))
        result=self.gate(True)
        self.assertEqual(result['status'],'PASS')
        self.assertTrue(all(p['committed_kv']=='IDENTICAL' for p in result['checkpoints']))
        raw=bytearray(drafts[2].read_bytes())
        raw[-1]^=1
        drafts[2].write_bytes(raw)
        failed=self.gate(True)
        self.assertEqual(failed['status'],'FAIL')
        self.assertEqual(failed['checkpoints'][2]['front']['first_different_field'],'source_after')


if __name__=='__main__':
    unittest.main()
