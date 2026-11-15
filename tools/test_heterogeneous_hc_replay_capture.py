"""Synthetic STFR/STL0 and full-size HC raw data; not GPU results."""
import json
from pathlib import Path
import struct
import tempfile
import unittest

import numpy as np
import heterogeneous_compare_parity as parity
import heterogeneous_layer0_arithmetic as l0
import heterogeneous_prefill_parity as prefill
import heterogeneous_hc_replay as h

PLAN='cuda:0:0-12,hip:0:12-30,hip:1:30-48'
LOG='STRATA_GR0_PATH down_q8=0 up_q8=0 down_scale=0 up_scale=0 qfuse=1 hc_q8_env=(unset)\n'
PCI='0000:04:00.0'


def checksum(blob, offset):
    digest=2166136261
    for i,v in enumerate(blob):
        if not offset<=i<offset+4:
            digest=((digest^v)*16777619)&0xffffffff
    struct.pack_into('<I',blob,offset,digest)
    return blob


def document(path, value):
    path.write_text(json.dumps(value))


class CaptureTests(unittest.TestCase):
    def setUp(self):
        tmp=tempfile.TemporaryDirectory();self.addCleanup(tmp.cleanup)
        self.root=Path(tmp.name);self.p=self.root/'actual';self.r=self.root/'replay';self.r.mkdir()
        for mode in ('inputs','native','serial-a','serial-b'):(self.r/mode).mkdir()
        n,hc,rank=h.N,h.HC,h.LR;d=n*hc
        inp,pf,ver=h.shapes()
        embedding=np.full(n,1,dtype='<f4')
        inputs={k:np.zeros(count,dtype=h.dtype(k)) for k,count in inp.items()}
        inputs['residual.f32']=np.tile(embedding,hc);inputs['norm_weight.f32'].fill(1)
        prompt={k:np.zeros(count,dtype=h.dtype(k)) for k,count in pf.items()}
        prompt['scale.f32'].fill(1);prompt['normalized.bf16'].fill(0x3f80);prompt['mixed.f32'].fill(.5)
        serial={k:prompt[k].copy() for k in ver};serial['mixed.f32'][0]=.51
        q8=bytes([3])*(n//32*36)
        for k,a in inputs.items():
            Path(f'{self.p}.hc.{k}').write_bytes(a.tobytes());(self.r/'inputs'/k).write_bytes(a.tobytes())
        for k,a in prompt.items():
            Path(f'{self.p}.hc.{k}').write_bytes(a.tobytes());(self.r/'native'/k).write_bytes(a.tobytes())
        for req,mode in ((1,'serial-a'),(3,'serial-b')):
            for k,a in serial.items():(self.r/mode/k).write_bytes(a.tobytes())
            (self.r/mode/'mixed.q8_1').write_bytes(q8)
            for k,a in {'residual.f32':inputs['residual.f32'],'scale.f32':serial['scale.f32'],
                        'inject.f32':serial['inject.f32']}.items():
                Path(f'{self.p}.hc.serial-r{req}.{k}').write_bytes(a.tobytes())
            b=bytearray(l0.HEADER.pack(l0.MAGIC,1,0,n,10240,len(q8),1,48,req,1,0,123,0,0))
            b+=struct.pack('<i',1)+embedding.tobytes()+serial['mixed.f32'].tobytes()+q8+bytes(10240*4)
            Path(f'{self.p}.stage0.r{req}.w1.stl0').write_bytes(checksum(b,64))
        Path(f'{self.p}.first.r2.embedding.f32').write_bytes(embedding.tobytes())
        Path(f'{self.p}.first.r2.hc_mixed.f32').write_bytes(prompt['mixed.f32'].tobytes())
        for req,mode in enumerate(prefill.MODES,1):
            stem=f'{self.p}.{mode}.w2'
            document(Path(stem+'.json'),dict(schema=parity.SCHEMA,plan=PLAN,model_fingerprint=9,
                plan_fingerprint=123,pack_index_sha256='a'*64,gguf_shards='/model/a;/model/b',profile_path='/profile',
                window_id=2,rows=1,n_keep=1,pos0=1,token_ids=[2]))
            payload=struct.pack('<iI4f',3,0,0.,1.,2.,4.)
            b=bytearray(parity.HEADER.pack(parity.MAGIC,1,1,2,1,4,0,req,2,123,len(payload),0,0)+payload)
            Path(stem+'.stfr').write_bytes(checksum(b,52))
        for stage,suffix in ((0,'out'),(1,'in')):
            Path(f'{self.p}.stage{stage}.r2.w1.{suffix}.f32').write_bytes(bytes(d*4))
        document(Path(f'{self.p}.prefill.json'),dict(schema=prefill.PREFILL_SCHEMA,plan=PLAN,model_fingerprint=9,
            prefix_rows=1,chunk=1,hc=hc,n_embd=n,token_ids=[1,2],state_capture=True,
            first_token_capture=True,hc_replay_capture=True))
        document(self.r/'complete.json',dict(schema='strata-native-hc-replay/v1',cuda_device=0,pci=PCI,
            qfuse=True,q8_written=True,variant=1,n_embd=n,hc=hc,rank=rank,rows=1))

    def gate(self):
        return h.analyze(self.p,LOG,self.r,PCI)

    def test_replay_and_changed_phase(self):
        g=self.gate()
        self.assertEqual(g['status'],'REPRODUCED_NATIVE_HC_OPERATOR')
        self.assertEqual(g['first_different_observed_common_field'],'mixed.f32')
        self.assertEqual(g['production_admission'],'NOT_QUALIFIED')
        path=self.r/'native'/'down.f32';a=np.frombuffer(path.read_bytes(),dtype='<f4').copy();a[0]=1
        path.write_bytes(a.tobytes());g=self.gate()
        self.assertEqual(g['status'],'NOT_REPRODUCED')
        self.assertFalse(g['prompt_intermediates']['down.f32']['bytes_equal'])

    def test_different_replay_inputs(self):
        path=self.r/'inputs'/'norm_weight.f32';a=np.frombuffer(path.read_bytes(),dtype='<f4').copy();a[0]=2
        path.write_bytes(a.tobytes())
        with self.assertRaisesRegex(ValueError,'different input bytes'):self.gate()

    def test_wrong_serial_request_and_residual(self):
        a=Path(f'{self.p}.stage0.r1.w1.stl0');b=Path(f'{self.p}.stage0.r3.w1.stl0');raw=b.read_bytes();b.write_bytes(a.read_bytes())
        with self.assertRaisesRegex(ValueError,'STL0 header'):self.gate()
        b.write_bytes(raw)
        path=Path(f'{self.p}.hc.serial-r3.residual.f32');arr=np.frombuffer(path.read_bytes(),dtype='<f4').copy();arr[0]=0
        path.write_bytes(arr.tobytes())
        with self.assertRaisesRegex(ValueError,'serial HC residual'):self.gate()


if __name__=='__main__':unittest.main()
