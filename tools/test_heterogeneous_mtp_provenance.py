import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import heterogeneous_mtp_provenance as provenance


class ProvenanceTests(unittest.TestCase):
    def setUp(self):
        d=tempfile.TemporaryDirectory();self.addCleanup(d.cleanup);root=Path(d.name)
        self.index=root/'index.txt';self.index.write_bytes(b'test index')
        self.shards=[root/'target-1.gguf',root/'target-2.gguf']
        for i,p in enumerate(self.shards):p.write_bytes(b'GGUF'+bytes([i]))
        self.audit=root/'audit.json';self.record=root/'record.json'
        self.digest=provenance.sha(self.index)
        self.patch=patch.object(provenance,'INDEX',self.digest);self.patch.start();self.addCleanup(self.patch.stop)
        self.audit.write_text(json.dumps(dict(status='PASS_ASSET_LAYOUT',raw_tensor_hashes_verified=True,
            pinned_revision=provenance.REVISION,target_index_sha256=self.digest,runtime_sha256=provenance.RUNTIME,
            target_shards=[str(p) for p in self.shards])))
        self.record.write_text(json.dumps(dict(schema='strata-target-conversion-provenance/v1',source_repo=provenance.REPO,
            source_revision=provenance.REVISION,target_index_sha256=self.digest,source_snapshot_manifest_sha256='f'*64,
            converter_revision='e'*40,target_shards=[dict(filename=p.name,bytes=p.stat().st_size,sha256=provenance.sha(p)) for p in self.shards])))

    def test_unresolved_without_external_record(self):
        r=provenance.review(self.audit,self.index,self.shards)
        self.assertEqual(r['status'],'UNRESOLVED');self.assertEqual(r['general_admission'],'NOT QUALIFIED')

    def test_match_external_record_not_admission(self):
        r=provenance.review(self.audit,self.index,self.shards,self.record,provenance.sha(self.record))
        self.assertEqual(r['status'],'MATCHED_AUTHORIZED_RECORD');self.assertEqual(r['general_admission'],'NOT QUALIFIED')

    def test_bad_digest_and_changed_bytes(self):
        with self.assertRaises(ValueError):provenance.review(self.audit,self.index,self.shards,self.record,'0'*64)
        self.shards[1].write_bytes(b'GGUFz')
        with self.assertRaisesRegex(ValueError,'bytes differ'):provenance.review(self.audit,self.index,self.shards,self.record,provenance.sha(self.record))

    def test_wrong_source_revision(self):
        d=json.loads(self.record.read_text());d['source_revision']='0'*40;self.record.write_text(json.dumps(d))
        with self.assertRaisesRegex(ValueError,'source or converter'):provenance.review(self.audit,self.index,self.shards,self.record,provenance.sha(self.record))


if __name__=='__main__':unittest.main()
