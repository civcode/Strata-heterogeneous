#!/usr/bin/env python3
"""Read-only target provenance review; no inferred or self-issued attestation."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import sys

REPO='Qwen/Qwen3.8-Flash-Next'
REVISION='de4b8e4d43b917e7706784d8bb445c9af86a3540'
INDEX='c344c01f59f547ed8a59003857a82eb90499e5299154241dbca787acb54b5b7c'
RUNTIME={'dense.txt':'991456cf4eb266eb8a888c349e548d11bc640f5dbd9149439aa9786af0f732fd',
         'dense.bin':'c724dc0b0822ada5d2977bf5bde821605feabaa64ea2e0045b67ca656329070a',
         'experts.bin':'09398406be61f1f54c93861f449e48b8df0bfccbc9ec9b2b7636775a6ea9244f',
         'draft_vocab.bin':'b1e1d3a7a9e4bf862dcd5923ce661fb59bbd07907e594df5cf86a62ac235cb91'}


def require(ok,message):
    if not ok:raise ValueError(message)


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(8<<20),b''):h.update(block)
    return h.hexdigest()


def hex_digest(value, length=64):
    return type(value) is str and bool(re.fullmatch('[0-9a-f]{'+str(length)+'}',value))


def review(audit_path,index_path,shards,record_path=None,trusted_digest=None):
    audit=json.loads(audit_path.read_text())
    require(audit.get('status')=='PASS_ASSET_LAYOUT' and audit.get('raw_tensor_hashes_verified') is True and
            audit.get('pinned_revision')==REVISION and audit.get('target_index_sha256')==INDEX and
            audit.get('runtime_sha256')==RUNTIME,'not the approved read-only asset audit')
    require(len(shards)==2 and len({p.resolve() for p in shards})==2 and
            [str(p.resolve()) for p in shards]==[str(Path(p).resolve()) for p in audit['target_shards']] and
            all(p.is_file() for p in shards) and sha(index_path)==INDEX,'supplied target files do not match approved audit/index')
    result=dict(status='UNRESOLVED',expected_source_repo=REPO,expected_source_revision=REVISION,
                target_index_sha256=INDEX,target_metadata=audit.get('target_metadata',{}),
                target_shards=[dict(path=str(p),bytes=p.stat().st_size) for p in shards],
                general_admission='NOT QUALIFIED',evidence_trust='operator must independently establish original conversion record provenance')
    require((record_path is None)==(trusted_digest is None),'conversion record and independently trusted digest must be supplied together')
    if record_path is None:
        result['reason']='No independently trusted original conversion record; GGUF metadata and MTP raw/layout hashes do not prove target source revision.'
        return result
    require(hex_digest(trusted_digest) and record_path.stat().st_size<=65536 and sha(record_path)==trusted_digest,
            'conversion record does not match externally trusted digest')
    record=json.loads(record_path.read_text())
    require(record.get('schema')=='strata-target-conversion-provenance/v1' and record.get('source_repo')==REPO and
            record.get('source_revision')==REVISION and record.get('target_index_sha256')==INDEX and
            hex_digest(record.get('source_snapshot_manifest_sha256')) and hex_digest(record.get('converter_revision'),40),
            'conversion record source or converter identity mismatch')
    expected=record['target_shards']
    require(type(expected) is list and len(expected)==2,'conversion record must bind both ordered target shards')
    for row,p,out in zip(expected,shards,result['target_shards']):
        require(row.get('filename')==p.name and type(row.get('bytes')) is int and row['bytes']==p.stat().st_size and
                hex_digest(row.get('sha256')),'conversion record shard identity mismatch')
        actual=sha(p)
        require(actual==row['sha256'],'target shard bytes differ from conversion record')
        out['sha256']=actual
    result.update(status='MATCHED_AUTHORIZED_RECORD',record_sha256=trusted_digest,
                  source_snapshot_manifest_sha256=record['source_snapshot_manifest_sha256'],
                  converter_revision=record['converter_revision'],
                  reason='Bytes and claimed revision match supplied trusted record; authenticity of record must be reviewed externally.')
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--asset-audit',type=Path,required=True)
    p.add_argument('--target-index',type=Path,required=True)
    p.add_argument('--target-shard',type=Path,action='append',required=True)
    p.add_argument('--conversion-record',type=Path)
    p.add_argument('--trusted-record-sha256',help='digest obtained independently from original conversion evidence; do not compute from a self-issued record')
    a=p.parse_args()
    try:
        r=review(a.asset_audit,a.target_index,a.target_shard,a.conversion_record,a.trusted_record_sha256)
        print(json.dumps(r,indent=2))
        return 0 if r['status']=='MATCHED_AUTHORIZED_RECORD' else 1
    except (OSError,ValueError,TypeError,AttributeError,KeyError) as exc:
        print(f'INVALID: {exc}',file=sys.stderr);return 2


if __name__=='__main__':sys.exit(main())
