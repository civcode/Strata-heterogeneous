#!/usr/bin/env python3
"""Read-only admission of vanilla Qwen Flash-Next MTP assets for the Q2_0 fork.

PASS means pinned manifest identity, matching target geometry and exact runtime
relayout from the local packed draft GGUF. It is not model numerical parity,
proof of the target conversion's checkpoint revision, or GPU memory admission.
No fetch, repack, setup, GPU initialization or writes to asset directories.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
from pathlib import Path
import struct
import sys

H, HC, FF, NE, VOCAB = 2560, 4, 640, 512, 248320
BLOB = 3 * H * FF * 18 // 64
GU = 'mtp.layers.0.mlp.experts.gate_up_proj'
DN = 'mtp.layers.0.mlp.experts.down_proj'


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest() if hasattr(hashlib, 'file_digest') else stream_sha(f)


def stream_sha(f):
    digest = hashlib.sha256()
    for block in iter(lambda: f.read(8 << 20), b''):
        digest.update(block)
    return digest.hexdigest()


def pinned_source(path):
    # Do not import mtp_fetch: it configures TLS and remote source discovery.
    fields = {}
    for node in ast.parse(path.read_text()).body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in ('SHA256', 'PINNED_REVISION'):
                    fields[target.id] = ast.literal_eval(node.value)
    require(len(fields['SHA256']) == 31, 'unexpected vanilla pinned MTP inventory')
    return fields['PINNED_REVISION'], fields['SHA256']


def check_manifest(root, pins, verify_raw=False):
    rows = json.loads((root / 'mtp-manifest.json').read_text())
    require(isinstance(rows, list) and len(rows) == len(pins), 'manifest tensor count')
    found = {}
    for row in rows:
        name = row['name']
        require(name in pins and name not in found and row['sha256'] == pins[name],
                f'unpinned or duplicate manifest tensor: {name}')
        shape = row['shape']
        require(row['dtype'] == 'BF16' and 1 <= len(shape) <= 3 and
                all(type(d) is int and d > 0 for d in shape) and
                row['bytes'] == 2 * math.prod(shape), f'invalid manifest shape: {name}')
        relative = Path(row['file'])
        require(relative == Path('tensors') / (name + '.bin'), f'unexpected raw tensor path: {name}')
        raw = root / relative
        require(raw.resolve().is_relative_to(root.resolve()) and raw.is_file() and
                raw.stat().st_size == row['bytes'], f'raw tensor missing/size mismatch: {name}')
        if verify_raw:
            require(sha(raw) == pins[name], f'raw tensor hash mismatch: {name}')
        found[name] = row
    require(set(found) == set(pins), 'missing pinned tensors')
    require(found[GU]['shape'] == [NE, 2 * FF, H] and found[DN]['shape'] == [NE, H, FF],
            'draft expert geometry mismatch')
    return found, hashlib.sha256(json.dumps(dict(pins), sort_keys=True).encode()).hexdigest()


def dense_index(path, dense_size):
    result, end = {}, 0
    for line in path.read_text().splitlines():
        parts = line.split()
        require(len(parts) == 6, 'malformed dense.txt row')
        name, kind, *numbers = parts
        rows, cols, offset, size = map(int, numbers)
        require(name not in result and rows > 0 and cols > 0 and size > 0 and
                offset == (end + 255) // 256 * 256, f'invalid dense extent: {name}')
        widths = {'bf16': (1, 2), 'f32': (1, 4), 'q8_0': (32, 34)}
        require(kind in widths, f'invalid dense kind: {name}')
        block, width = widths[kind]
        require(cols % block == 0 and size == rows * (cols // block) * width and
                offset + size <= dense_size, f'invalid dense size: {name}')
        result[name] = (kind, rows, cols, offset, size)
        end = offset + size
    require(bool(result) and end == dense_size, 'dense.bin extent mismatch')
    return result


def check_vocab(path, n_vocab=VOCAB):
    size = path.stat().st_size
    require(0 < size <= 4 * n_vocab and size % 4 == 0, 'invalid draft vocabulary extent')
    ids = [row[0] for row in struct.iter_unpack('<i', path.read_bytes())]
    require(len(set(ids)) == len(ids) and all(0 <= i < n_vocab for i in ids),
            'duplicate or out-of-range draft vocabulary ID')
    return len(ids)


def short_name(name):
    return name.removeprefix('mtp.').removeprefix('layers.0.')


def field(reader, name, text=False):
    item = reader.get_field(name)
    require(item is not None and len(item.data) == 1, f'missing/non-scalar GGUF field: {name}')
    data = item.parts[item.data[0]]
    return data.tobytes().decode('utf-8') if text else int(data[0])


def audit(root, pack, shards, expected_index, verify_raw):
    # Import the actual vanilla runtime conversion routines; no new quantizer.
    import numpy as np
    import mtp_rt as rt
    import gguf

    require(len(shards) == 2 and len(set(shards)) == 2, 'expected the two approved Q2_0 shards')
    require(sha(pack / 'index.txt') == expected_index, 'target pack index differs from approved config')
    revision, pins = pinned_source(Path(__file__).with_name('mtp_fetch.py'))
    manifest, source = check_manifest(root, pins, verify_raw)
    target = [gguf.GGUFReader(str(path), mode='r') for path in shards]
    primary = target[0]
    require(field(primary, 'general.architecture', True) == 'qwen4exp', 'target architecture mismatch')
    for key, want in {'embedding_length': H, 'block_count': 48, 'expert_count': NE,
                      'expert_feed_forward_length': FF, 'attention.head_count': 24,
                      'attention.head_count_kv': 2, 'hyper_connection.count': HC}.items():
        require(field(primary, 'qwen4exp.' + key) == want, f'target geometry mismatch: {key}')
    required = {'token_embd.weight', 'output.weight'}
    heads = [t for reader in target for t in reader.tensors if t.name in required]
    require(len(heads) == 2 and {t.name for t in heads} == required and
            all(tuple(int(x) for x in t.shape) == (H, VOCAB) for t in heads), 'target embedding/head geometry mismatch')
    vocab = primary.get_field('tokenizer.ggml.tokens')
    require(vocab is not None and len(vocab.data) == VOCAB, 'target tokenizer vocabulary extent mismatch')
    target_names = {}
    for key in ('general.name', 'general.basename', 'general.source.repo_url', 'general.base_model.0.name',
                'general.base_model.0.repo_url', 'general.base_model.0.version'):
        if primary.get_field(key) is not None:
            target_names[key] = field(primary, key, True)
    draft_path = root / 'mtp-q2_0.gguf'
    draft = gguf.GGUFReader(str(draft_path), mode='r')
    require(field(draft, 'general.architecture', True) == 'qwen4exp-mtp' and
            field(draft, 'strata.mtp.expert_format', True) == 'q2_0' and
            field(draft, 'strata.mtp.source_sha256', True) == source, 'draft GGUF source/format metadata mismatch')
    tensors = {t.name: t for t in draft.tensors}
    require(len(tensors) == len(draft.tensors) and set(tensors) == set(manifest), 'draft GGUF inventory mismatch')
    for name, t in tensors.items():
        require(tuple(int(x) for x in t.shape) == tuple(reversed(manifest[name]['shape'])), f'GGUF tensor shape mismatch: {name}')
    runtime = root / 'rt'
    dense, experts = runtime / 'dense.bin', runtime / 'experts.bin'
    idx = dense_index(runtime / 'dense.txt', dense.stat().st_size)
    require(set(idx) == {short_name(n) for n in tensors if n not in (GU, DN)}, 'runtime dense inventory mismatch')
    with dense.open('rb') as f:
        for name, t in tensors.items():
            if name in (GU, DN):
                continue
            short = short_name(name)
            kind, rows, cols, offset, size = idx[short]
            data = np.asarray(t.data)
            if int(t.tensor_type) == 0:
                arr = data.astype(np.float32) + 1.0
                raw, shape = arr.tobytes(), ('f32', 1, arr.size)
            else:
                require(t.tensor_type == gguf.GGMLQuantizationType.BF16, f'unsupported dense source type: {name}')
                u16 = data.view(np.uint16) if data.dtype != np.uint16 else data
                u16 = u16.reshape(data.shape[0], -1)
                r, c = u16.shape
                if short in rt.Q8:
                    raw, shape = rt.q8_0((u16.astype(np.uint32) << 16).view(np.float32)), ('q8_0', r, c)
                else:
                    raw, shape = u16.tobytes(), ('bf16', r, c)
            require((kind, rows, cols) == shape and size == len(raw), f'runtime dense shape/type mismatch: {name}')
            f.seek(offset)
            require(f.read(size) == raw, f'runtime dense bytes differ from vanilla conversion: {name}')
    require(rt.BLOB == BLOB and experts.stat().st_size == NE * BLOB, 'draft expert blob extent mismatch')
    gu, dn = (np.asarray(tensors[n].data) for n in (GU, DN))
    require(all(tensors[n].tensor_type == gguf.GGMLQuantizationType.Q2_0 for n in (GU, DN)) and
            gu.shape == (NE, 2 * FF, H // 64 * 18) and dn.shape == (NE, H, FF // 64 * 18),
            'draft GGUF expert layout mismatch')
    with experts.open('rb') as f:
        for e in range(NE):
            require(f.read(BLOB) == rt.blob_of(gu[e], dn[e]), f'runtime expert relayout differs: {e}')
    n_subset = check_vocab(runtime / 'draft_vocab.bin')
    return {'status': 'PASS_ASSET_LAYOUT', 'pinned_revision': revision, 'source_manifest_sha256': source,
            'raw_tensor_hashes_verified': verify_raw, 'draft_dense_tensors': len(idx), 'expert_blobs': NE,
            'expert_bytes': experts.stat().st_size, 'dense_bytes': dense.stat().st_size,
            'draft_vocab_rows': n_subset, 'target_index_sha256': expected_index, 'target_metadata': target_names,
            'target_checkpoint_revision_verified': False, 'gpu_qualified': False,
            'runtime_sha256': {p: sha(runtime / p) for p in ('dense.txt', 'dense.bin', 'experts.bin', 'draft_vocab.bin')},
            'packed_draft_sha256': sha(draft_path), 'target_shards': [str(p) for p in shards]}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mtp-root', required=True, type=Path)
    p.add_argument('--target-pack', required=True, type=Path)
    p.add_argument('--target-shard', required=True, action='append', type=Path)
    p.add_argument('--target-index-sha256', required=True)
    p.add_argument('--verify-raw', action='store_true')
    a = p.parse_args()
    try:
        result = audit(a.mtp_root, a.target_pack, a.target_shard, a.target_index_sha256, a.verify_raw)
    except (OSError, ValueError, TypeError, KeyError, IndexError, AttributeError, ImportError, RuntimeError) as exc:
        print(f'MTP ASSET ADMISSION FAILED: {exc}', file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2))
    return 0


if __name__ == '__main__':
    sys.exit(main())
