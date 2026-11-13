#!/usr/bin/env python3
"""M2.14 CPU-only, integrity-checked comparison of internal stage-0 phases.

STP0/v1 stores [12 layers][4 phases][2560 FP32] for one committed token.
Phases: first HC mixed, attention/GDN bo, second HC mixed, MoE-combined bo.
A phase's relative L2 threshold is diagnostic, never final-logit parity.
"""
from __future__ import annotations

import argparse
from array import array
import json
import math
from pathlib import Path
import struct
import sys

import heterogeneous_compare_parity as parity

HEADER = struct.Struct('<8IQQqQII')
MAGIC = 0x30505453  # 'STP0'
PHASES = ('hc0_mixed', 'attention_gdn_bo', 'hc1_mixed', 'moe_bo')
LAYERS = 12
WIDTH = 2560
FLOATS = LAYERS * len(PHASES) * WIDTH
REFERENCE_PLAN = 'hip:0:0-48'
HETERO_PLAN = 'cuda:0:0-12,hip:0:12-30,hip:1:30-48'


def require(ok: bool, message: str) -> None:
    if not ok:
        raise parity.ParityError(message)


def read_packet(path: Path) -> dict:
    require(path.is_file(), f'missing STP0 packet: {path}')
    size = path.stat().st_size
    require(size == HEADER.size + 4 + FLOATS * 4, 'STP0 size mismatch')
    blob = path.read_bytes()
    (magic, version, stage, layers, phases, width, rows, total,
     request, window, pos0, fingerprint, checksum, reserved) = HEADER.unpack_from(blob)
    require(magic == MAGIC and version == 1 and stage == 0 and
            layers == LAYERS and phases == len(PHASES) and width == WIDTH and
            rows == 1 and total == 48 and reserved == 0 and
            request == 1 and window > 0 and pos0 >= 0 and fingerprint > 0,
            'STP0 header, dimensions or identity invalid')
    digest = 2166136261
    for index, value in enumerate(blob):
        if 64 <= index < 68:
            continue
        digest = ((digest ^ value) * 16777619) & 0xffffffff
    require(digest == checksum, 'STP0 checksum mismatch')
    token, = struct.unpack_from('<i', blob, HEADER.size)
    require(token >= 0, 'STP0 invalid token')
    samples = array('f')
    samples.frombytes(blob[HEADER.size + 4:])
    if sys.byteorder != 'little':
        samples.byteswap()
    require(len(samples) == FLOATS and all(math.isfinite(x) for x in samples),
            'STP0 nonfinite or incomplete phase values')
    return {'request': request, 'window': window, 'pos0': pos0,
            'fingerprint': fingerprint, 'token': token, 'samples': samples}


def read_run(prefix: Path) -> list[tuple[dict, dict]]:
    windows = parity.read_run(prefix)  # requires successful native shutdown
    result = []
    for meta, stfr in windows:
        require(meta['rows'] == 1 and meta['n_keep'] == 1,
                'STP0 supports only committed single-row windows')
        packet = read_packet(Path(f'{prefix}.stage0.w{meta["window_id"]}.stp0'))
        require(packet['window'] == stfr['window'] and
                packet['request'] == stfr['request'] and
                packet['pos0'] == meta['pos0'] and
                packet['fingerprint'] == stfr['fingerprint'] and
                packet['token'] == meta['token_ids'][0],
                'STP0 token, position, request or plan identity mismatch')
        result.append((meta, packet))
    return result


def metric(a: array, b: array) -> dict:
    require(len(a) == len(b) == WIDTH, 'STP0 phase shape mismatch')
    ref2 = diff2 = max_abs = 0.0
    mismatch = 0
    for x, y in zip(a, b):
        x, y = float(x), float(y)
        d = x - y
        ref2 += x * x
        diff2 += d * d
        max_abs = max(max_abs, abs(d))
        mismatch += (x != y)
    return {'relative_l2': math.sqrt(diff2) / max(math.sqrt(ref2), 1e-30),
            'rmse': math.sqrt(diff2 / WIDTH),
            'max_abs': max_abs, 'mismatched_values': mismatch}


def compare(reference: list[tuple[dict, dict]],
            heterogeneous: list[tuple[dict, dict]], threshold: float) -> dict:
    require(bool(reference) and len(reference) == len(heterogeneous),
            'missing or unequal committed windows')
    require(math.isfinite(threshold) and threshold > 0,
            'invalid phase diagnostic threshold')
    require(reference[0][0]['plan'] == REFERENCE_PLAN and
            heterogeneous[0][0]['plan'] == HETERO_PLAN,
            'STP0 requires unsplit HIP reference and original three-GPU plan')
    metrics, first_rows = [], []
    for (rm, r), (hm, h) in zip(reference, heterogeneous):
        for field in ('model_fingerprint', 'pack_index_sha256', 'gguf_shards',
                      'profile_path', 'pos0', 'rows', 'n_keep', 'window_id', 'token_ids'):
            require(rm[field] == hm[field], f'STP0 model/input mismatch: {field}')
        require(r['token'] == h['token'] and r['pos0'] == h['pos0'],
                'STP0 mismatched token/position')
        first = None
        for layer in range(LAYERS):
            for phase, name in enumerate(PHASES):
                begin = (layer * len(PHASES) + phase) * WIDTH
                result = metric(r['samples'][begin:begin + WIDTH],
                                h['samples'][begin:begin + WIDTH])
                if first is None and result['relative_l2'] > threshold:
                    first = {'layer': layer, 'phase': name,
                             'relative_l2': result['relative_l2']}
                metrics.append({'window': rm['window_id'],
                                'position': rm['pos0'],
                                'layer': layer, 'phase': name, **result})
        first_rows.append({'window': rm['window_id'], 'position': rm['pos0'],
                           'input_token': r['token'],
                           'first_phase_above_threshold': first})
    return {'status': 'DIAGNOSTIC_ONLY', 'threshold': threshold,
            'caution': 'STP0 phase thresholds are not M2.10 KL or proof of kernel root cause',
            'first_crossings': first_rows, 'phase_metrics': metrics}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference-prefix', required=True, type=Path)
    parser.add_argument('--heterogeneous-prefix', required=True, type=Path)
    parser.add_argument('--relative-threshold', default=0.001, type=float)
    parser.add_argument('--report', type=Path)
    args = parser.parse_args()
    try:
        result = compare(read_run(args.reference_prefix),
                         read_run(args.heterogeneous_prefix),
                         args.relative_threshold)
        print('M2.14 STAGE-0 PHASE DIAGNOSTIC ONLY')
        for item in result['first_crossings']:
            print(f'position={item["position"]} token={item["input_token"]} '
                  f'first_above_threshold={item["first_phase_above_threshold"]}')
            for m in result['phase_metrics']:
                if m['position'] != item['position']:
                    continue
                print(f'  layer={m["layer"]:02d} phase={m["phase"]:<17} '
                      f'relL2={m["relative_l2"]:.8g} '
                      f'rmse={m["rmse"]:.8g} maxAbs={m["max_abs"]:.8g}')
        if args.report:
            require(not args.report.exists(), 'report already exists; choose a fresh path')
            args.report.write_text(json.dumps(result, indent=2) + '\n',
                                   encoding='utf-8')
        return 0
    except (parity.ParityError, ValueError, OSError, OverflowError, MemoryError,
            struct.error) as exc:
        print(f'M2.14 stage-zero phase comparison failed: {exc}', file=sys.stderr)
        return 2


if __name__ == '__main__':
    sys.exit(main())
