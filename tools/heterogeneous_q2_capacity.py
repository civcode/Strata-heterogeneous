#!/usr/bin/env python3
"""Analytical Q2_0 owner lower bounds, NOT a VRAM admission check.

Specific to the measured 0039 Flash-Next geometry, 2K resident stage arenas,
max_t=8 and the indicated placements. See report-0041 for exclusions. The
increment follows session_bytes/qsa_state_bytes/qsa_buffers_bytes; it is not
an extrapolation of total process VRAM. Only 256-aligned sizes accepted, preserving native arena alignment.
"""
import argparse
import json

# session, weight arena, resident expert slots, QSA count; all observed in 0039
OWNERS = {
    'mixed': [(56300032, 2357877248, 10616832000, 3),
              (64429568, 2628297088, 12740198400, 5),
              (55461888, 2438140544, 10616832000, 4)],
    'hip': [(83203072, 3024275200, 16986931200, 6),
            (83203072, 3115493120, 16986931200, 6)],
}
BASE_CELLS = 2048
RESERVE = 1 << 30
# NativeDense::weight_bytes_for, 0042 Q2 metadata and selected device defaults.
# These preflight estimates do not include all load-time/runtime allocations.
DENSE_ESTIMATES = {'mixed': (480043520, 596049920, 543535616),
                   'hip': (766984192, 852644864)}


def estimate(placement, cells, mode):
    if placement not in OWNERS or mode not in ('int8', 'f16'):
        raise ValueError('unsupported placement or KV mode')
    if not BASE_CELLS <= cells <= 262144 or cells % 256:
        raise ValueError('cells must be a multiple of 256 between 2048 and 262144')
    result = []
    for base, weight, experts, q in OWNERS[placement]:
        # Q8: 1024 code bytes + 32 bytes of FP16 scales per cell;
        # F16: 2048 bytes. Indexer 128 and page table 1 byte/cell/layer.
        kv = 1056 if mode == 'int8' else 2048
        # Q8 pool bookkeeping and final arena alignment make the simple
        # precision delta an overestimate by up to 512 B for these owners.
        # Both F16 baselines are measured separately.
        f16_measured = (62394624, 74587392, 63588096) if placement == 'mixed' else (95392512, 95392512)
        owner_index = len(result)
        if mode == 'int8':
            baseline = base
        else:
            baseline = f16_measured[owner_index]
        increment = (cells - BASE_CELLS) * (q * (kv + 128 + 1) + 256 + 4)
        session = baseline + increment
        # Entry-stage PLE: emb_dev + ple_run_scratch_bytes (ple.cu/layer.cpp).
        ple = 274352 if owner_index == 0 else 0
        # Ungrouped final verifier allocates a device input on non-entry stages.
        handoff = 0 if owner_index == 0 else (4*2560 + 2560 + 4)*4*8
        # verify.cpp prints count.used rounded to one decimal MiB (66.1 here).
        verifier_floor = 66 * (1 << 20)
        persistent_floor = session + weight + experts + ple + handoff + verifier_floor
        result.append(dict(qsa_layers=q, session_bytes=session, pack_weight_bytes=weight,
                           expert_slot_bytes=experts, reserve_bytes=RESERVE,
                           ple_device_bytes=ple, verifier_device_floor_bytes=verifier_floor,
                           handoff_device_bytes=handoff, persistent_floor_bytes=persistent_floor,
                           subtotal_not_peak_bytes=persistent_floor + RESERVE,
                           native_dense_estimate_bytes=DENSE_ESTIMATES[placement][owner_index],
                           temporary_peak_bytes=None,
                           actual_free_vram_bytes=None, admission='UNKNOWN'))
    return dict(status='LOWER_BOUND_NOT_ADMISSION', placement=placement, cells=cells,
                kv=mode, owners=result, assumptions='ungrouped max_t=8; no streaming or elastic KV',
                unknowns=['native-dense actual allocator overhead and HC copies/scratch', 'prefill workspace and lifetime',
                          'graph, runtime, library, transport and transient peaks', 'current free VRAM'])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--placement', choices=OWNERS, required=True)
    p.add_argument('--cells', type=int, required=True)
    p.add_argument('--kv', choices=('int8', 'f16'), required=True)
    args = p.parse_args()
    try:
        print(json.dumps(estimate(args.placement, args.cells, args.kv), indent=2))
    except ValueError as e:
        p.error(str(e))


if __name__ == '__main__':
    main()
