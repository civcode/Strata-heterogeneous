# Report 0067

Status: **Completed, with skips and measurement limitations.** The hardening port builds and the requested local checks ran. This does not establish that the earlier host crash is fixed.

## Builds and CPU tests

New Release/Ninja trees `build-0067-host`, `build-0067-hip`, `build-0067-cuda`: all built at -j28 with tests and hetero enabled. HIP used ROCm core-10.1, gfx1201, STRATA_MMQ=ON and STRATA_PREFILL_MMQ=ON; CUDA used 13.0, sm_89. No source compile fixes were needed. HIP configuration initially could not find hip-config.cmake; specifying CMAKE_PREFIX_PATH=/opt/rocm/core-10.1 fixed the build configuration.

Host CTest: 225 passed, 6 skipped, 0 failed (231 total). Skip messages:
- io_priority_test: SKIP (Windows only).
- native_hadamard_test, tq_cpu_parity, tq_cpu_parity_avx2: this ggml has no trellis types; requires the agention fork.
- expert_parity, pool_test: needs pack/full/experts.bin, a Q2_0 expert pack/model file not in this tree.

## GPU tests and reproducer

Tests ran individually using CTest -j1, HIP before CUDA, with the journal and core list checked between test processes. HIP: 95 passed, 0 skipped, 0 failed, including peer_native_* and the listed transfer and ownership families, HSA_ENABLE_SDMA=0. CUDA: 17 passed, 3 skipped, 0 failed (20 total); conversation_prepare_wrong-device, conversation_prepare_second and conversation_prepare_device skip silently with return 77 when fewer than two CUDA devices are present. Only one CUDA card is present. Native HIP-only ownership test targets were not listed in CUDA CTest.

The 0066 QSA reproducer, STRATA_TOPK_PAGEABLE_READBACK=1 and HSA_ENABLE_SDMA=0, passed its complete selftest once each with HIP_VISIBLE_DEVICES=0 and 1. As in report 0066 this exercises blocking pageable QSA readback in isolation, not snapshot async readback under contention, and does not prove the historical issue absent.

## HIP-only server

Used the report 0065 copied configuration with only exe changed to build-0067-hip/strata; original safe worktree log/cwd remained. Native two-R9700 layer split and MTP, with HSA_ENABLE_SDMA=0. Both short greedy requests were identical and succeeded with identical answers. The long request returned normally but reached max_tokens while thinking and had no final answer. The long prompt was 6361 tokens, not the target ~5000, and was not the identical report 0065 prompt; these numbers are not a controlled benchmark. Short prompts also differ from report 0065.

| Request | Prompt tokens (reused/read) | Prefill tok/s | Decode tok/s | Drafts accepted | 0065 baseline (prompt; prefill; decode; drafts) |
|---|---:|---:|---:|---:|---|
| Short | 63 (0/63) | 209.3 | 115.1 | 56/79 | 69; 215.5; 85.0; 105/184 |
| Repeat | 63 (58/5) | 162.9 | 119.8 | 57/76 | 69 (64/5); 171.4; 84.9; 106/186 |
| Long | 6361 (0/6361) | 2830.6 | 122.2 | 27/41 | 5397; 2283.0; 96.1; 25/45 |

Cache reuse: 58 tokens on the repeat. Loaded AMD SMI used/free MB: card 0 26645/5979, card 1 24316/8308. One SIGINT to the server shut down server, engine and vision process cooperatively; none remained. Afterwards card 0 and 1 each reported 57 MB used / 32567 MB free, i.e. residual system use, not literal zero global VRAM.

GPU-process test wall time: approximately 19 seconds summed CTest wall times, 1 second reproducer, and approximately 65 seconds server including startup/shutdown (about 85 seconds total; below both the five-minute standing budget and 15-minute handoff budget).

## Kernel/core safety

Before any GPU process and after the last GPU process, captured matching last-boot kernel lines and the since-boot core list below. Baseline: 314 matching kernel lines, 15 pre-existing core records. Post-run: **no new matching kernel lines or core records**; boot remained 2026-11-17 10:42:15. Checked between GPU test processes and after each request; no fault or new crash detected. The 15 pre-existing cores belong to strata-monitor processes and predate this handoff.

### Baseline kernel lines (post-run identical)

```text
Nov 17 10:42:19 oberon kernel: NMI watchdog: Enabled. Permanently consumes one hw-PMU counter.
Nov 17 10:42:19 oberon kernel: workingset: timestamp_bits=36 max_order=24 bucket_order=0
Nov 17 10:42:19 oberon kernel: Estimated ratio of average max frequency by base frequency (times 1024): 1197
Nov 17 10:42:19 oberon systemd[1]: Reached target integritysetup.target - Local Integrity Protected Volumes.
Nov 17 10:42:19 oberon systemd[1]: Reached target veritysetup.target - Local Verity Protected Volumes.
Nov 17 10:42:19 oberon systemd[1]: Mounting dev-mqueue.mount - POSIX Message Queue File System...
Nov 17 10:42:19 oberon systemd-journald[694]: Collecting audit messages is disabled.
Nov 17 10:42:19 oberon systemd[1]: Mounted dev-mqueue.mount - POSIX Message Queue File System.
Nov 17 10:42:21 oberon kernel: amdgpu: Virtual CRAT table created for CPU
Nov 17 10:42:21 oberon kernel: amdgpu: Topology: Add CPU node
Nov 17 10:42:21 oberon kernel: amdgpu 0000:03:00.0: enabling device (0006 -> 0007)
Nov 17 10:42:21 oberon kernel: amdgpu 0000:03:00.0: initializing kernel modesetting (IP DISCOVERY 0x1002:0x7551 0x1458:0x242F 0xC0).
Nov 17 10:42:21 oberon kernel: amdgpu 0000:03:00.0: register mmio base: 0xDFC00000
Nov 17 10:42:21 oberon kernel: amdgpu 0000:03:00.0: register mmio size: 524288
Nov 17 10:42:21 oberon kernel: amdgpu 0000:03:00.0: detected ip block number 0 <common_v1_0_0> (soc24_common)
Nov 17 10:42:21 oberon kernel: amdgpu 0000:03:00.0: detected ip block number 1 <gmc_v12_0_0> (gmc_v12_0)
Nov 17 10:42:21 oberon kernel: amdgpu 0000:03:00.0: detected ip block number 2 <ih_v7_0_0> (ih_v7_0)
Nov 17 10:42:21 oberon kernel: amdgpu 0000:03:00.0: detected ip block number 3 <psp_v14_0_0> (psp)
Nov 17 10:42:21 oberon kernel: amdgpu 0000:03:00.0: detected ip block number 4 <smu_v14_0_0> (smu)
Nov 17 10:42:21 oberon kernel: amdgpu 0000:03:00.0: detected ip block number 5 <dce_v1_0_0> (dm)
Nov 17 10:42:21 oberon kernel: amdgpu 0000:03:00.0: detected ip block number 6 <gfx_v12_0_0> (gfx_v12_0)
Nov 17 10:42:21 oberon kernel: amdgpu 0000:03:00.0: detected ip block number 7 <sdma_v7_0_0> (sdma_v7_0)
Nov 17 10:42:21 oberon kernel: amdgpu 0000:03:00.0: detected ip block number 8 <vcn_v5_0_0> (vcn_v5_0_0)
Nov 17 10:42:21 oberon kernel: amdgpu 0000:03:00.0: detected ip block number 9 <jpeg_v5_0_0> (jpeg_v5_0_0)
Nov 17 10:42:21 oberon kernel: amdgpu 0000:03:00.0: detected ip block number 10 <mes_v12_0_0> (mes_v12_0)
Nov 17 10:42:21 oberon kernel: amdgpu 0000:03:00.0: Fetched VBIOS from VFCT
Nov 17 10:42:21 oberon kernel: amdgpu 0000:03:00.0: [drm] ATOM BIOS: 113-R9700AT-F40
Nov 17 10:42:21 oberon kernel: amdgpu 0000:03:00.0: Trusted Memory Zone (TMZ) feature not supported
Nov 17 10:42:21 oberon kernel: amdgpu 0000:03:00.0: MEM ECC is active.
Nov 17 10:42:21 oberon kernel: amdgpu 0000:03:00.0: SRAM ECC is not presented.
Nov 17 10:42:21 oberon kernel: amdgpu 0000:03:00.0: RAS INFO: ras initialized successfully, hardware ability[101] ras_mask[101]
Nov 17 10:42:21 oberon kernel: amdgpu 0000:03:00.0: vm size is 262144 GB, 4 levels, block size is 9-bit, fragment size is 9-bit
Nov 17 10:42:21 oberon kernel: amdgpu 0000:03:00.0: VRAM: 32624M 0x0000008000000000 - 0x00000087F6FFFFFF (32624M used)
Nov 17 10:42:21 oberon kernel: amdgpu 0000:03:00.0: GART: 512M 0x0000000000000000 - 0x000000001FFFFFFF
Nov 17 10:42:21 oberon kernel: amdgpu 0000:03:00.0: [drm] Detected VRAM RAM=32624M, BAR=32768M
Nov 17 10:42:21 oberon kernel: amdgpu 0000:03:00.0: [drm] RAM width 256bits GDDR6
Nov 17 10:42:21 oberon kernel: amdgpu 0000:03:00.0:  32624M of VRAM memory ready
Nov 17 10:42:21 oberon kernel: amdgpu 0000:03:00.0:  30971M of GTT memory ready.
Nov 17 10:42:21 oberon kernel: amdgpu 0000:03:00.0: [drm] GART: num cpu pages 131072, num gpu pages 131072
Nov 17 10:42:21 oberon kernel: amdgpu 0000:03:00.0: [drm] PCIE GART of 512M enabled (table at 0x00000087D6B00000).
Nov 17 10:42:21 oberon kernel: amdgpu 0000:03:00.0: [drm] Loading DMUB firmware via PSP: version=0x0A000800
Nov 17 10:42:21 oberon kernel: amdgpu 0000:03:00.0: [VCN instance 0] Found VCN firmware Version ENC: 1.12 DEC: 9 VEP: 0 Revision: 15
Nov 17 10:42:21 oberon kernel: amdgpu 0000:03:00.0: MES: vmid_mask_mmhub 0x0000ff00, vmid_mask_gfxhub 0x0000ff00
Nov 17 10:42:21 oberon kernel: amdgpu 0000:03:00.0: MES: gfx_hqd_mask 0x000000fe, compute_hqd_mask 0x0000000c, sdma_hqd_mask 0x000000fc
Nov 17 10:42:21 oberon kernel: audit: type=1400 audit(1794908541.543:164): apparmor="STATUS" operation="profile_load" profile="unconfined" name="snap-update-ns.mesa-2604" pid=1492 comm="apparmor_parser"
Nov 17 10:42:21 oberon kernel: audit: type=1400 audit(1794908541.543:165): apparmor="STATUS" operation="profile_load" profile="unconfined" name="snap-update-ns.mesa-2404" pid=1491 comm="apparmor_parser"
Nov 17 10:42:21 oberon kernel: amdgpu 0000:03:00.0: GECC is disabled, set amdgpu_ras_enable=1 to enable GECC in next boot cycle if needed
Nov 17 10:42:21 oberon kernel: amdgpu 0000:03:00.0: RAP: optional rap ta ucode is not available
Nov 17 10:42:21 oberon kernel: amdgpu 0000:03:00.0: SECUREDISPLAY: optional securedisplay ta ucode is not available
Nov 17 10:42:21 oberon kernel: amdgpu 0000:03:00.0: smu driver if version = 0x0000002e, smu fw if version = 0x00000033, smu fw program = 0, smu fw version = 0x00684c00 (104.76.0)
Nov 17 10:42:21 oberon kernel: amdgpu 0000:03:00.0: SMU is initialized successfully!
Nov 17 10:42:21 oberon kernel: amdgpu 0000:03:00.0: [drm] Display Core v3.2.369 initialized on DCN 4.0.1
Nov 17 10:42:21 oberon kernel: amdgpu 0000:03:00.0: [drm] DP-HDMI FRL PCON supported
Nov 17 10:42:21 oberon kernel: amdgpu 0000:03:00.0: [drm] DMUB hardware initialized: version=0x0A000800
Nov 17 10:42:21 oberon kernel: snd_hda_intel 0000:03:00.1: bound 0000:03:00.0 (ops amdgpu_dm_audio_component_bind_ops [amdgpu])
Nov 17 10:42:21 oberon kernel: amdgpu 0000:03:00.0: [drm] DP-1: PSR support 0, DC PSR ver -1, sink PSR ver 0 DPCD caps 0x0 su_y_granularity 0
Nov 17 10:42:21 oberon kernel: amdgpu 0000:03:00.0: [drm] DP-2: PSR support 0, DC PSR ver -1, sink PSR ver 0 DPCD caps 0x0 su_y_granularity 0
Nov 17 10:42:21 oberon kernel: amdgpu 0000:03:00.0: [drm] DP-3: PSR support 0, DC PSR ver -1, sink PSR ver 0 DPCD caps 0x0 su_y_granularity 0
Nov 17 10:42:21 oberon kernel: amdgpu 0000:03:00.0: [drm] HDMI-A-1: PSR support 0, DC PSR ver -1, sink PSR ver 0 DPCD caps 0x0 su_y_granularity 0
Nov 17 10:42:21 oberon kernel: amdgpu 0000:03:00.0: program CP_MES_CNTL : 0x4000000
Nov 17 10:42:21 oberon kernel: amdgpu 0000:03:00.0: program CP_MES_CNTL : 0xc000000
Nov 17 10:42:21 oberon kernel: amdgpu: Virtual CRAT table created for GPU
Nov 17 10:42:21 oberon kernel: amdgpu: Topology: Add GPU node [0x1002:0x7551]
Nov 17 10:42:21 oberon kernel: amdgpu 0000:03:00.0: SE 4, SH per SE 2, CU per SH 8, active_cu_number 64
Nov 17 10:42:21 oberon kernel: amdgpu 0000:03:00.0: ring gfx_0.0.0 uses VM inv eng 0 on hub 0
Nov 17 10:42:21 oberon kernel: amdgpu 0000:03:00.0: ring comp_1.0.0 uses VM inv eng 1 on hub 0
Nov 17 10:42:21 oberon kernel: amdgpu 0000:03:00.0: ring comp_1.1.0 uses VM inv eng 4 on hub 0
Nov 17 10:42:21 oberon kernel: amdgpu 0000:03:00.0: ring comp_1.0.1 uses VM inv eng 7 on hub 0
Nov 17 10:42:21 oberon kernel: amdgpu 0000:03:00.0: ring comp_1.1.1 uses VM inv eng 8 on hub 0
Nov 17 10:42:21 oberon kernel: amdgpu 0000:03:00.0: ring sdma0 uses VM inv eng 9 on hub 0
Nov 17 10:42:21 oberon kernel: amdgpu 0000:03:00.0: ring sdma1 uses VM inv eng 10 on hub 0
Nov 17 10:42:21 oberon kernel: amdgpu 0000:03:00.0: ring vcn_unified_0 uses VM inv eng 0 on hub 8
Nov 17 10:42:21 oberon kernel: amdgpu 0000:03:00.0: ring jpeg_dec uses VM inv eng 1 on hub 8
Nov 17 10:42:21 oberon kernel: amdgpu: HMM registered 32624MB device memory
Nov 17 10:42:21 oberon kernel: amdgpu 0000:03:00.0: Using BACO for runtime pm
Nov 17 10:42:21 oberon kernel: amdgpu 0000:03:00.0: [drm] Registered 4 planes with drm panic
Nov 17 10:42:21 oberon kernel: [drm] Initialized amdgpu 3.64.0 for 0000:03:00.0 on minor 1
Nov 17 10:42:21 oberon kernel: amdgpu 0000:03:00.0: [drm] Cannot find any crtc or sizes
Nov 17 10:42:21 oberon kernel: amdgpu 0000:07:00.0: enabling device (0006 -> 0007)
Nov 17 10:42:21 oberon kernel: amdgpu 0000:07:00.0: initializing kernel modesetting (IP DISCOVERY 0x1002:0x7551 0x1458:0x242F 0xC0).
Nov 17 10:42:21 oberon kernel: amdgpu 0000:07:00.0: register mmio base: 0xDFA00000
Nov 17 10:42:21 oberon kernel: amdgpu 0000:07:00.0: register mmio size: 524288
Nov 17 10:42:21 oberon kernel: amdgpu 0000:07:00.0: detected ip block number 0 <common_v1_0_0> (soc24_common)
Nov 17 10:42:21 oberon kernel: amdgpu 0000:07:00.0: detected ip block number 1 <gmc_v12_0_0> (gmc_v12_0)
Nov 17 10:42:21 oberon kernel: amdgpu 0000:07:00.0: detected ip block number 2 <ih_v7_0_0> (ih_v7_0)
Nov 17 10:42:21 oberon kernel: amdgpu 0000:07:00.0: detected ip block number 3 <psp_v14_0_0> (psp)
Nov 17 10:42:21 oberon kernel: amdgpu 0000:07:00.0: detected ip block number 4 <smu_v14_0_0> (smu)
Nov 17 10:42:21 oberon kernel: amdgpu 0000:07:00.0: detected ip block number 5 <dce_v1_0_0> (dm)
Nov 17 10:42:21 oberon kernel: amdgpu 0000:07:00.0: detected ip block number 6 <gfx_v12_0_0> (gfx_v12_0)
Nov 17 10:42:21 oberon kernel: amdgpu 0000:07:00.0: detected ip block number 7 <sdma_v7_0_0> (sdma_v7_0)
Nov 17 10:42:21 oberon kernel: amdgpu 0000:07:00.0: detected ip block number 8 <vcn_v5_0_0> (vcn_v5_0_0)
Nov 17 10:42:21 oberon kernel: amdgpu 0000:07:00.0: detected ip block number 9 <jpeg_v5_0_0> (jpeg_v5_0_0)
Nov 17 10:42:21 oberon kernel: amdgpu 0000:07:00.0: detected ip block number 10 <mes_v12_0_0> (mes_v12_0)
Nov 17 10:42:21 oberon kernel: amdgpu 0000:07:00.0: VFCT bus number mismatch: table 3 != runtime 7, matching by device identity (vendor 0x1002 device 0x7551)
Nov 17 10:42:21 oberon kernel: amdgpu 0000:07:00.0: Fetched VBIOS from VFCT
Nov 17 10:42:21 oberon kernel: amdgpu 0000:07:00.0: [drm] ATOM BIOS: 113-R9700AT-F40
Nov 17 10:42:21 oberon kernel: amdgpu 0000:07:00.0: Trusted Memory Zone (TMZ) feature not supported
Nov 17 10:42:21 oberon kernel: amdgpu 0000:07:00.0: MEM ECC is active.
Nov 17 10:42:21 oberon kernel: amdgpu 0000:07:00.0: SRAM ECC is not presented.
Nov 17 10:42:21 oberon kernel: amdgpu 0000:07:00.0: RAS INFO: ras initialized successfully, hardware ability[101] ras_mask[101]
Nov 17 10:42:21 oberon kernel: amdgpu 0000:07:00.0: vm size is 262144 GB, 4 levels, block size is 9-bit, fragment size is 9-bit
Nov 17 10:42:21 oberon kernel: amdgpu 0000:07:00.0: VRAM: 32624M 0x0000008000000000 - 0x00000087F6FFFFFF (32624M used)
Nov 17 10:42:21 oberon kernel: amdgpu 0000:07:00.0: GART: 512M 0x0000000000000000 - 0x000000001FFFFFFF
Nov 17 10:42:21 oberon kernel: amdgpu 0000:07:00.0: [drm] Detected VRAM RAM=32624M, BAR=32768M
Nov 17 10:42:21 oberon kernel: amdgpu 0000:07:00.0: [drm] RAM width 256bits GDDR6
Nov 17 10:42:21 oberon kernel: amdgpu 0000:07:00.0:  32624M of VRAM memory ready
Nov 17 10:42:21 oberon kernel: amdgpu 0000:07:00.0:  30971M of GTT memory ready.
Nov 17 10:42:21 oberon kernel: amdgpu 0000:07:00.0: [drm] GART: num cpu pages 131072, num gpu pages 131072
Nov 17 10:42:21 oberon kernel: amdgpu 0000:07:00.0: [drm] PCIE GART of 512M enabled (table at 0x00000087D6B00000).
Nov 17 10:42:21 oberon kernel: amdgpu 0000:07:00.0: [drm] Loading DMUB firmware via PSP: version=0x0A000800
Nov 17 10:42:21 oberon kernel: amdgpu 0000:07:00.0: [VCN instance 0] Found VCN firmware Version ENC: 1.12 DEC: 9 VEP: 0 Revision: 15
Nov 17 10:42:21 oberon kernel: amdgpu 0000:07:00.0: MES: vmid_mask_mmhub 0x0000ff00, vmid_mask_gfxhub 0x0000ff00
Nov 17 10:42:21 oberon kernel: amdgpu 0000:07:00.0: MES: gfx_hqd_mask 0x000000fe, compute_hqd_mask 0x0000000c, sdma_hqd_mask 0x000000fc
Nov 17 10:42:22 oberon kernel: amdgpu 0000:07:00.0: GECC is disabled, set amdgpu_ras_enable=1 to enable GECC in next boot cycle if needed
Nov 17 10:42:22 oberon kernel: amdgpu 0000:07:00.0: RAP: optional rap ta ucode is not available
Nov 17 10:42:22 oberon kernel: amdgpu 0000:07:00.0: SECUREDISPLAY: optional securedisplay ta ucode is not available
Nov 17 10:42:22 oberon kernel: amdgpu 0000:07:00.0: smu driver if version = 0x0000002e, smu fw if version = 0x00000033, smu fw program = 0, smu fw version = 0x00684c00 (104.76.0)
Nov 17 10:42:22 oberon kernel: amdgpu 0000:07:00.0: SMU is initialized successfully!
Nov 17 10:42:22 oberon kernel: amdgpu 0000:07:00.0: [drm] Display Core v3.2.369 initialized on DCN 4.0.1
Nov 17 10:42:22 oberon kernel: amdgpu 0000:07:00.0: [drm] DP-HDMI FRL PCON supported
Nov 17 10:42:22 oberon kernel: amdgpu 0000:07:00.0: [drm] DMUB hardware initialized: version=0x0A000800
Nov 17 10:42:22 oberon kernel: snd_hda_intel 0000:07:00.1: bound 0000:07:00.0 (ops amdgpu_dm_audio_component_bind_ops [amdgpu])
Nov 17 10:42:22 oberon kernel: amdgpu 0000:07:00.0: [drm] DP-4: PSR support 0, DC PSR ver -1, sink PSR ver 0 DPCD caps 0x0 su_y_granularity 0
Nov 17 10:42:22 oberon kernel: amdgpu 0000:07:00.0: [drm] DP-5: PSR support 0, DC PSR ver -1, sink PSR ver 0 DPCD caps 0x0 su_y_granularity 0
Nov 17 10:42:22 oberon kernel: amdgpu 0000:07:00.0: [drm] DP-6: PSR support 0, DC PSR ver -1, sink PSR ver 0 DPCD caps 0x0 su_y_granularity 0
Nov 17 10:42:22 oberon kernel: amdgpu 0000:07:00.0: [drm] HDMI-A-2: PSR support 0, DC PSR ver -1, sink PSR ver 0 DPCD caps 0x0 su_y_granularity 0
Nov 17 10:42:22 oberon kernel: amdgpu 0000:07:00.0: program CP_MES_CNTL : 0x4000000
Nov 17 10:42:22 oberon kernel: amdgpu 0000:07:00.0: program CP_MES_CNTL : 0xc000000
Nov 17 10:42:22 oberon kernel: amdgpu: Virtual CRAT table created for GPU
Nov 17 10:42:22 oberon kernel: amdgpu: Topology: Add GPU node [0x1002:0x7551]
Nov 17 10:42:22 oberon kernel: amdgpu 0000:07:00.0: SE 4, SH per SE 2, CU per SH 8, active_cu_number 64
Nov 17 10:42:22 oberon kernel: amdgpu 0000:07:00.0: ring gfx_0.0.0 uses VM inv eng 0 on hub 0
Nov 17 10:42:22 oberon kernel: amdgpu 0000:07:00.0: ring comp_1.0.0 uses VM inv eng 1 on hub 0
Nov 17 10:42:22 oberon kernel: amdgpu 0000:07:00.0: ring comp_1.1.0 uses VM inv eng 4 on hub 0
Nov 17 10:42:22 oberon kernel: amdgpu 0000:07:00.0: ring comp_1.0.1 uses VM inv eng 7 on hub 0
Nov 17 10:42:22 oberon kernel: amdgpu 0000:07:00.0: ring comp_1.1.1 uses VM inv eng 8 on hub 0
Nov 17 10:42:22 oberon kernel: amdgpu 0000:07:00.0: ring sdma0 uses VM inv eng 9 on hub 0
Nov 17 10:42:22 oberon kernel: amdgpu 0000:07:00.0: ring sdma1 uses VM inv eng 10 on hub 0
Nov 17 10:42:22 oberon kernel: amdgpu 0000:07:00.0: ring vcn_unified_0 uses VM inv eng 0 on hub 8
Nov 17 10:42:22 oberon kernel: amdgpu 0000:07:00.0: ring jpeg_dec uses VM inv eng 1 on hub 8
Nov 17 10:42:22 oberon kernel: amdgpu: HMM registered 32624MB device memory
Nov 17 10:42:22 oberon kernel: amdgpu 0000:07:00.0: Using BACO for runtime pm
Nov 17 10:42:22 oberon kernel: amdgpu 0000:07:00.0: [drm] Registered 4 planes with drm panic
Nov 17 10:42:22 oberon kernel: [drm] Initialized amdgpu 3.64.0 for 0000:07:00.0 on minor 3
Nov 17 10:42:22 oberon kernel: amdgpu 0000:07:00.0: [drm] Cannot find any crtc or sizes
Nov 17 10:42:22 oberon kernel: amdgpu 0000:7e:00.0: enabling device (0006 -> 0007)
Nov 17 10:42:22 oberon kernel: amdgpu 0000:7e:00.0: initializing kernel modesetting (IP DISCOVERY 0x1002:0x13C0 0x1043:0x8877 0xC1).
Nov 17 10:42:22 oberon kernel: amdgpu 0000:7e:00.0: register mmio base: 0xDF900000
Nov 17 10:42:22 oberon kernel: amdgpu 0000:7e:00.0: register mmio size: 524288
Nov 17 10:42:22 oberon kernel: amdgpu 0000:7e:00.0: detected ip block number 0 <common_v1_0_0> (nv_common)
Nov 17 10:42:22 oberon kernel: amdgpu 0000:7e:00.0: detected ip block number 1 <gmc_v10_0_0> (gmc_v10_0)
Nov 17 10:42:22 oberon kernel: amdgpu 0000:7e:00.0: detected ip block number 2 <ih_v5_0_0> (navi10_ih)
Nov 17 10:42:22 oberon kernel: amdgpu 0000:7e:00.0: detected ip block number 3 <psp_v13_0_0> (psp)
Nov 17 10:42:22 oberon kernel: amdgpu 0000:7e:00.0: detected ip block number 4 <smu_v13_0_0> (smu)
Nov 17 10:42:22 oberon kernel: amdgpu 0000:7e:00.0: detected ip block number 5 <dce_v1_0_0> (dm)
Nov 17 10:42:22 oberon kernel: amdgpu 0000:7e:00.0: detected ip block number 6 <gfx_v10_0_0> (gfx_v10_0)
Nov 17 10:42:22 oberon kernel: amdgpu 0000:7e:00.0: detected ip block number 7 <sdma_v5_2_0> (sdma_v5_2)
Nov 17 10:42:22 oberon kernel: amdgpu 0000:7e:00.0: detected ip block number 8 <vcn_v3_0_0> (vcn_v3_0)
Nov 17 10:42:22 oberon kernel: amdgpu 0000:7e:00.0: detected ip block number 9 <jpeg_v3_0_0> (jpeg_v3_0)
Nov 17 10:42:22 oberon kernel: amdgpu 0000:7e:00.0: Fetched VBIOS from VFCT
Nov 17 10:42:22 oberon kernel: amdgpu 0000:7e:00.0: [drm] ATOM BIOS: 102-RAPHAEL-008
Nov 17 10:42:22 oberon kernel: amdgpu 0000:7e:00.0: vgaarb: deactivate vga console
Nov 17 10:42:22 oberon kernel: amdgpu 0000:7e:00.0: Trusted Memory Zone (TMZ) feature disabled as experimental (default)
Nov 17 10:42:22 oberon kernel: amdgpu 0000:7e:00.0: vm size is 262144 GB, 4 levels, block size is 9-bit, fragment size is 9-bit
Nov 17 10:42:22 oberon kernel: amdgpu 0000:7e:00.0: VRAM: 2048M 0x000000F400000000 - 0x000000F47FFFFFFF (2048M used)
Nov 17 10:42:22 oberon kernel: amdgpu 0000:7e:00.0: GART: 1024M 0x0000000000000000 - 0x000000003FFFFFFF
Nov 17 10:42:22 oberon kernel: amdgpu 0000:7e:00.0: [drm] Detected VRAM RAM=2048M, BAR=2048M
Nov 17 10:42:22 oberon kernel: amdgpu 0000:7e:00.0: [drm] RAM width 128bits DDR5
Nov 17 10:42:22 oberon kernel: amdgpu 0000:7e:00.0:  2048M of VRAM memory ready
Nov 17 10:42:22 oberon kernel: amdgpu 0000:7e:00.0:  30971M of GTT memory ready.
Nov 17 10:42:22 oberon kernel: amdgpu 0000:7e:00.0: [drm] GART: num cpu pages 262144, num gpu pages 262144
Nov 17 10:42:22 oberon kernel: amdgpu 0000:7e:00.0: [drm] PCIE GART of 1024M enabled (table at 0x000000F47FC00000).
Nov 17 10:42:22 oberon kernel: amdgpu 0000:7e:00.0: [drm] Loading DMUB firmware via PSP: version=0x05001C00
Nov 17 10:42:22 oberon kernel: amdgpu 0000:7e:00.0: [drm] use_doorbell being set to: [true]
Nov 17 10:42:22 oberon kernel: amdgpu 0000:7e:00.0: [VCN instance 0] Found VCN firmware Version ENC: 1.30 DEC: 3 VEP: 0 Revision: 4
Nov 17 10:42:22 oberon kernel: amdgpu 0000:7e:00.0: reserve 0xa00000 from 0xf47e000000 for PSP TMR
Nov 17 10:42:22 oberon kernel: amdgpu 0000:7e:00.0: RAS: optional ras ta ucode is not available
Nov 17 10:42:22 oberon kernel: amdgpu 0000:7e:00.0: RAP: optional rap ta ucode is not available
Nov 17 10:42:22 oberon kernel: amdgpu 0000:7e:00.0: SECUREDISPLAY: optional securedisplay ta ucode is not available
Nov 17 10:42:22 oberon kernel: amdgpu 0000:7e:00.0: SMU is initialized successfully!
Nov 17 10:42:22 oberon kernel: amdgpu 0000:7e:00.0: [drm] Display Core v3.2.369 initialized on DCN 3.1.5
Nov 17 10:42:22 oberon kernel: amdgpu 0000:7e:00.0: [drm] DP-HDMI FRL PCON supported
Nov 17 10:42:22 oberon kernel: amdgpu 0000:7e:00.0: [drm] DMUB hardware initialized: version=0x05001C00
Nov 17 10:42:22 oberon kernel: snd_hda_intel 0000:7e:00.1: bound 0000:7e:00.0 (ops amdgpu_dm_audio_component_bind_ops [amdgpu])
Nov 17 10:42:22 oberon kernel: amdgpu 0000:7e:00.0: [drm] HDMI-A-3: PSR support 0, DC PSR ver -1, sink PSR ver 0 DPCD caps 0x0 su_y_granularity 0
Nov 17 10:42:22 oberon kernel: amdgpu 0000:7e:00.0: [drm] DP-7: PSR support 0, DC PSR ver -1, sink PSR ver 0 DPCD caps 0x0 su_y_granularity 0
Nov 17 10:42:22 oberon kernel: amdgpu 0000:7e:00.0: [drm] DP-8: PSR support 0, DC PSR ver -1, sink PSR ver 0 DPCD caps 0x0 su_y_granularity 0
Nov 17 10:42:22 oberon kernel: amdgpu 0000:7e:00.0: kiq ring mec 2 pipe 1 q 0
Nov 17 10:42:22 oberon kernel: amdgpu: Virtual CRAT table created for GPU
Nov 17 10:42:22 oberon kernel: amdgpu: Topology: Add GPU node [0x1002:0x13c0]
Nov 17 10:42:22 oberon kernel: amdgpu 0000:7e:00.0: SE 1, SH per SE 1, CU per SH 2, active_cu_number 2
Nov 17 10:42:22 oberon kernel: amdgpu 0000:7e:00.0: ring gfx_0.0.0 uses VM inv eng 0 on hub 0
Nov 17 10:42:22 oberon kernel: amdgpu 0000:7e:00.0: ring gfx_0.1.0 uses VM inv eng 1 on hub 0
Nov 17 10:42:22 oberon kernel: amdgpu 0000:7e:00.0: ring comp_1.0.0 uses VM inv eng 4 on hub 0
Nov 17 10:42:22 oberon kernel: amdgpu 0000:7e:00.0: ring comp_1.1.0 uses VM inv eng 5 on hub 0
Nov 17 10:42:22 oberon kernel: amdgpu 0000:7e:00.0: ring comp_1.2.0 uses VM inv eng 6 on hub 0
Nov 17 10:42:22 oberon kernel: amdgpu 0000:7e:00.0: ring comp_1.3.0 uses VM inv eng 7 on hub 0
Nov 17 10:42:22 oberon kernel: amdgpu 0000:7e:00.0: ring comp_1.0.1 uses VM inv eng 8 on hub 0
Nov 17 10:42:22 oberon kernel: amdgpu 0000:7e:00.0: ring comp_1.1.1 uses VM inv eng 9 on hub 0
Nov 17 10:42:22 oberon kernel: amdgpu 0000:7e:00.0: ring comp_1.2.1 uses VM inv eng 10 on hub 0
Nov 17 10:42:22 oberon kernel: amdgpu 0000:7e:00.0: ring comp_1.3.1 uses VM inv eng 11 on hub 0
Nov 17 10:42:22 oberon kernel: amdgpu 0000:7e:00.0: ring kiq_0.2.1.0 uses VM inv eng 12 on hub 0
Nov 17 10:42:22 oberon kernel: amdgpu 0000:7e:00.0: ring sdma0 uses VM inv eng 13 on hub 0
Nov 17 10:42:22 oberon kernel: amdgpu 0000:7e:00.0: ring vcn_dec_0 uses VM inv eng 0 on hub 8
Nov 17 10:42:22 oberon kernel: amdgpu 0000:7e:00.0: ring vcn_enc_0.0 uses VM inv eng 1 on hub 8
Nov 17 10:42:22 oberon kernel: amdgpu 0000:7e:00.0: ring vcn_enc_0.1 uses VM inv eng 4 on hub 8
Nov 17 10:42:22 oberon kernel: amdgpu 0000:7e:00.0: ring jpeg_dec uses VM inv eng 5 on hub 8
Nov 17 10:42:22 oberon kernel: amdgpu 0000:7e:00.0: Runtime PM not available
Nov 17 10:42:22 oberon kernel: amdgpu 0000:7e:00.0: [drm] Registered 4 planes with drm panic
Nov 17 10:42:22 oberon kernel: [drm] Initialized amdgpu 3.64.0 for 0000:7e:00.0 on minor 4
Nov 17 10:42:22 oberon kernel: amdgpu 0000:7e:00.0: [drm] Failed to setup vendor infoframe on connector HDMI-A-3: -22
Nov 17 10:42:22 oberon kernel: fbcon: amdgpudrmfb (fb0) is primary device
Nov 17 10:42:22 oberon kernel: amdgpu 0000:7e:00.0: [drm] REG_WAIT timeout 1us * 100000 tries - optc31_disable_crtc line:145
Nov 17 10:42:22 oberon kernel: amdgpu 0000:7e:00.0: [drm] fb0: amdgpudrmfb frame buffer device
Nov 17 11:32:58 oberon kernel: amdgpu 0000:03:00.0: [drm] PCIE GART of 512M enabled (table at 0x00000087D6B00000).
Nov 17 11:32:58 oberon kernel: amdgpu 0000:03:00.0: PSP is resuming...
Nov 17 11:32:58 oberon kernel: amdgpu 0000:03:00.0: GECC is disabled, set amdgpu_ras_enable=1 to enable GECC in next boot cycle if needed
Nov 17 11:32:58 oberon kernel: amdgpu 0000:03:00.0: RAP: optional rap ta ucode is not available
Nov 17 11:32:58 oberon kernel: amdgpu 0000:03:00.0: SECUREDISPLAY: optional securedisplay ta ucode is not available
Nov 17 11:32:58 oberon kernel: amdgpu 0000:03:00.0: SMU is resuming...
Nov 17 11:32:58 oberon kernel: amdgpu 0000:03:00.0: smu driver if version = 0x0000002e, smu fw if version = 0x00000033, smu fw program = 0, smu fw version = 0x00684c00 (104.76.0)
Nov 17 11:32:59 oberon kernel: amdgpu 0000:03:00.0: SMU is resumed successfully!
Nov 17 11:32:59 oberon kernel: amdgpu 0000:03:00.0: program CP_MES_CNTL : 0x4000000
Nov 17 11:32:59 oberon kernel: amdgpu 0000:03:00.0: program CP_MES_CNTL : 0xc000000
Nov 17 11:32:59 oberon kernel: amdgpu 0000:03:00.0: [drm] DMUB hardware initialized: version=0x0A000800
Nov 17 11:32:59 oberon kernel: amdgpu 0000:03:00.0: [drm] Cannot find any crtc or sizes
Nov 17 11:32:59 oberon kernel: amdgpu 0000:03:00.0: ring gfx_0.0.0 uses VM inv eng 0 on hub 0
Nov 17 11:32:59 oberon kernel: amdgpu 0000:03:00.0: ring comp_1.0.0 uses VM inv eng 1 on hub 0
Nov 17 11:32:59 oberon kernel: amdgpu 0000:03:00.0: ring comp_1.1.0 uses VM inv eng 4 on hub 0
Nov 17 11:32:59 oberon kernel: amdgpu 0000:03:00.0: ring comp_1.0.1 uses VM inv eng 7 on hub 0
Nov 17 11:32:59 oberon kernel: amdgpu 0000:03:00.0: ring comp_1.1.1 uses VM inv eng 8 on hub 0
Nov 17 11:32:59 oberon kernel: amdgpu 0000:03:00.0: ring sdma0 uses VM inv eng 9 on hub 0
Nov 17 11:32:59 oberon kernel: amdgpu 0000:03:00.0: ring sdma1 uses VM inv eng 10 on hub 0
Nov 17 11:32:59 oberon kernel: amdgpu 0000:03:00.0: ring vcn_unified_0 uses VM inv eng 0 on hub 8
Nov 17 11:32:59 oberon kernel: amdgpu 0000:03:00.0: ring jpeg_dec uses VM inv eng 1 on hub 8
Nov 17 11:32:59 oberon kernel: amdgpu 0000:03:00.0: [drm] Cannot find any crtc or sizes
Nov 17 11:32:59 oberon kernel: amdgpu 0000:07:00.0: [drm] PCIE GART of 512M enabled (table at 0x00000087D6B00000).
Nov 17 11:32:59 oberon kernel: amdgpu 0000:07:00.0: PSP is resuming...
Nov 17 11:32:59 oberon kernel: amdgpu 0000:07:00.0: GECC is disabled, set amdgpu_ras_enable=1 to enable GECC in next boot cycle if needed
Nov 17 11:32:59 oberon kernel: amdgpu 0000:07:00.0: RAP: optional rap ta ucode is not available
Nov 17 11:32:59 oberon kernel: amdgpu 0000:07:00.0: SECUREDISPLAY: optional securedisplay ta ucode is not available
Nov 17 11:32:59 oberon kernel: amdgpu 0000:07:00.0: SMU is resuming...
Nov 17 11:32:59 oberon kernel: amdgpu 0000:07:00.0: smu driver if version = 0x0000002e, smu fw if version = 0x00000033, smu fw program = 0, smu fw version = 0x00684c00 (104.76.0)
Nov 17 11:32:59 oberon kernel: amdgpu 0000:07:00.0: SMU is resumed successfully!
Nov 17 11:32:59 oberon kernel: amdgpu 0000:07:00.0: program CP_MES_CNTL : 0x4000000
Nov 17 11:32:59 oberon kernel: amdgpu 0000:07:00.0: program CP_MES_CNTL : 0xc000000
Nov 17 11:32:59 oberon kernel: amdgpu 0000:07:00.0: [drm] DMUB hardware initialized: version=0x0A000800
Nov 17 11:32:59 oberon kernel: amdgpu 0000:07:00.0: [drm] Cannot find any crtc or sizes
Nov 17 11:32:59 oberon kernel: amdgpu 0000:07:00.0: ring gfx_0.0.0 uses VM inv eng 0 on hub 0
Nov 17 11:32:59 oberon kernel: amdgpu 0000:07:00.0: ring comp_1.0.0 uses VM inv eng 1 on hub 0
Nov 17 11:32:59 oberon kernel: amdgpu 0000:07:00.0: ring comp_1.1.0 uses VM inv eng 4 on hub 0
Nov 17 11:32:59 oberon kernel: amdgpu 0000:07:00.0: ring comp_1.0.1 uses VM inv eng 7 on hub 0
Nov 17 11:32:59 oberon kernel: amdgpu 0000:07:00.0: ring comp_1.1.1 uses VM inv eng 8 on hub 0
Nov 17 11:32:59 oberon kernel: amdgpu 0000:07:00.0: ring sdma0 uses VM inv eng 9 on hub 0
Nov 17 11:32:59 oberon kernel: amdgpu 0000:07:00.0: ring sdma1 uses VM inv eng 10 on hub 0
Nov 17 11:32:59 oberon kernel: amdgpu 0000:07:00.0: ring vcn_unified_0 uses VM inv eng 0 on hub 8
Nov 17 11:32:59 oberon kernel: amdgpu 0000:07:00.0: ring jpeg_dec uses VM inv eng 1 on hub 8
Nov 17 11:32:59 oberon kernel: amdgpu 0000:07:00.0: [drm] Cannot find any crtc or sizes
Nov 17 12:00:58 oberon kernel: strata-monitor-[149746]: segfault at 28 ip 00005696a2f1d108 sp 00007ffe493f26e0 error 4 in strata-monitor-test[34108,5696a2ef9000+101000] likely on CPU 14 (core 14, socket 0)
Nov 17 12:01:00 oberon kernel: strata-monitor-[149781]: segfault at 28 ip 00005d48d863d108 sp 00007ffe6c565100 error 4 in strata-monitor-test[34108,5d48d8619000+101000] likely on CPU 6 (core 6, socket 0)
Nov 17 12:01:07 oberon kernel: strata-monitor-[149836]: segfault at 28 ip 0000643fb4adf108 sp 00007ffe65bdf400 error 4 in strata-monitor-test[34108,643fb4abb000+101000] likely on CPU 28 (core 12, socket 0)
Nov 17 12:26:59 oberon kernel: strata-monitor[154448]: segfault at 722d34361a38 ip 000063b92bbd0adf sp 0000722d353fd680 error 6 in strata-monitor[169adf,63b92ba81000+176000] likely on CPU 4 (core 4, socket 0)
Nov 17 12:27:01 oberon kernel: strata-monitor[154512]: segfault at 7e62c839e5e8 ip 00006098b9552adf sp 00007e62c8bfd680 error 6 in strata-monitor[169adf,6098b9403000+176000] likely on CPU 16 (core 0, socket 0)
Nov 17 12:27:04 oberon kernel: strata-monitor[154580]: segfault at 7dbe40e12968 ip 000063289ecb8adf sp 00007dbe41ffd680 error 6 in strata-monitor[169adf,63289eb69000+176000] likely on CPU 0 (core 0, socket 0)
Nov 17 12:27:06 oberon kernel: strata-monitor[154651]: segfault at 735b51b5ce00 ip 0000570f98af5adf sp 0000735b52bfd680 error 6 in strata-monitor[169adf,570f989a6000+176000] likely on CPU 3 (core 3, socket 0)
Nov 17 12:27:09 oberon kernel: strata-monitor[154717]: segfault at 774f7037e628 ip 0000581f91affadf sp 0000774f70bfd680 error 6 in strata-monitor[169adf,581f919b0000+176000] likely on CPU 4 (core 4, socket 0)
Nov 17 12:27:11 oberon kernel: strata-monitor[154784]: segfault at 5aa ip 00007c62214a00b4 sp 00007c62213fd6b8 error 4 in libc.so.6[a00b4,7c6221428000+189000] likely on CPU 4 (core 4, socket 0)
Nov 17 12:27:13 oberon kernel: strata-monitor[154843]: segfault at 733bec42d880 ip 0000595f56763adf sp 0000733bed5fd680 error 6 in strata-monitor[169adf,595f56614000+176000] likely on CPU 4 (core 4, socket 0)
Nov 17 12:27:16 oberon kernel: strata-monitor[154909]: segfault at 7e9a0aba2610 ip 00005bd6e4037adf sp 00007e9a0bbfd680 error 6 in strata-monitor[169adf,5bd6e3ee8000+176000] likely on CPU 10 (core 10, socket 0)
Nov 17 12:27:19 oberon kernel: strata-monitor[154986]: segfault at 701b88741e10 ip 00005f608f2bcadf sp 0000701b897fd680 error 6 in strata-monitor[169adf,5f608f16d000+176000] likely on CPU 0 (core 0, socket 0)
Nov 17 13:10:31 oberon kernel: amdgpu 0000:03:00.0: [drm] PCIE GART of 512M enabled (table at 0x00000087D6B00000).
Nov 17 13:10:31 oberon kernel: amdgpu 0000:03:00.0: PSP is resuming...
Nov 17 13:10:32 oberon kernel: amdgpu 0000:03:00.0: GECC is disabled, set amdgpu_ras_enable=1 to enable GECC in next boot cycle if needed
Nov 17 13:10:32 oberon kernel: amdgpu 0000:03:00.0: RAP: optional rap ta ucode is not available
Nov 17 13:10:32 oberon kernel: amdgpu 0000:03:00.0: SECUREDISPLAY: optional securedisplay ta ucode is not available
Nov 17 13:10:32 oberon kernel: amdgpu 0000:03:00.0: SMU is resuming...
Nov 17 13:10:32 oberon kernel: amdgpu 0000:03:00.0: smu driver if version = 0x0000002e, smu fw if version = 0x00000033, smu fw program = 0, smu fw version = 0x00684c00 (104.76.0)
Nov 17 13:10:32 oberon kernel: amdgpu 0000:03:00.0: SMU is resumed successfully!
Nov 17 13:10:32 oberon kernel: amdgpu 0000:03:00.0: program CP_MES_CNTL : 0x4000000
Nov 17 13:10:32 oberon kernel: amdgpu 0000:03:00.0: program CP_MES_CNTL : 0xc000000
Nov 17 13:10:32 oberon kernel: amdgpu 0000:03:00.0: [drm] DMUB hardware initialized: version=0x0A000800
Nov 17 13:10:32 oberon kernel: amdgpu 0000:03:00.0: [drm] Cannot find any crtc or sizes
Nov 17 13:10:32 oberon kernel: amdgpu 0000:03:00.0: ring gfx_0.0.0 uses VM inv eng 0 on hub 0
Nov 17 13:10:32 oberon kernel: amdgpu 0000:03:00.0: ring comp_1.0.0 uses VM inv eng 1 on hub 0
Nov 17 13:10:32 oberon kernel: amdgpu 0000:03:00.0: ring comp_1.1.0 uses VM inv eng 4 on hub 0
Nov 17 13:10:32 oberon kernel: amdgpu 0000:03:00.0: ring comp_1.0.1 uses VM inv eng 7 on hub 0
Nov 17 13:10:32 oberon kernel: amdgpu 0000:03:00.0: ring comp_1.1.1 uses VM inv eng 8 on hub 0
Nov 17 13:10:32 oberon kernel: amdgpu 0000:03:00.0: ring sdma0 uses VM inv eng 9 on hub 0
Nov 17 13:10:32 oberon kernel: amdgpu 0000:03:00.0: ring sdma1 uses VM inv eng 10 on hub 0
Nov 17 13:10:32 oberon kernel: amdgpu 0000:03:00.0: ring vcn_unified_0 uses VM inv eng 0 on hub 8
Nov 17 13:10:32 oberon kernel: amdgpu 0000:03:00.0: ring jpeg_dec uses VM inv eng 1 on hub 8
Nov 17 13:10:32 oberon kernel: amdgpu 0000:03:00.0: [drm] Cannot find any crtc or sizes
Nov 17 13:10:32 oberon kernel: amdgpu 0000:07:00.0: [drm] PCIE GART of 512M enabled (table at 0x00000087D6B00000).
Nov 17 13:10:32 oberon kernel: amdgpu 0000:07:00.0: PSP is resuming...
Nov 17 13:10:32 oberon kernel: amdgpu 0000:07:00.0: GECC is disabled, set amdgpu_ras_enable=1 to enable GECC in next boot cycle if needed
Nov 17 13:10:32 oberon kernel: amdgpu 0000:07:00.0: RAP: optional rap ta ucode is not available
Nov 17 13:10:32 oberon kernel: amdgpu 0000:07:00.0: SECUREDISPLAY: optional securedisplay ta ucode is not available
Nov 17 13:10:32 oberon kernel: amdgpu 0000:07:00.0: SMU is resuming...
Nov 17 13:10:32 oberon kernel: amdgpu 0000:07:00.0: smu driver if version = 0x0000002e, smu fw if version = 0x00000033, smu fw program = 0, smu fw version = 0x00684c00 (104.76.0)
Nov 17 13:10:32 oberon kernel: amdgpu 0000:07:00.0: SMU is resumed successfully!
Nov 17 13:10:32 oberon kernel: amdgpu 0000:07:00.0: program CP_MES_CNTL : 0x4000000
Nov 17 13:10:32 oberon kernel: amdgpu 0000:07:00.0: program CP_MES_CNTL : 0xc000000
Nov 17 13:10:32 oberon kernel: amdgpu 0000:07:00.0: [drm] DMUB hardware initialized: version=0x0A000800
Nov 17 13:10:32 oberon kernel: amdgpu 0000:07:00.0: [drm] Cannot find any crtc or sizes
Nov 17 13:10:32 oberon kernel: amdgpu 0000:07:00.0: ring gfx_0.0.0 uses VM inv eng 0 on hub 0
Nov 17 13:10:32 oberon kernel: amdgpu 0000:07:00.0: ring comp_1.0.0 uses VM inv eng 1 on hub 0
Nov 17 13:10:32 oberon kernel: amdgpu 0000:07:00.0: ring comp_1.1.0 uses VM inv eng 4 on hub 0
Nov 17 13:10:32 oberon kernel: amdgpu 0000:07:00.0: ring comp_1.0.1 uses VM inv eng 7 on hub 0
Nov 17 13:10:32 oberon kernel: amdgpu 0000:07:00.0: ring comp_1.1.1 uses VM inv eng 8 on hub 0
Nov 17 13:10:32 oberon kernel: amdgpu 0000:07:00.0: ring sdma0 uses VM inv eng 9 on hub 0
Nov 17 13:10:32 oberon kernel: amdgpu 0000:07:00.0: ring sdma1 uses VM inv eng 10 on hub 0
Nov 17 13:10:32 oberon kernel: amdgpu 0000:07:00.0: ring vcn_unified_0 uses VM inv eng 0 on hub 8
Nov 17 13:10:32 oberon kernel: amdgpu 0000:07:00.0: ring jpeg_dec uses VM inv eng 1 on hub 8
Nov 17 13:10:32 oberon kernel: amdgpu 0000:07:00.0: [drm] Cannot find any crtc or sizes
```

### Baseline coredumpctl list (post-run identical)

```text
TIME                           PID  UID  GID SIG     COREFILE EXE                                                                  SIZE
Tue 2026-11-17 12:00:58 CET 149746 1000 1000 SIGSEGV missing  /home/chris/workspace/strata-monitor-ftxui/bin/strata-monitor-test      -
Tue 2026-11-17 12:01:00 CET 149781 1000 1000 SIGSEGV missing  /home/chris/workspace/strata-monitor-ftxui/bin/strata-monitor-test      -
Tue 2026-11-17 12:01:08 CET 149836 1000 1000 SIGSEGV missing  /home/chris/workspace/strata-monitor-ftxui/bin/strata-monitor-test      -
Tue 2026-11-17 12:20:02 CET 152827 1000 1000 SIGABRT present  /home/chris/workspace/strata-monitor-ftxui/bin/strata-monitor-test 159.5K
Tue 2026-11-17 12:26:59 CET 154447 1000 1000 SIGSEGV present  /home/chris/workspace/strata-monitor-ftxui/bin/strata-monitor       13.1M
Tue 2026-11-17 12:27:02 CET 154511 1000 1000 SIGSEGV present  /home/chris/workspace/strata-monitor-ftxui/bin/strata-monitor       13.9M
Tue 2026-11-17 12:27:04 CET 154579 1000 1000 SIGSEGV present  /home/chris/workspace/strata-monitor-ftxui/bin/strata-monitor       13.5M
Tue 2026-11-17 12:27:06 CET 154650 1000 1000 SIGSEGV present  /home/chris/workspace/strata-monitor-ftxui/bin/strata-monitor       13.7M
Tue 2026-11-17 12:27:09 CET 154716 1000 1000 SIGSEGV present  /home/chris/workspace/strata-monitor-ftxui/bin/strata-monitor       13.7M
Tue 2026-11-17 12:27:11 CET 154783 1000 1000 SIGSEGV present  /home/chris/workspace/strata-monitor-ftxui/bin/strata-monitor       12.9M
Tue 2026-11-17 12:27:14 CET 154842 1000 1000 SIGSEGV present  /home/chris/workspace/strata-monitor-ftxui/bin/strata-monitor       13.3M
Tue 2026-11-17 12:27:17 CET 154908 1000 1000 SIGSEGV present  /home/chris/workspace/strata-monitor-ftxui/bin/strata-monitor       13.7M
Tue 2026-11-17 12:27:19 CET 154985 1000 1000 SIGSEGV present  /home/chris/workspace/strata-monitor-ftxui/bin/strata-monitor       13.2M
Tue 2026-11-17 12:30:44 CET 156838 1000 1000 SIGSEGV present  /home/chris/workspace/strata-monitor-ftxui/bin/strata-monitor       10.7M
Tue 2026-11-17 12:33:48 CET 159039 1000 1000 SIGSEGV present  /home/chris/workspace/strata-monitor-ftxui/bin/strata-monitor       10.5M
```
