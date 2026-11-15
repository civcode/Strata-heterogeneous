# Report 0038 — direct verified real-model candidate

## Chat AI implementation (prior to Oberon qualification)
Base: report 0037 commit 4066b61c8602a30027cc24aa9b68cd93f10926bb. Implementation 50d90353813b1a8e9d8b5865e4e57ddab07b0837, plus bounded CLI preflight and CPU adapter regression in handoff commit.

Opt-in native CLI `--direct-verified-shared`, with grouped HIP overlap and full logits. The same size-sealed shared activation slot is independently registered before HELLO, with separate device-local aliases and mode-specific capability. CUDA uses the unchanged native publisher and ready flag; after native drain, the original finite/checksum and identity checks run over the same mapped bytes. HIP consumes native mapped data without H2D staging. Original all-stage accepted-prefix commit and poisoned-session handling retained. This serial verified CUDA→HIP boundary is not real-model full prelaunch. No defaults or vanilla native kernels changed. Real-model finite gate independent of CPU admission, and fully prelaunched real-model path, remain outstanding.

Local: standalone C++ binding and prior-layout fixtures passed. Binding source/test hashes checked against uploaded blobs. Existing Python model-free supervisor 5/5 passed using fake workers. Both synthetic-header probe syntax checks passed (not native compilation). Integrated CPU protocol tests and new CUDA adapter fixture, native builds, actual model parity/lifecycle/throughput NOT_RUN: local GitHub DNS resolution prevented full checkout. No GitHub Actions.

## Oberon qualification
PENDING. Append actual results, fixes, and remaining blockers here per handoff 0038.
