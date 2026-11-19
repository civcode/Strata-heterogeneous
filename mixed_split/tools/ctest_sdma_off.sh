#!/bin/sh
# CTest with HSA_ENABLE_SDMA=0 (.ai/PROJECT.md, the first rule: SDMA has a known bug in the AMD driver).
#   mixed_split/tools/ctest_sdma_off.sh --test-dir build-XXXX-hip -R conversation_readback_test --output-on-failure
if [ -n "${HSA_ENABLE_SDMA:-}" ] && [ "$HSA_ENABLE_SDMA" != "0" ]; then
  echo "HSA_ENABLE_SDMA=$HSA_ENABLE_SDMA: refused (.ai/PROJECT.md, the first rule)" >&2; exit 2
fi
HSA_ENABLE_SDMA=0 exec ctest "$@"
