# The CUDA/HIP seam of a mixed layer split (.ai/SEAM.md): shared hand-off memory, the front link, the worker process.
# Host-only C++: no GPU runtime is used or linked here.  Builds on its own (cmake -S hetero) or from the top level
# with -DSTRATA_HETERO=ON.
cmake_minimum_required(VERSION 3.20)

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  project(strata_hetero LANGUAGES CXX)
  enable_testing()
endif()

if(NOT CMAKE_SYSTEM_NAME STREQUAL "Linux")
  message(FATAL_ERROR "strata_hetero needs Linux (memfd, AF_UNIX descriptor passing, posix_spawn)")
endif()

find_package(Threads REQUIRED)

add_library(strata_hetero STATIC
  src/shared_region.cpp
  src/front_link.cpp
  src/worker_process.cpp)
target_include_directories(strata_hetero PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_compile_features(strata_hetero PUBLIC cxx_std_17)
target_link_libraries(strata_hetero PUBLIC Threads::Threads)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  target_compile_options(strata_hetero PRIVATE -Wall -Wextra -Wpedantic -Werror)
endif()

# drives a front stage worker (an engine started with --front-stage) without a host engine; host-only
add_executable(hetero_front_client tools/front_client.cpp)
target_link_libraries(hetero_front_client PRIVATE strata_hetero)

option(STRATA_HETERO_TESTS "Build the CPU tests of the mixed-split seam" ON)
if(STRATA_HETERO_TESTS)
  add_executable(hetero_mock_worker tests/mock_worker.cpp)
  target_link_libraries(hetero_mock_worker PRIVATE strata_hetero)
  add_executable(hetero_tests tests/hetero_tests.cpp)
  target_link_libraries(hetero_tests PRIVATE strata_hetero)
  if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(hetero_mock_worker PRIVATE -Wall -Wextra -Wpedantic -Werror)
    target_compile_options(hetero_tests PRIVATE -Wall -Wextra -Wpedantic -Werror)
  endif()
  add_test(NAME hetero_tests COMMAND hetero_tests $<TARGET_FILE:hetero_mock_worker>)
  set_tests_properties(hetero_tests PROPERTIES TIMEOUT 60)
endif()

# ---- the shared hand-off probe: one source, built by a CUDA build and by a HIP build (probe/shared_handoff_probe.cu) ----
# Only from the top level, where a GPU engine is configured.  UNVERIFIED: written without a GPU toolchain at hand;
# the target wiring follows flag_order_test / hip_flag_order_test in cmake/strata_gpu_engine.cmake.
if(NOT CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  if(STRATA_ENABLE_HIP)
    set_source_files_properties(probe/shared_handoff_probe.cu PROPERTIES LANGUAGE HIP)
    add_executable(hip_hetero_shared_probe probe/shared_handoff_probe.cu)
    target_include_directories(hip_hetero_shared_probe PRIVATE "${CMAKE_SOURCE_DIR}/include")
    target_link_libraries(hip_hetero_shared_probe PRIVATE strata_hip_runtime Threads::Threads)
  elseif(STRATA_ENABLE_CUDA AND NOT STRATA_HIP_GFX906)
    add_executable(hetero_shared_probe probe/shared_handoff_probe.cu)
    target_include_directories(hetero_shared_probe PRIVATE "${CMAKE_SOURCE_DIR}/include")
    target_link_libraries(hetero_shared_probe PRIVATE strata_core Threads::Threads)
  endif()
endif()
