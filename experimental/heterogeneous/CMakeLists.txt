cmake_minimum_required(VERSION 3.20)
project(strata_heterogeneous_host LANGUAGES CXX)

if(NOT CMAKE_SYSTEM_NAME STREQUAL "Linux")
  message(FATAL_ERROR "The first heterogeneous host transport prototype requires Linux AF_UNIX sockets and posix_spawn")
endif()

option(STRATA_HETERO_HOST_TESTS "Build CPU-only cross-backend host protocol tests" ON)
add_library(strata_heterogeneous_host STATIC src/host.cpp src/session.cpp src/process.cpp src/shared.cpp src/activation.cpp src/device.cpp src/final_result.cpp src/final_pipeline.cpp src/stage_weights.cpp
  src/cuda_worker_config.cpp src/cuda_worker_bootstrap.cpp
  src/window_admission.cpp)
target_include_directories(strata_heterogeneous_host PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_compile_features(strata_heterogeneous_host PUBLIC cxx_std_17)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  target_compile_options(strata_heterogeneous_host PRIVATE -Wall -Wextra -Wpedantic -Werror)
endif()

# Optional CUDA-only adapter archive. A future model-loading worker must link
# this with strata_engine from a CUDA-only engine build; no engine is fetched
# or compiled by this standalone project. Host/mock targets never link cudart.
option(STRATA_HETERO_CUDA_VERIFIER "Compile the real CUDA non-final verifier adapter" OFF)
if(STRATA_HETERO_CUDA_VERIFIER)
  if(STRATA_ENABLE_HIP OR STRATA_HIP_GFX906 OR STRATA_ENABLE_SYCL OR STRATA_ENABLE_METAL)
    message(FATAL_ERROR "The verifier adapter requires a separate CUDA-only build")
  endif()
  find_package(CUDAToolkit REQUIRED)
  add_library(strata_heterogeneous_cuda_verifier STATIC
    src/cuda_verifier.cpp src/cuda_final_verifier.cpp src/cuda_stage_resources.cpp
    src/cuda_file_experts.cpp src/cuda_native_stage.cpp)
  target_include_directories(strata_heterogeneous_cuda_verifier PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../../include)
  target_link_libraries(strata_heterogeneous_cuda_verifier PUBLIC strata_heterogeneous_host CUDA::cudart)
  target_compile_options(strata_heterogeneous_cuda_verifier PRIVATE -Wall -Wextra -Wpedantic -Werror)
endif()

# M1.16: native executable is available only when this subproject is
# deliberately added by the top-level CUDA-only Strata build. Do not
# attempt to link the executable in standalone CPU or HIP configurations.
option(STRATA_HETERO_BUILD_CUDA_WORKER "Build opt-in native CUDA model-loading worker" OFF)
if(STRATA_HETERO_BUILD_CUDA_WORKER)
  if(NOT STRATA_HETERO_CUDA_VERIFIER OR NOT TARGET strata_engine)
    message(FATAL_ERROR "CUDA worker requires the parent CUDA strata_engine target and STRATA_HETERO_CUDA_VERIFIER=ON")
  endif()
  add_executable(strata_heterogeneous_cuda_worker src/cuda_worker_main.cpp)
  target_link_libraries(strata_heterogeneous_cuda_worker PRIVATE
    strata_heterogeneous_cuda_verifier strata_engine)
  target_compile_features(strata_heterogeneous_cuda_worker PRIVATE cxx_std_17)
  if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(strata_heterogeneous_cuda_worker PRIVATE -Wall -Wextra -Wpedantic -Werror)
  endif()
endif()

# Two independent CPU mock executables. They deliberately do not initialize
# either GPU runtime, but verify that each worker has a backend-specific image.
foreach(_mock_backend IN ITEMS cuda hip)
  add_executable(strata_heterogeneous_mock_${_mock_backend}_worker src/mock_worker.cpp)
  target_link_libraries(strata_heterogeneous_mock_${_mock_backend}_worker PRIVATE strata_heterogeneous_host)
  target_compile_definitions(strata_heterogeneous_mock_${_mock_backend}_worker PRIVATE
    STRATA_HETERO_MOCK_BACKEND=$<IF:$<STREQUAL:${_mock_backend},cuda>,1,2>)
  if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(strata_heterogeneous_mock_${_mock_backend}_worker PRIVATE -Wall -Wextra -Wpedantic -Werror)
  endif()
endforeach()

add_executable(strata_heterogeneous_demo src/demo.cpp)
target_link_libraries(strata_heterogeneous_demo PRIVATE strata_heterogeneous_host)
target_compile_features(strata_heterogeneous_demo PRIVATE cxx_std_17)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  target_compile_options(strata_heterogeneous_demo PRIVATE -Wall -Wextra -Wpedantic -Werror)
endif()

if(STRATA_HETERO_HOST_TESTS)
  enable_testing()
  # M1.18: bounded protocol gate and forked CPU worker failures.
  add_executable(strata_heterogeneous_window_admission_tests
    tests/window_admission_tests.cpp)
  target_link_libraries(strata_heterogeneous_window_admission_tests PRIVATE
    strata_heterogeneous_host)
  add_test(NAME strata_heterogeneous_window_admission_tests
    COMMAND strata_heterogeneous_window_admission_tests)
  set_tests_properties(strata_heterogeneous_window_admission_tests PROPERTIES TIMEOUT 20)
  add_executable(strata_heterogeneous_window_admission_process_tests
    tests/window_admission_process_tests.cpp)
  target_link_libraries(strata_heterogeneous_window_admission_process_tests PRIVATE
    strata_heterogeneous_host)
  add_test(NAME strata_heterogeneous_window_admission_process_tests
    COMMAND strata_heterogeneous_window_admission_process_tests)
  set_tests_properties(strata_heterogeneous_window_admission_process_tests PROPERTIES TIMEOUT 30)
  if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    foreach(_t strata_heterogeneous_window_admission_tests
               strata_heterogeneous_window_admission_process_tests)
      target_compile_options(${_t} PRIVATE -Wall -Wextra -Wpedantic -Werror)
    endforeach()
  endif()
  # M1.17 NativeDense + PLE lifetime and failure contracts. Compile the
  # actual owner and new CUDA native-stage sources against CPU-only CUDA/core
  # doubles; no GPU or model files are opened.
  add_executable(strata_heterogeneous_cuda_native_stage_contract_tests
    tests/cuda_native_stage_tests.cpp src/cuda_native_stage.cpp
    src/cuda_stage_resources.cpp)
  target_include_directories(strata_heterogeneous_cuda_native_stage_contract_tests BEFORE
    PRIVATE tests/cuda_native_stubs)
  target_link_libraries(strata_heterogeneous_cuda_native_stage_contract_tests PRIVATE
    strata_heterogeneous_host)
  if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(strata_heterogeneous_cuda_native_stage_contract_tests PRIVATE
      -Wall -Wextra -Wpedantic -Werror)
  endif()
  add_test(NAME strata_heterogeneous_cuda_native_stage_contract_tests
    COMMAND strata_heterogeneous_cuda_native_stage_contract_tests)
  set_tests_properties(strata_heterogeneous_cuda_native_stage_contract_tests PROPERTIES
    TIMEOUT 20)
  # M1.16 startup config and supervisor argv parsers: strictly CPU-only.
  add_executable(strata_heterogeneous_cuda_worker_config_tests tests/cuda_worker_config_tests.cpp)
  target_link_libraries(strata_heterogeneous_cuda_worker_config_tests PRIVATE strata_heterogeneous_host)
  add_test(NAME strata_heterogeneous_cuda_worker_config_tests
    COMMAND strata_heterogeneous_cuda_worker_config_tests)
  set_tests_properties(strata_heterogeneous_cuda_worker_config_tests PROPERTIES TIMEOUT 20)
  add_executable(strata_heterogeneous_cuda_worker_bootstrap_tests tests/cuda_worker_bootstrap_tests.cpp)
  target_link_libraries(strata_heterogeneous_cuda_worker_bootstrap_tests PRIVATE strata_heterogeneous_host)
  add_test(NAME strata_heterogeneous_cuda_worker_bootstrap_tests
    COMMAND strata_heterogeneous_cuda_worker_bootstrap_tests)
  set_tests_properties(strata_heterogeneous_cuda_worker_bootstrap_tests PROPERTIES TIMEOUT 20)
  if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(strata_heterogeneous_cuda_worker_config_tests PRIVATE -Wall -Wextra -Wpedantic -Werror)
    target_compile_options(strata_heterogeneous_cuda_worker_bootstrap_tests PRIVATE -Wall -Wextra -Wpedantic -Werror)
  endif()
  # Compile the SAME adapter source against explicit CPU API doubles. These
  # tests validate contract/lifecycle only, never CUDA or model numerics.
  add_executable(strata_heterogeneous_cuda_verifier_contract_tests
    tests/cuda_verifier_tests.cpp src/cuda_verifier.cpp)
  target_include_directories(strata_heterogeneous_cuda_verifier_contract_tests BEFORE PRIVATE tests/cuda_stubs)
  target_link_libraries(strata_heterogeneous_cuda_verifier_contract_tests PRIVATE strata_heterogeneous_host)
  target_compile_options(strata_heterogeneous_cuda_verifier_contract_tests PRIVATE -Wall -Wextra -Wpedantic -Werror)
  add_test(NAME strata_heterogeneous_cuda_verifier_contract_tests COMMAND strata_heterogeneous_cuda_verifier_contract_tests)
  set_tests_properties(strata_heterogeneous_cuda_verifier_contract_tests PROPERTIES TIMEOUT 20)
  # Compile the actual final CUDA adapter against CPU CUDA/Verifier doubles.
  # The host/mock build remains runtime-free and no GPU is initialized.
  add_executable(strata_heterogeneous_cuda_final_verifier_contract_tests
    tests/cuda_final_verifier_tests.cpp src/cuda_final_verifier.cpp)
  target_include_directories(strata_heterogeneous_cuda_final_verifier_contract_tests BEFORE PRIVATE
    tests/cuda_final_stubs tests/cuda_stubs)
  target_link_libraries(strata_heterogeneous_cuda_final_verifier_contract_tests PRIVATE strata_heterogeneous_host)
  if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(strata_heterogeneous_cuda_final_verifier_contract_tests PRIVATE
      -Wall -Wextra -Wpedantic -Werror)
  endif()
  add_test(NAME strata_heterogeneous_cuda_final_verifier_contract_tests
    COMMAND strata_heterogeneous_cuda_final_verifier_contract_tests)
  set_tests_properties(strata_heterogeneous_cuda_final_verifier_contract_tests PROPERTIES TIMEOUT 20)
  # M1.15: compile the production file/cache/pool provider source against
  # explicit CUDA, expert-source, cache and CPU-pool doubles (no GPU).
  add_executable(strata_heterogeneous_cuda_file_experts_contract_tests
    tests/cuda_file_experts_tests.cpp src/cuda_file_experts.cpp)
  target_include_directories(strata_heterogeneous_cuda_file_experts_contract_tests BEFORE PRIVATE
    tests/cuda_expert_stubs)
  target_link_libraries(strata_heterogeneous_cuda_file_experts_contract_tests PRIVATE
    strata_heterogeneous_host)
  if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(strata_heterogeneous_cuda_file_experts_contract_tests PRIVATE
      -Wall -Wextra -Wpedantic -Werror)
  endif()
  add_test(NAME strata_heterogeneous_cuda_file_experts_contract_tests
    COMMAND strata_heterogeneous_cuda_file_experts_contract_tests)
  set_tests_properties(strata_heterogeneous_cuda_file_experts_contract_tests PROPERTIES TIMEOUT 20)
  # Native pack/index.txt stage selection: CPU-only, does not initialize CUDA.
  # M1.14: compile the actual CUDA resource owner against CPU-only CUDA,
  # WeightTable and session doubles; do not initialize the GPU.
  add_executable(strata_heterogeneous_cuda_stage_resources_contract_tests
    tests/cuda_stage_resources_tests.cpp src/cuda_stage_resources.cpp)
  target_include_directories(strata_heterogeneous_cuda_stage_resources_contract_tests BEFORE PRIVATE
    tests/cuda_resource_stubs)
  target_link_libraries(strata_heterogeneous_cuda_stage_resources_contract_tests PRIVATE
    strata_heterogeneous_host)
  if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(strata_heterogeneous_cuda_stage_resources_contract_tests PRIVATE
      -Wall -Wextra -Wpedantic -Werror)
  endif()
  add_test(NAME strata_heterogeneous_cuda_stage_resources_contract_tests
    COMMAND strata_heterogeneous_cuda_stage_resources_contract_tests)
  set_tests_properties(strata_heterogeneous_cuda_stage_resources_contract_tests PROPERTIES TIMEOUT 20)
  add_executable(strata_heterogeneous_stage_weights_tests tests/stage_weights_tests.cpp)
  target_link_libraries(strata_heterogeneous_stage_weights_tests PRIVATE strata_heterogeneous_host)
  if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(strata_heterogeneous_stage_weights_tests PRIVATE -Wall -Wextra -Wpedantic -Werror)
  endif()
  add_test(NAME strata_heterogeneous_stage_weights_tests COMMAND strata_heterogeneous_stage_weights_tests)
  set_tests_properties(strata_heterogeneous_stage_weights_tests PROPERTIES TIMEOUT 20)
  add_executable(strata_heterogeneous_host_tests tests/host_tests.cpp)
  target_link_libraries(strata_heterogeneous_host_tests PRIVATE strata_heterogeneous_host)
  target_compile_features(strata_heterogeneous_host_tests PRIVATE cxx_std_17)
  if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(strata_heterogeneous_host_tests PRIVATE -Wall -Wextra -Wpedantic -Werror)
  endif()
  add_executable(strata_heterogeneous_activation_tests tests/activation_tests.cpp)
  target_link_libraries(strata_heterogeneous_activation_tests PRIVATE strata_heterogeneous_host)
  if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(strata_heterogeneous_activation_tests PRIVATE -Wall -Wextra -Wpedantic -Werror)
  endif()
  add_test(NAME strata_heterogeneous_activation_tests COMMAND strata_heterogeneous_activation_tests)
  set_tests_properties(strata_heterogeneous_activation_tests PROPERTIES TIMEOUT 20)
  # Standalone, GPU-free protocol test. No change to v4 activation framing.
  # Opt-in final-stage worker state machine: real forked CPU-only stages.
  add_executable(strata_heterogeneous_final_pipeline_tests tests/final_pipeline_tests.cpp)
  target_link_libraries(strata_heterogeneous_final_pipeline_tests PRIVATE strata_heterogeneous_host)
  if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(strata_heterogeneous_final_pipeline_tests PRIVATE -Wall -Wextra -Wpedantic -Werror)
  endif()
  add_test(NAME strata_heterogeneous_final_pipeline_tests COMMAND strata_heterogeneous_final_pipeline_tests)
  set_tests_properties(strata_heterogeneous_final_pipeline_tests PROPERTIES TIMEOUT 25)
  add_executable(strata_heterogeneous_final_result_tests tests/final_result_tests.cpp)
  target_link_libraries(strata_heterogeneous_final_result_tests PRIVATE strata_heterogeneous_host)
  if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(strata_heterogeneous_final_result_tests PRIVATE -Wall -Wextra -Wpedantic -Werror)
  endif()
  add_test(NAME strata_heterogeneous_final_result_tests COMMAND strata_heterogeneous_final_result_tests)
  set_tests_properties(strata_heterogeneous_final_result_tests PROPERTIES TIMEOUT 20)
  add_executable(strata_heterogeneous_window_tests tests/window_tests.cpp)
  target_link_libraries(strata_heterogeneous_window_tests PRIVATE strata_heterogeneous_host)
  if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(strata_heterogeneous_window_tests PRIVATE -Wall -Wextra -Wpedantic -Werror)
  endif()
  add_test(NAME strata_heterogeneous_window_tests COMMAND strata_heterogeneous_window_tests)
  set_tests_properties(strata_heterogeneous_window_tests PROPERTIES TIMEOUT 20)
  add_executable(strata_heterogeneous_device_tests tests/device_tests.cpp)
  target_link_libraries(strata_heterogeneous_device_tests PRIVATE strata_heterogeneous_host)
  if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(strata_heterogeneous_device_tests PRIVATE -Wall -Wextra -Wpedantic -Werror)
  endif()
  add_test(NAME strata_heterogeneous_device_tests COMMAND strata_heterogeneous_device_tests)
  set_tests_properties(strata_heterogeneous_device_tests PROPERTIES TIMEOUT 20)
  add_executable(strata_heterogeneous_shared_tests tests/shared_tests.cpp)
  target_link_libraries(strata_heterogeneous_shared_tests PRIVATE strata_heterogeneous_host)
  if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(strata_heterogeneous_shared_tests PRIVATE -Wall -Wextra -Wpedantic -Werror)
  endif()
  add_test(NAME strata_heterogeneous_shared_tests COMMAND strata_heterogeneous_shared_tests)
  set_tests_properties(strata_heterogeneous_shared_tests PROPERTIES TIMEOUT 20)
  add_executable(strata_heterogeneous_hang_worker tests/hang_worker.cpp)
  add_executable(strata_heterogeneous_process_tests tests/process_tests.cpp)
  target_link_libraries(strata_heterogeneous_process_tests PRIVATE strata_heterogeneous_host)
  if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(strata_heterogeneous_process_tests PRIVATE -Wall -Wextra -Wpedantic -Werror)
  endif()
  add_test(NAME strata_heterogeneous_process_tests COMMAND strata_heterogeneous_process_tests)
  set_tests_properties(strata_heterogeneous_process_tests PROPERTIES TIMEOUT 20)
  add_test(NAME strata_heterogeneous_host_tests COMMAND strata_heterogeneous_host_tests)
  set_tests_properties(strata_heterogeneous_host_tests PROPERTIES TIMEOUT 20)
  add_test(NAME strata_heterogeneous_exec_smoke COMMAND strata_heterogeneous_demo
    cuda:0:0-3,hip:0:3-7,hip:1:7-10 10 2 6 12345 8)
  set_tests_properties(strata_heterogeneous_exec_smoke PROPERTIES TIMEOUT 20 PASS_REGULAR_EXPRESSION "PASS:")
  add_test(NAME strata_heterogeneous_shared_smoke COMMAND strata_heterogeneous_demo
    cuda:0:0-3,hip:0:3-7,hip:1:7-10 10 2 6 12345 8 --shared)
  set_tests_properties(strata_heterogeneous_shared_smoke PROPERTIES TIMEOUT 20 PASS_REGULAR_EXPRESSION "PASS:")
endif()
