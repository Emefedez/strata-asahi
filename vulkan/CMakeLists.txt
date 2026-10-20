# vulkan/ - Strata on Apple GPUs under Linux (Asahi, Honeykrisp): the Metal engine's design on Vulkan.  Its own CMake
# project like metal/ and sycl/, never part of the CUDA / HIP builds.
#
#   * src/mtl_rt_vk.cpp: metal/src/mtl_rt.h on Vulkan, so the Metal engine's kernel host code (metal/src/engine) and
#     the parity programs (metal/tests) build unchanged;
#   * kernels/*.slang: the Slang ports of metal/kernels/*.metal, same entry points and function constants, compiled
#     at run time (Slang) as Metal compiles its MSL;
#   * tests/: Vulkan-only checks.  The Metal parity programs are built as their kernels are ported (STRATA_VK_TESTS).
#
#   cmake -S vulkan -B build-vulkan -G Ninja -DCMAKE_BUILD_TYPE=Release [-DSLANG_DIR=~/.local/share/slang/2026.18.2]
#   ninja -C build-vulkan
# Runs on the Vulkan driver of honeykrisp-strata (VK_DRIVER_FILES=<its ICD>), which the engine needs for the device
# heap above 50% of RAM, bf16 and cooperative matrices; the parity programs also run on the stock Mesa.
cmake_minimum_required(VERSION 3.24)
project(strata_vulkan LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
set(CMAKE_CXX_FLAGS_RELEASE "-O2")

get_filename_component(STRATA_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/.." ABSOLUTE)
set(METAL_DIR "${STRATA_ROOT}/metal")

find_package(Vulkan REQUIRED)
find_program(GLSLANG glslangValidator REQUIRED)
set(SLANG_DIR "$ENV{HOME}/.local/share/slang/2026.18.2" CACHE PATH "Slang release (include/slang.h, lib/libslang.so)")
if(NOT EXISTS "${SLANG_DIR}/include/slang.h")
  message(FATAL_ERROR "SLANG_DIR='${SLANG_DIR}' has no include/slang.h (https://github.com/shader-slang/slang/releases)")
endif()
add_library(slang SHARED IMPORTED)
set_target_properties(slang PROPERTIES IMPORTED_LOCATION "${SLANG_DIR}/lib/libslang.so"
                                       INTERFACE_INCLUDE_DIRECTORIES "${SLANG_DIR}/include")

# The handshake kernel, compiled at build time and embedded (kHandshakeSpv).
set(HS_H "${CMAKE_CURRENT_BINARY_DIR}/handshake_spv.h")
add_custom_command(OUTPUT ${HS_H}
  COMMAND ${GLSLANG} --target-env vulkan1.3 -V --vn kHandshakeSpv -o ${HS_H} ${CMAKE_CURRENT_SOURCE_DIR}/src/handshake.comp
  DEPENDS src/handshake.comp)

add_library(mtl_rt STATIC src/mtl_rt_vk.cpp ${METAL_DIR}/src/ref_quant.cpp ${METAL_DIR}/src/gguf_lite.cpp ${HS_H})
# vulkan/src first: its resource_path.h is the Linux one
target_include_directories(mtl_rt BEFORE PUBLIC src)
target_include_directories(mtl_rt PUBLIC ${METAL_DIR}/src ${STRATA_ROOT}/third_party/ggml ${CMAKE_CURRENT_BINARY_DIR})
target_compile_definitions(mtl_rt PRIVATE STRATA_VULKAN_KERNELS_DEFAULT="${CMAKE_CURRENT_SOURCE_DIR}/kernels")
target_link_libraries(mtl_rt PUBLIC Vulkan::Vulkan slang)
set_target_properties(mtl_rt PROPERTIES BUILD_RPATH "${SLANG_DIR}/lib")

file(GLOB STRATA_VK_OWN_TESTS CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/tests/*.cpp)
foreach(src ${STRATA_VK_OWN_TESTS})
  get_filename_component(t ${src} NAME_WE)
  add_executable(${t} ${src})
  target_link_libraries(${t} PRIVATE mtl_rt)
  set_target_properties(${t} PROPERTIES BUILD_RPATH "${SLANG_DIR}/lib")
endforeach()

# The Metal parity programs whose kernels have a Slang port.
set(STRATA_VK_TESTS "" CACHE STRING "metal/tests programs to build against the Vulkan runtime (;-separated)")
foreach(t ${STRATA_VK_TESTS})
  add_executable(${t} ${METAL_DIR}/tests/${t}.cpp)
  target_link_libraries(${t} PRIVATE mtl_rt)
  set_target_properties(${t} PROPERTIES BUILD_RPATH "${SLANG_DIR}/lib")
endforeach()

# ---- the engine (strata) ------------------------------------------------------------------------------------------
# metal/CMakeLists.txt's engine on Vulkan: the shared host sources from ../src UNCHANGED (metal/include/metal_compat
# gives them the CUDA runtime names, as on the Mac), the device runtime src/gpu/runtime_vk.cpp, and the Metal engine's
# kernel host code metal/src/engine/*.cpp over mtl_rt_vk.cpp.  The build keeps STRATA_GPU_METAL: the host code's Metal
# paths are the unified-memory design, which is this one too.
#   cmake -S vulkan -B build-vulkan -G Ninja -DSTRATA_ENABLE_VULKAN=ON -DSTRATA_GGML_DIR=<llama.cpp at 3cf03257>
option(STRATA_ENABLE_VULKAN "Build the engine (strata) for Vulkan" OFF)
set(STRATA_GGML_DIR "" CACHE PATH "llama.cpp checkout at the pinned commit")
if(NOT STRATA_ENABLE_VULKAN)
  return()
endif()
if(NOT EXISTS "${STRATA_GGML_DIR}/ggml/CMakeLists.txt")
  message(FATAL_ERROR "STRATA_GGML_DIR='${STRATA_GGML_DIR}' is not a llama.cpp checkout")
endif()
set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
set(GGML_STATIC ON CACHE BOOL "" FORCE)
set(GGML_METAL OFF CACHE BOOL "" FORCE)
set(GGML_VULKAN OFF CACHE BOOL "" FORCE)
set(GGML_BLAS OFF CACHE BOOL "" FORCE)
set(GGML_OPENMP OFF CACHE BOOL "" FORCE)
set(GGML_BACKEND_DL OFF CACHE BOOL "" FORCE)
set(GGML_NATIVE ON CACHE BOOL "" FORCE)
set(GGML_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(GGML_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
add_subdirectory(${STRATA_GGML_DIR}/ggml ${CMAKE_BINARY_DIR}/ggml EXCLUDE_FROM_ALL)

add_library(strata_gpu_vulkan STATIC src/gpu/runtime_vk.cpp)
target_include_directories(strata_gpu_vulkan BEFORE PUBLIC ${METAL_DIR}/include/metal_compat ${METAL_DIR}/include
                           ${STRATA_ROOT}/include)
target_compile_definitions(strata_gpu_vulkan PUBLIC STRATA_GPU_METAL=1 STRATA_GPU_VULKAN=1)

set(S ${STRATA_ROOT}/src)
set(STRATA_VK_HOST_SOURCES
  ${S}/core/device.cu ${S}/core/pinned.cu ${S}/platform/memory.cpp ${S}/core/graph.cpp ${S}/core/weights.cpp ${S}/core/layout.cpp
  ${S}/core/layer.cpp ${S}/core/session.cpp ${S}/core/expert_source.cpp ${S}/core/remote_experts.cpp ${S}/core/expert_cache.cpp
  ${S}/core/native_head.cpp ${S}/core/native_dense.cpp ${S}/core/verify.cpp ${S}/core/mtp.cpp
  ${S}/core/conversation_snapshot.cpp ${S}/core/conversation_state.cpp ${S}/core/conversation_memory.cpp
  ${S}/core/peer_experts.cpp ${S}/core/arch_defaults.cpp ${S}/core/vmm.cpp ${S}/core/conversation_file.cpp
  ${S}/kernels/ngram.cpp ${S}/ngram/ple_reader.cpp ${S}/platform/direct_file.cpp
  ${S}/prefill/prefill.cpp
  ${S}/spec/suffix_drafter.cpp ${S}/spec/controller.cpp ${S}/spec/draft_policy.cpp ${S}/spec/draft_source.cpp
  ${S}/kernels/cpu/pool.cpp ${S}/kernels/cpu/expert_layout.cpp ${S}/kernels/cpu/native_expert.cpp
  ${S}/kernels/cpu/hadamard.cpp ${S}/kernels/cpu/tq_cpu.cpp)
set_source_files_properties(${S}/core/device.cu ${S}/core/pinned.cu PROPERTIES LANGUAGE CXX)

add_library(strata_vk_host STATIC ${STRATA_VK_HOST_SOURCES})
target_include_directories(strata_vk_host PUBLIC ${STRATA_ROOT}/include ${S}/core ${STRATA_ROOT}/third_party/ggml
                           ${STRATA_GGML_DIR}/ggml/include ${STRATA_GGML_DIR}/ggml/src ${STRATA_GGML_DIR}/ggml/src/ggml-cpu)
target_compile_definitions(strata_vk_host PUBLIC STRATA_NATIVE_EXPERTS=1 STRATA_VERSION="0.1.46-vulkan")
target_compile_options(strata_vk_host PRIVATE -Wno-unused-parameter -Wno-deprecated-declarations)
target_link_libraries(strata_vk_host PUBLIC strata_gpu_vulkan ggml-cpu ggml-base)

# The Metal engine's kernel host code, unchanged where it can be (the macOS-only calls have Linux branches in it).
file(GLOB STRATA_VK_ENGINE_SOURCES CONFIGURE_DEPENDS ${METAL_DIR}/src/engine/*.cpp)
list(APPEND STRATA_VK_ENGINE_SOURCES src/engine_stubs_linux.cpp)
add_library(strata_vk_kernels STATIC ${STRATA_VK_ENGINE_SOURCES})
target_include_directories(strata_vk_kernels BEFORE PRIVATE src)
target_include_directories(strata_vk_kernels PRIVATE ${METAL_DIR}/src ${STRATA_ROOT}/third_party/ggml)
target_compile_definitions(strata_vk_kernels PRIVATE
  STRATA_METAL_ENGINE_KERNELS_DEFAULT="${CMAKE_CURRENT_SOURCE_DIR}/kernels/engine"
  STRATA_METAL_DATA_DEFAULT="${STRATA_ROOT}/data")
target_link_libraries(strata_vk_kernels PUBLIC strata_vk_host mtl_rt)

add_executable(strata ${S}/program/generate.cpp)
target_link_libraries(strata PRIVATE strata_vk_kernels strata_vk_host strata_vk_kernels)
set_target_properties(strata PROPERTIES BUILD_RPATH "${SLANG_DIR}/lib" INSTALL_RPATH "${SLANG_DIR}/lib")
