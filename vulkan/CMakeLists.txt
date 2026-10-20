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
