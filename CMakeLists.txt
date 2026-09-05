# CMakeLists.txt — build alternativo a Make (Fase 3; CI lo usa en Linux).
# Mantiene `make` como vía principal en Windows/MinGW.
cmake_minimum_required(VERSION 3.16)
project(gguf2bin2 C)
set(CMAKE_C_STANDARD 99)

option(G2B_AVX2 "kernels AVX2" ON)
option(G2B_OPENMP "OpenMP" ON)
option(G2B_BUILD_TESTS "herramientas de test" ON)

set(G2B_CORE
  src/l1_gguf.c src/l2_codec.c src/l3_math.c src/l4_gbin.c
  src/model.c src/kv.c src/forward_dense.c src/forward_lfm2.c
  src/forward_hybrid.c src/forward_prefill.c
  src/l6_token.c src/l7_vulkan.c src/l8_cyber.c
  src/g2b_api.c src/os_mm.c src/sampler.c src/opts.c src/g2bx_io.c)

add_library(g2bcore STATIC ${G2B_CORE})
target_include_directories(g2bcore PUBLIC include PRIVATE src)
if(MSVC)
  target_compile_options(g2bcore PRIVATE /O2 /W3)
else()
  target_compile_options(g2bcore PRIVATE -O3 -ffast-math -Wall -Wextra)
  if(G2B_AVX2)
    target_compile_options(g2bcore PRIVATE -mavx2 -mfma -mf16c)
  endif()
endif()
if(G2B_OPENMP)
  find_package(OpenMP)
  if(OpenMP_C_FOUND)
    target_link_libraries(g2bcore PUBLIC OpenMP::OpenMP_C)
  endif()
endif()
if(NOT WIN32)
  target_link_libraries(g2bcore PUBLIC m dl)
endif()

add_executable(gguf2bin2 src/main.c)
target_link_libraries(gguf2bin2 PRIVATE g2bcore)
if(NOT WIN32 AND NOT MSVC)
  target_link_libraries(gguf2bin2 PRIVATE m)
endif()

if(G2B_BUILD_TESTS)
  enable_testing()
  foreach(t kvtest prefilltest q4bcheck iq1check q3kcheck apitest selftest)
    add_executable(${t} tools/${t}.c)
    target_link_libraries(${t} PRIVATE g2bcore)
    if(NOT WIN32 AND NOT MSVC)
      target_link_libraries(${t} PRIVATE m)
    endif()
  endforeach()
  add_test(NAME smoke-synth COMMAND gguf2bin2 synth ${CMAKE_BINARY_DIR}/t.g2bx)
  add_test(NAME smoke-run COMMAND gguf2bin2 run ${CMAKE_BINARY_DIR}/t.g2bx -n 4 -t 0)
  add_test(NAME selftest COMMAND selftest ${CMAKE_BINARY_DIR}/t.g2bx ${CMAKE_BINARY_DIR})
  set_tests_properties(smoke-run selftest PROPERTIES DEPENDS smoke-synth)
endif()
