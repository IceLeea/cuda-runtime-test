# CUDA Runtime API 测试模块

# ---------------------------------------------------------------------------
# IPC 测试
# ---------------------------------------------------------------------------
add_executable(cuda_ipc_test cuda_ipc_test.cpp)
target_link_libraries(cuda_ipc_test PRIVATE gtest gmock gtest_main CUDA::cudart)
add_test(NAME CudaIpcTest COMMAND cuda_ipc_test)

# ---------------------------------------------------------------------------
# Library 测试（LIB-001 ~ LIB-014）
#
# 流程：library_kernels.cu --nvcc--> library_kernels.fatbin（多架构 cubin 容器）
#                         --embed_cubin.cmake--> library_kernels_cubin.hpp
# library_test.cpp 把字节数组喂给 cudaLibraryLoadData / cuLibraryLoadFromFile。
# 注意：nvcc 的 -cubin 只允许单一 gencode，多 GPU 架构必须用 -fatbin 打包。
# ---------------------------------------------------------------------------

# 探测本机 GPU 计算能力，为其生成 gencode 参数；
# 探测不到（无 GPU / 无 nvidia-smi）时回退到常见架构 75;89
set(CUDA_TEST_ARCHS "75;89")
find_program(NVIDIA_SMI_EXECUTABLE nvidia-smi)
if(NVIDIA_SMI_EXECUTABLE)
  execute_process(
    COMMAND ${NVIDIA_SMI_EXECUTABLE} --query-gpu=compute_cap --format=csv,noheader
    OUTPUT_VARIABLE _nvsmi_cc
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET)
  if(_nvsmi_cc)
    set(CUDA_TEST_ARCHS "")
    string(REPLACE "\n" ";" _nvsmi_cc_list "${_nvsmi_cc}")
    foreach(_cc IN LISTS _nvsmi_cc_list)
      string(STRIP "${_cc}" _cc)
      string(REPLACE "." "" _sm "${_cc}")
      if(_sm MATCHES "^[0-9]+$")
        list(APPEND CUDA_TEST_ARCHS ${_sm})
      endif()
    endforeach()
    list(REMOVE_DUPLICATES CUDA_TEST_ARCHS)
  endif()
endif()
if(NOT CUDA_TEST_ARCHS)
  set(CUDA_TEST_ARCHS "75;89")
endif()

set(LIBRARY_KERNEL_FATBIN ${CMAKE_CURRENT_BINARY_DIR}/library_kernels.fatbin)
set(LIBRARY_KERNEL_CUBIN_HEADER ${CMAKE_CURRENT_BINARY_DIR}/library_kernels_cubin.hpp)

set(_gencode_args)
foreach(_arch IN LISTS CUDA_TEST_ARCHS)
  list(APPEND _gencode_args -gencode)
  list(APPEND _gencode_args "arch=compute_${_arch},code=sm_${_arch}")
endforeach()

add_custom_command(
  OUTPUT ${LIBRARY_KERNEL_FATBIN}
  COMMAND ${CMAKE_CUDA_COMPILER} ${_gencode_args} -fatbin
          -o ${LIBRARY_KERNEL_FATBIN}
          ${CMAKE_CURRENT_SOURCE_DIR}/library_kernels.cu
  DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/library_kernels.cu
  COMMENT "Compiling library test kernels to fatbin (${CUDA_TEST_ARCHS})"
  VERBATIM)

add_custom_command(
  OUTPUT ${LIBRARY_KERNEL_CUBIN_HEADER}
  COMMAND ${CMAKE_COMMAND}
          -DCUBIN=${LIBRARY_KERNEL_FATBIN}
          -DOUT=${LIBRARY_KERNEL_CUBIN_HEADER}
          -P ${CMAKE_CURRENT_SOURCE_DIR}/embed_cubin.cmake
  DEPENDS ${LIBRARY_KERNEL_FATBIN} ${CMAKE_CURRENT_SOURCE_DIR}/embed_cubin.cmake
  COMMENT "Embedding fatbin into ${LIBRARY_KERNEL_CUBIN_HEADER}"
  VERBATIM)

add_executable(library_test
  library_test.cpp
  ${LIBRARY_KERNEL_CUBIN_HEADER})
target_compile_features(library_test PRIVATE cxx_std_17)
target_include_directories(library_test PRIVATE ${CMAKE_CURRENT_BINARY_DIR})
target_link_libraries(library_test PRIVATE gtest gtest_main CUDA::cudart CUDA::cuda_driver)
add_test(NAME LibraryTest COMMAND library_test)

# 独立辅助进程：在干净进程中复现会崩溃的驱动场景（如重复卸载），
# 由 library_test fork+exec 调用，路径通过编译定义传入
add_executable(library_test_helper
  library_test_helper.cpp
  ${LIBRARY_KERNEL_CUBIN_HEADER})
target_compile_features(library_test_helper PRIVATE cxx_std_17)
target_include_directories(library_test_helper PRIVATE ${CMAKE_CURRENT_BINARY_DIR})
target_link_libraries(library_test_helper PRIVATE CUDA::cuda_driver)
target_compile_definitions(library_test PRIVATE
  LIBRARY_TEST_HELPER="$<TARGET_FILE:library_test_helper>")
