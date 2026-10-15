cmake_minimum_required(VERSION 3.14)

# ---------------------------------------------------------------------------
# 自动选择满足要求的 CUDA Toolkit（必须在 project() 之前：CUDA 语言的
# 编译器与 find_package(CUDAToolkit) 需要是同一个 Toolkit）。
#
# Library 模块需要 CUDA >= 12.8（Runtime Library API：cudaLibraryLoadData
# / cudaLibraryGetKernel / cudaLaunchKernelEx）。本机默认 /usr/local/cuda
# 是 12.4，不满足；这里在常见安装路径中自动挑选更高版本。用户显式指定
# CUDAToolkit_ROOT / CMAKE_CUDA_COMPILER 时跳过自动选择。
# ---------------------------------------------------------------------------
function(_cuda_kit_version out root)
  set(${out} "" PARENT_SCOPE)
  if(NOT EXISTS "${root}/version.json")
    return()
  endif()
  file(READ "${root}/version.json" _vj)
  string(REGEX MATCH "\"version\"[ \t]*:[ \t]*\"([0-9]+)\\.([0-9]+)" _m "${_vj}")
  if(CMAKE_MATCH_1)
    set(${out} "${CMAKE_MATCH_1}.${CMAKE_MATCH_2}" PARENT_SCOPE)
  endif()
endfunction()

# 已配置（缓存/命令行）的 Toolkit 是否满足要求？满足则直接沿用
set(_cuda_auto_select ON)
if(DEFINED CMAKE_CUDA_COMPILER AND
   NOT CMAKE_CUDA_COMPILER MATCHES "^/usr/local/cuda/bin/nvcc$" AND
   NOT CMAKE_CUDA_COMPILER MATCHES "/cuda-12\\.[0-7]")
  set(_cuda_auto_select OFF)  # 用户显式指定的其它编译器，不干预
endif()
if(_cuda_auto_select AND DEFINED CUDAToolkit_ROOT)
  _cuda_kit_version(_root_ver "${CUDAToolkit_ROOT}")
  if(NOT _root_ver)
    set(_cuda_auto_select OFF)  # 无法识别版本的自定义路径，尊重用户选择
  elseif(NOT _root_ver VERSION_LESS 12.8)
    set(_cuda_auto_select OFF)
  endif()
endif()
if(_cuda_auto_select)
  _cuda_kit_version(_default_ver "/usr/local/cuda")
  if(_default_ver AND NOT _default_ver VERSION_LESS 12.8)
    set(_cuda_auto_select OFF)  # 默认 Toolkit 已满足要求
  endif()
endif()
if(_cuda_auto_select)
  # 优先本机已验证的 13.0，其次 12.8，最后按版本升序找其它可用版本
  set(_candidates "/usr/local/cuda-13.0" "/usr/local/cuda-12.8")
  file(GLOB _others "/usr/local/cuda-*")
  list(SORT _others)
  list(APPEND _candidates ${_others})
  list(REMOVE_DUPLICATES _candidates)
  foreach(_c IN LISTS _candidates)
    if(NOT EXISTS "${_c}/bin/nvcc" OR NOT EXISTS "${_c}/include/cuda_runtime_api.h")
      continue()
    endif()
    _cuda_kit_version(_v "${_c}")
    if(_v AND NOT _v VERSION_LESS 12.8)
      set(CUDAToolkit_ROOT "${_c}" CACHE PATH "CUDA Toolkit root" FORCE)
      set(CMAKE_CUDA_COMPILER "${_c}/bin/nvcc" CACHE FILEPATH "CUDA compiler" FORCE)
      # 清除可能被先前失败配置污染的 find_package 结果缓存
      unset(CUDAToolkit_DIR CACHE)
      unset(CUDAToolkit_INCLUDE_DIR CACHE)
      unset(CUDAToolkit_LIBRARY_DIR CACHE)
      unset(CUDAToolkit_TARGET_DIR CACHE)
      if(_default_ver)
        message(STATUS "默认 CUDA ${_default_ver} 不满足 >= 12.8，自动改用 ${_c}（${_v}）")
      else()
        message(STATUS "自动选择 CUDA Toolkit：${_c}（${_v}）")
      endif()
      break()
    endif()
  endforeach()
endif()

project(cuda-runtime-test LANGUAGES CXX CUDA)

# ---------------------------------------------------------------------------
# googletest：首次配置需要联网拉取，之后缓存在 build/_deps 下
# ---------------------------------------------------------------------------
include(FetchContent)

FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG v1.14.0
)

set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

# CUDA Toolkit：提供 CUDA::cudart / CUDA::cuda_driver 等 imported target。
# Library 模块用到 CUDA >= 12.8 才提供的 Runtime Library API
# （cudaLibraryLoadData / cudaLibraryGetKernel / cudaLaunchKernelEx）。
# 默认版本不足时由文件顶部的逻辑自动选择更高版本；也可显式指定：
#   cmake -S . -B build -DCUDAToolkit_ROOT=/usr/local/cuda-13.0 \
#         -DCMAKE_CUDA_COMPILER=/usr/local/cuda-13.0/bin/nvcc
find_package(CUDAToolkit 12.8 REQUIRED)

enable_testing()

# 各测试模块放在子目录中（runtime API / driver API / IPC / ...）
add_subdirectory(runtime)
