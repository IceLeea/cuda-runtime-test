# embed_cubin.cmake
# 把二进制 cubin 转成 C++ 头文件中的字节数组。
# 用法: cmake -DCUBIN=<cubin路径> -DOUT=<头文件路径> -P embed_cubin.cmake

if(NOT DEFINED CUBIN OR NOT DEFINED OUT)
  message(FATAL_ERROR "embed_cubin.cmake 需要 -DCUBIN 与 -DOUT 参数")
endif()

file(READ "${CUBIN}" _hex HEX)
string(REGEX REPLACE "([0-9a-fA-F][0-9a-fA-F])" "0x\\1," _bytes "${_hex}")

file(WRITE "${OUT}" "// 本文件由构建系统自动生成（embed_cubin.cmake），请勿手动修改\n")
file(APPEND "${OUT}" "#pragma once\n")
file(APPEND "${OUT}" "#include <cstddef>\n\n")
file(APPEND "${OUT}" "inline const unsigned char g_libraryCubinData[] = {${_bytes}};\n")
file(APPEND "${OUT}" "inline const std::size_t g_libraryCubinSize = sizeof(g_libraryCubinData);\n")
