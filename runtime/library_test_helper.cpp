// library_test_helper.cpp
//
// 供 library_test 在【独立进程】中复现会破坏本进程状态的驱动场景
// （如重复卸载导致段错误）。由测试用例 fork+exec 调用：
//
//   library_test_helper double-unload
//
// 退出码约定：
//   0   = 场景完整执行且未崩溃（说明驱动已符合契约）
//   1   = 场景中接口按契约返回了错误（未崩溃）
//   127 = 参数错误 / 初始化失败
//
// 注意：必须独立进程运行——在已初始化 CUDA 的进程中 fork 出的子进程
// 无法复现该崩溃（见 LIB_011 注释），且把崩溃隔离在子进程外。

#include <cuda.h>

#include <cstdio>
#include <cstring>

#include "library_kernels_cubin.hpp"  // 构建时由 embed_cubin.cmake 生成

int main(int argc, char** argv) {
    if (argc < 2 || std::strcmp(argv[1], "double-unload") != 0) {
        std::fprintf(stderr, "usage: %s double-unload\n", argv[0]);
        return 127;
    }

    if (cuInit(0) != CUDA_SUCCESS) return 127;

    CUlibrary lib = nullptr;
    CUresult e = cuLibraryLoadData(&lib, g_libraryCubinData,
                                   nullptr, nullptr, 0, nullptr, nullptr, 0);
    if (e != CUDA_SUCCESS || lib == nullptr) return 127;

    e = cuLibraryUnload(lib);
    if (e != CUDA_SUCCESS) return 127;

    // 第二次卸载：契约应返回错误；本机驱动实测在此处段错误
    CUresult e2 = cuLibraryUnload(lib);
    std::fprintf(stderr, "second unload returned %d (no crash)\n", (int)e2);
    return e2 == CUDA_SUCCESS ? 0 : 1;
}
