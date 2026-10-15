#pragma once

// Library 模块测试（LIB-001 ~ LIB-014）的公共工具：
//   - 内嵌 cubin 数据（由构建系统从 library_kernels.cu 生成）
//   - 驱动上下文初始化
//   - 临时文件写入
// 供 library_test.cpp 使用。

#include <cuda.h>
#include <cuda_runtime.h>

#include <sys/types.h>

#include <cstddef>
#include <cstdlib>
#include <string>
#include <unistd.h>

#include "library_kernels_cubin.hpp"  // 构建时由 embed_cubin.cmake 生成

namespace library_test {

// 内嵌 cubin（含 lib_kernel_set / lib_kernel_add 两个内核）
inline const unsigned char* CubinData() { return g_libraryCubinData; }
inline std::size_t CubinSize() { return g_libraryCubinSize; }

// 测试用内核名（与 library_kernels.cu 中 extern "C" 的符号名一致）
inline constexpr const char* kKernelSetName = "lib_kernel_set";
inline constexpr const char* kKernelAddName = "lib_kernel_add";

// cubin 中定义的内核数量
inline constexpr int kExpectedKernelCount = 2;

// 保留设备 0 的主上下文并设为当前（调用前须已 cuInit 成功）。
// 运行时 API（cudaFree(nullptr)）随后复用同一个主上下文。
inline bool EnsurePrimaryContext() {
    CUdevice dev = 0;
    if (cuDeviceGet(&dev, 0) != CUDA_SUCCESS) return false;
    CUcontext ctx = nullptr;
    if (cuDevicePrimaryCtxRetain(&ctx, dev) != CUDA_SUCCESS) return false;
    return cuCtxSetCurrent(ctx) == CUDA_SUCCESS;
}

// RAII 临时文件：构造时用 mkstemp 创建并写入数据，析构时自动删除，
// 保证任何 ASSERT 提前退出都不会在 /tmp 遗留文件。
class TempFileGuard {
public:
    TempFileGuard() = default;
    ~TempFileGuard() {
        if (!path_.empty()) {
            ::unlink(path_.c_str());
            path_.clear();
        }
    }
    TempFileGuard(const TempFileGuard&) = delete;
    TempFileGuard& operator=(const TempFileGuard&) = delete;

    // 创建临时文件并写入 size 字节；失败返回 false（此时不持有文件）
    bool Write(const void* data, std::size_t size) {
        char tmpl[] = "/tmp/cuda_library_test_XXXXXX";
        int fd = ::mkstemp(tmpl);
        if (fd < 0) return false;
        path_ = tmpl;

        const char* p = static_cast<const char*>(data);
        std::size_t written = 0;
        while (written < size) {
            ssize_t n = ::write(fd, p + written, size - written);
            if (n <= 0) {
                ::close(fd);
                ::unlink(path_.c_str());
                path_.clear();
                return false;
            }
            written += static_cast<std::size_t>(n);
        }
        ::close(fd);
        return true;
    }

    const std::string& path() const { return path_; }

private:
    std::string path_;
};

}  // namespace library_test
