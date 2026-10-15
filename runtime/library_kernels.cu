// library_kernels.cu
//
// Library 模块（library.md）测试专用的最小内核集。构建系统用 nvcc 将其
// 编译为 cubin（见 runtime/CMakeLists.txt），再通过 embed_cubin.cmake 转成
// 字节数组嵌入测试二进制，供 cudaLibraryLoadData / cuLibraryLoadFromFile 加载。
//
// extern "C" 保证入口符号不被名称修饰，测试才能用
// cudaLibraryGetKernel 按名字 "lib_kernel_set" / "lib_kernel_add" 取到内核。

extern "C" __global__ void lib_kernel_set(int* out, int val) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        *out = val;
    }
}

extern "C" __global__ void lib_kernel_add(int* out, int a, int b) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        *out = a + b;
    }
}
