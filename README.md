# cuda-runtime-test
For Runtime or Driver API function tests(stress test)

## Testing

### 环境要求
- CMake >= 3.14
- CUDA Toolkit **>= 12.8**（Library 模块需要 Runtime Library API：`cudaLibraryLoadData` / `cudaLibraryGetKernel` / `cudaLaunchKernelEx`）
- 首次配置需联网（FetchContent 拉取 googletest v1.14.0）

> 本机默认 Toolkit 是 12.4，缺少 Library API。请显式指定已安装的 13.0
> （驱动 580 支持到 CUDA 13.0），如下面的构建命令所示。

### 构建与运行
```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build --output-on-failure
```

> 默认 Toolkit（`/usr/local/cuda`，本机为 12.4）低于 12.8 时，CMake 会
> 自动在本机已安装版本中挑选更高的（优先 13.0，其次 12.8），无需手动
> 指定。需要指定特定版本时再显式传参：
> ```bash
> cmake -S . -B build -DCUDAToolkit_ROOT=/usr/local/cuda-13.0 \
>       -DCMAKE_CUDA_COMPILER=/usr/local/cuda-13.0/bin/nvcc
> ```

也可以直接运行测试二进制：
```bash
./build/runtime/cuda_ipc_test                    # IPC 全部用例
./build/runtime/library_test                     # Library 全部用例（LIB-001~014）
./build/runtime/library_test --gtest_filter='LibraryTest.LIB_008_*'   # 按用例过滤
```

### 测试模块
| 模块 | 目标 | 说明 |
| --- | --- | --- |
| runtime | `cuda_ipc_test` | CUDA Runtime API 跨进程 IPC 用例（内存句柄/事件句柄，12 个用例） |
| runtime | `library_test` | Library 加载/枚举/查询/启动/卸载及生命周期用例（LIB-001~014，见 `runtime/library.md`） |

> 说明：
> - `library_test` 的内核源为 `runtime/library_kernels.cu`，构建时编译成
>   fatbin（按本机 GPU 架构打包多个 cubin）并嵌入测试二进制
>   （`runtime/embed_cubin.cmake`），架构按 `nvidia-smi` 探测的本机 GPU
>   生成，探测失败时回退到 sm_75/sm_89。
> - 跨进程用例的待命子进程必须在本进程任何 CUDA 调用之前 fork，
>   因此请勿使用 `--gtest_repeat`（第二轮 fork 时父进程已初始化 CUDA，
>   子进程无法建立自己的 context）。
