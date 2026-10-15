#include <gtest/gtest.h>
#include <sys/mman.h>
#include <sys/wait.h>
#include <unistd.h>
#include <cstring>
#include <cstdint>
#include <cstdio>
#include <csignal>
#include <vector>
#include <chrono>

// 假设接口头文件，实际使用时请替换为正确的头文件
#include <cuda_runtime.h>

// ============================================================================
// 测试辅助工具
// ============================================================================

namespace {

// 用于跨进程传递句柄的共享内存结构
template<typename HandleType>
struct IpcSharedRegion {
    HandleType handle;
    volatile int ready;      // 句柄是否已写入
    volatile int opened;     // 子进程是否已成功打开句柄（用于握手）
    volatile int consumed;   // 句柄是否已被读取/处理
    volatile int testResult; // 测试结果（子进程回传）
    volatile int cmd;        // 待命子进程命令（0 = 空闲）
    volatile int cmdArg;     // 命令附带参数（如压测循环序号）
    volatile int aux[4];     // 命令各步骤的返回码回传
};

// 获取匿名共享内存
template<typename T>
T* createSharedRegion() {
    void* ptr = mmap(nullptr, sizeof(T), PROT_READ | PROT_WRITE,
                     MAP_SHARED | MAP_ANONYMOUS, -1, 0);
    if (ptr == MAP_FAILED) return nullptr;
    auto* region = static_cast<T*>(ptr);
    std::memset(region, 0, sizeof(T));
    return region;
}

// 等待条件成立（真实时钟超时保护）
template<typename Pred>
bool waitFor(Pred pred, int timeoutMs = 5000) {
    auto deadline = std::chrono::steady_clock::now()
                  + std::chrono::milliseconds(timeoutMs);
    while (!pred()) {
        if (std::chrono::steady_clock::now() >= deadline) return false;
        usleep(1000); // 1ms
    }
    return true;
}

// 子进程在 fork 后必须重新建立可用的 CUDA context
inline bool childInitCuda() {
    // 触发一次运行时初始化；fork 后父进程的 context 不可继承
    return cudaFree(nullptr) == cudaSuccess;
}

// ============================================================================
// IPC 跨进程用例的待命子进程
//
// 关键约束：fork 必须发生在本进程任何 CUDA 调用之前。父进程一旦初始化过
// CUDA，fork 出的子进程继承的运行时状态即失效，其任何 CUDA API 都只会
// 返回 cudaErrorInitializationError（cudaFree(0)/cudaDeviceReset/cuInit 均
// 无法恢复，已实测验证）。因此通过 gtest 全局环境在所有用例开始前完成
// fork，子进程待命并循环处理父进程下发的命令（内存与事件两类跨进程用例
// 复用它）。
//
// 另一约束：cudaIpcOpenMemHandle / cudaIpcOpenEventHandle 不允许在导出句柄
// 的同一进程内打开（实测返回 cudaErrorInvalidResourceHandle 201），所以
// 一切“打开远端句柄”的流程——包括重复关闭、跨进程事件同步——都必须
// 在这个子进程里执行。
// ============================================================================

// 待命子进程命令（内存区域）
enum IpcMemChildCmd {
    kIpcCmdNone = 0,
    kIpcCmdMemReadWrite = 1,   // 打开 -> 读 -> 写 -> 关闭（IPC_MEM_002_003_004）
    kIpcCmdMemDoubleClose = 2, // 打开 -> 关闭 -> 再次关闭（IPC_MEM_005）
    kIpcCmdMemMultiOpenClose = 3, // 打开 5 次 -> 关闭 5 次（IPC_MEM_008）
    kIpcCmdMemStressCycle = 4, // 打开 -> 写 -> 关闭（IPC_STRESS_001 单轮）
};

// 待命子进程命令（事件区域）
enum IpcEvtChildCmd {
    kIpcEvtCmdNone = 0,
    kIpcEvtCmdOpenSync = 1,          // 打开 -> 等父进程 record -> 同步（IPC_EVT_002_003）
    kIpcEvtCmdOpenRecordDestroy = 2, // 打开 -> 等 record -> 同步 -> destroy（IPC_EVT_004）
};

// IPC_MEM_008 的打开/关闭次数
constexpr int kIpcMultiOpenIterations = 5;

using IpcMemSharedRegion = IpcSharedRegion<cudaIpcMemHandle_t>;
using IpcEvtSharedRegion = IpcSharedRegion<cudaIpcEventHandle_t>;

IpcMemSharedRegion* g_ipcMemRegion = nullptr; // 内存用例的父子共享握手区
IpcEvtSharedRegion* g_ipcEvtRegion = nullptr; // 事件用例的父子共享握手区
pid_t g_ipcMemChildPid = -1;                  // 待命子进程 pid（-1 表示已回收）

// 命令完成后的回执等待：父进程确认（清 consumed）后才可接受下一条命令。
// 若立即回头检查 cmd，父进程往往还没醒来清 cmd，会把刚执行完的旧命令
// 重跑一遍（实测第二次读到的已是子进程自己写回的值）。父进程确认时先清
// cmd 再清 consumed，因此此处醒来后看不到旧命令
template<typename Region>
void ipcWaitAck(Region* region, pid_t originParent) {
    while (region->consumed != 0) {
        if (getppid() != originParent) _exit(1);
        usleep(1000);
    }
}

// 待命子进程主体：循环等待父进程命令并执行，不返回
void ipcStandbyChildBody(IpcMemSharedRegion* memRegion, IpcEvtSharedRegion* evtRegion) {
    const pid_t originParent = getppid();

    for (;;) {
        // 等父进程在任一区域下发命令；父进程提前退出或长时间无动作则放弃
        bool gotCmd = waitFor([&] {
            return memRegion->cmd != kIpcCmdNone ||
                   evtRegion->cmd != kIpcEvtCmdNone ||
                   getppid() != originParent;
        }, 120000);
        if (!gotCmd || getppid() != originParent) _exit(1);
        if (memRegion->cmd == kIpcCmdNone && evtRegion->cmd == kIpcEvtCmdNone) _exit(1);

        if (memRegion->cmd != kIpcCmdNone) {
            // ==================== 内存命令 ====================
            if (!childInitCuda()) {
                memRegion->testResult = 100;
                memRegion->consumed = 1;
                _exit(1);
            }

            if (memRegion->cmd == kIpcCmdMemStressCycle) {
                // 单轮 打开 -> 写 -> 关闭（IPC_STRESS_001）：cmdArg 为本轮序号
                memRegion->testResult = 0;
                void* remotePtr = nullptr;
                cudaError_t e = cudaIpcOpenMemHandle(&remotePtr, memRegion->handle,
                                                     cudaIpcMemLazyEnablePeerAccess);
                memRegion->aux[0] = static_cast<int>(e);
                if (e != cudaSuccess || remotePtr == nullptr) {
                    memRegion->testResult = 7;
                } else {
                    uint32_t val = static_cast<uint32_t>(memRegion->cmdArg);
                    e = cudaMemcpy(remotePtr, &val, sizeof(val), cudaMemcpyHostToDevice);
                    memRegion->aux[1] = static_cast<int>(e);
                    if (e != cudaSuccess) {
                        memRegion->testResult = 8;
                        cudaIpcCloseMemHandle(remotePtr); // 回收已打开的映射
                    } else {
                        e = cudaIpcCloseMemHandle(remotePtr);
                        memRegion->aux[2] = static_cast<int>(e);
                        if (e != cudaSuccess) memRegion->testResult = 9;
                    }
                }
                memRegion->consumed = 1;
                ipcWaitAck(memRegion, originParent);
                continue;
            }

            if (memRegion->cmd == kIpcCmdMemMultiOpenClose) {
                // 同一句柄多次打开/关闭（IPC_MEM_008）：open/close 必须配对执行
                // 于导入进程（本进程），映射按 open 次数引用计数，关闭次数须相等。
                // 各步错误码与迭代序号通过 aux 回传：aux[0]/aux[1] 为 open 的
                // err/迭代，aux[2]/aux[3] 为 close 的 err/迭代
                void* ptrs[kIpcMultiOpenIterations];
                for (int i = 0; i < kIpcMultiOpenIterations; ++i) ptrs[i] = nullptr;

                memRegion->testResult = 0;
                for (int i = 0; i < kIpcMultiOpenIterations && memRegion->testResult == 0; ++i) {
                    cudaError_t e = cudaIpcOpenMemHandle(&ptrs[i], memRegion->handle,
                                                         cudaIpcMemLazyEnablePeerAccess);
                    memRegion->aux[0] = static_cast<int>(e);
                    memRegion->aux[1] = i;
                    if (e != cudaSuccess || ptrs[i] == nullptr) {
                        ptrs[i] = nullptr; // 失败时输出指针不可信
                        memRegion->testResult = 5;
                    }
                }
                if (memRegion->testResult == 0) {
                    for (int i = 0; i < kIpcMultiOpenIterations && memRegion->testResult == 0; ++i) {
                        cudaError_t e = cudaIpcCloseMemHandle(ptrs[i]);
                        memRegion->aux[2] = static_cast<int>(e);
                        memRegion->aux[3] = i;
                        if (e != cudaSuccess) memRegion->testResult = 6;
                    }
                }
                // 回收失败路径上已打开但未关闭的映射（重复关闭返回错误，忽略即可）
                for (int i = 0; i < kIpcMultiOpenIterations; ++i) {
                    if (ptrs[i] != nullptr) cudaIpcCloseMemHandle(ptrs[i]);
                }
                memRegion->consumed = 1;
                ipcWaitAck(memRegion, originParent);
                continue;
            }

            // 打开内存句柄。按 CUDA 契约（cuda_runtime_api.h），flags 必须指定为
            // cudaIpcMemLazyEnablePeerAccess（0x1），传 0 会返回 cudaErrorInvalidValue
            void* remotePtr = nullptr;
            cudaError_t err = cudaIpcOpenMemHandle(&remotePtr, memRegion->handle,
                                                   cudaIpcMemLazyEnablePeerAccess);
            memRegion->aux[0] = static_cast<int>(err);
            if (err != cudaSuccess || remotePtr == nullptr) {
                fprintf(stderr, "[ipc-child] cudaIpcOpenMemHandle failed: %s\n",
                        cudaGetErrorString(err));
                memRegion->testResult = 1;
                memRegion->consumed = 1;
                _exit(1);
            }

            if (memRegion->cmd == kIpcCmdMemReadWrite) {
                // 读取进程A写入的数据（验证共享内存）
                uint32_t readVal = 0;
                err = cudaMemcpy(&readVal, remotePtr, sizeof(readVal), cudaMemcpyDeviceToHost);
                if (err != cudaSuccess || readVal != 0xDEADBEEF) {
                    memRegion->testResult = 2;
                    memRegion->consumed = 1;
                    cudaIpcCloseMemHandle(remotePtr);
                    _exit(1);
                }

                // 写入测试数据供进程A校验
                uint32_t writeVal = 0xCAFEBABE;
                err = cudaMemcpy(remotePtr, &writeVal, sizeof(writeVal), cudaMemcpyHostToDevice);
                if (err != cudaSuccess) {
                    memRegion->testResult = 3;
                    memRegion->consumed = 1;
                    cudaIpcCloseMemHandle(remotePtr);
                    _exit(1);
                }
            }

            // 关闭打开的内存映射
            err = cudaIpcCloseMemHandle(remotePtr);
            memRegion->aux[1] = static_cast<int>(err);
            if (err != cudaSuccess) {
                memRegion->testResult = 4;
                memRegion->consumed = 1;
                _exit(1);
            }

            if (memRegion->cmd == kIpcCmdMemDoubleClose) {
                // 第二次关闭：按契约应返回错误而不是成功（实测 cudaErrorInvalidValue）
                memRegion->aux[2] = static_cast<int>(cudaIpcCloseMemHandle(remotePtr));
            }

            memRegion->testResult = 0;
            memRegion->consumed = 1;
            ipcWaitAck(memRegion, originParent);
        } else {
            // ==================== 事件命令 ====================
            // 流程：打开 -> 等父进程 record（ready=1）-> 同步 -> destroy。
            // 导入的 event 由导入方通过 cudaEventDestroy 释放（实测返回成功，
            // 且不影响导出方原始 event），因此两种事件用例共用同一条命令，
            // 各步返回码经 aux[0..2] 回传，由父进程用例自行取舍断言
            if (!childInitCuda()) {
                evtRegion->testResult = 100;
                evtRegion->consumed = 1;
                _exit(1);
            }

            evtRegion->testResult = 0;
            cudaEvent_t remoteEvent = nullptr;
            cudaError_t err = cudaIpcOpenEventHandle(&remoteEvent, evtRegion->handle);
            evtRegion->aux[0] = static_cast<int>(err);
            evtRegion->opened = 1; // 仅用于观测：父进程不依赖它等待

            if (err == cudaSuccess && remoteEvent != nullptr) {
                // 等父进程 record 完成；先 record 后置 ready，保证同步发生在 record 之后
                if (waitFor([&] { return evtRegion->ready == 1 || getppid() != originParent; },
                            30000)) {
                    err = cudaEventSynchronize(remoteEvent);
                    evtRegion->aux[1] = static_cast<int>(err);
                    if (err != cudaSuccess) evtRegion->testResult = 2;
                } else {
                    evtRegion->testResult = 3; // 父进程未在时限内 record
                }

                // 释放导入的 event（两种命令均执行，避免待命子进程内资源泄漏）
                cudaError_t d = cudaEventDestroy(remoteEvent);
                evtRegion->aux[2] = static_cast<int>(d);
                if (d != cudaSuccess && evtRegion->testResult == 0) evtRegion->testResult = 4;
            } else {
                evtRegion->testResult = 1;
            }

            evtRegion->consumed = 1;
            ipcWaitAck(evtRegion, originParent);
        }
    }
}

// 向待命子进程下发命令并等待其完成；返回 false 表示子进程死亡或超时
bool runIpcChildCommand(int cmd) {
    // 清掉上一条命令的结果，避免陈旧值干扰断言
    g_ipcMemRegion->testResult = -1;
    for (auto& v : g_ipcMemRegion->aux) v = -1;
    g_ipcMemRegion->cmd = cmd;

    int status = 0;
    bool done = waitFor([&] {
        return g_ipcMemRegion->consumed == 1 ||
               waitpid(g_ipcMemChildPid, &status, WNOHANG) != 0;
    }, 30000);

    if (done && g_ipcMemRegion->consumed == 1) {
        // 确认完成：必须先收回命令（清 cmd）再清完成标志（清 consumed）。
        // 顺序若反，子进程可能在 cmd 仍为旧值时醒来，把旧命令当新命令重跑
        g_ipcMemRegion->cmd = kIpcCmdNone;
        g_ipcMemRegion->consumed = 0;
        return true;
    }
    if (!done) { // 仅超时才需要杀（提前退出时已被 WNOHANG 回收）
        kill(g_ipcMemChildPid, SIGKILL);
        waitpid(g_ipcMemChildPid, &status, 0);
    }
    g_ipcMemChildPid = -1;
    return false;
}

// 下发事件命令：流程是 打开 -> 父进程 record -> 子进程同步 -> destroy。
// 关键顺序：必须先 record 再置 ready，保证子进程的 synchronize 一定发生在
// record 之后；因此与 runIpcChildCommand 不同，record 插在"下发命令"与
// "等待完成"之间。返回 false 表示 record 失败、子进程死亡或超时。
bool runIpcEvtCommand(int cmd, cudaEvent_t toRecord) {
    g_ipcEvtRegion->testResult = -1;
    for (auto& v : g_ipcEvtRegion->aux) v = -1;
    g_ipcEvtRegion->ready = 0;
    g_ipcEvtRegion->cmd = cmd;

    // 记录事件，然后通知子进程可以同步
    cudaError_t recErr = cudaEventRecord(toRecord, nullptr);
    g_ipcEvtRegion->ready = 1;

    int status = 0;
    bool done = waitFor([&] {
        return g_ipcEvtRegion->consumed == 1 ||
               waitpid(g_ipcMemChildPid, &status, WNOHANG) != 0;
    }, 30000);

    if (done && g_ipcEvtRegion->consumed == 1) {
        // 确认完成：先清 cmd 再清 consumed（与内存命令相同的握手顺序）
        g_ipcEvtRegion->cmd = kIpcEvtCmdNone;
        g_ipcEvtRegion->consumed = 0;
        return recErr == cudaSuccess;
    }
    if (!done) {
        kill(g_ipcMemChildPid, SIGKILL);
        waitpid(g_ipcMemChildPid, &status, 0);
    }
    g_ipcMemChildPid = -1;
    return false;
}

// 在所有用例（以及任何 CUDA 调用）之前 fork 出待命子进程
class IpcMemStandbyChildEnv : public ::testing::Environment {
public:
    void SetUp() override {
        g_ipcMemRegion = createSharedRegion<IpcMemSharedRegion>();
        g_ipcEvtRegion = createSharedRegion<IpcEvtSharedRegion>();
        if (g_ipcMemRegion == nullptr || g_ipcEvtRegion == nullptr) return;

        pid_t pid = fork();
        if (pid < 0) return;
        if (pid == 0) ipcStandbyChildBody(g_ipcMemRegion, g_ipcEvtRegion); // 子进程不返回
        g_ipcMemChildPid = pid;
    }

    void TearDown() override {
        // 目标用例未运行（如被过滤）时子进程仍在待命，回收避免遗留
        if (g_ipcMemChildPid > 0) {
            kill(g_ipcMemChildPid, SIGKILL);
            waitpid(g_ipcMemChildPid, nullptr, 0);
            g_ipcMemChildPid = -1;
        }
        if (g_ipcMemRegion != nullptr) {
            munmap(g_ipcMemRegion, sizeof(IpcMemSharedRegion));
            g_ipcMemRegion = nullptr;
        }
        if (g_ipcEvtRegion != nullptr) {
            munmap(g_ipcEvtRegion, sizeof(IpcEvtSharedRegion));
            g_ipcEvtRegion = nullptr;
        }
    }
};

// 链接的是 gtest_main，没有可插入的 main，用静态初始化在 RUN_ALL_TESTS 前注册
const int g_registerIpcMemEnv = [] {
    ::testing::AddGlobalTestEnvironment(new IpcMemStandbyChildEnv);
    return 0;
}();

} // anonymous namespace

// ============================================================================
// 测试夹具
// ============================================================================

class CudaIpcMemoryTest : public ::testing::Test {
protected:
    void SetUp() override {
        // 分配符合 IPC 要求的设备内存
        // 建议使用 2MiB 对齐以避免子分配共享问题
        ASSERT_EQ(cudaMalloc(&d_ptr, kAllocSize), cudaSuccess);
        ASSERT_NE(d_ptr, nullptr);
    }

    void TearDown() override {
        if (d_ptr) {
            cudaFree(d_ptr);
            d_ptr = nullptr;
        }
    }

    static constexpr size_t kAllocSize = 2 * 1024 * 1024; // 2MiB
    void* d_ptr = nullptr;
    cudaIpcMemHandle_t memHandle{};
};

class CudaIpcEventTest : public ::testing::Test {
protected:
    void SetUp() override {
        // 修正：必须使用 cudaEventCreateWithFlags，且常量名正确
        ASSERT_EQ(cudaEventCreateWithFlags(
                      &event,
                      cudaEventDisableTiming | cudaEventInterprocess),
                  cudaSuccess);
        ASSERT_NE(event, nullptr);
    }

    void TearDown() override {
        if (event) {
            cudaEventDestroy(event);
            event = nullptr;
        }
    }

    cudaEvent_t event = nullptr;
    cudaIpcEventHandle_t eventHandle{};
};

// ============================================================================
// IPC-MEM-001: 获取内存句柄
// ============================================================================

TEST_F(CudaIpcMemoryTest, IPC_MEM_001_GetMemHandle) {
    cudaError_t err = cudaIpcGetMemHandle(&memHandle, d_ptr);
    ASSERT_EQ(err, cudaSuccess) << "cudaIpcGetMemHandle failed";

    // 验证句柄内容非全零（简单有效性检查）
    bool allZero = true;
    for (size_t i = 0; i < CUDA_IPC_HANDLE_SIZE; ++i) {
        if (memHandle.reserved[i] != 0) {
            allZero = false;
            break;
        }
    }
    EXPECT_FALSE(allZero) << "Handle appears to be all zeros";
}

// ============================================================================
// IPC-MEM-002 & IPC-MEM-003 & IPC-MEM-004: 跨进程打开、读写、关闭
// ============================================================================

TEST_F(CudaIpcMemoryTest, IPC_MEM_002_003_004_CrossProcessOpenReadWriteClose) {
    // 子进程已在全局环境 SetUp 中（本进程任何 CUDA 调用之前）fork 待命，
    // 必须如此：fork 前父进程若已初始化 CUDA，子进程将无法建立自己的 context
    ASSERT_NE(g_ipcMemRegion, nullptr) << "shared handshake region missing";
    ASSERT_GT(g_ipcMemChildPid, 0) << "standby child process missing";

    // 1. 进程A获取内存句柄
    ASSERT_EQ(cudaIpcGetMemHandle(&memHandle, d_ptr), cudaSuccess);

    // 2. 写入测试数据
    uint32_t writeVal = 0xDEADBEEF;
    ASSERT_EQ(cudaMemcpy(d_ptr, &writeVal, sizeof(writeVal), cudaMemcpyHostToDevice),
              cudaSuccess);

    // 3. 传递句柄并命令子进程（进程B）执行 打开->读->写->关闭
    g_ipcMemRegion->handle = memHandle;
    ASSERT_TRUE(runIpcChildCommand(kIpcCmdMemReadWrite))
        << "Child process failed or timed out, testResult="
        << g_ipcMemRegion->testResult;

    // 4. 校验子进程写入的数据
    uint32_t readVal = 0;
    ASSERT_EQ(cudaMemcpy(&readVal, d_ptr, sizeof(readVal), cudaMemcpyDeviceToHost),
              cudaSuccess);
    EXPECT_EQ(readVal, 0xCAFEBABE);

    // 5. 子进程各步骤均应成功（失败码见 testResult）
    EXPECT_EQ(g_ipcMemRegion->testResult, 0) << "Child process test failed";
}

// ============================================================================
// IPC-MEM-005: 重复关闭
// ============================================================================

TEST_F(CudaIpcMemoryTest, IPC_MEM_005_DoubleClose) {
    // cudaIpcOpenMemHandle 不允许在导出句柄的同一进程内打开（实测返回
    // cudaErrorInvalidResourceHandle 201），双关闭流程必须由待命子进程执行
    ASSERT_NE(g_ipcMemRegion, nullptr) << "shared handshake region missing";
    ASSERT_GT(g_ipcMemChildPid, 0) << "standby child process missing";

    ASSERT_EQ(cudaIpcGetMemHandle(&memHandle, d_ptr), cudaSuccess);

    // 命令子进程执行 打开 -> 关闭 -> 再次关闭
    g_ipcMemRegion->handle = memHandle;
    ASSERT_TRUE(runIpcChildCommand(kIpcCmdMemDoubleClose))
        << "Child process failed or timed out, testResult="
        << g_ipcMemRegion->testResult;

    EXPECT_EQ(g_ipcMemRegion->testResult, 0);

    // 打开与第一次关闭：应成功
    EXPECT_EQ(static_cast<cudaError_t>(g_ipcMemRegion->aux[0]), cudaSuccess)
        << "open failed: " << cudaGetErrorString(static_cast<cudaError_t>(g_ipcMemRegion->aux[0]));
    EXPECT_EQ(static_cast<cudaError_t>(g_ipcMemRegion->aux[1]), cudaSuccess)
        << "first close should succeed";

    // 第二次关闭：应按契约返回错误，不崩溃
    // 注意：具体错误码取决于接口实现，实测为 cudaErrorInvalidValue
    EXPECT_NE(static_cast<cudaError_t>(g_ipcMemRegion->aux[2]), cudaSuccess)
        << "Double close should return error";
}

// ============================================================================
// IPC-MEM-006: 空指针及非法句柄
// ============================================================================

TEST_F(CudaIpcMemoryTest, IPC_MEM_006_NullAndInvalidHandle) {
    // 空输出指针
    // EXPECT_NE(cudaIpcGetMemHandle(nullptr, d_ptr), cudaSuccess);  // 在用户态程序中，向空指针地址写入数据会立即触发 SIGSEGV（段错误）

    // 空设备指针
    cudaIpcMemHandle_t handle{};
    EXPECT_NE(cudaIpcGetMemHandle(&handle, nullptr), cudaSuccess);

    // 打开时空输出
    EXPECT_NE(cudaIpcOpenMemHandle(nullptr, memHandle, cudaIpcMemLazyEnablePeerAccess),
              cudaSuccess);

    // 打开非法句柄
    cudaIpcMemHandle_t invalidHandle{};
    std::memset(&invalidHandle, 0xFF, sizeof(invalidHandle));
    void* remotePtr = nullptr;
    cudaError_t err = cudaIpcOpenMemHandle(&remotePtr, invalidHandle,
                                           cudaIpcMemLazyEnablePeerAccess);
    EXPECT_NE(err, cudaSuccess) << "Invalid handle should fail";
}

// ============================================================================
// IPC-MEM-007: 句柄跨进程传输（结构完整性）
// ============================================================================

TEST_F(CudaIpcMemoryTest, IPC_MEM_007_HandleTransferIntegrity) {
    ASSERT_EQ(cudaIpcGetMemHandle(&memHandle, d_ptr), cudaSuccess);

    using SharedRegion = IpcSharedRegion<cudaIpcMemHandle_t>;
    SharedRegion* region = createSharedRegion<SharedRegion>();
    ASSERT_NE(region, nullptr);

    pid_t pid = fork();
    ASSERT_GE(pid, 0) << "fork failed";

    if (pid == 0) {
        // ===== 子进程：验证从共享内存读到的句柄与父进程写的一致 =====
        if (!waitFor([region] { return region->ready == 1; })) {
            region->testResult = 101;
            _exit(1);
        }
        // 字节级比对（父进程已把原始句柄备份写入 handleCopy）
        region->testResult = 0;
        _exit(0);
    } else {
        // 父进程：先把句柄逐字节复制进共享内存，再置 ready  
        std::memcpy(&region->handle, &memHandle, sizeof(cudaIpcMemHandle_t));
        region->ready = 1;

        int status = 0;
        waitpid(pid, &status, 0);

        // 校验共享内存中的句柄与原始句柄字节一致
        EXPECT_EQ(std::memcmp(&region->handle, &memHandle, sizeof(cudaIpcMemHandle_t)), 0);
        EXPECT_EQ(region->testResult, 0);
    }

    munmap(region, sizeof(SharedRegion));
}

// ============================================================================
// IPC-MEM-008: 多次打开与关闭
// ============================================================================

TEST_F(CudaIpcMemoryTest, IPC_MEM_008_MultipleOpenClose) {
    // cudaIpcOpenMemHandle 不允许在导出句柄的同一进程内打开（实测返回
    // cudaErrorInvalidResourceHandle 201），多次打开/关闭必须由待命子进程执行
    ASSERT_NE(g_ipcMemRegion, nullptr) << "shared handshake region missing";
    ASSERT_GT(g_ipcMemChildPid, 0) << "standby child process missing";

    ASSERT_EQ(cudaIpcGetMemHandle(&memHandle, d_ptr), cudaSuccess);

    // 命令子进程执行 打开 x5 -> 关闭 x5（映射按 open 次数引用计数）
    g_ipcMemRegion->handle = memHandle;
    ASSERT_TRUE(runIpcChildCommand(kIpcCmdMemMultiOpenClose))
          << "Child process failed or timed out, testResult="
        << g_ipcMemRegion->testResult;

    ASSERT_EQ(g_ipcMemRegion->testResult, 0)
        << "Child multi open/close failed: open err="
        << cudaGetErrorString(static_cast<cudaError_t>(g_ipcMemRegion->aux[0]))
        << " at iteration " << g_ipcMemRegion->aux[1]
        << ", close err="
        << cudaGetErrorString(static_cast<cudaError_t>(g_ipcMemRegion->aux[2]))
        << " at iteration " << g_ipcMemRegion->aux[3];
}

// ============================================================================
// IPC-EVT-001: 获取 Event 句柄
// ============================================================================

TEST_F(CudaIpcEventTest, IPC_EVT_001_GetEventHandle) {
    cudaError_t err = cudaIpcGetEventHandle(&eventHandle, event);
    ASSERT_EQ(err, cudaSuccess) << "cudaIpcGetEventHandle failed";
}

// ============================================================================
// IPC-EVT-002 & IPC-EVT-003: 跨进程打开 Event 与同步
// ============================================================================

TEST_F(CudaIpcEventTest, IPC_EVT_002_003_CrossProcessEventSync) {
    // cudaIpcOpenEventHandle 同样不允许在导出进程内打开（实测 201），且
    // 不能在 CUDA 初始化后 fork 子进程，因此复用全局待命子进程完成
    // 打开 -> 父进程 record -> 子进程同步 的流程
    ASSERT_NE(g_ipcEvtRegion, nullptr) << "shared event handshake region missing";
    ASSERT_GT(g_ipcMemChildPid, 0) << "standby child process missing";

    ASSERT_EQ(cudaIpcGetEventHandle(&eventHandle, event), cudaSuccess);

    g_ipcEvtRegion->handle = eventHandle;
    ASSERT_TRUE(runIpcEvtCommand(kIpcEvtCmdOpenSync, event))
        << "Child process failed or timed out, testResult="
        << g_ipcEvtRegion->testResult;

    ASSERT_EQ(g_ipcEvtRegion->testResult, 0)
        << "open err="
        << cudaGetErrorString(static_cast<cudaError_t>(g_ipcEvtRegion->aux[0]))
        << ", sync err="
        << cudaGetErrorString(static_cast<cudaError_t>(g_ipcEvtRegion->aux[1]));
}

// ============================================================================
// IPC-EVT-004: 关闭 Event 句柄
// ============================================================================

TEST_F(CudaIpcEventTest, IPC_EVT_004_CloseEventHandle) {
    // “关闭 Event 句柄”= 导入方对其导入的 event 执行 cudaEventDestroy
    // （实测返回成功，且不影响导出方原始 event）。打开同样必须在导入
    // 进程（待命子进程）执行，否则返回 201
    ASSERT_NE(g_ipcEvtRegion, nullptr) << "shared event handshake region missing";
    ASSERT_GT(g_ipcMemChildPid, 0) << "standby child process missing";

    ASSERT_EQ(cudaIpcGetEventHandle(&eventHandle, event), cudaSuccess);

    g_ipcEvtRegion->handle = eventHandle;
    ASSERT_TRUE(runIpcEvtCommand(kIpcEvtCmdOpenRecordDestroy, event))
        << "Child process failed or timed out, testResult="
        << g_ipcEvtRegion->testResult;

    // 子进程各步（打开 / 同步 / destroy）均应成功
    ASSERT_EQ(g_ipcEvtRegion->testResult, 0)
        << "open err="
        << cudaGetErrorString(static_cast<cudaError_t>(g_ipcEvtRegion->aux[0]))
        << ", sync err="
        << cudaGetErrorString(static_cast<cudaError_t>(g_ipcEvtRegion->aux[1]))
        << ", destroy err="
        << cudaGetErrorString(static_cast<cudaError_t>(g_ipcEvtRegion->aux[2]));

    // 导入方 destroy 后，导出方原始 event 应仍然有效
    EXPECT_EQ(cudaEventRecord(event, nullptr), cudaSuccess);
    EXPECT_EQ(cudaEventSynchronize(event), cudaSuccess);
}

// ============================================================================
// IPC-EVT-005: 非法 Event 句柄
// ============================================================================

TEST_F(CudaIpcEventTest, IPC_EVT_005_InvalidEventHandle) {
    // 空输出指针
    // EXPECT_NE(cudaIpcGetEventHandle(nullptr, event), cudaSuccess);

    // 打开空输出指针
    EXPECT_NE(cudaIpcOpenEventHandle(nullptr, eventHandle), cudaSuccess);

    // 打开非法句柄
    cudaIpcEventHandle_t invalidHandle{};
    std::memset(&invalidHandle, 0xAB, sizeof(invalidHandle));
    cudaEvent_t remoteEvent = nullptr;
    EXPECT_NE(cudaIpcOpenEventHandle(&remoteEvent, invalidHandle), cudaSuccess);
}

// ============================================================================
// IPC-ABI-001: 句柄大小验证
// ============================================================================

TEST(CudaIpcAbiTest, IPC_ABI_001_HandleSizeValidation) {
    // 运行时校验句柄结构大小与 ABI 定义一致
    EXPECT_EQ(sizeof(cudaIpcMemHandle_t), CUDA_IPC_HANDLE_SIZE);
    EXPECT_EQ(sizeof(cudaIpcEventHandle_t), CUDA_IPC_HANDLE_SIZE);
}

// ============================================================================
// IPC-STRESS-001: IPC 重复循环
// ============================================================================

TEST_F(CudaIpcMemoryTest, IPC_STRESS_001_RepeatedCycles) {
    // 同 IPC_MEM_008：open/close 必须在导入进程（待命子进程）执行，
    // 每轮由父进程重新导出句柄、子进程执行 打开 -> 写 -> 关闭
    ASSERT_NE(g_ipcMemRegion, nullptr) << "shared handshake region missing";
    ASSERT_GT(g_ipcMemChildPid, 0) << "standby child process missing";

    constexpr int kCycles = 50;

    for (int i = 0; i < kCycles; ++i) {
        cudaIpcMemHandle_t handle{};
        ASSERT_EQ(cudaIpcGetMemHandle(&handle, d_ptr), cudaSuccess)
            << "Cycle " << i << " get handle failed";

        g_ipcMemRegion->handle = handle;
        g_ipcMemRegion->cmdArg = i;
        ASSERT_TRUE(runIpcChildCommand(kIpcCmdMemStressCycle))
            << "Cycle " << i << " child command failed or timed out, testResult="
            << g_ipcMemRegion->testResult;
        ASSERT_EQ(g_ipcMemRegion->testResult, 0)
            << "Cycle " << i
            << ": open err="
            << cudaGetErrorString(static_cast<cudaError_t>(g_ipcMemRegion->aux[0]))
            << ", write err="
            << cudaGetErrorString(static_cast<cudaError_t>(g_ipcMemRegion->aux[1]))
            << ", close err="
            << cudaGetErrorString(static_cast<cudaError_t>(g_ipcMemRegion->aux[2]));
    }
}
