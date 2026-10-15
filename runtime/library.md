library模块功能
验证设备代码库加载、Kernel 枚举、Kernel 查询、函数获取、Kernel 启动及卸载流程完整，并检查对象生命周期和异常路径。

涉及接口：

cudaLibraryLoadData
cudaLibraryGetKernel
cudaLaunchKernelEx
cuLibraryLoadFromFile
cuLibraryEnumerateKernels
cuLibraryGetKernelCount
cuKernelGetFunction
cuLibraryUnload
测试内容：

测试用例设计

编号	测试场景	        核心步骤	验收标准
LIB-001	从内存加载 Library	准备合法数据，调用 cudaLibraryLoadData	 成功加载并返回有效对象
LIB-002	从文件加载 Library	调用 cuLibraryLoadFromFile 加载合法文件	 加载成功，文件内容有效
LIB-003	非法文件加载	    使用不存在、损坏或格式不合法的文件	       返回可诊断错误，不崩溃
LIB-004	查询 Kernel         数量	调用 cuLibraryGetKernelCount	返回值与实际 Kernel 数量一致
LIB-005	枚举 Kernel	        调用 cuLibraryEnumerateKernels	        返回的 Kernel 列表与 Library 内容一致
LIB-006	获取指定 Kernel     调用cudaLibraryGetKernel	            能正确获取目标 Kernel
LIB-007	查询 Kernel 函数	调用 cuKernelGetFunction	            返回有效函数对象，符合接口定义
LIB-008	扩展 Kernel 启动	使用 cudaLaunchKernelEx 启动合法 Kernel	 Kernel 执行成功，输出数据正确
LIB-009	非法启动参数	     测试不合法的 Grid、Block 或 Kernel 参数  返回契约规定的错误
LIB-010	卸载 Library	    调用 cuLibraryUnload	                成功卸载，资源被正确释放
LIB-011	重复卸载	        对已卸载对象再次调用卸载	              按契约处理，不发生异常崩溃
LIB-012	多 Library 并存	    加载不同 Library 并分别查询、执行、卸载	  对象间无状态串扰
LIB-013	生命周期边界	     在卸载后尝试使用相关对象	              按生命周期契约处理，避免非法访问
LIB-014	重复加载与卸载	     循环执行完整流程	                      无持续资源泄漏、崩溃或状态污染