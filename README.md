# dsh-llama-server-autostart

**DeepSeek Harness（dsh）本地引擎生命周期管理插件** —— 随 dsh 启动在后台无窗口拉起本地 llama-server（llama.cpp 编译版），退出时清理，对已运行实例执行收养而非重复拉起。

## 遇到的问题

### 1. 手动启动链路繁琐且易错
本地 35B 推理依赖 llama-server（OpenAI 兼容接口）——每次开 dsh 前要手动开终端、设 CUDA PATH、拼十几行参数（router 模式/KV 量化/投机解码/模板文件……83 个参数）。拼错一个就静默失败或性能大幅退化。

### 2. 进程残留与双重实例
dsh 退出后 llama-server 不退出 → 下次启动时端口占用 → 再起一个 → **两个实例叠加 30GB+ 内存**（实测疑云现场：父进程树排查后发现是新旧两代实例并存）。

### 3. 端口探测误判（netstat 正则缺陷）
收养逻辑依赖 netstat 判断 8081 端口归属——正则曾把无关进程的监听也匹配成"llama 已在跑"，导致该启动时不启动（dsh 挂在不可用接口上）。

### 4. 沙箱派生进程被 CPU 限速
从 dsh 的沙箱终端直接启动 llama-server 会被 CPU 限速（推理速度只有正常值的几分之一）——必须走计划任务或插件的无窗口 spawn。

## 途径

1. **probe-then-spawn**：先探测 8081 是否已有健康服务（TCP connect + 超时），有则收养（记录 pid，退出时只清理本次拉起的），无则无窗口 spawn——`windowsHide: true` 规避终端闪窗
2. **参数全走 .env.llama 环境变量**：83 个参数从 profile 的 `.env.llama` 读取（`LLAMA_PROFILES`/`CTX`/`NGL`/`NCMOE`/`SPEC_TYPE`/`REASONING_BUDGET`……），模型配置与代码彻底解耦——改参数不用动代码
3. **退出清理精确化**：只杀"本次拉起的 pid"，不误伤用户手动启动的实例（收养的实例退出时保留）
4. **CUDA PATH 前置**：spawn 环境自动前置 `D:\CUDA\v12.5\bin`（多候选探测），不依赖系统 PATH（重装系统后免疫）
5. **配置分层**：插件配置（cordis 挂载）→ 进程环境变量（`.env.llama`）→ 代码默认值三级，任何一层可覆盖

## 参考的论文与社区实践

本插件以工程实践为主（进程管理/端口探测），无直接论文依赖。结构与包形态参考：

| 来源 | 用于 |
|---|---|
| **dsh-session-trash-bin（社区插件）** | bundle 包形态逆向：package.json 的 `dsh.bundle.patch` 字段 + 包内 cordis.patch.yml 的 insert 声明 |
| **dsh 插件管理器 schema**（plugin_manager 工具描述） | install_bundle 的 npm 安装规范、enable/disable 生命周期 |
| **实际失效现场** | 收养逻辑的每个分支都来自真实事故（双重实例 30G 内存疑云、netstat 正则误判、沙箱限速）|

## 机制与实测

| 机制 | 行为 | 实测 |
|---|---|---|
| **probe-then-spawn** | TCP 探测 8081 → 无服务才 spawn | 收养已运行实例（不重复拉起）；spawn→就绪 2 秒 |
| **退出清理** | 只杀本次 spawn 的 pid | 收养实例保留；无进程残留 |
| **收养逻辑** | 8081 已有健康服务 → 记录并收养 | 双重实例疑云现场修复（30GB 内存降回正常）|
| **参数链路** | `.env.llama` → resolveParams → 83 参数 router 模式 | router/模板/KV q8_0/投机解码/预算全部正确；命令行可逐项核对 |
| **CUDA PATH 前置** | 多候选探测（D:\CUDA\v12.5\bin 等） | 不依赖系统 PATH；重装系统免疫 |
| **双实例支持** | `LLAMA_PROFILES` 环境变量（35b/9b） | 按需多模型 |

## 解决到什么程度

- llama-server 全参数正确拉起（router 模式 83 参数逐项核对：`--models-preset`/`--chat-template-file`/`--cache-type-k q8_0`/`--fit off`/投机解码/重复惩罚）
- spawn→就绪 2 秒；退出清理精确；收养不误伤
- **重要教训（2026-11-17）**：本插件的 node_modules 目录曾被 `pnpm update` 用 GitHub 上的旧版覆盖（新版从未上传）——导致 llama-server 以旧参数（直连模式/无模板/q4_0 KV）运行。修复后确立纪律：**本地维护的插件必须先推 GitHub 再 pnpm update**。本仓库当前版本已含全部新版逻辑（收养/netstat 正则/退出清理/参数链路）
- 配套发现：bundle 包的 `dsh.bundle.patch` overlay 声明的 cordis.patch.yml **必须在包内真实存在**（缺失会让 dsh 标记插件"异常"）——本仓库已补齐

## 配置说明

配置几乎全部走 `.env.llama` 环境变量（挂载时 config 可为空）：

```
LLAMA_PROFILES=35b          # 实例列表（逗号分隔；未设则双实例）
MODEL / CTX / NGL / NCMOE   # 模型/上下文/GPU 层数/CPU MoE 层数
SPEC_TYPE / REASONING_BUDGET / CTX_CHECKPOINTS  # 投机解码/思考预算/检查点
LLAMA_SERVER_DIR            # llama-server.exe 所在目录
```

完整生产值参见仓库内 `.env.llama` 示例（本机：CTX 98304 / NGL 99 / NCMOE 40 / reasoning-budget 4096 / router 模式）。

## License
MIT
