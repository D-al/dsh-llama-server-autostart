# dsh-llama-server-autostart

随 DeepSeek Harness (dsh) 启动，在后台**无窗口**（不占任务栏）拉起本地 **llama-server**（llama.cpp 官方服务端，OpenAI 兼容 `/v1/chat/completions`）；退出 dsh 时按 PID 树精确关掉本次自己拉起的那个，不误伤其他实例。

这是 [`dsh-ollama-autostart`](../dsh-ollama-autostart) 的姊妹插件——「路线②」：

| | Ollama（路线①） | llama-server（路线②） |
| --- | --- | --- |
| 定位 | 省心封装 | 性能与可控性 |
| 端口 | 11434 | 8080 / 8081 |
| 上下文 | 默认 4096，需 Modelfile/环境变量调 | 命令行直接 `-c 57344` |
| KV cache 量化 | `OLLAMA_KV_CACHE_TYPE=q8_0`（插件已传） | `--cache-type-k/v q4_0`（插件已传） |
| 采样/模板 | Modelfile 调 | GGUF 内嵌模板 + 请求参数 |
| 多实例 | 单实例 | `LLAMA_PROFILES` 支持多模型并存 |

两者可并存（端口不同），在 dsh 的 provider 配置里切换即可。

## 前提

1. 已编译 llama.cpp：源码在 `D:\llama.cpp\src`，产物 `D:\llama.cpp\build\bin\llama-server.exe`。
2. 模型文件（按需）：`Qwen3.5-35B-A3B-UD-Q3_K_XL.gguf` + `Qwen3.5-35B-A3B-mmproj-F16.gguf`（视觉）。

## 安装

**方式一（推荐）**：在 dsh 插件管理器用 GitHub 链接安装：
`https://github.com/D-al/dsh-llama-server-autostart`

> ⚠️ **必须先完全退出 dsh 再安装/更新**。正在运行的 dsh 会锁住 `node_modules` 目录，导致安装报 `EPERM`（表现为"没有写入权限"）。完全退出（任务管理器确认无 `DeepSeek Harness.exe` 残留）后再装。

**方式二（手动）**：把本仓库克隆到 `~/.dsh/proiles/<profile>/node_modules/dsh-llama-server-autostart/`，并在该 profile 的 `package.json` 里同时加：
- `dependencies`: `"dsh-llama-server-autostart": "git+https://github.com/D-al/dsh-llama-server-autostart.git"`
- `dsh.profile.bundles`: `"dsh-llama-server-autostart"`

然后**完全重启 dsh**。

> ⚠️ **两个位置都要加**。只加 `bundles` 不加 `dependencies`（或反之）会导致插件静默不加载——表现为"dsh 调用不了模型"（没人拉起 llama-server）。

## 配置

配置文件是 profile 根目录的 **`.env.llama`**（插件启动时读取，**不是**死文件）。
优先级：`进程环境变量 > .env.llama > 插件内默认值`。

### 多实例

`LLAMA_PROFILES=35b,9b`（逗号分隔）会拉起多个实例，每个实例用 `LLAMA_<PROFILE>_*` 前缀键配置，顺序错峰启动。
未设 `LLAMA_PROFILES` 但设了 `LLAMA_MODEL`（平铺键）时走旧版单实例逻辑。

### 旋钮总览（`LLAMA_<PROFILE>_*`，单实例时可省略前缀）

| 变量 | 对应 llama-server 参数 | 默认 | 说明 |
| --- | --- | --- | --- |
| `MODEL` | `-m` | 9B GGUF | 主模型路径 |
| `MMPROJ` | `--mmproj` | 9B mmproj | 留**空串**=禁用视觉（不回退默认） |
| `MMPROJ_OFFLOAD` | `--no-mmproj-offload` | `1` | `0`=视觉编码器跑 CPU，**释放 ~900MB 显存** |
| `PORT` | `--port` | `8080` | 服务端口 |
| `CTX` | `-c` | `57344` | 上下文长度 |
| `NGL` | `-ngl` | `99` | 卸载到显存的层数 |
| `THREADS` | `-t` | `8` | 计算线程数 |
| `TB` | `-tb` | 空（不追加） | 批处理线程数 |
| `CPU_MOE` | `--cpu-moe` | 空 | `1`=所有 MoE 专家权重留 CPU |
| `N_CPU_MOE` | `--n-cpu-moe` | 空 | 前 N 层专家留 CPU |
| `MOE_CACHE_MIB` | `--moe-cache-mib` | 空 | 显存中 CPU 专家缓存大小（MiB） |
| `LOAD_MODE` | `-lm` | `mmap+mlock` | `none`/`mmap`/`mlock`/`mmap+mlock` |
| `BATCH` | `-b` | `2048` | 批处理大小（长 prompt 预填充） |
| `UBATCH` | `-ub` | `2048` | 物理批大小 |
| `SPEC_TYPE` | `--spec-type` | `none` | 投机解码类型 |
| `DRAFT` | `--spec-draft-model` | 空 | 外部草稿模型 |
| `REASONING_BUDGET` | `--reasoning-budget` | `512` | 思考 token 预算，**`0`=完全不思考** |
| `REASONING_FORMAT` | `--reasoning-format` | `deepseek-legacy` | 思考呈现格式 |
| `SERVER_DIR` | — | `D:/llama.cpp/build/bin` | `llama-server.exe` 目录 |
| `RUN_LOG` / `DEBUG_LOG` | — | 插件旁 | 日志路径 |

### ⚠️ `REASONING_BUDGET=0` 语义（重要）

设 `0` 会**真的**传 `--reasoning-budget 0`（立即结束思考）。若留空/不设，则用默认 `512`。
**不要**把它和"不设"混淆——历史上留空会让 llama-server 用 `-1`（无限思考），表现为
"卡半天、思考吃光 token、`content` 为空没有答案"。

### 35B-A3B MoE 实测调优结论（8GB 显存）

| 项 | 结论 |
| --- | --- |
| 解码瓶颈 | **内存带宽**（专家在 CPU，每 token 读 ~0.8GB）；roofline ≈53 tok/s |
| 上下文影响 | 解码随上下文下滑：1K=19.0 / 16K=17.3 / 32K=15.7 / **48K=14.5** tok/s |
| 真旋钮 | 只有 `--moe-cache-mib`（专家缓存进显存=把字节搬离 RAM） |
| 无效/危险 | `-tb`、`--cpu-mask/range`、投机解码、KV `q8_0`、线程 16、`--no-op-offload`（会卡死）、`--n-cpu-moe`（净亏 10%） |
| 定稿 | `ctx 57344 + moe-cache 3072 + no-mmproj-offload + reasoning-budget 0` |
| 预填充 | 48K 冷预填充约 72s（668 tok/s），旋钮无法改善 |
| 稳定性 | 显存余量需 >500MiB，长 prompt 预填充要额外显存；**装得下 ≠ 能用** |

完整实测数据见 `bench/` 目录（`bench35b_*.mjs` / `bench35b_*.json`）。

## 调试

- 插件日志：`llama-debug.log`（apply / 探测 / 启动 / 关闭全过程，含解析到的最终参数）。
- 运行日志：`llama-server-run.log`（模型加载、每请求计时：`prompt eval time` / `eval time`）。
- 手动验证：`curl http://127.0.0.1:8081/v1/models`。
- 请求阶段实时状态：`curl http://127.0.0.1:8081/slots`（`is_processing` / `n_prompt_tokens_processed` 可判断是否卡死）。
- 端口在模型加载**前**就 bind，`waitReady` 探到即返回；真正就绪看日志里的 `listening on`。

## Requirements

Windows with a compiled llama.cpp (CUDA build recommended). Install via dsh's plugin manager with the git URL (**fully exit dsh first**), then fully restart dsh. Config lives in `.env.llama` at the profile root. Supports multi-instance via `LLAMA_PROFILES` with per-instance `LLAMA_<PROFILE>_*` keys.
