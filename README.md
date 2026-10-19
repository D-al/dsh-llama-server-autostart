# dsh-llama-server-autostart

随 DeepSeek Harness (dsh) 启动，在后台**无窗口**（不占任务栏）拉起本地 **llama-server**（llama.cpp 官方服务端，OpenAI 兼容 `/v1/chat/completions`）；退出 dsh 时按 PID 树精确关掉本次自己拉起的那个，不误伤其他实例。

这是 [`dsh-ollama-autostart`](../dsh-ollama-autostart) 的姊妹插件——「路线②」：

| | Ollama（路线①） | llama-server（路线②） |
| --- | --- | --- |
| 定位 | 省心封装 | 性能与可控性 |
| 端口 | 11434 | 8080 |
| 上下文 | 默认 4096，需 Modelfile/环境变量调 | 命令行直接 `-c 32768` |
| KV cache 量化 | `OLLAMA_KV_CACHE_TYPE=q8_0`（插件已传） | `--cache-type-k/v q8_0`（插件已传） |
| 采样/模板 | Modelfile 调 | GGUF 内嵌模板 + 请求参数 |

两者可并存（端口不同），在 dsh 的 provider 配置里切换即可。

## 前提

1. 已编译 llama.cpp：源码在 `D:\llama.cpp\src`，产物 `D:\llama.cpp\build\bin\llama-server.exe`（CUDA 13.4，sm_89）。
2. 模型文件：`D:\llama.cpp\models\Qwen3.5-9B-Q4_K_M.gguf`（+ `mmproj-F16.gguf` 视觉）。

## 安装

方式一（推荐）：在 dsh 插件管理器用 GitHub 链接安装（仓库就绪后）：
`https://github.com/D-al/dsh-llama-server-autostart`

方式二（手动）：把本仓库克隆到 `~/.dsh/profiles/<profile>/node_modules/dsh-llama-server-autostart/`，并在该 profile 的 `package.json` 的 `dsh.profile.bundles` 里加上 `dsh-llama-server-autostart`，然后**完全重启 dsh**。

dsh 侧 provider 配置（`cordis.patch.yml`，llm-pi-ai）：

```yaml
      llama:
        displayName: llama.cpp 本地
        api: openai-completions
        baseURL: http://127.0.0.1:8080/v1
        headers:
          Authorization: "Bearer llama"
        defaultContextWindow: 32768
        defaultMaxTokens: 8192
        timeoutMs: 600000
        streamIdleTimeoutMs: 600000
```

## 配置（环境变量）

| 变量 | 说明 | 默认值 |
| --- | --- | --- |
| `LLAMA_PORT` | 服务端口 | `8080` |
| `LLAMA_SERVER_DIR` | `llama-server.exe` 所在目录 | `D:/llama.cpp/build/bin` |
| `LLAMA_MODEL` | 主模型 GGUF 路径 | `D:/llama.cpp/models/Qwen3.5-9B-Q4_K_M.gguf` |
| `LLAMA_MMPROJ` | 视觉 projector 路径（留空禁用视觉） | `D:/llama.cpp/models/mmproj-F16.gguf` |
| `LLAMA_CTX` | 上下文长度 | `32768` |
| `LLAMA_RUN_LOG` | llama-server 运行日志路径 | 插件目录旁 `llama-server-run.log` |
| `LLAMA_DEBUG_LOG` | 插件调试日志路径 | 插件目录旁 `llama-debug.log` |

## 调试

- 插件日志：`llama-debug.log`（apply / 探测 / 启动 / 关闭全过程）。
- 运行日志：`llama-server-run.log`（模型加载、HTTP 请求记录）。
- 手动验证：`curl http://127.0.0.1:8080/v1/models`。
- 首次启动模型加载约 20-60s，端口 listen 即就绪；期间请求会排队等待。

## Requirements

Windows with a compiled llama.cpp (CUDA build recommended). The plugin locates `llama-server.exe` via `LLAMA_SERVER_DIR` → `D:/llama.cpp/build/bin`. Install via dsh's plugin manager with the git URL, then fully restart dsh. Configurable via env vars: `LLAMA_PORT`, `LLAMA_SERVER_DIR`, `LLAMA_MODEL`, `LLAMA_MMPROJ`, `LLAMA_CTX`, `LLAMA_RUN_LOG`, `LLAMA_DEBUG_LOG`.
