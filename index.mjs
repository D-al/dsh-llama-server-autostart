// index.mjs — dsh-llama-server-autostart
//
// 随 DeepSeek Harness (dsh) 启动, 在后台无窗口(不占任务栏)拉起本地 llama-server
// (llama.cpp 官方服务端, OpenAI 兼容 /v1/chat/completions); 退出 dsh 时按 PID 树
// 精确关掉本次自己拉起的那个。
// 定位: dsh-ollama-autostart 的"路线②"姊妹插件 —— Ollama 图省心, llama.cpp 图性能
// 与可控性(上下文/量化/投机解码全开放)。两者可并存, 默认端口不同(11434 vs 8080)。
//
// 设计原则(与 dsh-ollama-autostart 一致):
//   - 全程 try/catch, 任何失败都静默, 绝不拖垮 dsh 启动。
//   - 不 import 任何框架包, 只用 Node 内置 API。
//   - 无窗口启动配方: stdio 重定向到真实日志文件 + windowsHide + detached +
//     CREATE_NO_WINDOW(llama-server 是控制台程序, 必须显式给 NO_WINDOW)。
import { spawn, execFile } from 'node:child_process'
import net from 'node:net'
import fs from 'node:fs'
import path from 'node:path'
import { fileURLToPath } from 'node:url'

const __dirname = path.dirname(fileURLToPath(import.meta.url))

// 调试日志: 默认写在插件目录旁; 可用 LLAMA_DEBUG_LOG 覆盖。
const LOG = process.env.LLAMA_DEBUG_LOG || path.join(__dirname, 'llama-debug.log')
function dbg(msg) {
  try { fs.appendFileSync(LOG, new Date().toISOString() + ' ' + msg + '\n') } catch (e) { /* 忽略 */ }
}

const DEFAULT_PORT = 8080
const DEFAULT_DIR = 'D:/llama.cpp/build/bin'          // llama-server.exe 所在目录
const DEFAULT_MODEL = 'D:/llama.cpp/models/Qwen3.5-9B-Q4_K_M.gguf'
const DEFAULT_MMPROJ = 'D:/llama.cpp/models/mmproj-F16.gguf'
const DEFAULT_CTX = '49152'

// 端口探测优先级: LLAMA_PORT > config.llamaPort > 8080
function resolvePort(config) {
  if (process.env.LLAMA_PORT) {
    const n = Number.parseInt(process.env.LLAMA_PORT, 10)
    if (Number.isFinite(n) && n > 0) return n
  }
  if (config && config.llamaPort) {
    const n = Number.parseInt(String(config.llamaPort), 10)
    if (Number.isFinite(n) && n > 0) return n
  }
  return DEFAULT_PORT
}

function probe(port) {
  return new Promise((resolve) => {
    const sock = net.connect(port, '127.0.0.1')
    sock.on('connect', () => { sock.destroy(); resolve(true) })
    sock.on('error', () => { sock.destroy(); resolve(false) })
    sock.setTimeout(1500, () => { sock.destroy(); resolve(false) })
  })
}

// ───────────────────────── Cordis 插件 ─────────────────────────
export const name = 'llama-server-autostart'

export function apply(ctx, config) {
  dbg('apply called')
  const port = resolvePort(config)
  const dir = process.env.LLAMA_SERVER_DIR || (config && config.llamaServerDir) || DEFAULT_DIR
  const exe = path.join(dir, 'llama-server.exe')
  const model = process.env.LLAMA_MODEL || (config && config.llamaModel) || DEFAULT_MODEL
  const mmproj = process.env.LLAMA_MMPROJ || (config && config.llamaMmproj) || DEFAULT_MMPROJ
  const ctxSize = String(process.env.LLAMA_CTX || (config && config.llamaCtx) || DEFAULT_CTX)
  const runLog = process.env.LLAMA_RUN_LOG || path.join(__dirname, 'llama-server-run.log')

  ctx.effect(() => {
    let child = null
    const boot = async () => {
      try {
        const up = await probe(port)
        dbg('probe(' + port + ') = ' + up)
        if (up) { dbg('llama-server 已在运行, 跳过'); return }

        if (!fs.existsSync(exe)) { dbg('未找到 llama-server.exe: ' + exe + ' (先完成 llama.cpp 编译)'); return }
        if (!fs.existsSync(model)) { dbg('未找到模型: ' + model); return }
        dbg('启动 llama-server: exe=' + exe + ' model=' + model + ' port=' + port + ' ctx=' + ctxSize)

        // 启动参数: 全 GPU 层 + flash attention + KV cache q4_0 量化(8GB 显卡跑 40K
        // 上下文不溢出且不掉速的关键, 比 q8_0 省一半显存);
        // mmproj 提供视觉能力(与 ollama 的 qwen3.5:9b vision 对齐)。
        // 注意: 本机 llama.cpp 0.7.0-dev 已不支持 -mm / --speculative-decode /
        // --kv-cache-quant(会导致 "error: invalid argument" 进程秒退 code=1), 故不写。
        const args = [
          '-m', model,
          '-c', ctxSize,
          '-ngl', '99',
          // 新版 llama.cpp(0.7.0+) 的 -fa/--flash-attn 需要显式值(on|off|auto);
          // 只写 -fa 会把后面的 --cache-type-k 当成它的值, 报 "unknown value for --flash-attn" 后退出。
          '-fa', 'on',
          '--cache-type-k', 'q4_0',
          '--cache-type-v', 'q4_0',
          // 多线程推理 (CPU 线程数, 8 为保守值)
          '-t', '8',
          // 批处理大小 (提升长 prompt 预填充吞吐量; 9B 模型在 4060 8GB 上 512 安全)
          '-b', '512',
          '--port', String(port),
          '--host', '127.0.0.1',
        ]
        if (mmproj && fs.existsSync(mmproj)) args.push('--mmproj', mmproj)

        // 无窗口启动: stdio 给真实日志文件 + windowsHide + detached + CREATE_NO_WINDOW。
        // 前置 CUDA 运行库目录到 PATH, 确保 llama-server.exe 能加载 cudart/cublas
        // (本机 CUDA 已迁移到 D:\CUDA\v12.5; 此处 D 优先、C 兜底, 迁移过渡期都兼容)。
        const cudaBinCandidates = [
          'D:\\CUDA\\v12.5\\bin',
          'C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.5\\bin'
        ]
        const spawnEnv = { ...process.env }
        const cudaBin = cudaBinCandidates.find(p => fs.existsSync(p))
        if (cudaBin) {
          spawnEnv.PATH = cudaBin + ';' + (spawnEnv.PATH || '')
        }
        const logFd = fs.openSync(runLog, 'a')
        child = spawn(exe, args, {
          cwd: dir,
          env: spawnEnv,
          detached: true,
          stdio: ['ignore', logFd, logFd],
          windowsHide: true,
          creationFlags: 0x08000000, // CREATE_NO_WINDOW
        })
        fs.closeSync(logFd) // 句柄已被子进程继承, 父进程即可关闭
        child.unref()
        dbg('spawn issued pid=' + child.pid)
        child.on('error', (e) => dbg('spawn ERROR: ' + (e && e.message)))
        child.on('exit', (code) => dbg('llama-server 退出 code=' + code))

        // 等端口就绪(6.6GB 模型加载 + CUDA 初始化约 20-60s, 给 180s)
        const t0 = Date.now()
        while (Date.now() - t0 < 180000) {
          await new Promise(r => setTimeout(r, 1500))
          if (await probe(port)) {
            dbg('llama-server 就绪, 用时 ' + ((Date.now() - t0) / 1000).toFixed(1) + 's')
            return
          }
        }
        dbg('llama-server 180s 内未就绪(详见 ' + runLog + ')')
      } catch (e) { dbg('boot EXCEPTION: ' + (e && e.message)) }
    }
    boot()

    return () => {
      dbg('disposer called')
      try {
        if (child && child.pid) {
          // 只杀本次拉起的这个(含进程树)
          execFile('taskkill.exe', ['/PID', String(child.pid), '/T', '/F'], { windowsHide: true }, () => {})
          dbg('taskkill /T /F pid=' + child.pid)
        } else {
          dbg('llama-server 非本次启动, 不关')
        }
      } catch (e) { dbg('disposer EXCEPTION: ' + (e && e.message)) }
    }
  })
}
