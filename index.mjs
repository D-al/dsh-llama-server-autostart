// index.mjs — dsh-llama-server-autostart
//
// 随 DeepSeek Harness (dsh) 启动，在后台无窗口 (不占任务栏) 拉起本地 llama-server
// (llama.cpp 官方服务端，OpenAI 兼容 /v1/chat/completions); 退出 dsh 时按 PID 树
// 精确关掉本次自己拉起的那个。
// 定位：dsh-ollama-autostart 的"路线②"姊妹插件 —— Ollama 图省心，llama.cpp 图性能
// 与可控性 (上下文/量化/投机解码全开放)。两者可并存，默认端口不同 (11434 vs 8080)。
//
// 设计原则 (与 dsh-ollama-autostart 一致):
//   - 全程 try/catch, 任何失败都静默，绝不拖垮 dsh 启动。
//   - 不 import 任何框架包，只用 Node 内置 API。
//   - 无窗口启动配方：stdio 重定向到真实日志文件 + windowsHide + detached +
//     CREATE_NO_WINDOW(llama-server 是控制台程序，必须显式给 NO_WINDOW)。
import { spawn, execFile } from 'node:child_process'
import net from 'node:net'
import fs from 'node:fs'
import path from 'node:path'
import { fileURLToPath } from 'node:url'

const __dirname = path.dirname(fileURLToPath(import.meta.url))

// 调试日志：默认写在插件目录旁; 可用 LLAMA_DEBUG_LOG 覆盖。
const LOG = process.env.LLAMA_DEBUG_LOG || path.join(__dirname, 'llama-debug.log')
function dbg(msg) {
  try { fs.appendFileSync(LOG, new Date().toISOString() + ' ' + msg + '\n') } catch (e) { /* 忽略 */ }
}

const DEFAULT_PORT = 8080
const DEFAULT_DIR = 'D:/llama.cpp/build/bin'          // llama-server.exe 所在目录
const DEFAULT_MODEL = 'D:/llama.cpp/models/Qwen3.5-9B-Q4_K_M.gguf'
const DEFAULT_MMPROJ = 'D:/llama.cpp/models/mmproj-F16.gguf'
// 投机解码的草稿模型: 同系列 0.8B(Q4_0, 483MB), 架构 qwen35、词表 248320 与主模型一致。
// 想关闭投机解码: 把环境变量 LLAMA_DRAFT 设为空字符串即可(文件不存在也会自动跳过)。
// 实测结论(237 个样本): 开启后生成速度 20 → 15.83 tok/s, 慢约 21%, 且多占 ~600MiB
// 显存(57344 上下文时仅剩 135MiB)。原因: 0.8B 草稿需串行跑 6 次 batch=1 前向,
// 每次重搬 483MiB 权重; 0.8B→9B 接受率过低, 提议大量被否决, 入不敷出。故默认关闭。
// 想再试: 设 LLAMA_DRAFT='D:/llama.cpp/models/Qwen3.5-0.8B-Q4_0.gguf'
const DEFAULT_DRAFT = ''
// 57344(56K): 实测加载草稿模型后 8192 时仍空闲 747MiB, 按 ~9.6MiB/1k 折算可容纳 ~7 万 token;
// 取 57344 足以覆盖现有最大会话(实测最大请求 50906 tokens)且留 ~500MiB 余量。
const DEFAULT_CTX = '57344'
// GPU 层数: 99 = 全部卸载到显存(dense 模型用)。MoE 大模型受 8GB 显存限制时,
// 用 LLAMA_NGL 指定层数 + LLAMA_CPU_MOE=1 把专家权重留在 CPU —— 大内存机器跑 35B-A3B 的关键配置。
const DEFAULT_NGL = '99'
const DEFAULT_CPU_MOE = ''        // '1' = 所有 MoE 专家权重留在 CPU
const DEFAULT_N_CPU_MOE = ''      // 数字 = 前 N 层的专家留在 CPU
const DEFAULT_MOE_CACHE_MIB = ''  // 显存中给 CPU 专家的缓存大小(MiB)
const DEFAULT_THREADS = '8'
// 加载模式: 默认 mmap+mlock(9B dense 用, 锁内存防换出);
// MoE 大模型用 cpu-moe 时, mmap 的按需分页会在 CPU 专家权重上产生大量缺页停顿,
// 拖慢解码。llama-server 官方建议改 --load-mode none(整模型预读进堆, 消除页错误),
// 实测 35B-A3B 解码 15.8 → 22.8 tok/s。故 35B 配置须设 LLAMA_LOAD_MODE=none。
const DEFAULT_LOAD_MODE = 'mmap+mlock'
// 批处理大小: 提升长 prompt 预填充吞吐。CPU+GPU 混合 MoE 对 batch 极敏感——官方 llama.cpp
// 指南指出 op-offload 在 batch≥阈值时把 CPU 专家整体搬上 GPU 做 prompt 处理, 实测 512→2048
// 使 16k 预填充从 ~326 翻倍到 ~665 tok/s, 解码不受影响; 再大(4096)显存逼近 8GB 易 OOM, 故默认 2048。
const DEFAULT_BATCH = '2048'
const DEFAULT_UBATCH = '2048'

// 读取插件旁的 .env.llama —— 此前这个文件是"死"的(写了整套参数但插件从不读, 导致
// 里面过时的 LLAMA_CTX=32768 等从未生效, 也让切模型/调参必须改代码)。
// 优先级: 真实进程环境变量 > .env.llama > 代码内默认值。
// 注意要退两级: __dirname 是 .../profiles/desktop/node_modules/dsh-llama-server-autostart,
// .env.llama 在 profile 根 .../profiles/desktop/.env.llama
const ENV_FILE = process.env.LLAMA_ENV_FILE || path.join(__dirname, '..', '..', '.env.llama')
const ENV_FILE_MAP = {}
try {
  const txt = fs.readFileSync(ENV_FILE, 'utf8')
  for (const raw of txt.split(/\r?\n/)) {
    const line = raw.trim()
    if (!line || line.startsWith('#')) continue
    const i = line.indexOf('=')
    if (i < 1) continue
    ENV_FILE_MAP[line.slice(0, i).trim()] = line.slice(i + 1).trim()
  }
  dbg('已加载 env 文件 ' + ENV_FILE + ' (' + Object.keys(ENV_FILE_MAP).length + ' 项)')
} catch { /* 无 env 文件, 纯用默认值 */ }

/** 取值: 进程环境变量优先, 其次 .env.llama, 最后代码默认(undefined 表示两处都没设)。 */
function getenv(key) {
  const v = process.env[key]
  return v === undefined ? ENV_FILE_MAP[key] : v
}

// 端口探测优先级：LLAMA_PORT > config.llamaPort > 8080
function resolvePort(config) {
  if (getenv('LLAMA_PORT')) {
    const n = Number.parseInt(getenv('LLAMA_PORT'), 10)
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
  const dir = getenv('LLAMA_SERVER_DIR') || (config && config.llamaServerDir) || DEFAULT_DIR
  const exe = path.join(dir, 'llama-server.exe')
  const model = getenv('LLAMA_MODEL') || (config && config.llamaModel) || DEFAULT_MODEL
  // mmproj(视觉投影): 默认 9B 的 mmproj 路径仅对 9B 正确。
  // 关键: 若显式把 LLAMA_MMPROJ 设为空串(如 35B 块), 表示"禁用视觉" ——
  //   因为 9B 与 35B 的 mmproj 不通用, 绝不能回退到 DEFAULT_MMPROJ。
  // 仅当该键完全未出现(undefined)时, 才用默认/配置(对 9B 是正确默认)。
  const mmprojRaw = getenv('LLAMA_MMPROJ')
  let mmproj
  if (mmprojRaw === undefined) {
    mmproj = (config && config.llamaMmproj) || DEFAULT_MMPROJ
  } else if (mmprojRaw === '') {
    mmproj = ''   // 显式禁用视觉
  } else {
    mmproj = mmprojRaw
  }
  const draft = getenv('LLAMA_DRAFT') !== undefined
    ? getenv('LLAMA_DRAFT')
    : ((config && config.llamaDraft) || DEFAULT_DRAFT)
  // n-gram 投机解码: 不需要草稿模型, 直接从上下文里找重复 n-gram 提议后续 token。
  // 零额外显存(对比 0.8B 草稿方案省 ~600MiB 且无串行小模型前向)。设 LLAMA_SPEC_TYPE=none 关闭。
  // 实测: n-gram 投机解码接受率仅 27.7%(62 accepted / 224 generated, mean len 9.86),
  // 生成速度 15.28 tok/s, 反而比不开启(~20 tok/s)慢约 24%, 故默认关闭('none')。
  // 根因: Qwen3.5 是 GatedDeltaNet 混合架构, 线性注意力的循环状态更新无法像纯注意力
  // 那样靠增大 batch 摊薄开销, 且接受率过低 ⇒ 投机解码在这类模型 + 8GB 卡上不划算。
  // 想再试: LLAMA_SPEC_TYPE=ngram-simple(或 ngram-cache / ngram-map-k 等)
  const specType = getenv('LLAMA_SPEC_TYPE') || (config && config.llamaSpecType) || 'none'
  const ctxSize = String(getenv('LLAMA_CTX') || (config && config.llamaCtx) || DEFAULT_CTX)
  const runLog = getenv('LLAMA_RUN_LOG') || path.join(__dirname, 'llama-server-run.log')
  // MoE / CPU 卸载相关(仅大模型需要; dense 9B 用默认值即可)
  const ngl = String(getenv('LLAMA_NGL') || DEFAULT_NGL)
  const threads = String(getenv('LLAMA_THREADS') || DEFAULT_THREADS)
  const cpuMoe = getenv('LLAMA_CPU_MOE') || DEFAULT_CPU_MOE
  const nCpuMoe = getenv('LLAMA_N_CPU_MOE') || DEFAULT_N_CPU_MOE
  const moeCacheMib = getenv('LLAMA_MOE_CACHE_MIB') || DEFAULT_MOE_CACHE_MIB
  const loadMode = getenv('LLAMA_LOAD_MODE') || DEFAULT_LOAD_MODE
  const batchSize = String(getenv('LLAMA_BATCH') || DEFAULT_BATCH)
  const ubatchSize = String(getenv('LLAMA_UBATCH') || DEFAULT_UBATCH)

  ctx.effect(() => {
    let child = null
    const boot = async () => {
      try {
        const up = await probe(port)
        dbg('probe(' + port + ') = ' + up)
        if (up) { dbg('llama-server 已在运行，跳过'); return }

        if (!fs.existsSync(exe)) { dbg('未找到 llama-server.exe: ' + exe + ' (先完成 llama.cpp 编译)'); return }
        if (!fs.existsSync(model)) { dbg('未找到模型：' + model); return }
        dbg('启动 llama-server: exe=' + exe + ' model=' + model + ' port=' + port + ' ctx=' + ctxSize + ' ngl=' + ngl + ' cpuMoe=' + (cpuMoe || nCpuMoe || '-'))

        // 启动参数：GPU 层数可调 + flash attention + KV cache q4_0 量化 + 多线程/批处理优化
        // 注意：本机 llama.cpp 为 0.7.0-dev 构建，以下 flag 已实测支持；
        //   -mm / --speculative-decode / --kv-cache-quant 在该版本不存在 (会导致
        //   "error: invalid argument" 且进程秒退 code=1)，故一律不写：
        //     · mmap 默认开启，无 -mm 开关；
        //     · KV 量化已由 --cache-type-k/v 覆盖，--kv-cache-quant 已废弃；
        //     · 投机解码该版本**支持**, 但旗标已改名: 用 --spec-draft-model(-md) 配
        //       --spec-draft-n-max; 旧名 --draft/--draft-max 已被移除(--help 有明确说明)。
        const args = [
          '-m', model,
          '-c', ctxSize,
          '-ngl', ngl,
          // Flash Attention 启用 (显著提升推理速度)
          '-fa', 'on',
          // KV cache 量化: q4_0 比 q8_0 省一半显存, 是 8GB 显卡跑 40K 上下文不溢出且不掉速的关键
          '--cache-type-k', 'q4_0',
          '--cache-type-v', 'q4_0',
          // 多线程推理 (CPU 线程数; MoE 专家在 CPU 上跑时可调大)
          '-t', threads,
          // 批处理大小 (提升长 prompt 预填充吞吐量; CPU+GPU 混合 MoE 对此极敏感)。
          // 9B dense 全 GPU 用 2048 安全且更快; 35B-A3B cpu-moe 下 2048 使预填充翻倍(326→665 tok/s),
          // 再大(4096)显存逼近 8GB 易 OOM。可用 LLAMA_BATCH / LLAMA_UBATCH 覆盖。
          '-b', batchSize,
          '-ub', ubatchSize,
          // 加载模式(可配): 默认 mmap+mlock 把模型锁在物理内存, 避免被换出造成偶发卡顿;
          // 9B dense 用默认即可。MoE 大模型(cpu-moe)须设 LLAMA_LOAD_MODE=none ——
          // mmap 的按需分页会让 CPU 专家权重频繁缺页, 拖慢解码(实测 15.8→22.8 tok/s 的关键)。
          '-lm', loadMode,
          // 进程优先级提到 high(2), 减少调度抖动。不用 realtime(3): 风险高、收益低。
          '--prio', '2',
          // 显式绑定端口与回环地址 (与 cordis provider baseURL 127.0.0.1:8080 对齐;
          // 不依赖 llama-server 默认值, 避免默认 host 变化导致外部可访问)
          '--port', String(port),
          '--host', '127.0.0.1',
        ]
        if (mmproj && fs.existsSync(mmproj)) args.push('--mmproj', mmproj)
        // MoE / CPU 卸载: 大模型(如 Qwen3.5-35B-A3B)放不进 8GB 显存时, 把专家权重留在 CPU。
        // 三项都可独立开关; dense 模型(9B)设空即可, 不会追加任何参数。
        if (cpuMoe === '1') args.push('--cpu-moe')          // 全部专家权重留 CPU
        if (nCpuMoe) args.push('--n-cpu-moe', nCpuMoe)       // 仅前 N 层专家留 CPU
        if (moeCacheMib) args.push('--moe-cache-mib', moeCacheMib) // 显存里缓存热专家(MiB)
        // 投机解码: 0.8B 草稿提议 → 9B 主模型把 N 个候选放进同一批验证。
        // 关键收益: 权重每批只读一次而非 N 次, 因此带宽受限场景同样有效。
        // 草稿文件不存在时自动跳过, 便于回退。
        if (draft && fs.existsSync(draft)) {
          args.push(
            '--spec-draft-model', draft,
            '--spec-draft-n-max', '6',      // 一次起草 6 个 token(默认 3)
            '--spec-draft-n-ubatch', '256', // 草稿物理批调小, 省显存
            '-ctkd', 'q4_0',                // 草稿 KV 量化, 再省一点显存
            '-ctvd', 'q4_0'
          )
        }
        // n-gram 投机解码(默认开启): 无草稿模型, 靠上下文重复模式提议, 零额外显存。
        // 对代码/工具调用/路径等重复模式内容接受率较高。
        if (specType && specType !== 'none') {
          args.push(
            '--spec-type', specType,
            '--spec-draft-n-max', '8',        // 每轮最多验证 8+1 个 token
            '--spec-ngram-simple-size-n', '12', // 查找用 n-gram 长度(默认 12)
            '--spec-ngram-simple-size-m', '32', // 每次起草长度(默认 48, 收保守些)
            '--spec-ngram-simple-min-hits', '2' // 至少命中 2 次才提议, 提高接受率
          )
        }

        // 无窗口启动：stdio 给真实日志文件 + windowsHide + detached + CREATE_NO_WINDOW。
        // 前置 CUDA 运行库目录到 PATH，确保 llama-server.exe 能加载 cudart/cublas
        // (本机 CUDA 已迁移到 D:\CUDA\v12.5；此处 D 优先、C 兜底，迁移过渡期都兼容)。
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
        fs.closeSync(logFd) // 句柄已被子进程继承，父进程即可关闭
        child.unref()
        dbg('spawn issued pid=' + child.pid)
        child.on('error', (e) => dbg('spawn ERROR: ' + (e && e.message)))
        child.on('exit', (code) => dbg('llama-server 退出 code=' + code))

        // 等端口就绪 (6.6GB 模型加载 + CUDA 初始化约 20-60s, 给 180s)
        const t0 = Date.now()
        while (Date.now() - t0 < 180000) {
          await new Promise(r => setTimeout(r, 1500))
          if (await probe(port)) {
            dbg('llama-server 就绪，用时 ' + ((Date.now() - t0) / 1000).toFixed(1) + 's')
            return
          }
        }
        dbg('llama-server 180s 内未就绪 (详见 ' + runLog + ')')
      } catch (e) { dbg('boot EXCEPTION: ' + (e && e.message)) }
    }
    boot()

    return () => {
      dbg('disposer called')
      try {
        if (child && child.pid) {
          // 只杀本次拉起的这个 (含进程树)
          execFile('taskkill.exe', ['/PID', String(child.pid), '/T', '/F'], { windowsHide: true }, () => {})
          dbg('taskkill /T /F pid=' + child.pid)
        } else {
          dbg('llama-server 非本次启动，不关')
        }
      } catch (e) { dbg('disposer EXCEPTION: ' + (e && e.message)) }
    }
  })
}
