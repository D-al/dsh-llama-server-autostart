// index.mjs — dsh-llama-server-autostart
//
// 随 DeepSeek Harness (dsh) 启动，在后台无窗口 (不占任务栏) 拉起本地 llama-server
// (llama.cpp 官方服务端，OpenAI 兼容 /v1/chat/completions); 退出 dsh 时按 PID 树
// 精确关掉本次自己拉起的那些。
// 定位：dsh-ollama-autostart 的"路线②"姊妹插件 —— Ollama 图省心，llama.cpp 图性能
// 与可控性 (上下文/量化/投机解码全开放)。两者可并存，默认端口不同 (11434 vs 8080/8081)。
//
// 设计原则 (与 dsh-ollama-autostart 一致):
//   - 全程 try/catch, 任何失败都静默，绝不拖垮 dsh 启动。
//   - 不 import 任何框架包，只用 Node 内置 API。
//   - 无窗口启动配方：stdio 重定向到真实日志文件 + windowsHide + detached +
//     CREATE_NO_WINDOW(llama-server 是控制台程序，必须显式给 NO_WINDOW)。
//
// 双实例模式 (2026-10-20 新增):
//   - 通过 LLAMA_PROFILES 指定要启动的多个模型实例(逗号分隔), 每个实例用前缀
//     LLAMA_<PROFILE>_ 的一组键配置(同原单模型键名, 加前缀即可)。
//   - 例: LLAMA_PROFILES=35b,9b 会起两个服务 —— 35B@8081 (默认, 专家跑CPU) + 9B@8080 (带视觉)。
//   - dsh 前端(cordis.patch.yml)把两个模型 id 分别指到各自端口, 下拉框即"瞬时切换",
//     无需改配置 / 重启 dsh / 重载模型。
//   - 兼容旧单模型配置: 若设了 LLAMA_MODEL(平铺键)且无 LLAMA_PROFILES, 走 legacy 单实例。
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
// 视觉投影(mmproj)是否卸载到显存: '1'=跑显存(默认, 视觉快); '0'=跑 CPU(--no-mmproj-offload)。
// 实测(48K 长上下文, 2026-10-21): 设 0 可释放 ~900MB 显存, 把它投给 --moe-cache-mib(2048→3072),
//   解码 14.4 → 17.2 tok/s(+19.7%), 且显存余量反而更大(1108→1222MiB)。
//   代价: 视觉请求 5.7s → 16.5s(编码器改跑 CPU)。⇒ 文本为主的使用场景强烈建议设 0。
const DEFAULT_MMPROJ_OFFLOAD = '1'
// 投机解码的草稿模型: 同系列 0.8B(Q4_0, 483MB), 架构 qwen35、词表 248320 与主模型一致。
// 想关闭投机解码: 把环境变量 LLAMA_<P>_DRAFT 设为空字符串即可(文件不存在也会自动跳过)。
// 旧结论(9B dense 全GPU, 237 样本): 开启后 20→15.83 tok/s 慢21%, 草稿串行前向+低接受率入不敷出。
// 但 35B cpu-moe 上重测(2026-10-20): 外部 0.8B 草稿解码 +15%(22.4→25.8)、@16K 零额外显存,
// 因 CPU 瓶颈下减少目标模型前向次数即赚。故在 35B 上可选启用(设 LLAMA_35B_DRAFT=路径)。
// 当前默认关闭(方案B选 ngram-simple 而非外部草稿, 免多载一个模型)。
// 想启用: 设 LLAMA_<P>_DRAFT='D:/llama.cpp/models/Qwen3.5-0.8B-Q4_0.gguf'
const DEFAULT_DRAFT = ''
// 57344(56K): 实测加载草稿模型后 8192 时仍空闲 747MiB, 按 ~9.6MiB/1k 折算可容纳 ~7 万 token;
// 取 57344 足以覆盖现有最大会话(实测最大请求 50906 tokens)且留 ~500MiB 余量。
const DEFAULT_CTX = '57344'
// GPU 层数: 99 = 全部卸载到显存(dense 模型用)。MoE 大模型受 8GB 显存限制时,
// 用 LLAMA_<P>_NGL 指定层数 + LLAMA_<P>_CPU_MOE=1 把专家权重留在 CPU —— 大内存机器跑 35B-A3B 的关键配置。
const DEFAULT_NGL = '99'
const DEFAULT_CPU_MOE = ''        // '1' = 所有 MoE 专家权重留在 CPU
const DEFAULT_N_CPU_MOE = ''      // 数字 = 前 N 层的专家留在 CPU
const DEFAULT_MOE_CACHE_MIB = ''  // 显存中给 CPU 专家的缓存大小(MiB)
const DEFAULT_THREADS = '8'
// 加载模式: 默认 mmap+mlock(9B dense 用, 锁内存防换出);
// MoE 大模型用 cpu-moe 时, mmap 的按需分页会在 CPU 专家权重上产生大量缺页停顿,
// 拖慢解码。llama-server 官方建议改 --load-mode none(整模型预读进堆, 消除页错误),
// 实测 35B-A3B 解码 15.8 → 22.8 tok/s。故 35B 配置须设 LLAMA_35B_LOAD_MODE=none。
const DEFAULT_LOAD_MODE = 'mmap+mlock'
// 批处理大小: 提升长 prompt 预填充吞吐。CPU+GPU 混合 MoE 对 batch 极敏感——官方 llama.cpp
// 指南指出 op-offload 在 batch≥阈值时把 CPU 专家整体搬上 GPU 做 prompt 处理, 实测 512→2048
// 使 16k 预填充从 ~326 翻倍到 ~665 tok/s, 解码不受影响; 再大(4096)显存逼近 8GB 易 OOM, 故默认 2048。
const DEFAULT_BATCH = '2048'
const DEFAULT_UBATCH = '2048'
const DEFAULT_TB = ''             // 批处理线程数(-tb)。sweep 单测曾称 -tb 8/16 解码 +16%(22.4→26.0), 但隔离端口 3 次复测均值 22.5≈基线(样本 21.5/21.8/24.1)——增益未复现, 判定测量噪声。故默认空(不追加), 留作可选: 设 LLAMA_<P>_TB=8 自测。
const DEFAULT_SPEC_TYPE = 'none'  // 投机解码默认关。旧结论(9B dense)称 n-gram 慢24%, 但 35B cpu-moe 重测: ngram-simple 解码 +17%(22.4→26.1)、外部0.8B草稿 +15%(22.4→25.8), 因 CPU 瓶颈下减少目标模型前向次数即赚。启用: LLAMA_<P>_SPEC_TYPE=ngram-simple
// 思考预算(reasoning budget): 限制单次生成里"思考"token 上限, 直击感知延迟。
// 实测 Qwen3.5-35B 默认 thinking 对"12×12"这种简单题都挥霍 2000 token / 99.9s,
// 设 256 后砍到 264 token / 13.6s 且仍答对。512 是平衡值: 既杀掉琐碎题的过度思考,
// 又给多步推理留余地。设为 0 即关闭(回归无限思考)。可用 LLAMA_<P>_REASONING_BUDGET 覆盖。
const DEFAULT_REASONING_BUDGET = '512'
// 思考(reasoning)在响应里的呈现方式 —— 直接决定客户端能不能"看见思维链"。
//   deepseek(默认): 思考放进 message.reasoning_content 独立字段, 客户端若不渲染该字段 ⇒ 用户【看不见思维链】,
//     表现为"卡很久才出答案"却没有任何中间输出(实测 dsh 即此情况)。
//   deepseek-legacy: 保留 <think> 标签在 message.content 里, 同时仍填 reasoning_content ⇒ 前端能直接显示思维链。
// 可用 LLAMA_<P>_REASONING_FORMAT 覆盖; 设空串则不追加该参数(由 llama-server 决定)。
const DEFAULT_REASONING_FORMAT = 'deepseek-legacy'

// 读取插件旁的 .env.llama —— 此前这个文件是"死"的(写了整套参数但插件从不读, 导致
// 里面过时的参数从未生效, 也让切模型/调参必须改代码)。现已修复, 且支持多实例前缀键。
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

function probe(port) {
  return new Promise((resolve) => {
    const sock = net.connect(port, '127.0.0.1')
    sock.on('connect', () => { sock.destroy(); resolve(true) })
    sock.on('error', () => { sock.destroy(); resolve(false) })
    sock.setTimeout(1500, () => { sock.destroy(); resolve(false) })
  })
}

async function waitReady(port, ms) {
  const t0 = Date.now()
  while (Date.now() - t0 < ms) {
    await new Promise(r => setTimeout(r, 1500))
    if (await probe(port)) return true
  }
  return false
}

// 解析某实例的参数。
//   profile: '9b' | '35b' | '__legacy__'(兼容旧单模型 LLAMA_MODEL 等平铺键)
//   返回构建启动参数所需的全部字段; 端口/模型缺失时调用方负责跳过。
function resolveParams(profile, config) {
  const prefix = profile === '__legacy__' ? 'LLAMA_' : `LLAMA_${profile.toUpperCase()}_`
  const penv = (k, d) => { const v = getenv(prefix + k); return v === undefined ? d : v }

  const model = penv('MODEL') || (profile === '__legacy__' && (config && config.llamaModel)) || DEFAULT_MODEL
  // mmproj(视觉投影): 默认 9B 的 mmproj 路径仅对 9B 正确。
  // 关键: 若显式把 LLAMA_<P>_MMPROJ 设为空串(如 35B 块), 表示"禁用视觉" ——
  //   因为 9B 与 35B 的 mmproj 不通用, 绝不能回退到 DEFAULT_MMPROJ。
  // 仅当该键完全未出现(undefined)时, 才用默认(对 9B 是正确默认)。
  const mmprojRaw = getenv(prefix + 'MMPROJ')
  let mmproj
  if (mmprojRaw === undefined) {
    mmproj = (profile === '__legacy__' && (config && config.llamaMmproj)) || DEFAULT_MMPROJ
  } else if (mmprojRaw === '') {
    mmproj = ''   // 显式禁用视觉
  } else {
    mmproj = mmprojRaw
  }
  // 视觉编码器是否卸载到显存: '1'(默认)=显存, 视觉快; '0'=CPU, 释放~900MB显存换专家缓存(实测解码+19.7%)
  const mmprojOffload = penv('MMPROJ_OFFLOAD') || DEFAULT_MMPROJ_OFFLOAD
  const portRaw = penv('PORT')
  const port = portRaw ? Number.parseInt(portRaw, 10)
    : (profile === '__legacy__' && config && config.llamaPort) ? Number.parseInt(String(config.llamaPort), 10)
    : DEFAULT_PORT
  // 草稿模型: 默认空(关)。文件不存在时启动参数里自动跳过。
  const draftRaw = getenv(prefix + 'DRAFT')
  const draft = draftRaw !== undefined
    ? draftRaw
    : ((profile === '__legacy__' && (config && config.llamaDraft)) || DEFAULT_DRAFT)
  // n-gram 投机解码: 默认 'none'(关)。仅当设了具体类型才开启。
  const specType = penv('SPEC_TYPE') || (profile === '__legacy__' && (config && config.llamaSpecType)) || DEFAULT_SPEC_TYPE
  const ctxSize = String(penv('CTX') || (profile === '__legacy__' && (config && config.llamaCtx)) || DEFAULT_CTX)
  const ngl = String(penv('NGL') || DEFAULT_NGL)
  const threads = String(penv('THREADS') || DEFAULT_THREADS)
  const cpuMoe = penv('CPU_MOE') || DEFAULT_CPU_MOE
  const nCpuMoe = penv('N_CPU_MOE') || DEFAULT_N_CPU_MOE
  const moeCacheMib = penv('MOE_CACHE_MIB') || DEFAULT_MOE_CACHE_MIB
  const loadMode = penv('LOAD_MODE') || DEFAULT_LOAD_MODE
  const batchSize = String(penv('BATCH') || DEFAULT_BATCH)
  const ubatchSize = String(penv('UBATCH') || DEFAULT_UBATCH)
  const tb = penv('TB') || DEFAULT_TB
  // 思考预算: 仅当 >0 时追加 --reasoning-budget; 0 或空 = 关闭(无限思考)。
  const reasoningBudget = penv('REASONING_BUDGET')
    || (profile === '__legacy__' && (config && config.llamaReasoningBudget)) || DEFAULT_REASONING_BUDGET
  // 思考呈现格式: 非空才追加 --reasoning-format(见常量处说明; deepseek-legacy 让前端可见思维链)。
  const reasoningFormat = penv('REASONING_FORMAT') || DEFAULT_REASONING_FORMAT

  return { profile, model, mmproj, mmprojOffload, port, draft, specType, ctxSize, ngl, threads, tb, cpuMoe, nCpuMoe, moeCacheMib, loadMode, batchSize, ubatchSize, reasoningBudget, reasoningFormat }
}

// 由解析后的参数构造 llama-server 启动参数数组。
function buildArgs(p) {
  const args = [
    '-m', p.model,
    '-c', p.ctxSize,
    '-ngl', p.ngl,
    // Flash Attention 启用 (显著提升推理速度)
    '-fa', 'on',
    // KV cache 量化: q4_0 比 q8_0 省一半显存, 是 8GB 显卡跑长上下文不溢出且不掉速的关键
    '--cache-type-k', 'q4_0',
    '--cache-type-v', 'q4_0',
    // 多线程推理 (CPU 线程数; MoE 专家在 CPU 上跑时可调大)
    '-t', p.threads,
    // 批处理大小 (提升长 prompt 预填充吞吐量; CPU+GPU 混合 MoE 对此极敏感)。
    // 9B dense 全 GPU 用 2048 安全且更快; 35B-A3B cpu-moe 下 2048 使预填充翻倍(326→665 tok/s),
    // 再大(4096)显存逼近 8GB 易 OOM。可用 LLAMA_<P>_BATCH / LLAMA_<P>_UBATCH 覆盖。
    '-b', p.batchSize,
    '-ub', p.ubatchSize,
    // 加载模式(可配): 默认 mmap+mlock 把模型锁在物理内存, 避免被换出造成偶发卡顿;
    // 9B dense 用默认即可。MoE 大模型(cpu-moe)须设 LLAMA_35B_LOAD_MODE=none ——
    // mmap 的按需分页会让 CPU 专家权重频繁缺页, 拖慢解码(实测 15.8→22.8 tok/s 的关键)。
    '-lm', p.loadMode,
    // 进程优先级提到 high(2), 减少调度抖动。不用 realtime(3): 风险高、收益低。
    '--prio', '2',
    // 显式绑定端口与回环地址 (与 cordis provider baseURL 对齐;
    // 不依赖 llama-server 默认值, 避免默认 host 变化导致外部可访问)
    '--port', String(p.port),
    '--host', '127.0.0.1',
  ]
  // 批处理线程数(-tb, 可配): sweep 单测曾显示 -tb 8 解码 +16%, 但 3 次复测均值 22.5≈基线, 增益未复现(测量噪声)。
  // 故默认不追加; 设 LLAMA_<P>_TB=8 可自测。0/空=不追加(用 llama 默认= n_threads)。
  if (p.tb && Number(p.tb) > 0) args.push('-tb', String(p.tb))
  if (p.mmproj && fs.existsSync(p.mmproj)) {
    args.push('--mmproj', p.mmproj)
    // 视觉编码器卸载开关: '0' = 挪到 CPU, 释放 ~900MB 显存(可转投专家缓存, 实测解码 +19.7%)
    if (p.mmprojOffload === '0') args.push('--no-mmproj-offload')
  }
  // MoE / CPU 卸载: 大模型(如 Qwen3.5-35B-A3B)放不进 8GB 显存时, 把专家权重留在 CPU。
  // 三项都可独立开关; dense 模型(9B)设空即可, 不会追加任何参数。
  if (p.cpuMoe === '1') args.push('--cpu-moe')          // 全部专家权重留 CPU
  if (p.nCpuMoe) args.push('--n-cpu-moe', p.nCpuMoe)     // 仅前 N 层专家留 CPU
  if (p.moeCacheMib) args.push('--moe-cache-mib', p.moeCacheMib) // 显存里缓存热专家(MiB)
  // 投机解码: 0.8B 草稿提议 → 主模型把 N 个候选放进同一批验证。
  // 关键收益: 权重每批只读一次而非 N 次, 因此带宽受限场景同样有效。
  // 草稿文件不存在时自动跳过, 便于回退。
  if (p.draft && fs.existsSync(p.draft)) {
    args.push(
      '--spec-draft-model', p.draft,
      '--spec-draft-n-max', '6',      // 一次起草 6 个 token(默认 3)
      '--spec-draft-n-ubatch', '256', // 草稿物理批调小, 省显存
      '-ctkd', 'q4_0',                // 草稿 KV 量化, 再省一点显存
      '-ctvd', 'q4_0'
    )
  }
  // n-gram 投机解码(默认关闭): 无草稿模型, 靠上下文重复模式提议, 零额外显存。
  // 对代码/工具调用/路径等重复模式内容接受率较高。
  if (p.specType && p.specType !== 'none') {
    args.push(
      '--spec-type', p.specType,
      '--spec-draft-n-max', '8',          // 每轮最多验证 8+1 个 token
      '--spec-ngram-simple-size-n', '12', // 查找用 n-gram 长度(默认 12)
      '--spec-ngram-simple-size-m', '32', // 每次起草长度(默认 48, 收保守些)
      '--spec-ngram-simple-min-hits', '2' // 至少命中 2 次才提议, 提高接受率
    )
  }
  // 思考预算: 0/空则不追加(无限思考); >0 才限制单次生成的思考 token。
  // 思考预算: 显式设 0 = 立即结束思考(必须真的把 --reasoning-budget 0 传给 llama-server,
  //   否则它用默认 -1=无限思考, 表现为"卡半天/思考吃光token/无答案")。
  //   空串/未设 = 不追加该参数(交回 llama-server 默认)。注意用 >=0 而非 >0。
  if (p.reasoningBudget !== '' && p.reasoningBudget !== undefined && Number.isFinite(Number(p.reasoningBudget)) && Number(p.reasoningBudget) >= 0) {
    args.push('--reasoning-budget', String(p.reasoningBudget))
  }
  // 思考呈现格式: deepseek=放进 reasoning_content(多数前端不显示 ⇒ 看不到思维链);
  //   deepseek-legacy=把 <think> 留在 content ⇒ 前端可直接显示思维链。空串则不追加。
  if (p.reasoningFormat) args.push('--reasoning-format', p.reasoningFormat)
  return args
}

// ───────────────────────── Cordis 插件 ─────────────────────────
export const name = 'llama-server-autostart'

export function apply(ctx, config) {
  dbg('apply called (dual-instance)')
  const dir = getenv('LLAMA_SERVER_DIR') || (config && config.llamaServerDir) || DEFAULT_DIR
  const exe = path.join(dir, 'llama-server.exe')
  const runLog = getenv('LLAMA_RUN_LOG') || path.join(__dirname, 'llama-server-run.log')

  // 决定要拉起的实例列表
  const profilesRaw = getenv('LLAMA_PROFILES')
  let profiles = []
  if (profilesRaw) profiles = profilesRaw.split(',').map(s => s.trim()).filter(Boolean)
  if (profiles.length === 0) {
    // 无 LLAMA_PROFILES: 兼容旧单模型配置(设过 LLAMA_MODEL 则单实例), 否则默认双实例
    if (getenv('LLAMA_MODEL')) profiles = ['__legacy__']
    else profiles = ['35b', '9b']
  }

  ctx.effect(() => {
    const children = []   // 本次拉起的所有实例: { pid, port, tag }

    const bootAll = async () => {
      try {
        if (!fs.existsSync(exe)) { dbg('未找到 llama-server.exe: ' + exe + ' (先完成 llama.cpp 编译)'); return }

        for (const profile of profiles) {
          const p = resolveParams(profile, config)
          if (!p.model) { dbg('profile ' + profile + ' 缺模型, 跳过'); continue }
          dbg('准备实例 ' + profile + ': model=' + p.model + ' port=' + p.port +
            ' ctx=' + p.ctxSize + ' ngl=' + p.ngl + ' cpuMoe=' + (p.cpuMoe || p.nCpuMoe || '-') +
            ' load=' + p.loadMode + ' tb=' + (p.tb && Number(p.tb) > 0 ? p.tb : 'off') +
            ' mmprojOffload=' + p.mmprojOffload +
            ' spec=' + p.specType + ' reasoningBudget=' + (p.reasoningBudget !== '' && p.reasoningBudget !== undefined && Number.isFinite(Number(p.reasoningBudget)) && Number(p.reasoningBudget) >= 0 ? p.reasoningBudget : 'default(-1)'))

          const up = await probe(p.port)
          dbg('probe(' + p.port + ') = ' + up)
          if (up) { dbg('端口 ' + p.port + ' 已有服务, 跳过 ' + profile); continue }
          if (!fs.existsSync(p.model)) { dbg('未找到模型：' + p.model); continue }

          const args = buildArgs(p)
          dbg('启动 llama-server: model=' + p.model + ' port=' + p.port)

          // 无窗口启动：stdio 给真实日志文件 + windowsHide + detached + CREATE_NO_WINDOW。
          // 前置 CUDA 运行库目录到 PATH，确保 llama-server.exe 能加载 cudart/cublas
          // (本机 CUDA 已迁移到 D:\CUDA\v12.5；此处 D 优先、C 兜底，迁移过渡期都兼容)。
          const cudaBinCandidates = [
            'D:\\CUDA\\v12.5\\bin',
            'C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.5\\bin'
          ]
          const spawnEnv = { ...process.env }
          const cudaBin = cudaBinCandidates.find(x => fs.existsSync(x))
          if (cudaBin) spawnEnv.PATH = cudaBin + ';' + (spawnEnv.PATH || '')

          const logFd = fs.openSync(runLog, 'a')
          const child = spawn(exe, args, {
            cwd: dir,
            env: spawnEnv,
            detached: true,
            stdio: ['ignore', logFd, logFd],
            windowsHide: true,
            creationFlags: 0x08000000, // CREATE_NO_WINDOW
          })
          fs.closeSync(logFd) // 句柄已被子进程继承，父进程即可关闭
          child.unref()
          children.push({ pid: child.pid, port: p.port, tag: profile })
          dbg('spawn issued pid=' + child.pid + ' (' + profile + ')')
          child.on('error', (e) => dbg('spawn ERROR (' + profile + '): ' + (e && e.message)))
          child.on('exit', (code) => dbg('llama-server 退出 (' + profile + ') code=' + code))

          // 等该实例端口就绪后再起下一个(错峰加载, 避免瞬时内存/显存峰值叠加;
          // 同时确认它真的起来了)。TCP 端口在模型加载前就 bind, 故就绪很快, 不拖慢整体。
          const ok = await waitReady(p.port, 180000)
          dbg('实例 ' + profile + ' 端口 ' + p.port + ' 就绪=' + ok)
        }
      } catch (e) { dbg('bootAll EXCEPTION: ' + (e && e.message)) }
    }
    bootAll()

    return () => {
      dbg('disposer called, 关闭 ' + children.length + ' 个实例')
      for (const c of children) {
        try {
          if (c.pid) {
            // 只杀本次拉起的这个 (含进程树)
            execFile('taskkill.exe', ['/PID', String(c.pid), '/T', '/F'], { windowsHide: true }, () => {})
            dbg('taskkill /T /F pid=' + c.pid + ' (' + c.tag + ')')
          }
        } catch (e) { dbg('disposer EXCEPTION: ' + (e && e.message)) }
      }
    }
  })
}
