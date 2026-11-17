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
// GPU 层数: 默认 'auto' = 不传 -ngl, 让 llama.cpp 按当前空闲显存自动适配层数 ——
// 8GB 显卡跑 35B-A3B 的最稳方案(避免 -ngl 99 因装不下而 abort、也避免 cpu-moe 把专家全压 CPU 导致长 prompt 预填充崩到 ~0.06 tok/s)。
// 若想强制指定层数, 设 LLAMA_<P>_NGL=非负整数(如 99); 'auto' 或空则不追加 -ngl。
const DEFAULT_NGL = 'auto'
const DEFAULT_CPU_MOE = ''        // '1' = 所有 MoE 专家权重留在 CPU
const DEFAULT_N_CPU_MOE = ''      // 数字 = 前 N 层的专家留在 CPU
const DEFAULT_MOE_CACHE_MIB = ''  // 显存中给 CPU 专家的缓存大小(MiB)
// KV cache 量化类型: q4_0 是 8GB 显卡跑长上下文的默认选择(q8_0 精度更高但显存翻倍)。
// Qwen3.6-35B-A3B 是混合注意力: 每 4 层才 1 层吃 KV(2 个 KV head × 256 dim),
// 所以 q4_0 下每 token 仅 ~5.6KiB ⇒ 196K 上下文也只要 ~1.2GiB。
const DEFAULT_CTK = 'q4_0'
const DEFAULT_CTV = 'q4_0'
// 并发槽位数: 1 = 单对话独占整个上下文预算。默认 1(本地单人使用, 多 slot 会瓜分 KV)。
const DEFAULT_NP = '1'
const DEFAULT_THREADS = '8'
// MTP 草稿长度(--spec-draft-n-max): 每轮前向最多验证 N+1 个 token。
// ⚠ 这是 cpu-moe 场景下最影响解码速度的旋钮之一, 实测(35B-A3B @196K, 8GB):
//   N=8 → 17.4 tok/s / N=6 → 24.4 / N=4 → 27.5 (自由文本 acceptance≈0.3)。
//   原因: N 越大, 每轮验证的无效 token 越多, CPU 专家算力被白烧; 文本越"可预测"才越该调大。
//   设 LLAMA_<P>_NMAX 覆盖(默认 4, 兼顾速度与重复性文本)。
const DEFAULT_NMAX = '3'
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
  const ctk = penv('CTK') || DEFAULT_CTK
  const ctv = penv('CTV') || DEFAULT_CTV
  const np = String(penv('NP') || DEFAULT_NP)
  const nmax = String(penv('NMAX') || DEFAULT_NMAX)
  const tb = penv('TB') || DEFAULT_TB
  // 采样温度: 空 = 不追加(--temp), 用 llama-server 默认(1.0)。
  //   工具调用型 agent 用 1.0 偏"发散", 容易在工具调用边界跑偏(实测偶发提前 EOS), 建议 0.7。
  const temp = penv('TEMP') || ''
  // 采样 top-p: 空 = 不追加。配合 TEMP 使用。
  const topP = penv('TOP_P') || ''
  // 反重复: llama-server 默认 repeat_penalty=1.0(等于关闭), 长上下文里思考容易陷入
  // "同一句话循环"的退化。下面几个键都是"空=不追加"。
  const repeatPenalty = penv('REPEAT_PENALTY') || ''
  // 反重复窗口: 默认 64 token 太短, 跨请求(上一条消息)的复读罚不到。
  // 2026-11-06 实测: 消息级死循环(连续 5 步正文一字不差 + 重复读同一文件)时 64 完全无效。
  const repeatLastN = penv('REPEAT_LAST_N') || ''
  // DRY 采样: 对重复 n-gram 施加指数惩罚, 比 repeat_penalty 更精准(不误伤正常用词)。
  const dryMultiplier = penv('DRY_MULTIPLIER') || ''
  const dryBase = penv('DRY_BASE') || ''
  const dryAllowedLength = penv('DRY_ALLOWED_LENGTH') || ''
  const dryPenaltyLastN = penv('DRY_PENALTY_LAST_N') || ''
  // 采样补充: min-p / top-k(空=不追加), 投机解码草稿阈值
  const minP = penv('MIN_P') || ''
  const topK = penv('TOP_K') || ''
  const specDraftPMin = penv('SPEC_DRAFT_P_MIN') || ''
  // 上下文检查点数: 默认 32。社区报告该"检查点巡回"逻辑对混合注意力模型(如 qwen3.6)失效,
  // 导致每次请求都全文重算(实测 50K 上下文每次浪费 38~41s)。设 0 = 关闭检查点,回退到简单前缀复用。
  const ctxCheckpoints = penv('CTX_CHECKPOINTS') || ''
  // 宿主 prompt 缓存上限(MiB) 与 "新任务时保存空闲槽位" 开关。
  // 2026-11-06 定位: --cache-idle-slots 默认开启, 新任务到达时把槽位状态整份存进宿主缓存;
  // 对混合注意力模型(DeltaNet 循环状态)这次保存是单线程大拷贝, 实测把 60K 预填充从 850 拖到 277 t/s。
  const cacheRam = penv('CACHE_RAM') || ''
  const cacheIdleSlots = penv('CACHE_IDLE_SLOTS') || ''
  // 日志级别: 0=generic 1=error 2=warn 3=info(默认) 4=debug 5=trace。排障时设 4。
  const logVerbosity = penv('LOG_VERBOSITY') || ''
  // 图像 token 上限: 视觉编码耗时与图像 token 数近似线性。2542x1347 的截图默认要给 1600 个 token,
  // CPU 上单张编码 141 秒；限到 1024 可同时省显存与时间。空=用模型自带值。
  const imageMaxTokens = penv('IMAGE_MAX_TOKENS') || ''
  const imageMinTokens = penv('IMAGE_MIN_TOKENS') || ''
  // 服务端默认 logit bias（2026-11-07 新增，默认关闭）。
  // 用途：万一「只输出好」在去掉上下文污染后仍有残余，可直接在采样分布上压低短确认词，
  // 不用改模板/persona。格式 "TOKEN_ID-BIAS,..."，例如 "95887-100,96747-100"
  // （95887=好，96747=好的；-100 近似禁止）。请求体若自带 logit_bias 会覆盖这里。
  // ⚠ 全局禁字会影响正常中文表达，生产默认留空；治本靠 chat 模板不渲染退化历史消息。
  const logitBias = penv('LOGIT_BIAS') || ''
  // MoE cache 未命中的 CPU 分担比例（2026-11-10，需 llama.cpp >= b12xxx）。
  // 未命中的专家不再从内存上传到 GPU，而是直接在 CPU 上算。文档说明：用 -lm mlock/dio
  // 锁定内存时上传是全速、该参数往往无效；-lm none（本机）才是它发挥作用的情形。
  // 小 cache 时收益大（作者实测 cache 6464 MiB、share 0.95 → +97%），大 cache 时反而拖慢。
  const moeCacheCpuShare = penv('MOE_CACHE_CPU_SHARE') || ''
  // 提示批专用份额（需要 b12244-local 补丁：--moe-cache-cpu-share-prefill）。
  // 目的：解码批走大份额（CPU 算 miss 省 PCIe），增量预填充走 0（CPU 算不过来）。
  const moeCacheCpuSharePrefill = penv('MOE_CACHE_CPU_SHARE_PREFILL') || ''
  // MoE 专家缓存淘汰策略（仅 ik 支持）：lru（默认）或 classlru
  const moeCachePolicy = penv('MOE_CACHE_POLICY') || ''
  // 路由模式（2026-11-11）：非空时以 router 模式启动（不传 -m），由 --models-preset 指向的
  // INI 定义多个模型（Q3/Q4），请求体里的 "model" 字段决定加载哪一个；--models-max 限制同时驻留数，
  // --models-memory-margin 让 router 在装不下时按 LRU 卸载。8GB 卡上 models-max 必须是 1。
  // 注意：n-cpu-moe / moe-cache-mib 这两个"每模型不同"的参数要写进 INI 各自小节，
  // 不能再从环境变量传（CLI 优先级高于 preset，会把两个模型配成同一套）。
  const routerPreset = penv('ROUTER_PRESET') || ''
  const routerModelsMax = penv('ROUTER_MODELS_MAX') || ''
  const routerMemoryMargin = penv('ROUTER_MEMORY_MARGIN') || ''
  // 思考预算: 仅当 >0 时追加 --reasoning-budget; 0 或空 = 关闭(无限思考)。
  const reasoningBudget = penv('REASONING_BUDGET')
    || (profile === '__legacy__' && (config && config.llamaReasoningBudget)) || DEFAULT_REASONING_BUDGET
  // 预算耗尽时插入的收尾提示(在强制 </think> 之前)。留空 = 不插入(llama-server 默认 none)。
  // 用中文短句，避免强制收尾那一刻把上下文推回英文。
  const reasoningBudgetMessage = penv('REASONING_BUDGET_MESSAGE') || ''
  // 思考呈现格式: 非空才追加 --reasoning-format(见常量处说明; deepseek-legacy 让前端可见思维链)。
  const reasoningFormat = penv('REASONING_FORMAT') || DEFAULT_REASONING_FORMAT
  // 自定义 chat 模板文件: 非空才追加 --chat-template-file。用于注入"中文思考示范"等前缀改造。
  const chatTemplate = penv('CHAT_TEMPLATE') || ''
  // 历史思考保留: 默认 llama-server 会把历史每条 assistant 的 <think> 都渲染回 prompt(费 token,
  // 且会让模型模仿历史思考的"语言/风格", 淹没模板里的中文示范)。设 1 时追加 --no-reasoning-preserve。
  const noReasoningPreserve = String(penv('NO_REASONING_PRESERVE') || '') === '1'
  // 投机参数自调优（仅 ik）：ε-greedy 在线调 n_max，日志里输出 Autotune reuse: ...
  const specAutotune = String(penv('SPEC_AUTOTUNE') || '') === '1'
  // 引擎：只用自编 upstream b12244（router 模式）。
  // ik_llama.cpp 支持已于 2026-11-16 移除：实测 decode 24.1 vs 27.3、prefill 513 vs 950，
  // 且它的 cpu_share 是单一标量，无法复现「解码就地算 / 预填充上传」的分档配置。
  // 下面的 ik 分支保留但不可达（ik 恒为 false），删它们对生产无收益、有改坏风险。
  const engineReq = String(getenv('LLAMA_ENGINE') || 'upstream').toLowerCase()
  if (engineReq === 'ik') dbg('LLAMA_ENGINE=ik 已不再支持（2026-11-16 移除 ik 引擎），按 upstream 启动')
  const engine = 'upstream'

  return { engine, specAutotune, moeCachePolicy, moeCacheCpuSharePrefill, profile, model, mmproj, mmprojOffload, port, draft, specType, ctxSize, ngl, threads, tb, temp, topP, cpuMoe, nCpuMoe, moeCacheMib, loadMode, batchSize, ubatchSize, ctk, ctv, np, nmax, reasoningBudget, reasoningBudgetMessage, reasoningFormat, chatTemplate, noReasoningPreserve, repeatPenalty, dryMultiplier, dryBase, dryAllowedLength, dryPenaltyLastN, repeatLastN, minP, topK, specDraftPMin, ctxCheckpoints, cacheRam, cacheIdleSlots, logVerbosity, imageMaxTokens, imageMinTokens, logitBias, moeCacheCpuShare, routerPreset, routerModelsMax, routerMemoryMargin }
}

// 由解析后的参数构造 llama-server 启动参数数组。
function buildArgs(p) {
  const ik = p.engine === 'ik'
  // ik_llama.cpp 没有 router（源码里零引用 --models-preset/--models-max），只能单模型。
  // 若引擎选 ik 又配了 ROUTER_PRESET，这里退回单模型加载（-m），并在调试日志说明。
  const router = Boolean(p.routerPreset) && !ik
  if (ik && p.routerPreset) dbg('引擎=ik 不支持 router，已忽略 ROUTER_PRESET，改为单模型加载')
  const args = []
  if (router) {
    // 路由模式：不指定模型，交给 preset 里的多个小节按请求的 model 字段按需加载
    args.push('--models-preset', p.routerPreset)
    // 注意：这里不能用下面的 num()——它在本函数内是后声明的 const，提前用会踩 TDZ
    if (p.routerModelsMax !== '' && Number.isFinite(Number(p.routerModelsMax))) {
      args.push('--models-max', String(p.routerModelsMax))
    }
    if (p.routerMemoryMargin !== '' && Number.isFinite(Number(p.routerMemoryMargin))) {
      args.push('--models-memory-margin', String(p.routerMemoryMargin))
    }
  } else {
    args.push('-m', p.model)
  }
  args.push(
    '-c', p.ctxSize,
    // 槽位数: 1 = 单对话独占 ctx 预算(本地单人使用的最优解)
    '-np', p.np,
    // Flash Attention 启用 (显著提升推理速度)
    '-fa', 'on',
    // KV cache 量化(可配): 默认 q4_0, 比 q8_0 省一半显存; 需要更高精度可设 CTK/CTV=q8_0
    '--cache-type-k', p.ctk,
    '--cache-type-v', p.ctv,
    // 多线程推理 (CPU 线程数; MoE 专家在 CPU 上跑时可调大)
    '-t', p.threads,
    // 批处理大小 (提升长 prompt 预填充吞吐量; CPU+GPU 混合 MoE 对此极敏感)。
    // 9B dense 全 GPU 用 2048 安全且更快; 35B-A3B cpu-moe 下 2048 使预填充翻倍(326→665 tok/s),
    // 再大(4096)显存逼近 8GB 易 OOM。可用 LLAMA_<P>_BATCH / LLAMA_<P>_UBATCH 覆盖。
    '-b', p.batchSize,
    '-ub', ik ? (p.ubatchIk || p.ubatchSize) : p.ubatchSize,
    // 加载模式(可配): 默认 mmap+mlock 把模型锁在物理内存, 避免被换出造成偶发卡顿;
    // 9B dense 用默认即可。MoE 大模型(cpu-moe)须设 LLAMA_35B_LOAD_MODE=none ——
    // mmap 的按需分页会让 CPU 专家权重频繁缺页, 拖慢解码(实测 15.8→22.8 tok/s 的关键)。
    // ik 没有 -lm / --prio：等价的“不 mmap + 锁内存”是 -nmm --mlock
    ...(ik ? ['-nmm', '--mlock'] : [
      '-lm', p.loadMode,
      // 进程优先级提到 high(2), 减少调度抖动。不用 realtime(3): 风险高、收益低。
      '--prio', '2']),
    // 显式绑定端口与回环地址 (与 cordis provider baseURL 对齐;
    // 不依赖 llama-server 默认值, 避免默认 host 变化导致外部可访问)
    '--port', String(p.port),
    '--host', '127.0.0.1',
  )
  // ik 的 --jinja 默认是关的（上游已默认开）。不开的话任何带 tools 的请求都会 500：
  //   {"error":{"code":500,"message":"tools param requires --jinja flag"}}
  if (ik) args.push('--jinja')
  // 批处理线程数(-tb, 可配): sweep 单测曾显示 -tb 8 解码 +16%, 但 3 次复测均值 22.5≈基线, 增益未复现(测量噪声)。
  // 故默认不追加; 设 LLAMA_<P>_TB=8 可自测。0/空=不追加(用 llama 默认= n_threads)。
  if (p.tb && Number(p.tb) > 0) args.push('-tb', String(p.tb))
  if (p.mmproj && fs.existsSync(p.mmproj)) {
    args.push('--mmproj', p.mmproj)
    // 视觉编码器卸载开关: '0' = 挪到 CPU, 释放 ~900MB 显存(可转投专家缓存, 实测解码 +19.7%)
    if (p.mmprojOffload === '0') args.push('--no-mmproj-offload')
  }
  // GPU 层数: 非负整数 = 强制指定卸载层数; 'auto' 或空 = 不追加 -ngl,
  // 让 llama.cpp 按空闲显存自动适配。
  // ⚠ 显式给 -ngl 时必须同时关掉 --fit, 否则日志会报
  //   "failed to fit params to free device memory: n_gpu_layers already set by user to 99, abort"
  //   (只是警告不致命, 但说明 fit 逻辑在和我们抢参数, 关掉更可预测)。
  if (p.ngl && p.ngl !== 'auto') {
    args.push('-ngl', p.ngl)
    // ik 的 --fit 是裸开关（没有 -fit 短名），且源码里 --fit 与 --moe-cache-mib 互斥
    // （common.cpp: "error: --fit cannot be combined with --moe-cache-mib"），所以 ik 一律不传。
    if (!ik) args.push('-fit', 'off')
  }
  // MoE / CPU 卸载: 大模型(如 Qwen3.5-35B-A3B)放不进 8GB 显存时, 把专家权重留在 CPU。
  // 三项都可独立开关; dense 模型(9B)设空即可, 不会追加任何参数。
  if (!router) {
    if (p.cpuMoe === '1') args.push('--cpu-moe')          // 全部专家权重留 CPU
    const nCpuMoeValue = ik ? (p.nCpuMoeIk || p.nCpuMoe) : p.nCpuMoe
    if (nCpuMoeValue) args.push('--n-cpu-moe', nCpuMoeValue)     // 仅前 N 层专家留 CPU
    const cacheMib = ik ? (p.moeCacheMibIk || p.moeCacheMib) : p.moeCacheMib
    if (cacheMib) args.push('--moe-cache-mib', cacheMib) // 显存里缓存热专家(MiB)
  }
  // 投机解码: 0.8B 草稿提议 → 主模型把 N 个候选放进同一批验证。
  // 关键收益: 权重每批只读一次而非 N 次, 因此带宽受限场景同样有效。
  // 草稿文件不存在时自动跳过, 便于回退。
  if (p.draft && fs.existsSync(p.draft) && !ik) {
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
    if (ik) {
      // ik 的写法是 `--spec-type <type>:k=v,k=v`，没有 --spec-draft-* / --spec-ngram-* 这一族。
      // n_max 与 p_min 都塞进同一个字符串；--spec-autotune 会让它在线自调（推荐值在日志里）。
      const parts = [`n_max=${p.nmax}`]
      if (p.specDraftPMin !== '' && p.specDraftPMin !== undefined && Number.isFinite(Number(p.specDraftPMin))) {
        parts.push(`p_min=${Number(p.specDraftPMin)}`)
      }
      // 类型名映射：上游叫 draft-mtp，ik 叫 mtp（ik 的类型表：none/draft/dflash/dspark/mtp/
      // ngram-cache/ngram-simple/ngram-map-k/... ）
      const ikType = p.specType === 'draft-mtp' ? 'mtp' : p.specType
      args.push('--spec-type', `${ikType}:${parts.join(',')}`)
      if (p.specAutotune) args.push('--spec-autotune')
    } else {
      args.push(
        '--spec-type', p.specType,
        '--spec-draft-n-max', p.nmax,        // 每轮最多验证 N+1 个 token(默认 3; 见 DEFAULT_NMAX 注释)
        '--spec-ngram-simple-size-n', '12', // 查找用 n-gram 长度(默认 12)
        '--spec-ngram-simple-size-m', '32', // 每次起草长度(默认 48, 收保守些)
        '--spec-ngram-simple-min-hits', '2' // 至少命中 2 次才提议, 提高接受率
      )
    }
  }
  // 思考预算: 0/空则不追加(无限思考); >0 才限制单次生成的思考 token。
  // 思考预算: 显式设 0 = 立即结束思考(必须真的把 --reasoning-budget 0 传给 llama-server,
  //   否则它用默认 -1=无限思考, 表现为"卡半天/思考吃光token/无答案")。
  //   空串/未设 = 不追加该参数(交回 llama-server 默认)。注意用 >=0 而非 >0。
  if (p.reasoningBudget !== '' && p.reasoningBudget !== undefined && Number.isFinite(Number(p.reasoningBudget)) && Number(p.reasoningBudget) >= 0) {
    args.push('--reasoning-budget', String(p.reasoningBudget))
  }
  // 预算耗尽时在 </think> 之前插入的收尾提示。中文短句可把"最后的思考"从英文拉回中文。
  if (p.reasoningBudgetMessage) args.push('--reasoning-budget-message', p.reasoningBudgetMessage)
  // 思考呈现格式: deepseek=放进 reasoning_content(多数前端不显示 ⇒ 看不到思维链);
  //   deepseek-legacy=把 <think> 留在 content ⇒ 前端可直接显示思维链。空串则不追加。
  if (p.reasoningFormat) args.push('--reasoning-format', p.reasoningFormat)
  // 采样默认值: 只影响"请求里没写 temperature 的调用"(dsh 的 provider 就不写)。
  // 设了才会覆盖 llama-server 默认 1.0, 避免 agent 长链路在工具调用边界跑偏。
  // 温度来源：上游 router 模式由 preset 按模型给（TEMP 留空）；ik 无 preset，用 TEMP_IK 兜底
  const tempValue = ik ? (p.tempIk || p.temp) : p.temp
  if (tempValue !== '' && tempValue !== undefined && Number.isFinite(Number(tempValue))) {
    args.push('--temp', String(tempValue))
  }
  if (p.topP !== '' && p.topP !== undefined && Number.isFinite(Number(p.topP))) {
    args.push('--top-p', String(p.topP))
  }
  // 自定义 chat 模板(注入中文思考示范等): 留空则用模型内嵌模板。
  if (p.chatTemplate) args.push('--chat-template-file', p.chatTemplate)
  // 不把历史思考渲染进 prompt: 省 token, 且避免模型模仿历史(英文)思考语言。
  if (p.noReasoningPreserve && !ik) args.push('--no-reasoning-preserve')
  // 反重复(默认全关): 长上下文思考容易"同一句话循环"，这几个是社区标准做法。
  const num = (v) => v !== '' && v !== undefined && Number.isFinite(Number(v))
  if (num(p.repeatPenalty)) args.push('--repeat-penalty', String(p.repeatPenalty))
  if (num(p.repeatLastN)) args.push('--repeat-last-n', String(p.repeatLastN))
  // ⚠ DRY 在 ik 上的实现极慢：同一份参数，ik decode 10.6 t/s、上游 22 t/s（实测，见 2026-11-15 日志）。
  // 去掉 DRY 只留 repeat-penalty，ik 立刻回到 28.0 t/s。所以 ik 档位默认不传 DRY；
  // 反重复交给 --repeat-penalty 1.1 + --repeat-last-n 512（这项在两个引擎上都不贵）。
  if (!ik || p.dryIk) {
    if (num(p.dryMultiplier)) args.push('--dry-multiplier', String(p.dryMultiplier))
    if (num(p.dryBase)) args.push('--dry-base', String(p.dryBase))
    if (num(p.dryAllowedLength)) args.push('--dry-allowed-length', String(p.dryAllowedLength))
    if (num(p.dryPenaltyLastN)) args.push('--dry-penalty-last-n', String(p.dryPenaltyLastN))
  }
  if (num(p.minP)) args.push('--min-p', String(p.minP))
  if (num(p.topK)) args.push('--top-k', String(p.topK))
  // ik 的 p_min 已经嵌进 --spec-type 字符串里了，不能再单独传（它没有 --spec-draft-* 这一族）
  if (num(p.specDraftPMin) && !ik) args.push('--spec-draft-p-min', String(p.specDraftPMin))
  if (num(p.ctxCheckpoints)) args.push('--ctx-checkpoints', String(p.ctxCheckpoints))
  if (num(p.cacheRam)) args.push('--cache-ram', String(p.cacheRam))
  // '0' 或 'false' => 关闭新任务时保存空闲槽位(消除预填充停顿)
  // ik 没有 --no-cache-idle-slots（它的空闲槽位行为是内置的），跳过
  if (!ik && (p.cacheIdleSlots === '0' || p.cacheIdleSlots === 'false')) args.push('--no-cache-idle-slots')
  if (num(p.logVerbosity)) args.push(ik ? '--verbosity' : '--log-verbosity', String(p.logVerbosity))
  if (num(p.imageMaxTokens)) args.push('--image-max-tokens', String(p.imageMaxTokens))
  if (num(p.imageMinTokens)) args.push('--image-min-tokens', String(p.imageMinTokens))
  // 默认 logit bias（可选兜底，默认空）。llama.cpp 的 --logit-bias 每次只收一个 token。
  // 仅在 >0 时传参：旧版 llama.cpp(bool b12244 之前)不认识该参数，传 0 会让它启动失败
  const shareValue = ik ? (p.moeCacheCpuShareIk || p.moeCacheCpuShare) : p.moeCacheCpuShare
  if (num(shareValue) && Number(shareValue) > 0) args.push('--moe-cache-cpu-share', String(shareValue))
  // 专家缓存淘汰策略（ik 独有）：classlru 比 lru 更看重“同一层/同一类专家”的复用
  if (ik && p.moeCachePolicy) args.push('--moe-cache-policy', p.moeCachePolicy)
  // 提示批份额：>=0 才传（-1/空 = 沿用解码份额，即旧行为）
  if (!ik && num(p.moeCacheCpuSharePrefill) && Number(p.moeCacheCpuSharePrefill) >= 0) {
    args.push('--moe-cache-cpu-share-prefill', String(p.moeCacheCpuSharePrefill))
  }
  if (p.logitBias) {
    for (const item of String(p.logitBias).split(',').map(s => s.trim()).filter(Boolean)) {
      args.push('--logit-bias', item)
    }
  }
  // 把最终参数打进调试日志：两个引擎的参数名差异大，出问题时先看这一行
  dbg(`args[${p.engine}](${args.length}): ${args.join(' ')}`)
  return args
}

// ───────────────────────── Cordis 插件 ─────────────────────────
export const name = 'llama-server-autostart'

export function apply(ctx, config) {
  dbg('apply called (dual-instance)')
  // 二进制目录按引擎分开：上游用 LLAMA_SERVER_DIR，ik 用 LLAMA_SERVER_DIR_IK
  // （踩过坑：只切 LLAMA_ENGINE 但 LLAMA_SERVER_DIR 还指着 ik 目录时，会用 ik 的二进制跑上游参数，
  //   ik 不认识 --models-preset 直接退出，日志里只有一句 unknown argument）。
  const engineIk = String(getenv('LLAMA_ENGINE') || 'upstream').toLowerCase() === 'ik'
  const dir = (engineIk ? (getenv('LLAMA_SERVER_DIR_IK') || getenv('LLAMA_SERVER_DIR')) : getenv('LLAMA_SERVER_DIR'))
    || (config && config.llamaServerDir) || DEFAULT_DIR
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
    const children = []   // 本次拉起/收养的所有实例: { pid, port, tag }

    // 通过 netstat -ano 找到占用端口的 PID，并确认进程名是 llama-server
    const pidOfPort = (port) => new Promise((resolve) => {
      execFile('netstat.exe', ['-ano', '-p', 'TCP'], { windowsHide: true }, (e, stdout) => {
        if (e || !stdout) return resolve(null)
        // 匹配形如 "  TCP    0.0.0.0:8081   0.0.0.0:0   LISTENING   5692" 的 LISTENING 行
        const re = new RegExp(':' + Number(port) + '\\s.*LISTENING\\s+(\\d+)\\s*$')
        let pid = null
        for (const line of stdout.split('\n')) {
          const m = line.match(re)
          if (m) { pid = m[1]; break }
        }
        if (!pid) return resolve(null)
        execFile('tasklist.exe', ['/FI', 'PID eq ' + pid, '/FO', 'CSV', '/NH'], { windowsHide: true }, (e2, out2) => {
          if (e2 || !out2) return resolve(null)
          const name = String(out2).split(',')[0].replace(/^"/, '').replace(/"$/, '').trim().toLowerCase()
          resolve(name.includes('llama-server') ? Number(pid) : null)
        })
      })
    })

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
          if (up) {
            // 收养已运行的 llama-server（可能是计划任务/上次残留拉起的）：
            // 纳入 children，dsh 退出时 disposer 才能一并清理，否则 32G commit 孤儿驻留。
            // 只收养进程名确为 llama-server 的，避免误杀端口撞车的其他程序。
            try {
              const pid = await pidOfPort(p.port)
              if (pid) {
                children.push({ pid, port: p.port, tag: profile + '(adopted)' })
                dbg('端口 ' + p.port + ' 已有服务, 已收养 pid=' + pid + ' (' + profile + ')')
              } else {
                dbg('端口 ' + p.port + ' 已有服务但无法定位 PID, 跳过 ' + profile)
              }
            } catch (e) { dbg('收养失败: ' + (e && e.message) + ', 跳过 ' + profile) }
            continue
          }
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
