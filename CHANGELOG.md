# Changelog

所有迭代以本地时间记录（UTC+8）。

## [1.2.0] - 2026-11-20 01:05

### Added —— 全局帧率限制（推理期间降帧，dsh 关闭全释放）
- 新增 `frame_limiter.py`（随包发布）：推理中(`/slots` is_processing)对所有 3D 引擎占用 >=5% 的
  非系统窗口设 EcoQoS 效率模式（Chromium 自动降帧），不最小化、不遮挡；推理停止连 2 次后释放。
- `index.mjs` 的 `ctx.effect()` 在 llama-server 全部就绪后 spawn limiter（`--apply --fresh`，
  无窗口 detached）；配置 `FRAME_LIMIT_ENABLED/PY/VERBOSE/EXCLUDE_PID/LOG`，默认开启，
  `FRAME_LIMIT_ENABLED=0` 可关。
- **关闭全释放（三重保险）**：
  ① dispose 先 `SIGTERM` limiter 再同步跑 `--release-all`（Windows 下 kill 不触发 Python finally）；
  ② limiter 带 `--parent-pid`，每轮检测父进程存活，父死自动 `release_all` 退出；
  ③ 启动时清理上次残留的 state（防 Force kill 遗留 EcoQoS）。
- 实测：dsh 重启后 limiter 自动 spawn（pid 13100，`--apply --fresh`）；父进程死亡自愈测试
  `parent_gone → release_all_done → stop` 通过；`--release-all` 成功撤销残留 pid。

### Note
- 背景：单 GPU 无核显，nvidia-smi 聚合 util% 会显示 100%，但分引擎看是 3D(WorkBuddy/dwm UI)
  与 Copy(llama.cpp MoE 换入)竞争；聚合值误导，必须按 pid+engtype 判断。
- `frame_limiter.py` 与 harness 仓库 `dsh-behavior-guard/tools/frame_limiter.py` 同源。

## [1.1.1] - 2026-11-17 21:45

### Added
- 详细 README：问题（启动链路/进程残留/netstat 误判/沙箱限速）、途径、机制与实测
- bundle 包内 `cordis.patch.yml`（**补齐缺失的 bundle overlay**——package.json 声明了 `dsh.bundle.patch` 但文件缺失，dsh 每次启动报 ENOENT 并将插件标记"异常"；2026-11-17 21:20 实测修复）
- 版本戳与 tag（v1.1.1-20261117-2145）

### Fixed
- **版本覆盖事故**（2026-11-17 20:40）：`pnpm update` 用 GitHub 上的旧版覆盖了本地维护的新版（新版从未上传）——llama-server 以旧参数运行（-m 直连模式/无模板文件/q4_0 KV，门禁渲染断言 FAIL 暴露）。修复：从迁移备份恢复真新版 → force push（f388750）→ node_modules 覆盖 → dsh 重启后 router 模式/模板/q8_0 KV 全部正确
- 教训：**本地维护的插件必须先推 GitHub 再 pnpm update**；GitHub 仓库即版本真相源

## [1.1.0] - 2026-11-17 18:45（迁移备份时点）

### Added
- 收养逻辑：8081 已有健康服务时收养而非重复拉起（双重实例 30GB 内存疑云现场修复）
- netstat 端口归属正则修正（此前把无关进程监听误判为 llama）
- 退出清理精确化：只杀本次 spawn 的 pid，收养实例保留
- 参数链路全面化：83 参数 router 模式（--models-preset/--chat-template-file/--cache-type-k q8_0/--fit off/投机解码/DRY/预算）
- CUDA PATH 多候选前置（重装系统免疫）
- 双实例支持（LLAMA_PROFILES 环境变量）

## [1.0.x] - 2026-11 早期

### Added
- 基础版：probe-then-spawn + windowsHide + .env.llama 参数解耦 + 退出清理

[1.1.1]: https://github.com/D-al/dsh-llama-server-autostart/compare/v1.1.0...v1.1.1
[1.1.0]: https://github.com/D-al/dsh-llama-server-autostart/releases/tag/v1.1.0
