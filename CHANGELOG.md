# Changelog

所有迭代以本地时间记录（UTC+8）。

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
