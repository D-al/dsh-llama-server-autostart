#!/usr/bin/env python3
"""frame_limiter —— 推理期间对全部窗口自动降帧，dsh 退出时全部撤销。

用户场景：单 GPU(无核显)，nvidia-smi 聚合 util% 会显示 100%，但分引擎看是
3D 引擎(WorkBuddy/dwm 的 UI 渲染)与 Copy 引擎(llama.cpp MoE 专家换入)竞争。
保证推理的做法是：推理期间对所有抢 3D 的窗口设 EcoQoS(效率模式→Chromium 自动降帧)，
推理停止或 dsh 关闭后撤销全部 EcoQoS。

与 gpu_engine_guard.py 的区别：
  - gpu_engine_guard：单点动作(只挑一个最大 3D pid) + 最小化兜底，用于"Copy 饿死"急救。
  - frame_limiter(本脚本)：全局动作(所有超阈值 3D pid 都降帧) + 只 EcoQoS 不最小化，
    用于"推理全程屏蔽 UI 帧率竞争"。两者可并存，互不冲突(都幂等)。

降帧对象：所有 3D 引擎占用 >= FRAME_3D_MIN 的非系统 pid。
  - SKIP_PIDS 只排 dwm(4)/system(0)——dwm 是桌面合成，不该被降。
  - 不按进程名白名单：用户明确要求"对所有窗口"。
  - 自身 pid(调用者)可用 --exclude-pid 排除(如排除当前对话窗口)。

生命周期(关键)：
  1. 常驻轮询(默认 1.5s)：is_processing() 为 True → 对所有 3D pid 设 EcoQoS。
  2. is_processing() 连续 N 次为 False → 撤销全部已设 pid 的 EcoQoS(不退出，等下轮推理)。
  3. 收到 SIGTERM/SIGINT/父进程退出 → finally 块撤销全部后退出(保证不残留)。
  4. --release-all：独立入口，立即撤销本脚本记录里所有 pid 的 EcoQoS 并退出(用于异常残留兜底)。

状态文件：_frame_limiter_state.json(记录被降帧的 pid)，供 --release-all 与审计使用。
用法：
  python frame_limiter.py --apply            # 常驻守护
  python frame_limiter.py --apply --verbose  # 带采样日志
  python frame_limiter.py --release-all      # 撤销全部降帧并退出
  python frame_limiter.py --selftest         # 自检
"""
import argparse
import ctypes
import json
import os
import signal
import subprocess
import sys
import time
import datetime
from ctypes import wintypes

HERE = os.path.dirname(os.path.abspath(__file__))
LOG = os.path.join(HERE, "_frame_limiter.jsonl")
STATE = os.path.join(HERE, "_frame_limiter_state.json")
NOWIN = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0

SLOTS = "http://127.0.0.1:8081/slots?model=Qwen3.6-35B-A3B-UD-Q4_K_XL"  # router 必须带 model，否则 400 → is_processing 永远 False

# 判据阈值(环境变量可覆盖)
POLL = float(os.environ.get("FL_POLL", "1.5"))
FRAME_3D_MIN = float(os.environ.get("FL_3D_MIN", "5"))     # 3D 占用超过此值算"在渲染"，降帧
IDLE_RELEASE = int(os.environ.get("FL_IDLE_RELEASE", "2"))  # 连续 N 次无推理后释放
# 2026-11-21：粘性释放——WorkBuddy 3D 渲染是突发的(59%→0→59%)，单轮掉到阈值下就
# RELEASE 会让它在 THROTTLE/RELEASE 间横跳，每次释放都满帧抢核心。改为连续 N 轮
# 无 3D 才释放（hysteresis，与 Governor 的"降级靠 hysteresis"同理）。
RELEASE_STREAK = int(os.environ.get("FL_RELEASE_STREAK", "4"))  # 连续 N 轮无 3D 才释放
SKIP_PIDS = {4, 0}                                          # dwm/system 不降
SKIP_NAMES = {"dwm.exe", "system", "registry"}              # 按名再排一次(dwm 子进程 pid 不是 4)

PS_ENGINE = r'''
Get-Counter '\GPU Engine(*)\Utilization Percentage' -ErrorAction SilentlyContinue |
  ForEach-Object { $_.CounterSamples } |
  Where-Object { $_.Path -match 'engtype_3d' } |
  ForEach-Object {
    if ($_.Path -match 'pid_(\d+)') {
      "{0}|3d|{1:N2}" -f $Matches[1], $_.CookedValue
    }
  }
'''


def log(ev, **kw):
    rec = {"ts": datetime.datetime.now().isoformat(timespec="seconds"), "ev": ev, **kw}
    try:
        with open(LOG, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    except Exception:
        pass


def save_state(pids):
    try:
        with open(STATE, "w", encoding="utf-8") as f:
            json.dump({"pids": sorted(pids), "updated": datetime.datetime.now().isoformat(timespec="seconds")}, f)
    except Exception:
        pass


def load_state():
    try:
        with open(STATE, encoding="utf-8") as f:
            return set(json.load(f).get("pids", []))
    except Exception:
        return set()


# ── Windows EcoQoS(ProcessPowerThrottling=4, EXECUTION_SPEED=1) ──
class PPT_STATE(ctypes.Structure):
    _fields_ = [("Version", wintypes.DWORD), ("ControlType", wintypes.DWORD), ("ExecutionSpeed", wintypes.BYTE)]


def _k32():
    return ctypes.windll.kernel32


def eco_qos(pid, on):
    try:
        k = _k32()
        h = k.OpenProcess(0x0300, False, pid)
        if not h:
            return False, ctypes.get_last_error()
        s = PPT_STATE(1, 1, 1 if on else 0)
        ok = bool(k.SetProcessInformation(h, 4, ctypes.byref(s), ctypes.sizeof(s)))
        err = 0 if ok else ctypes.get_last_error()
        k.CloseHandle(h)
        return ok, err
    except Exception as e:
        return False, str(e)[:80]


def pid_name(pid):
    try:
        r = subprocess.run(["powershell", "-NoProfile", "-Command",
                            f"(Get-Process -Id {pid} -ErrorAction SilentlyContinue).ProcessName"],
                           capture_output=True, text=True, timeout=3, creationflags=NOWIN)
        return r.stdout.strip() or "?"
    except Exception:
        return "?"


def sample_3d():
    """返回 {pid: 3d%}，只保留超阈值且非系统的 pid。"""
    try:
        r = subprocess.run(["powershell", "-NoProfile", "-Command", PS_ENGINE],
                           capture_output=True, text=True, timeout=10, creationflags=NOWIN)
    except Exception as e:
        log("sample_err", err=str(e)[:120])
        return {}
    out = {}
    for line in r.stdout.splitlines():
        parts = line.strip().split("|")
        if len(parts) != 3:
            continue
        try:
            pid, val = int(parts[0]), float(parts[2])
        except ValueError:
            continue
        if pid in SKIP_PIDS or val < FRAME_3D_MIN:
            continue
        if pid_name(pid).lower() in SKIP_NAMES:
            continue
        out[pid] = max(out.get(pid, 0.0), val)
    return out


def is_processing():
    """llama-server router /slots 是否有实例正在推理。"""
    import urllib.request
    try:
        with urllib.request.urlopen(SLOTS, timeout=2) as r:
            slots = json.loads(r.read().decode("utf-8"))
        return any(bool(s.get("is_processing")) for s in slots)
    except Exception:
        return False


def release_all(active, reason):
    """撤销所有已降帧 pid 的 EcoQoS。"""
    n = 0
    for pid in list(active):
        ok, err = eco_qos(pid, False)
        log("RELEASE", pid=pid, name=pid_name(pid), ok=ok, err=str(err)[:60], reason=reason)
        n += 1
    active.clear()
    save_state(active)
    log("release_all_done", count=n, reason=reason)
    return n


def pid_alive(pid):
    """父进程存活检查(Windows/OpenOS 通用)。"""
    if not pid or pid <= 0:
        return True
    try:
        if os.name == "nt":
            SYNCHRONIZE = 0x00100000
            h = _k32().OpenProcess(SYNCHRONIZE, False, pid)
            if not h:
                return False
            k32 = _k32()
            k32.WaitForSingleObject.restype = wintypes.DWORD
            k32.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
            exited = k32.WaitForSingleObject(h, 0) == 0  # WAIT_OBJECT_0
            k32.CloseHandle(h)
            return not exited
        os.kill(pid, 0)
        return True
    except Exception:
        return False


def run(args):
    # 启动即释放状态文件里的任何残留 pid——防止上次被 Force kill 留下的 EcoQoS。
    # 本守护每次都从干净状态开始；fresh 参数只是显式表明这一点。
    stale = load_state()
    if stale:
        log("stale_state_found", pids=sorted(stale))
        release_all(stale, "startup_cleanup")
    active = set()
    absent = {}   # pid -> 连续无 3D 轮数（粘性释放，避免突发渲染导致横跳）
    idle = 0
    stop = {"flag": False}

    def on_signal(signum, _frame):
        stop["flag"] = True

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            signal.signal(sig, on_signal)
        except Exception:
            pass

    log("start", poll=POLL, min_3d=FRAME_3D_MIN, idle_release=IDLE_RELEASE,
        exclude=args.exclude_pid, fresh=args.fresh)
    try:
        while not stop["flag"]:
            try:
                # 父进程(dsh/autostart)已死 → 自动释放全部降帧并退出。
                # 这是"关掉 dsh 就结束限帧"的最后一道保险：即使 dispose 漏了、
                # 或者我们被 orphan 化，也能自愈，不残留 EcoQoS。
                if args.parent_pid and not pid_alive(args.parent_pid):
                    log("parent_gone", parent=args.parent_pid)
                    stop["flag"] = True
                    break
                proc = is_processing()
                if proc:
                    idle = 0
                    cur = sample_3d()
                    for pid, val in cur.items():
                        if pid in cur and pid in active:
                            absent[pid] = 0   # 又在渲染 → 清零缺席计数（保持限帧）
                            continue
                        if pid == args.exclude_pid:
                            continue
                        ok, err = eco_qos(pid, True)
                        if ok:
                            active.add(pid)
                            absent[pid] = 0
                            save_state(active)
                            log("THROTTLE", pid=pid, name=pid_name(pid), eng3d=round(val, 1))
                        elif args.verbose:
                            log("throttle_fail", pid=pid, err=str(err)[:60])
                    # 粘性释放：连续 RELEASE_STREAK 轮无 3D 才释放（避免突发渲染横跳）
                    for pid in list(active):
                        if pid in cur:
                            absent[pid] = 0
                        else:
                            absent[pid] = absent.get(pid, 0) + 1
                            if absent[pid] >= RELEASE_STREAK:
                                eco_qos(pid, False)
                                active.discard(pid)
                                absent.pop(pid, None)
                                save_state(active)
                                log("RELEASE", pid=pid, reason=f"no_3d_{absent[pid]}streak")
                    if args.verbose:
                        log("sample", proc=True, active=len(active), pids3d=len(cur))
                else:
                    idle += 1
                    if idle >= IDLE_RELEASE and active:
                        release_all(active, "idle")
                    if args.verbose and idle <= 2:
                        log("sample", proc=False, active=len(active))
                time.sleep(POLL)
            except KeyboardInterrupt:
                break
            except Exception as e:
                log("loop_err", err=str(e)[:160])
                time.sleep(POLL)
    finally:
        # 关键保证：任何退出路径都撤销全部降帧(dsh 关闭 → guard 被 kill → 这里执行)
        release_all(active, "exit")
        log("stop")


def selftest():
    print("== frame_limiter selftest ==")
    print("[1] is_processing():", is_processing())
    s = sample_3d()
    print(f"[2] sample_3d(): {len(s)} pid(s) over {FRAME_3D_MIN}%:", dict(list(s.items())[:5]))
    ok, err = eco_qos(os.getpid(), True)
    print(f"[3] EcoQoS on self pid={os.getpid()}: ok={ok} err={err}")
    if ok:
        ok2, _ = eco_qos(os.getpid(), False)
        print(f"    off: ok={ok2}")
    print("[4] should skip dwm(4):", 4 in SKIP_PIDS)
    print("selftest done")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--apply", action="store_true", help="常驻守护：推理期间全局降帧")
    g.add_argument("--release-all", action="store_true", help="撤销全部降帧并退出")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--fresh", action="store_true", help="忽略状态文件，从零开始")
    ap.add_argument("--exclude-pid", type=int, default=0, help="排除指定 pid(如当前对话窗口)")
    ap.add_argument("--parent-pid", type=int, default=0, help="父进程 pid；父进程退出时自动释放全部降帧并退出")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        selftest()
    elif args.release_all:
        n = release_all(load_state(), "manual")
        print(f"released {n} pid(s)")
    elif args.apply:
        run(args)
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
