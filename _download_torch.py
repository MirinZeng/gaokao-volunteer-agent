"""下载 torch cu126 轮子到 D 盘：可续传、可监控、卡住自动换源。

下载完用 pip install <本地文件> 安装，避免 pip 黑盒下载卡死。
"""

import sys
import time
from pathlib import Path

import requests

sys.stdout.reconfigure(encoding="utf-8")

WHEEL = "torch-2.9.1%2Bcu126-cp312-cp312-win_amd64.whl"
DEST_DIR = Path(r"D:\torch_wheel")
DEST = DEST_DIR / "torch-2.9.1+cu126-cp312-cp312-win_amd64.whl"
PART = DEST.with_suffix(".whl.part")

SOURCES = [
    ("上交大镜像", f"https://mirror.sjtu.edu.cn/pytorch-wheels/cu126/{WHEEL}"),
    ("pytorch官方", f"https://download.pytorch.org/whl/cu126/{WHEEL}"),
    ("阿里云镜像", f"https://mirrors.aliyun.com/pytorch-wheels/cu126/{WHEEL}"),
]

STALL_SECONDS = 45

session = requests.Session()
session.trust_env = False  # 强制直连，绕开注册表里的 Clash 代理

DEST_DIR.mkdir(parents=True, exist_ok=True)


def probe_size(url: str) -> int:
    resp = session.head(url, timeout=30, allow_redirects=True)
    resp.raise_for_status()
    return int(resp.headers.get("Content-Length") or 0)


def try_source(name: str, url: str, expected: int) -> bool:
    done = PART.stat().st_size if PART.exists() else 0
    if expected and done >= expected:
        return True
    headers = {"Range": f"bytes={done}-"} if done else {}
    print(f"[{name}] 从 {done / 1024 / 1024:,.1f} MB 处开始", flush=True)
    try:
        with session.get(url, headers=headers, stream=True, timeout=(20, 60)) as resp:
            if done and resp.status_code == 200:
                print(f"[{name}] 服务端不支持续传，从头开始", flush=True)
                done = 0
                PART.unlink(missing_ok=True)
            resp.raise_for_status()
            if not expected:
                expected = int(resp.headers.get("Content-Length") or 0) + done
            mode = "ab" if done else "wb"
            written = done
            window_start, window_bytes, last_print = time.time(), done, 0.0
            with PART.open(mode) as handle:
                for chunk in resp.iter_content(1024 * 512):
                    handle.write(chunk)
                    written += len(chunk)
                    now = time.time()
                    if now - last_print >= 5:
                        speed = (written - window_bytes) / (now - window_start) / 1024 / 1024
                        pct = f"{written / expected * 100:5.1f}%" if expected else "  ?  "
                        print(
                            f"[{name}] {written / 1024 / 1024:,.1f}"
                            f"/{expected / 1024 / 1024:,.1f} MB {pct} {speed:.2f} MB/s",
                            flush=True,
                        )
                        window_start, window_bytes, last_print = now, written, now
                    if now - window_start > STALL_SECONDS and written == window_bytes:
                        print(f"[{name}] 停滞 {STALL_SECONDS}s，换源", flush=True)
                        return False
    except Exception as error:  # noqa: BLE001
        print(f"[{name}] 出错：{type(error).__name__}: {error}", flush=True)
        return False
    if expected and written < expected:
        print(f"[{name}] 不完整 {written}/{expected}，换源续传", flush=True)
        return False
    return True


def main() -> None:
    print(f"目标：{DEST}", flush=True)
    expected = 0
    for name, url in SOURCES:
        try:
            expected = probe_size(url)
            print(f"探测到大小：{expected / 1024 / 1024:,.1f} MB（{name}）", flush=True)
            break
        except Exception as error:  # noqa: BLE001
            print(f"探测 {name} 失败：{error}", flush=True)

    started = time.time()
    for name, url in SOURCES:
        if try_source(name, url, expected):
            PART.replace(DEST)
            size = DEST.stat().st_size
            print("-" * 60, flush=True)
            print(
                f"下载完成：{size / 1024 / 1024:,.1f} MB，用时 {(time.time() - started) / 60:.1f} 分钟",
                flush=True,
            )
            print(f"文件：{DEST}", flush=True)
            return
    print("所有源均失败", flush=True)
    raise SystemExit(1)


if __name__ == "__main__":
    main()
