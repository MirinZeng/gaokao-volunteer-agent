"""从 ModelScope 直连下载 BAAI/bge-m3 到 D 盘。

- 强制直连（trust_env=False），绕开注册表里的 Clash 代理：代理无法处理 308 跳转，会卡死在 0 字节
- 跳过 onnx/ 与示例图片，省掉约 2.1 GB
- 支持断点续传（.part + Range），可随时中断重跑
"""

import sys
import time
from pathlib import Path

import requests

sys.stdout.reconfigure(encoding="utf-8")

TARGET = Path(r"D:\models\bge-m3")
REPO = "BAAI/bge-m3"
REVISION = "master"
API = f"https://www.modelscope.cn/api/v1/models/{REPO}/repo/files"
RAW = f"https://www.modelscope.cn/models/{REPO}/resolve/{REVISION}/"

SKIP_PREFIX = ("onnx/", "imgs/")
SKIP_EXT = (".jpg", ".jpeg", ".png", ".webp")

session = requests.Session()
session.trust_env = False  # 直连，忽略环境变量与注册表代理

TARGET.mkdir(parents=True, exist_ok=True)


def fetch_list():
    resp = session.get(API, params={"Revision": REVISION}, timeout=60)
    resp.raise_for_status()
    files = [f for f in resp.json()["Data"]["Files"] if f.get("Type") != "tree"]
    wanted = []
    for item in files:
        path = item["Path"]
        if path.startswith(SKIP_PREFIX) or path.endswith(SKIP_EXT):
            continue
        wanted.append((path, int(item.get("Size") or 0)))
    wanted.sort(key=lambda pair: pair[1])  # 小文件先下，便于快速验证
    return wanted


def download(path: str, size: int) -> None:
    dest = TARGET / path
    dest.parent.mkdir(parents=True, exist_ok=True)
    if size and dest.exists() and dest.stat().st_size == size:
        print(f"  [跳过] {path}  (已存在 {size / 1024 / 1024:,.1f} MB)", flush=True)
        return

    part = dest.parent / (dest.name + ".part")
    for attempt in range(1, 6):
        done = part.stat().st_size if part.exists() else 0
        headers = {"Range": f"bytes={done}-"} if done else {}
        try:
            with session.get(RAW + path, headers=headers, stream=True, timeout=90) as resp:
                if done and resp.status_code == 200:
                    done = 0  # 服务端不支持续传，从头来
                    part.unlink(missing_ok=True)
                resp.raise_for_status()
                mode = "ab" if done else "wb"
                written = done
                last = time.time()
                base = written
                with part.open(mode) as handle:
                    for chunk in resp.iter_content(1024 * 512):
                        handle.write(chunk)
                        written += len(chunk)
                        now = time.time()
                        if now - last >= 5:
                            pct = f"{written / size * 100:5.1f}%" if size else "  ?  "
                            speed = (written - base) / (now - last) / 1024 / 1024
                            print(
                                f"  {path}: {written / 1024 / 1024:,.1f}"
                                f"/{size / 1024 / 1024:,.1f} MB {pct}  {speed:.2f} MB/s",
                                flush=True,
                            )
                            last, base = now, written
            part.replace(dest)
            print(f"  [完成] {path}  ({dest.stat().st_size / 1024 / 1024:,.1f} MB)", flush=True)
            return
        except Exception as error:  # noqa: BLE001
            print(f"  [重试 {attempt}/5] {path}: {type(error).__name__}: {error}", flush=True)
            time.sleep(3)
    raise RuntimeError(f"下载失败：{path}")


def main() -> None:
    print(f"目标目录 : {TARGET}")
    print("代理     : 已禁用（直连 ModelScope）")
    print("获取文件清单 ...", flush=True)
    wanted = fetch_list()
    total = sum(size for _, size in wanted)
    print(f"共 {len(wanted)} 个文件，合计 {total / 1024 / 1024:,.1f} MB", flush=True)
    print("-" * 60, flush=True)

    started = time.time()
    for index, (path, size) in enumerate(wanted, 1):
        print(f"[{index}/{len(wanted)}] {path}", flush=True)
        download(path, size)

    elapsed = time.time() - started
    done_bytes = sum(
        (TARGET / p).stat().st_size for p, _ in wanted if (TARGET / p).exists()
    )
    print("-" * 60, flush=True)
    print(f"全部完成：{done_bytes / 1024 / 1024:,.1f} MB，用时 {elapsed / 60:.1f} 分钟", flush=True)


if __name__ == "__main__":
    main()
