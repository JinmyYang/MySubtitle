"""把要"删除"的东西移进项目垃圾桶，而不是真的删掉。

为什么需要这个脚本：这台机器把所有删除操作重定向到
`genie-trash\\win32-x64.exe`，它在项目路径上会失败弹错（0x800700e8），而且
**每删一个路径就新开一个 Windows Terminal 窗口**——2026-09-11 批量清理约 262 个
空目录，积压了 75 个终端窗口。移动是"改名"、不是删除，完全绕开这个机制。

约定：东西一律移进项目根下第一个 `项目垃圾桶_*` 目录（已在 .gitignore 里按前缀忽略），
由用户自己抽空清空。按批次建子目录，方便整批删。
**目录名按前缀发现、不写死**——用户改过名字，写死会凭空造出第二个垃圾桶。

用法:
    python scripts/to_trash.py results/xxx results/yyy     # 移入今天的批次
    python scripts/to_trash.py --batch 清理旧导出 results/xxx
    python scripts/to_trash.py --dry-run results/xxx      # 只看看会移动什么

安全护栏（缺了任何一条都可能造成事故，别为了省事关掉）:
  - 只接受项目目录**内部**的路径，项目外的一律拒绝
  - 拒绝把垃圾桶自己再搬进垃圾桶
  - 必须是同盘符：跨盘移动会退化成"复制 + 删除"，照样触发 genie-trash
  - 目录里含文件时默认拒绝，必须显式加 --force
  - 只用 os.rename（失败即报错），绝不退化成"复制 + 删除"
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
# 垃圾桶目录按前缀发现，不写死全名——用户改过两次名字，写死必然失联
TRASH_PREFIX = "项目垃圾桶_"
TRASH_FALLBACK = f"{TRASH_PREFIX}不要查看"


def trash_dir() -> Path:
    """定位垃圾桶（项目根下第一个 `项目垃圾桶_*` 目录）；没有就用默认名新建。"""
    found = sorted(p for p in ROOT.glob(f"{TRASH_PREFIX}*") if p.is_dir())
    return found[0] if found else ROOT / TRASH_FALLBACK


def _inside_project(p: Path) -> bool:
    try:
        p.resolve().relative_to(ROOT)
        return True
    except ValueError:
        return False


def _same_volume(a: Path, b: Path) -> bool:
    return os.path.splitdrive(str(a))[0].lower() == os.path.splitdrive(str(b))[0].lower()


def _file_count(p: Path) -> int:
    return sum(1 for f in p.rglob("*") if f.is_file()) if p.is_dir() else 0


def main() -> int:
    ap = argparse.ArgumentParser(description="把要删的东西移进项目垃圾桶（不真删）")
    ap.add_argument("paths", nargs="+", help="要移入垃圾桶的路径（必须在项目内）")
    ap.add_argument("--batch", default="", help="批次子目录名，默认用当天日期")
    ap.add_argument("--force", action="store_true", help="目录含文件时也移动")
    ap.add_argument("--dry-run", action="store_true", help="只打印计划，不移动")
    args = ap.parse_args()

    TRASH_DIR = trash_dir()
    if not _same_volume(ROOT, TRASH_DIR):
        print("[错误] 垃圾桶与项目不在同一盘符：跨盘移动会变成复制+删除，照样弹窗。")
        return 1
    dest_root = TRASH_DIR / (args.batch or time.strftime("%Y-%m-%d"))

    planned: list[Path] = []
    refused: list[tuple[str, str]] = []
    for raw in args.paths:
        p = Path(raw)
        if not p.is_absolute():
            p = ROOT / p
        if not p.exists():
            refused.append((str(p), "不存在"))
        elif not _inside_project(p):
            refused.append((str(p), "在项目外，拒绝"))
        elif p == TRASH_DIR or TRASH_DIR in p.parents:
            refused.append((str(p), "它已经在垃圾桶里了"))
        elif _file_count(p) and not args.force:
            refused.append((str(p), f"目录含 {_file_count(p)} 个文件（要移就加 --force）"))
        else:
            planned.append(p)

    for path, why in refused:
        print(f"[跳过] {path} —— {why}")
    if not planned:
        print("没有可移动的路径。")
        return 1 if refused else 0

    if args.dry_run:
        print(f"[dry-run] 将移动 {len(planned)} 项到 {dest_root}:")
        for p in planned:
            print("   ", p.relative_to(ROOT))
        return 0

    dest_root.mkdir(parents=True, exist_ok=True)
    moved = 0
    for p in planned:
        dest = dest_root / p.name
        if dest.exists():
            dest = dest_root / f"{p.name}_{int(time.time())}"
        try:
            os.rename(p, dest)          # 同盘 = 纯改名；失败就报错，绝不"复制+删除"
        except OSError as exc:
            print(f"[失败] {p}: {exc}")
            continue
        moved += 1
        print(f"[移入] {p.relative_to(ROOT)}")
    print(f"\n完成：移动 {moved} 项到 {dest_root.relative_to(ROOT)}"
          f"（该批次现共 {len(list(dest_root.iterdir()))} 项）")
    print("清空由你自行处理——这个脚本不会删除任何东西。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
