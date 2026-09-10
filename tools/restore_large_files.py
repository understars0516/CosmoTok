#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
restore_large_files.py — reassemble files that were split by split_large_files.py.

For every file ending in `.manifest.json` found (recursively, defaults to
repo root), reconstruct `original` from its `parts`, then verify:
  - byte size == total_size
  - sha256 matches manifest sha256

After a fresh clone/download of the repository, run:

    python3 tools/restore_large_files.py --root .

This will regenerate e.g. `weights/best_cosmogrid_adv.ckpt` from its parts.

Options:
    --root DIR        directory to walk for *.manifest.json (default: repo root)
    --max-parts N     safety limit on number of parts per file (default 10_000)
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def restore(manifest_path: Path) -> bool:
    with open(manifest_path, "r", encoding="utf-8") as mf:
        manifest = json.load(mf)
    original_rel = manifest["original"]
    dest = manifest_path.parent / original_rel
    total_size = manifest["total_size"]
    num_parts = manifest["num_parts"]

    print(f"[restore] {dst_str(dest)} <- {num_parts} parts from {manifest_path.name}")

    with open(dest, "wb") as out:
        for part in manifest["parts"]:
            part_path = manifest_path.parent / part["file"]
            if not part_path.is_file():
                print(f"    [ERROR] missing part: {part_path}", file=sys.stderr)
                return False
            out.write(part_path.read_bytes())

    ok_size = dest.stat().st_size == total_size
    ok_hash = sha256_file(dest) == manifest["sha256"]
    if ok_size and ok_hash:
        print(f"    OK  size={total_size} sha256 match")
        return True
    print(f"    FAIL size_ok={ok_size} hash_ok={ok_hash}", file=sys.stderr)
    return False


def dst_str(dest: Path) -> str:
    return str(dest).replace("\\", "/")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".", help="root dir to walk (default: .)")
    args = ap.parse_args()
    root = Path(args.root)
    manifests = sorted(root.rglob("*.manifest.json"))
    if not manifests:
        print("[restore] no *.manifest.json found under", root)
        return 0
    all_ok = True
    for mp in manifests:
        if not restore(mp):
            all_ok = False
    print("\n[restore] done:", "ALL OK" if all_ok else "SOME FILES FAILED")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())