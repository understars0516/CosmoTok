#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
split_large_files.py — split files larger than GitHub's 100 MB per-file limit
into smaller parts so they can be committed to the repository.

Each oversized file F is split into chunks of <max_bytes>. Output parts are:
    F.part000, F.part001, ...
plus a manifest file `F.manifest.json` containing, for each part:
  - name, byte_size, sha256
and the overall file's total_size + sha256 (for verification after restore).

Usage:
    python3 tools/split_large_files.py [files...] [--max-bytes 94371840]
    # default --max-bytes = 90 MiB (comfortably under GitHub's 100 MB hard limit)
"""

import argparse
import hashlib
import json
import os
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


def split_file(path: Path, max_bytes: int) -> dict:
    total_size = path.stat().st_size
    print(f"[split] {path.name}: {total_size} bytes -> chunks of {max_bytes}")
    parts = []
    idx = 0
    with open(path, "rb") as f:
        while True:
            chunk = f.read(max_bytes)
            if not chunk:
                break
            part_name = f"{path.name}.part{idx:03d}"
            part_path = Path(f"{path}.part{idx:03d}")
            with open(part_path, "wb") as pf:
                pf.write(chunk)
            parts.append(
                {
                    "name": part_name,
                    "file": part_path.name,
                    "byte_size": len(chunk),
                    "sha256": hashlib.sha256(chunk).hexdigest(),
                }
            )
            print(f"    -> {part_name} ({len(chunk)} bytes)")
            idx += 1
    manifest = {
        "original": path.name,
        "total_size": total_size,
        "num_parts": len(parts),
        "sha256": sha256_file(path),
        "parts": parts,
    }
    manifest_path = Path(f"{path}.manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as mf:
        json.dump(manifest, mf, indent=2)
    print(f"[split] manifest written: {manifest_path}")
    return manifest


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+", help="large files to split")
    ap.add_argument("--max-bytes", type=int, default=90 * 1024 * 1024,
                    help="max bytes per part (default 90 MiB)")
    args = ap.parse_args()
    for f in args.files:
        p = Path(f)
        if not p.is_file():
            print(f"[skip] not a file: {p}", file=sys.stderr)
            continue
        s = p.stat().st_size
        if s <= args.max_bytes:
            print(f"[skip] {p}: {s} bytes already under limit")
            continue
        split_file(p, args.max_bytes)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())