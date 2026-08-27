#!/usr/bin/env python3
"""Cross-platform backend launcher for `tauri dev` and local runs."""

from __future__ import annotations

import os
import sys
from pathlib import Path


def pick_python(root: Path) -> str:
    candidates = []
    if sys.platform == "win32":
        candidates.extend(
            [
                root / "venv" / "Scripts" / "python.exe",
                root / "oan2" / "Scripts" / "python.exe",
            ]
        )
    else:
        candidates.extend(
            [
                root / "venv" / "bin" / "python",
                root / "oan2" / "bin" / "python",
            ]
        )
    for path in candidates:
        if path.is_file() and os.access(path, os.X_OK):
            return str(path)
    return sys.executable


def main() -> None:
    root = Path(__file__).resolve().parent
    script = root / "main.py"
    python = pick_python(root)
    os.chdir(root)
    args = [python, str(script), *sys.argv[1:]]
    print(f"[launch] {' '.join(args)}", flush=True)
    os.execv(python, args)


if __name__ == "__main__":
    main()
