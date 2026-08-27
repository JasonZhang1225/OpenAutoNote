#!/usr/bin/env python3
"""
OpenAutoNote Installer
Installs the default (Qwen3-ASR cloud) stack, with optional local Whisper extras.
"""

import os
import subprocess
import sys


def main():
    print("=" * 50)
    print("  OpenAutoNote - Dependency Installer")
    print("=" * 50)
    print()

    here = os.path.dirname(os.path.abspath(__file__))
    os.chdir(here)

    extra = None
    if sys.platform == "darwin":
        print("[INFO] Detected macOS")
        print("  [1] Default (Qwen3-ASR 云端转写，推荐)")
        print("  [2] Default + local MLX Whisper")
        choice = input("Your choice (1/2, default 1): ").strip() or "1"
        req_file = "requirements.txt"
        if choice == "2":
            extra = "requirements_mac.txt"
    elif sys.platform == "win32":
        print("[INFO] Detected Windows")
        print("  [1] Default (Qwen3-ASR cloud STT, recommended)")
        print("  [2] Default + local faster-whisper (CPU)")
        print("  [3] Default + local faster-whisper (CUDA)")
        choice = input("Your choice (1/2/3, default 1): ").strip() or "1"
        req_file = "requirements.txt"
        if choice == "2":
            extra = "requirements_win_cpu.txt"
        elif choice == "3":
            extra = "requirements_win_cuda.txt"
    else:
        print(f"[INFO] Detected {sys.platform}")
        req_file = "requirements.txt"

    files = [req_file] + ([extra] if extra else [])
    for req_file in files:
        if not os.path.exists(req_file):
            print(f"[ERROR] Requirements file not found: {req_file}")
            print("Please run this script from python-core/.")
            sys.exit(1)

        cmd = [sys.executable, "-m", "pip", "install", "-r", req_file]
        print(f"[CMD] {' '.join(cmd)}")
        print("-" * 50)
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            print(f"[ERROR] Installation failed with code {e.returncode}")
            sys.exit(1)

    print()
    print("=" * 50)
    print("  Installation complete")
    print("=" * 50)
    print()
    print("Next steps:")
    print("  1. Set DASHSCOPE_API_KEY (see ../README.md or .env.example)")
    print("  2. Run: python main.py")
    print("  3. Open: http://127.0.0.1:8964")
    print()


if __name__ == "__main__":
    main()
