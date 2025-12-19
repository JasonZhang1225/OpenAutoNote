#!/usr/bin/env python3
"""
OpenAutoNote Installer
Helps install the correct dependencies based on your hardware.
"""

import subprocess
import sys
import os

def main():
    print("=" * 50)
    print("  OpenAutoNote - Dependency Installer")
    print("=" * 50)
    print()
    
    # Detect OS
    if sys.platform == "darwin":
        print("[INFO] Detected macOS (Apple Silicon optimized)")
        req_file = "requirements_mac.txt"
    elif sys.platform == "win32":
        print("[INFO] Detected Windows")
        print()
        print("Do you have an NVIDIA GPU with CUDA support?")
        print("  [Y] Yes - Install GPU-accelerated version (~3GB)")
        print("  [N] No  - Install CPU-only version (~200MB)")
        print()
        
        while True:
            choice = input("Your choice (y/n): ").strip().lower()
            if choice in ('y', 'yes'):
                req_file = "requirements_cuda.txt"
                print()
                print("[INFO] Installing CUDA 12.1 GPU version...")
                break
            elif choice in ('n', 'no'):
                req_file = "requirements_wincpu.txt"
                print()
                print("[INFO] Installing CPU-only version...")
                break
            else:
                print("Invalid input. Please enter 'y' or 'n'.")
    else:
        print(f"[INFO] Detected Linux ({sys.platform})")
        print("Using CUDA requirements by default. Edit if needed.")
        req_file = "requirements_cuda.txt"
    
    # Check if requirements file exists
    if not os.path.exists(req_file):
        print(f"[ERROR] Requirements file not found: {req_file}")
        print("Please ensure you are running this script from the project root.")
        sys.exit(1)
    
    print(f"[INFO] Using: {req_file}")
    print()
    
    # Run pip install
    cmd = [sys.executable, "-m", "pip", "install", "-r", req_file]
    print(f"[CMD] {' '.join(cmd)}")
    print("-" * 50)
    
    try:
        subprocess.run(cmd, check=True)
        print()
        print("=" * 50)
        print("  ✅ Installation Complete!")
        print("=" * 50)
        print()
        print("Next steps:")
        print("  1. Run the app: python main.py")
        print("  2. Open browser: http://localhost:8964")
        print()
    except subprocess.CalledProcessError as e:
        print()
        print(f"[ERROR] Installation failed with code {e.returncode}")
        print("Please check the error messages above.")
        sys.exit(1)

if __name__ == "__main__":
    main()
