"""
Torch CUDA Manager - Runtime download and installation of PyTorch CUDA packages
"""
import os
import sys
import subprocess
import platform
from typing import Optional, Tuple, Callable


def detect_cuda_version() -> Optional[str]:
    """
    Detect CUDA version using nvcc or by checking installed CUDA libraries.
    Returns CUDA version string (e.g., "11.8", "12.1") or None if not found.
    """
    try:
        # Method 1: Check nvcc
        result = subprocess.run(
            ["nvcc", "--version"], 
            capture_output=True, 
            text=True, 
            timeout=5
        )
        if result.returncode == 0:
            for line in result.stdout.split('\n'):
                if 'release' in line.lower():
                    # Parse "release 11.8" or similar
                    parts = line.lower().split('release')
                    if len(parts) > 1:
                        version = parts[1].strip().split(',')[0].strip()
                        # Extract major.minor (e.g., "11.8" from "11.8.0")
                        version_parts = version.split('.')
                        if len(version_parts) >= 2:
                            return f"{version_parts[0]}.{version_parts[1]}"
    except (subprocess.TimeoutExpired, FileNotFoundError, subprocess.SubprocessError):
        pass
    
    try:
        # Method 2: Check common CUDA library paths (Windows)
        if platform.system() == "Windows":
            cuda_paths = [
                r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA",
                r"C:\Program Files (x86)\NVIDIA GPU Computing Toolkit\CUDA",
            ]
            for base_path in cuda_paths:
                if os.path.exists(base_path):
                    # Find the highest version directory
                    versions = []
                    for item in os.listdir(base_path):
                        item_path = os.path.join(base_path, item)
                        if os.path.isdir(item_path) and item.replace('.', '').isdigit():
                            try:
                                version_parts = item.split('.')
                                if len(version_parts) >= 2:
                                    versions.append((float(item), item))
                            except ValueError:
                                pass
                    if versions:
                        versions.sort(reverse=True)
                        return versions[0][1]
    except Exception:
        pass
    
    # Method 3: Try importing torch and checking CUDA version if available
    try:
        import torch
        if torch.cuda.is_available():
            cuda_version = torch.version.cuda
            if cuda_version:
                # Extract major.minor (e.g., "11.8" from "11.8.0")
                version_parts = cuda_version.split('.')
                if len(version_parts) >= 2:
                    return f"{version_parts[0]}.{version_parts[1]}"
    except ImportError:
        pass
    
    return None


def get_torch_cuda_package(cuda_version: str) -> str:
    """
    Get the appropriate PyTorch package URL for the given CUDA version.
    Returns pip install command argument.
    """
    cuda_major_minor = cuda_version.split('.')[:2]
    cuda_version_str = ''.join(cuda_major_minor)  # "11.8" -> "118"
    
    # Map CUDA versions to PyTorch index URLs
    cuda_to_torch = {
        "12.1": ("cu121", "2.1.2+cu121"),
        "12.0": ("cu121", "2.1.2+cu121"),  # Use cu121 for 12.0
        "11.8": ("cu118", "2.1.2+cu118"),
        "11.7": ("cu118", "2.1.2+cu118"),  # Use cu118 for 11.7
    }
    
    # Try exact match first
    if cuda_version in cuda_to_torch:
        index_suffix, version = cuda_to_torch[cuda_version]
    elif cuda_major_minor[0] == "12":
        # CUDA 12.x -> use cu121
        index_suffix, version = cuda_to_torch["12.1"]
    elif cuda_major_minor[0] == "11":
        # CUDA 11.x -> use cu118
        index_suffix, version = cuda_to_torch["11.8"]
    else:
        # Default to cu121 for newer versions
        index_suffix, version = cuda_to_torch["12.1"]
    
    index_url = f"https://download.pytorch.org/whl/{index_suffix}"
    return f"torch=={version} --index-url {index_url}"


def check_torch_cuda_installed() -> Tuple[bool, Optional[str]]:
    """
    Check if torch with CUDA support is installed.
    Returns (is_installed, cuda_version_if_available)
    """
    try:
        import torch
        if torch.cuda.is_available():
            cuda_version = torch.version.cuda
            return True, cuda_version
        else:
            return False, None
    except ImportError:
        return False, None


def install_torch_cuda(
    cuda_version: Optional[str] = None,
    progress_callback: Optional[Callable[[str], None]] = None
) -> bool:
    """
    Install PyTorch with CUDA support.
    
    Args:
        cuda_version: CUDA version (e.g., "11.8", "12.1"). If None, auto-detects.
        progress_callback: Optional callback(progress_text) for progress updates.
    
    Returns:
        True if successful, False otherwise
    """
    if progress_callback:
        try:
            progress_callback("检测 CUDA 版本...")
        except Exception:
            pass
    
    # Detect CUDA version if not provided
    if cuda_version is None:
        cuda_version = detect_cuda_version()
        if cuda_version is None:
            if progress_callback:
                try:
                    progress_callback("错误: 未检测到 CUDA，无法安装 PyTorch CUDA 版本")
                except Exception:
                    pass
            return False
    
    if progress_callback:
        try:
            progress_callback(f"检测到 CUDA {cuda_version}，准备下载 PyTorch...")
        except Exception:
            pass
    
    # Get the appropriate package
    package_spec = get_torch_cuda_package(cuda_version)
    parts = package_spec.split(" --index-url ")
    package_name = parts[0]
    index_url = parts[1] if len(parts) > 1 else None
    
    if progress_callback:
        try:
            progress_callback(f"正在下载并安装 {package_name}...")
        except Exception:
            pass
    
    try:
        # Build pip install command
        cmd = [sys.executable, "-m", "pip", "install", package_name]
        if index_url:
            cmd.extend(["--index-url", index_url])
        
        # Run pip install
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600  # 10 minute timeout
        )
        
        if result.returncode == 0:
            if progress_callback:
                try:
                    progress_callback("PyTorch CUDA 安装成功！")
                except Exception:
                    pass
            return True
        else:
            if progress_callback:
                try:
                    progress_callback(f"安装失败: {result.stderr}")
                except Exception:
                    pass
            print(f"[TorchManager] Install failed: {result.stderr}")
            return False
    except subprocess.TimeoutExpired:
        if progress_callback:
            try:
                progress_callback("安装超时，请检查网络连接")
            except Exception:
                pass
        return False
    except Exception as e:
        if progress_callback:
            try:
                progress_callback(f"安装出错: {str(e)}")
            except Exception:
                pass
        print(f"[TorchManager] Install error: {e}")
        return False

