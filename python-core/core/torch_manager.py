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


def get_torch_cuda_package(cuda_version: str, use_mirror: bool = True) -> Tuple[str, str]:
    """
    Get the appropriate PyTorch package specification for the given CUDA version.
    Returns (package_spec, index_url) tuple.
    
    Args:
        cuda_version: CUDA version string (e.g., "11.8", "12.1", "12.6")
        use_mirror: Whether to use NJU mirror first
    
    Returns:
        Tuple of (package_spec, index_url)
    """
    cuda_major_minor = cuda_version.split('.')[:2]
    
    # Map CUDA versions to PyTorch wheel suffixes and versions
    # Format: (cu_suffix, torch_version)
    # Note: PyTorch version format is torch==X.Y.Z+cu{version}
    # For CUDA 12.6, use cu126 suffix, torch version depends on available builds
    cuda_to_torch = {
        "12.6": ("cu126", "2.1.2+cu121"),  # Use cu126 wheel path, but torch version may vary
        "12.5": ("cu121", "2.1.2+cu121"),
        "12.4": ("cu121", "2.1.2+cu121"),
        "12.3": ("cu121", "2.1.2+cu121"),
        "12.2": ("cu121", "2.1.2+cu121"),
        "12.1": ("cu121", "2.1.2+cu121"),
        "12.0": ("cu121", "2.1.2+cu121"),
        "11.8": ("cu118", "2.1.2+cu118"),
        "11.7": ("cu118", "2.1.2+cu118"),
    }
    
    # Determine CUDA wheel suffix and PyTorch version
    if cuda_version in cuda_to_torch:
        cu_suffix, torch_version = cuda_to_torch[cuda_version]
    elif len(cuda_major_minor) >= 2:
        try:
            major = int(cuda_major_minor[0])
            minor = int(cuda_major_minor[1])
            
            if major == 12:
                if minor >= 6:
                    cu_suffix = "cu126"
                    torch_version = "2.1.2+cu121"  # Try cu121 build first for 12.6
                elif minor >= 1:
                    cu_suffix = "cu121"
                    torch_version = "2.1.2+cu121"
                else:
                    cu_suffix = "cu121"
                    torch_version = "2.1.2+cu121"
            elif major == 11:
                cu_suffix = "cu118"
                torch_version = "2.1.2+cu118"
            elif major >= 13:
                # For CUDA 13+, try cu126
                cu_suffix = "cu126"
                torch_version = "2.1.2+cu121"
            else:
                # Default for older versions
                cu_suffix = "cu118"
                torch_version = "2.1.2+cu118"
        except (ValueError, IndexError):
            # Fallback for invalid version format
            cu_suffix = "cu121"
            torch_version = "2.1.2+cu121"
    else:
        cu_suffix = "cu121"
        torch_version = "2.1.2+cu121"
    
    # Choose index URL (mirror first, then official)
    if use_mirror:
        index_url = f"https://mirrors.nju.edu.cn/pytorch/whl/{cu_suffix}"
    else:
        index_url = f"https://download.pytorch.org/whl/{cu_suffix}"
    
    package_spec = f"torch=={torch_version}"
    return package_spec, index_url


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
    Tries NJU mirror first, falls back to official PyTorch index if failed.
    
    Args:
        cuda_version: CUDA version (e.g., "11.8", "12.1", "12.6"). If None, auto-detects.
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
    
    # Get the appropriate package spec (try mirror first)
    package_spec_mirror, index_url_mirror = get_torch_cuda_package(cuda_version, use_mirror=True)
    package_spec_official, index_url_official = get_torch_cuda_package(cuda_version, use_mirror=False)
    
    # Try mirror first
    if progress_callback:
        try:
            progress_callback(f"正在从南京大学镜像下载 {package_spec_mirror}...")
        except Exception:
            pass
    
    success = _try_install(package_spec_mirror, index_url_mirror, progress_callback, source_name="南京大学镜像")
    
    if success:
        return True
    
    # Fall back to official source
    if progress_callback:
        try:
            progress_callback(f"镜像源失败，切换到官方源下载 {package_spec_official}...")
        except Exception:
            pass
    
    success = _try_install(package_spec_official, index_url_official, progress_callback, source_name="PyTorch 官方源")
    
    if success:
        return True
    
    # Both failed
    if progress_callback:
        try:
            progress_callback("安装失败：镜像源和官方源都不可用，请检查网络连接")
        except Exception:
            pass
    return False


def _try_install(
    package_spec: str,
    index_url: str,
    progress_callback: Optional[Callable[[str], None]],
    source_name: str = "源"
) -> bool:
    """
    Try to install PyTorch from a specific index URL.
    Only installs torch (not torchvision/torchaudio) as they're not required for faster-whisper.
    
    Returns:
        True if successful, False otherwise
    """
    try:
        # Build pip install command
        # Only install torch, not torchvision/torchaudio (not needed for faster-whisper)
        cmd = [sys.executable, "-m", "pip", "install", package_spec, "--index-url", index_url]
        
        if progress_callback:
            try:
                progress_callback(f"正在从{source_name}下载并安装 {package_spec}...")
            except Exception:
                pass
        
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
                    progress_callback(f"PyTorch CUDA 安装成功！(使用{source_name})")
                except Exception:
                    pass
            print(f"[TorchManager] Successfully installed {package_spec} from {source_name}")
            return True
        else:
            error_msg = result.stderr if result.stderr else result.stdout
            # Extract useful error information
            error_preview = error_msg[:200] if error_msg else "未知错误"
            if progress_callback:
                try:
                    progress_callback(f"{source_name}下载失败: {error_preview}")
                except Exception:
                    pass
            print(f"[TorchManager] Install failed from {source_name}")
            print(f"[TorchManager] Error output: {error_msg}")
            return False
    except subprocess.TimeoutExpired:
        if progress_callback:
            try:
                progress_callback(f"从{source_name}下载超时（10分钟）")
            except Exception:
                pass
        print(f"[TorchManager] Timeout when installing from {source_name}")
        return False
    except Exception as e:
        if progress_callback:
            try:
                progress_callback(f"从{source_name}下载出错: {str(e)}")
            except Exception:
                pass
        print(f"[TorchManager] Error when installing from {source_name}: {e}")
        return False

