import subprocess
import sys
import re
import platform


def detect_cuda_version():
    """Detect installed CUDA version"""
    try:
        result = subprocess.run(
            ["nvcc", "--version"],
            capture_output=True,
            text=True,
            timeout=10
        )
        if result.returncode == 0:
            output = result.stdout
            match = re.search(r"release (\d+\.\d+)", output)
            if match:
                version = match.group(1)
                major = version.split(".")[0]
                minor = version.split(".")[1]
                return f"cu{major}{minor}"
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    try:
        result = subprocess.run(
            ["nvidia-smi"],
            capture_output=True,
            text=True,
            timeout=10
        )
        if result.returncode == 0:
            for line in result.stdout.split('\n'):
                if "CUDA Version" in line:
                    match = re.search(r"CUDA Version:\s*(\d+\.\d+)", line)
                    if match:
                        version = match.group(1)
                        major = version.split(".")[0]
                        minor = version.split(".")[1]
                        return f"cu{major}{minor}"
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    return None


def check_torch_cuda_installed():
    """Check if PyTorch with CUDA is installed"""
    try:
        import torch
        if torch.cuda.is_available():
            return True, torch.version.cuda
        return False, None
    except ImportError:
        return False, None


def get_pytorch_index_url(cuda_version):
    """Get PyTorch index URL for specific CUDA version"""
    cuda_map = {
        "cu118": "cu118",
        "cu121": "cu121",
        "cu124": "cu124",
        "cu125": "cu125",
        "cu126": "cu126",
    }

    cuda_suffix = cuda_map.get(cuda_version.lower(), "cu118")

    nju_mirror = f"https://mirrors.nju.edu.cn/pytorch/whl/{cuda_suffix}"
    official_url = f"https://download.pytorch.org/whl/{cuda_suffix}"

    return nju_mirror, official_url


def install_torch_cuda(cuda_version, progress_callback=None):
    """Install PyTorch with CUDA support"""
    if progress_callback is None:
        def no_op(msg):
            pass
        progress_callback = no_op

    def run_command(cmd, description):
        progress_callback(f"{description}...")
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=600
            )
            if result.returncode != 0:
                progress_callback(f"错误: {result.stderr[:200]}")
                return False, result.stderr
            return True, None
        except subprocess.TimeoutExpired:
            progress_callback("命令超时")
            return False, "timeout"
        except Exception as e:
            progress_callback(f"异常: {str(e)}")
            return False, str(e)

    progress_callback("检测Python版本...")
    python_version = f"{sys.version_info.major}.{sys.version_info.minor}"

    cuda_map = {
        "cu118": ("2.1.0", "2.4.0"),
        "cu121": ("2.1.0", "2.5.0"),
        "cu124": ("2.3.0", None),
        "cu125": ("2.4.0", None),
        "cu126": ("2.5.0", None),
    }

    cuda_info = cuda_map.get(cuda_version.lower())
    if not cuda_info:
        progress_callback(f"不支持的CUDA版本: {cuda_version}")
        return False, f"Unsupported CUDA version: {cuda_version}"

    torch_version_range = cuda_info

    progress_callback(f"开始安装 PyTorch (CUDA {cuda_version})...")

    nju_url, official_url = get_pytorch_index_url(cuda_version)

    install_success = False
    last_error = None

    for attempt, index_url in enumerate([nju_url, official_url]):
        progress_callback(f"尝试 {attempt + 1}: 使用 {'南京大学镜像' if 'nju' in index_url else '官方源'}...")

        uninstall_cmd = [
            sys.executable, "-m", "pip", "uninstall", "-y",
            "torch", "torchvision", "torchaudio"
        ]
        run_command(uninstall_cmd, "卸载旧版本...")

        install_cmd = [
            sys.executable, "-m", "pip", "install",
            "--upgrade",
            "--index-url", index_url,
            f"torch>={torch_version_range[0]}" if torch_version_range[1] is None else f"torch>={torch_version_range[0]},<{torch_version_range[1]}",
            "torchvision",
            "torchaudio",
            "--no-deps"
        ]

        success, error = run_command(install_cmd, "安装PyTorch...")
        if success:
            progress_callback("验证安装...")
            verify_success, _ = check_torch_cuda_installed()
            if verify_success:
                progress_callback("✅ PyTorch CUDA 安装成功！")
                install_success = True
                break
            else:
                progress_callback("安装后验证失败，尝试其他源...")
        else:
            last_error = error
            progress_callback(f"安装失败: {str(error)[:100]}")

    return install_success, last_error


def get_torch_install_status():
    """Get comprehensive torch installation status"""
    status = {
        "torch_installed": False,
        "cuda_available": False,
        "cuda_version": None,
        "torch_version": None,
        "device_name": None,
    }

    try:
        import torch
        status["torch_installed"] = True
        status["torch_version"] = torch.__version__

        if torch.cuda.is_available():
            status["cuda_available"] = True
            status["cuda_version"] = torch.version.cuda
            status["device_name"] = torch.cuda.get_device_name(0)
    except ImportError:
        pass

    return status
