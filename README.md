# OpenAutoNote 📝🤖

<div align="center">

<img src="https://github.com/JasonZhang1225/OpenAutoNote/assets/YOUR_IMAGE_PATH/logo.png" width="120" height="120" alt="OpenAutoNote Logo">

**基于 Tauri + Python 的本地化 AI 视频摘要工具**
<br>
*一键下载 • 极速转写 • 智能总结 • 隐私安全*

[![Release](https://img.shields.io/github/v/release/JasonZhang1225/OpenAutoNote?style=flat-square&color=blue)](https://github.com/JasonZhang1225/OpenAutoNote/releases)
[![Platform](https://img.shields.io/badge/Platform-macOS%20%7C%20Windows-lightgrey?style=flat-square)](https://github.com/JasonZhang1225/OpenAutoNote/releases)
[![Built with Tauri](https://img.shields.io/badge/built%20with-Tauri-orange?style=flat-square)](https://tauri.app/)
[![Backend](https://img.shields.io/badge/Backend-Python%203.10%20%7C%20NiceGUI-yellow?style=flat-square)](https://nicegui.io/)

[📥 下载最新版](https://github.com/JasonZhang1225/OpenAutoNote/releases) | [✨ 功能特性](#-功能特性) | [❓ 常见问题](#-常见问题-faq)

</div>

---

## 📖 简介 (Introduction)

**OpenAutoNote** 是一款现代化的桌面端效率工具，旨在利用本地 AI 算力，帮你快速“读”完视频。

不同于依赖云端 API 的收费工具，OpenAutoNote 运行在你的本地设备上，利用 **NVIDIA GPU** (Windows) 或 **Apple Silicon** (macOS) 进行硬件加速。它能自动下载 B 站或 YouTube 视频，将语音转写为文字，并生成结构化的内容摘要。

**你的数据，永远属于你。** 视频内容和笔记无需上传云端，隐私绝对安全。

## ✨ 核心特性 (Features)

### 🎥 全能视频解析
* **多平台支持**: 完美支持 **Bilibili** (含稍后再看列表、自动去除跟踪参数) 和 **YouTube**。
* **高速下载**: 内置 `aria2` 多线程下载引擎，带宽跑满。
* **音频提取**: 自动提取最佳音质轨道，无需人工干预。

### ⚡️ 本地 AI 引擎
* **极致性能**:
    * **macOS**: 基于 Apple **MLX** 框架深度优化，M1/M2/M3 芯片速度飞快。
    * **Windows**: 提供 **CUDA 加速版** (支持 NVIDIA 显卡) 及 CPU 通用版。
* **模型管理**: 支持在设置中一键下载和切换 AI 模型 (Tiny/Small/Medium/Large)，平衡速度与精度。
* **Faster-Whisper**: 转写速度可达视频时长的 30-50 倍。

### 🖥️ 现代化体验
* **极简界面**: 采用 Tauri v2 构建，轻量级，启动即用。
* **后台管理**: 内置“看门狗”机制和启动页 (Splash Screen)，自动清理残留进程，告别端口冲突。
* **实时进度**: 下载、转写、总结全流程进度条展示。

## 📸 预览 (Screenshots)

| **主界面 (Main UI)** | **AI 摘要结果 (Summary)** |
|:---:|:---:|
| <img src="docs/screenshot_main.png" alt="主界面" width="400"> | <img src="docs/screenshot_summary.png" alt="摘要结果" width="400"> |
| *支持链接解析与本地文件拖拽* | *自动生成时间轴与重点笔记* |

> *(请将您的截图重命名为 `screenshot_main.png` 和 `screenshot_summary.png` 并放入仓库的 `docs` 文件夹中)*

## 📥 下载与安装 (Download)

请前往 [Releases 页面](https://github.com/JasonZhang1225/OpenAutoNote/releases) 下载适合您系统的版本。

### 🪟 Windows 用户
* **NVIDIA 显卡用户 (推荐)**: 请下载 `OpenAutoNote_Windows_cuda_Setup.exe`。
    * *享受 GPU 加速，速度极快。安装包较大 (~700MB) 是正常的。*
* **无独立显卡用户**: 请下载 `OpenAutoNote_Windows_cpu_Setup.exe`。
* *注意：首次运行若被 Windows Defender 拦截，请点击“更多信息” -> “仍要运行”。*

### 🍎 macOS 用户
* 请下载 `OpenAutoNote_..._aarch64.dmg`。
* **仅支持 Apple Silicon (M1/M2/M3/M4) 芯片**。
* *注意：首次打开若提示“无法验证开发者”，请前往“系统设置” -> “隐私与安全性”中点击“仍要打开”。*

## ❓ 常见问题 (FAQ)

### 1. 为什么第一次转写时卡住了？
初次使用某个大小的模型（如 Small 或 Large）时，程序会自动从 Hugging Face 下载模型文件（约 500MB - 2GB）。
* 请确保您的**网络环境畅通**（中国大陆用户可能需要代理）。
* 您可以在“设置”页面提前手动下载所需的模型。

### 2. Windows 版安装时提示 "File error" 或进度条不动？
这通常是因为旧版本进程未完全退出。
* **解决方法**：打开任务管理器，结束所有 `api-server.exe` 或 `open-auto-note.exe` 进程，然后点击安装程序的“重试”即可。

### 3. macOS 版启动闪退？
* 请确保下载的是最新版本 (v1.3.2+)。早期版本可能存在签名或文件名兼容性问题。

## 🛠️ 本地开发 (Development)

如果你是开发者，欢迎贡献代码！

### 环境要求
* Node.js (v20+)
* Rust (最新稳定版)
* Python 3.10
* FFmpeg & Aria2 (需添加到系统 PATH)

### 快速开始
```bash
# 1. 克隆仓库
git clone [https://github.com/JasonZhang1225/OpenAutoNote.git](https://github.com/JasonZhang1225/OpenAutoNote.git)
cd OpenAutoNote

# 2. 准备 Python 后端
cd python-core
python -m venv venv
# Windows:
# venv\Scripts\activate
# pip install -r requirements_win_cpu.txt
# macOS:
# source venv/bin/activate
# pip install -r requirements_mac.txt

# 3. 启动开发模式
cd ..
npm install
npm run tauri dev
