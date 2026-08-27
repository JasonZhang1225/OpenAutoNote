**目前仍偏早期：桌面壳在 Linux 上无法完整复现 macOS 闪退；推荐先用 `python python-core/main.py` 跑通。打包版需要 ffmpeg，以及 `DASHSCOPE_API_KEY`（默认云端转写）。Windows 安装包仍可能有问题。**

# OpenAutoNote

<div align="center">

**基于 Tauri + Python 的 AI 视频笔记工具**
<br>
*下载 • 转写 • 多模态总结*

[![Release](https://img.shields.io/github/v/release/JasonZhang1225/OpenAutoNote?style=flat-square&color=blue)](https://github.com/JasonZhang1225/OpenAutoNote/releases)
[![Platform](https://img.shields.io/badge/Platform-macOS%20%7C%20Windows-lightgrey?style=flat-square)](https://github.com/JasonZhang1225/OpenAutoNote/releases)
[![Built with Tauri](https://img.shields.io/badge/built%20with-Tauri-orange?style=flat-square)](https://tauri.app/)
[![Backend](https://img.shields.io/badge/Backend-Python%203.10%20%7C%20NiceGUI-yellow?style=flat-square)](https://nicegui.io/)

[📥 下载](https://github.com/JasonZhang1225/OpenAutoNote/releases) · [语音转写](#-语音转写-stt) · [FAQ](#-常见问题-faq)

</div>

---

## 简介

OpenAutoNote 从 Bilibili / YouTube 下载视频（yt-dlp + 可选 aria2），把音频转成带时间戳的字幕，再把字幕和抽帧一起交给你配置的 LLM（OpenAI 兼容接口）生成结构化笔记。

默认转写走 **阿里云百炼 Qwen3-ASR**（异步文件转写，最长约 12 小时），不再依赖难装的本地 faster-whisper。视频文件仍在本地处理；转写时会把音频上传到 DashScope 临时存储。

## 核心特性

* **多平台下载**: Bilibili（含稍后再看 / BV 规范化）和 YouTube；检测到 `aria2c` 时走多线程。
* **统一 STT**: `qwen3-asr-flash-filetrans`（默认）/ `qwen3-asr-flash` / 可选本地 Qwen3-ASR / 遗留 faster-whisper。
* **多模态总结**: 字幕 + 关键帧送给 LLM。
* **桌面壳**: Tauri v2 启动页会等待 Python `/health`，sidecar 缺失时回退到系统 Python，失败时留在启动页而不是直接退出。

## 语音转写 (STT)

所有后端实现同一接口：`transcribe(audio_path) -> {text, segments, language, backend}`。

| 后端 id | 用途 | 默认？ |
| --- | --- | --- |
| `qwen3-asr-flash-filetrans` | 阿里云百炼异步文件转写，最长约 12 小时 / 2GB。适合整段视频。 | **是** |
| `qwen3-asr-flash` | 同步短音频，最长约 5 分钟 / 10MB。 | 否 |
| `qwen3-asr-local` | 可选本地 [Qwen3-ASR-1.7B](https://github.com/QwenLM/Qwen3-ASR)。默认不安装权重。 | 否 |
| `faster-whisper` | 本地 Whisper（Windows CUDA/CPU 或 macOS MLX）。遗留路径。 | 否 |

官方接口（不要自己发明）：

* 非实时转写：<https://www.alibabacloud.com/help/en/model-studio/non-realtime-speech-recognition-user-guide>
* Qwen3-ASR：<https://github.com/QwenLM/Qwen3-ASR>
* 临时文件 URL：<https://www.alibabacloud.com/help/en/model-studio/get-temporary-file-url>

`qwen3-asr-flash-filetrans` 只接受公网可访问的 `file_url`。本项目会先用 ffmpeg 抽音（若可用），再走 DashScope `getPolicy` 上传到临时 OSS，请求头带 `X-DashScope-OssResourceResolve: enable`。

### DASHSCOPE_API_KEY

**不要把 Key 写进源码或提交到 git。** 缺少 Key 时会直接报错并提示设置方法。

任选其一（环境变量优先）：

```bash
export DASHSCOPE_API_KEY=sk-...
# 可选：国际站
export DASHSCOPE_REGION=intl
```

或复制 `python-core/.env.example` 为 `python-core/.env`（已 gitignore）：

```
DASHSCOPE_API_KEY=sk-...
DASHSCOPE_REGION=beijing
OPENAUTONOTE_STT_BACKEND=qwen3-asr-flash-filetrans
```

或在应用 **设置 → API**：DashScope API Key / 地域 / 转写后端。

* 北京（默认）: `https://dashscope.aliyuncs.com`
* 国际站: `https://dashscope-intl.aliyuncs.com`（Key 与北京站不同）

申请 Key：<https://help.aliyun.com/zh/model-studio/get-api-key>

LLM 笔记仍用设置里的 OpenAI 兼容 `api_key` / `base_url` / `model_name`，与 DashScope STT Key 分开。

### 本地 Qwen3-ASR（可选）

默认安装**不会**拉取 PyTorch + 1.7B 权重。若要试本地后端：

```bash
pip install transformers accelerate torch torchaudio
# 参考 https://github.com/QwenLM/Qwen3-ASR 与 HuggingFace Qwen/Qwen3-ASR-1.7B
export OPENAUTONOTE_STT_BACKEND=qwen3-asr-local
```

未装齐时该后端会给出安装说明，而不是悄悄失败。

### 遗留 faster-whisper

```bash
# macOS Apple Silicon
pip install -r python-core/requirements_mac.txt
# Windows CPU / CUDA
pip install -r python-core/requirements_win_cpu.txt
pip install -r python-core/requirements_win_cuda.txt
```

设置中把转写后端改为 `faster-whisper`，并选择硬件模式（mlx / cuda / cpu）。

## 下载与安装

发行包见 [Releases](https://github.com/JasonZhang1225/OpenAutoNote/releases)。请同时安装 **ffmpeg**（以及可选 **aria2**）。

### macOS

* Apple Silicon DMG。若提示无法验证开发者：系统设置 → 隐私与安全性 → 仍要打开。
* 若应用停在「正在连接后端」：多半是 sidecar 没起来。用下面的开发方式跑 `python-core/main.py`，确认 `http://127.0.0.1:8964/health` 返回 `{"ok": true}`。
* GitHub issue #1（打开后 Connecting 再闪退）已在代码里按「sidecar 缺失 / 握手超时 / 关掉启动页误杀进程」处理。Linux 上无法复现原生 macOS `.app` 崩溃。

### Windows

* CUDA / CPU 安装包。若被 Defender 拦截：更多信息 → 仍要运行。
* 安装卡死时结束残留的 `api-server.exe` / `open-auto-note.exe` 再重试。

## 本地开发

需要：Node.js 20+、Rust、Python 3.10+、ffmpeg（建议 aria2）。

```bash
git clone https://github.com/JasonZhang1225/OpenAutoNote.git
cd OpenAutoNote

cd python-core
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env              # 填入 DASHSCOPE_API_KEY
python main.py                    # http://127.0.0.1:8964

# 可选：Tauri 壳（会等 /health，并避免重复拉起 sidecar）
cd ..
npm install
npm run tauri dev
```

健康检查：`curl -s http://127.0.0.1:8964/health`

测试：

```bash
cd python-core
python -m unittest discover -s tests -v
```

仅在作为 Tauri sidecar 启动时加 `--watchdog`（父进程退出则退出）。直接 `python main.py` 不会启用。

## 常见问题 (FAQ)

### 1. 转写立刻失败，提示 DASHSCOPE_API_KEY？

默认后端是云端 Qwen3-ASR。按上文配置 Key。北京站和国际站 Key 不通用。

### 2. 为什么第一次用 faster-whisper 会卡住？

本地 Whisper 会从 Hugging Face 拉模型。国内可在设置里打开镜像，或设 `HF_ENDPOINT=https://hf-mirror.com`。

### 3. macOS 打开后 Connecting / 启动中然后退出？

见 issue #1。新版本不会在 sidecar 失败时 `expect` panic，也不会在关掉启动页时把整个应用 `exit(1)`。请用带这些修复的构建，或直接跑 `python-core/main.py`。

### 4. Windows 安装 File error？

结束旧的 `api-server.exe` 再重试。
