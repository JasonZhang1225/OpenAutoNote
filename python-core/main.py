import os
import json
import re
import asyncio
import time
from nicegui import ui, run, app
import sys
import logging

from core.prompts import get_prompt, get_normal_prompt, get_chunk_prompt

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Retry decorator for API calls
async def retry_async(func, max_retries=3, initial_delay=1, backoff_factor=2, exceptions=(Exception,)):
    """
    Async retry decorator with exponential backoff
    
    Args:
        func: Async function to retry
        max_retries: Maximum number of retries
        initial_delay: Initial delay in seconds
        backoff_factor: Backoff factor for exponential delay
        exceptions: Tuple of exception types to retry on
    
    Returns:
        Result of the function call
    """
    retry_count = 0
    while retry_count <= max_retries:
        try:
            return await func()
        except exceptions as e:
            retry_count += 1
            if retry_count > max_retries:
                logger.error(f"Failed after {max_retries} retries: {e}")
                raise
            
            delay = initial_delay * (backoff_factor ** (retry_count - 1))
            logger.info(f"Retrying in {delay:.2f} seconds... (Attempt {retry_count}/{max_retries})")
            await asyncio.sleep(delay)

# Get the directory where this script is located
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Capture original streams for redirection
original_stdout = sys.stdout
original_stderr = sys.stderr
from openai import AsyncOpenAI

from core.downloader import download_video
from core.stt import (
    DEFAULT_STT_BACKEND,
    TranscriptionResult,
    TranscriberFactory,
    load_stt_env,
    resolve_stt_backend,
)
from core.visual_processor import process_video_for_vision, extract_frame
from core import model_manager
from core.torch_manager import check_torch_cuda_installed, install_torch_cuda, detect_cuda_version, get_torch_install_status
from core.utils import (
    build_multimodal_payload,
    timestamp_str_to_seconds,
    clean_bilibili_url,
)
from core.i18n import get_text
from core.storage import (
    load_history,
    save_history,
    add_session,
    get_session,
    delete_session,
    create_session,
    clear_all_history,
    rename_session,
    sync_history,
    load_chat_history,
    save_chat_history,
    validate_and_cleanup_sessions,
)

# --- Configuration & State ---
load_stt_env()
CONFIG_FILE = os.path.join(BASE_DIR, "user_config.json")
DEFAULT_CONFIG = {
    "api_key": "",
    "base_url": "https://api.openai.com/v1",
    "model_name": "gpt-4o",
    "asr_model": "small",
    "stt_backend": DEFAULT_STT_BACKEND,
    "dashscope_api_key": "",
    "dashscope_region": "beijing",
    "language": "Simplified Chinese (Default)",
    "hardware_mode": "mlx",
    "enable_vision": True,
    "vision_interval": 15,
    "vision_detail": "low",
    "detail_level": "Standard",
    "ui_language": "zh",
    "theme_mode": "light",
    "cookies_yt": "",
    "cookies_bili": "",
    "use_china_mirror": False,
    "enable_chunk_summary": False,
    "deep_thinking": False,
    "first_launch_completed": False,
    "remind_gpu_install": True,
}

# --- Hardware Detection ---
import platform


def detect_hardware():
    info = {"type": "cpu", "name": "Standard CPU", "valid_modes": ["cpu"]}

    # 1. Apple Silicon
    if platform.system() == "Darwin" and platform.processor() == "arm":
        info["type"] = "mlx"
        info["name"] = "Apple Silicon (M-Series)"
        info["valid_modes"] = ["mlx", "cpu"]
        return info

    # 2. NVIDIA CUDA
    try:
        import torch

        if torch.cuda.is_available():
            info["type"] = "cuda"
            info["name"] = torch.cuda.get_device_name(0)
            info["valid_modes"] = ["cuda", "cpu"]
            return info
    except ImportError:
        pass

    return info


hardware_info = detect_hardware()

# Default to the detected hardware for transcription (mlx on macOS, cuda if available on Windows, else cpu)
DEFAULT_CONFIG["hardware_mode"] = hardware_info["type"]


def check_first_launch_gpu_reminder():
    """Check first launch and show GPU installation reminder if needed"""
    if not state.config.get("first_launch_completed", False):
        try:
            cuda_version = detect_cuda_version()
            torch_status = get_torch_install_status()

            needs_reminder = (
                cuda_version is not None and
                not torch_status["cuda_available"] and
                state.config.get("remind_gpu_install", True)
            )

            if needs_reminder:
                with ui.dialog() as dialog, ui.card().classes("w-[500px] max-w-full"):
                    ui.label("🚀 开启 GPU 加速").classes("text-xl font-bold text-primary mb-4")
                    ui.label(f"检测到您的系统安装了 NVIDIA {cuda_version} 显卡，但尚未安装支持 CUDA 的 PyTorch 版本。").classes("text-sm text-gray-700 mb-2")
                    ui.label("安装 GPU 版 PyTorch 可以显著提升视频处理速度（通常提升 2-5 倍）").classes("text-sm text-gray-700 mb-4")

                    ui.label("安装命令：").classes("text-sm font-bold mt-2")
                    install_cmd = f"pip3 install torch torchvision --index-url https://mirrors.nju.edu.cn/pytorch/whl/{cuda_version}"
                    with ui.row().classes("w-full items-center gap-2 mt-1"):
                        ui.textarea(value=install_cmd).classes("flex-1 text-sm font-mono").props("readonly outlined")
                        def copy_cmd():
                            ui.run_javascript(f"navigator.clipboard.writeText('{install_cmd}')")
                            ui.notify("已复制到剪贴板", type="info", timeout=2000)
                        ui.button(icon="content_copy", on_click=copy_cmd).props("flat dense")

                    with ui.row().classes("w-full justify-end gap-2 mt-4"):
                        def skip_reminder():
                            state.config["first_launch_completed"] = True
                            state.config["remind_gpu_install"] = False
                            save_config(state.config)
                            dialog.close()

                        def complete_setup():
                            state.config["first_launch_completed"] = True
                            save_config(state.config)
                            dialog.close()
                            ui.notify("如已安装 PyTorch，请在设置中刷新硬件状态", type="info", timeout=3000)

                        ui.button("不再提醒", on_click=skip_reminder).props("flat")
                        ui.button("我已知晓", on_click=complete_setup).props("raised color=primary")

                dialog.open()

        except Exception as e:
            print(f"First launch check failed: {e}")


GENERATE_DIR = os.path.join(BASE_DIR, "generate")
if not os.path.exists(GENERATE_DIR):
    os.makedirs(GENERATE_DIR)
app.add_static_files("/generate", GENERATE_DIR)


def load_config():
    cfg = DEFAULT_CONFIG.copy()
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                cfg = {**DEFAULT_CONFIG, **json.load(f)}
        except Exception:
            cfg = DEFAULT_CONFIG.copy()

    # If stored hardware mode is invalid for this machine, fall back to detected type
    if cfg.get("hardware_mode") not in hardware_info["valid_modes"]:
        cfg["hardware_mode"] = hardware_info["type"]

    if not cfg.get("stt_backend"):
        cfg["stt_backend"] = DEFAULT_STT_BACKEND

    # Whisper model list only applies to the legacy local backend
    stt_backend = resolve_stt_backend(cfg)
    if stt_backend == "faster-whisper":
        supported_models = model_manager.get_supported_models(cfg["hardware_mode"])
        if cfg.get("asr_model") not in supported_models:
            cfg["asr_model"] = (
                "small"
                if "small" in supported_models
                else next(iter(supported_models.keys()), "small")
            )

    return cfg


def save_config(cfg):
    with open(CONFIG_FILE, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)


class State:
    def __init__(self):
        self.config = load_config()
        self.current_session = None
        self._deleting_session = False  # Track deletion state to prevent race conditions


state = State()


class WebLogger:
    def __init__(self, original_stream, ui_log_element):
        self.terminal = original_stream
        self.log_element = ui_log_element
        self._recursion_guard = False
        self._last_chunk_log_time = 0
        self._chunk_log_interval = 2.0  # Only log chunk updates every 2 seconds
        self._chunk_log_count = 0

    def write(self, message):
        # 1. Write to the real terminal (Keep backend working - all logs go here)
        self.terminal.write(message)

        # 2. Filter logic: Filter out progress bars and token-level AI API logs
        # yt-dlp/aria2 progress bars usually start with '\r' or contain 'ETA' or '[download]'
        if "\r" in message and ("%" in message or "ETA" in message or "[download]" in message):
            return
        
        # Filter out token-level progress logs (keep only paragraph-level logs)
        # Token-level logs are frequent updates that occur during streaming (per token/chunk)
        # Paragraph-level logs are milestone events (start, completion, errors)
        
        is_token_level_log = False
        
        # Pattern 1: API streaming progress logs (token-level updates during streaming)
        # Format: "[AI API] Progress: X chunks, Y chars..."
        if "[AI API] Progress:" in message and ("chars" in message or "chars..." in message):
            is_token_level_log = True
        
        # Pattern 2: Recursive chunk token-level progress updates during generation
        # Format: "[Recursive Chunk X/Y] Reasoning: Y chars..." or "Content: X chars..."
        # These are incremental updates during streaming
        if "[Recursive Chunk" in message:
            if (("Reasoning:" in message or "Content:" in message) and 
                "chars" in message and "chars..." in message):
                is_token_level_log = True
        
        # Pattern 3: AI API chunk content received logs (per-chunk updates during streaming)
        # These are detailed token-level updates
        if "[AI API] Chunk" in message:
            if any(phrase in message for phrase in [
                "Content received", 
                "Reasoning content received", 
                "Content treated as reasoning"
            ]):
                is_token_level_log = True
        
        # Skip token-level logs entirely for GUI terminal
        # All logs are still printed to backend terminal (done above at line 153)
        if is_token_level_log:
            return
        
        # Keep paragraph-level logs (examples below will be shown in GUI):
        # - "[AI API] Starting API call for: ..."
        # - "[AI API] API call completed. Total chunks: ..."
        # - "[AI API] Error during API call: ..."
        # - "[Recursive Chunk X/Y] Starting AI summary generation..."
        # - "[Recursive Chunk X] Summary generation completed..."
        # - "[Recursive Chunk X] Final chunk_full_response length: ..." (completion summary)
        # - "[Chunk X] Summary generation completed. Success: ..." (completion summary)
        # - Any error or warning messages
        
        # Keep paragraph-level logs (API call start/end, chunk completion, errors, etc.)
        # These are important for user visibility
        # Examples:
        # - "[AI API] Starting API call for: ..."
        # - "[AI API] API call completed..."
        # - "[Recursive Chunk X/Y] Starting AI summary generation..."
        # - "[Recursive Chunk X] Summary generation completed..."
        # - "[Chunk X] Summary generation completed..."
        # - "[AI API] Error..."
        # - Any error messages
        
        try:
            # Prevent infinite recursion: Don't push if we're already in a push operation
            if not self._recursion_guard:
                self._recursion_guard = True
                # Push to UI
                self.log_element.push(message)
                self._recursion_guard = False
        except Exception:
            self._recursion_guard = False
            pass  # Avoid errors if UI is disconnected

    def flush(self):
        # Use original stdout/stderr handles to avoid infinite recursion
        # when self is assigned to sys.stdout/sys.stderr
        try:
            sys.__stdout__.flush()
            sys.__stderr__.flush()
        except (AttributeError, ValueError, OSError):
            pass


# --- Atomic Finalization Helper ---
def finalize_task(task_id: str, raw_title: str, report_content: str, abstract_content: str = "", contents_content: str = "") -> tuple:
    """
    Atomic finalization: Save report.md, abstract.md, contents.md, update paths, then rename folder.
    Returns (final_folder_path, updated_report_content).
    """
    from core.utils import sanitize_filename

    clean_title = sanitize_filename(raw_title)
    old_folder = os.path.join(GENERATE_DIR, task_id)
    new_folder = os.path.join(GENERATE_DIR, clean_title)

    # Avoid overwriting existing folders
    if os.path.exists(new_folder):
        counter = 1
        base_folder = new_folder
        
        # Find the next available number suffix
        while os.path.exists(new_folder):
            new_folder = f"{base_folder}_{counter}"
            counter += 1


    report_path = os.path.join(old_folder, "report.md")
    abstract_path = os.path.join(old_folder, "abstract.md")
    contents_path = os.path.join(old_folder, "contents.md")

    if not os.path.exists(old_folder):
        print(f"[Finalize] Error: Folder {old_folder} not found.")
        return old_folder, report_content

    # 1. Update paths in report content BEFORE renaming
    # Extract the final folder name without path for URL replacement
    final_folder_name = os.path.basename(new_folder)
    updated_content = report_content.replace(
        f"/generate/{task_id}", f"/generate/{final_folder_name}"
    )

    # 2. Save report.md to disk
    try:
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(updated_content)
        print(f"[Finalize] Saved report.md to {report_path}")
    except Exception as e:
        print(f"[Finalize] Could not save report: {e}")

    # 3. Save abstract.md if content exists
    if abstract_content:
        try:
            with open(abstract_path, "w", encoding="utf-8") as f:
                f.write(abstract_content)
            print(f"[Finalize] Saved abstract.md to {abstract_path}")
        except Exception as e:
            print(f"[Finalize] Could not save abstract: {e}")

    # 4. Save contents.md if content exists
    if contents_content:
        try:
            with open(contents_path, "w", encoding="utf-8") as f:
                f.write(contents_content)
            print(f"[Finalize] Saved contents.md to {contents_path}")
        except Exception as e:
            print(f"[Finalize] Could not save contents: {e}")

    # 5. Rename folder (the atomic move)
    try:
        os.rename(old_folder, new_folder)
        print(f"[Finalize] Renamed folder: {old_folder} -> {new_folder}")
        return new_folder, updated_content
    except OSError as e:
        print(f"[Finalize] Rename failed: {e}")
        return old_folder, report_content


# --- Async Wrappers ---
async def async_download(url):
    return await run.io_bound(download_video, url)


async def async_transcribe(video_path, hardware_mode, progress_callback=None):
    def _t():
        actual_mode = hardware_mode
        stt_backend = resolve_stt_backend(state.config)

        # Torch / MLX fallbacks only matter for the legacy local Whisper path.
        if stt_backend == "faster-whisper":
            if hardware_mode == "cuda":
                torch_installed, _ = check_torch_cuda_installed()
                if not torch_installed:
                    cuda_version = detect_cuda_version()
                    if cuda_version:
                        print(f"[Torch Install] CUDA {cuda_version} detected, missing PyTorch CUDA build")

                        def progress_update(msg):
                            print(f"[Torch Install] {msg}")

                        success = install_torch_cuda(cuda_version, progress_update)
                        if success:
                            torch_installed, _ = check_torch_cuda_installed()
                            if not torch_installed:
                                actual_mode = "cpu"
                        else:
                            actual_mode = "cpu"
                    else:
                        actual_mode = "cpu"
                elif "cuda" not in hardware_info["valid_modes"]:
                    actual_mode = "cpu"
            elif hardware_mode == "mlx" and "mlx" not in hardware_info["valid_modes"]:
                actual_mode = "cpu"

        print(f"Transcribing with backend={stt_backend} hardware={actual_mode}...")

        # China Mirror Setup (local whisper weights)
        if state.config.get("use_china_mirror"):
            os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

        asr_model = state.config.get("asr_model", "small")
        factory_backend = actual_mode if stt_backend == "faster-whisper" else stt_backend
        transcriber = TranscriberFactory.get_transcriber(
            factory_backend,
            asr_model,
            hardware_mode=actual_mode,
            config=state.config,
        )

        result = transcriber.transcribe(video_path, progress_callback=progress_callback)
        if isinstance(result, TranscriptionResult):
            return result.segments
        return result

    stt_backend = resolve_stt_backend(state.config)
    if stt_backend == "faster-whisper":
        ui.notify("Checking/Loading local Whisper model...", type="info", timeout=3000)
    else:
        ui.notify(f"Starting STT ({stt_backend})...", type="info", timeout=3000)
    return await run.io_bound(_t)


async def async_vision(video_path, interval, output_dir):
    return await run.io_bound(
        process_video_for_vision, video_path, interval, output_dir
    )


def extract_toc_from_content(content):
    """
    从markdown内容中提取目录结构（H2和H3标题）
    返回目录markdown格式的字符串，格式为：
    ## 📑 目录
    - [标题1](#标题1)
    - [标题2](#标题2)
      - [子标题2.1](#子标题2.1)
    """
    if not content:
        return ""
    
    # 提取所有H2和H3标题
    h2_pattern = r"^##\s+(.+?)$"
    h3_pattern = r"^###\s+(.+?)$"
    
    lines = content.split('\n')
    toc_items = []
    current_h2 = None
    
    for line in lines:
        h2_match = re.match(h2_pattern, line)
        h3_match = re.match(h3_pattern, line)
        
        if h2_match:
            title = h2_match.group(1).strip()
            # 移除emoji和特殊字符以创建锚点
            anchor = re.sub(r'[^\w\s-]', '', title).strip()
            anchor = re.sub(r'[-\s]+', '-', anchor).lower()
            # 如果标题包含emoji，保留在显示文本中
            toc_items.append(('h2', title, anchor))
            current_h2 = len(toc_items) - 1
        elif h3_match:
            title = h3_match.group(1).strip()
            anchor = re.sub(r'[^\w\s-]', '', title).strip()
            anchor = re.sub(r'[-\s]+', '-', anchor).lower()
            toc_items.append(('h3', title, anchor))
    
    if not toc_items:
        return ""
    
    # 生成目录markdown
    toc_lines = ["## 📑 目录"]
    for level, title, anchor in toc_items:
        if level == 'h2':
            toc_lines.append(f"- [{title}](#{anchor})")
        elif level == 'h3':
            toc_lines.append(f"  - [{title}](#{anchor})")
    
    return '\n'.join(toc_lines)

def merge_tocs(existing_toc, new_toc):
    """
    合并两个目录，将新目录追加到现有目录后面
    如果existing_toc为空，直接返回new_toc
    """
    if not existing_toc:
        return new_toc
    if not new_toc:
        return existing_toc
    
    # 从existing_toc中提取除了"## 📑 目录"之外的所有行
    existing_lines = existing_toc.split('\n')
    existing_items = [line for line in existing_lines if line.strip() and not line.strip().startswith('## 📑')]
    
    # 从new_toc中提取目录项
    new_lines = new_toc.split('\n')
    new_items = [line for line in new_lines if line.strip() and not line.strip().startswith('## 📑')]
    
    # 合并
    if existing_items and new_items:
        merged_lines = ["## 📑 目录"] + existing_items + new_items
        return '\n'.join(merged_lines)
    elif existing_items:
        return existing_toc
    else:
        return new_toc

def split_transcript_into_chunks(segments, target_duration_minutes=15):
    """
    Split transcript into logical chunks based on content and target duration.
    Returns list of chunks, each containing segments and metadata.
    """
    if not segments:
        return []
    
    target_duration_seconds = target_duration_minutes * 60
    chunks = []
    current_chunk = []
    current_duration = 0
    
    for segment in segments:
        # Calculate duration of current segment
        start_time = segment.get('start', 0)
        end_time = segment.get('end', 0)
        duration = end_time - start_time
        
        # Check if adding this segment would exceed target duration
        if current_duration + duration > target_duration_seconds and current_chunk:
            # Finalize current chunk
            chunks.append({
                'segments': current_chunk,
                'start_time': current_chunk[0]['start'],
                'end_time': current_chunk[-1]['end'],
                'duration': current_duration,
                'text': ' '.join([s['text'] for s in current_chunk])
            })
            # Start new chunk
            current_chunk = []
            current_duration = 0
        
        # Add segment to current chunk
        current_chunk.append(segment)
        current_duration += duration
    
    # Add the last chunk
    if current_chunk:
        chunks.append({
            'segments': current_chunk,
            'start_time': current_chunk[0]['start'],
            'end_time': current_chunk[-1]['end'],
            'duration': current_duration,
            'text': ' '.join([s['text'] for s in current_chunk])
        })
    
    return chunks

async def process_chunk_recursively(chunk_index, total_chunks, chunk, dl_res, vision_frames, state, custom_prompt, complexity, chunk_context, step_ai, md_container, final_display_text, full_response, reasoning_exp, reasoning_label, task_id, assets_dir, processed_timestamps, accumulated_toc=""):
    """
    Recursively process a chunk, splitting it into smaller chunks if token limit is exceeded.
    Returns (processed_successfully, chunk_full_response, chunk_full_reasoning, updated_final_display_text, updated_full_response, chunk_toc)
    
    Args:
        accumulated_toc: 前面所有分块累积的目录，用于指导当前分块的输出结构
    """
    # Build context from previous chunks
    context_prompt = ""
    if chunk_context:
        context_prompt = f"\n\n前面部分的摘要内容（供参考上下文）：\n{chr(10).join(chunk_context)}\n\n"
    
    # 如果有累积的目录，添加到prompt中指导输出结构
    toc_guidance = ""
    if accumulated_toc:
        toc_guidance = f"\n\n**重要：前面部分的目录结构如下，请参考这个结构和层级继续规划输出：**\n{accumulated_toc}\n\n请根据以上目录结构，在你负责的部分中继续使用相同的层级和编号规范，确保整体文档结构连贯一致。"
    
    # Initialize chunk-specific variables
    chunk_full_response = ""
    chunk_full_reasoning = ""
    
    # Generate summary for this chunk
    print(f"[Recursive Chunk {chunk_index}/{total_chunks}] Starting AI summary generation...")
    chunk_content_received = False
    chunk_reasoning_received = False
    last_update_time = time.time()
    update_log_interval = 3.0  # Log updates every 3 seconds
    last_ui_update_time = time.time()
    ui_update_interval = 0.5  # Update UI at most every 0.5 seconds to reduce connection load
    
    async for chunk_type, chunk_text in generate_summary_stream_async(
        f"{dl_res['title']} - Chunk {chunk_index}/{total_chunks}",
        chunk['text'],
        chunk['segments'],
        vision_frames,
        state.config,
        custom_prompt + f"\n\n这是长视频的第 {chunk_index} 部分，共 {total_chunks} 部分。如果上一部分的最后一个章节编号是某个数字，请从此数字加1开始继续编号。专注于总结这一部分的具体内容。包含关键引用和结构化摘要。" + context_prompt + toc_guidance,
        complexity,
    ):
        # Remove detailed per-chunk logging to reduce connection load
        current_time = time.time()
        should_log_update = current_time - last_update_time >= update_log_interval
        
        if chunk_type == "reasoning":
            chunk_full_reasoning += chunk_text
            reasoning_exp.classes(remove="hidden")
            # Throttle UI updates for reasoning to reduce connection load
            if current_time - last_ui_update_time >= ui_update_interval:
                try:
                    reasoning_label.set_content(chunk_full_reasoning)
                    last_ui_update_time = current_time
                except Exception:
                    pass  # Ignore UI update errors to prevent connection issues
            chunk_reasoning_received = True
            if should_log_update:
                print(f"[Recursive Chunk {chunk_index}/{total_chunks}] Reasoning: {len(chunk_full_reasoning)} chars...")
                last_update_time = current_time
        
        elif chunk_type == "content":
            chunk_full_response += chunk_text
            chunk_content_received = True
            
            if full_response:
                current_full_response = full_response + "\n\n" + chunk_full_response
            else:
                current_full_response = chunk_full_response
            step_ai.props('caption="✍️ Writing report..."')

            # Image Logic Logic (Updated for assets_dir)
            display_text = current_full_response
            timestamps = re.findall(r"\[(\d{1,2}:\d{2})\]", display_text)
            for ts in timestamps:
                seconds = timestamp_str_to_seconds(ts)
                img_filename = f"frame_{seconds}.jpg"
                img_fs_path = os.path.join(assets_dir, img_filename)
                img_web_path = f"/generate/{task_id}/assets/{img_filename}"

                if ts not in processed_timestamps:
                    if not os.path.exists(img_fs_path):
                        await run.io_bound(
                            extract_frame,
                            dl_res["video_path"],
                            seconds,
                            img_fs_path,
                        )
                    processed_timestamps.add(ts)

                if os.path.exists(img_fs_path):
                    if f"![{ts}]" not in display_text:
                        display_text = display_text.replace(
                            f"[{ts}]", f"[{ts}]\n\n![{ts}]({img_web_path})"
                        )

            # Throttle UI updates for content to reduce connection load
            if current_time - last_ui_update_time >= ui_update_interval:
                try:
                    md_container.set_content(display_text)
                    last_ui_update_time = current_time
                except Exception:
                    pass  # Ignore UI update errors to prevent connection issues
            # Store the final display_text for finalization
            final_display_text = display_text
            if should_log_update:
                print(f"[Recursive Chunk {chunk_index}/{total_chunks}] Content: {len(chunk_full_response)} chars, display: {len(display_text)} chars...")
                last_update_time = current_time
        
        elif chunk_type == "error":
            # Check if this is a token limit error
            if "token_limit_error" in chunk_text:
                print(f"[Recursive Chunk {chunk_index}] Token limit exceeded, splitting into smaller chunks...")
                
                # Split this chunk into smaller chunks
                num_segments = len(chunk['segments'])
                if num_segments <= 1:
                    print(f"[Recursive Chunk {chunk_index}] Cannot split further, using as is...")
                    return False, "", "", final_display_text, full_response, ""
                
                # Split into two equal parts
                mid_point = num_segments // 2
                
                # Create first sub-chunk
                sub_chunk1 = {
                    'segments': chunk['segments'][:mid_point],
                    'start_time': chunk['segments'][0]['start'],
                    'end_time': chunk['segments'][mid_point-1]['end'],
                    'duration': chunk['segments'][mid_point-1]['end'] - chunk['segments'][0]['start'],
                    'text': ' '.join([s['text'] for s in chunk['segments'][:mid_point]])
                }
                
                # Create second sub-chunk
                sub_chunk2 = {
                    'segments': chunk['segments'][mid_point:],
                    'start_time': chunk['segments'][mid_point]['start'],
                    'end_time': chunk['segments'][-1]['end'],
                    'duration': chunk['segments'][-1]['end'] - chunk['segments'][mid_point]['start'],
                    'text': ' '.join([s['text'] for s in chunk['segments'][mid_point:]])
                }
                
                # Get vision frames for first sub-chunk
                sub_chunk1_vision_frames = [
                    frame for frame in vision_frames 
                    if sub_chunk1['start_time'] <= frame['timestamp'] <= sub_chunk1['end_time']
                ]
                
                # Get vision frames for second sub-chunk
                sub_chunk2_vision_frames = [
                    frame for frame in vision_frames 
                    if sub_chunk2['start_time'] <= frame['timestamp'] <= sub_chunk2['end_time']
                ]
                
                # Process first sub-chunk
                success1, resp1, reason1, final_display_text, full_response, toc1 = await process_chunk_recursively(
                    f"{chunk_index}.1", total_chunks, sub_chunk1, dl_res, sub_chunk1_vision_frames, state, 
                    custom_prompt, complexity, chunk_context, step_ai, md_container, final_display_text, full_response, 
                    reasoning_exp, reasoning_label, task_id, assets_dir, processed_timestamps, accumulated_toc
                )
                
                if success1 and resp1:
                    # Add to context for second sub-chunk
                    chunk_context.append(f"部分 {chunk_index}.1: {resp1[:100]}...")
                
                # Process second sub-chunk with accumulated TOC from first sub-chunk
                accumulated_toc_after_sub1 = merge_tocs(accumulated_toc, toc1)
                success2, resp2, reason2, final_display_text, full_response, toc2 = await process_chunk_recursively(
                    f"{chunk_index}.2", total_chunks, sub_chunk2, dl_res, sub_chunk2_vision_frames, state, 
                    custom_prompt, complexity, chunk_context, step_ai, md_container, final_display_text, full_response, 
                    reasoning_exp, reasoning_label, task_id, assets_dir, processed_timestamps, accumulated_toc_after_sub1
                )
                
                # Combine results and TOCs
                if success1 or success2:
                    combined_response = resp1 + ("\n\n---\n\n" if resp1 and resp2 else "") + resp2
                    combined_reasoning = reason1 + ("\n\n---\n\n" if reason1 and reason2 else "") + reason2
                    combined_toc = merge_tocs(toc1, toc2)
                    return True, combined_response, combined_reasoning, final_display_text, full_response, combined_toc
                else:
                    return False, "", "", final_display_text, full_response, ""
            else:
                # Other error, return failure
                print(f"[Recursive Chunk {chunk_index}] Non-token error occurred: {chunk_text}")
                return False, "", "", final_display_text, full_response, ""
    
    print(f"[Recursive Chunk {chunk_index}] Summary generation completed. Content received: {chunk_content_received}, Reasoning received: {chunk_reasoning_received}")
    print(f"[Recursive Chunk {chunk_index}] Final chunk_full_response length: {len(chunk_full_response)}")
    
    # Extract TOC from chunk content (don't include in the actual response, just extract for reference)
    chunk_toc = ""
    if chunk_content_received and chunk_full_response:
        chunk_toc = extract_toc_from_content(chunk_full_response)
        print(f"[Recursive Chunk {chunk_index}] Extracted TOC: {len(chunk_toc)} chars")
    
    return chunk_content_received, chunk_full_response, chunk_full_reasoning, final_display_text, full_response, chunk_toc

async def generate_segmented_content_async(
    chunk_idx, total_chunks, chunk_content, vision_frames, config, prev_abstracts="", on_stream=None
):
    """
    Generate segmented content for a chunk using the new segmented prompts.
    Returns the generated content.
    """
    if not config["api_key"]:
        return "Error: API Key missing."

    client = AsyncOpenAI(api_key=config["api_key"], base_url=config["base_url"])

    ui_lang = state.config["ui_language"]
    default_lang = ui_lang

    base_identity = get_prompt("chunk", "base_identity", ui_lang, default_lang=default_lang)
    language_style = get_prompt("chunk", "language_style", ui_lang, default_lang=default_lang)

    complexity_levels = {
        1: get_text("complexity_level_1", ui_lang),
        2: get_text("complexity_level_2", ui_lang),
        3: get_text("complexity_level_3", ui_lang),
        4: get_text("complexity_level_4", ui_lang),
        5: get_text("complexity_level_5", ui_lang),
    }
    complexity_instruction = complexity_levels.get(3, complexity_levels[3])

    if chunk_idx == 1:
        content_prompt = get_prompt("chunk", "chunk_first_content", ui_lang,
            chunk_idx=chunk_idx,
            total_chunks=total_chunks
        )
    else:
        content_prompt = get_prompt("chunk", "chunk_n_content", ui_lang,
            chunk_idx=chunk_idx,
            total_chunks=total_chunks,
            prev_chunk_count=chunk_idx - 1,
            prev_abstracts=prev_abstracts
        )

    deep_thinking_instruction = ""
    if config.get("deep_thinking", False):
        if ui_lang == "zh":
            deep_thinking_instruction = "\n\n特别注意：启用深度思考模式。请进行多角度分析，深入挖掘内容背后的含义和逻辑关系，提供更全面的见解和洞察。"
        else:
            deep_thinking_instruction = "\n\nSpecial note: Deep thinking mode is enabled. Please perform multi-angle analysis, dig deeper into the meaning and logical relationships behind the content, and provide more comprehensive insights and perspectives."

    complexity_requirement = get_prompt("chunk", "output_complexity_requirement", ui_lang)
    user_extra = get_prompt("chunk", "user_extra_requirement", ui_lang)

    system_prompt = f"""{base_identity}

{language_style}

{complexity_requirement}{complexity_instruction}

{content_prompt}

{deep_thinking_instruction}

Output only the formal content with headings, nothing else.
"""

    try:
        print(f"[Segmented] Generating content for chunk {chunk_idx}/{total_chunks}...")

        if config["enable_vision"] and vision_frames:
            user_content = build_multimodal_payload(
                f"Chunk {chunk_idx}/{total_chunks}", chunk_content, [], vision_frames, detail=config["vision_detail"]
            )
        else:
            user_content = chunk_content

        async def api_call():
            return await client.chat.completions.create(
                model=config["model_name"],
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content},
                ],
                stream=True,
            )

        response = await retry_async(api_call, max_retries=3, initial_delay=1, backoff_factor=2)

        content = ""
        async for chunk in response:
            delta = chunk.choices[0].delta
            chunk_text = getattr(delta, "content", None)
            if chunk_text:
                content += chunk_text
                if on_stream:
                    on_stream(content)

        print(f"[Segmented] Generated content for chunk {chunk_idx}: {len(content)} chars")
        return content

    except Exception as e:
        print(f"[Segmented] Error generating content for chunk {chunk_idx}: {e}")
        return f"Error generating content: {str(e)}"


async def generate_summary_stream_async(
    title, full_text, segments, vision_frames, config, custom_prompt="", complexity=3
):
    if not config["api_key"]:
        yield ("error", "Error: API Key missing.")
        return

    client = AsyncOpenAI(api_key=config["api_key"], base_url=config["base_url"])

    complexity_levels = {
        1: get_text("complexity_level_1", state.config["ui_language"]),
        2: get_text("complexity_level_2", state.config["ui_language"]),
        3: get_text("complexity_level_3", state.config["ui_language"]),
        4: get_text("complexity_level_4", state.config["ui_language"]),
        5: get_text("complexity_level_5", state.config["ui_language"]),
    }

    complexity_instruction = complexity_levels.get(complexity, complexity_levels[3])

    ui_lang = state.config["ui_language"]
    default_lang = ui_lang

    base_identity = get_prompt("normal", "base_identity", ui_lang, default_lang=default_lang)
    language_style = get_prompt("normal", "language_style", ui_lang, default_lang=default_lang)

    deep_thinking_instruction = ""
    if config.get("deep_thinking", False):
        if ui_lang == "zh":
            deep_thinking_instruction = "\n\n特别注意：启用深度思考模式。请进行多角度分析，深入挖掘内容背后的含义和逻辑关系，提供更全面的见解和洞察。"
        else:
            deep_thinking_instruction = "\n\nSpecial note: Deep thinking mode is enabled. Please perform multi-angle analysis, dig deeper into the meaning and logical relationships behind the content, and provide more comprehensive insights and perspectives."

    is_chunk_summary = "Chunk" in title

    if is_chunk_summary:
        if custom_prompt and custom_prompt.strip():
            user_extra = get_prompt("normal", "user_extra_requirement", ui_lang)
            complexity_requirement = get_prompt("normal", "output_complexity_requirement", ui_lang)
            chunk_requirements = get_prompt("chunk", "chunk_requirements", ui_lang)

            system_prompt = f"""{base_identity}

{user_extra}
{custom_prompt.strip()}

{complexity_requirement}{complexity_instruction}

{chunk_requirements}

{deep_thinking_instruction}

{language_style}"""
        else:
            complexity_requirement = get_prompt("normal", "output_complexity_requirement", ui_lang)
            chunk_requirements = get_prompt("chunk", "chunk_requirements", ui_lang)

            system_prompt = f"""{base_identity}

{complexity_requirement}{complexity_instruction}

{chunk_requirements}

{deep_thinking_instruction}

{language_style}"""
    else:
        if custom_prompt and custom_prompt.strip():
            user_extra = get_prompt("normal", "user_extra_requirement", ui_lang)
            complexity_requirement = get_prompt("normal", "output_complexity_requirement", ui_lang)
            full_requirements = get_prompt("normal", "full_requirements", ui_lang)
            core_layout = get_prompt("normal", "core_layout_requirements", ui_lang)
            one_liner = get_prompt("normal", "the_one_liner", ui_lang)
            one_liner_desc = get_prompt("normal", "the_one_liner_desc", ui_lang)
            structured_toc = get_prompt("normal", "structured_toc", ui_lang)
            structured_toc_desc = get_prompt("normal", "structured_toc_desc", ui_lang)
            structured_sections = get_prompt("normal", "structured_sections", ui_lang)
            structured_sections_desc = get_prompt("normal", "structured_sections_desc", ui_lang)
            data_comparison = get_prompt("normal", "data_comparison", ui_lang)
            math_formulas_desc = get_prompt("normal", "math_formulas_desc", ui_lang)
            visual_evidence = get_prompt("normal", "visual_evidence", ui_lang)
            visual_evidence_desc = get_prompt("normal", "visual_evidence_desc", ui_lang)

            system_prompt = f"""{base_identity}

{user_extra}
{custom_prompt.strip()}

{complexity_requirement}{complexity_instruction}

{full_requirements}

{core_layout}

{one_liner}
{one_liner_desc}

{structured_toc}
{structured_toc_desc}

{structured_sections}
{structured_sections_desc}

{data_comparison}
{math_formulas_desc}

{visual_evidence}
{visual_evidence_desc}

{language_style}"""
        else:
            complexity_requirement = get_prompt("normal", "output_complexity_requirement", ui_lang)
            full_requirements = get_prompt("normal", "full_requirements", ui_lang)
            core_layout = get_prompt("normal", "core_layout_requirements", ui_lang)
            one_liner = get_prompt("normal", "the_one_liner", ui_lang)
            one_liner_desc = get_prompt("normal", "the_one_liner_desc", ui_lang)
            structured_toc = get_prompt("normal", "structured_toc", ui_lang)
            structured_toc_desc = get_prompt("normal", "structured_toc_desc", ui_lang)
            structured_sections = get_prompt("normal", "structured_sections", ui_lang)
            structured_sections_desc = get_prompt("normal", "structured_sections_desc", ui_lang)
            data_comparison = get_prompt("normal", "data_comparison", ui_lang)
            data_comparison_desc = get_prompt("normal", "data_comparison_desc", ui_lang)
            math_formulas = get_prompt("normal", "math_formulas", ui_lang)
            math_formulas_desc = get_prompt("normal", "math_formulas_desc", ui_lang)
            visual_evidence = get_prompt("normal", "visual_evidence", ui_lang)
            visual_evidence_desc = get_prompt("normal", "visual_evidence_desc", ui_lang)

            system_prompt = f"""{base_identity}

{complexity_requirement}{complexity_instruction}

{full_requirements}

{core_layout}

{one_liner}
{one_liner_desc}

{structured_toc}
{structured_toc_desc}

{structured_sections}
{structured_sections_desc}

{data_comparison}
{data_comparison_desc}

{math_formulas}
{math_formulas_desc}

{visual_evidence}
{visual_evidence_desc}

{language_style}"""

    user_content = []
    if config["enable_vision"] and vision_frames:
        user_content = build_multimodal_payload(
            title, full_text, segments, vision_frames, detail=config["vision_detail"]
        )
    else:
        user_content = get_text(
            "user_content_prompt", state.config["ui_language"]
        ).format(title=title, full_text=full_text)

    try:
        print(f"[AI API] Starting API call for: {title}")
        print(f"[AI API] System prompt length: {len(system_prompt)}")
        print(f"[AI API] User content type: {type(user_content)}, length: {len(str(user_content)) if isinstance(user_content, str) else 'complex'}")
        
        try:
            response = await client.chat.completions.create(
                model=config["model_name"],
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content},
                ],
                stream=True,
            )
        except Exception as e:
            # Fallback for models that don't support multimodal content (e.g. 400 Bad Request)
            error_str = str(e)
            if config["enable_vision"] and vision_frames and ("ChatCompletionRequestMultiContent" in error_str or "InvalidParameter" in error_str or "400" in error_str):
                print(f"[AI API] Multimodal request failed ({error_str}), falling back to text-only...")
                user_content = get_text(
                    "user_content_prompt", state.config["ui_language"]
                ).format(title=title, full_text=full_text)
                
                response = await client.chat.completions.create(
                    model=config["model_name"],
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_content},
                    ],
                    stream=True,
                )
            else:
                raise e

        chunk_count = 0
        total_content_length = 0
        total_reasoning_length = 0
        last_log_time = time.time()
        log_interval = 5.0  # Log progress every 5 seconds
        
        async for chunk in response:
            chunk_count += 1
            delta = chunk.choices[0].delta
            content = getattr(delta, "content", None)
            reasoning = getattr(delta, "reasoning_content", None)

            # For models that support reasoning_content (like OpenAI o1 series)
            if reasoning:
                total_reasoning_length += len(reasoning)
                # Only log progress periodically to reduce output
                current_time = time.time()
                if current_time - last_log_time >= log_interval:
                    print(f"[AI API] Progress: {chunk_count} chunks, {total_reasoning_length} reasoning chars...")
                    last_log_time = current_time
                yield ("reasoning", reasoning)
            # For models that don't support reasoning_content, we'll use a portion of the content as thinking process
            elif content and "Thinking Process" in system_prompt:
                # If this is a specialized thinking prompt, treat content as reasoning
                total_reasoning_length += len(content)
                current_time = time.time()
                if current_time - last_log_time >= log_interval:
                    print(f"[AI API] Progress: {chunk_count} chunks, {total_reasoning_length} reasoning chars...")
                    last_log_time = current_time
                yield ("reasoning", content)
            elif content:
                total_content_length += len(content)
                # Only log progress periodically to reduce output
                current_time = time.time()
                if current_time - last_log_time >= log_interval:
                    print(f"[AI API] Progress: {chunk_count} chunks, {total_content_length} content chars...")
                    last_log_time = current_time
                yield ("content", content)
        
        print(f"[AI API] API call completed. Total chunks: {chunk_count}, Content length: {total_content_length}, Reasoning length: {total_reasoning_length}")

    except Exception as e:
        error_msg = str(e)
        print(f"[AI API] Error during API call: {error_msg}")
        
        is_token_limit_error = "Total tokens" in error_msg and "exceed max message tokens" in error_msg
        
        if is_token_limit_error:
            print(f"[AI API] Detected token limit error: {error_msg}")
            yield ("error", f"token_limit_error: {error_msg}")
        else:
            yield ("error", f"\n\n**Error:** {error_msg}")


async def generate_abstract_async(
    chunk_idx, total_chunks, chunk_content, prev_abstracts, config, on_stream=None
):
    """
    Generate abstract for a chunk. Returns the abstract text.
    """
    if not config["api_key"]:
        return "Error: API Key missing."

    client = AsyncOpenAI(api_key=config["api_key"], base_url=config["base_url"])

    complexity_levels = {
        1: get_text("complexity_level_1", state.config["ui_language"]),
        2: get_text("complexity_level_2", state.config["ui_language"]),
        3: get_text("complexity_level_3", state.config["ui_language"]),
        4: get_text("complexity_level_4", state.config["ui_language"]),
        5: get_text("complexity_level_5", state.config["ui_language"]),
    }

    ui_lang = state.config["ui_language"]
    default_lang = ui_lang

    base_identity = get_prompt("chunk", "base_identity", ui_lang, default_lang=default_lang)
    language_style = get_prompt("chunk", "language_style", ui_lang, default_lang=default_lang)

    if chunk_idx == 1:
        abstract_prompt = get_prompt("chunk", "chunk_first_abstract", ui_lang,
            chunk_idx=chunk_idx,
            total_chunks=total_chunks,
            chunk_idx_plus_1=chunk_idx + 1
        )
    else:
        abstract_prompt = get_prompt("chunk", "chunk_n_abstract", ui_lang,
            chunk_idx=chunk_idx,
            prev_abstracts=prev_abstracts
        )

    deep_thinking_instruction = ""
    if config.get("deep_thinking", False):
        if ui_lang == "zh":
            deep_thinking_instruction = "\n\n特别注意：启用深度思考模式。请进行多角度分析，深入挖掘内容背后的含义和逻辑关系，提供更全面的见解和洞察。"
        else:
            deep_thinking_instruction = "\n\nSpecial note: Deep thinking mode is enabled. Please perform multi-angle analysis, dig deeper into the meaning and logical relationships behind the content, and provide more comprehensive insights and perspectives."

    system_prompt = f"""{base_identity}

{language_style}

{abstract_prompt}

{deep_thinking_instruction}

Output only the abstract content, nothing else.
"""

    try:
        print(f"[Abstract] Generating abstract for chunk {chunk_idx}/{total_chunks}...")
        
        async def api_call():
            return await client.chat.completions.create(
                model=config["model_name"],
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"请根据以下第 {chunk_idx} 部分的内容生成摘要：\n\n{chunk_content}"},
                ],
                stream=True,
            )

        response = await retry_async(api_call, max_retries=3, initial_delay=1, backoff_factor=2)

        abstract = ""
        async for chunk in response:
            delta = chunk.choices[0].delta
            chunk_text = getattr(delta, "content", None)
            if chunk_text:
                abstract += chunk_text
                if on_stream:
                    on_stream(abstract)

        print(f"[Abstract] Generated abstract for chunk {chunk_idx}: {len(abstract)} chars")
        return abstract

    except Exception as e:
        print(f"[Abstract] Error generating abstract for chunk {chunk_idx}: {e}")
        return f"Error generating abstract: {str(e)}"


async def generate_final_contents_async(full_abstracts, config):
    """
    Generate final table of contents from all abstracts.
    """
    if not config["api_key"]:
        return "Error: API Key missing."

    client = AsyncOpenAI(api_key=config["api_key"], base_url=config["base_url"])

    if state.config["ui_language"] == "zh":
        base_identity = get_text("base_identity_zh", state.config["ui_language"])
    else:
        base_identity = get_text("base_identity_en", state.config["ui_language"])

    default_lang = state.config["ui_language"]
    language_style = get_text("language_style", state.config["ui_language"]).format(
        default_lang=default_lang
    )

    final_prompt = get_text("final_summary_prompt", state.config["ui_language"]).format(
        full_abstracts=full_abstracts
    )

    # Deep thinking mode addition
    deep_thinking_instruction = ""
    if config.get("deep_thinking", False):
        if state.config["ui_language"] == "zh":
            deep_thinking_instruction = "\n\n特别注意：启用深度思考模式。请进行多角度分析，深入挖掘内容背后的含义和逻辑关系，提供更全面的见解和洞察。"
        else:
            deep_thinking_instruction = "\n\nSpecial note: Deep thinking mode is enabled. Please perform multi-angle analysis, dig deeper into the meaning and logical relationships behind the content, and provide more comprehensive insights and perspectives."

    system_prompt = f"""{base_identity}

{language_style}

{final_prompt}

{deep_thinking_instruction}

Output only the contents.md content, nothing else.
"""

    try:
        print(f"[Final] Generating final table of contents...")
        
        async def api_call():
            return await client.chat.completions.create(
                model=config["model_name"],
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": "请根据所有部分的摘要生成最终的目录和梗概。"},
                ],
                stream=False,
            )
        
        response = await retry_async(api_call, max_retries=3, initial_delay=1, backoff_factor=2)

        contents = response.choices[0].message.content.strip()
        print(f"[Final] Generated table of contents: {len(contents)} chars")
        return contents

    except Exception as e:
        print(f"[Final] Error generating final contents: {e}")
        return f"Error generating contents: {str(e)}"


# --- UI Construction ---


@ui.page("/")
def index():
    # Apply MD3 Colors
    ui.colors(primary="#6750A4", secondary="#625B71", accent="#7D5260")
    ui.query("body").style(
        'background-color: #FFFBFE; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "Noto Sans", Helvetica, Arial, sans-serif, "Apple Color Emoji", "Segoe UI Emoji";'
    )

    # Check if CUDA mode is selected but PyTorch CUDA is not installed
    if state.config.get("hardware_mode") == "cuda":
        torch_installed, cuda_version = check_torch_cuda_installed()
        if not torch_installed:
            detected_cuda = detect_cuda_version()
            if detected_cuda:
                # Show notification to install PyTorch CUDA
                ui.timer(1.0, lambda: ui.notify(
                    f"检测到您选择了 CUDA 模式，但 PyTorch CUDA 未安装。请在设置中点击「下载 PyTorch CUDA」按钮进行安装。",
                    type="warning",
                    timeout=10000
                ), once=True)

    # --- Top Bar ---
    # --- Top Bar (Modern/Flat) ---
    with ui.header().classes(
        "bg-transparent text-[#1C1B1F] shadow-none h-24 items-center px-6"
    ):
        # Toggle Drawer
        ui.button(on_click=lambda: left_drawer.toggle(), icon="menu").props(
            "flat round color=primary size=lg"
        )
        ui.label("OpenAutoNote").classes("text-4xl font-bold ml-4 text-[#1C1B1F]")
        ui.space()

    # --- Custom CSS for Report ---
    # --- Custom CSS for Report & MD3 ---
    ui.add_head_html("""
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.css">
    <script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.js"></script>
    <script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/contrib/auto-render.min.js" onload="renderMathInElement(document.body);"></script>
    <script>
        function rerenderMath(element) {
            if (typeof renderMathInElement !== 'undefined' && element) {
                try {
                    renderMathInElement(element, {
                        delimiters: [
                            {left: '$$', right: '$$', display: true},
                            {left: '$', right: '$', display: false},
                            {left: '\\\\[', right: '\\\\]', display: true},
                            {left: '\\\\(', right: '\\\\)', display: false}
                        ]
                    });
                } catch (e) {
                    console.log('Math render error:', e);
                }
            }
        }
        
        function findAndRenderMath(root) {
            const rootElement = root || document.body;
            rerenderMath(rootElement);
        }
        
        if (typeof MutationObserver !== 'undefined') {
            let debounceTimer = null;
            const observer = new MutationObserver(function(mutations) {
                if (debounceTimer) {
                    clearTimeout(debounceTimer);
                }
                debounceTimer = setTimeout(function() {
                    mutations.forEach(function(mutation) {
                        if (mutation.type === 'childList' && mutation.addedNodes.length) {
                            mutation.addedNodes.forEach(function(node) {
                                if (node.nodeType === 1) {
                                    findAndRenderMath(node);
                                }
                            });
                        }
                        if (mutation.type === 'childList') {
                            const target = mutation.target;
                            if (target && target.classList) {
                                if (target.classList.contains('q-markdown') || 
                                    target.classList.contains('prose') || 
                                    target.classList.contains('report-content') ||
                                    target.closest('.q-markdown')) {
                                    findAndRenderMath(target);
                                }
                            }
                        }
                    });
                }, 200);
            });
            
            const startObserver = function() {
                try {
                    observer.observe(document.body, {
                        childList: true,
                        subtree: true,
                        characterData: true
                    });
                } catch (e) {
                    console.log('Observer start error:', e);
                }
            };
            
            if (document.readyState === 'loading') {
                document.addEventListener('DOMContentLoaded', startObserver);
            } else {
                startObserver();
            }
        }
    </script>
    <style>
        body {
            background-color: #FFFBFE; /* MD3 Background */
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "Noto Sans", Helvetica, Arial, sans-serif, "Apple Color Emoji", "Segoe UI Emoji";
        }
        .md3-card {
            background-color: #F3EDF7; /* Surface Container Low */
            border-radius: 24px;
            padding: 24px;
            /* No shadow, just color */
        }
        .md3-input .q-field__control {
            border-radius: 20px !important; /* Rounder inputs */
            padding-left: 12px;
            padding-right: 12px;
        }
        
        /* Report Styles - Compact Layout */
        .report-content { font-size: 1rem; }
        .report-content h1 { font-size: 1.8em; color: #1C1B1F; font-weight: 700; margin-bottom: 0.5em; margin-top: 0; }
        .report-content h2 { font-size: 1.4em; color: #49454F; font-weight: 600; margin-top: 1.2em; margin-bottom: 0.5em; border-bottom: 2px solid #E7E0EC; padding-bottom: 6px; }
        .report-content h3 { font-size: 1.15em; color: #49454F; font-weight: 600; margin-top: 1em; margin-bottom: 0.4em; }
        .report-content p { font-size: 0.95em; line-height: 1.6; color: #49454F; margin-bottom: 0.8em; }
        .report-content blockquote { border-left: 4px solid #6750A4; background: #F3EDF7; padding: 10px 14px; margin: 12px 0; border-radius: 0 12px 12px 0; color: #49454F; font-style: italic; font-size: 0.95em; }
        .report-content ul { list-style-type: none; padding-left: 0; margin-bottom: 0.8em; }
        .report-content li { margin-bottom: 0.3em; line-height: 1.6; padding-left: 0.3em; font-size: 0.95em; }
        /* Image Constraint - Smaller and compact */
        .report-content img { 
            max-width: 55%; 
            display: block; 
            margin: 16px auto; 
            border-radius: 12px; 
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            border: 1px solid #E7E0EC; 
        }
        .report-content table { width: 100%; border-collapse: separate; border-spacing: 0; margin: 12px 0; border: 1px solid #E7E0EC; border-radius: 12px; overflow: hidden; font-size: 0.9em; }
        .report-content th { background-color: #F3EDF7; padding: 8px 12px; border-bottom: 1px solid #E7E0EC; font-weight: 600; text-align: left; color: #1C1B1F; }
        .report-content td { padding: 8px 12px; border-bottom: 1px solid #E7E0EC; color: #49454F; }
        .report-content tr:last-child td { border-bottom: none; }
        .report-content tr:nth-child(even) { background-color: #FFFBFE; }
        /* LaTeX Math Styles */
        .katex { font-size: 1.1em !important; }
        .katex-display { margin: 16px 0 !important; }
    </style>
    """)

    # --- Settings Dialog (Function) ---
    def open_settings():
        with (
            ui.dialog() as dialog,
            ui.card().classes("w-full max-w-[500px] p-5 rounded-xl bg-white shadow-lg"),
        ):
            # --- 标题栏 ---
            with ui.row().classes("w-full items-center justify-between mb-2"):
                ui.label(
                    get_text("nav_settings_title", state.config["ui_language"])
                ).classes("text-lg font-bold text-gray-800")
                ui.button(icon="close", on_click=dialog.close).props(
                    "flat round dense color=grey size=sm"
                )

            # --- Tabs ---
            with ui.tabs().classes("w-full text-primary") as tabs:
                tab_api = ui.tab(
                    get_text("tab_api", state.config["ui_language"])
                ).classes("h-10 min-h-0 text-sm")
                tab_gen = ui.tab(
                    get_text("tab_gen", state.config["ui_language"])
                ).classes("h-10 min-h-0 text-sm")
                tab_hw = ui.tab(
                    get_text("tab_hardware", state.config["ui_language"])
                ).classes("h-10 min-h-0 text-sm")
                tab_sys = ui.tab(
                    get_text("tab_system", state.config["ui_language"])
                ).classes("h-10 min-h-0 text-sm")

            # --- 面板区域 ---
            with ui.tab_panels(tabs, value=tab_gen).classes(
                "w-full mt-2 bg-transparent max-h-[60vh] overflow-y-auto"
            ):
                # API 设置
                with ui.tab_panel(tab_api).classes("p-1 flex flex-col gap-3"):
                    ui.input(
                        get_text("lbl_api_key", state.config["ui_language"]),
                        password=True,
                    ).bind_value(state.config, "api_key").classes("w-full").props(
                        'outlined dense item-aligned input-class="text-sm"'
                    )
                    ui.input(
                        get_text("lbl_base_url", state.config["ui_language"])
                    ).bind_value(state.config, "base_url").classes("w-full").props(
                        'outlined dense item-aligned input-class="text-sm"'
                    )
                    ui.input(
                        get_text("lbl_model", state.config["ui_language"])
                    ).bind_value(state.config, "model_name").classes("w-full").props(
                        'outlined dense item-aligned input-class="text-sm"'
                    )

                    ui.separator().classes("opacity-30 my-1")
                    ui.label(
                        get_text("lbl_stt_section", state.config["ui_language"])
                    ).classes("text-sm font-bold text-gray-700")
                    ui.select(
                        label=get_text("lbl_stt_backend", state.config["ui_language"]),
                        options={
                            "qwen3-asr-flash-filetrans": get_text(
                                "stt_backend_filetrans", state.config["ui_language"]
                            ),
                            "qwen3-asr-flash": get_text(
                                "stt_backend_flash", state.config["ui_language"]
                            ),
                            "qwen3-asr-local": get_text(
                                "stt_backend_local", state.config["ui_language"]
                            ),
                            "faster-whisper": get_text(
                                "stt_backend_whisper", state.config["ui_language"]
                            ),
                        },
                    ).bind_value(state.config, "stt_backend").classes("w-full").props(
                        "outlined dense options-dense"
                    )
                    ui.input(
                        get_text("lbl_dashscope_key", state.config["ui_language"]),
                        password=True,
                    ).bind_value(state.config, "dashscope_api_key").classes(
                        "w-full"
                    ).props(
                        'outlined dense item-aligned input-class="text-sm" placeholder="DASHSCOPE_API_KEY"'
                    )
                    ui.select(
                        label=get_text(
                            "lbl_dashscope_region", state.config["ui_language"]
                        ),
                        options={
                            "beijing": get_text(
                                "dashscope_region_beijing",
                                state.config["ui_language"],
                            ),
                            "intl": get_text(
                                "dashscope_region_intl", state.config["ui_language"]
                            ),
                        },
                    ).bind_value(state.config, "dashscope_region").classes(
                        "w-full"
                    ).props("outlined dense options-dense")
                    ui.label(
                        get_text("dashscope_key_hint", state.config["ui_language"])
                    ).classes("text-xs text-gray-500")
                    
                    # 深度思考开关
                    with ui.row().classes("w-full justify-between items-center px-1"):
                        ui.label(
                            get_text("lbl_deep_thinking", state.config["ui_language"])
                        ).classes("text-sm text-gray-700")
                        ui.switch().bind_value(state.config, "deep_thinking").props(
                            "dense color=primary size=sm"
                        )
                    
                    # API连通性测试按钮
                    test_api_btn = ui.button(
                        get_text("btn_test_api", state.config["ui_language"]),
                        on_click=None  # Will be set after function definition
                    ).classes("w-full").props("outline color=primary")
                    
                    async def test_api_connectivity():
                        if not state.config["api_key"]:
                            ui.notify("请先设置API密钥", type="negative")
                            print("[API Test] Error: API Key missing.")
                            return
                        if not state.config["base_url"]:
                            ui.notify("请先设置Base URL", type="negative")
                            print("[API Test] Error: Base URL missing.")
                            return
                        if not state.config["model_name"]:
                            ui.notify("请先设置模型名称", type="negative")
                            print("[API Test] Error: Model name missing.")
                            return
                        
                        # 启用加载状态
                        test_api_btn.set_text("测试中...")
                        test_api_btn.props('loading')
                        
                        print("[API Test] Starting API connectivity test...")
                        
                        try:
                            from openai import AsyncOpenAI
                            client = AsyncOpenAI(
                                api_key=state.config["api_key"], 
                                base_url=state.config["base_url"]
                            )
                            
                            # 测试API连通性，使用文本+图片请求
                            # First, try a simple text request
                            test_prompt = "Hello, this is a connectivity test. Please respond with 'Connection successful' only."
                            
                            # Try text-only request first
                            response = await client.chat.completions.create(
                                model=state.config["model_name"],
                                messages=[{"role": "user", "content": test_prompt}],
                                max_tokens=20,
                                temperature=0.7
                            )
                            
                            # 记录详细的响应信息
                            print(f"[API Test] Full response type: {type(response)}")
                            print(f"[API Test] Full response: {response}")
                            print(f"[API Test] Response choices: {response.choices}")
                            print(f"[API Test] First choice: {response.choices[0]}")
                            print(f"[API Test] Message: {response.choices[0].message}")
                            
                            # 检查响应内容是否存在
                            message_content = response.choices[0].message.content
                            if message_content is None:
                                # 某些API（如Google Gemini）可能返回None内容，我们需要处理这种情况
                                result = "Connection successful (no content returned)"
                                print(f"[API Test] Text API test successful but no content returned: {message_content}")
                            else:
                                result = message_content.strip()
                                print(f"[API Test] Text API test successful: {result}")
                            
                            # If text request succeeded, try a multimodal request if vision is enabled
                            if state.config.get("enable_vision", True):
                                try:
                                    # Create a simple multimodal request with a base64 encoded placeholder image
                                    # This is a minimal base64 encoded 16x16 pixel PNG image to meet minimum dimension requirements
                                    # Simple red square with white border - compatible with all APIs including Google Gemini
                                    # Generated specifically to meet Google Gemini API image processing requirements
                                    placeholder_image = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg=="
                                    
                                    # 构建多模态请求
                                    multimodal_messages = [
                                        {
                                            "role": "user",
                                            "content": [
                                                {"type": "text", "text": "This is a test of multimodal API connectivity. Please respond with 'Multimodal test successful' only."},
                                                {
                                                    "type": "image_url",
                                                    "image_url": {
                                                        "url": f"data:image/png;base64,{placeholder_image}"
                                                    }
                                                }
                                            ]
                                        }
                                    ]
                                    
                                    print(f"[API Test] Multimodal request messages: {multimodal_messages}")
                                    print(f"[API Test] Image URL length: {len(multimodal_messages[0]['content'][1]['image_url']['url'])}")
                                    print(f"[API Test] Image URL format: {multimodal_messages[0]['content'][1]['image_url']['url'][:50]}...")
                                    print(f"[API Test] Base64 image data length: {len(placeholder_image)}")
                                    print(f"[API Test] Base64 image data: {placeholder_image}")
                                    
                                    multimodal_response = await client.chat.completions.create(
                                        model=state.config["model_name"],
                                        messages=multimodal_messages,
                                        max_tokens=30,
                                        temperature=0.7
                                    )
                                    
                                    # 记录详细的多模态响应信息
                                    print(f"[API Test] Multimodal response type: {type(multimodal_response)}")
                                    print(f"[API Test] Multimodal response: {multimodal_response}")
                                    print(f"[API Test] Multimodal response choices: {multimodal_response.choices}")
                                    print(f"[API Test] Multimodal first choice: {multimodal_response.choices[0]}")
                                    print(f"[API Test] Multimodal message: {multimodal_response.choices[0].message}")
                                    
                                    # 检查多模态响应内容是否存在
                                    multimodal_content = multimodal_response.choices[0].message.content
                                    if multimodal_content is None:
                                        multimodal_result = "Multimodal test successful (no content returned)"
                                        print(f"[API Test] Multimodal API test successful but no content returned: {multimodal_content}")
                                    else:
                                        multimodal_result = multimodal_content.strip()
                                        print(f"[API Test] Multimodal API test successful: {multimodal_result}")
                                    ui.notify(f"API连接成功 - 文本: {result}, 多模态: {multimodal_result}", type="positive")
                                except Exception as multimodal_error:
                                    # If multimodal fails, still report text success
                                    print(f"[API Test] Multimodal API test failed: {str(multimodal_error)}")
                                    ui.notify(f"API连接成功 - 文本: {result}, 多模态测试失败: {str(multimodal_error)}", type="warning")
                            else:
                                print("[API Test] Vision disabled, skipping multimodal test")
                                ui.notify(f"API连接成功: {result}", type="positive")
                            
                            print("[API Test] API connectivity test completed successfully")
                            
                        except Exception as e:
                            print(f"[API Test] API connectivity test failed: {str(e)}")
                            ui.notify(f"API连接失败: {str(e)}", type="negative")
                        finally:
                            # 恢复按钮状态
                            test_api_btn.set_text(get_text("btn_test_api", state.config["ui_language"]))
                            test_api_btn.props(remove='loading')
                    
                    test_api_btn.on('click', test_api_connectivity)

                # 生成设置
                with ui.tab_panel(tab_gen).classes("p-1 flex flex-col gap-3"):
                    with ui.row().classes("w-full justify-between items-center px-1"):
                        ui.label(
                            get_text("lbl_enable_vision", state.config["ui_language"])
                        ).classes("text-sm text-gray-700")
                        ui.switch().bind_value(state.config, "enable_vision").props(
                            "dense color=primary size=sm"
                        )

                    ui.separator().classes("opacity-30 my-1")

                    with ui.row().classes("w-full items-center gap-3"):
                        ui.label(
                            get_text("lbl_vision_interval", state.config["ui_language"])
                        ).classes("text-xs text-gray-500 w-16")
                        slider = (
                            ui.slider(min=5, max=60, step=5)
                            .bind_value(state.config, "vision_interval")
                            .classes("flex-grow")
                            .props("dense color=primary")
                        )
                        ui.label().bind_text_from(
                            slider, "value", lambda v: f"{v}s"
                        ).classes("text-xs font-mono w-6 text-right")

                    ui.select(
                        label=get_text(
                            "lbl_vision_detail", state.config["ui_language"]
                        ),
                        options={
                            "low": get_text(
                                "vision_detail_low", state.config["ui_language"]
                            ),
                            "high": get_text(
                                "vision_detail_high", state.config["ui_language"]
                            ),
                            "auto": get_text(
                                "vision_detail_auto", state.config["ui_language"]
                            ),
                        },
                        value=state.config.get("vision_detail", "low"),
                    ).bind_value(state.config, "vision_detail").classes("w-full").props(
                        "outlined dense options-dense"
                    )

                    ui.separator().classes("opacity-30 my-1")

                    # --- AI Model Management ---
                    ui.label("AI Model Management").classes(
                        "text-sm font-bold text-gray-700"
                    )
                    
                    # Button to open model save location
                    def open_model_folder():
                        import subprocess
                        import platform
                        import os
                        
                        # Get model cache directory
                        model_cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
                        
                        # Open directory based on OS
                        if platform.system() == 'Darwin':  # macOS
                            subprocess.run(['open', model_cache_dir])
                        elif platform.system() == 'Windows':  # Windows
                            subprocess.run(['explorer', model_cache_dir])
                        else:  # Linux
                            subprocess.run(['xdg-open', model_cache_dir])
                    
                    ui.button(
                        "Open Model Folder", 
                        icon="folder_open", 
                        on_click=open_model_folder
                    ).props("flat dense color=primary size=sm").classes("mt-2")

                    model_rows = {}

                    # Initialize with a placeholder to avoid validation error
                    current_model = state.config.get("asr_model", "small")
                    model_select = (
                        ui.select(
                            label="Current Model",
                            options={
                                current_model: current_model
                            },  # Temporary placeholder
                            value=current_model,
                        )
                        .bind_value(state.config, "asr_model")
                        .classes("w-full")
                        .props("outlined dense options-dense")
                    )

                    model_status_area = ui.column().classes("w-full gap-2")

                    async def refresh_model_ui():
                        statuses = model_manager.get_model_statuses(
                            state.config.get("hardware_mode", hardware_info["type"])
                        )

                        # Build new options dict (don't modify in place)
                        new_options = {
                            m[
                                "key"
                            ]: f"{m['display']} ({'Installed' if m['installed'] else 'Not downloaded'})"
                            for m in statuses
                        }

                        # Update select: completely replace options
                        model_select.options = new_options
                        if model_select.value not in new_options:
                            model_select.value = next(iter(new_options.keys()), "small")
                            state.config["asr_model"] = model_select.value
                        model_select.update()

                        # Clear tracking dict and UI container BEFORE rebuilding
                        model_rows.clear()
                        model_status_area.clear()
                        for m in statuses:
                            with model_status_area:
                                with ui.row().classes(
                                    "w-full items-center justify-between bg-white rounded-lg p-2 shadow-sm"
                                ):
                                    ui.label(m["display"]).classes(
                                        "text-sm font-medium"
                                    )
                                    status_text = (
                                        "✅ Installed"
                                        if m["installed"]
                                        else "☁️ Not downloaded"
                                    )
                                    status_label = ui.label(status_text).classes(
                                        "text-xs text-gray-600"
                                    )

                                    progress = (
                                        ui.linear_progress(value=0)
                                        .classes("w-32")
                                        .style("display:none")
                                        .props("rounded color=primary")
                                    )

                                    if m["installed"]:
                                        btn = ui.button("Ready").props("flat disabled")
                                    else:

                                        async def handle_download(model_key=m["key"]):
                                            btn.disable()
                                            status_label.text = "Downloading..."
                                            progress.style("display:block")
                                            progress.props("indeterminate")
                                            mirror = (
                                                "https://hf-mirror.com"
                                                if state.config.get("use_china_mirror")
                                                else None
                                            )
                                            try:
                                                await run.io_bound(
                                                    model_manager.download_model,
                                                    model_key,
                                                    state.config.get(
                                                        "hardware_mode",
                                                        hardware_info["type"],
                                                    ),
                                                    mirror,
                                                    None,
                                                )
                                                ui.notify(
                                                    f"Model '{model_key}' downloaded.",
                                                    type="positive",
                                                )
                                            except Exception as e:
                                                ui.notify(
                                                    f"Download failed: {str(e)}",
                                                    type="negative",
                                                )
                                            progress.style("display:none")
                                            btn.enable()
                                            await refresh_model_ui()

                                        btn = ui.button(
                                            "Download",
                                            on_click=lambda mk=m[
                                                "key"
                                            ]: handle_download(mk),
                                        ).props("color=primary outline")

                                    model_rows[m["key"]] = {
                                        "status_label": status_label,
                                        "btn": btn,
                                        "progress": progress,
                                    }

                    ui.timer(0.1, refresh_model_ui, once=True)

                    ui.label(
                        get_text("cookies_netscape_format", state.config["ui_language"])
                    ).classes("text-sm font-bold text-gray-700")
                    ui.label(
                        get_text("paste_cookies_directly", state.config["ui_language"])
                    ).classes("text-xs text-gray-500 mb-1 w-full").style(
                        "white-space: pre-wrap; overflow-wrap: anywhere;"
                    )
                    ui.textarea(
                        label=get_text(
                            "lbl_youtube_cookie", state.config["ui_language"]
                        )
                    ).bind_value(state.config, "cookies_yt").classes("w-full").props(
                        'outlined dense rows=3 placeholder="# Netscape HTTP Cookie File..."'
                    )
                    ui.textarea(
                        label=get_text(
                            "lbl_bilibili_cookie", state.config["ui_language"]
                        )
                    ).bind_value(state.config, "cookies_bili").classes("w-full").props(
                        "outlined dense rows=3"
                    )

                # 硬件设置
                with ui.tab_panel(tab_hw).classes("p-1 gap-4 flex flex-col"):
                    # Status Badge
                    status_color = (
                        "positive" if hardware_info["type"] != "cpu" else "warning"
                    )
                    with ui.row().classes(
                        f"w-full bg-{status_color}-50 p-3 rounded-lg border border-{status_color}-200 items-center gap-3"
                    ):
                        ui.icon("memory", size="sm").classes(f"text-{status_color}")
                        # Hardware Status Section with Refresh Button
                        with ui.row().classes("w-full items-center gap-2"):
                            with ui.column().classes("gap-0"):
                                ui.label("Detected Hardware:").classes(
                                    "text-xs text-gray-500 uppercase font-bold"
                                )
                                ui.label(hardware_info["name"]).classes(
                                    "text-sm font-medium text-gray-900"
                                )
                            
                            ui.space()
                            
                            def refresh_hardware_status():
                                """Refresh hardware status display"""
                                current_status = get_torch_install_status()
                                
                                status_parts = []
                                if current_status["torch_installed"]:
                                    status_parts.append(f"PyTorch: {current_status['torch_version'][:8]}")
                                    if current_status["cuda_available"]:
                                        status_parts.append(f"CUDA: {current_status['cuda_version']}")
                                        status_parts.append(f"GPU: {current_status['device_name']}")
                                        hardware_status_label.text = "✅ GPU 已就绪"
                                        hardware_status_label.classes("text-sm font-medium text-green-600")
                                    else:
                                        hardware_status_label.text = "⚠️ PyTorch 无 CUDA"
                                        hardware_status_label.classes("text-sm font-medium text-yellow-600")
                                else:
                                    hardware_status_label.text = "❌ 未安装 PyTorch"
                                    hardware_status_label.classes("text-sm font-medium text-red-600")
                                
                                if status_parts:
                                    hardware_detail_label.text = " | ".join(status_parts)
                                    hardware_detail_label.classes("text-xs text-gray-500")
                                else:
                                    hardware_detail_label.text = "点击刷新获取状态"
                                    hardware_detail_label.classes("text-xs text-gray-400")
                            
                            refresh_btn = ui.button(icon="refresh").props("flat round dense").tooltip("刷新硬件状态").on_click(
                                lambda: (refresh_hardware_status(), ui.notify("状态已刷新", type="info", timeout=1000))
                            )
                        
                        # Hardware Status Detail Label
                        hardware_status_label = ui.label("点击刷新获取状态").classes("text-sm font-medium text-gray-400")
                        hardware_detail_label = ui.label("").classes("text-xs text-gray-400")
                        
                        ui.separator()

                    hardware_select = ui.select(
                        label=get_text(
                            "lbl_hardware_mode", state.config["ui_language"]
                        ),
                        options={
                            "cpu": get_text(
                                "hardware_mode_cpu", state.config["ui_language"]
                            ),
                            "cuda": get_text(
                                "hardware_mode_cuda", state.config["ui_language"]
                            ),
                            "mlx": get_text(
                                "hardware_mode_mlx", state.config["ui_language"]
                            ),
                        },
                        value=state.config.get("hardware_mode", "mlx"),
                    ).bind_value(state.config, "hardware_mode").classes("w-full").props(
                        "outlined dense"
                    )
                    
                    # Torch CUDA installation helper for CUDA mode
                    torch_status_row = ui.row().classes("w-full items-center gap-2 hidden")
                    progress_label = ui.label("").classes("text-xs text-gray-600 hidden")
                    torch_install_btn = ui.button(
                        "下载 PyTorch CUDA", icon="download"
                    ).classes("hidden").props("outlined")
                    
                    async def check_and_update_torch_status():
                        """Check torch status and show install button if needed"""
                        if hardware_select.value == "cuda":
                            torch_installed, cuda_version = check_torch_cuda_installed()
                            if not torch_installed:
                                detected_cuda = detect_cuda_version()
                                if detected_cuda:
                                    torch_install_btn.text = f"下载 PyTorch CUDA (检测到 CUDA {detected_cuda})"
                                    torch_status_row.classes(remove="hidden")
                                    torch_install_btn.classes(remove="hidden")
                                else:
                                    torch_status_row.classes(add="hidden")
                            else:
                                torch_status_row.classes(add="hidden")
                        else:
                            torch_status_row.classes(add="hidden")
                    
                    async def install_torch_handler():
                        """Handle torch installation"""
                        torch_install_btn.disable()
                        torch_install_btn.text = "正在下载..."
                        progress_label.classes(remove="hidden")
                        progress_label.text = "检测 CUDA 版本..."
                        
                        def progress_callback(msg):
                            progress_label.text = msg
                            progress_label.update()
                        
                        detected_cuda = detect_cuda_version()
                        if not detected_cuda:
                            ui.notify("❌ 未检测到 CUDA，无法安装 PyTorch CUDA 版本", type="negative")
                            torch_install_btn.enable()
                            torch_install_btn.text = "下载 PyTorch CUDA"
                            return
                        
                        success = await run.io_bound(
                            install_torch_cuda,
                            detected_cuda,
                            progress_callback
                        )
                        
                        if success:
                            ui.notify("✅ PyTorch CUDA 安装成功！请重启应用以生效。", type="positive", timeout=5000)
                            torch_status_row.classes(add="hidden")
                            progress_label.classes(add="hidden")
                        else:
                            ui.notify("❌ PyTorch CUDA 安装失败，请查看终端日志", type="negative", timeout=5000)
                            torch_install_btn.enable()
                            torch_install_btn.text = "重试下载 PyTorch CUDA"
                    
                    torch_install_btn.on_click(install_torch_handler)
                    hardware_select.on("update:model-value", check_and_update_torch_status)
                    
                    # Initial check after a short delay
                    ui.timer(0.5, check_and_update_torch_status, once=True)

                # 系统设置
                with ui.tab_panel(tab_sys).classes("p-1"):
                    ui.select(
                        {"zh": "中文", "en": "English"},
                        label=get_text("lbl_ui_lang", state.config["ui_language"]),
                    ).bind_value(state.config, "ui_language").on_value_change(
                        lambda: (
                            save_config(state.config),
                            ui.run_javascript("location.reload()"),
                        )
                    ).classes("w-full").props("outlined dense")

            # 底部按钮
            with ui.row().classes("w-full justify-end mt-4 gap-2"):
                ui.button(
                    get_text("close", state.config["ui_language"]),
                    on_click=dialog.close,
                ).props("flat dense color=primary")
                ui.button(
                    get_text("save", state.config["ui_language"]),
                    on_click=lambda: (save_config(state.config), dialog.close()),
                ).props("unelevated dense color=primary")

        dialog.open()

    # --- Left Sidebar (Floating/Rounded) ---
    with (
        ui.left_drawer(value=True)
        .props("width=240")
        .classes(
            "bg-[#F7F2FA] rounded-r-[28px] column no-wrap shadow-sm border-none"
        ) as left_drawer
    ):
        # History List
        with ui.scroll_area().classes("col flex-grow q-pa-xs"):
            # Header row with title and clear all button
            with ui.row().classes("w-full items-center justify-between px-2 py-1"):
                ui.label(
                    get_text("nav_history_title", state.config["ui_language"])
                ).classes("text-xs font-bold text-grey-6 uppercase tracking-wide")

                # Clear All Button
                async def confirm_clear_all():
                    with ui.dialog() as confirm_dialog, ui.card().classes("p-4"):
                        ui.label(
                            get_text(
                                "confirm_delete_all_history",
                                state.config["ui_language"],
                            )
                        ).classes("text-base mb-4")
                        ui.label(
                            get_text(
                                "operation_irreversible", state.config["ui_language"]
                            )
                        ).classes("text-sm text-red-500 mb-4")
                        with ui.row().classes("w-full justify-end gap-2"):
                            ui.button(
                                get_text("cancel", state.config["ui_language"]),
                                on_click=confirm_dialog.close,
                            ).props("flat")

                            async def do_clear():
                                await run.io_bound(clear_all_history, delete_files=True)
                                confirm_dialog.close()
                                new_note_handler()
                                history_list.refresh()
                                ui.notify(
                                    get_text(
                                        "all_records_deleted",
                                        state.config["ui_language"],
                                    ),
                                    type="positive",
                                )

                            ui.button(
                                get_text("delete_all", state.config["ui_language"]),
                                on_click=do_clear,
                            ).props("color=negative")
                    confirm_dialog.open()

                ui.button(icon="delete_sweep", on_click=confirm_clear_all).props(
                    "flat round dense size=xs color=grey-5"
                ).tooltip(get_text("btn_clear_history", state.config["ui_language"]))

            # Sync history on load (remove orphan entries)
            sync_history()
            
            # Validate and cleanup sessions (remove invalid/stale sessions)
            validate_and_cleanup_sessions()

            @ui.refreshable
            def history_list():
                sessions = load_history()
                if not sessions:
                    ui.label(
                        get_text("nav_no_history", state.config["ui_language"])
                    ).classes("text-grey-5 text-sm italic px-3 py-2")
                    return

                # Fixed-height list items with context menu
                with ui.column().classes("w-full gap-0"):
                    for s in sessions:
                        session_id = s["id"]
                        session_title = s["title"]

                        # Pre-create rename dialog for this item
                        rename_dialog = ui.dialog()
                        with rename_dialog, ui.card().classes("p-4 min-w-[300px]"):
                            ui.label(
                                get_text("rename", state.config["ui_language"])
                            ).classes("text-lg font-bold mb-3")
                            new_name_input = ui.input(
                                get_text("new_title", state.config["ui_language"]),
                                value=session_title,
                            ).classes("w-full mb-3")
                            with ui.row().classes("w-full justify-end gap-2"):
                                ui.button(
                                    get_text("cancel", state.config["ui_language"]),
                                    on_click=rename_dialog.close,
                                ).props("flat")

                                def make_save_handler(dlg, inp, sid):
                                    async def save_rename():
                                        if inp.value and inp.value.strip():
                                            await run.io_bound(rename_session, sid, inp.value.strip())
                                            dlg.close()
                                            history_list.refresh()
                                            if (
                                                state.current_session
                                                and state.current_session["id"] == sid
                                            ):
                                                state.current_session = get_session(sid)
                                                main_content.refresh()
                                            ui.notify(
                                                get_text(
                                                    "rename_successful",
                                                    state.config["ui_language"],
                                                ),
                                                type="positive",
                                            )

                                    return save_rename

                                ui.button(
                                    get_text("save", state.config["ui_language"]),
                                    on_click=make_save_handler(
                                        rename_dialog, new_name_input, session_id
                                    ),
                                ).props("color=primary")

                        # Fixed height item - single line, no wrap
                        with ui.element("div").classes("w-full"):
                            # Add status indicator
                            status = s.get("status", "completed")
                            status_icon = ""
                            if status == "processing":
                                status_icon = "⏳ "
                            elif status == "downloaded":
                                status_icon = "📥 "
                            elif status == "transcribed":
                                status_icon = "📝 "
                            elif status == "completed":
                                status_icon = "✅ "
                            elif status == "error":
                                status_icon = "❌ "
                            
                            # Display progress if available
                            progress_text = s.get("progress", "")
                            display_text = f"{status_icon}{s['title']}"
                            if progress_text:
                                display_text += f" ({progress_text})"
                            
                            # Add loading animation for processing status
                            if status == "processing":
                                ui.button(
                                    display_text,
                                    on_click=lambda e, id=session_id: load_session(id),
                                ).props("flat align=left no-caps").classes(
                                    "w-full h-9 px-2 text-left text-sm text-grey-9 rounded-lg hover:bg-[#E8DEF8] overflow-hidden animate-pulse"
                                ).style(
                                    "white-space: nowrap; text-overflow: ellipsis; justify-content: flex-start;"
                                )
                            else:
                                ui.button(
                                    display_text,
                                    on_click=lambda e, id=session_id: load_session(id),
                                ).props("flat align=left no-caps").classes(
                                    "w-full h-9 px-2 text-left text-sm text-grey-9 rounded-lg hover:bg-[#E8DEF8] overflow-hidden"
                                ).style(
                                    "white-space: nowrap; text-overflow: ellipsis; justify-content: flex-start;"
                                )

                            # Context menu
                            with ui.context_menu() as menu:

                                def open_rename(dlg=rename_dialog):
                                    dlg.open()

                                def make_delete_handler(sid):
                                    async def delete_handler():
                                        try:
                                            await delete_sess_handler(sid)
                                        except Exception as e:
                                            print(f"Error in delete_handler: {e}")
                                            ui.notify(f"删除失败: {e}", type="negative")
                                    return delete_handler

                                ui.menu_item(
                                    get_text("rename", state.config["ui_language"]),
                                    on_click=open_rename,
                                )
                                ui.separator()
                                ui.menu_item(
                                    get_text("delete", state.config["ui_language"]),
                                    on_click=make_delete_handler(session_id),
                                ).classes("text-red-500")

            history_list()

        # Bottom Actions
        with ui.column().classes("col-auto q-pa-sm w-full border-t border-gray-100"):
            ui.button(
                get_text("nav_settings_title", state.config["ui_language"]),
                icon="settings",
                on_click=open_settings,
            ).props(
                "flat rounded block align=left text-color-[#1C1B1F] w-full"
            ).classes("rounded-full hover:bg-[#E8DEF8]")

        # Floating Action Button (FAB) for New Note
        # We place it "fixed" absolute relative to drawer or main?
        # User requested "prominent FAB".
        # Let's put it inside the drawer at bottom right corner overlay? Or just top right of drawer list.
        # Actually classic Material Design puts FAB in main area bottom right, or sidebar top.
        # Let's put a nice button at the Top of Sidebar list actually, or overlay.
        # Implementation: Fixed position button inside drawer creates issues with scrolling.
        # Let's try placing it at the very top of drawer content as a "New Chat" primary action.
        # Wait, user asked: "placed at the top or bottom right of the sidebar."
        # Lets put it at the Top for clarity.
        pass  # We'll insert it via layout above history list if we want, but FAB implies floating.
        # Let's do a proper FAB in the Bottom Right of the Drawer?
        # ui.button(icon='add').props('fab color=primary').classes('absolute-bottom-right q-ma-md')
        # But this might overlap settings.
        # Let's put it top-right of sidebar.

    # FAB is better placed in the Drawer stack
    with left_drawer:
        ui.button(icon="add", on_click=lambda: new_note_handler()).props(
            "fab color=primary"
        ).classes("absolute-bottom-right q-ma-md shadow-lg z-10").tooltip(
            get_text("nav_new_note", state.config["ui_language"])
        )

    # --- Main Content Area ---
    main_container = ui.column().classes(
        "w-full max-w-5xl mx-auto q-pa-md items-stretch gap-8"
    )

    @ui.refreshable
    def main_content():
        # Clear container content but preserve layout classes
        main_container.clear()
        # Only reset classes if not in split view mode to preserve layout state
        if state.current_session:
            # In split view, keep the wider layout
            main_container.classes(remove="max-w-5xl mx-auto", add="w-full max-w-[95%]")
        else:
            # In normal view, use standard layout
            main_container.classes(remove="w-full max-w-[95%]", add="w-full max-w-5xl mx-auto")
        
        with main_container:
            if state.current_session:
                render_history_view(state.current_session)
            else:
                render_input_view()

    async def delete_sess_handler(sess_id):
        # Prevent concurrent deletions
        if state._deleting_session:
            return
            
        state._deleting_session = True
        
        try:
            # Handle current session if it's the one being deleted
            if state.current_session and state.current_session["id"] == sess_id:
                state.current_session = None
                main_content.refresh()
            
            # Optimistic UI update: Remove from UI first
            sessions = load_history()
            filtered_sessions = [s for s in sessions if s["id"] != sess_id]
            save_history(filtered_sessions)
            
            # Then delete files in background (this can be slower)
            await run.io_bound(delete_session, sess_id)
            
            # Reload the page to avoid DOM conflicts
            ui.run_javascript("location.reload()")
            
        except Exception as e:
            print(f"Error deleting session: {e}")
            ui.notify(f"删除失败: {e}", type="negative")
        finally:
            state._deleting_session = False

    def load_session(sess_id):
        state.current_session = get_session(sess_id)
        main_content.refresh()

    def new_note_handler():
        state.current_session = None
        main_content.refresh()

    # --- Logic: View Renderers ---

    def convert_timestamps_to_images(summary_content, project_dir):
        """
        Convert timestamp markers [MM:SS] to image tags using local video frames.
        
        Args:
            summary_content: The summary text with timestamp markers
            project_dir: The project directory containing the assets folder
            
        Returns:
            Modified content with timestamp markers replaced by image tags
        """
        if not summary_content or not project_dir:
            return summary_content
            
        # Pattern to match timestamp markers like [07:15] or [12:34-15:20]
        timestamp_pattern = r'\[((\d{1,2}):(\d{2}))(?:-(\d{1,2}):(\d{2}))?\]'
        
        def replace_timestamp(match):
            # Extract timestamp components
            full_match = match.group(0)
            start_minutes = int(match.group(2))
            start_seconds = int(match.group(3))
            
            # Convert to total seconds for frame filename
            total_seconds = start_minutes * 60 + start_seconds
            
            # Construct frame filename (using 4-digit format)
            frame_filename = f"frame_{total_seconds:04d}.jpg"
            
            # Construct image path
            image_path = os.path.join(project_dir, "assets", frame_filename)
            
            # Check if the frame file exists
            if os.path.exists(image_path):
                # Return Markdown image tag with timestamp as alt text
                return f"![Frame at {match.group(1)}]({image_path})"
            else:
                # If frame doesn't exist, return the original timestamp marker
                # This preserves the timestamp for debugging
                logger.warning(f"Frame not found: {image_path}")
                return full_match
        
        # Replace all timestamp markers with image tags
        modified_content = re.sub(timestamp_pattern, replace_timestamp, summary_content)
        
        return modified_content

    def render_history_view(session):
        # Layout classes are now managed by main_content() to prevent state corruption

        # State
        project_dir = session.get("project_dir", "")
        chat_messages = load_chat_history(project_dir) if project_dir else []

        # Selection state
        selection_state = {"text": "", "len": 0}

        # Splitter Layout
        splitter = ui.splitter(value=70).classes(
            "w-full h-[calc(100vh-140px)] rounded-2xl border border-gray-200 bg-[#FAFAFA] overflow-hidden shadow-sm"
        )

        # Chat Toggle FAB
        chat_fab = (
            ui.button(icon="smart_toy", on_click=lambda: splitter.set_value(70))
            .props("fab color=primary")
            .classes("fixed top-24 right-8 z-50 shadow-lg hidden")
            .tooltip(get_text("expand_ai_assistant", state.config["ui_language"]))
        )

        def update_fab_visibility():
            if splitter.value >= 98:
                chat_fab.classes(remove="hidden")
            else:
                chat_fab.classes(add="hidden")

        splitter.on("change", update_fab_visibility)

        # --- Left: Summary Content ---
        with splitter.before:
            with ui.scroll_area().classes(
                "h-full w-full bg-[#FAFAFA]"
            ) as scroll_container:
                content_div = ui.column().classes("w-full p-6 max-w-full mx-auto")
                with content_div:
                    # Header
                    ui.label(session["title"]).classes(
                        "text-3xl font-bold tracking-tight text-[#1C1B1F] leading-tight mb-2"
                    )
                    with ui.row().classes(
                        "items-center gap-2 text-[#49454F] text-sm mb-6"
                    ):
                        ui.link(session["video_url"], session["video_url"]).classes(
                            "hover:text-primary truncate max-w-md"
                        )
                        ui.label("•")
                        ui.label(session.get("timestamp", "")[:10])

                    # Transcript Expansion
                    with ui.expansion(
                        get_text(
                            "view_original_transcript", state.config["ui_language"]
                        ),
                        icon="description",
                    ).classes(
                        "w-full mb-4 bg-white rounded-xl border border-gray-100 shadow-sm"
                    ):
                        ui.label(session.get("transcript", "No transcript")).classes(
                            "text-grey-8 font-mono text-xs leading-relaxed p-4 whitespace-pre-wrap max-w-full break-words"
                        )

                    ui.separator().classes("mb-6")

                    # Summary Report - Convert timestamps to images before rendering
                    summary_content = session.get("summary", "")
                    processed_summary = convert_timestamps_to_images(summary_content, project_dir)
                    ui.markdown(processed_summary, extras=['latex', 'tables']).classes(
                        "w-full prose prose-lg prose-slate report-content max-w-none"
                    )

                    # Enable TOC anchor links after page loads with enhanced DOM safety
                    async def enable_toc_links():
                        await ui.run_javascript("""
                        (function() {
                            // Enhanced DOM safety with comprehensive error handling
                            function safeProcessTOC() {
                                try {
                                    const container = document.querySelector('.report-content');
                                    if (!container || !container.isConnected) {
                                        console.warn('TOC container not found or disconnected');
                                        return;
                                    }
                                    
                                    const headings = container.querySelectorAll('h2, h3');
                                    headings.forEach(h => {
                                        if (h && h.isConnected) {
                                            try {
                                                const text = h.textContent.replace(/^[\\s🎯⚡💰📊🔬📑🏙️🌆🏛️💼🌊👤]*/, '').trim();
                                                const id = text.replace(/[\\s:：]+/g, '-').toLowerCase();
                                                h.id = id;
                                            } catch (e) {
                                                console.warn('Failed to process heading:', e);
                                            }
                                        }
                                    });
                                    
                                    const links = container.querySelectorAll('a[href^="#"]');
                                    links.forEach(a => {
                                        if (a && a.isConnected) {
                                            try {
                                                a.style.cursor = 'pointer';
                                                
                                                // Remove existing listeners to prevent duplicates
                                                a.removeEventListener('click', handleTOCClick);
                                                a.addEventListener('click', handleTOCClick);
                                            } catch (e) {
                                                console.warn('Failed to setup TOC link:', e);
                                            }
                                        }
                                    });
                                    
                                    // TOC click handler with comprehensive DOM safety
                                    function handleTOCClick(e) {
                                        e.preventDefault();
                                        try {
                                            const href = this.getAttribute('href');
                                            const targetId = decodeURIComponent(href.substring(1));
                                            let target = document.getElementById(targetId);
                                            
                                            if (!target || !target.isConnected) {
                                                const searchText = targetId.replace(/-/g, '').toLowerCase();
                                                headings.forEach(h => {
                                                    if (h && h.isConnected) {
                                                        const hText = h.textContent.replace(/[\\s:：🎯⚡💰📊🔬📑🏙️🌆🏛️💼🌊👤]/g, '').toLowerCase();
                                                        if (hText.includes(searchText) || searchText.includes(hText)) {
                                                            target = h;
                                                        }
                                                    }
                                                });
                                            }
                                            
                                            if (target && target.isConnected) {
                                                target.scrollIntoView({ behavior: 'smooth', block: 'start' });
                                            }
                                        } catch (error) {
                                            console.warn('TOC navigation failed:', error);
                                        }
                                    }
                                    
                                } catch (error) {
                                    console.error('TOC processing failed:', error);
                                }
                            }
                            
                            safeProcessTOC();
                        })();
                        """)

                    ui.timer(0.5, enable_toc_links, once=True)

                # Capture Selection
                async def check_selection():
                    text = await ui.run_javascript("window.getSelection().toString()")
                    if text and len(text.strip()) > 0:
                        selection_state["text"] = text.strip()
                        selection_state["len"] = len(text.strip())
                        sel_indicator.classes(remove="hidden")
                        sel_label.text = get_text(
                            "selection_label", state.config["ui_language"]
                        ).format(count=selection_state["len"])
                    else:
                        selection_state["text"] = ""
                        selection_state["len"] = 0
                        sel_indicator.classes(add="hidden")

                # Bind mouseup to check selection
                scroll_container.on("mouseup", check_selection)

        # --- Right: Chat Panel ---
        with splitter.after:
            chat_panel = ui.column().classes(
                "h-full w-full bg-white border-l border-gray-200"
            )
            with chat_panel:
                # Chat Header
                with ui.row().classes(
                    "w-full items-center justify-between p-3 border-b border-gray-100"
                ):
                    with ui.row().classes("items-center gap-2"):
                        ui.icon("smart_toy", size="xs").classes("text-primary")
                        ui.label(
                            get_text("ai_assistant", state.config["ui_language"])
                        ).classes("font-bold text-sm text-gray-800")

                    # Toggle Fullscreen/Hide (modifies splitter)
                    def toggle_width():
                        if splitter.value > 90:
                            splitter.set_value(70)
                        else:
                            splitter.set_value(100)
                        update_fab_visibility()

                    ui.button(icon="last_page", on_click=toggle_width).props(
                        "flat round dense size=sm text-color=grey-6"
                    ).tooltip(get_text("collapse_expand", state.config["ui_language"]))

                # Chat Messages (Flat Design)
                chat_scroll = ui.scroll_area().classes("flex-grow w-full p-4")
                with chat_scroll:
                    messages_column = ui.column().classes("w-full gap-6")

                # Streaming State
                streaming_md = {"ref": None}

                def render_messages():
                    messages_column.clear()
                    with messages_column:
                        if not chat_messages:
                            ui.label(
                                get_text(
                                    "ask_questions_about_content",
                                    state.config["ui_language"],
                                )
                            ).classes(
                                "text-gray-400 text-xs italic text-center w-full mt-10"
                            )

                        for msg in chat_messages:
                            role = msg.get("role", "user")
                            content = msg.get("content", "")

                            with ui.column().classes("w-full gap-1"):
                                # Header
                                if role == "user":
                                    ui.label("You").classes(
                                        "font-bold text-xs text-gray-900 bg-gray-100 px-2 py-0.5 rounded-md self-start"
                                    )
                                else:
                                    ui.label(
                                        get_text(
                                            "ai_label", state.config["ui_language"]
                                        )
                                    ).classes(
                                        "font-bold text-xs text-primary bg-purple-50 px-2 py-0.5 rounded-md self-start"
                                    )

                                # Content
                                if role == "user":
                                    # Check for image
                                    if (
                                        get_text("image", state.config["ui_language"])
                                        in content
                                    ):
                                        ui.label(content).classes(
                                            "text-sm text-gray-800 whitespace-pre-wrap"
                                        )
                                    else:
                                        ui.label(content).classes(
                                            "text-sm text-gray-800 whitespace-pre-wrap"
                                        )
                                else:
                                    ui.markdown(content, extras=['latex', 'tables']).classes(
                                        "text-sm text-gray-800 prose prose-sm max-w-none"
                                    )

                render_messages()

                # Selection Indicator (Floating over input)
                with ui.row().classes(
                    "w-full px-4 py-1 hidden bg-yellow-50 border-t border-yellow-100 items-center justify-between"
                ) as sel_indicator:
                    sel_label = ui.label(
                        get_text("selection_label", state.config["ui_language"]).format(
                            count=0
                        )
                    ).classes("text-xs text-yellow-800")
                    ui.button(icon="close", on_click=lambda: check_selection()).props(
                        "flat round dense size=xs color=yellow-800"
                    )

                # Input Area
                with ui.column().classes(
                    "w-full p-3 border-t border-gray-200 gap-2 bg-gray-50"
                ):
                    # Image Preview
                    img_preview = ui.image().classes(
                        "w-32 h-32 object-cover rounded-lg hidden border border-gray-300"
                    )
                    uploaded_img = {"base64": None}

                    def handle_file(e):
                        import base64

                        content = e.content.read()
                        b64 = base64.b64encode(content).decode()
                        uploaded_img["base64"] = b64
                        img_preview.props(f'src="data:image/png;base64,{b64}"')
                        img_preview.classes(remove="hidden")

                    # Hidden Uploader
                    uploader = ui.upload(
                        on_upload=handle_file, auto_upload=True, max_files=1
                    ).classes("hidden")

                    with ui.row().classes("w-full items-end gap-2"):
                        # Custom Upload Button Trigger linked to hidden uploader
                        ui.button(
                            icon="image",
                            on_click=lambda: uploader.run_method("pickFiles"),
                        ).props("flat round dense color=grey-7").classes(
                            "w-8 h-8"
                        ).tooltip(get_text("upload_image", state.config["ui_language"]))

                        chat_input = (
                            ui.textarea(
                                placeholder=get_text(
                                    "shift_enter_newline", state.config["ui_language"]
                                )
                            )
                            .props("outlined dense rows=1 auto-grow bg-color=white")
                            .classes("flex-grow text-sm")
                            .on("keydown.enter.prevent", lambda: send_message())
                        )

                        async def send_message():
                            text = chat_input.value.strip()
                            if not text and not uploaded_img["base64"]:
                                return

                            # Construct Message
                            final_text = text
                            # Append selection context if exists
                            if selection_state["text"]:
                                final_text += f"\n\n【{get_text('quoted_context', state.config['ui_language'])}】\n{selection_state['text']}"
                                # Reset selection
                                await ui.run_javascript(
                                    "window.getSelection().removeAllRanges()"
                                )
                                check_selection()

                            user_msg = {"role": "user", "content": final_text}
                            if uploaded_img["base64"]:
                                user_msg["image"] = uploaded_img["base64"]
                                user_msg["content"] = (
                                    f"[{get_text('image', state.config['ui_language'])}] {final_text}"
                                )

                            chat_messages.append(user_msg)
                            chat_input.value = ""
                            img_preview.classes(add="hidden")
                            uploaded_img["base64"] = None

                            render_messages()
                            chat_scroll.scroll_to(percent=1.0)

                            # ... API Call (Stream) ...
                            # Reuse similar logic but with new flat UI
                            chat_messages.append({"role": "assistant", "content": ""})
                            with messages_column:
                                with ui.column().classes("w-full gap-1"):
                                    ui.label("AI").classes(
                                        "font-bold text-xs text-primary bg-purple-50 px-2 py-0.5 rounded-md self-start"
                                    )
                                    streaming_md["ref"] = ui.markdown("▌", extras=['latex', 'tables']).classes(
                    "text-sm text-gray-800 prose prose-sm max-w-none"
                )
                            chat_scroll.scroll_to(percent=1.0)

                            try:
                                from openai import AsyncOpenAI

                                client = AsyncOpenAI(
                                    api_key=state.config["api_key"],
                                    base_url=state.config["base_url"],
                                )

                                context = f"{get_text('ai_title_label', state.config['ui_language'])}{session.get('title')}\n{get_text('ai_summary_label', state.config['ui_language'])}{session.get('summary')[:1000]}"
                                api_msgs = [
                                    {
                                        "role": "system",
                                        "content": f"{get_text('ai_assistant_prompt', state.config['ui_language'])}\n{context}",
                                    }
                                ]

                                # Add history
                                for msg in chat_messages[:-1]:
                                    content_obj = msg["content"]
                                    if msg.get("image") and state.config.get(
                                        "enable_vision"
                                    ):
                                        content_obj = [
                                            {"type": "text", "text": msg["content"]},
                                            {
                                                "type": "image_url",
                                                "image_url": {
                                                    "url": f"data:image/jpeg;base64,{msg['image']}"
                                                },
                                            },
                                        ]
                                    api_msgs.append(
                                        {"role": msg["role"], "content": content_obj}
                                    )

                                try:
                                    response = await client.chat.completions.create(
                                        model=state.config["model_name"],
                                        messages=api_msgs,
                                        stream=True,
                                    )
                                except Exception as e:
                                    # Fallback for chat interface
                                    error_str = str(e)
                                    if "ChatCompletionRequestMultiContent" in error_str or "InvalidParameter" in error_str or "400" in error_str:
                                         # Rebuild messages as text-only
                                         text_only_msgs = []
                                         for m in api_msgs:
                                             content = m["content"]
                                             if isinstance(content, list):
                                                 # Extract text from list
                                                 text_part = next((item["text"] for item in content if item["type"] == "text"), "")
                                                 text_only_msgs.append({"role": m["role"], "content": text_part})
                                             else:
                                                 text_only_msgs.append(m)
                                         
                                         response = await client.chat.completions.create(
                                            model=state.config["model_name"],
                                            messages=text_only_msgs,
                                            stream=True,
                                        )
                                    else:
                                        raise e
                                full_res = ""
                                async for chunk in response:
                                    if chunk.choices[0].delta.content:
                                        full_res += chunk.choices[0].delta.content
                                        streaming_md["ref"].set_content(full_res + "▌")
                                        chat_scroll.scroll_to(percent=1.0)
                                streaming_md["ref"].set_content(full_res)
                                chat_messages[-1]["content"] = full_res
                                if project_dir:
                                    save_chat_history(project_dir, chat_messages)
                            except Exception as e:
                                streaming_md["ref"].set_content(f"Error: {e}")

                        ui.button(icon="send", on_click=send_message).props(
                            "round unelevated color=primary size=sm"
                        )

    def render_input_view():
        # Store local file path
        selected_local_file = {"path": None, "task_id": None}
        url_input_ref = {"input": None}

        # Create upload dialog with proper DOM initialization
        upload_dialog = ui.dialog()
        with upload_dialog, ui.card().classes("w-96"):
            ui.label("Upload Local Video").classes("text-xl font-bold mb-4")
            
            async def handle_upload(e):
                try:
                    import uuid
                    from datetime import datetime
                    
                    # Generate task ID
                    task_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    task_uuid = str(uuid.uuid4())[:8]
                    task_id = f"{task_timestamp}_{task_uuid}"
                    
                    base_temp = GENERATE_DIR
                    task_dir = os.path.join(base_temp, task_id)
                    raw_dir = os.path.join(task_dir, "raw")
                    assets_dir = os.path.join(task_dir, "assets")
                    
                    os.makedirs(raw_dir, exist_ok=True)
                    os.makedirs(assets_dir, exist_ok=True)
                    
                    file_name = e.file.name
                    target_path = os.path.join(raw_dir, file_name)
                    
                    # Read content from e.file (async)
                    content = await e.file.read()
                    with open(target_path, 'wb') as f:
                        f.write(content)
                    
                    ui.notify(f"Uploaded: {file_name}")
                    
                    # Update UI state instead of running analysis immediately
                    if url_input_ref["input"]:
                        url_input_ref["input"].value = f"Local File: {file_name}"
                        url_input_ref["input"].disable()
                    selected_local_file["path"] = target_path
                    selected_local_file["task_id"] = task_id
                    
                    # Close dialog after upload completes with delay and DOM safety
                    await asyncio.sleep(0.1)  # Allow UI to update
                    
                    # Add DOM safety cleanup before closing
                    await ui.run_javascript("""
                    (function() {
                        const domMonitor = window.oanDomSafetyMonitor;
                        if (!domMonitor) return;
                        
                        // Cleanup upload dialog references
                        const uploadDialog = domMonitor.safeQuerySelector('.q-dialog, .q-uploader, [class*="upload"], [class*="dialog"]');
                        if (uploadDialog && uploadDialog.isConnected) {
                            // Remove safety wrappers from all elements
                            const allElements = uploadDialog.querySelectorAll('*');
                            allElements.forEach(element => {
                                domMonitor.cleanupElementReferences(element);
                            });
                            
                            // Remove specific upload button safety wrappers
                            const uploadButtons = uploadDialog.querySelectorAll('button, input, .q-btn');
                            uploadButtons.forEach(btn => {
                                if (btn && btn._safe_onclick) {
                                    btn.onclick = btn._safe_onclick;
                                    delete btn._safe_onclick;
                                }
                            });
                        }
                        
                        // Stop upload monitoring
                        domMonitor.stop();
                        
                        console.log('Upload dialog DOM safety cleanup completed');
                    })();
                    """, timeout=2.0)
                    
                    upload_dialog.close()
                except Exception as e:
                    print(f"Error in handle_upload: {e}")
                    ui.notify(f"Upload failed: {e}", type="negative")
            
            # Create uploader with proper error handling and DOM safety
            uploader_ref = {"uploader": None}
            try:
                uploader_ref["uploader"] = ui.upload(
                    auto_upload=True,
                    on_upload=handle_upload,
                    max_files=1
                ).props("accept=.mp4,.mov,.mkv,.mp3,.wav,.m4a flat bordered").classes("w-full")
            except Exception as e:
                print(f"Uploader creation error: {e}")
                ui.label("Upload component failed to load").classes("text-red-500")
            with ui.row().classes("w-full justify-end mt-4"):
                ui.button(get_text("cancel", state.config["ui_language"]), on_click=upload_dialog.close).props("flat color=primary")

        # Input Card
        with ui.card().classes("w-full max-w-3xl self-center md3-card shadow-none"):
            ui.label(
                get_text("create_new_summary", state.config["ui_language"])
            ).classes("text-xl font-bold mb-4 text-[#1C1B1F]")

            # Capsule-Style Input with Button inside
            with ui.row().classes("w-full gap-2 items-center"):
                with (
                    ui.input(
                        placeholder=get_text("paste_url_here", state.config["ui_language"])
                    )
                    .props('rounded device outlined item-aligned input-class="ml-4"')
                    .classes(
                        "flex-grow text-lg rounded-full bg-white shadow-sm md3-input"
                    ) as url_input
                ):
                    pass
                
                # Store reference for handle_upload
                url_input_ref["input"] = url_input

                # Upload Button (Triggers Dialog) with proper error handling and DOM safety
                async def open_upload_dialog():
                    try:
                        # Open the dialog first
                        upload_dialog.open()
                        
                        # Add small delay to ensure dialog is rendered
                        await asyncio.sleep(0.1)
                        
                        # Initialize DOM safety for upload dialog
                        await ui.run_javascript("""
                        (function() {
                            // Enhanced upload dialog DOM safety
                            const domMonitor = window.oanDomSafetyMonitor;
                            if (!domMonitor) return;
                            
                            // Start upload-specific monitoring
                            domMonitor.startUploadMonitor();
                            
                            // Add safety wrappers to upload dialog elements
                            function enhanceUploadDialogSafety() {
                                try {
                                    // Find upload dialog container
                                    const uploadDialog = domMonitor.safeQuerySelector('.q-dialog, .q-uploader, [class*="upload"], [class*="dialog"]');
                                    if (!uploadDialog || !uploadDialog.isConnected) {
                                        console.warn('Upload dialog not found for safety enhancement');
                                        return;
                                    }
                                    
                                    // Enhance all elements in upload dialog
                                    const allElements = uploadDialog.querySelectorAll('*');
                                    allElements.forEach(element => {
                                        if (element && element.isConnected) {
                                            domMonitor.ensureElementSafety(element);
                                        }
                                    });
                                    
                                    // Add specific safety for uploader buttons and inputs
                                    const uploadButtons = uploadDialog.querySelectorAll('button, input, .q-btn, .q-uploader__file, .q-uploader__list');
                                    uploadButtons.forEach(btn => {
                                        if (btn && btn.isConnected) {
                                            // Add safety wrapper for click handlers
                                            if (typeof btn.onclick === 'function' && !btn._safe_onclick) {
                                                btn._safe_onclick = btn.onclick;
                                                btn.onclick = function(e) {
                                                    if (!this.isConnected) {
                                                        console.warn('Attempted click on disconnected upload button');
                                                        return false;
                                                    }
                                                    try {
                                                        return this._safe_onclick(e);
                                                    } catch (error) {
                                                        console.warn('Upload button click failed:', error);
                                                        return false;
                                                    }
                                                };
                                            }
                                        }
                                    });
                                    
                                    console.log('Upload dialog DOM safety enhanced');
                                } catch (error) {
                                    console.warn('Upload dialog safety enhancement failed:', error);
                                }
                            }
                            
                            // Run enhancement with retry
                            function enhanceWithRetry(retryCount = 0) {
                                if (retryCount >= 3) {
                                    console.warn('Upload dialog safety enhancement failed after 3 retries');
                                    return;
                                }
                                
                                setTimeout(() => {
                                    enhanceUploadDialogSafety();
                                    
                                    // Verify enhancement worked
                                    const uploadDialog = domMonitor.safeQuerySelector('.q-dialog, .q-uploader');
                                    if (!uploadDialog && retryCount < 2) {
                                        enhanceWithRetry(retryCount + 1);
                                    }
                                }, retryCount * 100 + 50); // 50ms, 150ms, 250ms delays
                            }
                            
                            enhanceWithRetry();
                            
                        })();
                        """, timeout=3.0)
                        
                    except Exception as e:
                        print(f"Error opening upload dialog: {e}")
                        ui.notify("Failed to open upload dialog", type="negative")

                ui.button(icon="file_upload", on_click=lambda: open_upload_dialog()) \
                    .props("round unelevated color=secondary text-color=white") \
                    .classes("w-12 h-12 shadow-sm") \
                    .tooltip("Upload local video file")

            # Complexity Selector - Moved from Advanced Options
            with ui.row().classes("w-full items-center gap-2 mt-3"):
                ui.label(
                    get_text("complexity", state.config["ui_language"])
                ).classes("text-sm text-grey-7 w-16")
                complexity_select = (
                    ui.select(
                        {
                            1: get_text(
                                "complexity_option_1",
                                state.config["ui_language"],
                            ),
                            2: get_text(
                                "complexity_option_2",
                                state.config["ui_language"],
                            ),
                            3: get_text(
                                "complexity_option_3",
                                state.config["ui_language"],
                            ),
                            4: get_text(
                                "complexity_option_4",
                                state.config["ui_language"],
                            ),
                            5: get_text(
                                "complexity_option_5",
                                state.config["ui_language"],
                            ),
                        },
                        value=3,
                    )
                    .props("outlined dense")
                    .classes("flex-grow")
                )
                
                # Chunk Summary Toggle (below complexity selector)
                with ui.row().classes("w-full items-center gap-2 mt-2"):
                    chunk_summary_toggle = (
                        ui.checkbox(
                            get_text("enable_chunk_summary", state.config["ui_language"]),
                            value=bool(state.config.get("enable_chunk_summary", False))
                        )
                        .on("update:model-value", lambda e: (
                            state.config.update({"enable_chunk_summary": bool(e.args)}),
                            save_config(state.config)
                        ))
                    )
                    ui.label(
                        get_text("chunk_summary_hint", state.config["ui_language"])
                    ).classes("text-xs text-gray-600")

            # Advanced Options Row
            with ui.expansion(
                get_text("custom_prompt_settings", state.config["ui_language"]), icon="tune"
            ).classes("w-full mt-3 bg-[#F3EDF7] rounded-xl"):
                with ui.column().classes("w-full gap-3 p-2"):
                    # Custom Prompt Textarea
                    ui.label(
                        get_text("custom_prompt", state.config["ui_language"])
                    ).classes("text-sm text-grey-7")
                    custom_prompt_input = (
                        ui.textarea(
                            placeholder=get_text(
                                "custom_prompt_placeholder", state.config["ui_language"]
                            )
                        )
                        .props("outlined rows=3")
                        .classes("w-full")
                    )

            # Start Button
            with ui.row().classes("w-full justify-end mt-3"):
                btn_start = (
                    ui.button(
                        get_text("start_analysis", state.config["ui_language"]),
                        icon="arrow_forward",
                        on_click=lambda: run_analysis(
                            url_input.value,
                            custom_prompt_input.value,
                            complexity_select.value,
                            local_file_path=selected_local_file["path"],
                            pre_task_id=selected_local_file["task_id"],
                            enable_chunk_summary=chunk_summary_toggle.value
                        ),
                    )
                    .props("unelevated color=primary size=lg")
                    .classes("rounded-full px-6")
                )

        # Stepper Container (Hidden initially)
        stepper_container = ui.column().classes(
            "w-full max-w-3xl self-center mt-8 transition-all"
        )
        
        # Progress display for history records
        def update_history_progress(session_id, progress_text):
            sessions = load_history()
            for s in sessions:
                if s["id"] == session_id:
                    s["progress"] = progress_text
                    save_history(sessions)
                    if 'history_list' in globals():
                        history_list.refresh()
                    elif 'history_list' in locals():
                        locals()['history_list'].refresh()
                    break

        # Result Card (Hidden initially)
        result_card = ui.card().classes(
            "w-full max-w-4xl self-center mt-8 md3-card shadow-none hidden"
        )

        # Transcript Section (Hidden initially)
        transcript_card = ui.card().classes(
            "w-full max-w-4xl self-center mt-8 md3-card shadow-none hidden"
        )
        with transcript_card:
            ui.label(get_text("transcript_label", state.config["ui_language"])).classes("text-xl font-bold mb-4 text-[#1C1B1F]")
            
            # Transcript Expansion
            transcript_expander = ui.expansion(
                get_text("view_original_transcript", state.config["ui_language"]), icon="description"
            ).classes("w-full mt-4 bg-blue-50 rounded")
            with transcript_expander:
                transcript_label = ui.markdown().classes(
                    "text-sm text-blue-800 p-4 max-w-full break-words whitespace-pre-wrap overflow-auto max-h-[50vh]"
                )
        
        # Load transcript from session if available
        if state.current_session and state.current_session.get("transcript"):
            transcript_card.classes(remove="hidden")
            transcript_label.set_content(state.current_session.get("transcript", "No transcript available"))
        
        # Add a timestamp to track last refresh for throttling
        last_progress_refresh = 0
        
        # Update progress in history records with throttling
        def update_progress(session_id, step, progress):
            nonlocal last_progress_refresh
            current_time = time.time()
            
            # Only update every 1 second to avoid flickering
            if current_time - last_progress_refresh < 1.0:
                return
                
            last_progress_refresh = current_time
            
            sessions = load_history()
            for s in sessions:
                if s["id"] == session_id:
                    s["progress"] = f"{step}: {progress}"
                    save_history(sessions)
                    if 'history_list' in globals():
                        history_list.refresh()
                    elif 'history_list' in locals():
                        locals()['history_list'].refresh()
                    break

        # Thread-safe UI updater helper
        # Capture the main loop in the closure
        try:
            _main_loop = asyncio.get_running_loop()
        except RuntimeError:
            _main_loop = asyncio.get_event_loop()

        def queue_ui_update(func):
            _main_loop.call_soon_threadsafe(func)

        # ANSI escape code cleaner
        import re as regex_module

        ansi_escape_pattern = regex_module.compile(
            r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])"
        )

        def clean_ansi(text):
            return ansi_escape_pattern.sub("", str(text) if text else "")

        async def run_analysis(url, custom_prompt="", complexity=3, local_file_path=None, pre_task_id=None, enable_chunk_summary=False):
            if not url and not local_file_path:
                ui.notify("Please enter a URL or upload a file", type="warning")
                return

            # Disable input
            btn_start.disable()

            # --- 0. Prepare Directory Structure ---
            import uuid
            from datetime import datetime

            if pre_task_id:
                task_id = pre_task_id
                # Reconstruct paths
                base_temp = GENERATE_DIR
                task_dir = os.path.join(base_temp, task_id)
                raw_dir = os.path.join(task_dir, "raw")
                assets_dir = os.path.join(task_dir, "assets")
            else:
                task_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                task_uuid = str(uuid.uuid4())[:8]
                task_id = f"{task_timestamp}_{task_uuid}"

                base_temp = GENERATE_DIR
                task_dir = os.path.join(base_temp, task_id)
                raw_dir = os.path.join(task_dir, "raw")
                assets_dir = os.path.join(task_dir, "assets")

                # Create Strict Hierarchy
                os.makedirs(raw_dir, exist_ok=True)
                os.makedirs(assets_dir, exist_ok=True)

            # --- Create Temporary History Record ---
            temp_session = {
                "id": task_id,
                "title": "Processing...",
                "video_url": url,
                "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
                "status": "processing",
                "project_dir": task_dir
            }
            add_session(temp_session)
            # Refresh history list to show the new record
            if 'history_list' in globals():
                history_list.refresh()
            elif 'history_list' in locals():
                locals()['history_list'].refresh()

            # Setup Stepper
            stepper_container.clear()
            with stepper_container:
                # Stepper (Horizontal, Soft)
                with (
                    ui.stepper()
                    .props("flat alternative-labels animated")
                    .classes("w-full shadow-none bg-transparent oan-stepper") as stepper
                ):
                    step_dl = ui.step("Download Video").props("icon=cloud_download")
                    step_ts = ui.step("Transcribe Audio").props("icon=graphic_eq")
                    step_ai = ui.step("AI Analysis").props("icon=psychology")

                # Initialize
                stepper.value = step_dl

            # Add a small delay to ensure the stepper is rendered before scrolling
            await asyncio.sleep(0.3)
            try:
                # Auto-scroll to show progress with enhanced DOM safety
                await ui.run_javascript("""
                (function() {
                    // Enhanced DOM safety with retry mechanism using global monitor
                    function safeScrollToStepper(retryCount = 0) {
                        // Use global DOM safety monitor if available
                        const domMonitor = window.oanDomSafetyMonitor;
                        
                        const stepper = domMonitor ? 
                            domMonitor.safeQuerySelector('.oan-stepper') : 
                            document.querySelector('.oan-stepper');
                        
                        if (stepper && stepper.isConnected) {
                            try {
                                // Use safe scrollIntoView if available
                                if (domMonitor && domMonitor.safeGetProperty(stepper, 'scrollIntoView')) {
                                    stepper.scrollIntoView({behavior: 'smooth', block: 'center'});
                                } else {
                                    stepper.scrollIntoView({behavior: 'smooth', block: 'center'});
                                }
                                return true;
                            } catch (e) {
                                console.warn('Scroll to stepper failed:', e);
                            }
                        }
                        
                        // Retry if element not found (max 3 retries)
                        if (retryCount < 3) {
                            setTimeout(() => safeScrollToStepper(retryCount + 1), 200);
                        }
                        return false;
                    }
                    safeScrollToStepper();
                })();
                """, timeout=5.0)
            except Exception as e:
                print(f"Auto-scroll failed: {e}")
            
            # Initialize global DOM safety monitoring
            await ui.run_javascript("""
            (function() {
                // Global DOM Safety Monitor
                if (window.oanDomSafetyMonitor) return;
                
                window.oanDomSafetyMonitor = {
                    observers: new Map(),
                    
                    // Enhanced DOM element access with safety checks
                    safeQuerySelector: function(selector, context = document) {
                        try {
                            const element = context.querySelector(selector);
                            return element && element.isConnected ? element : null;
                        } catch (e) {
                            console.warn('DOM query failed:', e);
                            return null;
                        }
                    },
                    
                    // Safe element property access
                    safeGetProperty: function(element, property) {
                        if (!element || !element.isConnected) {
                            console.warn('Attempted to access property on null/disconnected element:', property);
                            return null;
                        }
                        try {
                            return element[property];
                        } catch (e) {
                            console.warn('Property access failed:', e);
                            return null;
                        }
                    },
                    
                    // Safe nextSibling access with retry
                    safeNextSibling: function(element, maxRetries = 3, retryDelay = 100) {
                        if (!element || !element.isConnected) {
                            console.warn('Attempted to get nextSibling of null/disconnected element');
                            return null;
                        }
                        
                        function tryGetSibling(retryCount = 0) {
                            try {
                                const sibling = element.nextSibling;
                                if (sibling && sibling.isConnected) {
                                    return sibling;
                                }
                                
                                if (retryCount < maxRetries) {
                                    setTimeout(() => tryGetSibling(retryCount + 1), retryDelay);
                                }
                                return null;
                            } catch (e) {
                                console.warn('nextSibling access failed:', e);
                                return null;
                            }
                        }
                        
                        return tryGetSibling();
                    },
                    
                    // Monitor DOM mutations for upload-related changes
                startUploadMonitor: function() {
                    const uploadContainer = this.safeQuerySelector('.q-dialog, .q-uploader, [class*="upload"], [class*="dialog"]');
                    if (!uploadContainer) return;
                    
                    const observer = new MutationObserver((mutations) => {
                        mutations.forEach((mutation) => {
                            if (mutation.type === 'childList') {
                                mutation.addedNodes.forEach((node) => {
                                    if (node.nodeType === 1) { // Element node
                                        this.ensureElementSafety(node);
                                        
                                        // Special handling for upload-specific elements
                                        if (node.classList && (
                                            node.classList.contains('q-uploader__file') ||
                                            node.classList.contains('q-uploader__list') ||
                                            node.classList.contains('q-btn') ||
                                            node.tagName === 'BUTTON' ||
                                            node.tagName === 'INPUT'
                                        )) {
                                            this.enhanceUploadElementSafety(node);
                                        }
                                    }
                                });
                                
                                mutation.removedNodes.forEach((node) => {
                                    if (node.nodeType === 1) {
                                        this.cleanupElementReferences(node);
                                        
                                        // Special cleanup for upload elements
                                        if (node.classList && node.classList.contains('q-uploader__file')) {
                                            this.cleanupUploadElementReferences(node);
                                        }
                                    }
                                });
                            }
                        });
                    });
                    
                    observer.observe(uploadContainer, {
                        childList: true,
                        subtree: true
                    });
                    
                    this.observers.set('upload', observer);
                },
                
                // Enhanced safety for upload-specific elements
                enhanceUploadElementSafety: function(element) {
                    if (!element || !element.isConnected) return;
                    
                    // Add safety wrapper for click events
                    if (typeof element.onclick === 'function' && !element._safe_onclick) {
                        element._safe_onclick = element.onclick;
                        element.onclick = function(e) {
                            if (!this.isConnected) {
                                console.warn('Attempted click on disconnected upload element');
                                return false;
                            }
                            try {
                                return this._safe_onclick(e);
                            } catch (error) {
                                console.warn('Upload element click failed:', error);
                                return false;
                            }
                        };
                    }
                    
                    // Add safety for nextSibling access (common source of errors)
                    if (!element._safe_nextSibling) {
                        element._safe_nextSibling = element.nextSibling;
                        Object.defineProperty(element, 'nextSibling', {
                            get: function() {
                                if (!this.isConnected) {
                                    console.warn('Attempted to access nextSibling on disconnected element');
                                    return null;
                                }
                                try {
                                    return this._safe_nextSibling;
                                } catch (error) {
                                    console.warn('nextSibling access failed:', error);
                                    return null;
                                }
                            }
                        });
                    }
                },
                
                // Cleanup upload element references
                cleanupUploadElementReferences: function(element) {
                    if (element._safe_onclick) {
                        element.onclick = element._safe_onclick;
                        delete element._safe_onclick;
                    }
                    if (element._safe_nextSibling) {
                        delete element._safe_nextSibling;
                        delete Object.getOwnPropertyDescriptor(element, 'nextSibling');
                    }
                },
                    
                    // Ensure element safety by adding safety wrappers
                    ensureElementSafety: function(element) {
                        if (!element || !element.isConnected) return;
                        
                        // Add safety wrapper to common problematic methods
                        const originalMethods = ['scrollIntoView', 'querySelector', 'querySelectorAll'];
                        originalMethods.forEach(method => {
                            if (typeof element[method] === 'function' && !element[`_safe_${method}`]) {
                                element[`_safe_${method}`] = element[method];
                                element[method] = function(...args) {
                                    if (!this.isConnected) {
                                        console.warn(`Attempted ${method} on disconnected element`);
                                        return null;
                                    }
                                    try {
                                        return this[`_safe_${method}`](...args);
                                    } catch (e) {
                                        console.warn(`${method} failed:`, e);
                                        return null;
                                    }
                                };
                            }
                        });
                    },
                    
                    // Cleanup element references when removed
                    cleanupElementReferences: function(element) {
                        // Remove any safety wrappers
                        ['_safe_scrollIntoView', '_safe_querySelector', '_safe_querySelectorAll'].forEach(method => {
                            if (element[method]) {
                                delete element[method];
                            }
                        });
                    },
                    
                    // Start all monitors
                    start: function() {
                        this.startUploadMonitor();
                        console.log('OAN DOM Safety Monitor started');
                    },
                    
                    // Stop all monitors
                    stop: function() {
                        this.observers.forEach((observer, key) => {
                            observer.disconnect();
                        });
                        this.observers.clear();
                    }
                };
                
                // Start monitoring when DOM is ready
                if (document.readyState === 'loading') {
                    document.addEventListener('DOMContentLoaded', () => {
                        window.oanDomSafetyMonitor.start();
                    });
                } else {
                    window.oanDomSafetyMonitor.start();
                }
                
                // Override global console.error to catch DOM errors
                const originalConsoleError = console.error;
                console.error = function(...args) {
                    const message = args.join(' ');
                    if (message.includes('Cannot read properties of null') || 
                        message.includes('nextSibling') ||
                        message.includes('is not connected')) {
                        console.warn('DOM Safety Monitor intercepted error:', message);
                        return;
                    }
                    originalConsoleError.apply(console, args);
                };
                
            })();
            """, timeout=3.0)

            try:
                # 1. Download
                with step_dl:
                    # Progress Bar UI (No overlay, just bar + status below)
                    dl_progress = ui.linear_progress(value=0).props(
                        "rounded color=primary size=12px"
                    )
                    dl_status = ui.label("Initializing...").classes(
                        "text-sm text-grey-7 mt-2"
                    )

                # Callback for yt-dlp
                def dl_hook(d):
                    if d["status"] == "downloading":
                        try:
                            total = (
                                d.get("total_bytes")
                                or d.get("total_bytes_estimate")
                                or 1
                            )
                            downloaded = d.get("downloaded_bytes", 0)
                            percent = downloaded / total
                            speed = clean_ansi(d.get("_speed_str", "N/A"))
                            eta = clean_ansi(d.get("_eta_str", "N/A"))

                            # Update UI
                            def _update_dl():
                                dl_progress.value = percent
                                dl_status.text = (
                                    f"{percent:.1%} | Speed: {speed} | ETA: {eta}"
                                )
                            queue_ui_update(_update_dl)
                        except Exception:
                            pass
                    elif d["status"] == "finished":
                        queue_ui_update(
                            lambda: setattr(dl_status, "text", "Processing...")
                        )

                if local_file_path:
                    # Skip download, mock dl_res
                    dl_progress.value = 1.0
                    dl_status.text = "✅ Local file loaded"
                    dl_status.classes(add="text-green-700")
                    step_dl.props("icon=check color=positive")
                    
                    dl_res = {
                        "success": True,
                        "title": os.path.basename(local_file_path),
                        "video_path": local_file_path,
                        "duration": 0,
                        "description": "Local file upload",
                        "uploader": "User",
                        "upload_date": datetime.now().strftime("%Y%m%d")
                    }
                    
                    # Update history record with progress
                    update_progress(task_id, "Upload", "Completed")
                    stepper.next()
                else:
                    # Run download with domain-based cookie selection
                    cookies_yt = state.config.get("cookies_yt", "")
                    cookies_bili = state.config.get("cookies_bili", "")
                    url = clean_bilibili_url(url)
                    dl_res = await run.io_bound(
                        download_video,
                        url,
                        raw_dir,
                        cookies_yt,
                        cookies_bili,
                        True,
                        dl_hook,
                    )

                    if not dl_res["success"]:
                        # Visual Error Feedback
                        dl_progress.props("color=negative")  # Turn progress bar red
                        dl_progress.value = 1.0  # Fill it to show "failed"
                        dl_status.text = f"❌ Error: {dl_res.get('error', 'Unknown error')}"
                        dl_status.classes(add="text-red-600")

                        # Update step icon to error
                        step_dl.props("icon=error color=negative")

                        ui.notify(
                            f"Download Failed: {dl_res.get('error')}",
                            type="negative",
                            position="top",
                            close_button=True,
                            timeout=0,
                        )
                        btn_start.enable()
                        return

                    # Mark done - success state
                    dl_progress.props("color=positive")
                    dl_status.text = f"✅ Downloaded: {dl_res.get('title', 'Video')}"
                    dl_status.classes(add="text-green-700")
                    step_dl.props("icon=check color=positive")
                    
                    # Update history record with progress
                    update_progress(task_id, "Download", "Completed")
                    
                    try:
                        # Add a small delay to ensure UI is updated
                        await asyncio.sleep(0.1)
                        stepper.next()  # Move to TS
                    except RuntimeError as e:
                        print(f"Client already disconnected during download: {e}")
                        return

                # 2. Transcribe
                with step_ts:
                    ui.spinner().classes("q-ma-md")
                    lbl_ts = ui.label("Processing Audio...").classes(
                        "text-grey-6 italic"
                    )

                # Don't show transcript card during transcription - will show after AI starts

                # Define progress callback for real-time updates
                last_transcript_update = 0
                def transcript_progress_callback(transcript):
                    nonlocal last_transcript_update
                    current_time = time.time()
                    
                    # Throttle updates to once every 1.0 seconds to prevent websocket timeout
                    if current_time - last_transcript_update < 1.0:
                        return
                    last_transcript_update = current_time

                    def _update_ts():
                        # Update main page transcript only (removed stepper transcript update)
                        transcript_label.set_content(transcript)
                        # Note: We do NOT save history here to avoid IO blocking
                    queue_ui_update(_update_ts)

                try:
                    segments = await async_transcribe(
                        dl_res["video_path"], state.config["hardware_mode"], transcript_progress_callback
                    )
                except Exception as transcribe_error:
                    err = str(transcribe_error)
                    print(f"[STT] Transcription failed: {err}")
                    lbl_ts.text = f"❌ {err}"
                    step_ts.props("icon=error color=negative")
                    ui.notify(
                        f"转写失败: {err}",
                        type="negative",
                        position="top",
                        close_button=True,
                        timeout=0,
                    )
                    btn_start.enable()
                    return
                if not segments:
                    segments = []
                transcript_text = " ".join([s.get("text", "") for s in segments])
                lbl_ts.text = f"Transcribed {len(segments)} segments."
                
                # Show transcript card after transcription complete
                transcript_card.classes(remove="hidden")
                transcript_label.set_content("\n".join([f"[{s['start']:.2f}-{s['end']:.2f}] {s['text']}" for s in segments]))

                step_ts.props(add="done")
                
                # Update history record with transcription progress
                update_progress(task_id, "Transcription", "Completed")
                
                try:
                    # Add a small delay to ensure UI is updated
                    await asyncio.sleep(0.1)
                    stepper.next()  # Move to AI
                except RuntimeError as e:
                    print(f"Client already disconnected during transcription: {e}")
                    return

                # Add spinner to AI step
                with step_ai:
                    ai_spinner = ui.spinner("dots", size="lg").classes("q-ma-md")
                    lbl_ai = ui.label("Preparing AI analysis...").classes(
                        "text-grey-6 italic"
                    )

                # 3. Vision Extraction
                vision_frames = []
                if state.config["enable_vision"]:
                    try:
                        # Optional: Show sub-progress
                        # Pass assets_dir directly. visual_processor was updated to write there.
                        vision_frames = await async_vision(
                            dl_res["video_path"],
                            state.config["vision_interval"],
                            assets_dir,
                        )
                    except Exception as e:
                        print(f"Vision extraction failed: {str(e)}")
                        # Continue without vision frames
                        vision_frames = []

                # 4. Chunk Processing
                try:
                    result_card.classes(remove="hidden")
                    result_card.clear()
                    with result_card:
                        ui.label(dl_res["title"]).classes("text-2xl font-bold mb-4")

                        # Reasoning Expander
                        reasoning_exp = ui.expansion(
                            "Thinking Process (AI Reasoning)", icon="psychology"
                        ).classes("w-full mb-4 bg-purple-50 rounded hidden")
                        with reasoning_exp:
                            reasoning_label = ui.markdown(extras=['latex', 'tables']).classes(
                                "text-sm text-purple-800 p-2"
                            )

                        md_container = ui.markdown(extras=['latex', 'tables']).classes(
                        "w-full prose prose-lg max-w-none report-content"
                    )
                except RuntimeError:
                    print("Client disconnected during UI update.")
                    return

                # Decide whether to use chunk processing based on user setting
                full_reasoning = ""
                processed_timestamps = set()
                full_response = ""
                final_display_text = ""
                
                if enable_chunk_summary:
                    # New segmented summary mode
                    chunks = split_transcript_into_chunks(segments, target_duration_minutes=15)
                    all_abstracts = []
                    full_report_content = ""
                    full_abstract_content = ""
                    
                    # Process each chunk with segmented approach
                    for i, chunk in enumerate(chunks, 1):
                        if i > 1:
                            await asyncio.sleep(0.3)
                        
                        step_ai.props(f'caption="正在生成第 {i}/{len(chunks)} 部分内容..."')
                        
                        chunk_start_time = chunk['start_time']
                        chunk_end_time = chunk['end_time']
                        chunk_vision_frames = [
                            frame for frame in vision_frames 
                            if chunk_start_time <= frame['timestamp'] <= chunk_end_time
                        ]
                        
                        # Generate segmented content for this chunk
                        prev_abstracts_str = "\n\n".join(all_abstracts) if all_abstracts else ""
                        chunk_content = ""
                        separator = f"\n\n{'=' * 60}\n第 {i} 部分\n{'=' * 60}\n\n"
                        
                        def on_content_stream(partial_content):
                            nonlocal chunk_content, full_report_content
                            chunk_content = partial_content
                            # Update full report content with separator and current chunk content
                            if i == 1:
                                updated_full_content = separator + chunk_content
                            else:
                                updated_full_content = full_report_content.rsplit(separator, 1)[0] + separator + chunk_content
                            # Update UI
                            current_display = f"{updated_full_content}\n\n---\n\n**目前已生成摘要 {i-1}/{len(chunks)} 部分**"
                            try:
                                md_container.set_content(current_display)
                            except Exception:
                                pass
                        await generate_segmented_content_async(
                            i, len(chunks), chunk['text'], chunk_vision_frames, state.config, prev_abstracts_str, on_stream=on_content_stream
                        )
                        
                        # Add chunk separator and content to full report
                        if i == 1:
                            full_report_content = separator + chunk_content
                        else:
                            full_report_content = full_report_content.rsplit(separator, 1)[0] + separator + chunk_content
                        
                        # Generate abstract for this chunk
                        step_ai.props(f'caption="正在生成第 {i}/{len(chunks)} 部分摘要..."')
                        abstract = ""
                        
                        def on_abstract_stream(partial_abstract):
                            nonlocal abstract
                            abstract = partial_abstract
                            # Update UI with current abstract
                            current_display = f"{full_report_content}\n\n---\n\n**目前已生成摘要 {i}/{len(chunks)} 部分**\n{'-' * 40}\n{abstract}"
                            try:
                                md_container.set_content(current_display)
                            except Exception:
                                pass
                        
                        await generate_abstract_async(
                            i, len(chunks), chunk_content, prev_abstracts_str, state.config, on_stream=on_abstract_stream
                        )
                        
                        all_abstracts.append(abstract)
                        full_abstract_content += f"\n{'-' * 40}\n{abstract}\n"
                        
                        # Update UI with current progress
                        current_display = f"{full_report_content}\n\n---\n\n**目前已生成摘要 {i}/{len(chunks)} 部分**"
                        try:
                            md_container.set_content(current_display)
                        except Exception:
                            pass
                        
                        update_progress(task_id, "AI Analysis", f"Chunk {i}/{len(chunks)} complete")
                    
                    # Generate final contents.md from all abstracts
                    step_ai.props(f'caption="正在生成最终目录和梗概..."')
                    full_abstracts_text = "\n\n".join(all_abstracts)
                    final_contents = await generate_final_contents_async(full_abstracts_text, state.config)
                    
                    # Combine final contents with full report for report.md
                    final_report = f"{final_contents}\n\n---\n\n{full_report_content}"
                    
                    # Update UI with final result
                    try:
                        md_container.set_content(final_report)
                    except Exception:
                        pass
                    
                    full_response = final_report
                    final_display_text = final_report
                    
                    # Prepare files for download
                    report_content = final_report
                    abstract_content = full_abstract_content.strip()
                    contents_content = final_contents
                else:
                    # Non-chunk mode: process entire video at once
                    step_ai.props('caption="✍️ Generating summary..."')
                    transcript_text = " ".join([s["text"] for s in segments])
                    
                    # Process image timestamps for non-chunk mode
                    last_ui_update_time = time.time()
                    ui_update_interval = 0.5  # Update UI at most every 0.5 seconds
                    
                    async for chunk_type, chunk_text in generate_summary_stream_async(
                        dl_res["title"],
                        transcript_text,
                        segments,
                        vision_frames,
                        state.config,
                        custom_prompt,
                        complexity,
                    ):
                        current_time = time.time()
                        
                        if chunk_type == "reasoning":
                            full_reasoning += chunk_text
                            reasoning_exp.classes(remove="hidden")
                            # Throttle UI updates for reasoning
                            if current_time - last_ui_update_time >= ui_update_interval:
                                try:
                                    reasoning_label.set_content(full_reasoning)
                                    last_ui_update_time = current_time
                                except Exception:
                                    pass
                        
                        elif chunk_type == "content":
                            full_response += chunk_text
                            
                            # Process timestamps and images
                            display_text = full_response
                            timestamps = re.findall(r"\[(\d{1,2}:\d{2})\]", display_text)
                            for ts in timestamps:
                                seconds = timestamp_str_to_seconds(ts)
                                img_filename = f"frame_{seconds}.jpg"
                                img_fs_path = os.path.join(assets_dir, img_filename)
                                img_web_path = f"/generate/{task_id}/assets/{img_filename}"
                                
                                if ts not in processed_timestamps:
                                    if not os.path.exists(img_fs_path):
                                        await run.io_bound(
                                            extract_frame,
                                            dl_res["video_path"],
                                            seconds,
                                            img_fs_path,
                                        )
                                    processed_timestamps.add(ts)
                                
                                if os.path.exists(img_fs_path):
                                    if f"![{ts}]" not in display_text:
                                        display_text = display_text.replace(
                                            f"[{ts}]", f"[{ts}]\n\n![{ts}]({img_web_path})"
                                        )
                            
                            # Throttle UI updates for content
                            if current_time - last_ui_update_time >= ui_update_interval:
                                try:
                                    md_container.set_content(display_text)
                                    last_ui_update_time = current_time
                                except Exception:
                                    pass
                            
                            final_display_text = display_text
                        
                        elif chunk_type == "error":
                            print(f"[Error] {chunk_text}")
                    
                    # Final UI update for non-chunk mode with delay for robustness
                    try:
                        md_container.set_content(final_display_text)
                        await asyncio.sleep(0.3)  # Delay to ensure UI is updated
                    except Exception:
                        pass

                    # Initialize empty files for non-chunk mode
                    abstract_content = ""
                    contents_content = ""

                # Clear AI step spinner and label
                ai_spinner.delete()
                lbl_ai.delete()

                step_ai.props('caption="Completed"').props(add="done")

                # Update history record as completed
                # Update history record as completed
                update_progress(task_id, "AI Analysis", "Completed")

                ui.notify("Analysis Complete!", type="positive")

                # --- PHASE C: Atomic Finalization ---
                # Use final_display_text which contains the correct image paths
                # 添加调试日志以验证内容
                print(f"[Finalize] Debug: final_display_text exists: {'final_display_text' in dir()}")
                if "final_display_text" in dir():
                    print(f"[Finalize] Debug: final_display_text length: {len(final_display_text)}")
                
                final_content_for_save = (
                    final_display_text
                    if "final_display_text" in dir()
                    else full_response
                )
                
                print(f"[Finalize] Debug: final_content_for_save length: {len(final_content_for_save)}")
                print(f"[Finalize] Debug: final_content_for_save content preview: {final_content_for_save[:200]}")
                
                final_task_dir, final_report = finalize_task(
                    task_id, dl_res["title"], final_content_for_save, abstract_content, contents_content
                )

                # Update displayed content with corrected paths
                md_container.set_content(final_report)

                # Enable TOC anchor links via JavaScript with enhanced DOM safety
                await ui.run_javascript("""
                (function() {
                    // Enhanced DOM safety with comprehensive error handling using global monitor
                    function safeProcessTOC() {
                        try {
                            // Use global DOM safety monitor if available
                            const domMonitor = window.oanDomSafetyMonitor;
                            
                            // Find all headings in report-content with safety checks
                            const container = domMonitor ? 
                                domMonitor.safeQuerySelector('.report-content') : 
                                document.querySelector('.report-content');
                            
                            if (!container || !container.isConnected) {
                                console.warn('TOC container not found or disconnected');
                                return;
                            }
                            
                            const headings = container.querySelectorAll('h2, h3');
                            headings.forEach(h => {
                                if (h && h.isConnected) {
                                    try {
                                        // Create anchor ID from heading text (remove emojis and whitespace)
                                        const text = h.textContent.replace(/^[\\s🎯⚡💰📊🔬📑🏙️🌆🏛️💼🌊👤]*/, '').trim();
                                        const id = text.replace(/[\\s:：]+/g, '-').toLowerCase();
                                        h.id = id;
                                    } catch (e) {
                                        console.warn('Failed to process heading:', e);
                                    }
                                }
                            });
                            
                            // Enable smooth scroll for all anchor links with safety
                            const links = container.querySelectorAll('a[href^="#"]');
                            links.forEach(a => {
                                if (a && a.isConnected) {
                                    try {
                                        a.style.cursor = 'pointer';
                                        
                                        // Remove existing listeners to prevent duplicates
                                        a.removeEventListener('click', handleTOCClick);
                                        a.addEventListener('click', handleTOCClick);
                                    } catch (e) {
                                        console.warn('Failed to setup TOC link:', e);
                                    }
                                }
                            });
                            
                            // TOC click handler with comprehensive DOM safety
                            function handleTOCClick(e) {
                                e.preventDefault();
                                try {
                                    const href = this.getAttribute('href');
                                    const targetId = decodeURIComponent(href.substring(1));
                                    
                                    // Try exact match first
                                    let target = document.getElementById(targetId);
                                    
                                    if (!target || !target.isConnected) {
                                        // Fuzzy match: find heading containing the text
                                        const searchText = targetId.replace(/-/g, '').toLowerCase();
                                        headings.forEach(h => {
                                            if (h && h.isConnected) {
                                                const hText = h.textContent.replace(/[\\s:：🎯⚡💰📊🔬📑🏙️🌆🏛️💼🌊👤]/g, '').toLowerCase();
                                                if (hText.includes(searchText) || searchText.includes(hText)) {
                                                    target = h;
                                                }
                                            }
                                        });
                                    }
                                    
                                    if (target && target.isConnected) {
                                        // Use safe scrollIntoView if available
                                        if (domMonitor && domMonitor.safeGetProperty(target, 'scrollIntoView')) {
                                            target.scrollIntoView({ behavior: 'smooth', block: 'start' });
                                        } else {
                                            target.scrollIntoView({ behavior: 'smooth', block: 'start' });
                                        }
                                    }
                                } catch (error) {
                                    console.warn('TOC navigation failed:', error);
                                }
                            }
                            
                        } catch (error) {
                            console.error('TOC processing failed:', error);
                        }
                    }
                    
                    // Use retry mechanism for TOC processing
                    function processTOCWithRetry(retryCount = 0) {
                        if (retryCount >= 3) {
                            console.warn('TOC processing failed after 3 retries');
                            return;
                        }
                        
                        setTimeout(() => {
                            safeProcessTOC();
                            
                            // Verify TOC was processed successfully
                            const container = document.querySelector('.report-content');
                            if (container && container.isConnected) {
                                const headings = container.querySelectorAll('h2[id], h3[id]');
                                if (headings.length === 0 && retryCount < 2) {
                                    processTOCWithRetry(retryCount + 1);
                                }
                            }
                        }, retryCount * 200 + 100); // 100ms, 300ms, 500ms delays
                    }
                    
                    processTOCWithRetry();
                })();
                """)
                # Update existing temporary session instead of creating new one
                try:
                    sessions = load_history()
                    session_updated = False
                    for i, s in enumerate(sessions):
                        if s["id"] == task_id:
                            # 确保我们保存的是包含最终总结的正确内容
                            sessions[i] = create_session(
                                dl_res["title"],
                                url,
                                final_report,  # 保存最终报告内容
                                transcript_text,
                                final_task_dir,
                                state.config,
                            )
                            save_history(sessions)
                            session_updated = True
                            break
                    
                    # If we couldn't find and update the session, log this issue
                    if not session_updated:
                        print(f"[Warning] Could not find temporary session {task_id} to update")
                        # Try to add as a new session as fallback
                        new_session = create_session(
                            dl_res["title"],
                            url,
                            final_report,  # 保存最终报告内容
                            transcript_text,
                            final_task_dir,
                            state.config,
                        )
                        add_session(new_session)
                except Exception as update_error:
                    print(f"[Error] Failed to update session {task_id}: {update_error}")
                    # Try to add as a new session as fallback
                    try:
                        new_session = create_session(
                            dl_res["title"],
                            url,
                            final_report,  # 保存最终报告内容
                            transcript_text,
                            final_task_dir,
                            state.config,
                        )
                        add_session(new_session)
                    except Exception as fallback_error:
                        print(f"[Error] Failed to create fallback session: {fallback_error}")
                history_list.refresh()

            except Exception as e:
                import traceback
                print(traceback.format_exc())
                try:
                    ui.notify(f"Critical Error: {str(e)}", type="negative")
                except RuntimeError:
                    print("Client already disconnected, cannot show notification.")
                # Update history record with error status
                try:
                    sessions = load_history()
                    for s in sessions:
                        if s["id"] == task_id:
                            s["status"] = "error"
                            s["progress"] = f"Error: {str(e)}"
                            save_history(sessions)
                            if 'history_list' in globals():
                                history_list.refresh()
                            elif 'history_list' in locals():
                                locals()['history_list'].refresh()
                            break
                except Exception as update_e:
                    print(f"Failed to update history with error: {update_e}")
            finally:
                # Ensure btn_start exists before trying to enable it
                if 'btn_start' in globals():
                    btn_start.enable()

    # --- System Log View ---
    with ui.expansion(
        get_text("system_terminal", state.config["ui_language"]), icon="terminal"
    ).classes("w-full mt-4 bg-black text-green-400 rounded-lg"):
        log_view = ui.log(max_lines=1000).classes(
            "w-full h-64 bg-black text-green-400 font-mono text-xs p-2 rounded"
        )

    # Redirect stdout/stderr to this user's log view
    # Note: In a multi-user app, this would redirect output to the LAST connected user.
    # For a local single-user tool, this is acceptable.
    sys.stdout = WebLogger(original_stdout, log_view)
    sys.stderr = WebLogger(original_stderr, log_view)

    # Initial Render
    main_content()


def start_parent_watchdog():
    """
    Monitors the parent process (Tauri). If it dies, we die.
    """
    import threading
    import time
    import sys
    import os
    import psutil

    def watchdog():
        ppid = os.getppid()
        print(f"[Watchdog] Monitoring Parent PID: {ppid}")

        try:
            parent = psutil.Process(ppid)
        except psutil.NoSuchProcess:
            print("[Watchdog] Parent not found at startup. Exiting.")
            sys.exit(0)

        while True:
            try:
                # Check if parent is running and valid
                if not parent.is_running() or parent.status() == psutil.STATUS_ZOMBIE:
                    print(f"[Watchdog] Parent {ppid} died. Exiting.")
                    sys.exit(0)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                print(f"[Watchdog] Lost access to Parent {ppid}. Exiting.")
                sys.exit(0)

            time.sleep(1.0)

    t = threading.Thread(target=watchdog, daemon=True)
    t.start()


@app.get("/health")
def health():
    """Liveness probe for the Tauri splash / sidecar handshake."""
    return {"ok": True, "service": "openautonote"}


if __name__ in {"__main__", "__mp_main__"}:
    import multiprocessing

    multiprocessing.freeze_support()  # Critical for PyInstaller

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8964, help="Port to bind to")
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Bind address (default 127.0.0.1)",
    )
    parser.add_argument(
        "--watchdog",
        action="store_true",
        help="Exit if the parent process (Tauri) dies",
    )
    parser.add_argument(
        "--secret", type=str, default="gemini_secret", help="Storage secret"
    )
    args = parser.parse_args()

    # Only watch the Tauri parent when launched as a sidecar, not `python main.py`.
    if args.watchdog or os.environ.get("OPENAUTONOTE_WATCHDOG") == "1":
        start_parent_watchdog()

    check_first_launch_gpu_reminder()

    # NOTE: 'show=False' is critical for headless
    # 'reload=False' is recommended for the frozen binary
    print(f"Starting OpenAutoNote on {args.host}:{args.port}...")
    ui.run(
        title="OpenAutoNote",
        host=args.host,
        port=args.port,
        show=False,  # <--- Headless usage
        reload=False,  # <--- Disable auto-reload for sidecar
        storage_secret=args.secret,
        favicon="🚀",
    )

