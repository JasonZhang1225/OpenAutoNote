import os
import json
import re
import asyncio
import time
from nicegui import ui, run, app
import sys

# Get the directory where this script is located
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Capture original streams for redirection
original_stdout = sys.stdout
original_stderr = sys.stderr
from openai import AsyncOpenAI

from core.downloader import download_video
from core.transcriber import TranscriberFactory
from core.visual_processor import process_video_for_vision, extract_frame
from core import model_manager
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
CONFIG_FILE = os.path.join(BASE_DIR, "user_config.json")
DEFAULT_CONFIG = {
    "api_key": "",
    "base_url": "https://api.openai.com/v1",
    "model_name": "gpt-4o",
    "asr_model": "small",
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

    # Ensure ASR model is valid for the current hardware backend
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


state = State()


class WebLogger:
    def __init__(self, original_stream, ui_log_element):
        self.terminal = original_stream
        self.log_element = ui_log_element
        self._recursion_guard = False

    def write(self, message):
        # 1. Write to the real terminal (Keep backend working)
        self.terminal.write(message)

        # 2. Filter logic: Only filter out progress bars and very short whitespace messages
        # yt-dlp/aria2 progress bars usually start with '\r' or contain 'ETA' or '[download]'
        if not ("\r" in message and ("%" in message or "ETA" in message or "[download]" in message)):
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
def finalize_task(task_id: str, raw_title: str, report_content: str) -> tuple:
    """
    Atomic finalization: Save report.md, update paths, then rename folder.
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

    # 3. Rename folder (the atomic move)
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
        # Safety Fallback
        actual_mode = hardware_mode
        if hardware_mode == "cuda" and "cuda" not in hardware_info["valid_modes"]:
            ui.notify("⚠️ CUDA not found, falling back to CPU", type="warning")
            actual_mode = "cpu"
        elif hardware_mode == "mlx" and "mlx" not in hardware_info["valid_modes"]:
            ui.notify(
                "⚠️ Apple Neural Engine not found, falling back to CPU", type="warning"
            )
            actual_mode = "cpu"

        print(f"Transcribing with {actual_mode}...")

        # China Mirror Setup
        if state.config.get("use_china_mirror"):
            os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

        asr_model = state.config.get("asr_model", "small")
        transcriber = TranscriberFactory.get_transcriber(actual_mode, asr_model)
        
        # Pass progress callback to transcriber for real-time updates
        return transcriber.transcribe(video_path, progress_callback=progress_callback)

    ui.notify("Checking/Loading MLX Model...", type="info", timeout=3000)
    return await run.io_bound(_t)


async def async_vision(video_path, interval, output_dir):
    return await run.io_bound(
        process_video_for_vision, video_path, interval, output_dir
    )


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

async def process_chunk_recursively(chunk_index, total_chunks, chunk, dl_res, vision_frames, state, custom_prompt, complexity, chunk_context, step_ai, md_container, final_display_text, full_response, reasoning_exp, reasoning_label, task_id, assets_dir, processed_timestamps):
    """
    Recursively process a chunk, splitting it into smaller chunks if token limit is exceeded.
    Returns (processed_successfully, chunk_full_response, chunk_full_reasoning, updated_final_display_text, updated_full_response)
    """
    # Build context from previous chunks
    context_prompt = ""
    if chunk_context:
        context_prompt = f"\n\nPrevious chunk summaries for context:\n{chr(10).join(chunk_context)}\n\n"
    
    # Initialize chunk-specific variables
    chunk_full_response = ""
    chunk_full_reasoning = ""
    
    # Generate summary for this chunk
    print(f"[Recursive Chunk {chunk_index}] Starting AI summary generation...")
    chunk_content_received = False
    chunk_reasoning_received = False
    
    async for chunk_type, chunk_text in generate_summary_stream_async(
        f"{dl_res['title']} - Chunk {chunk_index}/{total_chunks}",
        chunk['text'],
        chunk['segments'],
        vision_frames,
        state.config,
        custom_prompt + f"\n\nThis is part {chunk_index} of {total_chunks} of a larger video. Please continue the numbering from previous sections if applicable. Focus on summarizing this specific chunk. Include key quotes and structured summary." + context_prompt,
        complexity,
    ):
        print(f"[Recursive Chunk {chunk_index}] Received chunk_type: {chunk_type}, text_length: {len(chunk_text)}")
        
        if chunk_type == "reasoning":
            chunk_full_reasoning += chunk_text
            reasoning_exp.classes(remove="hidden")
            reasoning_label.set_content(chunk_full_reasoning)
            chunk_reasoning_received = True
            print(f"[Recursive Chunk {chunk_index}] Reasoning content updated, total length: {len(chunk_full_reasoning)}")
        
        elif chunk_type == "content":
            chunk_full_response += chunk_text
            chunk_content_received = True
            print(f"[Recursive Chunk {chunk_index}] Content received, chunk_full_response length: {len(chunk_full_response)}")
            
            # Combine all chunk summaries so far
            current_full_response = full_response + ("\n\n---\n\n" if full_response else "") + chunk_full_response
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

            md_container.set_content(display_text)
            # Store the final display_text for finalization
            final_display_text = display_text
            print(f"[Recursive Chunk {chunk_index}] Display text updated, length: {len(display_text)}")
        
        elif chunk_type == "error":
            # Check if this is a token limit error
            if "token_limit_error" in chunk_text:
                print(f"[Recursive Chunk {chunk_index}] Token limit exceeded, splitting into smaller chunks...")
                
                # Split this chunk into smaller chunks
                num_segments = len(chunk['segments'])
                if num_segments <= 1:
                    print(f"[Recursive Chunk {chunk_index}] Cannot split further, using as is...")
                    return False, "", "", final_display_text, full_response
                
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
                success1, resp1, reason1, final_display_text, full_response = await process_chunk_recursively(
                    f"{chunk_index}.1", total_chunks, sub_chunk1, dl_res, sub_chunk1_vision_frames, state, 
                    custom_prompt, complexity, chunk_context, step_ai, md_container, final_display_text, full_response, 
                    reasoning_exp, reasoning_label
                )
                
                if success1 and resp1:
                    # Add to context for second sub-chunk
                    chunk_context.append(f"Chunk {chunk_index}.1: {resp1[:100]}...")
                
                # Process second sub-chunk
                success2, resp2, reason2, final_display_text, full_response = await process_chunk_recursively(
                    f"{chunk_index}.2", total_chunks, sub_chunk2, dl_res, sub_chunk2_vision_frames, state, 
                    custom_prompt, complexity, chunk_context, step_ai, md_container, final_display_text, full_response, 
                    reasoning_exp, reasoning_label
                )
                
                # Combine results
                if success1 or success2:
                    combined_response = resp1 + ("\n\n---\n\n" if resp1 and resp2 else "") + resp2
                    combined_reasoning = reason1 + ("\n\n---\n\n" if reason1 and reason2 else "") + reason2
                    return True, combined_response, combined_reasoning, final_display_text, full_response
                else:
                    return False, "", "", final_display_text, full_response
            else:
                # Other error, return failure
                print(f"[Recursive Chunk {chunk_index}] Non-token error occurred: {chunk_text}")
                return False, "", "", final_display_text, full_response
    
    print(f"[Recursive Chunk {chunk_index}] Summary generation completed. Content received: {chunk_content_received}, Reasoning received: {chunk_reasoning_received}")
    print(f"[Recursive Chunk {chunk_index}] Final chunk_full_response length: {len(chunk_full_response)}")
    
    return chunk_content_received, chunk_full_response, chunk_full_reasoning, final_display_text, full_response

async def generate_summary_stream_async(
    title, full_text, segments, vision_frames, config, custom_prompt="", complexity=3
):
    if not config["api_key"]:
        yield "Error: API Key missing."
        return

    client = AsyncOpenAI(api_key=config["api_key"], base_url=config["base_url"])

    # Complexity level descriptions
    complexity_levels = {
        1: get_text("complexity_level_1", state.config["ui_language"]),
        2: get_text("complexity_level_2", state.config["ui_language"]),
        3: get_text("complexity_level_3", state.config["ui_language"]),
        4: get_text("complexity_level_4", state.config["ui_language"]),
        5: get_text("complexity_level_5", state.config["ui_language"]),
    }

    complexity_instruction = complexity_levels.get(complexity, complexity_levels[3])

    # Base system identity (always included)
    # 根据UI语言动态生成base_identity内容
    if state.config["ui_language"] == "zh":
        base_identity = get_text("base_identity_zh", state.config["ui_language"])
    else:
        base_identity = get_text("base_identity_en", state.config["ui_language"])

    # Build system prompt
    # 根据UI语言设置默认生成语言
    default_lang = state.config["ui_language"]

    # 构建语言风格部分
    language_style = get_text("language_style", state.config["ui_language"]).format(
        default_lang=default_lang
    )

    # Check if this is a chunk summary
    is_chunk_summary = "Chunk" in title
    
    if custom_prompt and custom_prompt.strip():
        # Custom prompt is ADDED to base, not replacing it
        system_prompt = f"""{base_identity}

{get_text("user_extra_requirement", state.config["ui_language"])}
{custom_prompt.strip()}

{get_text("output_complexity_requirement", state.config["ui_language"])}{complexity_instruction}"""
    else:
        system_prompt = f"""{base_identity}

{get_text("output_complexity_requirement", state.config["ui_language"])}{complexity_instruction}
{get_text("core_layout_requirements", state.config["ui_language"])}

{get_text("the_one_liner", state.config["ui_language"])}
{get_text("the_one_liner_desc", state.config["ui_language"])}

{get_text("structured_toc", state.config["ui_language"])}
{get_text("structured_toc_desc", state.config["ui_language"])}

{get_text("structured_sections", state.config["ui_language"])}
{get_text("structured_sections_desc", state.config["ui_language"])}

{get_text("data_comparison", state.config["ui_language"])}
{get_text("data_comparison_desc", state.config["ui_language"])}

{get_text("math_formulas", state.config["ui_language"])}
{get_text("math_formulas_desc", state.config["ui_language"])}

{get_text("visual_evidence", state.config["ui_language"])}
{get_text("visual_evidence_desc", state.config["ui_language"])}

{language_style}
    """
    
    # Add chunk-specific requirements if needed
    if is_chunk_summary:
        system_prompt += f"\n\n{get_text('chunk_summary_requirements', state.config['ui_language'])}"

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
        
        async for chunk in response:
            chunk_count += 1
            delta = chunk.choices[0].delta
            content = getattr(delta, "content", None)
            reasoning = getattr(delta, "reasoning_content", None)

            # For models that support reasoning_content (like OpenAI o1 series)
            if reasoning:
                total_reasoning_length += len(reasoning)
                print(f"[AI API] Chunk {chunk_count}: Reasoning content received, length: {len(reasoning)}")
                yield ("reasoning", reasoning)
            # For models that don't support reasoning_content, we'll use a portion of the content as thinking process
            elif content and "Thinking Process" in system_prompt:
                # If this is a specialized thinking prompt, treat content as reasoning
                total_reasoning_length += len(content)
                print(f"[AI API] Chunk {chunk_count}: Content treated as reasoning, length: {len(content)}")
                yield ("reasoning", content)
            elif content:
                total_content_length += len(content)
                print(f"[AI API] Chunk {chunk_count}: Content received, length: {len(content)}")
                yield ("content", content)
        
        print(f"[AI API] API call completed. Total chunks: {chunk_count}, Content length: {total_content_length}, Reasoning length: {total_reasoning_length}")

    except Exception as e:
        error_msg = str(e)
        print(f"[AI API] Error during API call: {error_msg}")
        
        # Check if this is a token limit error
        is_token_limit_error = "Total tokens" in error_msg and "exceed max message tokens" in error_msg
        
        if is_token_limit_error:
            print(f"[AI API] Detected token limit error: {error_msg}")
            yield ("error", f"token_limit_error: {error_msg}")
        else:
            yield ("error", f"\n\n**Error:** {error_msg}")


# --- UI Construction ---


@ui.page("/")
def index():
    # Apply MD3 Colors
    ui.colors(primary="#6750A4", secondary="#625B71", accent="#7D5260")
    ui.query("body").style(
        'background-color: #FFFBFE; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "Noto Sans", Helvetica, Arial, sans-serif, "Apple Color Emoji", "Segoe UI Emoji";'
    )

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
                        with ui.column().classes("gap-0"):
                            ui.label("Detected Hardware:").classes(
                                "text-xs text-gray-500 uppercase font-bold"
                            )
                            ui.label(hardware_info["name"]).classes(
                                "text-sm font-medium text-gray-900"
                            )

                    ui.separator()

                    ui.select(
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

                            def do_clear():
                                clear_all_history(delete_files=True)
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
                                    def save_rename():
                                        if inp.value and inp.value.strip():
                                            rename_session(sid, inp.value.strip())
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
                            with ui.context_menu():

                                def open_rename(dlg=rename_dialog):
                                    dlg.open()

                                def delete_item(sid=session_id):
                                    delete_sess_handler(sid)

                                ui.menu_item(
                                    get_text("rename", state.config["ui_language"]),
                                    on_click=open_rename,
                                )
                                ui.separator()
                                ui.menu_item(
                                    get_text("delete", state.config["ui_language"]),
                                    on_click=delete_item,
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
        main_container.clear()
        # Reset to default classes when refreshing
        main_container.classes(remove="w-full max-w-[95%]", add="w-full max-w-5xl mx-auto")
        with main_container:
            if state.current_session:
                render_history_view(state.current_session)
            else:
                render_input_view()

    def delete_sess_handler(sess_id):
        delete_session(sess_id)
        history_list.refresh()
        if state.current_session and state.current_session["id"] == sess_id:
            new_note_handler()

    def load_session(sess_id):
        state.current_session = get_session(sess_id)
        main_content.refresh()

    def new_note_handler():
        state.current_session = None
        main_content.refresh()

    # --- Logic: View Renderers ---

    def render_history_view(session):
        # Widen container for split view
        main_container.classes(remove="max-w-5xl mx-auto", add="w-full max-w-[95%]")

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

                    # Summary Report
                    ui.markdown(session.get("summary", "")).classes(
                        "w-full prose prose-lg prose-slate report-content max-w-none"
                    )

                    # Enable TOC anchor links after page loads
                    async def enable_toc_links():
                        await ui.run_javascript("""
                        (function() {
                            const container = document.querySelector('.report-content');
                            if (!container) return;
                            
                            const headings = container.querySelectorAll('h2, h3');
                            headings.forEach(h => {
                                const text = h.textContent.replace(/^[\\s🎯⚡💰📊🔬📑🏙️🌆🏛️💼🌊👤]*/, '').trim();
                                const id = text.replace(/[\\s:：]+/g, '-').toLowerCase();
                                h.id = id;
                            });
                            
                            container.querySelectorAll('a[href^="#"]').forEach(a => {
                                a.style.cursor = 'pointer';
                                a.addEventListener('click', function(e) {
                                    e.preventDefault();
                                    const href = this.getAttribute('href');
                                    const targetId = decodeURIComponent(href.substring(1));
                                    let target = document.getElementById(targetId);
                                    if (!target) {
                                        const searchText = targetId.replace(/-/g, '').toLowerCase();
                                        headings.forEach(h => {
                                            const hText = h.textContent.replace(/[\\s:：🎯⚡💰📊🔬📑🏙️🌆🏛️💼🌊👤]/g, '').toLowerCase();
                                            if (hText.includes(searchText) || searchText.includes(hText)) {
                                                target = h;
                                            }
                                        });
                                    }
                                    if (target) {
                                        target.scrollIntoView({ behavior: 'smooth', block: 'start' });
                                    }
                                });
                            });
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
                                    ui.markdown(content).classes(
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
                                    streaming_md["ref"] = ui.markdown("▌").classes(
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
        # Define upload dialog first
        upload_dialog = None
        # Store local file path
        selected_local_file = {"path": None, "task_id": None}

        async def handle_upload(e):
            if upload_dialog:
                upload_dialog.close()
                
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
            url_input.value = f"Local File: {file_name}"
            url_input.disable()
            selected_local_file["path"] = target_path
            selected_local_file["task_id"] = task_id

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

                # Upload Button (Triggers Dialog)
                ui.button(icon="file_upload", on_click=lambda: upload_dialog.open()) \
                    .props("round unelevated color=secondary text-color=white") \
                    .classes("w-12 h-12 shadow-sm") \
                    .tooltip("Upload local video file")

            # Upload Dialog (Hidden by default)
            with ui.dialog() as upload_dialog, ui.card().classes("w-96"):
                ui.label("Upload Local Video").classes("text-xl font-bold mb-4")
                ui.upload(
                    auto_upload=True,
                    on_upload=handle_upload,
                    max_files=1
                ).props("accept=.mp4,.mov,.mkv,.mp3,.wav,.m4a flat bordered").classes("w-full")
                with ui.row().classes("w-full justify-end mt-4"):
                    ui.button(get_text("cancel", state.config["ui_language"]), on_click=upload_dialog.close).props("flat color=primary")

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
                            pre_task_id=selected_local_file["task_id"]
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

        async def run_analysis(url, custom_prompt="", complexity=3, local_file_path=None, pre_task_id=None):
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
                    .classes("w-full shadow-none bg-transparent") as stepper
                ):
                    step_dl = ui.step("Download Video").props("icon=cloud_download")
                    step_ts = ui.step("Transcribe Audio").props("icon=graphic_eq")
                    step_ai = ui.step("AI Analysis").props("icon=psychology")

                # Initialize
                stepper.value = step_dl

            # Auto-scroll to show progress
            await ui.run_javascript(
                'document.querySelector(".q-stepper").scrollIntoView({behavior: "smooth", block: "center"})'
            )

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
                        stepper.next()  # Move to TS
                    except RuntimeError:
                        print("Client already disconnected during download.")
                        return

                # 2. Transcribe
                with step_ts:
                    ui.spinner().classes("q-ma-md")
                    lbl_ts = ui.label("Processing Audio...").classes(
                        "text-grey-6 italic"
                    )
                    
                    # Add real-time transcript display in stepper
                stepper_transcript_expander = ui.expansion(
                    get_text("transcript_label", state.config["ui_language"]), icon="description"
                ).classes("w-full mt-4 bg-blue-50 rounded")
                with stepper_transcript_expander:
                    stepper_transcript_label = ui.markdown().classes(
                        "text-sm text-blue-800 p-2 max-w-full break-words whitespace-pre-wrap overflow-auto max-h-[30vh]"
                    )

                # Don't show transcript card during transcription - will show after AI starts

                # Define progress callback for real-time updates
                def transcript_progress_callback(transcript):
                    def _update_ts():
                        # Update stepper transcript
                        stepper_transcript_label.set_content(transcript)
                        # Update main page transcript
                        transcript_label.set_content(transcript)
                        # Update history record with partial transcript
                        sessions = load_history()
                        for s in sessions:
                            if s["id"] == task_id:
                                s["transcript"] = transcript
                                save_history(sessions)
                                # Refresh history list to show updated transcript
                                if 'history_list' in globals():
                                    history_list.refresh()
                                elif 'history_list' in locals():
                                    locals()['history_list'].refresh()
                                break
                    queue_ui_update(_update_ts)

                segments = await async_transcribe(
                    dl_res["video_path"], state.config["hardware_mode"], transcript_progress_callback
                )
                transcript_text = " ".join([s["text"] for s in segments])
                lbl_ts.text = f"Transcribed {len(segments)} segments."
                
                # Show transcript card after transcription complete
                transcript_card.classes(remove="hidden")
                transcript_label.set_content("\n".join([f"[{s['start']:.2f}-{s['end']:.2f}] {s['text']}" for s in segments]))

                step_ts.props(add="done")
                
                # Update history record with transcription progress
                update_progress(task_id, "Transcription", "Completed")
                
                try:
                    stepper.next()  # Move to AI
                except RuntimeError:
                    print("Client already disconnected during transcription.")
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
                            reasoning_label = ui.markdown(extras=['latex']).classes(
                                "text-sm text-purple-800 p-2"
                            )

                        md_container = ui.markdown(extras=['latex']).classes(
                        "w-full prose prose-lg max-w-none report-content"
                    )
                except RuntimeError:
                    print("Client disconnected during UI update.")
                    return

                # Split transcript into chunks
                chunks = split_transcript_into_chunks(segments, target_duration_minutes=15)
                chunk_summaries = []
                full_reasoning = ""
                processed_timestamps = set()

                # Process each chunk
                chunk_context = []
                chunk_briefs = []
                full_response = ""
                final_display_text = ""
                
                for i, chunk in enumerate(chunks, 1):
                    step_ai.props(f'caption="Processing Chunk {i}/{len(chunks)}..."')
                    
                    # Get only vision frames that fall within this chunk's time range
                    chunk_start_time = chunk['start_time']
                    chunk_end_time = chunk['end_time']
                    chunk_vision_frames = [
                        frame for frame in vision_frames 
                        if chunk_start_time <= frame['timestamp'] <= chunk_end_time
                    ]
                    
                    # Process chunk recursively
                    success, chunk_full_response, chunk_full_reasoning, final_display_text, _ = await process_chunk_recursively(
                        i, len(chunks), chunk, dl_res, chunk_vision_frames, state, custom_prompt, complexity, 
                        chunk_context, step_ai, md_container, final_display_text, full_response, 
                        reasoning_exp, reasoning_label, task_id, assets_dir, processed_timestamps
                    )
                    
                    print(f"[Chunk {i}] Summary generation completed. Success: {success}, Response length: {len(chunk_full_response)}")
                    
                    # Add completed chunk to context if successful
                    if success and chunk_full_response:
                        # Extract structured brief for context and TOC
                        # 1. Try to find the One-Liner (Blockquote)
                        one_liner_match = re.search(r"^>\s*(.*?)(?:\n|$)", chunk_full_response, re.MULTILINE)
                        one_liner = one_liner_match.group(1).strip() if one_liner_match else ""
                        
                        # 2. Extract all H2/H3 headers to capture topics
                        headers = re.findall(r"^(?:##|###)\s+(.*)", chunk_full_response, re.MULTILINE)
                        headers_text = "\n".join([f"- {h}" for h in headers])
                        
                        # 3. Construct Brief
                        if one_liner or headers:
                            brief = f"{one_liner}\n\nKey Topics:\n{headers_text}"
                        else:
                            brief = chunk_full_response[:500] + "..."
                            
                        chunk_context.append(f"Chunk {i} Summary:\n{brief}")
                        chunk_briefs.append(f"Chunk {i} Summary:\n{brief}")
                        
                        # Update full_response with the new chunk content
                        full_response += ("\n\n---\n\n" if full_response else "") + chunk_full_response
                    
                    # Update progress in history records
                    update_progress(task_id, "AI Analysis", f"Chunk {i}/{len(chunks)} complete")

                # Generate final summary from all chunk summaries
                if len(chunks) > 1:
                    step_ai.props('caption="Generating Final Summary..."')
                    final_summary_prompt = f"""
                    Please provide a comprehensive final summary of this video based on the following chunk summaries:
                    
                    {full_response}
                    
                    The final summary should:
                    1. Synthesize key insights from all chunks
                    2. Identify overarching themes and connections
                    3. Provide a cohesive narrative of the entire video
                    4. Include the most important quotes and takeaways
                    """
                    
                    final_summary_text = ""
                    # Store the content before final summary
                    content_before_final = full_response
                    
                    async for chunk_type, chunk_text in generate_summary_stream_async(
                        f"{dl_res['title']} - Final Summary",
                        final_summary_prompt,
                        [],
                        [],  # No vision frames for final summary to avoid token limit
                        state.config,
                        custom_prompt,
                        complexity,
                    ):
                        if chunk_type == "content":
                            # Accumulate final summary text
                            final_summary_text += chunk_text
                            # Append final summary to the end instead of prepending
                            full_response = f"{content_before_final}\n\n---\n\n# Final Summary\n\n{final_summary_text}"
                            md_container.set_content(full_response)
                            # 确保最终总结被正确保存到 final_display_text
                            final_display_text = full_response

                # Generate Table of Contents
                if len(chunks) > 1:
                    step_ai.props('caption="Generating Table of Contents..."')
                    toc_prompt = f"""
                    Based on the following brief summaries of the video parts, please generate a concise Table of Contents (TOC) for the final report.
                    The TOC should list the main sections/topics covered in each chunk.
                    
                    Brief Summaries:
                    {chr(10).join(chunk_briefs)}
                    
                    Format:
                    # Table of Contents
                    - [Section Title]
                    - [Section Title]
                    ...
                    """
                    
                    toc_text = ""
                    content_before_toc = full_response
                    
                    async for chunk_type, chunk_text in generate_summary_stream_async(
                        f"{dl_res['title']} - TOC",
                        toc_prompt,
                        [],
                        [],
                        state.config,
                        custom_prompt,
                        complexity,
                    ):
                        if chunk_type == "content":
                            toc_text += chunk_text
                            # Prepend TOC to the beginning
                            full_response = f"{toc_text}\n\n---\n\n{content_before_toc}"
                            md_container.set_content(full_response)
                            final_display_text = full_response

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
                    task_id, dl_res["title"], final_content_for_save
                )

                # Update displayed content with corrected paths
                md_container.set_content(final_report)

                # Enable TOC anchor links via JavaScript
                await ui.run_javascript("""
                (function() {
                    // Find all headings in report-content
                    const container = document.querySelector('.report-content');
                    if (!container) return;
                    
                    const headings = container.querySelectorAll('h2, h3');
                    headings.forEach(h => {
                        // Create anchor ID from heading text (remove emojis and whitespace)
                        const text = h.textContent.replace(/^[\\s🎯⚡💰📊🔬📑🏙️🌆🏛️💼🌊👤]*/, '').trim();
                        const id = text.replace(/[\\s:：]+/g, '-').toLowerCase();
                        h.id = id;
                    });
                    
                    // Enable smooth scroll for all anchor links
                    container.querySelectorAll('a[href^="#"]').forEach(a => {
                        a.style.cursor = 'pointer';
                        a.addEventListener('click', function(e) {
                            e.preventDefault();
                            const href = this.getAttribute('href');
                            const targetId = decodeURIComponent(href.substring(1));
                            // Try exact match first, then fuzzy match
                            let target = document.getElementById(targetId);
                            if (!target) {
                                // Fuzzy match: find heading containing the text
                                const searchText = targetId.replace(/-/g, '').toLowerCase();
                                headings.forEach(h => {
                                    const hText = h.textContent.replace(/[\\s:：🎯⚡💰📊🔬📑🏙️🌆🏛️💼🌊👤]/g, '').toLowerCase();
                                    if (hText.includes(searchText) || searchText.includes(hText)) {
                                        target = h;
                                    }
                                });
                            }
                            if (target) {
                                target.scrollIntoView({ behavior: 'smooth', block: 'start' });
                            }
                        });
                    });
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


if __name__ in {"__main__", "__mp_main__"}:
    import multiprocessing

    multiprocessing.freeze_support()  # Critical for PyInstaller

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8964, help="Port to bind to")
    parser.add_argument(
        "--secret", type=str, default="gemini_secret", help="Storage secret"
    )
    args = parser.parse_args()

    # Start Watchdog
    start_parent_watchdog()

    # NOTE: 'show=False' is critical for headless
    # 'reload=False' is recommended for the frozen binary
    print(f"Starting OpenAutoNote on port {args.port}...")
    ui.run(
        title="OpenAutoNote",
        port=args.port,
        show=False,  # <--- Headless usage
        reload=False,  # <--- Disable auto-reload for sidecar
        storage_secret=args.secret,
        favicon="🚀",
    )
