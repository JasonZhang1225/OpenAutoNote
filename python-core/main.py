import os
import json
import re
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
from core.utils import build_multimodal_payload, timestamp_str_to_seconds
from core.i18n import get_text
from core.storage import (
    load_history,
    add_session,
    get_session,
    delete_session,
    create_session,
    clear_all_history,
    rename_session,
    sync_history,
    load_chat_history,
    save_chat_history,
)

# --- Configuration & State ---
CONFIG_FILE = os.path.join(BASE_DIR, "user_config.json")
DEFAULT_CONFIG = {
    "api_key": "",
    "base_url": "https://api.openai.com/v1",
    "model_name": "gpt-4o",
    "language": "Simplified Chinese (Default)",
    "hardware_mode": "mlx",
    "enable_vision": False,
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
import subprocess


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

GENERATE_DIR = os.path.join(BASE_DIR, "generate")
if not os.path.exists(GENERATE_DIR):
    os.makedirs(GENERATE_DIR)
app.add_static_files("/generate", GENERATE_DIR)


def load_config():
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                return {**DEFAULT_CONFIG, **json.load(f)}
        except Exception:
            return DEFAULT_CONFIG.copy()
    return DEFAULT_CONFIG.copy()


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

    def write(self, message):
        # 1. Write to the real terminal (Keep backend working)
        self.terminal.write(message)

        # 2. Filter logic: Ignore empty lines or progress bars
        # yt-dlp/aria2 progress bars usually start with '\r' or contain 'ETA' or '[download]'
        # We also filter out empty whitespace messages to keep UI clean
        # Note: We check if message is just a newline to avoid double spacing if ui.log adds its own,
        # but ui.log.push usually handles strings. NiceGUI log adds new div per push.
        if message.strip() and "\r" not in message and "[download]" not in message:
            try:
                # Push to UI
                self.log_element.push(message.strip())
            except Exception:
                pass  # Avoid errors if UI is disconnected

    def flush(self):
        self.terminal.flush()


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
        import uuid

        new_folder = f"{new_folder}_{str(uuid.uuid4())[:8]}"

    report_path = os.path.join(old_folder, "report.md")

    if not os.path.exists(old_folder):
        print(f"[Finalize] Error: Folder {old_folder} not found.")
        return old_folder, report_content

    # 1. Update paths in report content BEFORE renaming
    updated_content = report_content.replace(
        f"/generate/{task_id}", f"/generate/{clean_title}"
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


async def async_transcribe(video_path, hardware_mode):
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

        transcriber = TranscriberFactory.get_transcriber(actual_mode)
        return transcriber.transcribe(video_path)

    ui.notify("Checking/Loading MLX Model...", type="info", timeout=3000)
    return await run.io_bound(_t)


async def async_vision(video_path, interval, output_dir):
    return await run.io_bound(
        process_video_for_vision, video_path, interval, output_dir
    )


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

{get_text("visual_evidence", state.config["ui_language"])}
{get_text("visual_evidence_desc", state.config["ui_language"])}

{language_style}
    """

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
        response = await client.chat.completions.create(
            model=config["model_name"],
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
            stream=True,
        )

        async for chunk in response:
            delta = chunk.choices[0].delta
            content = getattr(delta, "content", None)
            reasoning = getattr(delta, "reasoning_content", None)

            if reasoning:
                yield ("reasoning", reasoning)
            if content:
                yield ("content", content)

    except Exception as e:
        yield ("error", f"\n\n**Error:** {str(e)}")


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
                "w-full mt-2 bg-transparent"
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
                            ui.button(
                                s["title"],
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
        main_container.classes(remove="max-w-5xl", add="w-full max-w-[95%]")

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
                content_div = ui.column().classes("w-full p-6 max-w-5xl mx-auto")
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
                            "text-grey-8 font-mono text-xs leading-relaxed p-4 whitespace-pre-wrap"
                        )

                    ui.separator().classes("mb-6")

                    # Summary Report
                    ui.markdown(session.get("summary", "")).classes(
                        "w-full prose prose-lg prose-slate report-content max-w-none"
                    )

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

                                response = await client.chat.completions.create(
                                    model=state.config["model_name"],
                                    messages=api_msgs,
                                    stream=True,
                                )
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
        # Input Card
        with ui.card().classes("w-full max-w-3xl self-center md3-card shadow-none"):
            ui.label(
                get_text("create_new_summary", state.config["ui_language"])
            ).classes("text-xl font-bold mb-4 text-[#1C1B1F]")

            # Capsule-Style Input with Button inside
            with (
                ui.input(
                    placeholder=get_text("paste_url_here", state.config["ui_language"])
                )
                .props('rounded device outlined item-aligned input-class="ml-4"')
                .classes(
                    "w-full text-lg rounded-full bg-white shadow-sm md3-input"
                ) as url_input
            ):
                pass  # Button moved outside

            # Advanced Options Row
            with ui.expansion(
                get_text("advanced_options", state.config["ui_language"]), icon="tune"
            ).classes("w-full mt-3 bg-[#F3EDF7] rounded-xl"):
                with ui.column().classes("w-full gap-3 p-2"):
                    # Complexity Dropdown
                    with ui.row().classes("w-full items-center gap-2"):
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
                        ),
                    )
                    .props("unelevated color=primary size=lg")
                    .classes("rounded-full px-6")
                )

        # Stepper Container (Hidden initially)
        stepper_container = ui.column().classes(
            "w-full max-w-3xl self-center mt-8 transition-all"
        )

        # Result Card (Hidden initially)
        result_card = ui.card().classes(
            "w-full max-w-4xl self-center mt-8 md3-card shadow-none hidden"
        )

        # Thread-safe UI updater helper
        def queue_ui_update(func):
            func()

        # ANSI escape code cleaner
        import re as regex_module

        ansi_escape_pattern = regex_module.compile(
            r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])"
        )

        def clean_ansi(text):
            return ansi_escape_pattern.sub("", str(text) if text else "")

        async def run_analysis(url, custom_prompt="", complexity=3):
            if not url:
                ui.notify("Please enter a URL", type="warning")
                return

            # Disable input
            btn_start.disable()

            # --- 0. Prepare Directory Structure ---
            import uuid
            from datetime import datetime

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
                            dl_progress.value = percent
                            dl_status.text = (
                                f"{percent:.1%} | Speed: {speed} | ETA: {eta}"
                            )
                        except:
                            pass
                    elif d["status"] == "finished":
                        queue_ui_update(
                            lambda: setattr(dl_status, "text", "Processing...")
                        )

                # Run download with domain-based cookie selection
                cookies_yt = state.config.get("cookies_yt", "")
                cookies_bili = state.config.get("cookies_bili", "")
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
                stepper.next()  # Move to TS

                # 2. Transcribe
                with step_ts:
                    ui.spinner().classes("q-ma-md")
                    lbl_ts = ui.label("Processing Audio...").classes(
                        "text-grey-6 italic"
                    )

                segments = await async_transcribe(
                    dl_res["video_path"], state.config["hardware_mode"]
                )
                transcript_text = " ".join([s["text"] for s in segments])
                lbl_ts.text = f"Transcribed {len(segments)} segments."

                step_ts.props(add="done")
                stepper.next()  # Move to AI

                # Add spinner to AI step
                with step_ai:
                    ai_spinner = ui.spinner("dots", size="lg").classes("q-ma-md")
                    lbl_ai = ui.label("Preparing AI analysis...").classes(
                        "text-grey-6 italic"
                    )

                # 3. Vision Extraction
                vision_frames = []
                if state.config["enable_vision"]:
                    # Optional: Show sub-progress
                    # Pass assets_dir directly. visual_processor was updated to write there.
                    vision_frames = await async_vision(
                        dl_res["video_path"],
                        state.config["vision_interval"],
                        assets_dir,
                    )

                # 4. Generate Stream
                result_card.classes(remove="hidden")
                result_card.clear()
                with result_card:
                    ui.label(dl_res["title"]).classes("text-2xl font-bold mb-4")

                    # Reasoning Expander
                    reasoning_exp = ui.expansion(
                        "Thinking Process (AI Reasoning)", icon="psychology"
                    ).classes("w-full mb-4 bg-purple-50 rounded hidden")
                    with reasoning_exp:
                        reasoning_label = ui.markdown().classes(
                            "text-sm text-purple-800 p-2"
                        )

                    md_container = ui.markdown().classes(
                        "w-full prose prose-lg report-content"
                    )

                full_response = ""
                full_reasoning = ""
                processed_timestamps = set()

                # Helper for UI updates (NiceGUI usually handles this, but for safety in callbacks we define helper if needed)
                # But here we are in main loop so direct update is fine.

                async for chunk_type, chunk_text in generate_summary_stream_async(
                    dl_res["title"],
                    transcript_text,
                    segments,
                    vision_frames,
                    state.config,
                    custom_prompt,
                    complexity,
                ):
                    if chunk_type == "reasoning":
                        full_reasoning += chunk_text
                        reasoning_exp.classes(remove="hidden")
                        reasoning_label.set_content(full_reasoning)
                        # Auto-expand if first chunk? Maybe keep collapsed to not annoy. User preferred collapsed.
                        step_ai.props('caption="🤔 AI is thinking..."')

                    elif chunk_type == "content":
                        full_response += chunk_text
                        step_ai.props('caption="✍️ Writing report..."')

                        # Image Logic Logic (Updated for assets_dir)
                        display_text = full_response
                        timestamps = re.findall(r"\[(\d{1,2}:\d{2})\]", full_response)
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

                # Clear AI step spinner and label
                ai_spinner.delete()
                lbl_ai.delete()

                step_ai.props('caption="Completed"').props(add="done")

                ui.notify("Analysis Complete!", type="positive")

                # --- PHASE C: Atomic Finalization ---
                # Use final_display_text which contains the correct image paths
                final_content_for_save = (
                    final_display_text
                    if "final_display_text" in dir()
                    else full_response
                )
                final_task_dir, final_report = finalize_task(
                    task_id, dl_res["title"], final_content_for_save
                )

                # Update displayed content with corrected paths
                md_container.set_content(final_report)

                # Save to History (use final path)
                new_sess = create_session(
                    dl_res["title"],
                    url,
                    final_report,
                    transcript_text,
                    final_task_dir,
                    state.config,
                )
                add_session(new_sess)
                history_list.refresh()

            except Exception as e:
                ui.notify(f"Critical Error: {str(e)}", type="negative")
                import traceback

                print(traceback.format_exc())
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
