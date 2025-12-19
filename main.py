import os
import asyncio
import json
import re
from pathlib import Path
from nicegui import ui, run, app
import sys

# Capture original streams for redirection
original_stdout = sys.stdout
original_stderr = sys.stderr
from openai import AsyncOpenAI

from core.downloader import download_video
from core.transcriber import TranscriberFactory
from core.visual_processor import process_video_for_vision, extract_frame
from core.utils import build_multimodal_payload, timestamp_str_to_seconds
from core.i18n import get_text
from core.storage import load_history, add_session, get_session, delete_session, create_session, clear_all_history

# --- Configuration & State ---
CONFIG_FILE = "user_config.json"
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
    "theme_mode": "light"
}

if not os.path.exists('generate'):
    os.makedirs('generate')
app.add_static_files('/generate', 'generate') 

def load_config():
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
                return {**DEFAULT_CONFIG, **json.load(f)}
        except:
            return DEFAULT_CONFIG.copy()
    return DEFAULT_CONFIG.copy()

def save_config(cfg):
    with open(CONFIG_FILE, 'w', encoding='utf-8') as f:
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
        if message.strip() and '\r' not in message and '[download]' not in message:
            try:
                # Push to UI
                self.log_element.push(message.strip())
            except:
                pass # Avoid errors if UI is disconnected

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
    old_folder = f"generate/{task_id}"
    new_folder = f"generate/{clean_title}"
    
    # Avoid overwriting existing folders
    if os.path.exists(new_folder):
        import uuid
        new_folder = f"{new_folder}_{str(uuid.uuid4())[:8]}"
    
    report_path = os.path.join(old_folder, "report.md")
    
    if not os.path.exists(old_folder):
        print(f"[Finalize] Error: Folder {old_folder} not found.")
        return old_folder, report_content
    
    # 1. Update paths in report content BEFORE renaming
    updated_content = report_content.replace(f"/generate/{task_id}", f"/generate/{clean_title}")
    
    # 2. Save report.md to disk
    try:
        with open(report_path, 'w', encoding='utf-8') as f:
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
        print(f"Transcribing with {hardware_mode}...")
        transcriber = TranscriberFactory.get_transcriber(hardware_mode)
        return transcriber.transcribe(video_path)
    return await run.io_bound(_t)

async def async_vision(video_path, interval, output_dir):
    return await run.io_bound(process_video_for_vision, video_path, interval, output_dir)

async def generate_summary_stream_async(title, full_text, segments, vision_frames, config):
    if not config["api_key"]:
        yield "Error: API Key missing."
        return

    client = AsyncOpenAI(api_key=config["api_key"], base_url=config["base_url"])
    
    system_prompt = """
你是一名**资深科技主编**和**视觉叙事专家**。
你的风格是：**专业、犀利、结构化**，类似于 "The Verge", "Notion Blog" 或 "少数派" 的深度文章。

任务目标：将视频内容转化为一份**视觉化、杂志级的深度报告**。

### 核心布局要求 (Strict Layout)
1.  **一句话金句 (The One-Liner)**
    -   在开头必须使用 Blockquote (`>`) 提炼出视频最核心的价值或结论。
    -   例如：`> 💡 **核心洞察**：Firefox 的衰落并非技术落后，而是移动互联网时代商业模式的必然溃败。`

2.  **结构化章节**
    -   使用 H2 (`##`) 划分主要模块。
    -   **Emoji 列表**：禁止使用普通的黑点 bullet。必须根据上下文使用 Emoji：
        -   🎯 核心观点 / 目标
        -   ⚡ 技术亮点 / 痛点
        -   💰 商业 / 成本
        -   ⚠️ 风险 / 警告
        -   🛠️ 解决方案 / 步骤

3.  **数据对比 (必须使用表格)**
    -   如果视频中出现对比（如 A vs B，今年 vs 去年），**必须**输出 Markdown Table。
    -   例如：
        | 特性 | Firefox | Chrome |
        | :--- | :--- | :--- |
        | 内核 | Gecko | Blink |

4.  **视觉证据 (Selective Images)**
    -   **原则**：宁缺毋滥。仅在关键时刻（PPT图表、独特产品细节）插入截图。
    -   **位置**：将截图时间戳 `[MM:SS]` 直接插入在最相关的段落之后，不要堆砌在最后。

### 语言风格
-   **中文**：使用流畅、专业的简体中文。
-   **拒绝流水账**：不要说“视频首先讲了...然后讲了...”，直接陈述事实和观点。
"""
    
    user_content = []
    if config["enable_vision"] and vision_frames:
        user_content = build_multimodal_payload(title, full_text, segments, vision_frames, detail=config["vision_detail"])
    else:
        user_content = f"视频标题: {title}\n字幕内容:\n{full_text}\n\n请总结上述内容。"

    try:
        response = await client.chat.completions.create(
            model=config["model_name"],
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content}
            ],
            stream=True
        )
        
        async for chunk in response:
            delta = chunk.choices[0].delta
            content = getattr(delta, 'content', None)
            reasoning = getattr(delta, 'reasoning_content', None)
            
            if reasoning:
                yield ('reasoning', reasoning)
            if content:
                yield ('content', content)

    except Exception as e:
        yield ('error', f"\n\n**Error:** {str(e)}")


# --- UI Construction ---

@ui.page('/')
def index():
    # Apply some global styles via query
    ui.colors(primary='#5898d4', secondary='#26a69a', accent='#9c27b0')
    ui.query('body').classes('bg-grey-1 text-slate-900 font-sans')
    
    # --- Top Bar ---
    with ui.header().classes('bg-white text-slate-800 shadow-none border-b border-gray-200 h-16 items-center px-4'):
        # Toggle Drawer
        ui.button(on_click=lambda: left_drawer.toggle(), icon='menu').props('flat round color=grey-8')
        ui.label('OpenAutoNote').classes('text-xl font-bold ml-2 text-primary')
        ui.space()

    # --- Custom CSS for Report ---
    ui.add_head_html('''
    <style>
        .report-content { font-size: 1.15rem; }
        .report-content h1 { font-size: 2.2em; color: #2c3e50; font-weight: 700; margin-bottom: 0.8em; margin-top: 0; }
        .report-content h2 { font-size: 1.6em; color: #34495e; font-weight: 600; margin-top: 1.8em; margin-bottom: 0.8em; border-bottom: 2px solid #f1f5f9; padding-bottom: 10px; }
        .report-content h3 { font-size: 1.3em; color: #475569; font-weight: 600; margin-top: 1.4em; }
        .report-content p { font-size: 1em; line-height: 1.8; color: #334155; margin-bottom: 1.25em; }
        .report-content blockquote { border-left: 5px solid #6366f1; background: #f8fafc; padding: 16px 20px; margin: 24px 0; border-radius: 0 8px 8px 0; color: #475569; font-style: italic; }
        .report-content ul { list-style-type: none; padding-left: 0; margin-bottom: 1.5em; }
        .report-content li { margin-bottom: 0.6em; line-height: 1.8; padding-left: 0.5em; }
        /* Image Constraint Fix */
        .report-content img { 
            max-width: 80%; 
            max-height: 600px;
            display: block; 
            margin: 30px auto; 
            border-radius: 12px; 
            box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1); 
            border: 1px solid #e2e8f0; 
        }
        .report-content table { width: 100%; border-collapse: separate; border-spacing: 0; margin: 24px 0; border: 1px solid #e2e8f0; border-radius: 8px; overflow: hidden; }
        .report-content th { background-color: #f8fafc; padding: 12px 16px; border-bottom: 1px solid #e2e8f0; font-weight: 600; text-align: left; color: #475569; }
        .report-content td { padding: 12px 16px; border-bottom: 1px solid #e2e8f0; color: #334155; }
        .report-content tr:last-child td { border-bottom: none; }
        .report-content tr:nth-child(even) { background-color: #fcfcfc; }
    </style>
    ''')

    # --- Settings Dialog (Function) ---
    def open_settings():
        with ui.dialog() as dialog, ui.card().classes('w-[500px] p-6'):
            ui.label(get_text("nav_settings_title", state.config["ui_language"])).classes('text-xl font-bold mb-4')

            with ui.tabs().classes('w-full') as tabs:
                tab_api = ui.tab(get_text("tab_api", state.config["ui_language"]))
                tab_gen = ui.tab(get_text("tab_gen", state.config["ui_language"]))
                tab_hw = ui.tab(get_text("tab_hardware", state.config["ui_language"]))
                tab_sys = ui.tab(get_text("tab_system", state.config["ui_language"]))

            with ui.tab_panels(tabs, value=tab_gen).classes('w-full mt-2'):
                
                # --- API ---
                with ui.tab_panel(tab_api):
                    ui.input(get_text("lbl_api_key", state.config["ui_language"]), password=True).bind_value(state.config, 'api_key').classes('w-full').props('outlined dense')
                    ui.input(get_text("lbl_base_url", state.config["ui_language"])).bind_value(state.config, 'base_url').classes('w-full').props('outlined dense')
                    ui.input(get_text("lbl_model", state.config["ui_language"])).bind_value(state.config, 'model_name').classes('w-full').props('outlined dense')

                # --- Generation ---
                with ui.tab_panel(tab_gen):
                    with ui.row().classes('w-full justify-between items-center mb-4'):
                        ui.label(get_text("lbl_enable_vision", state.config["ui_language"])).classes('text-gray-700')
                        ui.switch().bind_value(state.config, 'enable_vision')

                    ui.separator().classes('my-4')
                    
                    ui.label(get_text("lbl_vision_interval", state.config["ui_language"])).classes('text-sm text-gray-500 mt-2')
                    with ui.row().classes('w-full items-center gap-2'):
                        slider = ui.slider(min=5, max=60, step=5).bind_value(state.config, 'vision_interval').classes('flex-grow')
                        ui.label().bind_text_from(slider, 'value', lambda v: f"{v}s").classes('w-10 text-right font-mono')
                    
                    ui.label(get_text("lbl_vision_detail", state.config["ui_language"])).classes('text-sm text-gray-500 mt-4')
                    ui.select(
                        options={'low': 'Low (Fast, 720p)', 'high': 'High (Detail, 1080p)', 'auto': 'Auto'},
                        value=state.config.get('vision_detail', 'low')
                    ).bind_value(state.config, 'vision_detail').classes('w-full')
                    
                    ui.separator().classes('my-4')
                    
                    # Cookie Settings
                    ui.label('🍪 Cookies (Optional)').classes('text-sm text-gray-700 font-medium')
                    ui.input('YouTube Cookies Path', placeholder='cookies_youtube.txt').bind_value(state.config, 'cookies_yt').classes('w-full').props('outlined dense')
                    ui.input('Bilibili Cookies Path', placeholder='cookies_bilibili.txt').bind_value(state.config, 'cookies_bili').classes('w-full').props('outlined dense')
                    ui.label('Tip: Export cookies using browser extensions like "Get cookies.txt LOCALLY"').classes('text-xs text-gray-400')

                # --- Hardware ---
                with ui.tab_panel(tab_hw):
                    ui.label(get_text("lbl_hardware_mode", state.config["ui_language"])).classes('text-sm text-gray-500')
                    
                    device_options = {
                        'cpu': 'Standard CPU (Slow)',
                        'cuda': 'NVIDIA GPU (CUDA)',
                        'mlx': 'Apple Silicon (MLX) ⚡️'
                    }
                    
                    ui.select(
                        options=device_options,
                        value=state.config.get('hardware_mode', 'mlx')
                    ).bind_value(state.config, 'hardware_mode').classes('w-full')
                    
                    ui.label("Recommended: MLX for Mac, CUDA for NVIDIA Windows.").classes('text-xs text-gray-400 mt-1')

                # --- System ---
                with ui.tab_panel(tab_sys):
                     ui.select({'zh': '中文', 'en': 'English'}, label=get_text("lbl_ui_lang", state.config["ui_language"])).bind_value(state.config, 'ui_language').on_value_change(lambda: ui.open('/')).classes('w-full')

            with ui.row().classes('w-full justify-end mt-6'):
                # Note: get_text for save/close missing, using hardcoded
                ui.button('Close', on_click=dialog.close).props('flat')
                ui.button('Save', on_click=lambda: (save_config(state.config), dialog.close())).props('unelevated color=primary')
        
        dialog.open()

    # --- Left Sidebar ---
    with ui.left_drawer(value=True).classes('bg-white border-r border-gray-200 column no-wrap') as left_drawer:
        # History List
        with ui.scroll_area().classes('col flex-grow q-pa-sm'):
            ui.label(get_text("nav_history_title", state.config["ui_language"])).classes('text-xs font-bold text-grey-6 mb-2 mt-2 uppercase tracking-wide')
            
            @ui.refreshable
            def history_list():
                sessions = load_history()
                if not sessions:
                    ui.label(get_text("nav_no_history", state.config["ui_language"])).classes('text-grey-5 text-sm italic q-ml-sm')
                    return
                
                # Quasar List
                with ui.list().props('dense separator=false'):
                    for s in sessions:
                        # Item Container
                        with ui.item(on_click=lambda id=s['id']: load_session(id)).props('clickable v-ripple').classes('rounded-borders hover:bg-blue-50 group mb-1'):
                            # Icon
                            with ui.item_section().props('avatar min-width=0'):
                                ui.icon('article', color='grey-5').props('size=xs')
                            
                            # Label with truncation
                            with ui.item_section():
                                ui.label(s['title']).classes('text-body2 text-grey-9 truncate w-40')
                            
                            # Hover Action (Delete)
                            with ui.item_section().props('side'):
                                ui.button(icon='delete', on_click=lambda e, id=s['id']: delete_sess_handler(id)) \
                                    .props('flat round dense size=sm color=grey-4').classes('opacity-0 group-hover:opacity-100 transition-opacity')

            history_list()

        # Bottom Actions
        with ui.column().classes('col-auto q-pa-sm w-full border-t border-gray-200'):
            ui.button('Settings', icon='settings', on_click=open_settings).props('flat block align=left text-color=grey-8 w-full')

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
        pass # We'll insert it via layout above history list if we want, but FAB implies floating.
        # Let's do a proper FAB in the Bottom Right of the Drawer?
        # ui.button(icon='add').props('fab color=primary').classes('absolute-bottom-right q-ma-md') 
        # But this might overlap settings. 
        # Let's put it top-right of sidebar.

    # FAB is better placed in the Drawer stack
    with left_drawer:
        ui.button(icon='add', on_click=lambda: new_note_handler()) \
            .props('fab color=primary') \
            .classes('absolute-bottom-right q-ma-md shadow-lg z-10') \
            .tooltip(get_text("nav_new_note", state.config["ui_language"]))

    # --- Main Content Area ---
    main_container = ui.column().classes('w-full max-w-5xl mx-auto q-pa-md items-stretch')
    
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
        if state.current_session and state.current_session['id'] == sess_id:
            new_note_handler()

    def load_session(sess_id):
        state.current_session = get_session(sess_id)
        main_content.refresh()

    def new_note_handler():
        state.current_session = None
        main_content.refresh()

    # --- Logic: View Renderers ---
    
    def render_history_view(session):
        with ui.card().classes('w-full q-pa-lg no-shadow border border-gray-200 rounded-xl'):
            # Header
            ui.label(session['title']).classes('text-3xl font-bold tracking-tight text-slate-900 leading-tight mb-2')
            with ui.row().classes('items-center gap-4 text-grey-6 text-sm mb-6'):
                ui.link(session['video_url'], session['video_url']).classes('hover:text-primary')
                ui.label('•')
                ui.label(session.get('timestamp', '')[:10])

            # Transcript Expander
            with ui.expansion('查看原文 (Transcript)', icon='description').classes('w-full mb-6 bg-grey-1 rounded-lg border border-gray-200'):
                ui.label(session.get('transcript', 'No transcript')).classes('text-grey-8 font-mono text-sm leading-relaxed p-4')

            ui.separator().classes('mb-6')

            # Markdown Content
            ui.markdown(session.get('summary', '')).classes('w-full prose prose-lg prose-slate report-content')

    def render_input_view():
        # Title Header
        # ui.label('OpenAutoNote').classes('text-4xl font-extrabold text-slate-800 mb-8 self-center')
        
        # Input Card
        with ui.card().classes('w-full max-w-3xl self-center q-pa-lg shadow-sm border border-gray-200 rounded-xl bg-white'):
            ui.label('Create New Summary').classes('text-xl font-bold mb-4 text-slate-700')
            
            # Capsule-Style Input with Button inside
            with ui.input(placeholder="Paste video URL here (Bilibili/YouTube)...").props('rounded outlined dense').classes('w-full text-lg') as url_input:
                with url_input.add_slot('append'):
                    btn_start = ui.button(icon='arrow_forward', on_click=lambda: run_analysis(url_input.value)).props('round flat color=primary dense').tooltip('Start Analysis')

        # Stepper Container (Hidden initially)
        stepper_container = ui.column().classes('w-full max-w-3xl self-center mt-8 transition-all')
        
        # Result Card (Hidden initially)
        result_card = ui.card().classes('w-full max-w-4xl self-center mt-8 q-pa-xl shadow-sm border border-gray-200 rounded-xl hidden')
        
        # Thread-safe UI updater helper
        def queue_ui_update(func):
             func()
        
        # ANSI escape code cleaner
        import re as regex_module
        ansi_escape_pattern = regex_module.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
        def clean_ansi(text):
            return ansi_escape_pattern.sub('', str(text) if text else '')
        
        async def run_analysis(url):
            if not url:
                ui.notify('Please enter a URL', type='warning')
                return
            
            # Disable input
            btn_start.disable()
            
            # --- 0. Prepare Directory Structure ---
            import uuid
            from datetime import datetime
            
            task_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            task_uuid = str(uuid.uuid4())[:8]
            task_id = f"{task_timestamp}_{task_uuid}"
            
            base_temp = "generate"
            task_dir = os.path.join(base_temp, task_id)
            raw_dir = os.path.join(task_dir, "raw")
            assets_dir = os.path.join(task_dir, "assets")
            
            # Create Strict Hierarchy
            os.makedirs(raw_dir, exist_ok=True)
            os.makedirs(assets_dir, exist_ok=True)
            
            # Setup Stepper
            stepper_container.clear()
            with stepper_container:
                # Stepper (No header-nav to prevent user skipping steps)
                with ui.stepper().props('vertical animated').classes('w-full shadow-sm bg-white rounded-xl border border-gray-200') as stepper:
                    step_dl = ui.step('Download Video').props('icon=cloud_download')
                    step_ts = ui.step('Transcribe Audio').props('icon=graphic_eq')
                    step_ai = ui.step('AI Analysis').props('icon=psychology')
                
                # Initialize
                stepper.value = step_dl
            
            try:
                # 1. Download
                with step_dl:
                    # Progress Bar UI (No overlay, just bar + status below)
                    dl_progress = ui.linear_progress(value=0).props('rounded color=primary size=12px')
                    dl_status = ui.label('Initializing...').classes('text-sm text-grey-7 mt-2')

                # Callback for yt-dlp
                def dl_hook(d):
                    if d['status'] == 'downloading':
                        try:
                            total = d.get('total_bytes') or d.get('total_bytes_estimate') or 1
                            downloaded = d.get('downloaded_bytes', 0)
                            percent = downloaded / total
                            speed = clean_ansi(d.get('_speed_str', 'N/A'))
                            eta = clean_ansi(d.get('_eta_str', 'N/A'))
                            
                            # Update UI
                            dl_progress.value = percent
                            dl_status.text = f"{percent:.1%} | Speed: {speed} | ETA: {eta}"
                        except:
                            pass
                    elif d['status'] == 'finished':
                        queue_ui_update(lambda: setattr(dl_status, 'text', 'Processing...'))

                # Run download with domain-based cookie selection
                cookies_yt = state.config.get('cookies_yt', '')
                cookies_bili = state.config.get('cookies_bili', '')
                dl_res = await run.io_bound(download_video, url, raw_dir, cookies_yt, cookies_bili, True, dl_hook)
                
                if not dl_res['success']:
                    # Visual Error Feedback
                    dl_progress.props('color=negative')  # Turn progress bar red
                    dl_progress.value = 1.0  # Fill it to show "failed"
                    dl_status.text = f"❌ Error: {dl_res.get('error', 'Unknown error')}"
                    dl_status.classes(add='text-red-600')
                    
                    # Update step icon to error
                    step_dl.props('icon=error color=negative')
                    
                    ui.notify(f"Download Failed: {dl_res.get('error')}", type='negative', position='top', close_button=True, timeout=0)
                    btn_start.enable()
                    return
                
                # Mark done - success state
                dl_progress.props('color=positive')
                dl_status.text = f"✅ Downloaded: {dl_res.get('title', 'Video')}"
                dl_status.classes(add='text-green-700')
                step_dl.props('icon=check color=positive')
                stepper.next() # Move to TS
                
                # 2. Transcribe
                with step_ts:
                    ui.spinner().classes('q-ma-md')
                    lbl_ts = ui.label('Processing Audio...').classes('text-grey-6 italic')
                
                segments = await async_transcribe(dl_res['video_path'], state.config['hardware_mode'])
                transcript_text = " ".join([s['text'] for s in segments])
                lbl_ts.text = f"Transcribed {len(segments)} segments."
                
                step_ts.props(add='done')
                stepper.next() # Move to AI
                
                # 3. Vision Extraction
                vision_frames = []
                if state.config['enable_vision']:
                     # Optional: Show sub-progress
                     # Pass assets_dir directly. visual_processor was updated to write there.
                     vision_frames = await async_vision(dl_res['video_path'], state.config['vision_interval'], assets_dir)
                
                # 4. Generate Stream
                result_card.classes(remove='hidden')
                result_card.clear()
                with result_card:
                    ui.label(dl_res['title']).classes('text-2xl font-bold mb-4')
                    
                    # Reasoning Expander
                    reasoning_exp = ui.expansion('Thinking Process (AI Reasoning)', icon='psychology').classes('w-full mb-4 bg-purple-50 rounded hidden')
                    with reasoning_exp:
                        reasoning_label = ui.markdown().classes('text-sm text-purple-800 p-2')
                    
                    md_container = ui.markdown().classes('w-full prose prose-lg report-content')
                
                full_response = ""
                full_reasoning = ""
                processed_timestamps = set()
                
                # Helper for UI updates (NiceGUI usually handles this, but for safety in callbacks we define helper if needed)
                # But here we are in main loop so direct update is fine.

                async for chunk_type, chunk_text in generate_summary_stream_async(
                    dl_res['title'], transcript_text, segments, vision_frames, state.config
                ):
                    if chunk_type == 'reasoning':
                        full_reasoning += chunk_text
                        reasoning_exp.classes(remove='hidden')
                        reasoning_label.set_content(full_reasoning)
                        # Auto-expand if first chunk? Maybe keep collapsed to not annoy. User preferred collapsed.
                        step_ai.props('caption="🤔 AI is thinking..."')

                    elif chunk_type == 'content':
                        full_response += chunk_text
                        step_ai.props('caption="✍️ Writing report..."')
                        
                        # Image Logic Logic (Updated for assets_dir)
                        display_text = full_response
                        timestamps = re.findall(r'\[(\d{1,2}:\d{2})\]', full_response)
                        for ts in timestamps:
                            seconds = timestamp_str_to_seconds(ts)
                            img_filename = f"frame_{seconds}.jpg"
                            img_fs_path = os.path.join(assets_dir, img_filename)
                            img_web_path = f"/generate/{task_id}/assets/{img_filename}"
                            
                            if ts not in processed_timestamps:
                                 if not os.path.exists(img_fs_path):
                                     await run.io_bound(extract_frame, dl_res['video_path'], seconds, img_fs_path)
                                 processed_timestamps.add(ts)
                            
                            if os.path.exists(img_fs_path):
                                if f"![{ts}]" not in display_text:
                                    display_text = display_text.replace(f"[{ts}]", f"[{ts}]\n\n![{ts}]({img_web_path})")

                        md_container.set_content(display_text)
                        # Store the final display_text for finalization
                        final_display_text = display_text
                    
                step_ai.props('caption="Completed"').props(add='done')

                ui.notify('Analysis Complete!', type='positive')
                
                # --- PHASE C: Atomic Finalization ---
                # Use final_display_text which contains the correct image paths
                final_content_for_save = final_display_text if 'final_display_text' in dir() else full_response
                final_task_dir, final_report = finalize_task(task_id, dl_res['title'], final_content_for_save)
                
                # Update displayed content with corrected paths
                md_container.set_content(final_report)
                
                # Save to History (use final path)
                new_sess = create_session(dl_res['title'], url, final_report, transcript_text, final_task_dir, state.config)
                add_session(new_sess)
                history_list.refresh()

            except Exception as e:
                ui.notify(f"Critical Error: {str(e)}", type='negative')
                import traceback
                print(traceback.format_exc())
            finally:
                btn_start.enable()

    # --- System Log View ---
    with ui.expansion('📟 系统终端 / System Terminal', icon='terminal').classes('w-full mt-4 bg-black text-green-400 rounded-lg'):
        log_view = ui.log(max_lines=1000).classes('w-full h-64 bg-black text-green-400 font-mono text-xs p-2 rounded')
        
    # Redirect stdout/stderr to this user's log view
    # Note: In a multi-user app, this would redirect output to the LAST connected user.
    # For a local single-user tool, this is acceptable.
    sys.stdout = WebLogger(original_stdout, log_view)
    sys.stderr = WebLogger(original_stderr, log_view)

    # Initial Render
    main_content()

ui.run(title='OpenAutoNote', port=8964, reload=True, storage_secret='gemini_secret')
