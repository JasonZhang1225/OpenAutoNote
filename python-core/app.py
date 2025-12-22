import streamlit as st
import os
import re
import json
import time
from pathlib import Path
from openai import OpenAI

# Core modules
from core.downloader import download_video
from core.transcriber import TranscriberFactory
from core.visual_processor import extract_frame, process_video_for_vision
from core.utils import build_multimodal_payload, seconds_to_hms, timestamp_str_to_seconds
from core.i18n import get_text
from core.storage import load_history, add_session, get_session, delete_session, clear_all_history, create_session

# --- Configuration Management ---
CONFIG_FILE = "user_config.json"

def load_config() -> dict:
    """Load user configuration from JSON file"""
    default_config = {
        "api_key": "",
        "base_url": "https://api.openai.com/v1",
        "model_name": "gpt-4o",
        "language": "Simplified Chinese (Default)",
        "hardware_mode": "mlx",
        "enable_vision": False,
        "vision_interval": 15,
        "vision_detail": "low",
        "detail_level": "Standard",
        "ui_language": "zh" # Default UI language
    }
    
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
                saved_config = json.load(f)
                default_config.update(saved_config)
        except Exception as e:
            print(f"[Config] Error loading config: {e}")
    
    return default_config

def save_config(config: dict):
    """Save user configuration to JSON file"""
    try:
        with open(CONFIG_FILE, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"[Config] Error saving config: {e}")

# --- Init & CSS ---
st.set_page_config(page_title="OpenAutoNote", page_icon="📝", layout="wide")

# Custom CSS to hide native elements and style sidebar
st.markdown("""
<style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Clean up sidebar */
    [data-testid="stSidebar"] {
        padding-top: 2rem;
    }
    
    /* Settings Expander Styling */
    .streamlit-expanderHeader {
        font-weight: bold;
        color: #555;
    }
</style>
""", unsafe_allow_html=True)

# --- Session State Management ---
if "config" not in st.session_state:
    st.session_state.config = load_config()

# UI Language shortcut
ui_lang = st.session_state.config.get("ui_language", "zh")

if "active_session_id" not in st.session_state:
    st.session_state.active_session_id = None # None means "New Note" view

# --- Helper Functions ---

def switch_session(session_id):
    st.session_state.active_session_id = session_id
    st.rerun()

def new_note():
    st.session_state.active_session_id = None
    st.rerun()

# --- STREAM GENERATOR Logic (Moved from old app.py) ---
def generate_summary_stream(title, full_text, segments, vision_frames, api_key, base_url, model_name, language, detail_level, enable_vision, vision_interval, vision_detail):
    if not api_key:
        yield "Error: Please enter an API Key."
        return

    client = OpenAI(api_key=api_key, base_url=base_url)
    
    # System Prompt (Updated with latest requirements)
    system_prompt = """
你是一个高级内容主编和视觉分析专家。
你将接收到视频的【字幕文本】和对应的【关键帧截图】。

任务核心目标：**生成一份结构清晰、重点突出的框架性总结，并精选极少数具有代表性的画面。**

关键要求：
1. **结构化输出**：
   - 严禁流水账。不要按时间顺序复述，而是要提取视频的核心框架（如：核心观点、主要模块、关键数据、结论）。
   - 使用多级标题、项目符号列表来组织内容。

2. **精选视觉证据 (Selective Visualization)**：
   - **不要**每句话都配图。
   - **仅在**画面出现关键图表、核心PPT页面、独特产品细节或具有高度概括性的视觉元素时，才插入时间戳 `[MM:SS]`。
   - 如果画面只是人物大头照或无关紧要的背景，**绝对不要**插入时间戳。
   - 目标是：整篇总结中只穿插 3-5 张最关键的“高光时刻”截图。

3. **语言与格式**：
   - 使用简体中文。
   - 每一部分先给出核心结论，再展开细节。
   - 格式示例：
     ## 1. [核心模块名称]
     - 核心观点...
     - 关键细节... (`[MM:SS]` <--- 仅在此处需要展示关键图表时插入)
"""
    
    user_content = []
    if enable_vision and vision_frames:
        user_content = build_multimodal_payload(title, full_text, segments, vision_frames, detail=vision_detail)
    else:
        user_content = f"视频标题: {title}\n字幕内容:\n{full_text}\n\n请总结上述内容。"

    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content}
            ],
            stream=True
        )
        yield from response
    except Exception as e:
        # Fallback logic
        error_str = str(e)
        if enable_vision and vision_frames and ("ChatCompletionRequestMultiContent" in error_str or "InvalidParameter" in error_str or "400" in error_str):
             print(f"Multimodal request failed ({error_str}), falling back to text-only...")
             user_content = f"视频标题: {title}\n字幕内容:\n{full_text}\n\n请总结上述内容。"
             response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content}
                ],
                stream=True
            )
             yield from response
        else:
            import traceback
            error_msg = f"LLM Error: {str(e)}\n\nDetails: {traceback.format_exc()}"
            print(error_msg)
        yield error_msg


# --- Settings Dialog ---
@st.dialog("⚙️ Settings")
def settings_dialog():
    # Tabs for categorized settings
    tab_api, tab_gen, tab_hw, tab_sys = st.tabs([
        get_text("tab_api", ui_lang), 
        get_text("tab_gen", ui_lang),
        get_text("tab_hardware", ui_lang),
        get_text("tab_system", ui_lang)
    ])
    
    with tab_api:
        new_api_key = st.text_input(get_text("lbl_api_key", ui_lang), value=st.session_state.config["api_key"], type="password")
        new_base_url = st.text_input(get_text("lbl_base_url", ui_lang), value=st.session_state.config["base_url"])
        new_model = st.text_input(get_text("lbl_model", ui_lang), value=st.session_state.config["model_name"])
        
        if new_api_key != st.session_state.config["api_key"]:
            st.session_state.config["api_key"] = new_api_key
            save_config(st.session_state.config)
        
        if new_base_url != st.session_state.config["base_url"]:
            st.session_state.config["base_url"] = new_base_url
            save_config(st.session_state.config)
            
        if new_model != st.session_state.config["model_name"]:
            st.session_state.config["model_name"] = new_model
            save_config(st.session_state.config)

    with tab_gen:
        allow_vision = st.toggle(get_text("lbl_enable_vision", ui_lang), value=st.session_state.config["enable_vision"])
        
        v_interval = st.slider(get_text("lbl_vision_interval", ui_lang), 5, 60, st.session_state.config["vision_interval"], disabled=not allow_vision)
        
        v_detail = st.selectbox(get_text("lbl_vision_detail", ui_lang), ["low", "high", "auto"], 
                               index=["low", "high", "auto"].index(st.session_state.config["vision_detail"]), 
                               disabled=not allow_vision)
                               
        out_lang = st.selectbox(get_text("lbl_output_lang", ui_lang), ["Simplified Chinese (Default)", "English"],
                               index=0 if "Chinese" in st.session_state.config["language"] else 1)
        
        changes = False
        if allow_vision != st.session_state.config["enable_vision"]:
            st.session_state.config["enable_vision"] = allow_vision
            changes = True
        if v_interval != st.session_state.config["vision_interval"]:
            st.session_state.config["vision_interval"] = v_interval
            changes = True
        if v_detail != st.session_state.config["vision_detail"]:
            st.session_state.config["vision_detail"] = v_detail
            changes = True
        if out_lang != st.session_state.config["language"]:
            st.session_state.config["language"] = out_lang
            changes = True
        
        if changes:
            save_config(st.session_state.config)

    with tab_hw:
        hw_options = ["mlx", "cpu", "cuda"]
        curr_hw = st.session_state.config["hardware_mode"]
        hw_idx = hw_options.index(curr_hw) if curr_hw in hw_options else 0
        
        new_hw = st.selectbox(get_text("lbl_hardware_mode", ui_lang), hw_options, index=hw_idx)
        
        if new_hw != curr_hw:
            st.session_state.config["hardware_mode"] = new_hw
            save_config(st.session_state.config)

    with tab_sys:
        lang_opts = {"zh": "🇨🇳 中文", "en": "🇺🇸 English"}
        curr_ui = st.session_state.config.get("ui_language", "zh")
        display_names = list(lang_opts.values())
        curr_disp = lang_opts.get(curr_ui, "🇨🇳 中文")
        
        selected_disp = st.selectbox(get_text("lbl_ui_lang", ui_lang), display_names, index=display_names.index(curr_disp))
        
        new_ui_lang = [k for k, v in lang_opts.items() if v == selected_disp][0]
        
        if new_ui_lang != curr_ui:
            st.session_state.config["ui_language"] = new_ui_lang
            save_config(st.session_state.config)
            st.rerun()
            
        if st.button(get_text("btn_clear_history", ui_lang)):
            clear_all_history()
            st.rerun()

# --- SIDEBAR: Navigation & Settings ---
with st.sidebar:
    st.title("OpenAutoNote")
    
    if st.button(get_text("nav_new_note", ui_lang), type="primary", use_container_width=True):
        new_note()

    st.divider()

    st.caption(get_text("nav_history_title", ui_lang))
    history_sessions = load_history()
    
    if not history_sessions:
        st.info(get_text("nav_no_history", ui_lang))
    else:
        for s in history_sessions:
            col1, col2 = st.columns([0.8, 0.2])
            with col1:
                display_title = (s['title'][:18] + '..') if len(s['title']) > 18 else s['title']
                if st.button(display_title, key=f"btn_{s['id']}", help=s['title']):
                    switch_session(s['id'])
            with col2:
                if st.button(get_text("nav_delete", ui_lang), key=f"del_{s['id']}"):
                    delete_session(s['id'])
                    st.rerun()

    st.divider()

    # Settings triggers Dialog
    if st.button(get_text("nav_settings_title", ui_lang), icon="⚙️", use_container_width=True):
        settings_dialog()

# --- MAIN AREA ---

# A. HISTORY VIEW
if st.session_state.active_session_id:
    session = get_session(st.session_state.active_session_id)
    if not session:
        st.error("Session not found.")
        time.sleep(1)
        new_note()
    else:
        st.header(session['title'])
        st.caption(f"Source: {session['video_url']} | Time: {session['timestamp']}")
        
        # Transcript Expander
        with st.expander(get_text("transcript_label", ui_lang)):
            st.text(session.get("transcript", "No transcript available."))
            
        # Summary content
        full_text = session.get("summary", "")
        project_dir = session.get("project_dir", "")
        
        # Render markdown with images
        # We need to parse images again to render them? 
        # Actually simplest way is just render text, and if text has image placeholders or if we can re-detect timestamps?
        # The stored summary likely contains the text. Creating logic to re-insert images is tricky if we don't store image positions.
        # BUT: The 'generate_summary_stream' was rendering images dynamically based on timestamps in text.
        # So we can reuse that rendering logic!
        
        lines = full_text.split('\n')
        for line in lines:
            st.markdown(line)
            # Detect Timestamp [MM:SS]
            match = re.search(r'\[(\d{1,2}:\d{2}(?::\d{2})?)\]', line)
            if match:
                ts_str = match.group(1)
                seconds = timestamp_str_to_seconds(ts_str)
                # Try to find image in project dir
                # Look for project_dir in session or reconstruct it
                target_img_name = f"frame_render_{seconds}.jpg"
                
                # Check temp or session project dir
                img_path = None
                
                # If we recorded project_dir
                if project_dir and os.path.exists(project_dir):
                     # Frames were saved in 'frames' subdir usually? Or temp.
                     # In app.py before: process_video_for_vision -> project_dir/frames
                     # But extract_frame -> temp/frame_render_X.jpg
                     
                     # Check classic temp location first (fragile but persistent for this session?)
                     # For history, ideally we saved images to project_dir. 
                     # Let's assume for now we just try project_dir/frames/frame_X.jpg or similar.
                     # Fix: In generation, we used `extract_frame` to `temp/`.
                     # Ideally we should have moved them to `project_dir`.
                     pass
                
                # Fallback: check if existing temp file exists (only works if not cleaned)
                temp_path = f"temp/{target_img_name}"
                if os.path.exists(temp_path):
                    st.image(temp_path, caption=f"Time: {ts_str}", width=600)
                # Else: we can't show image unless we re-extract or saved it properly.
                # Future Task: Save renders to project folder.

# B. GENERATION VIEW (New Note)
else:
    st.title("OpenAutoNote")
    
    url_input = st.text_input("URL", placeholder=get_text("input_placeholder", ui_lang), label_visibility="collapsed")
    
    if st.button(get_text("btn_start", ui_lang), type="primary"):
        # Validate Config
        cfg = st.session_state.config
        if not url_input or not cfg["api_key"]:
            st.warning("Please check URL and API Key in Settings.")
            st.stop()
            
        # 1. Download
        with st.status(get_text("status_downloading", ui_lang), expanded=True) as status:
            dl_res = download_video(url_input)
            
            if not dl_res.get("success", False):
                status.update(label=get_text("status_download_failed", ui_lang), state="error")
                st.error(dl_res.get('error'))
                st.stop()
                
            video_path = dl_res.get("video_path")
            video_title = dl_res.get("title", "Unknown")
            project_dir = dl_res.get("project_dir")
            status.update(label=f"Downloaded: {video_title}", state="complete")
            
        # 2. Transcribe
        with st.status(get_text("status_transcribing", ui_lang), expanded=True) as status:
            try:
                transcriber = TranscriberFactory.get_transcriber(cfg["hardware_mode"])
                segments = transcriber.transcribe(video_path)
                transcript_text = " ".join([seg['text'] for seg in segments])
                status.update(label=f"Transcribed {len(segments)} segments", state="complete")
            except Exception as e:
                st.error(str(e))
                st.stop()

        # 3. Vision
        vision_frames = []
        if cfg["enable_vision"]:
            with st.status(get_text("status_vision_analyzing", ui_lang), expanded=True) as status:
                vision_frames = process_video_for_vision(video_path, cfg["vision_interval"], output_dir=project_dir)
                status.update(label=f"Extracted {len(vision_frames)} visual samples", state="complete")
        
        # 4. Generate
        st.markdown(f"### {get_text('header_summary', ui_lang)}")
        
        thinking_expander = st.expander(get_text("expander_thinking", ui_lang), expanded=False)
        thinking_placeholder = thinking_expander.empty()
        summary_placeholder = st.empty()
        
        full_text = ""
        reasoning_text = ""
        
        stream = generate_summary_stream(
            video_title, transcript_text, segments, vision_frames, 
            cfg["api_key"], cfg["base_url"], cfg["model_name"], cfg["language"], 
            cfg["detail_level"], cfg["enable_vision"], cfg["vision_interval"], cfg["vision_detail"]
        )
        
        if stream:
            for chunk in stream:
                if isinstance(chunk, str):
                    st.error(chunk)
                    break
                try:
                    delta = chunk.choices[0].delta
                    r_content = getattr(delta, 'reasoning_content', None)
                    content = getattr(delta, 'content', None)
                    
                    if r_content:
                        reasoning_text += r_content
                        thinking_placeholder.markdown(reasoning_text)
                    
                    if content:
                        full_text += content
                        with summary_placeholder.container():
                            lines = full_text.split('\n')
                            for line in lines:
                                st.markdown(line)
                                match = re.search(r'\[(\d{1,2}:\d{2}(?::\d{2})?)\]', line)
                                if match:
                                    ts_str = match.group(1)
                                    seconds = timestamp_str_to_seconds(ts_str)
                                    # Use temp folder for dynamic render
                                    img_path = f"temp/frame_render_{seconds}.jpg"
                                    if not os.path.exists(img_path):
                                        extract_frame(video_path, seconds, img_path)
                                    if os.path.exists(img_path):
                                        st.image(img_path, caption=f"Time: {ts_str}", width=600)
                except Exception as e:
                    pass
            
            # Save Session to History
            new_sess = create_session(video_title, url_input, full_text, transcript_text, project_dir, cfg)
            add_session(new_sess)
            st.success("Saved to History!")
            st.session_state.active_session_id = new_sess["id"]
            time.sleep(1)
            st.rerun() # Rerun to switch to view mode
