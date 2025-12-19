import streamlit as st
import os
import time
import re
from pathlib import Path
from openai import OpenAI
from core.downloader import download_video
from core.transcriber import TranscriberFactory
from core.visual_processor import extract_frame, process_video_for_vision
from core.utils import build_multimodal_payload, seconds_to_hms, timestamp_str_to_seconds
import json

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
        "detail_level": "Standard"
    }
    
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
                saved_config = json.load(f)
                # Merge with defaults to handle missing keys
                default_config.update(saved_config)
        except Exception as e:
            print(f"[Config] Error loading config: {e}")
    
    return default_config

def save_config(config: dict):
    """Save user configuration to JSON file"""
    try:
        with open(CONFIG_FILE, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        print(f"[Config] Configuration saved to {CONFIG_FILE}")
    except Exception as e:
        print(f"[Config] Error saving config: {e}")

# --- Configuration & UI Setup ---
st.set_page_config(page_title="OpenAutoNote", page_icon="📝", layout="wide")

st.title("📝 OpenAutoNote: Multimodal Video Summary")

# Load saved configuration
config = load_config()

# --- Sidebar ---
with st.sidebar:
    st.header("⚙️ Configuration")
    
    st.subheader("LLM Settings")
    api_key = st.text_input("API Key", value=config["api_key"], type="password", help="OpenAI or DeepSeek API Key")
    base_url = st.text_input("Base URL", value=config["base_url"], help="API Base URL")
    model_name = st.text_input("Model Name", value=config["model_name"], help="Vision-capable model required for multimodal analysis")
    
    st.subheader("Language & Preferences")
    language_options = ["Simplified Chinese (Default)", "English"]
    language_index = language_options.index(config["language"]) if config["language"] in language_options else 0
    language = st.selectbox("Output Language", options=language_options, index=language_index)
    
    st.subheader("👁️ Vision Analysis Settings")
    enable_vision = st.toggle("Enable Multimodal Analysis", value=config["enable_vision"])
    vision_interval = st.slider("Vision Interval (sec)", 5, 60, config["vision_interval"], disabled=not enable_vision)
    vision_detail = st.selectbox("Image Detail", ["low", "high", "auto"], index=["low", "high", "auto"].index(config["vision_detail"]), disabled=not enable_vision)
    
    if enable_vision and vision_interval < 10:
        st.warning("⚠️ High frequency sampling will consume significant tokens.")

    st.subheader("Hardware")
    hardware_options = ["mlx", "cpu", "cuda"]
    hardware_index = hardware_options.index(config["hardware_mode"]) if config["hardware_mode"] in hardware_options else 0
    hardware_mode = st.selectbox(
        "Acceleration Mode",
        options=hardware_options,
        format_func=lambda x: {
            "mlx": "Apple Neural Engine (MLX)",
            "cpu": "CPU (Universal)",
            "cuda": "NVIDIA CUDA"
        }[x],
        index=hardware_index
    )
    
    st.subheader("Summary Options")
    detail_options = ["Brief", "Standard", "Detailed"]
    detail_index = detail_options.index(config["detail_level"]) if config["detail_level"] in detail_options else 1
    detail_level = st.select_slider("Detail Level", options=detail_options, value=detail_options[detail_index])
    
    st.markdown("---")
    st.caption(f"System: {os.uname().machine}")

# --- Helper Functions imported from core.utils ---

def generate_summary_stream(title, full_text, segments, vision_frames, api_key, base_url, model_name, language, detail_level, enable_vision, vision_interval, vision_detail):
    if not api_key:
        yield "Error: Please enter an API Key."
        return

    client = OpenAI(api_key=api_key, base_url=base_url)
    
    lang_instruction = "Simplified Chinese" if "Simplified Chinese" in language else "English"
    
    system_prompt = """
你是一个专业的视频内容分析专家。
你将接收到视频的【字幕文本】和对应的【关键帧截图】。

任务要求：
1. **多模态分析**：必须结合画面内容（如PPT文字、产品细节）和语音内容。如果画面补充了语音未提及的信息，请重点指出。
2. **语言控制**：
   - 无论输入是什么语言，**必须使用简体中文 (Simplified Chinese)** 输出总结。
   - 如果用户指定了其他语言，请忽略此条。
3. **格式规范**：
   - 使用 Markdown 格式。
   - 在关键结论后引用时间戳，格式为 `[MM:SS]`。
   - 遇到繁体中文输入，请自动转换为简体中文。
"""
    
    user_content = []
    
    if enable_vision and vision_frames:
        # Patch B: Use the new build_multimodal_payload
        user_content = build_multimodal_payload(title, full_text, segments, vision_frames, detail=vision_detail)
    else:
        # Text only mode
        user_content = f"视频标题: {title}\n字幕内容:\n{full_text}\n\n请总结上述内容。"


    # Debug: Log payload structure
    if enable_vision and vision_frames:
        print(f"[DEBUG] Sending multimodal payload with {len(user_content)} content blocks")
        print(f"[DEBUG] Vision frames: {len(vision_frames)}, Segments: {len(segments)}")
    else:
        print(f"[DEBUG] Sending text-only payload, length: {len(full_text)} chars")

    try:
        print(f"[DEBUG] About to call API: {base_url}")
        print(f"[DEBUG] Model: {model_name}")
        print(f"[DEBUG] API Key (first 10): {api_key[:10]}...")
        
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content}
            ],
            stream=True
        )
        
        print(f"[DEBUG] API response received, type: {type(response)}")
        return response
    except Exception as e:
        import traceback
        error_msg = f"LLM Error: {str(e)}\n\nDetails: {traceback.format_exc()}"
        st.error(error_msg)
        print(error_msg)  # Also print to console for debugging
        return None

# --- Main Logic ---

url_input = st.text_input("Enter Video URL", placeholder="https://www.bilibili.com/video/...")

if st.button("🚀 Start Multimodal Processing", type="primary"):
    if not url_input or not api_key:
        st.warning("Please check URL and API Key.")
        st.stop()
    
    # Save current configuration
    current_config = {
        "api_key": api_key,
        "base_url": base_url,
        "model_name": model_name,
        "language": language,
        "hardware_mode": hardware_mode,
        "enable_vision": enable_vision,
        "vision_interval": vision_interval,
        "vision_detail": vision_detail,
        "detail_level": detail_level
    }
    save_config(current_config)
    
    # 1. Download
    with st.status("📥 Downloading...", expanded=True) as status:
        dl_res = download_video(url_input)
        if not dl_res["success"]:
            status.update(label="Download Failed", state="error")
            st.error(dl_res["error"])
            st.stop()
        video_path = dl_res["video_path"]
        video_title = dl_res["title"]
        project_dir = dl_res["project_dir"]  # NEW: get project directory

        # [Patch A] Validation
        if not video_path or not os.path.exists(video_path):
            st.error("❌ 视频下载失败，请检查 URL 或网络连接（是否需要梯子？）。")
            st.stop()

        status.update(label=f"Downloaded: {video_title}", state="complete")
        st.info(f"📁 Project directory: `{project_dir}`")
        
    # 2. Transcribe
    with st.status("🎙️ Transcribing...", expanded=True) as status:
        try:
            transcriber = TranscriberFactory.get_transcriber(hardware_mode)
            segments = transcriber.transcribe(video_path)
            
            # [Patch A] Empty transcription check
            transcript_text = " ".join([seg['text'] for seg in segments])
            if not transcript_text or len(transcript_text.strip()) == 0:
                st.warning("⚠️ 视频中未检测到语音，无法生成总结。 (Transcription result is empty)")
                st.stop()
                
            status.update(label=f"Transcribed {len(segments)} segments", state="complete")
        except Exception as e:
            status.update(label="Transcription Failed", state="error")
            st.error(str(e))
            st.stop()

    # 3. Vision Processing (Optional)
    vision_frames = []
    if enable_vision:
        with st.status("👁️ Analyzing Vision...", expanded=True) as status:
            # Pass project_dir to save frames in subdirectory
            vision_frames = process_video_for_vision(video_path, vision_interval, output_dir=project_dir)
            status.update(label=f"Extracted {len(vision_frames)} visual samples", state="complete")
            if vision_frames:
                st.info(f"🖼️ Frames saved to: `{project_dir}/frames/`")

    # 4. Summary & Render
    st.markdown("### 📝 Smart Summary")
    
    summary_placeholder = st.empty()
    full_text = ""
    
    # Generator for stream
    stream = generate_summary_stream(
        video_title, transcript_text, segments, vision_frames, api_key, base_url, model_name, 
        language, detail_level, enable_vision, vision_interval, vision_detail
    )
    
    print(f"[DEBUG] Stream object received: {stream is not None}")
    
    if stream:
        print("[DEBUG] Entering stream processing loop...")
        chunk_count = 0
        content_count = 0
        
        for chunk in stream:
            chunk_count += 1
            
            if isinstance(chunk, str): # Error message
                print(f"[DEBUG] Received error string: {chunk}")
                st.error(chunk)
                break
            
            print(f"[DEBUG] Chunk {chunk_count}: {chunk}")
            
            
            try:
                delta = chunk.choices[0].delta
                
                # 火山引擎的推理模型使用 reasoning_content 而不是 content!
                content = getattr(delta, 'content', None) or getattr(delta, 'reasoning_content', None)
                
                if chunk_count <= 3:
                    print(f"[DEBUG] Chunk {chunk_count} - content: {repr(getattr(delta, 'content', None))}, reasoning_content: {repr(getattr(delta, 'reasoning_content', None))}")
                
                if content:
                    content_count += 1
                    full_text += content
                    
                    # Rendering Logic: Clear and Re-render fully on each chunk (simple but effective for interleaving)
                    # Optimization: Only re-render if newline? For now, re-render all to support dynamic layout.
                    
                    with summary_placeholder.container():
                        # Split by lines to find timestamps
                        lines = full_text.split('\n')
                        for line in lines:
                            st.markdown(line)
                            # Detect Timestamp [MM:SS]
                            match = re.search(r'\[(\d{1,2}:\d{2}(?::\d{2})?)\]', line)
                            if match:
                                ts_str = match.group(1)
                                seconds = timestamp_str_to_seconds(ts_str)
                                # Render precise frame
                                # To avoid re-extracting every refresh, check existence or cache?
                                # For simplicity, we overwrite/check existence.
                                img_path = f"temp/frame_render_{seconds}.jpg"
                                if not os.path.exists(img_path):
                                    extract_frame(video_path, seconds, img_path)
                                
                                if os.path.exists(img_path):
                                    st.image(img_path, caption=f"Time: {ts_str}", width=600)
            except Exception as e:
                print(f"[DEBUG] Error processing chunk {chunk_count}: {e}")
                import traceback
                traceback.print_exc()
        
        print(f"[DEBUG] Stream processing complete. Total chunks: {chunk_count}, Content chunks: {content_count}")
        print(f"[DEBUG] Final text length: {len(full_text)}")
    else:
        error_msg = "⚠️ LLM 没有返回任何响应。请检查：\n1. API Key 是否正确\n2. Base URL 是否正确\n3. 模型名称是否支持\n4. 网络连接是否正常"
        print(f"[DEBUG] {error_msg}")
        st.error(error_msg)

    st.success("Processing Complete!")
i