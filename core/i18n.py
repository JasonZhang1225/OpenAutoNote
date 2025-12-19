# Multi-language support dictionary

TRANSLATIONS = {
    # --- Sidebar / Navigation ---
    "nav_new_note": {
        "zh": "📝 新建笔记",
        "en": "📝 New Note"
    },
    "nav_history_title": {
        "zh": "📜 历史记录",
        "en": "📜 History"
    },
    "nav_settings_title": {
        "zh": "⚙️ 设置",
        "en": "⚙️ Settings"
    },
    "nav_no_history": {
        "zh": "暂无历史记录",
        "en": "No history found"
    },
    "nav_delete": {
        "zh": "🗑️",
        "en": "🗑️"
    },

    # --- Main Page ---
    "page_title": {
        "zh": "OpenAutoNote 智能笔记",
        "en": "OpenAutoNote Smart Summary"
    },
    "input_placeholder": {
        "zh": "请输入视频链接 (Bilibili/YouTube)...",
        "en": "Enter Video URL (Bilibili/YouTube)..."
    },
    "btn_start": {
        "zh": "🚀 开始智能分析",
        "en": "🚀 Start Analysis"
    },
    "status_downloading": {
        "zh": "📥 正在下载视频...",
        "en": "📥 Downloading video..."
    },
    "status_download_failed": {
        "zh": "❌ 下载失败",
        "en": "❌ Download Failed"
    },
    "status_transcribing": {
        "zh": "🎙️ 正在转录语音...",
        "en": "🎙️ Transcribing audio..."
    },
    "status_vision_analyzing": {
        "zh": "👁️ 正在分析视觉画面...",
        "en": "👁️ Analyzing visual content..."
    },
    "header_summary": {
        "zh": "📝 智能总结",
        "en": "📝 Smart Summary"
    },
    "expander_thinking": {
        "zh": "🤔 思考过程",
        "en": "🤔 Thinking Process"
    },
    "transcript_label": {
        "zh": "📄 字幕原文",
        "en": "📄 Transcript"
    },

    # --- Settings Tabs ---
    "tab_api": {"zh": "API 设置", "en": "API Settings"},
    "tab_gen": {"zh": "生成设置", "en": "Generation"},
    "tab_hardware": {"zh": "硬件加速", "en": "Hardware"},
    "tab_system": {"zh": "系统", "en": "System"},

    # API Tab
    "lbl_api_key": {"zh": "API 密钥", "en": "API Key"},
    "lbl_base_url": {"zh": "Base URL", "en": "Base URL"},
    "lbl_model": {"zh": "模型名称", "en": "Model Name"},
    
    # Generation Tab
    "lbl_enable_vision": {"zh": "启用多模态 (视觉分析)", "en": "Enable Multimodal (Vision)"},
    "lbl_vision_interval": {"zh": "截图间隔 (秒)", "en": "Vision Interval (sec)"},
    "lbl_vision_detail": {"zh": "图片清晰度", "en": "Image Detail"},
    "lbl_detail_level": {"zh": "总结详细程度", "en": "Summary Detail Level"},
    "lbl_output_lang": {"zh": "输出语言", "en": "Output Language"},
    
    # Hardware Tab
    "lbl_hardware_mode": {"zh": "加速模式", "en": "Acceleration Mode"},
    
    # System Tab
    "lbl_ui_lang": {"zh": "界面语言 / UI Language", "en": "UI Language"},
    "lbl_reset_app": {"zh": "重置所有设置", "en": "Reset All Settings"},
    "btn_clear_history": {"zh": "清空历史记录", "en": "Clear All History"},
}

def get_text(key: str, lang: str = "zh") -> str:
    """Retrieve translated text for a given key."""
    # Fallback to 'zh' if lang not found, or key not found
    lang_map = TRANSLATIONS.get(key, {})
    return lang_map.get(lang, lang_map.get("zh", key))
