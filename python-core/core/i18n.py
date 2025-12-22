# Multi-language support dictionary

TRANSLATIONS = {
    # --- Sidebar / Navigation ---
    "nav_new_note": {"zh": "新建笔记", "en": "New Note"},
    "nav_history_title": {"zh": "历史记录", "en": "History"},
    "nav_settings_title": {"zh": "设置", "en": "Settings"},
    "nav_no_history": {"zh": "暂无历史记录", "en": "No history found"},
    "nav_delete": {"zh": "🗑️", "en": "🗑️"},
    # --- Main Page ---
    "page_title": {"zh": "OpenAutoNote智能笔记", "en": "OpenAutoNote Smart Summary"},
    "input_placeholder": {
        "zh": "请输入视频链接 (Bilibili/YouTube)...",
        "en": "Enter Video URL (YouTube/Bilibili)...",
    },
    "btn_start": {"zh": "🚀 开始智能分析", "en": "🚀 Start Analysis"},
    "status_downloading": {"zh": "📥 正在下载视频...", "en": "📥 Downloading video..."},
    "status_download_failed": {"zh": "❌ 下载失败", "en": "❌ Download Failed"},
    "status_transcribing": {"zh": "🎙️ 正在转录语音...", "en": "🎙️ Transcribing audio..."},
    "status_vision_analyzing": {
        "zh": "👁️ 正在分析视觉画面...",
        "en": "👁️ Analyzing visual content...",
    },
    "header_summary": {"zh": "📝 智能总结", "en": "📝 Smart Summary"},
    "expander_thinking": {"zh": "🤔 思考过程", "en": "🤔 Thinking Process"},
    "transcript_label": {"zh": "📄 字幕原文", "en": "📄 Transcript"},
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
    "lbl_enable_vision": {
        "zh": "启用多模态 (视觉分析)",
        "en": "Enable Multimodal (Vision)",
    },
    "lbl_vision_interval": {"zh": "截图间隔 (秒)", "en": "Vision Interval (sec)"},
    "lbl_vision_detail": {"zh": "图片清晰度", "en": "Image Detail"},
    "lbl_detail_level": {"zh": "总结详细程度", "en": "Summary Detail Level"},
    "lbl_output_lang": {"zh": "输出语言", "en": "Output Language"},
    # Hardware Tab
    "lbl_hardware_mode": {"zh": "加速模式", "en": "Acceleration Mode"},
    # System Tab
    "lbl_ui_lang": {"zh": "界面语言", "en": "UI Language"},
    "lbl_reset_app": {"zh": "重置所有设置", "en": "Reset All Settings"},
    "btn_clear_history": {"zh": "清空历史记录", "en": "Clear All History"},
    # --- Additional UI Elements ---
    "create_new_summary": {"zh": "新建总结", "en": "Create New Summary"},
    "paste_url_here": {
        "zh": "粘贴视频链接 (B站/YouTube)...",
        "en": "Paste video URL here (YouTube/Bilibili)...",
    },
    "advanced_options": {"zh": "高级选项", "en": "Advanced Options"},
    "custom_prompt_settings": {"zh": "自定义 prompt 设置", "en": "Custom Prompt Settings"},
    "complexity": {"zh": "复杂度", "en": "Complexity"},
    "custom_prompt": {"zh": "自定义 Prompt", "en": "Custom Prompt"},
    "custom_prompt_placeholder": {
        "zh": "输入自定义 Prompt，留空则使用系统默认的报告模板...",
        "en": "Enter custom Prompt, leave blank for default professional report template...",
    },
    "start_analysis": {"zh": "开始总结", "en": "Start Generation"},
    "view_original_transcript": {
        "zh": "查看原文 (Transcript)",
        "en": "View Original (Transcript)",
    },
    "ai_assistant": {"zh": "AI 助手", "en": "AI Assistant"},
    "ask_questions_about_content": {
        "zh": "针对左侧内容，随时向我提问...",
        "en": "Feel free to ask questions about the content on the left...",
    },
    "confirm_delete_all_history": {
        "zh": "确定要删除所有历史记录和本地文件吗？",
        "en": "Are you sure you want to delete all history and local files?",
    },
    "operation_irreversible": {
        "zh": "此操作不可撤销！",
        "en": "This operation is irreversible!",
    },
    "cancel": {"zh": "取消", "en": "Cancel"},
    "delete_all": {"zh": "删除全部", "en": "Delete All"},
    "all_records_deleted": {"zh": "所有记录已删除", "en": "All records deleted"},
    "rename": {"zh": "重命名", "en": "Rename"},
    "delete": {"zh": "删除", "en": "Delete"},
    "rename_successful": {"zh": "重命名成功", "en": "Rename successful"},
    "collapse_expand": {"zh": "收起/展开", "en": "Collapse/Expand"},
    "upload_image": {"zh": "上传图片", "en": "Upload Image"},
    "shift_enter_newline": {
        "zh": "Shift+Enter 换行...",
        "en": "Shift+Enter for new line...",
    },
    "quoted_context": {"zh": "引用上下文", "en": "Quoted Context"},
    "image": {"zh": "图片", "en": "Image"},
    "close": {"zh": "关闭", "en": "Close"},
    "save": {"zh": "保存", "en": "Save"},
    "new_title": {"zh": "新标题", "en": "New Title"},
    "cookies_netscape_format": {
        "zh": "🍪 Cookies (Netscape Format)",
        "en": "🍪 Cookies (Netscape Format)",
    },
    "paste_cookies_directly": {
        "zh": "直接粘贴您的 cookies。\n如有疑问，请搜索如何获取 cookie。",
        "en": "Paste your cookies directly.\nAsk AI or search to know how to get your cookie.",
    },
    "youtube_cookie": {"zh": "YouTube Cookie", "en": "YouTube Cookie"},
    "bilibili_cookie": {"zh": "Bilibili Cookie", "en": "Bilibili Cookie"},
    "lbl_youtube_cookie": {"zh": "YouTube Cookie", "en": "YouTube Cookie"},
    "lbl_bilibili_cookie": {"zh": "Bilibili Cookie", "en": "Bilibili Cookie"},
    "expand_ai_assistant": {"zh": "展开 AI 助手", "en": "Expand AI Assistant"},
    # --- Complexity Levels ---
    "complexity_level_1": {
        "zh": "极简概要：只输出最核心的3-5个要点，复杂度为1（总共1-5）。",
        "en": "Ultra Simple Summary: Output only the 3-5 core points, complexity level 1 (out of 5).",
    },
    "complexity_level_2": {
        "zh": "简洁总结：输出主要观点和关键信息，复杂度为2（总共1-5）。",
        "en": "Concise Summary: Output main points and key information, complexity level 2 (out of 5).",
    },
    "complexity_level_3": {
        "zh": "标准报告：完整结构化总结，包含主要章节和细节，复杂度为3（总共1-5）。",
        "en": "Standard Report: Complete structured summary with main sections and details, complexity level 3 (out of 5).",
    },
    "complexity_level_4": {
        "zh": "详细报告：深入分析每个要点，包含更多细节和示例，复杂度为4（总共1-5）。",
        "en": "Detailed Report: In-depth analysis of each point with more details and examples, complexity level 4 (out of 5).",
    },
    "complexity_level_5": {
        "zh": "深度解析：全面详尽的分析报告，包含所有细节、引申讨论和多角度分析，复杂度为5（总共1-5）。",
        "en": "Deep Analysis: Comprehensive and detailed report with all details, extended discussions and multi-angle analysis, complexity level 5 (out of 5).",
    },
    # --- Style Guidelines ---
    "writing_style": {
        "zh": "专业、犀利、结构化",
        "en": "professional, sharp, and structured",
    },
    "avoid_chronological": {
        "zh": "拒绝流水账：不要说'视频首先讲了...然后讲了...'，直接陈述事实和观点。",
        "en": "Avoid chronological narrative: Don't say 'the video first talked about... then talked about...', directly state facts and viewpoints.",
    },
    # --- User Prompt ---
    "user_content_prompt": {
        "zh": "视频标题: {title}\n字幕内容:\n{full_text}\n\n请总结上述内容。",
        "en": "Video Title: {title}\nTranscript:\n{full_text}\n\nPlease summarize the above content.",
    },
    # --- Context Menu ---
    "context_menu_rename": {"zh": "重命名", "en": "Rename"},
    "context_menu_delete": {"zh": "删除", "en": "Delete"},
    # --- Complexity Options ---
    "complexity_option_1": {
        "zh": "⚡ 极简（默认不生成目录）",
        "en": "⚡ Ultra Simple (Default No Table of Contents)",
    },
    "complexity_option_2": {
        "zh": "📝 简洁（默认不生成目录）",
        "en": "📝 Concise (Default No Table of Contents)",
    },
    "complexity_option_3": {
        "zh": "📊 标准（系统判断是否生成目录）",
        "en": "📊 Standard (System Judgment on Whether to Generate Table of Contents)",
    },
    "complexity_option_4": {
        "zh": "📚 详细（默认生成目录）",
        "en": "📚 Detailed (Default Table of Contents)",
    },
    "complexity_option_5": {
        "zh": "🔬 深度（默认生成目录）",
        "en": "🔬 Deep (Default Table of Contents)",
    },
    # --- System Terminal ---
    "system_terminal": {"zh": "📟 系统日志终端", "en": "📟 System Log Terminal"},
    # --- Selection Label ---
    "selection_label": {"zh": "已选中 {count} 字", "en": "Selected {count} characters"},
    # --- AI Assistant ---
    "ai_title_label": {"zh": "【标题】", "en": "【Title】"},
    "ai_summary_label": {"zh": "【总结】", "en": "【Summary】"},
    "ai_assistant_prompt": {
        "zh": "你是助手。基于以下信息回答：",
        "en": "You are an assistant. Answer based on the following information:",
    },
    "ai_label": {"zh": "AI", "en": "AI"},
    # --- System Prompt ---
    "base_identity_zh": {
        "zh": '你是一个 AI 笔记软件，叫做 OpenAutoNote，系统会给你输入音频转文字生成的文本和截取的视频帧。按要求为用户生成总结，帮助他快速了解视频内容和进行学习。\n你的风格是：**专业、犀利、结构化**，类似于 "The Verge", "Notion Blog" 或 "少数派" 的深度文章。\n\n任务目标：将视频内容转化为一份**视觉化、杂志级的深度报告**。\n\n重要提醒：由于语音识别模型的精度限制，转录文本中可能存在同音字或近音字的错误。请在分析时注意识别并纠正这些潜在的错误，确保最终生成的内容准确且符合逻辑。\n\n**CRITICAL INSTRUCTION: OUTPUT LANGUAGE**\n你必须使用 **{default_lang}** 进行输出。如果界面语言是中文，所有标题、正文、解释必须使用中文。严禁中英混杂。',
        "en": 'You are an AI note-taking software called OpenAutoNote. The system will input text generated from audio transcription and captured video frames. Generate a summary according to requirements to help users quickly understand the video content and learn.\nYour style is: **professional, sharp, and structured**, similar to in-depth articles from "The Verge", "Notion Blog" or "少数派".\n\nTask objective: Transform video content into a **visual, magazine-level in-depth report**.\n\nImportant Reminder: Due to limitations in speech recognition model accuracy, the transcribed text may contain homophone or near-homophone errors. Please pay attention to identify and correct these potential errors during analysis to ensure the final generated content is accurate and logically consistent.\n\n**CRITICAL INSTRUCTION: OUTPUT LANGUAGE**\nYou MUST use **{default_lang}** for output. If the UI language is Chinese, all headers, body text, and explanations MUST be in Chinese. Do NOT mix English and Chinese.',
    },
    "base_identity_en": {
        "zh": '你是一个 AI 笔记软件，叫做 OpenAutoNote，系统会给你输入音频转文字生成的文本和截取的视频帧。按要求为用户生成总结，帮助他快速了解视频内容和进行学习。\n你的风格是：**专业、犀利、结构化**，类似于 "The Verge", "Notion Blog" 或 "少数派" 的深度文章。\n\n任务目标：将视频内容转化为一份**视觉化、杂志级的深度报告**。\n\n重要提醒：由于语音识别模型的精度限制，转录文本中可能存在同音字或近音字的错误。请在分析时注意识别并纠正这些潜在的错误，确保最终生成的内容准确且符合逻辑。\n\n**CRITICAL INSTRUCTION: OUTPUT LANGUAGE**\n你必须使用 **{default_lang}** 进行输出。如果界面语言是中文，所有标题、正文、解释必须使用中文。严禁中英混杂。',
        "en": 'You are an AI note-taking software called OpenAutoNote. The system will input text generated from audio transcription and captured video frames. Generate a summary according to requirements to help users quickly understand the video content and learn.\nYour style is: **professional, sharp, and structured**, similar to in-depth articles from "The Verge", "Notion Blog" or "The Verge".\n\nTask objective: Transform video content into a **visual, magazine-level in-depth report**.\n\nImportant Reminder: Due to limitations in speech recognition model accuracy, the transcribed text may contain homophone or near-homophone errors. Please pay attention to identify and correct these potential errors during analysis to ensure the final generated content is accurate and logically consistent.\n\n**CRITICAL INSTRUCTION: OUTPUT LANGUAGE**\nYou MUST use **{default_lang}** for output. If the UI language is Chinese, all headers, body text, and explanations MUST be in Chinese. Do NOT mix English and Chinese.',
    },
    "language_style": {
        "zh": '\n### 语言风格\n-   **语言选择**：除非用户在prompt中明确指定语言，否则请使用与UI界面相同的语言（当前界面语言：{default_lang}）。\n-   **中文**：如果使用中文，请使用流畅、专业的简体中文。\n-   **英文**：如果使用英文，请使用流畅、专业的英文。\n-   **拒绝流水账**：不要说"视频首先讲了...然后讲了..."，直接陈述事实和观点。\n-   **格式规范**：不要将整个输出包裹在代码块中（如 ```markdown ... ```）。直接输出 Markdown 内容。',
        "en": '\n### Language Style\n-   **Language Selection**：Unless the user explicitly specifies a language in the prompt, please use the same language as the UI interface (current interface language: {default_lang}).\n-   **Chinese**：If using Chinese, please use fluent, professional Simplified Chinese.\n-   **English**：If using English, please use fluent, professional English.\n-   **Avoid Narrative Flow**：Do not say "The video first talked about... then talked about...", directly state facts and viewpoints.\n-   **Formatting**: Do NOT wrap the entire output in a code block (e.g. ```markdown ... ```). Output raw Markdown content directly.',
    },
    "user_extra_requirement": {
        "zh": "【用户额外要求 - 优先满足】：",
        "en": "[User Additional Requirements - Priority Satisfaction]: ",
    },
    "output_complexity_requirement": {
        "zh": "【输出复杂度要求】：",
        "en": "[Output Complexity Requirement]: ",
    },
    "core_layout_requirements": {
        "zh": "### 核心布局要求 (Strict Layout)",
        "en": "### Core Layout Requirements (Strict Layout)",
    },
    "the_one_liner": {
        "zh": "1.  **一句话金句 (The One-Liner)**",
        "en": "1.  **The One-Liner**",
    },
    "the_one_liner_desc": {
        "zh": "    -   在开头必须使用 Blockquote (`>`) 提炼出视频最核心的价值或结论。\n    -   例如：`> 💡 **核心洞察**：Firefox 的衰落并非技术落后，而是移动互联网时代商业模式的必然溃败。`",
        "en": "    -   At the beginning, you must use a Blockquote (`>`) to extract the most core value or conclusion of the video.\n    -   Example: `> 💡 **Core Insight**: Firefox's decline is not due to technical backwardness, but the inevitable failure of its business model in the mobile Internet era.`",
    },
    "structured_toc": {
        "zh": "2.  **结构化目录 (Table of Contents)**",
        "en": "2.  **Structured Table of Contents**",
    },
    "structured_toc_desc": {
        "zh": "    -   在复杂度为 4 和 5 时必须生成目录！在复杂度为 3 时，你根据视频内容复杂度和内容量决定是否生成目录。在复杂度为 1、2 时不生成目录。\n    -   紧接着核心洞察之后，必须生成一个**可点击跳转的目录**。\n    -   格式要求：使用 Markdown 锚点链接，例如：\n    ```\n    ## 📑 目录\n    - [🎯 章节一标题](#章节一标题)\n    - [⚡ 章节二标题](#章节二标题)\n    - [💰 章节三标题](#章节三标题)\n    ```\n    -   每个章节标题在正文中使用 `## 🎯 章节一标题` 格式，确保锚点链接可以正确跳转。",
        "en": "    -   Must be generated for complexity levels 4 and 5! For complexity level 3, decide whether to generate based on video content complexity and volume. Do not generate for complexity levels 1 and 2.\n    -   Immediately after the core insight, you must generate a **clickable table of contents**.\n    -   Format requirements: Use Markdown anchor links, for example:\n    ```\n    ## 📑 Table of Contents\n    - [🎯 Chapter 1 Title](#chapter-1-title)\n    - [⚡ Chapter 2 Title](#chapter-2-title)\n    - [💰 Chapter 3 Title](#chapter-3-title)\n    ```\n    -   Each chapter title in the main text uses the format `## 🎯 Chapter 1 Title` to ensure anchor links can jump correctly.",
    },
    "structured_sections": {
        "zh": "3.  **结构化章节**",
        "en": "3.  **Structured Sections**",
    },
    "structured_sections_desc": {
        "zh": "    -   使用 H2 (`##`) 划分主要模块，每个标题前加上对应的 Emoji。\n    -   **Emoji 列表**：禁止使用普通的黑点 bullet。必须根据上下文使用 Emoji：\n        -   🎯 核心观点 / 目标\n        -   ⚡ 技术亮点 / 痛点\n        -   💰 商业 / 成本\n        -   ⚠️ 风险 / 警告\n        -   🛠️ 解决方案 / 步骤\n        -   📊 数据分析\n        -   🔮 未来展望",
        "en": "    -   Use H2 (`##`) to divide main modules, with corresponding Emoji before each title.\n    -   **Emoji List**: Prohibit the use of ordinary black dot bullets. Must use Emoji according to context:\n        -   🎯 Core Views / Goals\n        -   ⚡ Technical Highlights / Pain Points\n        -   💰 Business / Cost\n        -   ⚠️ Risks / Warnings\n        -   🛠️ Solutions / Steps\n        -   📊 Data Analysis\n        -   🔮 Future Outlook",
    },
    "data_comparison": {
        "zh": "4.  **数据对比 (必须使用表格)**",
        "en": "4.  **Data Comparison (Must Use Tables)**",
    },
    "data_comparison_desc": {
        "zh": "    -   如果视频中出现对比（如 A vs B，今年 vs 去年），**必须**输出标准 Markdown Table 格式。\n    -   表格必须包含表头和分隔线，例如：\n    ```\n    | 项目 | 数值1 | 数值2 |\n    |------|-------|-------|\n    | 指标A | 100 | 200 |\n    | 指标B | 300 | 400 |\n    ```",
        "en": "    -   If comparisons appear in the video (such as A vs B, this year vs last year), **must** output standard Markdown Table format.\n    -   Tables must include headers and separator lines, for example:\n    ```\n    | Item | Value1 | Value2 |\n    |------|-------|-------|\n    | MetricA | 100 | 200 |\n    | MetricB | 300 | 400 |\n    ```",
    },
    "visual_evidence": {
        "zh": "5.  **视觉证据 (Selective Images)**",
        "en": "5.  **Visual Evidence (Selective Images)**",
    },
    "visual_evidence_desc": {
        "zh": "    -   **原则**：宁缺毋滥。仅在关键时刻（PPT图表、独特产品细节）插入截图。\n    -   **位置**：将截图时间戳 `[MM:SS]` 直接插入在最相关的段落之后。",
        "en": "    -   **Principle**: Better to have none than too many. Only insert screenshots at critical moments (PPT charts, unique product details).\n    -   **Position**: Insert the screenshot timestamp `[MM:SS]` directly after the most relevant paragraph.",
    },
    "math_formulas": {
        "zh": "6.  **数学公式 (必须使用LaTeX格式)**",
        "en": "6.  **Mathematical Formulas (Must Use LaTeX Format)**",
    },
    "math_formulas_desc": {
        "zh": "    -   如果视频中出现数学公式、方程或符号，**必须**使用LaTeX格式输出。\n    -   **严禁**使用 `\[ ... \]` 或 `\( ... \)` 格式。\n    -   行内公式**必须**使用 `$...$` 包裹，例如：`$E=mc^2$`\n    -   块级公式**必须**使用 `$$...$$` 包裹，例如：`$$\\sum_{i=1}^n x_i$$`\n    -   确保 LaTeX 语法标准，兼容 KaTeX。",
        "en": "    -   If mathematical formulas, equations or symbols appear in the video, **must** output using LaTeX format.\n    -   **STRICTLY FORBIDDEN**: Do NOT use `\[ ... \]` or `\( ... \)` format.\n    -   Inline formulas **MUST** use `$...$` wrapping, e.g., `$E=mc^2$`\n    -   Block formulas **MUST** use `$$...$$` wrapping, e.g., `$$\\sum_{i=1}^n x_i$$`\n    -   Ensure standard LaTeX syntax compatible with KaTeX.",
    },
    "chunk_summary_requirements": {
        "zh": "### 分块总结要求\n\n1.  **结构化摘要**：每块必须包含清晰的结构，包括核心观点、主要内容和关键要点。\n2.  **关键引用**：必须提取视频中最重要的引用和对话，使用 `> ` 格式引用。\n3.  **时间范围**：在开头明确标注该块的时间范围，例如 `[00:00-15:00]`。\n4.  **独立性**：每块摘要应独立完整，能够单独理解该部分内容。\n5.  **上下文保留**：保留足够的上下文信息，以便后续生成总总结时能够理解各块之间的联系。\n6.  **思考过程**：在生成内容时，请先输出您的思考过程，解释您是如何分析和组织信息的。这部分内容将作为AI思考过程显示给用户。\n7.  **连续编号**：请注意这是系列报告的一部分。如果上一部分的最后一个章节编号是 2，那么这一部分必须从 3 开始编号。保持整体结构的连贯性。",
        "en": "### Chunk Summary Requirements\n\n1.  **Structured Summary**: Each chunk must have a clear structure including core viewpoints, main content, and key points.\n2.  **Key Quotes**: Must extract the most important quotes and dialogues from the video, using `> ` format for quotes.\n3.  **Time Range**: Clearly indicate the time range of the chunk at the beginning, for example `[00:00-15:00]`.\n4.  **Independence**: Each chunk summary should be independently complete, allowing understanding of that section on its own.\n5.  **Context Preservation**: Retain sufficient contextual information to enable understanding of connections between chunks when generating the final summary.\n6.  **Thinking Process**: When generating content, please first output your thinking process, explaining how you analyzed and organized the information. This content will be displayed to users as the AI thinking process.\n7.  **Continuous Numbering**: Note that this is part of a series report. If the last section number in the previous part was 2, this part MUST start with section 3. Maintain structural continuity.",
    },
    # --- Default Config ---
    "default_language": {"zh": "简体中文 (默认)", "en": "Simplified Chinese (Default)"},
    "default_detail_level": {"zh": "标准", "en": "Standard"},
    "vision_detail_low": {"zh": "低 (快速, 推荐, 720p)", "en": "Low (Fast, Recommanded 720p)"},
    "vision_detail_high": {"zh": "高 (细节, 1080p)", "en": "High (Detail, 1080p)"},
    "vision_detail_auto": {"zh": "自动", "en": "Auto"},
    "hardware_mode_cpu": {"zh": "CPU (faster-whisper)", "en": "CPU (faster-whisper)"},
    "hardware_mode_cuda": {
        "zh": "NVIDIA CUDA (faster-whisper)",
        "en": "NVIDIA CUDA (faster-whisper)",
    },
    "hardware_mode_mlx": {
        "zh": "Apple Silicon (mlx-whisper)",
        "en": "Apple Silicon (mlx-whisper)",
    },
}


def get_text(key: str, lang: str = "zh") -> str:
    """Retrieve translated text for a given key."""
    # Fallback to 'zh' if lang not found, or key not found
    lang_map = TRANSLATIONS.get(key, {})
    return lang_map.get(lang, lang_map.get("zh", key))
