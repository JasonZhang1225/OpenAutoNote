import re
from typing import List, Dict, Any


def clean_bilibili_url(url: str) -> str:
    """
    Extract the BV ID from any Bilibili URL (video, watch-later, share links)
    and return a normalized video URL.
    """
    match = re.search(r"(BV[a-zA-Z0-9]{10})", url)
    if match:
        bvid = match.group(1)
        return f"https://www.bilibili.com/video/{bvid}"
    return url

def sanitize_filename(filename: str) -> str:
    """
    Sanitize filename by removing/replacing filesystem-unsafe characters.
    Supports Windows, macOS, and Linux file systems.
    """
    # Replace common unsafe characters with underscores
    unsafe_chars = r'[<>:"/\\|?*]'
    sanitized = re.sub(unsafe_chars, '_', filename)
    
    # Remove leading/trailing spaces and dots (Windows issue)
    sanitized = sanitized.strip('. ')
    
    # Replace multiple consecutive underscores with single underscore
    sanitized = re.sub(r'_+', '_', sanitized)
    
    # Limit length to avoid filesystem issues (max 255 chars on most systems)
    max_length = 200  # Leave room for extensions and subdirs
    if len(sanitized) > max_length:
        sanitized = sanitized[:max_length].rstrip('_')
    
    # Fallback if empty after sanitization
    if not sanitized:
        sanitized = "untitled"
    
    return sanitized

def build_multimodal_payload(title: str, full_text: str, segments: List[Dict[str, Any]], visual_samples: List[Dict[str, Any]], detail: str = "low") -> List[Dict[str, Any]]:
    """
    Logic Patch B: Construct "Text-Image Interleaved" message payload.
    Slices the video timeline based on visual sample intervals and fills text slots accordingly.
    """
    messages_content = [{"type": "text", "text": f"视频标题: {title}\n请结合以下画面和文字生成总结：\n"}]
    
    # If no visual data, return text only
    if not visual_samples:
        messages_content.append({"type": "text", "text": full_text})
        return messages_content

    # --- Core Algorithm: Dual Pointer Merge ---
    # visual_samples is sorted by timestamp: [{'timestamp': 5, 'base64': '...'}, {'timestamp': 10...}]
    
    last_end_time = 0.0
    
    for sample in visual_samples:
        current_time = sample['timestamp']
        img_b64 = sample['base64']
        
        # 1. Collect text buffer for current window (last_end_time -> current_time)
        current_text_block = ""
        for seg in segments:
            # Check if the segment's midpoint falls within the current time window
            mid_point = (seg['start'] + seg['end']) / 2
            if last_end_time <= mid_point < current_time:
                current_text_block += seg['text'] + " "
        
        # 2. Assemble: Text first, then Image
        if current_text_block:
            messages_content.append({"type": "text", "text": f"\n(Time {int(last_end_time)}-{int(current_time)}s): {current_text_block}"})
        
        # 3. Insert screenshot at this moment (Low Detail to save tokens)
        messages_content.append({
            "type": "image_url", 
            "image_url": {
                "url": f"data:image/jpeg;base64,{img_b64}",
                "detail": detail
            }
        })
        
        last_end_time = current_time
        
    # 4. Add remaining text (content after the last frame)
    remaining_text = ""
    for seg in segments:
        if (seg['start'] + seg['end']) / 2 >= last_end_time:
            remaining_text += seg['text'] + " "
            
    if remaining_text:
        messages_content.append({"type": "text", "text": f"\n(Remaining): {remaining_text}"})
    
    # Add final instruction
    messages_content.append({"type": "text", "text": "\n\n请根据以上画面和字幕内容生成详细总结。"})
    
    return messages_content

def seconds_to_hms(seconds):
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    if h > 0:
        return f"{int(h):02d}:{int(m):02d}:{int(s):02d}"
    return f"{int(m):02d}:{int(s):02d}"

def timestamp_str_to_seconds(timestamp_str):
    parts = list(map(int, timestamp_str.split(':')))
    if len(parts) == 2:
        return parts[0] * 60 + parts[1]
    elif len(parts) == 3:
        return parts[0] * 3600 + parts[1] * 60 + parts[2]
    return 0
