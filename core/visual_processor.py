import cv2
import os

def extract_frame(video_path: str, timestamp_seconds: float, output_path: str, ensure_dir: bool = True) -> bool:
    """
    Extracts a frame from the video at the specific timestamp and saves it to output_path.
    Returns True if successful, False otherwise.
    """
    # Ensure output directory exists
    if ensure_dir:
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
    if not os.path.exists(video_path):
        return False

    try:
        cap = cv2.VideoCapture(video_path)
        
        # Calculate frame position
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_no = int(fps * timestamp_seconds)
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
        ret, frame = cap.read()
        
        if ret:
            cv2.imwrite(output_path, frame)
            cap.release()
            return True
        else:
            cap.release()
            return False
    except Exception as e:
        print(f"Error extracting frame: {e}")
        return False

import base64
import numpy as np
from typing import List, Dict, Any

def process_video_for_vision(video_path: str, interval: int = 15, output_dir: str = None) -> List[Dict[str, Any]]:
    """
    Samples frames from video at fixed intervals, resizes them, and converts to Base64.
    Returns a list of dicts: [{'timestamp': int, 'base64': str}]
    
    Args:
        video_path: Path to video file
        interval: Seconds between frame samples
        output_dir: Optional directory to save frame images (for debugging)
    """
    if not os.path.exists(video_path):
        return []

    results = []
    try:
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps
        
        current_time = 0
        while current_time < duration:
            frame_no = int(fps * current_time)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
            ret, frame = cap.read()
            
            if ret:
                # Resize to max 512px (width or height)
                h, w = frame.shape[:2]
                scale = 512 / max(h, w)
                if scale < 1:
                    new_w = int(w * scale)
                    new_h = int(h * scale)
                    frame = cv2.resize(frame, (new_w, new_h))

                # Encode to JPEG Base64
                _, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 60])
                base64_str = base64.b64encode(buffer).decode('utf-8')
                
                results.append({
                    "timestamp": int(current_time),
                    "base64": base64_str
                })
                
                # Optionally save frame to disk for debugging
                if output_dir:
                    frame_path = os.path.join(output_dir, "frames", f"frame_{int(current_time):04d}.jpg")
                    os.makedirs(os.path.dirname(frame_path), exist_ok=True)
                    cv2.imwrite(frame_path, frame)
            
            current_time += interval
            
        cap.release()
        return results
    except Exception as e:
        print(f"Error processing video for vision: {e}")
        return []
