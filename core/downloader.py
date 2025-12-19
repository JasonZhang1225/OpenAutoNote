import os
import yt_dlp
from pathlib import Path
from core.utils import sanitize_filename

def download_video(url: str, output_dir: str = "temp", cookie_file: str = "cookies.txt") -> dict:
    """
    Downloads video from URL using yt-dlp.
    Returns a dictionary containing 'title', 'video_path', 'project_dir', etc.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Check if cookie file exists
    cookie_path = None
    if cookie_file and os.path.exists(cookie_file):
        cookie_path = cookie_file

    ydl_opts = {
        'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best',  # Prefer MP4
        'outtmpl': '%(id)s.%(ext)s',  # Temporary, will be moved
        'noplaylist': True,
        'quiet': True,
        'no_warnings': True,
    }

    if cookie_path:
        ydl_opts['cookiefile'] = cookie_path

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            # Extract info first to get title
            info = ydl.extract_info(url, download=False)
            title = info.get('title', 'Unknown Title')
            
            # Create project directory based on sanitized title
            sanitized_title = sanitize_filename(title)
            project_dir = os.path.join(output_dir, sanitized_title)
            Path(project_dir).mkdir(parents=True, exist_ok=True)
            
            # Update output template to use project directory
            ydl_opts['outtmpl'] = f'{project_dir}/%(id)s.%(ext)s'
            
            # Download to project directory
            ydl.params.update(ydl_opts)
            info = ydl.extract_info(url, download=True)
            video_path = ydl.prepare_filename(info)
            
            return {
                "success": True,
                "title": title,
                "video_path": video_path,
                "project_dir": project_dir,
                "duration": info.get('duration'),
                "error": None
            }
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }
