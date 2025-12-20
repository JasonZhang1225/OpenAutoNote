import os
import yt_dlp
import traceback
import shutil
import time
from pathlib import Path
from core.utils import sanitize_filename
from datetime import datetime


def download_video(
    url: str,
    output_dir: str = "generate",
    cookies_yt: str = None,
    cookies_bili: str = None,
    strict_output: bool = False,
    progress_hook=None,
) -> dict:
    """
    Downloads video from URL using yt-dlp with bulletproof settings.
    Features:
    - Auto-detects and uses aria2c if available (multi-connection download).
    - Aggressive retry settings for HTTP and fragment errors.
    - Python-level retry wrapper for global resilience.
    - Accepts cookie CONTENT (Netscape format) directly, not file paths.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Auto-select cookie based on URL domain
    # Write cookie content to temp file if provided
    cookie_path = None
    temp_cookie_file = None

    if "youtube" in url.lower() or "youtu.be" in url.lower():
        if cookies_yt and cookies_yt.strip():
            temp_cookie_file = os.path.join(output_dir, ".cookies_yt.txt")
            with open(temp_cookie_file, "w", encoding="utf-8") as f:
                f.write(cookies_yt)
            cookie_path = temp_cookie_file
            print("[Downloader] Using YouTube cookies (from config)")
    elif "bilibili" in url.lower():
        if cookies_bili and cookies_bili.strip():
            temp_cookie_file = os.path.join(output_dir, ".cookies_bili.txt")
            with open(temp_cookie_file, "w", encoding="utf-8") as f:
                f.write(cookies_bili)
            cookie_path = temp_cookie_file
            print("[Downloader] Using Bilibili cookies (from config)")

    # --- Bulletproof Base Options ---
    base_opts = {
        "format": "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
        "noplaylist": True,
        "quiet": False,  # Show progress for debugging
        "no_warnings": False,
        # Aggressive Retry Settings
        "retries": 10,
        "fragment_retries": 10,
        "skip_unavailable_fragments": False,
        "socket_timeout": 30,
        "continuedl": True,
        # File Overwrite
        "overwrites": True,
    }

    # Auto-detect aria2c for multi-connection download
    # NOTE: Disable aria2 if progress_hook is provided, as aria2 doesn't support progress callbacks
    if shutil.which("aria2c") and not progress_hook:
        print("[Downloader] aria2c detected! Enabling multi-connection download.")
        base_opts["external_downloader"] = "aria2c"
        base_opts["external_downloader_args"] = {
            "aria2c": ["-x", "16", "-k", "1M", "-s", "16", "--continue=true"]
        }
    elif progress_hook:
        print("[Downloader] Using native downloader for progress tracking.")
    else:
        print("[Downloader] aria2c not found. Using default downloader.")

    if progress_hook:
        base_opts["progress_hooks"] = [progress_hook]

    if cookie_path:
        base_opts["cookiefile"] = cookie_path

    # --- Python-Level Retry Wrapper ---
    MAX_GLOBAL_RETRIES = 3
    RETRY_COOLDOWN = 5  # seconds

    try:
        # --- Pass 1: Extract Metadata (No download) ---
        print(f"[Downloader] Extracting metadata for: {url}")
        meta_opts = base_opts.copy()
        meta_opts["quiet"] = True  # Keep metadata extraction quiet

        with yt_dlp.YoutubeDL(meta_opts) as ydl:
            info = ydl.extract_info(url, download=False)

            if not isinstance(info, dict):
                raise ValueError(f"yt-dlp returned unexpected type: {type(info)}")

            if "entries" in info:
                if len(info["entries"]) > 0:
                    info = info["entries"][0]
                else:
                    raise ValueError("Got empty playlist entries.")

            title = info.get("title", "Unknown Title")
            duration = info.get("duration")

        # --- Prepare Paths ---
        sanitized_title = sanitize_filename(title)

        if strict_output:
            final_dir = output_dir
            download_opts = base_opts.copy()
            download_opts["outtmpl"] = f"{final_dir}/%(id)s.%(ext)s"

            print(f"[Downloader] Downloading strictly to: {final_dir}")

            # --- Download with Python-Level Retry ---
            last_error = None
            for attempt in range(MAX_GLOBAL_RETRIES):
                try:
                    print(f"[Downloader] Attempt {attempt + 1}/{MAX_GLOBAL_RETRIES}...")
                    with yt_dlp.YoutubeDL(download_opts) as ydl:
                        info = ydl.extract_info(url, download=True)
                        video_path = ydl.prepare_filename(info)
                    break  # Success!
                except Exception as e:
                    last_error = e
                    print(f"[Downloader] Attempt {attempt + 1} failed: {str(e)}")
                    if attempt < MAX_GLOBAL_RETRIES - 1:
                        print(f"[Downloader] Retrying in {RETRY_COOLDOWN}s...")
                        time.sleep(RETRY_COOLDOWN)
                    else:
                        raise last_error

            print(f"[Downloader] Success! Video at: {video_path}")
            return {
                "success": True,
                "title": title,
                "video_path": video_path,
                "project_dir": final_dir,
                "duration": duration,
                "error": None,
            }

        else:
            # Legacy behavior (non-strict)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            temp_dir = os.path.join(output_dir, f"temp_{timestamp}")
            Path(temp_dir).mkdir(parents=True, exist_ok=True)

            download_opts = base_opts.copy()
            download_opts["outtmpl"] = f"{temp_dir}/%(id)s.%(ext)s"

            print(f"[Downloader] Downloading to: {temp_dir}")

            # --- Download with Python-Level Retry ---
            last_error = None
            for attempt in range(MAX_GLOBAL_RETRIES):
                try:
                    print(f"[Downloader] Attempt {attempt + 1}/{MAX_GLOBAL_RETRIES}...")
                    with yt_dlp.YoutubeDL(download_opts) as ydl:
                        info = ydl.extract_info(url, download=True)
                        video_path_temp = ydl.prepare_filename(info)
                    break  # Success!
                except Exception as e:
                    last_error = e
                    print(f"[Downloader] Attempt {attempt + 1} failed: {str(e)}")
                    if attempt < MAX_GLOBAL_RETRIES - 1:
                        print(f"[Downloader] Retrying in {RETRY_COOLDOWN}s...")
                        time.sleep(RETRY_COOLDOWN)
                    else:
                        raise last_error

            final_dir = os.path.join(output_dir, sanitized_title)
            if os.path.exists(final_dir):
                final_dir = os.path.join(output_dir, f"{sanitized_title}_{timestamp}")

            os.rename(temp_dir, final_dir)

            video_filename = os.path.basename(video_path_temp)
            video_path = os.path.join(final_dir, video_filename)

            print(f"[Downloader] Success! Video at: {video_path}")

            return {
                "success": True,
                "title": title,
                "video_path": video_path,
                "project_dir": final_dir,
                "duration": duration,
                "error": None,
            }

    except Exception as e:
        error_details = traceback.format_exc()
        print(f"[Downloader] Fatal Error:\n{error_details}")
        return {
            "success": False,
            "error": f"{str(e)}",
            "title": None,
            "video_path": None,
            "project_dir": None,
            "duration": None,
        }
