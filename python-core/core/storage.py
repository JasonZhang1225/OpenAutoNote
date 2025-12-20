import json
import os
import shutil
import uuid
from datetime import datetime
from typing import List, Dict, Optional

# Get the directory where this module is located (python-core/core/)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HISTORY_FILE = os.path.join(BASE_DIR, "user_history.json")


def load_history() -> List[Dict]:
    """Load all sessions from history file."""
    if not os.path.exists(HISTORY_FILE):
        return []
    try:
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
            # Ensure it's a list
            if isinstance(data, dict) and "sessions" in data:
                return data["sessions"]
            elif isinstance(data, list):
                return data
            return []
    except Exception as e:
        print(f"[Storage] Error loading history: {e}")
        return []


def save_history(sessions: List[Dict]):
    """Save sessions to history file."""
    try:
        with open(HISTORY_FILE, "w", encoding="utf-8") as f:
            json.dump({"sessions": sessions}, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"[Storage] Error saving history: {e}")


def create_session(
    title: str,
    video_url: str,
    summary: str,
    transcript: str,
    project_dir: str,
    config: Dict = None,
) -> Dict:
    """Create a new session object."""
    return {
        "id": str(uuid.uuid4()),
        "timestamp": datetime.now().isoformat(),
        "title": title,
        "video_url": video_url,
        "summary": summary,
        "transcript": transcript,
        "project_dir": project_dir,
        "config_snapshot": config or {},
    }


def add_session(session: Dict):
    """Add a session to history (prepend) and save."""
    sessions = load_history()
    # Prepend to keep newest first
    sessions.insert(0, session)
    save_history(sessions)


def get_session(session_id: str) -> Optional[Dict]:
    """Get specific session by ID."""
    sessions = load_history()
    for s in sessions:
        if s["id"] == session_id:
            return s
    return None


def delete_session(session_id: str, delete_files: bool = True):
    """Delete a session by ID and optionally remove local project files."""
    sessions = load_history()

    # Find and delete local files if requested
    if delete_files:
        for s in sessions:
            if s["id"] == session_id:
                project_dir = s.get("project_dir")
                if project_dir and os.path.exists(project_dir):
                    try:
                        shutil.rmtree(project_dir)
                        print(f"[Storage] Deleted project folder: {project_dir}")
                    except Exception as e:
                        print(f"[Storage] Error deleting folder {project_dir}: {e}")
                break

    # Remove from history
    sessions = [s for s in sessions if s["id"] != session_id]
    save_history(sessions)


def clear_all_history(delete_files: bool = True):
    """Clear all history and optionally delete all project files."""
    if delete_files:
        sessions = load_history()
        for s in sessions:
            project_dir = s.get("project_dir")
            if project_dir and os.path.exists(project_dir):
                try:
                    shutil.rmtree(project_dir)
                    print(f"[Storage] Deleted project folder: {project_dir}")
                except Exception as e:
                    print(f"[Storage] Error deleting folder {project_dir}: {e}")

    if os.path.exists(HISTORY_FILE):
        os.remove(HISTORY_FILE)


def rename_session(session_id: str, new_title: str) -> bool:
    """Rename a session and its local folder."""
    from core.utils import sanitize_filename

    sessions = load_history()

    for s in sessions:
        if s["id"] == session_id:
            old_title = s["title"]
            old_dir = s.get("project_dir")

            # Update title in session
            s["title"] = new_title

            # Rename local folder if it exists
            if old_dir and os.path.exists(old_dir):
                parent_dir = os.path.dirname(old_dir)
                new_dir_name = sanitize_filename(new_title)
                new_dir = os.path.join(parent_dir, new_dir_name)

                # Avoid overwriting
                if os.path.exists(new_dir) and new_dir != old_dir:
                    import uuid

                    new_dir = os.path.join(
                        parent_dir, f"{new_dir_name}_{str(uuid.uuid4())[:8]}"
                    )

                try:
                    os.rename(old_dir, new_dir)
                    s["project_dir"] = new_dir

                    # Update paths in summary/report if they reference old path
                    if s.get("summary"):
                        old_basename = os.path.basename(old_dir)
                        new_basename = os.path.basename(new_dir)
                        s["summary"] = s["summary"].replace(
                            f"/generate/{old_basename}", f"/generate/{new_basename}"
                        )

                    print(f"[Storage] Renamed folder: {old_dir} -> {new_dir}")
                except Exception as e:
                    print(f"[Storage] Error renaming folder: {e}")
                    return False

            save_history(sessions)
            print(f"[Storage] Renamed session: {old_title} -> {new_title}")
            return True

    return False


def sync_history():
    """Remove history entries whose project folders no longer exist."""
    sessions = load_history()
    original_count = len(sessions)

    # Filter out sessions with missing project directories
    valid_sessions = []
    for s in sessions:
        project_dir = s.get("project_dir")
        if project_dir and os.path.exists(project_dir):
            valid_sessions.append(s)
        else:
            print(
                f"[Storage] Removing orphan entry: {s.get('title', 'Unknown')} (folder missing)"
            )

    removed_count = original_count - len(valid_sessions)
    if removed_count > 0:
        save_history(valid_sessions)
        print(f"[Storage] Synced history: removed {removed_count} orphan entries")

    return removed_count


# ===== Chat History Functions =====


def get_chat_file_path(project_dir: str) -> str:
    """Get the chat history file path for a project."""
    chat_dir = os.path.join(project_dir, "chat")
    os.makedirs(chat_dir, exist_ok=True)
    return os.path.join(chat_dir, "history.json")


def load_chat_history(project_dir: str) -> List[Dict]:
    """Load chat history for a specific session."""
    chat_file = get_chat_file_path(project_dir)
    if os.path.exists(chat_file):
        try:
            with open(chat_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            print(f"[Storage] Error loading chat history: {e}")
    return []


def save_chat_history(project_dir: str, messages: List[Dict]):
    """Save chat history for a specific session."""
    chat_file = get_chat_file_path(project_dir)
    try:
        with open(chat_file, "w", encoding="utf-8") as f:
            json.dump(messages, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"[Storage] Error saving chat history: {e}")
