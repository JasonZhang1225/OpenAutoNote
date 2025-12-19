import json
import os
import uuid
from datetime import datetime
from typing import List, Dict, Optional

HISTORY_FILE = "user_history.json"

def load_history() -> List[Dict]:
    """Load all sessions from history file."""
    if not os.path.exists(HISTORY_FILE):
        return []
    try:
        with open(HISTORY_FILE, 'r', encoding='utf-8') as f:
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
        with open(HISTORY_FILE, 'w', encoding='utf-8') as f:
            json.dump({"sessions": sessions}, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"[Storage] Error saving history: {e}")

def create_session(title: str, video_url: str, summary: str, transcript: str, project_dir: str, config: Dict = None) -> Dict:
    """Create a new session object."""
    return {
        "id": str(uuid.uuid4()),
        "timestamp": datetime.now().isoformat(),
        "title": title,
        "video_url": video_url,
        "summary": summary,
        "transcript": transcript,
        "project_dir": project_dir,
        "config_snapshot": config or {}
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

def delete_session(session_id: str):
    """Delete a session by ID."""
    sessions = load_history()
    sessions = [s for s in sessions if s["id"] != session_id]
    save_history(sessions)

def clear_all_history():
    """Clear all history."""
    if os.path.exists(HISTORY_FILE):
        os.remove(HISTORY_FILE)
