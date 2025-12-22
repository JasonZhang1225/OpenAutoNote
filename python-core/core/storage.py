import json
import os
import shutil
import uuid
from datetime import datetime
from typing import List, Dict, Optional

# Add TimeoutError import
from builtins import TimeoutError

# Get the directory where this module is located (python-core/core/)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HISTORY_FILE = os.path.join(BASE_DIR, "user_history.json")

# Cross-platform file locking
try:
    import fcntl  # Unix/Linux/Mac
    HAS_FCNTL = True
except ImportError:
    HAS_FCNTL = False

try:
    import msvcrt  # Windows
    HAS_MSVCRT = True
except ImportError:
    HAS_MSVCRT = False


class FileLock:
    """Cross-platform file lock implementation"""
    
    def __init__(self, file_path: str, timeout: float = 10.0):
        self.file_path = file_path
        self.timeout = timeout
        self.lock_file_path = file_path + ".lock"
        self.lock_file = None
    
    def __enter__(self):
        self.acquire()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()
    
    def acquire(self):
        """Acquire file lock with timeout"""
        import time
        
        start_time = time.time()
        while True:
            try:
                # Try to create and open lock file exclusively
                self.lock_file = open(self.lock_file_path, 'w')
                if HAS_FCNTL:
                    # Unix/Linux/Mac - use flock
                    fcntl.flock(self.lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                elif HAS_MSVCRT:
                    # Windows - use locking
                    # Write a byte to the file and lock it
                    self.lock_file.write('L')
                    self.lock_file.flush()
                    msvcrt.locking(self.lock_file.fileno(), msvcrt.LK_NBLCK, 1)
                else:
                    # Fallback: Just create lock file and hope for the best
                    # This is not atomic but better than nothing
                    pass
                break
            except (IOError, OSError):
                if self.lock_file:
                    self.lock_file.close()
                    self.lock_file = None
                
                if time.time() - start_time > self.timeout:
                    raise TimeoutError(f"Could not acquire lock for {self.file_path} within {self.timeout} seconds")
                
                # Wait a bit before retrying
                time.sleep(0.1)
    
    def release(self):
        """Release file lock"""
        if self.lock_file:
            try:
                if HAS_FCNTL:
                    # Unix/Linux/Mac - unlock
                    fcntl.flock(self.lock_file.fileno(), fcntl.LOCK_UN)
                elif HAS_MSVCRT:
                    # Windows - unlock
                    msvcrt.locking(self.lock_file.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    # Fallback: nothing to do
                    pass
            except (IOError, OSError):
                pass
            finally:
                self.lock_file.close()
                self.lock_file = None
                # Try to remove lock file
                try:
                    os.remove(self.lock_file_path)
                except (IOError, OSError):
                    pass


def _backup_corrupt_history_file(error: Exception) -> Optional[str]:
    """Backup the corrupt history file for manual inspection."""
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = f"{HISTORY_FILE}.corrupt-{timestamp}"
        shutil.copy2(HISTORY_FILE, backup_path)
        print(f"[Storage] Backed up corrupt history to {backup_path}: {error}")
        return backup_path
    except Exception as backup_err:
        print(f"[Storage] Failed to back up corrupt history: {backup_err}")
        return None


def load_history() -> List[Dict]:
    """Load all sessions from history file."""
    if not os.path.exists(HISTORY_FILE):
        return []
    try:
        with FileLock(HISTORY_FILE):
            with open(HISTORY_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                # Ensure it's a list
                if isinstance(data, dict) and "sessions" in data:
                    return data["sessions"]
                elif isinstance(data, list):
                    return data
                return []
    except json.JSONDecodeError as e:
        # Keep a copy of the broken file and reset to a clean state
        _backup_corrupt_history_file(e)
        try:
            save_history([])
        except Exception as reset_err:
            print(f"[Storage] Error resetting history after corruption: {reset_err}")
        print(f"[Storage] Error loading history: {e}")
        return []
    except Exception as e:
        print(f"[Storage] Error loading history: {e}")
        return []


def save_history(sessions: List[Dict]):
    """Save sessions to history file."""
    try:
        with FileLock(HISTORY_FILE):
            tmp_path = f"{HISTORY_FILE}.tmp"
            try:
                with open(tmp_path, "w", encoding="utf-8") as f:
                    json.dump({"sessions": sessions}, f, indent=2, ensure_ascii=False)
                # Atomic replace to avoid truncated files on crashes
                os.replace(tmp_path, HISTORY_FILE)
            finally:
                if os.path.exists(tmp_path):
                    try:
                        os.remove(tmp_path)
                    except (IOError, OSError):
                        pass
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

                # Avoid overwriting existing folders
                if os.path.exists(new_dir) and new_dir != old_dir:
                    counter = 1
                    base_dir = new_dir
                    
                    # Find the next available number suffix
                    while os.path.exists(new_dir):
                        new_dir = f"{base_dir}_{counter}"
                        counter += 1

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
            with FileLock(chat_file):
                with open(chat_file, "r", encoding="utf-8") as f:
                    return json.load(f)
        except Exception as e:
            print(f"[Storage] Error loading chat history: {e}")
    return []


def save_chat_history(project_dir: str, messages: List[Dict]):
    """Save chat history for a specific session."""
    chat_file = get_chat_file_path(project_dir)
    try:
        with FileLock(chat_file):
            with open(chat_file, "w", encoding="utf-8") as f:
                json.dump(messages, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"[Storage] Error saving chat history: {e}")


def validate_and_cleanup_sessions():
    """Validate session IDs and cleanup temporary sessions that failed during processing."""
    import re
    from datetime import datetime, timedelta
    
    # Pattern for valid UUID v4
    uuid_pattern = re.compile(r'^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$', re.I)
    
    # Pattern for temporary session IDs (timestamp_uuid)
    temp_session_pattern = re.compile(r'^\d{8}_\d{6}_[0-9a-f]{8}$')
    
    sessions = load_history()
    valid_sessions = []
    cleaned_count = 0
    
    for session in sessions:
        session_id = session.get("id", "")
        
        # Check if it's a valid UUID
        if uuid_pattern.match(session_id):
            valid_sessions.append(session)
        # Check if it's a temporary session ID
        elif temp_session_pattern.match(session_id):
            # Check if the session is too old (more than 1 day old)
            try:
                # Extract timestamp from session ID
                timestamp_part = session_id.split('_')[0] + session_id.split('_')[1]
                session_time = datetime.strptime(timestamp_part, "%Y%m%d%H%M%S")
                current_time = datetime.now()
                
                # If session is older than 1 day, consider it stale and remove it
                if current_time - session_time > timedelta(days=1):
                    # Delete associated files if they exist
                    project_dir = session.get("project_dir")
                    if project_dir and os.path.exists(project_dir):
                        try:
                            shutil.rmtree(project_dir)
                            print(f"[Storage] Deleted stale project folder: {project_dir}")
                        except Exception as e:
                            print(f"[Storage] Error deleting stale folder {project_dir}: {e}")
                    cleaned_count += 1
                else:
                    # Keep recent temporary sessions
                    valid_sessions.append(session)
            except ValueError:
                # If we can't parse the timestamp, treat it as invalid
                cleaned_count += 1
        else:
            # Invalid session ID format, remove it
            cleaned_count += 1
    
    # Save cleaned sessions if we removed any
    if cleaned_count > 0:
        save_history(valid_sessions)
        print(f"[Storage] Cleaned up {cleaned_count} invalid/stale sessions")
    
    return valid_sessions


def is_valid_uuid(session_id: str) -> bool:
    """Check if a session ID is a valid UUID."""
    import re
    uuid_pattern = re.compile(r'^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$', re.I)
    return bool(uuid_pattern.match(session_id))


def convert_temp_session_to_valid(session_id: str) -> Optional[str]:
    """Convert a temporary session ID to a valid UUID if possible."""
    import re
    temp_session_pattern = re.compile(r'^\d{8}_\d{6}_[0-9a-f]{8}$')
    
    if temp_session_pattern.match(session_id):
        # Convert to valid UUID
        return str(uuid.uuid4())
    
    return None
