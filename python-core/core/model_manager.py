import os
import platform
import shutil
from typing import Callable, Dict, List, Optional

from huggingface_hub import scan_cache_dir, snapshot_download

# Supported models per backend family.
# MLX targets Apple; CTranslate2 targets CPU/CUDA (Windows/Linux).
SUPPORTED_MODELS: Dict[str, Dict[str, Dict[str, str]]] = {
    "ctranslate2": {
        "tiny": {"repo_id": "Systran/faster-whisper-tiny", "display": "Tiny"},
        "base": {"repo_id": "Systran/faster-whisper-base", "display": "Base"},
        "small": {"repo_id": "Systran/faster-whisper-small", "display": "Small"},
        "medium": {
            "repo_id": "Systran/faster-whisper-medium",
            "display": "Medium",
        },
        "large-v3": {
            "repo_id": "Systran/faster-whisper-large-v3",
            "display": "Large-v3",
        },
    },
    "mlx": {
        "tiny": {
            "repo_id": "mlx-community/whisper-tiny-mlx",
            "display": "Tiny (MLX)",
        },
        "base": {
            "repo_id": "mlx-community/whisper-base-mlx",
            "display": "Base (MLX)",
        },
        "small": {
            "repo_id": "mlx-community/whisper-small-mlx",
            "display": "Small (MLX)",
        },
        "medium": {
            "repo_id": "mlx-community/whisper-medium-mlx",
            "display": "Medium (MLX)",
        },
        "large-v3": {
            "repo_id": "mlx-community/whisper-large-v3-mlx",
            "display": "Large-v3 (MLX)",
        },
    },
}


def _backend_family(hardware_mode: str) -> str:
    if hardware_mode == "mlx":
        return "mlx"
    return "ctranslate2"


def _scan_repo_cache():
    try:
        return scan_cache_dir()
    except Exception:
        return None


def _repo_cache_path(repo_id: str) -> Optional[str]:
    """
    Check if a model exists in the Hugging Face cache.
    Uses scan_cache_dir first, then falls back to direct filesystem check.
    Supports multiple naming conventions for MLX models.
    """
    from pathlib import Path

    # Method 1: Try huggingface_hub's scan_cache_dir
    cache = _scan_repo_cache()
    if cache:
        for repo in cache.repos:
            if getattr(repo, "repo_id", None) == repo_id:
                path = getattr(repo, "repo_path", None) or getattr(
                    repo, "cache_dir", None
                )
                if path:
                    return str(path)

    # Method 2: Direct filesystem check (fallback)
    cache_dir = Path(os.path.expanduser("~/.cache/huggingface/hub"))

    # Build list of candidate folder names to check
    # HF uses format: models--{org}--{model}
    base_folder = "models--" + repo_id.replace("/", "--")

    # For MLX models, check both with and without -mlx suffix
    # e.g., whisper-tiny-mlx vs whisper-tiny
    candidates = [base_folder]

    if base_folder.endswith("-mlx"):
        # Also check without -mlx suffix
        candidates.append(base_folder[:-4])  # Remove "-mlx"
    elif "whisper-" in base_folder and not base_folder.endswith("-mlx"):
        # Also check with -mlx suffix
        candidates.append(base_folder + "-mlx")

    # Check each candidate
    for folder_name in candidates:
        model_path = cache_dir / folder_name
        if model_path.exists():
            # Verify it has actual model files (not just an empty folder)
            snapshots_dir = model_path / "snapshots"
            if snapshots_dir.exists() and any(snapshots_dir.iterdir()):
                print(f"[ModelCheck] ✅ Found: {folder_name}")
                return str(model_path)

    print(f"[ModelCheck] ❌ Not found: {candidates}")
    return None


def get_supported_models(hardware_mode: str) -> Dict[str, Dict[str, str]]:
    family = _backend_family(hardware_mode)
    return SUPPORTED_MODELS.get(family, {})


def get_model_statuses(hardware_mode: str) -> List[Dict[str, Optional[str]]]:
    models = get_supported_models(hardware_mode)
    statuses: List[Dict[str, Optional[str]]] = []
    for key, meta in models.items():
        repo_id = meta.get("repo_id", "")
        cache_path = _repo_cache_path(repo_id)
        statuses.append(
            {
                "key": key,
                "display": meta.get("display", key),
                "repo_id": repo_id,
                "installed": cache_path is not None,
                "path": cache_path,
            }
        )
    return statuses


def resolve_model_path(model_key: str, hardware_mode: str) -> str:
    models = get_supported_models(hardware_mode)
    meta = models.get(model_key)
    if not meta:
        # Fallback to "small" if unknown
        meta = models.get("small") or next(iter(models.values()), None)
    repo_id = meta.get("repo_id", "") if meta else ""

    # For MLX backend, always return repo_id since mlx_whisper handles cache internally
    if hardware_mode == "mlx":
        return repo_id

    # For ctranslate2/faster-whisper, we can use cached path or repo_id
    cache_path = _repo_cache_path(repo_id) if repo_id else None
    return cache_path or repo_id


def download_model(
    model_key: str,
    hardware_mode: str,
    mirror: Optional[str] = None,
    progress_cb: Optional[Callable[[float], None]] = None,
) -> str:
    models = get_supported_models(hardware_mode)
    if model_key not in models:
        raise ValueError(f"Unknown model: {model_key}")

    repo_id = models[model_key]["repo_id"]

    # Optional mirror support
    env = os.environ.copy()
    if mirror:
        env["HF_ENDPOINT"] = mirror.rstrip("/")

    if progress_cb:
        try:
            progress_cb(0.0)
        except Exception:
            pass

    path = snapshot_download(
        repo_id=repo_id, resume_download=True, local_files_only=False
    )

    if progress_cb:
        try:
            progress_cb(1.0)
        except Exception:
            pass

    return path


def delete_model(model_key: str, hardware_mode: str) -> bool:
    models = get_supported_models(hardware_mode)
    if model_key not in models:
        return False
    repo_id = models[model_key]["repo_id"]
    cache_path = _repo_cache_path(repo_id)
    if cache_path and os.path.exists(cache_path):
        shutil.rmtree(cache_path, ignore_errors=True)
        return True
    return False
