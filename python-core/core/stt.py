"""Unified speech-to-text interface.

Backends (see TranscriberFactory):
  - qwen3-asr-flash-filetrans  (default; DashScope async, up to 12h)
  - qwen3-asr-flash            (DashScope sync, short clips <= 5 min)
  - qwen3-asr-local            (optional local Qwen3-ASR-1.7B; stub unless installed)
  - faster-whisper             (legacy local; also accepts mlx/cpu/cuda aliases)
"""

from __future__ import annotations

import json
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional


DEFAULT_STT_BACKEND = "qwen3-asr-flash-filetrans"

BACKEND_IDS = (
    "qwen3-asr-flash-filetrans",
    "qwen3-asr-flash",
    "qwen3-asr-local",
    "faster-whisper",
)

# DashScope HTTP roots. Classic domains remain valid per Alibaba docs.
DASHSCOPE_ENDPOINTS = {
    "beijing": "https://dashscope.aliyuncs.com/api/v1",
    "cn": "https://dashscope.aliyuncs.com/api/v1",
    "intl": "https://dashscope-intl.aliyuncs.com/api/v1",
    "international": "https://dashscope-intl.aliyuncs.com/api/v1",
    "singapore": "https://dashscope-intl.aliyuncs.com/api/v1",
}

DEFAULT_DASHSCOPE_REGION = "beijing"

_ENV_LOADED = False


class DashScopeAuthError(RuntimeError):
    """Raised when DASHSCOPE_API_KEY is missing."""


def missing_key_message() -> str:
    return (
        "未配置 DASHSCOPE_API_KEY，无法调用 Qwen3-ASR。请任选一种方式设置（不要把 Key 写进代码或提交到 git）：\n"
        "  1) 环境变量: export DASHSCOPE_API_KEY=sk-...\n"
        "  2) python-core/.env 文件: DASHSCOPE_API_KEY=sk-...\n"
        "  3) 应用设置 → API → DashScope API Key（写入已 gitignore 的 user_config.json）\n"
        "申请 Key: https://help.aliyun.com/zh/model-studio/get-api-key\n"
        "默认使用北京站 https://dashscope.aliyuncs.com ；国际站请设置 DASHSCOPE_REGION=intl "
        "或配置项 dashscope_region=intl。"
    )


@dataclass
class TranscriptionResult:
    """Normalized STT output used by the rest of the pipeline."""

    text: str
    segments: List[Dict[str, Any]] = field(default_factory=list)
    language: Optional[str] = None
    backend: str = ""
    raw: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "segments": self.segments,
            "language": self.language,
            "backend": self.backend,
        }


class BaseTranscriber(ABC):
    backend_id: str = ""

    @abstractmethod
    def transcribe(
        self, audio_path: str, progress_callback: Optional[Callable[[str], None]] = None
    ) -> TranscriptionResult:
        """Transcribe a local audio/video file."""


def _parse_dotenv(path: Path) -> None:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("'").strip('"')
        if key and key not in os.environ:
            os.environ[key] = value


def load_stt_env() -> None:
    """Load python-core/.env and cwd/.env without overriding existing env vars."""
    global _ENV_LOADED
    if _ENV_LOADED:
        return
    here = Path(__file__).resolve().parent.parent
    for candidate in (here / ".env", Path.cwd() / ".env"):
        if candidate.is_file():
            _parse_dotenv(candidate)
    _ENV_LOADED = True


def _read_user_config() -> Dict[str, Any]:
    cfg_path = Path(__file__).resolve().parent.parent / "user_config.json"
    if not cfg_path.is_file():
        return {}
    try:
        with cfg_path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
        return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def resolve_dashscope_api_key(config: Optional[Dict[str, Any]] = None) -> str:
    load_stt_env()
    key = (os.environ.get("DASHSCOPE_API_KEY") or "").strip()
    if key:
        return key
    merged = dict(_read_user_config())
    if config:
        merged.update({k: v for k, v in config.items() if v not in (None, "")})
    key = (merged.get("dashscope_api_key") or "").strip()
    if key:
        return key
    raise DashScopeAuthError(missing_key_message())


def resolve_dashscope_region(config: Optional[Dict[str, Any]] = None) -> str:
    load_stt_env()
    region = (
        os.environ.get("DASHSCOPE_REGION")
        or (config or {}).get("dashscope_region")
        or _read_user_config().get("dashscope_region")
        or DEFAULT_DASHSCOPE_REGION
    )
    return str(region).strip().lower() or DEFAULT_DASHSCOPE_REGION


def resolve_dashscope_base_url(config: Optional[Dict[str, Any]] = None) -> str:
    load_stt_env()
    explicit = (
        os.environ.get("DASHSCOPE_BASE_URL")
        or (config or {}).get("dashscope_base_url")
        or ""
    ).strip()
    if explicit:
        return explicit.rstrip("/")
    region = resolve_dashscope_region(config)
    return DASHSCOPE_ENDPOINTS.get(region, DASHSCOPE_ENDPOINTS[DEFAULT_DASHSCOPE_REGION])


def resolve_stt_backend(config: Optional[Dict[str, Any]] = None) -> str:
    load_stt_env()
    backend = (
        os.environ.get("OPENAUTONOTE_STT_BACKEND")
        or (config or {}).get("stt_backend")
        or _read_user_config().get("stt_backend")
        or DEFAULT_STT_BACKEND
    )
    backend = str(backend).strip()
    aliases = {
        "qwen3-asr": DEFAULT_STT_BACKEND,
        "qwen3": DEFAULT_STT_BACKEND,
        "filetrans": DEFAULT_STT_BACKEND,
        "whisper": "faster-whisper",
        "mlx": "faster-whisper",
        "cpu": "faster-whisper",
        "cuda": "faster-whisper",
    }
    return aliases.get(backend, backend) or DEFAULT_STT_BACKEND


def segments_from_sentences(
    sentences: List[Dict[str, Any]], time_unit: str = "ms"
) -> List[Dict[str, Any]]:
    """Convert DashScope sentence objects into pipeline segments (seconds)."""
    segments: List[Dict[str, Any]] = []
    divisor = 1000.0 if time_unit == "ms" else 1.0
    for item in sentences:
        text = (item.get("text") or "").strip()
        if not text:
            continue
        start_raw = item.get("begin_time", item.get("start", 0)) or 0
        end_raw = item.get("end_time", item.get("end", start_raw)) or start_raw
        try:
            start = float(start_raw) / divisor
            end = float(end_raw) / divisor
        except (TypeError, ValueError):
            start, end = 0.0, 0.0
        segments.append({"start": start, "end": end, "text": text})
    return segments


def parse_filetrans_transcript_json(
    data: Dict[str, Any], backend_id: str = DEFAULT_STT_BACKEND
) -> TranscriptionResult:
    """Parse the JSON downloaded from transcription_url."""
    transcripts = data.get("transcripts") or []
    if not transcripts and isinstance(data.get("output"), dict):
        transcripts = data["output"].get("transcripts") or []

    sentences: List[Dict[str, Any]] = []
    full_parts: List[str] = []
    language = None
    for transcript in transcripts:
        if not isinstance(transcript, dict):
            continue
        t_text = (transcript.get("text") or "").strip()
        if t_text:
            full_parts.append(t_text)
        for sentence in transcript.get("sentences") or []:
            if isinstance(sentence, dict):
                sentences.append(sentence)
                language = language or sentence.get("language")

    segments = segments_from_sentences(sentences, time_unit="ms")
    text = " ".join(full_parts).strip()
    if not text:
        text = " ".join(s["text"] for s in segments)
    if not segments and text:
        segments = [{"start": 0.0, "end": 0.0, "text": text}]
    return TranscriptionResult(
        text=text, segments=segments, language=language, backend=backend_id, raw=data
    )


class TranscriberFactory:
    @staticmethod
    def get_transcriber(
        backend: str = DEFAULT_STT_BACKEND,
        model_name: str = "small",
        hardware_mode: str = "cpu",
        config: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> BaseTranscriber:
        """Return a transcriber for the given backend id.

        ``model_name`` / ``hardware_mode`` apply to the legacy faster-whisper/MLX path.
        Unknown cloud aliases fall through to the default filetrans backend only when
        they look like qwen3; anything else raises.
        """
        requested = (backend or "").strip() or DEFAULT_STT_BACKEND
        aliases = {
            "qwen3-asr": DEFAULT_STT_BACKEND,
            "qwen3": DEFAULT_STT_BACKEND,
            "filetrans": DEFAULT_STT_BACKEND,
            "whisper": "faster-whisper",
        }
        resolved = aliases.get(requested, requested)

        if resolved == "qwen3-asr-flash-filetrans":
            from core.stt_dashscope import DashScopeFiletransTranscriber

            return DashScopeFiletransTranscriber(config=config)
        if resolved == "qwen3-asr-flash":
            from core.stt_dashscope import DashScopeFlashTranscriber

            return DashScopeFlashTranscriber(config=config)
        if resolved == "qwen3-asr-local":
            from core.stt_local import Qwen3AsrLocalTranscriber

            return Qwen3AsrLocalTranscriber()
        if resolved == "faster-whisper" or requested in ("mlx", "cpu", "cuda", "faster-whisper"):
            from core.stt_local import build_whisper_transcriber

            mode = requested if requested in ("mlx", "cpu", "cuda") else hardware_mode
            asr_model = kwargs.get("asr_model") or model_name
            return build_whisper_transcriber(mode, asr_model)

        raise ValueError(
            f"未知 STT 后端: {backend!r}。可选: {', '.join(BACKEND_IDS)}"
        )
