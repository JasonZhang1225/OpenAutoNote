"""Local STT backends: faster-whisper / MLX (legacy) and optional Qwen3-ASR-1.7B."""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

from core.stt import BaseTranscriber, TranscriptionResult


LOCAL_QWEN3_HINT = (
    "本地后端 qwen3-asr-local 未启用。Qwen3-ASR-1.7B 需要单独安装较大的 "
    "PyTorch / transformers 权重，默认依赖不会拉取以免把安装体积打爆。\n"
    "可选安装（自行承担依赖体积）：\n"
    "  pip install transformers accelerate torch torchaudio\n"
    "  模型: https://huggingface.co/Qwen/Qwen3-ASR-1.7B\n"
    "  上游: https://github.com/QwenLM/Qwen3-ASR\n"
    "装好后设置 OPENAUTONOTE_STT_BACKEND=qwen3-asr-local，并保证能 "
    "`import transformers`。日常长视频请继续使用默认的 qwen3-asr-flash-filetrans。"
)


def _segments_to_result(
    segments: List[Dict[str, Any]], backend: str, language: Optional[str] = None
) -> TranscriptionResult:
    text = " ".join(s.get("text", "") for s in segments).strip()
    return TranscriptionResult(
        text=text, segments=segments, language=language, backend=backend
    )


class MlxTranscriber(BaseTranscriber):
    backend_id = "faster-whisper"

    def __init__(self, model_name: str):
        self.model_name = model_name

    def transcribe(
        self, audio_path: str, progress_callback: Optional[Callable[[str], None]] = None
    ) -> TranscriptionResult:
        try:
            import mlx_whisper
        except ImportError as exc:
            raise ImportError(
                "mlx-whisper 未安装。请 pip install mlx-whisper，或改用 Qwen3-ASR 云端后端。"
            ) from exc

        result = mlx_whisper.transcribe(
            audio_path,
            initial_prompt="以下是简体中文句子。",
            path_or_hf_repo=self.model_name,
        )

        segments: List[Dict[str, Any]] = []
        full_transcript = ""
        for seg in result.get("segments", []):
            segment_text = str(seg.get("text", "")).strip()
            segments.append(
                {"start": seg["start"], "end": seg["end"], "text": segment_text}
            )
            full_transcript += f"[{seg['start']:.2f}-{seg['end']:.2f}] {segment_text}\n"
            if progress_callback:
                progress_callback(full_transcript.strip())
        language = result.get("language")
        return _segments_to_result(segments, "faster-whisper", language)


class FasterTranscriber(BaseTranscriber):
    backend_id = "faster-whisper"

    def __init__(self, model_name: str, device="cpu", compute_type="int8"):
        self.device = device
        self.compute_type = compute_type
        self.model_name = model_name

    def transcribe(
        self, audio_path: str, progress_callback: Optional[Callable[[str], None]] = None
    ) -> TranscriptionResult:
        try:
            from faster_whisper import WhisperModel
        except ImportError as exc:
            raise ImportError(
                "faster-whisper 未安装。请 pip install faster-whisper，或改用默认 Qwen3-ASR 云端后端。"
            ) from exc

        from core import model_manager

        print(f"[Transcriber] Loading Model (Device: {self.device})...")
        model_path = model_manager.resolve_model_path(
            self.model_name, "cuda" if self.device == "cuda" else "cpu"
        )
        model = WhisperModel(
            model_path, device=self.device, compute_type=self.compute_type
        )
        print("[Transcriber] Model Loaded Successfully!")

        segments_generator, info = model.transcribe(
            audio_path, beam_size=5, initial_prompt="以下是简体中文句子。"
        )

        segments: List[Dict[str, Any]] = []
        full_transcript = ""
        for segment in segments_generator:
            segment_text = segment.text.strip()
            segments.append(
                {
                    "start": segment.start,
                    "end": segment.end,
                    "text": segment_text,
                }
            )
            full_transcript += f"[{segment.start:.2f}-{segment.end:.2f}] {segment_text}\n"
            if progress_callback:
                progress_callback(full_transcript.strip())
        language = getattr(info, "language", None)
        return _segments_to_result(segments, "faster-whisper", language)


class Qwen3AsrLocalTranscriber(BaseTranscriber):
    """Optional local Qwen3-ASR-1.7B. Stub unless transformers + weights are present."""

    backend_id = "qwen3-asr-local"
    model_id = "Qwen/Qwen3-ASR-1.7B"

    def transcribe(
        self, audio_path: str, progress_callback: Optional[Callable[[str], None]] = None
    ) -> TranscriptionResult:
        try:
            import transformers  # noqa: F401
        except ImportError as exc:
            raise RuntimeError(LOCAL_QWEN3_HINT) from exc

        # The upstream API is still evolving (https://github.com/QwenLM/Qwen3-ASR).
        # If a future `qwen_asr` package is installed, use it; otherwise fail loudly
        # rather than guessing a transformers class that may not exist.
        try:
            import qwen_asr  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                LOCAL_QWEN3_HINT
                + "\n已检测到 transformers，但未找到 qwen_asr 包，拒绝猜测未文档化的调用方式。"
            ) from exc

        transcribe_fn = getattr(qwen_asr, "transcribe", None)
        if transcribe_fn is None:
            raise RuntimeError(LOCAL_QWEN3_HINT)

        if progress_callback:
            progress_callback("Running local Qwen3-ASR...")
        raw = transcribe_fn(audio_path)
        if isinstance(raw, str):
            text = raw
            segments = [{"start": 0.0, "end": 0.0, "text": text}]
            language = None
        elif isinstance(raw, dict):
            text = (raw.get("text") or "").strip()
            segments = raw.get("segments") or [{"start": 0.0, "end": 0.0, "text": text}]
            language = raw.get("language")
        else:
            text = str(raw)
            segments = [{"start": 0.0, "end": 0.0, "text": text}]
            language = None
        return TranscriptionResult(
            text=text, segments=segments, language=language, backend=self.backend_id
        )


def build_whisper_transcriber(hardware_mode: str, model_name: str) -> BaseTranscriber:
    from core import model_manager

    if hardware_mode == "mlx":
        resolved_model = model_manager.resolve_model_path(model_name, "mlx")
        return MlxTranscriber(model_name=resolved_model)
    if hardware_mode == "cuda":
        resolved_model = model_manager.resolve_model_path(model_name, "cuda")
        return FasterTranscriber(
            model_name=resolved_model, device="cuda", compute_type="float16"
        )
    resolved_model = model_manager.resolve_model_path(model_name, "cpu")
    return FasterTranscriber(model_name=resolved_model, device="cpu")
