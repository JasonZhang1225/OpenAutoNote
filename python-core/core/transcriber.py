import abc
from typing import List, Dict, Any

from core import model_manager


class BaseTranscriber(abc.ABC):
    @abc.abstractmethod
    def transcribe(self, audio_path: str) -> List[Dict[str, Any]]:
        pass


class MlxTranscriber(BaseTranscriber):
    def __init__(self, model_name: str):
        self.model_name = model_name

    def transcribe(self, audio_path: str) -> List[Dict[str, Any]]:
        try:
            from nicegui import ui

            import mlx_whisper
        except ImportError:
            raise ImportError(
                "mlx-whisper not installed. Please install it to use MLX backend."
            )

        # mlx_whisper.transcribe returns a dict with 'text' and 'segments'
        # Adding initial_prompt to guide it towards Simplified Chinese
        try:
            result = mlx_whisper.transcribe(
                audio_path,
                initial_prompt="以下是简体中文句子。",
                model=self.model_name,
            )
            ui.notify("Model Loaded & Transcription Started!", type="positive")
        except Exception as e:
            ui.notify(f"Model Error: {str(e)}", type="negative")
            raise e

        segments = []
        for seg in result.get("segments", []):
            segments.append(
                {"start": seg["start"], "end": seg["end"], "text": seg["text"].strip()}
            )
        return segments


class FasterTranscriber(BaseTranscriber):
    def __init__(self, model_name: str, device="cpu", compute_type="int8"):
        self.device = device
        self.compute_type = compute_type
        self.model_name = model_name

    def transcribe(self, audio_path: str) -> List[Dict[str, Any]]:
        try:
            from nicegui import ui
            from faster_whisper import WhisperModel
        except ImportError:
            raise ImportError("faster-whisper not installed.")

        ui.notify(
            f"Loading Model (Device: {self.device})... this may take a while if downloading.",
            type="info",
            timeout=None,
        )

        try:
            model_path = model_manager.resolve_model_path(
                self.model_name, "cuda" if self.device == "cuda" else "cpu"
            )
            model = WhisperModel(
                model_path, device=self.device, compute_type=self.compute_type
            )
            ui.notify("Model Loaded Successfully!", type="positive")
        except Exception as e:
            ui.notify(f"Model Load Failed: {str(e)}", type="negative")
            raise e

        segments_generator, info = model.transcribe(
            audio_path, beam_size=5, initial_prompt="以下是简体中文句子。"
        )

        segments = []
        for segment in segments_generator:
            segments.append(
                {
                    "start": segment.start,
                    "end": segment.end,
                    "text": segment.text.strip(),
                }
            )
        return segments


class TranscriberFactory:
    @staticmethod
    def get_transcriber(backend: str, model_name: str) -> BaseTranscriber:
        if backend == "mlx":
            resolved_model = model_manager.resolve_model_path(model_name, "mlx")
            return MlxTranscriber(model_name=resolved_model)
        elif backend == "cpu":
            resolved_model = model_manager.resolve_model_path(model_name, "cpu")
            return FasterTranscriber(model_name=resolved_model, device="cpu")
        elif backend == "cuda":
            resolved_model = model_manager.resolve_model_path(model_name, "cuda")
            return FasterTranscriber(
                model_name=resolved_model, device="cuda", compute_type="float16"
            )
        else:
            raise ValueError(f"Unknown backend: {backend}")
