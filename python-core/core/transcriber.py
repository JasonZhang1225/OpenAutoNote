import os
import abc
from typing import List, Dict, Any


class BaseTranscriber(abc.ABC):
    @abc.abstractmethod
    def transcribe(self, audio_path: str) -> List[Dict[str, Any]]:
        pass


class MlxTranscriber(BaseTranscriber):
    def transcribe(self, audio_path: str) -> List[Dict[str, Any]]:
        try:
            from nicegui import ui

            ui.notify("Checking/Loading MLX Model...", type="info", timeout=3000)
            import mlx_whisper
        except ImportError:
            raise ImportError(
                "mlx-whisper not installed. Please install it to use MLX backend."
            )

        # mlx_whisper.transcribe returns a dict with 'text' and 'segments'
        # Adding initial_prompt to guide it towards Simplified Chinese
        try:
            result = mlx_whisper.transcribe(
                audio_path, initial_prompt="以下是简体中文句子。"
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
    def __init__(self, device="cpu", compute_type="int8"):
        self.device = device
        self.compute_type = compute_type

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
            model = WhisperModel(
                "small", device=self.device, compute_type=self.compute_type
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
    def get_transcriber(backend: str) -> BaseTranscriber:
        if backend == "mlx":
            return MlxTranscriber()
        elif backend == "cpu":
            return FasterTranscriber(device="cpu")
        elif backend == "cuda":
            # Placeholder for NVIDIA/CUDA support
            return FasterTranscriber(device="cuda")
        else:
            raise ValueError(f"Unknown backend: {backend}")
