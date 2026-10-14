"""Backward-compatible STT exports.

New code should import from ``core.stt``. This module keeps the historical
``TranscriberFactory.get_transcriber(hardware_mode, model_name)`` call shape
working while routing cloud backends through the unified interface.
"""

from core.stt import (  # noqa: F401
    BACKEND_IDS,
    DEFAULT_STT_BACKEND,
    BaseTranscriber,
    DashScopeAuthError,
    TranscriptionResult,
    TranscriberFactory,
)
from core.stt_local import FasterTranscriber, MlxTranscriber  # noqa: F401
