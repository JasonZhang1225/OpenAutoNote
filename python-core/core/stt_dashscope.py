"""DashScope Qwen3-ASR backends (REST, no invented SDK wrappers).

Docs:
  https://www.alibabacloud.com/help/en/model-studio/non-realtime-speech-recognition-user-guide
  https://help.aliyun.com/zh/model-studio/qwen-speech-recognition
  https://www.alibabacloud.com/help/en/model-studio/get-temporary-file-url
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Callable, Dict, Optional
from urllib.parse import urlparse

import requests

from core.stt import (
    BaseTranscriber,
    DashScopeAuthError,
    TranscriptionResult,
    missing_key_message,
    parse_filetrans_transcript_json,
    resolve_dashscope_api_key,
    resolve_dashscope_base_url,
    segments_from_sentences,
)

FILETRANS_MODEL = "qwen3-asr-flash-filetrans"
FLASH_MODEL = "qwen3-asr-flash"

# filetrans: up to 12h / 2GB. flash: 5 min / 10MB.
FLASH_MAX_SECONDS = 5 * 60
FLASH_MAX_BYTES = 10 * 1024 * 1024

POLL_INTERVAL_SEC = 3.0
DEFAULT_TIMEOUT_SEC = 3 * 60 * 60  # 3h wall clock for long videos


def _notify(progress_callback: Optional[Callable[[str], None]], message: str) -> None:
    print(f"[STT] {message}")
    if progress_callback:
        try:
            progress_callback(message)
        except Exception:
            pass


def _headers(api_key: str, async_enable: bool = False, oss_resolve: bool = False) -> Dict[str, str]:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    if async_enable:
        headers["X-DashScope-Async"] = "enable"
    if oss_resolve:
        headers["X-DashScope-OssResourceResolve"] = "enable"
    return headers


def upload_local_file_to_dashscope(
    api_key: str,
    base_url: str,
    file_path: str,
    model_name: str,
) -> str:
    """Upload a local file to DashScope temp OSS and return an oss:// URL.

    Matches the official getPolicy + form POST flow. The resulting oss:// URL
    must be used with header X-DashScope-OssResourceResolve: enable.
    """
    policy_url = f"{base_url.rstrip('/')}/uploads"
    resp = requests.get(
        policy_url,
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        params={"action": "getPolicy", "model": model_name},
        timeout=60,
    )
    if resp.status_code != 200:
        raise RuntimeError(
            f"获取 DashScope 上传凭证失败 (HTTP {resp.status_code}): {resp.text[:500]}"
        )
    body = resp.json()
    policy = body.get("data") or body
    required = ("upload_dir", "oss_access_key_id", "signature", "policy", "upload_host")
    missing = [k for k in required if not policy.get(k)]
    if missing:
        raise RuntimeError(f"上传凭证缺少字段 {missing}: {body}")

    file_name = Path(file_path).name
    key = f"{policy['upload_dir'].rstrip('/')}/{file_name}"
    with open(file_path, "rb") as fh:
        form = {
            "OSSAccessKeyId": (None, policy["oss_access_key_id"]),
            "Signature": (None, policy["signature"]),
            "policy": (None, policy["policy"]),
            "x-oss-object-acl": (None, policy.get("x_oss_object_acl", "private")),
            "x-oss-forbid-overwrite": (None, policy.get("x_oss_forbid_overwrite", "true")),
            "key": (None, key),
            "success_action_status": (None, "200"),
            "file": (file_name, fh),
        }
        up = requests.post(policy["upload_host"], files=form, timeout=600)
    if up.status_code not in (200, 204):
        raise RuntimeError(f"上传音频到 DashScope OSS 失败 (HTTP {up.status_code}): {up.text[:500]}")
    return f"oss://{key}"


def prepare_media_for_upload(media_path: str) -> str:
    """Prefer a compact audio track for upload; fall back to the original file."""
    path = Path(media_path)
    if not path.is_file():
        raise FileNotFoundError(f"找不到音视频文件: {media_path}")

    audio_exts = {".wav", ".mp3", ".m4a", ".flac", ".ogg", ".aac", ".opus", ".amr", ".wma"}
    if path.suffix.lower() in audio_exts:
        return str(path)

    import shutil
    import subprocess

    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        print("[STT] ffmpeg 不在 PATH，将直接上传原始文件（filetrans 支持常见视频容器）。")
        return str(path)

    out = path.with_suffix(".oan-audio.mp3")
    if out.is_file() and out.stat().st_mtime >= path.stat().st_mtime:
        return str(out)

    cmd = [
        ffmpeg,
        "-y",
        "-i",
        str(path),
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-b:a",
        "64k",
        str(out),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0 or not out.is_file():
        print(f"[STT] ffmpeg 抽音失败，回退原文件: {proc.stderr[-400:] if proc.stderr else 'unknown'}")
        return str(path)
    return str(out)


def _looks_like_url(value: str) -> bool:
    parsed = urlparse(value)
    return parsed.scheme in ("http", "https", "oss")


def _download_json(url: str, timeout: int = 120) -> Dict[str, Any]:
    resp = requests.get(url, timeout=timeout)
    resp.raise_for_status()
    return resp.json()


def _extract_task_id(payload: Dict[str, Any]) -> Optional[str]:
    output = payload.get("output") or {}
    if isinstance(output, dict):
        return output.get("task_id") or output.get("taskId")
    return payload.get("task_id")


def _extract_transcription_url(payload: Dict[str, Any]) -> Optional[str]:
    output = payload.get("output") or {}
    if not isinstance(output, dict):
        return None
    result = output.get("result")
    if isinstance(result, dict) and result.get("transcription_url"):
        return result.get("transcription_url")
    results = output.get("results")
    if isinstance(results, list):
        for item in results:
            if isinstance(item, dict) and item.get("transcription_url"):
                return item.get("transcription_url")
    if output.get("transcription_url"):
        return output.get("transcription_url")
    return None


class DashScopeFiletransTranscriber(BaseTranscriber):
    """qwen3-asr-flash-filetrans — async file transcription, up to 12 hours."""

    backend_id = FILETRANS_MODEL
    model_name = FILETRANS_MODEL

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}

    def transcribe(
        self, audio_path: str, progress_callback: Optional[Callable[[str], None]] = None
    ) -> TranscriptionResult:
        try:
            api_key = resolve_dashscope_api_key(self.config)
        except DashScopeAuthError:
            raise DashScopeAuthError(missing_key_message()) from None

        base_url = resolve_dashscope_base_url(self.config)
        media = audio_path
        if not _looks_like_url(audio_path):
            media = prepare_media_for_upload(audio_path)
            _notify(progress_callback, f"正在上传音频到 DashScope 临时存储: {os.path.basename(media)}")
            file_url = upload_local_file_to_dashscope(
                api_key, base_url, media, self.model_name
            )
        else:
            file_url = audio_path

        oss_resolve = file_url.startswith("oss://")
        submit_url = f"{base_url.rstrip('/')}/services/audio/asr/transcription"
        payload = {
            "model": self.model_name,
            "input": {"file_url": file_url},
            "parameters": {
                "channel_id": [0],
                "enable_itn": False,
                "enable_words": False,
            },
        }
        _notify(progress_callback, f"提交异步转写任务 ({self.model_name})...")
        submit = requests.post(
            submit_url,
            headers=_headers(api_key, async_enable=True, oss_resolve=oss_resolve),
            data=json.dumps(payload),
            timeout=60,
        )
        if submit.status_code != 200:
            raise RuntimeError(
                f"提交 Qwen3-ASR filetrans 任务失败 (HTTP {submit.status_code}): {submit.text[:800]}"
            )
        submit_body = submit.json()
        task_id = _extract_task_id(submit_body)
        if not task_id:
            raise RuntimeError(f"filetrans 未返回 task_id: {submit_body}")

        _notify(progress_callback, f"任务已提交 task_id={task_id}，等待识别完成...")
        query_url = f"{base_url.rstrip('/')}/tasks/{task_id}"
        deadline = time.time() + DEFAULT_TIMEOUT_SEC
        last_status = ""
        while time.time() < deadline:
            time.sleep(POLL_INTERVAL_SEC)
            query = requests.get(
                query_url,
                headers=_headers(api_key, async_enable=True, oss_resolve=oss_resolve),
                timeout=60,
            )
            if query.status_code != 200:
                _notify(progress_callback, f"查询任务失败 HTTP {query.status_code}，重试中...")
                continue
            body = query.json()
            output = body.get("output") or {}
            status = str(output.get("task_status") or output.get("taskStatus") or "").upper()
            if status != last_status:
                _notify(progress_callback, f"转写状态: {status or 'PENDING'}")
                last_status = status
            if status == "SUCCEEDED":
                trans_url = _extract_transcription_url(body)
                if trans_url:
                    transcript = _download_json(trans_url)
                    result = parse_filetrans_transcript_json(transcript, self.backend_id)
                else:
                    # Some responses inline the transcript.
                    result = parse_filetrans_transcript_json(body, self.backend_id)
                    if not result.text:
                        raise RuntimeError(f"filetrans 成功但未返回 transcription_url: {body}")
                _notify(progress_callback, result.text or "转写完成")
                return result
            if status in {"FAILED", "CANCELED", "UNKNOWN", "CANCELLED"}:
                message = (
                    output.get("message")
                    or body.get("message")
                    or json.dumps(body, ensure_ascii=False)[:800]
                )
                raise RuntimeError(f"Qwen3-ASR filetrans 失败 ({status}): {message}")

        raise TimeoutError(f"Qwen3-ASR filetrans 超时（>{DEFAULT_TIMEOUT_SEC}s），task_id={task_id}")


class DashScopeFlashTranscriber(BaseTranscriber):
    """qwen3-asr-flash — sync multimodal generation, short clips only."""

    backend_id = FLASH_MODEL
    model_name = FLASH_MODEL

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}

    def transcribe(
        self, audio_path: str, progress_callback: Optional[Callable[[str], None]] = None
    ) -> TranscriptionResult:
        try:
            api_key = resolve_dashscope_api_key(self.config)
        except DashScopeAuthError:
            raise DashScopeAuthError(missing_key_message()) from None

        base_url = resolve_dashscope_base_url(self.config)
        media = audio_path
        if not _looks_like_url(audio_path):
            media = prepare_media_for_upload(audio_path)
            size = os.path.getsize(media)
            if size > FLASH_MAX_BYTES:
                raise RuntimeError(
                    f"qwen3-asr-flash 限制单文件 ≤10MB / 5 分钟，当前 {size} 字节。"
                    "长视频请改用默认后端 qwen3-asr-flash-filetrans。"
                )
            _notify(progress_callback, f"正在上传短音频: {os.path.basename(media)}")
            file_url = upload_local_file_to_dashscope(
                api_key, base_url, media, self.model_name
            )
        else:
            file_url = audio_path

        oss_resolve = file_url.startswith("oss://")
        url = f"{base_url.rstrip('/')}/services/aigc/multimodal-generation/generation"
        payload = {
            "model": self.model_name,
            "input": {
                "messages": [
                    {"role": "user", "content": [{"audio": file_url}]}
                ]
            },
            "parameters": {
                "asr_options": {"enable_itn": False},
            },
        }
        _notify(progress_callback, f"调用 {self.model_name} 同步识别...")
        resp = requests.post(
            url,
            headers=_headers(api_key, oss_resolve=oss_resolve),
            data=json.dumps(payload),
            timeout=300,
        )
        if resp.status_code != 200:
            raise RuntimeError(
                f"qwen3-asr-flash 调用失败 (HTTP {resp.status_code}): {resp.text[:800]}"
            )
        body = resp.json()
        text, language, segments = _parse_flash_response(body)
        if not text:
            raise RuntimeError(f"qwen3-asr-flash 未返回文本: {body}")
        _notify(progress_callback, text)
        return TranscriptionResult(
            text=text,
            segments=segments
            or [{"start": 0.0, "end": 0.0, "text": text}],
            language=language,
            backend=self.backend_id,
            raw=body,
        )


def _parse_flash_response(body: Dict[str, Any]) -> tuple:
    """Best-effort parse of DashScope multimodal ASR response shapes."""
    output = body.get("output") or {}
    text = ""
    language = None
    segments: list = []

    # Shape A: choices[0].message.content[0].text  (+ annotations)
    choices = output.get("choices") if isinstance(output, dict) else None
    if isinstance(choices, list) and choices:
        message = (choices[0] or {}).get("message") or {}
        content = message.get("content")
        if isinstance(content, list):
            parts = []
            for item in content:
                if isinstance(item, dict) and item.get("text"):
                    parts.append(item["text"])
            text = "".join(parts).strip()
        elif isinstance(content, str):
            text = content.strip()
        annotations = message.get("annotations") or []
        for ann in annotations:
            if not isinstance(ann, dict):
                continue
            language = language or ann.get("language")
            sentences = ann.get("sentences") or []
            if sentences:
                segments = segments_from_sentences(sentences, time_unit="ms")

    if not text and isinstance(output, dict):
        # Shape B: output.text / output.output.sentence.text
        text = (output.get("text") or "").strip()
        nested = output.get("output") if isinstance(output.get("output"), dict) else {}
        sentence = nested.get("sentence") if isinstance(nested, dict) else None
        if isinstance(sentence, dict):
            text = text or (sentence.get("text") or "").strip()
            language = language or sentence.get("language")

    return text, language, segments
