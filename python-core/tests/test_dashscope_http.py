import json
import os
import sys
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.stt_dashscope import (  # noqa: E402
    FILETRANS_MODEL,
    FLASH_MODEL,
    DashScopeFiletransTranscriber,
    _extract_task_id,
    _extract_transcription_url,
    _parse_flash_response,
)


class _Resp:
    def __init__(self, status_code, payload):
        self.status_code = status_code
        self._payload = payload
        self.text = json.dumps(payload)

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


class FiletransHttpTests(unittest.TestCase):
    def test_extract_task_id_and_url_shapes(self):
        self.assertEqual(_extract_task_id({"output": {"task_id": "abc"}}), "abc")
        self.assertEqual(
            _extract_transcription_url(
                {"output": {"result": {"transcription_url": "https://t.json"}}}
            ),
            "https://t.json",
        )
        self.assertEqual(
            _extract_transcription_url(
                {
                    "output": {
                        "results": [
                            {"subtask_status": "SUCCEEDED", "transcription_url": "https://u.json"}
                        ]
                    }
                }
            ),
            "https://u.json",
        )

    def test_submit_payload_matches_official_qwen3_filetrans_api(self):
        transcript = {
            "transcripts": [
                {
                    "text": "hello",
                    "sentences": [{"begin_time": 0, "end_time": 500, "text": "hello", "language": "en"}],
                }
            ]
        }
        calls = {"post": [], "get": []}

        def fake_post(url, headers=None, data=None, timeout=None):
            calls["post"].append({"url": url, "headers": headers, "data": json.loads(data)})
            return _Resp(200, {"output": {"task_id": "task-1"}})

        def fake_get(url, headers=None, timeout=None):
            calls["get"].append(url)
            if url.endswith("/tasks/task-1"):
                return _Resp(
                    200,
                    {
                        "output": {
                            "task_status": "SUCCEEDED",
                            "result": {"transcription_url": "https://example.invalid/result.json"},
                        }
                    },
                )
            return _Resp(200, transcript)

        env = {
            "DASHSCOPE_API_KEY": "sk-test",
            "DASHSCOPE_REGION": "beijing",
        }
        with mock.patch.dict(os.environ, env, clear=False):
            with mock.patch("core.stt_dashscope.time.sleep", return_value=None):
                with mock.patch("core.stt_dashscope.requests.post", side_effect=fake_post):
                    with mock.patch("core.stt_dashscope.requests.get", side_effect=fake_get):
                        result = DashScopeFiletransTranscriber().transcribe(
                            "https://example.invalid/audio.wav"
                        )

        self.assertEqual(result.text, "hello")
        self.assertEqual(result.backend, FILETRANS_MODEL)
        self.assertEqual(len(calls["post"]), 1)
        posted = calls["post"][0]
        self.assertTrue(posted["url"].endswith("/services/audio/asr/transcription"))
        self.assertEqual(posted["headers"]["X-DashScope-Async"], "enable")
        self.assertEqual(posted["headers"]["Authorization"], "Bearer sk-test")
        self.assertEqual(posted["data"]["model"], FILETRANS_MODEL)
        self.assertEqual(
            posted["data"]["input"]["file_url"], "https://example.invalid/audio.wav"
        )
        self.assertNotIn("file_urls", posted["data"]["input"])

    def test_flash_response_choices_shape(self):
        body = {
            "output": {
                "choices": [
                    {
                        "message": {
                            "content": [{"text": "hi there"}],
                            "annotations": [{"language": "en"}],
                        }
                    }
                ]
            }
        }
        text, language, _segments = _parse_flash_response(body)
        self.assertEqual(text, "hi there")
        self.assertEqual(language, "en")
        self.assertEqual(FLASH_MODEL, "qwen3-asr-flash")


if __name__ == "__main__":
    unittest.main()
