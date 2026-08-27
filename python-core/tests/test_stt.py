import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.stt import (  # noqa: E402
    DEFAULT_STT_BACKEND,
    DashScopeAuthError,
    TranscriberFactory,
    missing_key_message,
    parse_filetrans_transcript_json,
    resolve_dashscope_api_key,
    resolve_dashscope_base_url,
    resolve_stt_backend,
    segments_from_sentences,
)


def _without_stt_env():
    keys = (
        "OPENAUTONOTE_STT_BACKEND",
        "DASHSCOPE_API_KEY",
        "DASHSCOPE_REGION",
        "DASHSCOPE_BASE_URL",
    )
    return {k: v for k, v in os.environ.items() if k not in keys}


class ResolveConfigTests(unittest.TestCase):
    def test_default_backend(self):
        with mock.patch.dict(os.environ, _without_stt_env(), clear=True):
            with mock.patch("core.stt._read_user_config", return_value={}):
                self.assertEqual(resolve_stt_backend({}), DEFAULT_STT_BACKEND)
        self.assertEqual(DEFAULT_STT_BACKEND, "qwen3-asr-flash-filetrans")

    def test_backend_aliases(self):
        with mock.patch.dict(os.environ, _without_stt_env(), clear=True):
            with mock.patch("core.stt._read_user_config", return_value={}):
                self.assertEqual(
                    resolve_stt_backend({"stt_backend": "filetrans"}), DEFAULT_STT_BACKEND
                )
                self.assertEqual(
                    resolve_stt_backend({"stt_backend": "whisper"}), "faster-whisper"
                )

    def test_env_backend_overrides_config(self):
        with mock.patch.dict(os.environ, {"OPENAUTONOTE_STT_BACKEND": "qwen3-asr-flash"}, clear=False):
            self.assertEqual(resolve_stt_backend({"stt_backend": "faster-whisper"}), "qwen3-asr-flash")

    def test_missing_key_raises_helpful_error(self):
        env = {k: v for k, v in os.environ.items() if k != "DASHSCOPE_API_KEY"}
        with mock.patch.dict(os.environ, env, clear=True):
            with mock.patch("core.stt._read_user_config", return_value={}):
                with self.assertRaises(DashScopeAuthError) as ctx:
                    resolve_dashscope_api_key({})
        self.assertIn("DASHSCOPE_API_KEY", str(ctx.exception))
        self.assertIn("help.aliyun.com", str(ctx.exception))

    def test_key_from_env(self):
        with mock.patch.dict(os.environ, {"DASHSCOPE_API_KEY": "sk-test-env"}, clear=False):
            self.assertEqual(resolve_dashscope_api_key({"dashscope_api_key": "sk-config"}), "sk-test-env")

    def test_key_from_config_when_env_empty(self):
        env = {k: v for k, v in os.environ.items() if k != "DASHSCOPE_API_KEY"}
        with mock.patch.dict(os.environ, env, clear=True):
            with mock.patch("core.stt._read_user_config", return_value={}):
                self.assertEqual(
                    resolve_dashscope_api_key({"dashscope_api_key": "sk-from-settings"}),
                    "sk-from-settings",
                )

    def test_beijing_default_endpoint(self):
        with mock.patch.dict(os.environ, _without_stt_env(), clear=True):
            with mock.patch("core.stt._read_user_config", return_value={}):
                self.assertEqual(
                    resolve_dashscope_base_url({}),
                    "https://dashscope.aliyuncs.com/api/v1",
                )

    def test_intl_endpoint(self):
        with mock.patch.dict(os.environ, {"DASHSCOPE_REGION": "intl"}, clear=False):
            with mock.patch.dict(os.environ, {"DASHSCOPE_BASE_URL": ""}, clear=False):
                url = resolve_dashscope_base_url({"dashscope_region": "beijing"})
                # env region wins over config
                self.assertEqual(url, "https://dashscope-intl.aliyuncs.com/api/v1")

    def test_explicit_base_url(self):
        with mock.patch.dict(
            os.environ,
            {"DASHSCOPE_BASE_URL": "https://example.dashscope.test/api/v1/"},
            clear=False,
        ):
            self.assertEqual(
                resolve_dashscope_base_url({}),
                "https://example.dashscope.test/api/v1",
            )


class ParseResultTests(unittest.TestCase):
    def test_parse_qwen3_sentences_ms(self):
        data = {
            "transcripts": [
                {
                    "text": "你好世界",
                    "sentences": [
                        {
                            "begin_time": 120,
                            "end_time": 980,
                            "text": "你好",
                            "language": "zh",
                        },
                        {
                            "begin_time": 1000,
                            "end_time": 1800,
                            "text": "世界",
                            "language": "zh",
                        },
                    ],
                }
            ]
        }
        result = parse_filetrans_transcript_json(data)
        self.assertEqual(result.backend, DEFAULT_STT_BACKEND)
        self.assertEqual(result.text, "你好世界")
        self.assertEqual(result.language, "zh")
        self.assertEqual(len(result.segments), 2)
        self.assertAlmostEqual(result.segments[0]["start"], 0.12)
        self.assertAlmostEqual(result.segments[1]["end"], 1.8)
        self.assertEqual(result.segments[0]["text"], "你好")

    def test_parse_empty_falls_back_to_full_text_segment(self):
        result = parse_filetrans_transcript_json(
            {"transcripts": [{"text": "only text", "sentences": []}]}
        )
        self.assertEqual(result.text, "only text")
        self.assertEqual(result.segments[0]["text"], "only text")

    def test_segments_from_sentences_seconds(self):
        segs = segments_from_sentences(
            [{"start": 1, "end": 2, "text": "hi"}], time_unit="s"
        )
        self.assertEqual(segs[0]["start"], 1.0)


class FactoryTests(unittest.TestCase):
    def test_unknown_backend(self):
        with self.assertRaises(ValueError):
            TranscriberFactory.get_transcriber("not-a-real-backend")

    def test_filetrans_factory(self):
        t = TranscriberFactory.get_transcriber("qwen3-asr-flash-filetrans")
        self.assertEqual(t.backend_id, "qwen3-asr-flash-filetrans")

    def test_flash_factory(self):
        t = TranscriberFactory.get_transcriber("qwen3-asr-flash")
        self.assertEqual(t.backend_id, "qwen3-asr-flash")

    def test_local_stub_without_transformers(self):
        t = TranscriberFactory.get_transcriber("qwen3-asr-local")
        with tempfile.NamedTemporaryFile(suffix=".wav") as fh:
            with self.assertRaises(RuntimeError) as ctx:
                t.transcribe(fh.name)
        self.assertIn("Qwen3-ASR", str(ctx.exception))

    def test_missing_key_message_mentions_env_and_file(self):
        msg = missing_key_message()
        self.assertIn("DASHSCOPE_API_KEY", msg)
        self.assertIn(".env", msg)


class DotenvTests(unittest.TestCase):
    def test_dotenv_setdefault(self):
        from core.stt import _parse_dotenv

        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / ".env"
            path.write_text("OAN_TEST_DOTENV_KEY=from-file\n", encoding="utf-8")
            os.environ.pop("OAN_TEST_DOTENV_KEY", None)
            _parse_dotenv(path)
            self.assertEqual(os.environ.get("OAN_TEST_DOTENV_KEY"), "from-file")
            os.environ["OAN_TEST_DOTENV_KEY"] = "already-set"
            path.write_text("OAN_TEST_DOTENV_KEY=should-not-win\n", encoding="utf-8")
            _parse_dotenv(path)
            self.assertEqual(os.environ.get("OAN_TEST_DOTENV_KEY"), "already-set")
            os.environ.pop("OAN_TEST_DOTENV_KEY", None)


if __name__ == "__main__":
    unittest.main()
