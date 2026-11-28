import json
import os
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, patch

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
STAGE3_DIR = os.path.join(ROOT_DIR, "03_Agentic_Engine")
for p in (ROOT_DIR, STAGE3_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

import config
import llm_parser
import llm_providers
import utils


class TestLLMProviders(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self._orig_prefs = config.USER_PREFS_FILE
        self._orig_keys = config.LLM_KEYS_FILE
        config.USER_PREFS_FILE = os.path.join(self._tmp.name, "user_preferences.json")
        config.LLM_KEYS_FILE = os.path.join(self._tmp.name, ".llm_keys.json")

    def tearDown(self):
        config.USER_PREFS_FILE = self._orig_prefs
        config.LLM_KEYS_FILE = self._orig_keys
        self._tmp.cleanup()

    def test_llm_key_storage_masking_and_env_fallback(self):
        with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-env-1234567890abcd"}, clear=False):
            masked = utils.get_masked_llm_keys()
            self.assertTrue(masked["openai"]["has_key"])
            self.assertEqual(masked["openai"]["source"], "env")
            self.assertNotIn("1234567890", masked["openai"]["masked"])

            utils.save_llm_api_keys({"gemini": "AIzaSyTestSecretKey9988"})
            self.assertEqual(utils.get_llm_api_key("gemini"), "AIzaSyTestSecretKey9988")
            masked_after = utils.get_masked_llm_keys()
            self.assertEqual(masked_after["gemini"]["source"], "file")
            self.assertEqual(masked_after["gemini"]["masked"], "AIza...9988")

            # Masked placeholder should not overwrite stored secret
            utils.save_llm_api_keys({"gemini": "AIza...9988"})
            self.assertEqual(utils.get_llm_api_key("gemini"), "AIzaSyTestSecretKey9988")

            # Explicit empty string clears key
            utils.save_llm_api_keys({"gemini": ""})
            self.assertEqual(utils.get_llm_api_key("gemini"), "")

    def test_preference_validation_for_cloud_and_critic_depth(self):
        saved = utils.save_user_preferences(
            {
                "llm_backend": "anthropic",
                "cloud_model": "claude-sonnet-4-20250514",
                "critic_depth": "3",
            }
        )
        self.assertEqual(saved["llm_backend"], "anthropic")
        self.assertEqual(saved["cloud_model"], "claude-sonnet-4-20250514")
        self.assertEqual(saved["critic_depth"], "3")

        reverted = utils.save_user_preferences({"llm_backend": "invalid_provider", "critic_depth": "99"})
        self.assertEqual(reverted["llm_backend"], "anthropic")
        self.assertEqual(reverted["critic_depth"], "3")

    def test_capability_tiers_and_gemini_schema_conversion(self):
        self.assertEqual(llm_providers.infer_capability_tier("openvino", "qwen3-1.7b-ov"), "compact")
        self.assertEqual(llm_providers.infer_capability_tier("openvino", "qwen3-4b-ov"), "standard")
        self.assertEqual(llm_providers.infer_capability_tier("ollama", "qwen3:32b"), "frontier")
        self.assertEqual(llm_providers.infer_capability_tier("gemini", "gemini-2.5-pro"), "frontier")
        self.assertEqual(llm_providers.default_critic_budget("compact"), 1)
        self.assertEqual(llm_providers.default_critic_budget("standard"), 2)
        self.assertEqual(llm_providers.default_critic_budget("frontier"), 3)

        conv = llm_providers._to_gemini_schema(llm_parser._VIBE_CHECK_SCHEMA)
        self.assertEqual(conv["type"], "OBJECT")
        self.assertEqual(conv["properties"]["selected_indices"]["type"], "ARRAY")
        self.assertEqual(conv["properties"]["selected_indices"]["items"]["type"], "INTEGER")

    def test_cloud_backends_http_adapters_and_secret_redaction(self):
        fake_resp = MagicMock()
        fake_resp.status_code = 200
        fake_resp.json.return_value = {
            "candidates": [{"content": {"parts": [{"text": '{"ok": true}'}]}}]
        }
        with patch("requests.post", return_value=fake_resp) as post_mock:
            client = llm_providers.build_cloud_llm_client("gemini", cloud_model="gemini-2.5-flash", api_key="gem-secret-xyz")
            out = client.generate_json("sys", "usr", json_schema={"type": "object", "properties": {"ok": {"type": "boolean"}}})
            self.assertIn("ok", out)
            call_url = post_mock.call_args.args[0]
            call_headers = post_mock.call_args.kwargs["headers"]
            self.assertNotIn("gem-secret-xyz", call_url)
            self.assertEqual(call_headers["x-goog-api-key"], "gem-secret-xyz")

        # OpenAI adapter
        fake_resp.json.return_value = {"choices": [{"message": {"content": '{"ok": true}'}}]}
        with patch("requests.post", return_value=fake_resp) as post_mock:
            client = llm_providers.build_cloud_llm_client("openai", cloud_model="gpt-4o-mini", api_key="sk-open-123")
            out = client.generate_json("sys", "usr")
            self.assertIn("ok", out)
            self.assertEqual(post_mock.call_args.kwargs["headers"]["Authorization"], "Bearer sk-open-123")

        # Anthropic adapter (tool_use block)
        fake_resp.json.return_value = {"content": [{"type": "tool_use", "input": {"ok": True}}]}
        with patch("requests.post", return_value=fake_resp):
            client = llm_providers.build_cloud_llm_client(
                "anthropic", cloud_model="claude-3-5-haiku-latest", api_key="sk-ant-999"
            )
            out = client.generate_json("sys", "usr", json_schema={"type": "object", "properties": {"ok": {"type": "boolean"}}})
            self.assertEqual(json.loads(out)["ok"], True)

        # OpenRouter adapter redacts key in error message
        err_resp = MagicMock()
        err_resp.status_code = 401
        err_resp.text = "Invalid key sk-or-supersecret-123"
        with patch("requests.post", return_value=err_resp):
            client = llm_providers.build_cloud_llm_client("openrouter", api_key="sk-or-supersecret-123")
            with self.assertRaises(RuntimeError) as ctx:
                client.generate_json("sys", "usr")
            self.assertNotIn("sk-or-supersecret-123", str(ctx.exception))
            self.assertIn("[REDACTED]", str(ctx.exception))

    def test_get_llm_and_catalog_status_with_cloud_provider(self):
        utils.save_user_preferences({"llm_backend": "openai", "cloud_model": "gpt-4.1-mini"})
        utils.save_llm_api_keys({"openai": "sk-test-abcdef1234"})
        client = llm_parser.get_llm()
        self.assertEqual(client.backend, "openai")
        self.assertEqual(client.cloud_model, "gpt-4.1-mini")
        self.assertEqual(client.capability_tier, "standard")

        status = llm_parser.get_llm_catalog_status(base_dir=self._tmp.name, probe_ollama=False)
        self.assertEqual(status["active"]["llm_backend"], "openai")
        self.assertEqual(status["active"]["cloud_model"], "gpt-4.1-mini")
        self.assertTrue(status["cloud_providers"]["openai"]["has_key"])
        self.assertEqual(status["cloud_providers"]["openai"]["masked_key"], "sk-t...1234")


if __name__ == "__main__":
    unittest.main()
