import json
import os
import sys
import tempfile
import threading
import unittest
import urllib.request
from unittest.mock import MagicMock, patch

import numpy as np

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
STAGE2_DIR = os.path.join(ROOT_DIR, "02_Vector_Core")
STAGE3_DIR = os.path.join(ROOT_DIR, "03_Agentic_Engine")
for p in (ROOT_DIR, STAGE2_DIR, STAGE3_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

import agent_dj
import config
import discovery_engine
import utils
import web_app


class TestStudioStreamingAndChatHTTP(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self._orig_prefs = config.USER_PREFS_FILE
        self._orig_keys = config.LLM_KEYS_FILE
        self._orig_last_pl = discovery_engine.LAST_PL_PATH
        self._orig_arch = config.PLAYLIST_HISTORY_DIR
        self._orig_s2 = discovery_engine.STAGE2_DIR

        config.USER_PREFS_FILE = os.path.join(self._tmp.name, "user_preferences.json")
        config.LLM_KEYS_FILE = os.path.join(self._tmp.name, ".llm_keys.json")
        discovery_engine.LAST_PL_PATH = os.path.join(self._tmp.name, "last_pl.json")
        self.arch_dir = os.path.join(self._tmp.name, "history")
        self.s2_dir = os.path.join(self._tmp.name, "stage2")
        os.makedirs(self.arch_dir, exist_ok=True)
        os.makedirs(self.s2_dir, exist_ok=True)
        config.PLAYLIST_HISTORY_DIR = self.arch_dir
        discovery_engine.STAGE2_DIR = self.s2_dir

        ids = [f"loc_{i}" for i in range(5)]
        embs = np.eye(5, 8, dtype=np.float32)
        np.savez(os.path.join(self.s2_dir, "taste_vectors.npz"), track_ids=np.array(ids), embeddings=embs)
        meta = {
            tid: {
                "name": f"Track {i}",
                "artists": f"Artist {i}",
                "genres": "indie electronic",
                "cluster_id": 0,
                "recency_weight": 1.0,
                "audio_features": {"bpm": 118, "energy": 0.65, "valence": 0.5},
            }
            for i, tid in enumerate(ids)
        }
        utils.atomic_write_json(os.path.join(self.s2_dir, "track_metadata.json"), meta)
        utils.atomic_write_json(
            os.path.join(self.s2_dir, "taste_profile_summary.json"),
            {"embed_backend": "minilm", "master_taste_centroid": np.ones(8, dtype=np.float32).tolist(), "clusters": []},
        )

        self.server, self.app_ctx = web_app.create_server(host="127.0.0.1", port=0)
        port = self.server.server_address[1]
        self.base_url = f"http://127.0.0.1:{port}"
        self._thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self._thread.start()

    def tearDown(self):
        self.app_ctx["stop_event"].set()
        self.server.shutdown()
        self.server.server_close()
        config.USER_PREFS_FILE = self._orig_prefs
        config.LLM_KEYS_FILE = self._orig_keys
        discovery_engine.LAST_PL_PATH = self._orig_last_pl
        config.PLAYLIST_HISTORY_DIR = self._orig_arch
        discovery_engine.STAGE2_DIR = self._orig_s2
        self._tmp.cleanup()

    def _post_json(self, path, body):
        req = urllib.request.Request(
            f"{self.base_url}{path}",
            data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json", "Origin": self.base_url},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=8) as resp:
            return resp.status, json.loads(resp.read().decode("utf-8"))

    def _get_json(self, path):
        with urllib.request.urlopen(f"{self.base_url}{path}", timeout=8) as resp:
            return resp.status, json.loads(resp.read().decode("utf-8"))

    def test_sse_stream_start_progress_and_complete(self):
        def fake_generate(**kwargs):
            cb = kwargs.get("progress_cb")
            if cb:
                cb({"step": 1, "label": "Planning", "pct": 15, "detail": "Planning prompt"})
                cb({"step": 5, "label": "Curating Mix", "pct": 100, "detail": "Done"})
            return {
                "playlist": {
                    "playlist_name": "Streamed Mix",
                    "playlist_description": "SSE test",
                    "dj_notes": "Smooth progression.",
                    "critic_passes": 1,
                    "tracks": [{"id": "loc_0", "name": "Track 0", "artist": "Artist 0", "source": "Local Library"}],
                },
                "candidate_pool_size": 12,
                "critic_passes": 1,
            }

        with patch.object(self.app_ctx["warm_session"], "generate", side_effect=fake_generate):
            status, started = self._post_json("/api/studio/generate/start", {"prompt": "midnight groove", "critic_depth": "1"})
            self.assertEqual(status, 200)
            run_id = started["run_id"]
            self.assertEqual(started["status"], "running")

            with urllib.request.urlopen(f"{self.base_url}/api/studio/generate/stream?run_id={run_id}", timeout=8) as resp:
                self.assertEqual(resp.status, 200)
                body = resp.read().decode("utf-8")
            self.assertIn("event: snapshot", body)
            self.assertIn("event: complete", body)
            self.assertIn("Streamed Mix", body)

    def test_sse_stream_cancel(self):
        started_ev = threading.Event()

        def cancellable_generate(**kwargs):
            cancel_ev = kwargs["cancel_event"]
            started_ev.set()
            cancel_ev.wait(timeout=4.0)
            raise agent_dj.GenerationCancelledError("cancelled")

        with patch.object(self.app_ctx["warm_session"], "generate", side_effect=cancellable_generate):
            _, started = self._post_json("/api/studio/generate/start", {"prompt": "slow ambient"})
            run_id = started["run_id"]
            self.assertTrue(started_ev.wait(timeout=3.0))
            _, cancel_res = self._post_json("/api/studio/generate/cancel", {"run_id": run_id})
            self.assertTrue(cancel_res["cancelled"])

            with urllib.request.urlopen(f"{self.base_url}/api/studio/generate/stream?run_id={run_id}", timeout=8) as resp:
                body = resp.read().decode("utf-8")
            self.assertIn("event: cancelled", body)

    def test_lock_and_studio_chat_endpoints(self):
        ws = self.app_ctx["warm_session"]
        ws.embed_tokenizer = MagicMock()
        ws.embed_model = MagicMock()
        ws.embed_model.is_clap = False

        with patch("discovery_engine.embed_texts", side_effect=lambda docs, *a, **k: np.ones((len(docs), 8), dtype=np.float32)):
            ws.generate(prompt="electronic chill", size=4, fast=True, local_only=True, save_to_archive=True)

        tid0 = ws.last_playlist["tracks"][0]["id"]
        st, lock_res = self._post_json("/api/studio/lock", {"track_id": tid0, "locked": True})
        self.assertEqual(st, 200)
        self.assertTrue(lock_res["locked"])
        self.assertIn(tid0, lock_res["locked_ids"])

        st, chat_res = self._post_json("/api/studio/chat", {"message": "drop track 1 and keep it upbeat", "fast": True})
        self.assertEqual(st, 200)
        self.assertIn(tid0, [t["id"] for t in chat_res["playlist"]["tracks"]])
        self.assertEqual(len(chat_res["history"]), 2)

        st, chat_state = self._get_json("/api/studio/chat")
        self.assertEqual(st, 200)
        self.assertEqual(chat_state["locked_ids"], [tid0])
        self.assertEqual(len(chat_state["history"]), 2)

    def test_preferences_cloud_keys_masked_and_llm_test_endpoint(self):
        _, saved = self._post_json(
            "/api/preferences",
            {
                "preferences": {"llm_backend": "gemini", "cloud_model": "gemini-2.5-flash", "critic_depth": "2"},
                "llm_api_keys": {"gemini": "AIzaSecretCloudKey12345"},
            },
        )
        self.assertEqual(saved["preferences"]["llm_backend"], "gemini")

        _, got = self._get_json("/api/preferences")
        self.assertTrue(got["llm_api_keys_masked"]["gemini"]["has_key"])
        self.assertEqual(got["llm_api_keys_masked"]["gemini"]["masked"], "AIza...2345")
        self.assertNotIn("AIzaSecretCloudKey12345", json.dumps(got))

        fake_resp = MagicMock()
        fake_resp.status_code = 200
        fake_resp.json.return_value = {"candidates": [{"content": {"parts": [{"text": '{"ok": true}'}]}}] }
        with patch("requests.post", return_value=fake_resp):
            _, tested = self._post_json("/api/llm/test", {"backend": "gemini", "cloud_model": "gemini-2.5-flash"})
        self.assertTrue(tested["ok"])
        self.assertEqual(tested["model"], "gemini-2.5-flash")

    def test_ui_assets_contain_phase4_controls_and_endpoints(self):
        with open(os.path.join(ROOT_DIR, "web_ui", "index.html"), "r", encoding="utf-8") as f:
            html = f.read()
        with open(os.path.join(ROOT_DIR, "web_ui", "app.js"), "r", encoding="utf-8") as f:
            js = f.read()
        with open(os.path.join(ROOT_DIR, "web_ui", "styles.css"), "r", encoding="utf-8") as f:
            css = f.read()

        for dom_id in (
            "studioKnownRatio",
            "studioKnownRatioAuto",
            "studioTasteWeight",
            "studioTasteWeightAuto",
            "studioCriticDepth",
            "studioProgressPanel",
            "btnCancelGenerate",
            "resDjNotes",
            "studioChatThread",
            "studioLockedSummary",
            "cloudLlmConfigBox",
            "prefCloudModel",
            "prefCloudApiKey",
            "btnTestLlm",
            "prefCriticDepth",
        ):
            self.assertIn(dom_id, html, f"Missing DOM ID {dom_id} in index.html")

        for ep in (
            "/api/studio/generate/start",
            "/api/studio/generate/stream",
            "/api/studio/generate/cancel",
            "/api/studio/chat",
            "/api/studio/lock",
            "/api/llm/test",
        ):
            self.assertIn(ep, js, f"Missing endpoint reference {ep} in app.js")

        self.assertIn(".fine-tune-drawer", css)
        self.assertIn(".studio-progress-panel", css)
        self.assertIn(".track-lock-btn", css)


if __name__ == "__main__":
    unittest.main()
