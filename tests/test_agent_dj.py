import json
import os
import sys
import tempfile
import threading
import unittest
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
import llm_parser
import llm_providers
import utils


class TestAgentDJUnit(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self._orig_prefs = config.USER_PREFS_FILE
        config.USER_PREFS_FILE = os.path.join(self._tmp.name, "user_preferences.json")

    def tearDown(self):
        config.USER_PREFS_FILE = self._orig_prefs
        self._tmp.cleanup()

    def test_critic_budget_scales_with_capability_tier_and_overrides(self):
        compact_client = llm_providers.UnifiedLLMClient(backend="openvino", model_key="qwen3-1.7b-ov")
        standard_client = llm_providers.UnifiedLLMClient(backend="openvino", model_key="qwen3-4b-ov")
        frontier_client = llm_providers.UnifiedLLMClient(backend="openai", model_key="gpt-4.1", cloud_model="gpt-4.1")

        self.assertEqual(agent_dj.resolve_critic_budget(frontier_client, fast=True), 0)
        self.assertEqual(agent_dj.resolve_critic_budget(compact_client, fast=False), 1)
        self.assertEqual(agent_dj.resolve_critic_budget(standard_client, fast=False), 2)
        self.assertEqual(agent_dj.resolve_critic_budget(frontier_client, fast=False), 3)
        self.assertEqual(agent_dj.resolve_critic_budget(frontier_client, fast=False, override_depth="1"), 1)
        self.assertEqual(agent_dj.resolve_critic_budget(compact_client, fast=False, override_depth="0"), 0)

    def test_bounded_critic_dedupes_queries_and_caps_count(self):
        pipe = llm_providers.UnifiedLLMClient(backend="openai", cloud_model="gpt-4o-mini", api_key="sk-test")
        pipe.generate_json = MagicMock(
            return_value=json.dumps(
                {
                    "sufficient": False,
                    "critique": "Need deeper atmospheric electronic tracks.",
                    "new_queries": ["old query", "dark ambient house", "modular synthchill", "extra1", "extra2"],
                    "seed_tracks": [{"name": "Kiara", "artist": "Bonobo"}],
                }
            )
        )
        res = agent_dj.run_bounded_critic_step(
            pipe,
            "late night ambient electronica",
            {"local_count": 2, "discovered_count": 1},
            past_queries=["old query"],
            iteration=1,
        )
        self.assertFalse(res["sufficient"])
        self.assertEqual(res["new_queries"], ["dark ambient house", "modular synthchill", "extra1"])
        self.assertEqual(res["seed_tracks"], [{"name": "Kiara", "artist": "Bonobo"}])


class TestAgentDJSessionIntegration(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self._orig_s2 = discovery_engine.STAGE2_DIR
        self._orig_last_pl = discovery_engine.LAST_PL_PATH
        self._orig_arch = config.PLAYLIST_HISTORY_DIR
        self._orig_prefs = config.USER_PREFS_FILE

        self.s2_dir = os.path.join(self._tmp.name, "stage2")
        self.arch_dir = os.path.join(self._tmp.name, "history")
        os.makedirs(self.s2_dir, exist_ok=True)
        os.makedirs(self.arch_dir, exist_ok=True)

        discovery_engine.STAGE2_DIR = self.s2_dir
        discovery_engine.LAST_PL_PATH = os.path.join(self._tmp.name, "last_pl.json")
        config.PLAYLIST_HISTORY_DIR = self.arch_dir
        config.USER_PREFS_FILE = os.path.join(self._tmp.name, "user_preferences.json")

        ids = [f"loc_{i}" for i in range(6)]
        embs = np.eye(6, 8, dtype=np.float32)
        np.savez(os.path.join(self.s2_dir, "taste_vectors.npz"), track_ids=np.array(ids), embeddings=embs)

        meta = {}
        for i, tid in enumerate(ids):
            meta[tid] = {
                "name": f"Local Song {i}",
                "artists": f"Artist {i}",
                "genres": "electronic, house",
                "cluster_id": 0,
                "recency_weight": 1.0,
                "audio_features": {"bpm": 122, "energy": 0.72, "valence": 0.45, "key": 1, "mode": 1},
            }
        utils.atomic_write_json(os.path.join(self.s2_dir, "track_metadata.json"), meta)
        utils.atomic_write_json(
            os.path.join(self.s2_dir, "taste_profile_summary.json"),
            {
                "embed_backend": "minilm",
                "master_taste_centroid": np.ones(8, dtype=np.float32).tolist(),
                "clusters": [{"cluster_id": 0, "vibe_label": "Night House", "top_genres": ["house"], "weighted_share_pct": 100.0}],
            },
        )

    def tearDown(self):
        discovery_engine.STAGE2_DIR = self._orig_s2
        discovery_engine.LAST_PL_PATH = self._orig_last_pl
        config.PLAYLIST_HISTORY_DIR = self._orig_arch
        config.USER_PREFS_FILE = self._orig_prefs
        self._tmp.cleanup()

    def _make_session_with_mock_pipe(self):
        ws = discovery_engine.WarmDiscoverySession()
        ws.embed_tokenizer = MagicMock()
        ws.embed_model = MagicMock()
        ws.embed_model.is_clap = False
        ws.pipe = llm_providers.UnifiedLLMClient(backend="openai", cloud_model="gpt-4o-mini", api_key="sk-test")
        ws.llm_config_sig = ws._current_llm_pref_sig()
        ws.sp = MagicMock()
        return ws

    def test_generate_with_bounded_critic_reasons_and_cancellation(self):
        ws = self._make_session_with_mock_pipe()
        sp_track_a = {
            "id": "5abcdefghij0123456789A",
            "name": "Nightfall",
            "artists": [{"id": "a1", "name": "Kavinsky"}],
            "album": {"release_date": "2013-01-01"},
            "popularity": 60,
            "explicit": False,
        }
        sp_track_b = {
            "id": "5abcdefghij0123456789B",
            "name": "Genesis",
            "artists": [{"id": "a2", "name": "Justice"}],
            "album": {"release_date": "2007-06-01"},
            "popularity": 55,
            "explicit": False,
        }

        def fake_sp_search(q="", type="track", limit=10, offset=0):
            if offset > 0:
                return {"tracks": {"items": []}}
            if "critic" in q.lower():
                return {"tracks": {"items": [sp_track_b]}}
            return {"tracks": {"items": [sp_track_a]}}

        ws.sp.search.side_effect = fake_sp_search
        ws.sp.artists.return_value = {"artists": []}

        critic_calls = {"n": 0}

        def fake_gen_json(system_msg, user_msg, json_schema=None, **kwargs):
            if json_schema == llm_parser._PARSE_PROMPT_SCHEMA:
                return json.dumps(
                    {
                        "search_queries": ["synthwave drive"],
                        "recommended_tracks": [],
                        "playlist_name": "Neon Overdrive",
                        "playlist_description": "Moody midnight synthwave.",
                    }
                )
            if json_schema == agent_dj._AGENT_CRITIC_SCHEMA:
                critic_calls["n"] += 1
                if critic_calls["n"] == 1:
                    return json.dumps(
                        {
                            "sufficient": False,
                            "critique": "Add harder electro",
                            "new_queries": ["french electro critic"],
                            "seed_tracks": [],
                        }
                    )
                return json.dumps({"sufficient": True, "critique": "Ready", "new_queries": [], "seed_tracks": []})
            if json_schema == llm_parser._VIBE_CHECK_SCHEMA:
                return json.dumps(
                    {
                        "selected_indices": [0, 1, 2, 3, 4],
                        "playlist_name": "Midnight Telemetry",
                        "playlist_description": "Cohesive dark synth mix.",
                        "dj_notes": "Builds tension smoothly from groove to peak.",
                        "track_reasons": [
                            {"index": 0, "reason": "Anchor local groove"},
                            {"index": 3, "reason": "Adds driving analog bass"},
                        ],
                    }
                )
            return "{}"

        ws.pipe.generate_json = fake_gen_json

        progress_events = []
        with patch("discovery_engine.embed_texts", side_effect=lambda docs, *a, **k: np.ones((len(docs), 8), dtype=np.float32)), \
             patch("reccobeats_client.fetch_audio_features_batch", return_value={}), \
             patch("reccobeats_client.fetch_track_metadata_batch", return_value={}), \
             patch("reccobeats_client.fetch_reccobeats_recommendations", return_value=[]):
            res = ws.generate(
                prompt="dark midnight drive synthwave",
                size=5,
                fast=False,
                critic_depth="2",
                progress_cb=progress_events.append,
                save_to_archive=True,
            )

        self.assertEqual(res["critic_passes"], 2)
        self.assertEqual(res["playlist"]["dj_notes"], "Builds tension smoothly from groove to peak.")
        self.assertTrue(any(t.get("reason") for t in res["playlist"]["tracks"]))
        self.assertEqual(progress_events[-1]["pct"], 100)

        # Cancellation event stops before doing work
        cancel_ev = threading.Event()
        cancel_ev.set()
        with self.assertRaises(agent_dj.GenerationCancelledError):
            ws.generate(prompt="should cancel", size=5, fast=True, cancel_event=cancel_ev)

    def test_chat_refine_respects_locked_tracks_and_persists_turns(self):
        ws = self._make_session_with_mock_pipe()
        with patch("discovery_engine.embed_texts", side_effect=lambda docs, *a, **k: np.ones((len(docs), 8), dtype=np.float32)):
            ws.generate(prompt="house groove", size=5, fast=True, local_only=True, save_to_archive=True)

        initial_tracks = ws.last_playlist["tracks"]
        locked_track = initial_tracks[0]
        locked_id = locked_track["id"]

        lock_res = ws.toggle_track_lock(locked_id, locked=True)
        self.assertTrue(lock_res["locked"])
        self.assertIn(locked_id, lock_res["locked_ids"])

        # Ask AI DJ via chat to remove the locked track's artist + steer mood
        ws.pipe.generate_json = MagicMock(
            return_value=json.dumps(
                {
                    "assistant_reply": "Removed unlocked tracks by that artist while keeping your locked favorite.",
                    "action": "refine",
                    "drop_track_numbers": [1],
                    "remove_artists": [locked_track["artist"]],
                    "direction_prompt": "brighter melodic synth",
                    "discover_queries": ["melodic synth bright"],
                }
            )
        )
        ws.sp.search.return_value = {
            "tracks": {
                "items": [
                    {
                        "id": "5freshchatdiscovery01",
                        "name": "Fresh Chat Discovery",
                        "artists": [{"name": "RUFUS DU SOL"}],
                        "explicit": False,
                    }
                ]
            }
        }

        with patch("discovery_engine.embed_texts", side_effect=lambda docs, *a, **k: np.ones((len(docs), 8), dtype=np.float32)):
            chat_res = ws.chat_refine(f"drop track 1 and remove {locked_track['artist']}, add brighter melodic synth", fast=False)

        new_ids = [t["id"] for t in chat_res["playlist"]["tracks"]]
        self.assertIn(locked_id, new_ids)
        self.assertEqual(len(chat_res["history"]), 2)
        self.assertEqual(ws.get_chat_state()["locked_ids"], [locked_id])


if __name__ == "__main__":
    unittest.main()
