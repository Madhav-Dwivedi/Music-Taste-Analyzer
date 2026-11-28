import json
import os
import sys
import tempfile
import threading
import unittest
import urllib.request
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
STAGE1_DIR = os.path.join(ROOT_DIR, "01_Data_Extractor")
STAGE2_DIR = os.path.join(ROOT_DIR, "02_Vector_Core")
STAGE3_DIR = os.path.join(ROOT_DIR, "03_Agentic_Engine")
for _p in (ROOT_DIR, STAGE1_DIR, STAGE2_DIR, STAGE3_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import audio_scorer
import config
import deezer_audio_client as dz
import discovery_engine
import llm_parser
import reccobeats_client
import utils
import web_app


class TestAudioScorerUnit(unittest.TestCase):
    def test_is_real_clap_model_rejects_magic_mock(self):
        self.assertFalse(audio_scorer.is_real_clap_model(None))
        self.assertFalse(audio_scorer.is_real_clap_model(MagicMock()))
        self.assertFalse(audio_scorer.is_real_clap_model(SimpleNamespace(is_clap=False)))
        self.assertTrue(audio_scorer.is_real_clap_model(SimpleNamespace(is_clap=True)))

    def test_raw_clap_audio_cache_roundtrip_and_missing(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            self.assertEqual(audio_scorer.load_raw_clap_audio_cache(tmpdir), {})
            cache = {
                "nightcall::kavinsky": np.ones(512, dtype=np.float32) * 0.25,
                "midnight city::m83": np.arange(512, dtype=np.float32),
            }
            audio_scorer.save_raw_clap_audio_cache(cache, stage2_dir=tmpdir)
            loaded = audio_scorer.load_raw_clap_audio_cache(tmpdir)
            self.assertEqual(set(loaded.keys()), set(cache.keys()))
            np.testing.assert_allclose(loaded["nightcall::kavinsky"], cache["nightcall::kavinsky"], atol=1e-6)

    def test_compute_subcluster_centroid_respects_recency_and_exclusions(self):
        embs = np.array(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )
        tids = ["t0", "t1", "t2", "t3"]
        meta = {
            "t0": {"cluster_id": 1, "sub_cluster_id": 2, "recency_weight": 10.0, "excluded_from_taste": False},
            "t1": {"cluster_id": 1, "sub_cluster_id": 2, "recency_weight": 2.0, "excluded_from_taste": False},
            "t2": {"cluster_id": 1, "sub_cluster_id": 2, "recency_weight": 50.0, "excluded_from_taste": True},
            "t3": {"cluster_id": 1, "sub_cluster_id": 0, "recency_weight": 5.0, "excluded_from_taste": False},
        }
        centroid, indices = audio_scorer.compute_subcluster_centroid(embs, tids, meta, cluster_id=1, sub_cluster_id=2)
        self.assertEqual(indices, [0, 1])
        self.assertIsNotNone(centroid)
        self.assertGreater(centroid[0], centroid[1])
        self.assertAlmostEqual(float(centroid[2]), 0.0, places=5)
        self.assertAlmostEqual(float(np.linalg.norm(centroid)), 1.0, places=5)

        none_c, none_i = audio_scorer.compute_subcluster_centroid(embs, tids, meta, cluster_id=1, sub_cluster_id=99)
        self.assertIsNone(none_c)
        self.assertEqual(none_i, [])

    def test_score_local_library_candidates_clap_alignment_vs_taste_weight(self):
        # Centered Stage 2 vectors point to taste_vec ([0, 1, 0, 0]) for t0 and away for t1,
        # while raw CLAP audio cache makes t1 match prompt_vec ([1, 0, 0, 0]).
        local_embs = np.array(
            [
                [0.0, 1.0, 0.0, 0.0],
                [0.0, -1.0, 0.0, 0.0],
                [0.1, 0.2, 0.9, 0.0],
            ],
            dtype=np.float32,
        )
        local_embs /= np.linalg.norm(local_embs, axis=1, keepdims=True)
        local_ids = ["t0", "t1", "t2"]
        local_meta = {
            "t0": {"name": "Taste Song", "artists": "Art A"},
            "t1": {"name": "Prompt Song", "artists": "Art B"},
            "t2": {"name": "Unverified Song", "artists": "Art C"},
        }
        raw_cache = {
            "taste song::art a": np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32),
            "prompt song::art b": np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
        }
        prompt_vec = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        taste_vec = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)

        pure_prompt_scores, raw_sims = audio_scorer.score_local_library_candidates(
            prompt_vec, taste_vec, local_embs, local_ids, local_meta, [0, 1, 2],
            raw_audio_cache=raw_cache, is_clap=True, taste_weight=0.0,
        )
        self.assertGreater(pure_prompt_scores[1], pure_prompt_scores[0])
        self.assertAlmostEqual(float(raw_sims[1]), 1.0, places=5)
        self.assertAlmostEqual(float(raw_sims[0]), 0.0, places=5)

        pure_taste_scores, _ = audio_scorer.score_local_library_candidates(
            prompt_vec, taste_vec, local_embs, local_ids, local_meta, [0, 1, 2],
            raw_audio_cache=raw_cache, is_clap=True, taste_weight=1.0,
        )
        self.assertGreater(pure_taste_scores[0], pure_taste_scores[1])

        # MiniLM / non-CLAP fallback
        minilm_scores, _ = audio_scorer.score_local_library_candidates(
            prompt_vec, taste_vec, local_embs, local_ids, local_meta, [0, 1, 2],
            raw_audio_cache=None, is_clap=False, taste_weight=1.0,
        )
        self.assertGreater(minilm_scores[0], minilm_scores[1])

    def test_compute_scored_mmr_balances_relevance_and_diversity(self):
        embs = np.array(
            [
                [1.0, 0.0, 0.0],
                [0.99, 0.14, 0.0],  # near-duplicate of index 0
                [0.0, 1.0, 0.0],    # diverse orthogonal item
            ],
            dtype=np.float32,
        )
        embs /= np.linalg.norm(embs, axis=1, keepdims=True)
        rels = np.array([1.0, 0.85, 0.82], dtype=np.float32)
        picked = audio_scorer.compute_scored_mmr(rels, embs, top_k=2, lambda_param=0.5)
        self.assertEqual(picked[0], 0)
        self.assertEqual(picked[1], 2)

    def test_verify_and_rerank_discovery_candidates_caches_and_backfills(self):
        import clap_embedder

        with tempfile.TemporaryDirectory() as tmpdir:
            dz_cache_path = os.path.join(tmpdir, "dz.json")
            alb_cache_path = os.path.join(tmpdir, "alb.json")
            dsp_cache_path = os.path.join(tmpdir, "dsp.json")

            # Seed raw CLAP cache with one track + 5 background vectors so audio_mean is active
            bg_cache = {
                f"bg{i}::art": np.eye(1, 512, k=i + 10, dtype=np.float32)[0]
                for i in range(5)
            }
            cached_vec = np.zeros(512, dtype=np.float32)
            cached_vec[0] = 0.2
            cached_vec[1] = 0.98
            bg_cache["cached track::art a"] = cached_vec
            audio_scorer.save_raw_clap_audio_cache(bg_cache, stage2_dir=tmpdir)

            dz_cache = {
                "cached track::art a": {"found": True, "v": 2, "album_id": 111},
            }
            alb_cache = {"111": ["Synthpop"]}
            dsp_cache = {
                "cached track::art a": {"bpm": 118.0, "energy": 0.65, "valence": 0.55, "key": 0, "mode": 1, "source": "dsp"}
            }
            utils.atomic_write_json(dz_cache_path, dz_cache)
            utils.atomic_write_json(alb_cache_path, alb_cache)
            utils.atomic_write_json(dsp_cache_path, dsp_cache)

            candidates = [
                {"id": "c1", "name": "Cached Track", "artist": "Art A", "source": "New Discovery", "genres": "", "bpm": None},
                {"id": "c2", "name": "Fresh Hit", "artist": "Art B", "source": "New Discovery", "genres": "", "bpm": None},
                {"id": "c3", "name": "Previewless Pick", "artist": "Art C", "source": "AI Curator Pick", "genres": "synthwave", "bpm": 124.0},
            ]

            prompt_vec = np.zeros(512, dtype=np.float32)
            prompt_vec[0] = 1.0
            taste_vec = np.zeros(512, dtype=np.float32)
            taste_vec[1] = 1.0

            fresh_audio_vec = np.zeros(512, dtype=np.float32)
            fresh_audio_vec[0] = 0.99
            fresh_audio_vec[1] = 0.14

            fake_model = SimpleNamespace(is_clap=True, _clap_device="cpu")

            def fake_search_dz(title, artist, cache=None):
                if "Fresh" in title:
                    return {"found": True, "v": 2, "deezer_id": 222, "album_id": 333, "preview_url": "https://dz/2.mp3", "bpm": 134.0}
                return {"found": False, "v": 2}

            with (
                patch.object(dz, "DEEZER_CACHE_PATH", dz_cache_path),
                patch.object(dz, "DEEZER_ALBUM_GENRES_CACHE_PATH", alb_cache_path),
                patch.object(dz, "DSP_CACHE_PATH", dsp_cache_path),
                patch.object(dz, "search_deezer_track", side_effect=fake_search_dz),
                patch.object(dz, "download_preview_waveform", return_value=(np.ones(48000, dtype=np.float32), "https://dz/2.mp3")),
                patch.object(dz, "extract_dsp_audio_features", return_value={"bpm": 134.0, "energy": 0.88, "valence": 0.62, "key": 9, "mode": 0, "source": "dsp"}),
                patch.object(dz, "fetch_deezer_album_genres", return_value=["Electronic", "Synthwave"]),
                patch.object(clap_embedder, "embed_audio_waveform", return_value=fresh_audio_vec),
                patch.object(clap_embedder, "embed_texts_clap", return_value=np.zeros((1, 512), dtype=np.float32)),
            ):
                reranked, stats = audio_scorer.verify_and_rerank_discovery_candidates(
                    candidates,
                    prompt_vec,
                    taste_vec,
                    embed_tokenizer=None,
                    embed_model=fake_model,
                    stage2_dir=tmpdir,
                    max_verify=10,
                    taste_weight=0.1,
                )

            self.assertEqual(stats["checked"], 3)
            self.assertEqual(stats["audio_verified"], 2)
            self.assertEqual(stats["cache_hits"], 1)
            self.assertEqual(stats["dsp_enriched"], 2)
            self.assertEqual(stats["genres_enriched"], 2)
            # Fresh Hit matches prompt_vec (0.99) closest -> ranked #1
            self.assertEqual(reranked[0]["id"], "c2")
            self.assertTrue(reranked[0]["audio_verified"])
            self.assertEqual(reranked[0]["bpm"], 134.0)
            self.assertEqual(reranked[0]["camelot"], "8A")
            self.assertIn("electronic", reranked[0]["genres"])
            # Previewless Pick stays unverified cleanly
            pick_row = [r for r in reranked if r["id"] == "c3"][0]
            self.assertFalse(pick_row["audio_verified"])
            # Persisted fresh vector to clap_audio_cache.npz
            persisted = audio_scorer.load_raw_clap_audio_cache(tmpdir)
            self.assertIn("fresh hit::art b", persisted)

    def test_stratified_shortlist_prevents_local_starvation_and_keeps_ai_picks(self):
        local = [{"id": f"l{i}", "name": f"Local {i}", "artist": f"LA {i}", "source": "Local Library"} for i in range(60)]
        disc = (
            [{"id": f"d{i}", "name": f"New {i}", "artist": f"NA {i}", "source": "New Discovery"} for i in range(30)]
            + [{"id": "p1", "name": "Pick 1", "artist": "PA 1", "source": "AI Curator Pick"}]
            + [{"id": "p2", "name": "Pick 2", "artist": "PA 2", "source": "AI Curator Pick"}]
        )
        shortlist, overflow = audio_scorer.build_stratified_curator_shortlist(
            local, disc, target_size=30, target_local_count=9, max_candidates=45
        )
        self.assertEqual(len(shortlist), 45)
        loc_cnt = sum(1 for t in shortlist if t["source"] == "Local Library")
        new_cnt = len(shortlist) - loc_cnt
        self.assertGreaterEqual(loc_cnt, 11)
        self.assertGreaterEqual(new_cnt, 30)
        sl_ids = {t["id"] for t in shortlist}
        self.assertIn("p1", sl_ids)
        self.assertIn("p2", sl_ids)
        self.assertEqual(len(shortlist) + len(overflow), len(local) + len(disc))

    def test_format_candidate_for_curator_displays_audio_match_and_source(self):
        row = {
            "name": "Nightcall",
            "artist": "Kavinsky",
            "source": "AI Curator Pick",
            "audio_verified": True,
            "prompt_match": 0.427,
            "bpm": 91.0,
            "energy": 0.65,
            "valence": 0.31,
            "genres": "synthwave",
        }
        line = llm_parser._format_candidate_for_curator(0, row)
        self.assertIn("Src:AIPick", line)
        self.assertIn("AudioOK", line)
        self.assertIn("Match:0.43", line)
        self.assertIn("91.0BPM", line)


class TestPhase1Integration(unittest.TestCase):
    def test_warm_session_generate_clap_subcluster_overrides_and_verify_toggle(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            prefs_file = os.path.join(tmpdir, "prefs.json")
            fb_file = os.path.join(tmpdir, "fb.json")
            last_pl = os.path.join(tmpdir, "last.json")
            arch_dir = os.path.join(tmpdir, "history")
            s2_dir = os.path.join(tmpdir, "s2")
            os.makedirs(s2_dir, exist_ok=True)

            # Build 4 local tracks in C0 (2 in S0, 2 in S1) with raw CLAP vectors
            embs = np.zeros((4, 512), dtype=np.float32)
            embs[0, 0] = 1.0  # S0
            embs[1, 0] = 0.9; embs[1, 2] = 0.43  # S0
            embs[2, 1] = 1.0  # S1
            embs[3, 1] = 0.9; embs[3, 3] = 0.43  # S1
            tids = ["spLoc00000000000000001", "spLoc00000000000000002", "spLoc00000000000000003", "spLoc00000000000000004"]
            meta = {
                tids[0]: {"name": "S0 Track 1", "artists": "Art 0", "genres": "jazz", "cluster_id": 0, "sub_cluster_id": 0, "recency_weight": 10.0},
                tids[1]: {"name": "S0 Track 2", "artists": "Art 1", "genres": "jazz", "cluster_id": 0, "sub_cluster_id": 0, "recency_weight": 5.0},
                tids[2]: {"name": "S1 Track 1", "artists": "Art 2", "genres": "bossa nova", "cluster_id": 0, "sub_cluster_id": 1, "recency_weight": 12.0},
                tids[3]: {"name": "S1 Track 2", "artists": "Art 3", "genres": "bossa nova", "cluster_id": 0, "sub_cluster_id": 1, "recency_weight": 6.0},
            }
            raw_cache = {
                "s0 track 1::art 0": embs[0],
                "s0 track 2::art 1": embs[1],
                "s1 track 1::art 2": embs[2],
                "s1 track 2::art 3": embs[3],
            }
            audio_scorer.save_raw_clap_audio_cache(raw_cache, stage2_dir=s2_dir)

            with (
                patch.object(config, "USER_PREFS_FILE", prefs_file),
                patch.object(config, "USER_FEEDBACK_FILE", fb_file),
                patch.object(config, "PLAYLIST_HISTORY_DIR", arch_dir),
                patch.object(discovery_engine, "LAST_PL_PATH", last_pl),
                patch.object(discovery_engine, "STAGE2_DIR", s2_dir),
            ):
                ws = discovery_engine.WarmDiscoverySession()
                ws.local_ids = tids
                ws.local_embeddings = embs
                ws.local_meta = meta
                ws.local_id_set = set(tids)
                ws.local_fps = {utils.get_track_fingerprint(m["name"], m["artists"]) for m in meta.values()}
                ws.master_centroid = np.mean(embs, axis=0)
                ws.clusters = [
                    {
                        "cluster_id": 0,
                        "vibe_label": "Jazz",
                        "weighted_share_pct": 100.0,
                        "top_genres": ["jazz", "bossa nova"],
                        "centroid_vector": np.mean(embs, axis=0).tolist(),
                        "subclusters": [
                            {"sub_cluster_id": 0, "vibe_label": "Swing", "track_count": 2, "raw_percentage": 50.0, "top_genres": ["swing"]},
                            {"sub_cluster_id": 1, "vibe_label": "Bossa", "track_count": 2, "raw_percentage": 50.0, "top_genres": ["bossa nova"]},
                        ],
                    }
                ]
                ws._vectors_sig = ws._stage2_sig()

                fake_clap = SimpleNamespace(is_clap=True, _clap_device="cpu")
                ws.embed_model = fake_clap
                ws.embed_tokenizer = object()

                mock_sp = MagicMock()
                mock_sp.search.return_value = {
                    "tracks": {
                        "items": [
                            {
                                "id": "6GyFP1nfCDB8lbD2bG0Hq9",
                                "name": "New Bossa",
                                "artists": [{"id": "artNew0000000000000001", "name": "New Art"}],
                                "album": {"release_date": "2024-01-01"},
                                "explicit": False,
                            }
                        ]
                    }
                }
                mock_sp.artist.return_value = {"genres": ["bossa nova"]}
                ws.sp = mock_sp

                prompt_vec = np.zeros((1, 512), dtype=np.float32)
                prompt_vec[0, 1] = 1.0  # aligns with S1 Bossa

                with (
                    patch.object(ws, "_ensure_vectors_loaded"),
                    patch.object(ws, "warm_up"),
                    patch.object(discovery_engine, "embed_texts", side_effect=lambda texts, *_: np.repeat(prompt_vec, len(texts), axis=0)),
                    patch.object(reccobeats_client, "fetch_reccobeats_recommendations", return_value=[]),
                    patch.object(reccobeats_client, "fetch_track_metadata_batch", return_value={}),
                    patch.object(reccobeats_client, "fetch_audio_features_batch", return_value={}),
                    patch.object(
                        discovery_engine,
                        "verify_and_rerank_discovery_candidates",
                        wraps=lambda cands, *a, **kw: ([dict(c, audio_verified=True, prompt_match=0.88) for c in cands], {"checked": len(cands), "audio_verified": len(cands), "cache_hits": 0, "dsp_enriched": 0, "genres_enriched": 0}),
                    ) as mock_verify,
                ):
                    res_on = ws.generate(
                        prompt="warm acoustic bossa",
                        fast=True,
                        size=5,
                        cluster_id=0,
                        sub_cluster_id=1,
                        known_tracks_ratio=0.4,
                        taste_weight=0.5,
                        verify_audio=True,
                        provider="spotify",
                    )
                    mock_verify.assert_called_once()
                    self.assertEqual(res_on["playlist"]["sub_cluster_id"], 1)
                    self.assertEqual(res_on["audio_verification"]["audio_verified"], 1)
                    self.assertTrue(any(t.get("audio_verified") for t in res_on["playlist"]["tracks"]))

                    mock_verify.reset_mock()
                    res_off = ws.generate(
                        prompt="warm acoustic bossa",
                        fast=True,
                        size=5,
                        cluster_id=0,
                        sub_cluster_id=1,
                        known_tracks_ratio=0.4,
                        taste_weight=0.5,
                        verify_audio=False,
                        provider="spotify",
                    )
                    mock_verify.assert_not_called()
                    self.assertIsNone(res_off["audio_verification"])

    def test_http_studio_generate_forwards_phase1_parameters(self):
        tmpdir = tempfile.TemporaryDirectory()
        try:
            prefs_file = os.path.join(tmpdir.name, "prefs.json")
            with patch.object(config, "USER_PREFS_FILE", prefs_file):
                server, app_ctx = web_app.create_server(host="127.0.0.1", port=0)
                port = server.server_address[1]
                th = threading.Thread(target=server.serve_forever, daemon=True)
                th.start()
                try:
                    fake_res = {
                        "playlist": {"playlist_name": "Sub-Vibe Mix", "tracks": []},
                        "constraints": {},
                        "candidate_pool_size": 0,
                        "audio_verification": {"checked": 2, "audio_verified": 2},
                    }
                    with patch.object(app_ctx["warm_session"], "generate", return_value=fake_res) as mock_gen:
                        req = urllib.request.Request(
                            f"http://127.0.0.1:{port}/api/studio/generate",
                            data=json.dumps(
                                {
                                    "prompt": "bossa groove",
                                    "fast": False,
                                    "size": 25,
                                    "cluster_id": 0,
                                    "sub_cluster_id": 1,
                                    "known_tracks_ratio": 0.25,
                                    "taste_weight": 0.4,
                                    "verify_audio": True,
                                }
                            ).encode("utf-8"),
                            headers={"Content-Type": "application/json"},
                            method="POST",
                        )
                        with urllib.request.urlopen(req, timeout=5) as resp:
                            self.assertEqual(resp.status, 200)
                            body = json.loads(resp.read().decode("utf-8"))
                            self.assertEqual(body["audio_verification"]["audio_verified"], 2)

                        kw = mock_gen.call_args.kwargs
                        self.assertEqual(kw["cluster_id"], 0)
                        self.assertEqual(kw["sub_cluster_id"], 1)
                        self.assertAlmostEqual(kw["known_tracks_ratio"], 0.25)
                        self.assertAlmostEqual(kw["taste_weight"], 0.4)
                        self.assertIs(kw["verify_audio"], True)
                finally:
                    app_ctx["stop_event"].set()
                    server.shutdown()
                    server.server_close()
                    th.join(timeout=2)
        finally:
            tmpdir.cleanup()


if __name__ == "__main__":
    unittest.main()
