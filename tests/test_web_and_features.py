import http.client
import io
import json
import os
import subprocess
import sys
import tempfile
import threading
import time
import unittest
import urllib.error
import urllib.request
import zipfile
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
STAGE1_DIR = os.path.join(ROOT_DIR, "01_Data_Extractor")
STAGE2_DIR = os.path.join(ROOT_DIR, "02_Vector_Core")
STAGE3_DIR = os.path.join(ROOT_DIR, "03_Agentic_Engine")
for path in (ROOT_DIR, STAGE1_DIR, STAGE2_DIR, STAGE3_DIR):
    if path not in sys.path:
        sys.path.insert(0, path)

import automation_runner
import config
import discovery_engine
import history_analytics
import history_importer
import library_hygiene
import llm_parser
import npu_vectorizer
import reccobeats_client
import stage1_extractor
import taste_clusterer
import utils
import web_app
import ytmusic_client
import ytmusic_extractor


class TestAvoidanceFlowArchiveAndFreshness(unittest.TestCase):
    def test_artist_and_genre_avoidance_matching(self):
        # Exact artist match (no substring false positives)
        self.assertTrue(utils.matches_avoided_artist("Drake, Future", ["drake"]))
        self.assertFalse(utils.matches_avoided_artist("Mandrake", ["drake"]))
        self.assertFalse(utils.matches_avoided_artist("Infamous", ["nf"]))

        # Word-boundary genre match
        self.assertTrue(utils.matches_avoided_genre("contemporary country, pop", ["country"]))
        self.assertTrue(utils.matches_avoided_genre("lo-fi, chillhop", ["lo-fi"]))
        self.assertFalse(utils.matches_avoided_genre("trapeze pop", ["rap"]))
        self.assertTrue(utils.matches_avoided_genre(["nu metal", "rock"], ["metal"]))

    def test_dj_flow_sequencing_modes(self):
        tracks = [
            {"id": "1", "name": "Peak", "artist": "Art A", "bpm": 150.0, "energy": 0.95, "key": 0, "mode": 1},
            {"id": "2", "name": "Warm", "artist": "Art B", "bpm": 92.0, "energy": 0.25, "key": 7, "mode": 1},
            {"id": "3", "name": "Mid", "artist": "Art C", "bpm": 122.0, "energy": 0.60, "key": 9, "mode": 0},
            {"id": "4", "name": "Mid2", "artist": "Art A", "bpm": 124.0, "energy": 0.65, "key": 2, "mode": 1},
            {"id": "5", "name": "Chill", "artist": "Art D", "bpm": 85.0, "energy": 0.18, "key": 4, "mode": 0},
        ]
        ramp = utils.sequence_playlist_tracks(tracks, "energy_ramp")
        self.assertEqual([t["id"] for t in ramp], ["5", "2", "3", "4", "1"])

        descent = utils.sequence_playlist_tracks(tracks, "chill_descent")
        self.assertEqual([t["id"] for t in descent], ["1", "4", "3", "2", "5"])

        arc = utils.sequence_playlist_tracks(tracks, "warmup_peak_cooldown")
        self.assertEqual(len(arc), 5)
        self.assertLess(arc[0]["energy"], arc[2]["energy"])
        self.assertLess(arc[-1]["energy"], arc[2]["energy"])

        harmonic = utils.sequence_playlist_tracks(tracks, "harmonic")
        self.assertEqual(len(harmonic), 5)
        self.assertEqual(utils.format_camelot_label(0, 1), "8B")
        self.assertEqual(utils.format_camelot_label(9, 0), "8A")
        self.assertEqual(utils.camelot_distance((8, "B"), (8, "A")), 1)

        shuffled = utils.sequence_playlist_tracks(tracks, "shuffle", seed=42)
        self.assertEqual(len(shuffled), 5)

    def test_archive_crud_and_export_formats(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            payload = {
                "playlist_name": "Neon Drive",
                "playlist_description": "Late night synth",
                "prompt": "synthwave",
                "generated_at": "2026-11-20T12:00:00Z",
                "tracks": [
                    {
                        "id": "4cOdK2wGLETKBW3PvgPWqT",
                        "name": "Track | One",
                        "artist": "Artist A",
                        "source": "Local Library",
                        "bpm": 128.0,
                        "energy": 0.8,
                        "valence": 0.6,
                        "key": 0,
                        "mode": 1,
                    }
                ],
            }
            pid = utils.save_playlist_to_archive(payload, archive_dir=tmpdir)
            listed = utils.list_archived_playlists(archive_dir=tmpdir)
            self.assertEqual(len(listed), 1)
            self.assertEqual(listed[0]["id"], pid)

            loaded = utils.load_archived_playlist(pid, archive_dir=tmpdir)
            self.assertEqual(loaded["playlist_name"], "Neon Drive")
            self.assertIsNone(utils.load_archived_playlist("../../etc/passwd", archive_dir=tmpdir))

            csv_txt = utils.format_playlist_csv(loaded)
            self.assertIn("Track | One", csv_txt)
            self.assertIn("8B", csv_txt)
            m3u_txt = utils.format_playlist_m3u(loaded)
            self.assertIn("#EXTM3U", m3u_txt)
            md_txt = utils.format_playlist_markdown(loaded)
            self.assertIn("Track \\| One", md_txt)

            self.assertTrue(utils.delete_archived_playlist(pid, archive_dir=tmpdir))
            self.assertEqual(utils.list_archived_playlists(archive_dir=tmpdir), [])

    def test_stale_freshness_and_preferences_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            s1 = os.path.join(tmpdir, "01_Data_Extractor")
            s2 = os.path.join(tmpdir, "02_Vector_Core")
            os.makedirs(s1, exist_ok=True)
            os.makedirs(s2, exist_ok=True)

            vec_p = os.path.join(s2, "taste_vectors.npz")
            sum_p = os.path.join(s2, "taste_profile_summary.json")
            prof_p = os.path.join(s1, "user_music_profile.json")
            with open(vec_p, "wb") as f:
                f.write(b"v")
            with open(sum_p, "w", encoding="utf-8") as f:
                f.write("{}")
            time.sleep(0.02)
            with open(prof_p, "w", encoding="utf-8") as f:
                f.write("{}")
            os.utime(prof_p, (time.time() + 10, time.time() + 10))

            fresh = utils.compute_pipeline_freshness(tmpdir)
            self.assertTrue(fresh["stage2_exists"])
            self.assertTrue(fresh["stage2_stale"])

            prefs_path = os.path.join(tmpdir, "user_preferences.json")
            with patch.object(config, "USER_PREFS_FILE", prefs_path):
                saved = utils.save_user_preferences(
                    {
                        "avoid_artists": "Drake, NF",
                        "avoid_genres": ["country", "metal"],
                        "half_life_days": 90,
                        "default_playlist_size": 45,
                    }
                )
                self.assertEqual(saved["avoid_artists"], ["Drake", "NF"])
                self.assertEqual(saved["avoid_genres"], ["country", "metal"])
                self.assertEqual(saved["half_life_days"], 90.0)
                self.assertEqual(saved["default_playlist_size"], 45)


class TestAnalyticsExclusionsAndWarmSession(unittest.TestCase):
    def test_history_analytics_and_forgotten_favorites(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            s1 = os.path.join(tmpdir, "s1")
            s2 = os.path.join(tmpdir, "s2")
            os.makedirs(s1, exist_ok=True)
            os.makedirs(s2, exist_ok=True)

            stats = {
                "4cOdK2wGLETKBW3PvgPWqT": {
                    "track_name": "Old Gem",
                    "artist_name": "Good Artist",
                    "play_count": 20,
                    "skip_count": 1,
                    "total_ms_listened": 4200000,
                    "stream_timestamps": ["2022-05-10T18:00:00Z", "2022-05-11T19:00:00Z"],
                    "first_streamed_date": "2022-05-10T18:00:00Z",
                    "last_streamed_date": "2022-05-11T19:00:00Z",
                },
                "0VjIjW4GlUZAMYd2vXMi3b": {
                    "track_name": "Avoided Song",
                    "artist_name": "Avoided Artist",
                    "play_count": 30,
                    "skip_count": 2,
                    "total_ms_listened": 6000000,
                    "stream_timestamps": ["2022-06-01T10:00:00Z"],
                    "first_streamed_date": "2022-06-01T10:00:00Z",
                    "last_streamed_date": "2022-06-01T10:00:00Z",
                },
                "3ujzd7pcdAhILCnS3oGbaU": {
                    "track_name": "Skipped Often",
                    "artist_name": "Country Star",
                    "play_count": 4,
                    "skip_count": 12,
                    "total_ms_listened": 800000,
                    "stream_timestamps": ["2026-11-01T14:00:00Z"],
                    "first_streamed_date": "2026-11-01T14:00:00Z",
                    "last_streamed_date": "2026-11-01T14:00:00Z",
                },
            }
            utils.atomic_write_json(os.path.join(s1, "extended_history_stats.json"), stats)
            utils.atomic_write_json(
                os.path.join(s1, ".reccobeats_cache.json"),
                {"4cOdK2wGLETKBW3PvgPWqT": {"bpm": 122.0, "energy": 0.72, "valence": 0.65, "source": "reccobeats"}},
            )
            utils.atomic_write_json(
                os.path.join(s2, "track_metadata.json"),
                {"3ujzd7pcdAhILCnS3oGbaU": {"genres": "contemporary country"}},
            )

            analytics = history_analytics.compute_history_analytics(stage1_dir=s1, stage2_dir=s2)
            self.assertTrue(analytics["available"])
            self.assertEqual(analytics["overview"]["total_plays"], 54)
            self.assertEqual(len(analytics["yearly_evolution"]), 2)
            self.assertEqual(analytics["most_skipped"][0]["name"], "Skipped Often")

            ff = history_analytics.find_forgotten_favorites(
                min_plays=5,
                dormant_days=180,
                limit=10,
                avoid_artists=["Avoided Artist"],
                avoid_genres=["country"],
                stage1_dir=s1,
                stage2_dir=s2,
                now=datetime(2026, 11, 20, tzinfo=timezone.utc),
            )
            self.assertEqual(len(ff), 1)
            self.assertEqual(ff[0]["name"], "Old Gem")
            self.assertEqual(ff[0]["bpm"], 122.0)

    def test_clusterer_exclusions_and_skip_penalty(self):
        track_ids = ["t1", "t2", "t3"]
        metadata = {
            "t1": {"name": "Keep", "artists": "Good Band", "genres": "indie rock", "alt_ids": []},
            "t2": {"name": "Drop Artist", "artists": "Drake", "genres": "hip hop", "alt_ids": []},
            "t3": {"name": "Drop Genre", "artists": "Other", "genres": "modern country", "alt_ids": []},
        }
        kept_idx, excluded = taste_clusterer.filter_excluded_tracks(
            track_ids, metadata, avoid_artists=["drake"], avoid_genres=["country"]
        )
        self.assertEqual(kept_idx, [0])
        self.assertEqual(excluded, {"t2", "t3"})

        # Skip penalty (>60% skips with >=3 skips) down-weights track
        history_by_id = {
            "t1": {"play_count": 2, "skip_count": 8, "stream_timestamps": ["2026-10-01T10:00:00Z"]},
        }
        skip_ratios = {}
        w_pen = taste_clusterer.compute_recency_weights(
            ["t1"],
            metadata,
            {"tracks": {}, "recently_played": []},
            history_by_id=history_by_id,
            skip_penalty_enabled=True,
            skip_ratios_out=skip_ratios,
            now=datetime(2026, 11, 20, tzinfo=timezone.utc),
        )
        w_no_pen = taste_clusterer.compute_recency_weights(
            ["t1"],
            metadata,
            {"tracks": {}, "recently_played": []},
            history_by_id=history_by_id,
            skip_penalty_enabled=False,
            now=datetime(2026, 11, 20, tzinfo=timezone.utc),
        )
        self.assertAlmostEqual(skip_ratios["t1"], 0.8)
        self.assertLess(w_pen[0], w_no_pen[0] * 0.25)

    def test_llm_parser_session_avoidance_and_clean_deep_cuts(self):
        c = llm_parser.parse_prompt_heuristic(
            "clean only deep cuts workout no country music",
            avoid_artists=["Drake"],
            avoid_genres=["pop"],
        )
        self.assertTrue(c["clean_only"])
        self.assertTrue(c["deep_cuts"])
        self.assertIn("Drake", c["exclude_artists"])
        self.assertIn("country", [g.lower() for g in c["exclude_genres"]])
        self.assertIn("pop", [g.lower() for g in c["exclude_genres"]])
        self.assertFalse(any("country" in q.lower() or "pop" in q.lower() for q in c["search_queries"]))

        mock_pipe = MagicMock()
        mock_pipe.generate.return_value = json.dumps(
            {
                "search_queries": ["artist:Drake", "genre:country", "genre:synthwave"],
                "playlist_size": 25,
            }
        )
        res = llm_parser.parse_prompt(
            mock_pipe, "vibe", avoid_artists=["Drake"], avoid_genres=["country"]
        )
        prompt_sent = mock_pipe.generate.call_args.args[0]
        self.assertIn("Drake", prompt_sent)
        self.assertIn("country", prompt_sent)
        self.assertEqual(res["search_queries"], ["genre:synthwave"])

    def test_warm_session_swap_and_resequence(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            last_p = os.path.join(tmpdir, "last.json")
            arch_dir = os.path.join(tmpdir, "archive")
            with (
                patch.object(discovery_engine, "LAST_PL_PATH", last_p),
                patch.object(config, "PLAYLIST_HISTORY_DIR", arch_dir),
            ):
                ws = discovery_engine.WarmDiscoverySession()
                ws.last_playlist = {
                    "playlist_name": "Session Mix",
                    "flow_mode": "mmr",
                    "tracks": [
                        {"id": "a1", "name": "Track 1", "artist": "Art 1", "energy": 0.9, "bpm": 140.0},
                        {"id": "a2", "name": "Track 2", "artist": "Art 2", "energy": 0.3, "bpm": 90.0},
                    ],
                }
                ws.last_candidates = [
                    {"id": "a1", "name": "Track 1", "artist": "Art 1", "energy": 0.9, "bpm": 140.0},
                    {"id": "a2", "name": "Track 2", "artist": "Art 2", "energy": 0.3, "bpm": 90.0},
                    {"id": "a3", "name": "Track 3", "artist": "Art 3", "energy": 0.7, "bpm": 122.0},
                ]
                swapped = ws.swap_track(0)
                self.assertEqual(swapped["replacement"]["id"], "a3")
                self.assertEqual(swapped["playlist"]["tracks"][0]["id"], "a3")

                reseq = ws.resequence_current("energy_ramp")
                self.assertEqual([t["id"] for t in reseq["tracks"]], ["a2", "a3"])
                self.assertTrue(os.path.exists(last_p))


class TestNewStudioUpgrades(unittest.TestCase):
    def test_feedback_memory_and_forgotten_favorites_exclusion(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            fb_file = os.path.join(tmpdir, "user_feedback.json")
            s1_dir = os.path.join(tmpdir, "s1")
            s2_dir = os.path.join(tmpdir, "s2")
            os.makedirs(s1_dir, exist_ok=True)
            os.makedirs(s2_dir, exist_ok=True)

            now = datetime(2026, 6, 1, tzinfo=timezone.utc)
            old_ts = (now - timedelta(days=365)).strftime("%Y-%m-%dT10:00:00Z")
            hist = {
                "id_liked": {
                    "spotify_track_id": "id_liked",
                    "track_name": "Beloved Oldie",
                    "artist_name": "Good Artist",
                    "play_count": 20,
                    "skip_count": 0,
                    "total_ms_listened": 3600000,
                    "last_streamed_date": old_ts,
                    "fingerprint": "beloved oldie::good artist",
                },
                "id_bad": {
                    "spotify_track_id": "id_bad",
                    "track_name": "Disliked Oldie",
                    "artist_name": "Meh Artist",
                    "play_count": 25,
                    "skip_count": 0,
                    "total_ms_listened": 3800000,
                    "last_streamed_date": old_ts,
                    "fingerprint": "disliked oldie::meh artist",
                },
            }
            utils.atomic_write_json(os.path.join(s1_dir, "extended_history_stats.json"), hist)

            with patch.object(config, "USER_FEEDBACK_FILE", fb_file):
                utils.record_track_feedback({"id": "id_liked", "name": "Beloved Oldie", "artist": "Good Artist"}, "like")
                utils.record_track_feedback({"id": "id_bad", "name": "Disliked Oldie", "artist": "Meh Artist"}, "dislike")

                self.assertIn("id_liked", utils.get_liked_ids())
                d_ids, d_fps = utils.get_disliked_sets()
                self.assertIn("id_bad", d_ids)
                self.assertIn("disliked oldie::meh artist", d_fps)

                ff = history_analytics.find_forgotten_favorites(
                    stage1_dir=s1_dir,
                    stage2_dir=s2_dir,
                    min_plays=5,
                    dormant_days=180,
                    now=now,
                )
                self.assertEqual([t["id"] for t in ff], ["id_liked"])

                # Clear disliked and confirm it is unblocked
                utils.record_track_feedback({"id": "id_bad", "name": "Disliked Oldie", "artist": "Meh Artist"}, "clear")
                d_ids_after, _ = utils.get_disliked_sets()
                self.assertNotIn("id_bad", d_ids_after)

    def test_extract_uploaded_history_zip_and_zip_slip_rejection(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            buf = io.BytesIO()
            with zipfile.ZipFile(buf, "w") as zf:
                zf.writestr("Spotify Extended Streaming History/Streaming_History_Audio_2025.json", json.dumps([{"ts": "2025"}]))
                zf.writestr("Spotify Extended Streaming History/Streaming_History_Video_2025.json", json.dumps([{"ts": "2025"}]))
                zf.writestr("ReadMeFirst.pdf", b"ignored")

            res = utils.extract_uploaded_history_bytes("export.zip", buf.getvalue(), dest_dir=tmpdir)
            self.assertEqual(res["saved_files"], ["Streaming_History_Audio_2025.json"])
            self.assertTrue(os.path.isfile(os.path.join(tmpdir, "Streaming_History_Audio_2025.json")))
            self.assertFalse(os.path.isfile(os.path.join(tmpdir, "Streaming_History_Video_2025.json")))

            # Single JSON upload
            res_single = utils.extract_uploaded_history_bytes("endsong_0.json", b"[]", dest_dir=tmpdir)
            self.assertEqual(res_single["saved_files"], ["endsong_0.json"])

            # Zip-Slip path traversal must be rejected
            evil_buf = io.BytesIO()
            with zipfile.ZipFile(evil_buf, "w") as zf:
                zf.writestr("../../evil.json", b"[]")
            with self.assertRaises(ValueError):
                utils.extract_uploaded_history_bytes("evil.zip", evil_buf.getvalue(), dest_dir=tmpdir)

    def test_library_hygiene_and_spotify_dedupe(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            fb_file = os.path.join(tmpdir, "fb.json")
            s1_dir = os.path.join(tmpdir, "s1")
            os.makedirs(s1_dir, exist_ok=True)
            profile = {
                "audio_features": {"t1": {"bpm": 120.0, "energy": 0.7, "valence": 0.6}},
                "playlists": [
                    {
                        "id": "pl_1234567890",
                        "name": "Daily Mix Draft",
                        "tracks": [
                            {"id": "t1", "name": "Neon Lights", "artists": [{"name": "Synth"}]},
                            {"id": "t1b", "name": "Neon Lights (Remastered)", "artists": [{"name": "Synth"}]},
                            {"id": "t2", "name": "Overplayed Song", "artists": [{"name": "Skippy"}]},
                            {"id": "t3", "name": "Blocked Tune", "artists": [{"name": "Bad"}]},
                        ],
                    }
                ],
            }
            history = {
                "t2": {
                    "track_name": "Overplayed Song",
                    "artist_name": "Skippy",
                    "play_count": 2,
                    "skip_count": 8,
                    "fingerprint": "overplayed song - skippy",
                }
            }
            utils.atomic_write_json(os.path.join(s1_dir, "user_music_profile.json"), profile)
            utils.atomic_write_json(os.path.join(s1_dir, "extended_history_stats.json"), history)

            with patch.object(config, "USER_FEEDBACK_FILE", fb_file):
                utils.record_track_feedback({"id": "t3", "name": "Blocked Tune", "artist": "Bad"}, "dislike")
                report = library_hygiene.analyze_library_hygiene(stage1_dir=s1_dir, stage2_dir=tmpdir)
                self.assertTrue(report["available"])
                self.assertEqual(report["summary"]["duplicate_groups"], 1)
                self.assertEqual(report["summary"]["duplicate_extra_tracks"], 1)
                self.assertEqual(report["summary"]["chronically_skipped_count"], 1)
                self.assertEqual(report["summary"]["disliked_in_playlists_count"], 1)

            mock_sp = MagicMock()
            mock_sp._get.return_value = {
                "items": [
                    {"item": {"id": "a1", "name": "Track A", "artists": [{"name": "Art"}]}},
                    {"item": {"id": "a2", "name": "Track A - Radio Edit", "artists": [{"name": "Art"}]}},
                    {"item": {"id": "b1", "name": "Track B", "artists": [{"name": "Other"}]}},
                ],
                "next": None,
            }
            deduped = library_hygiene.dedupe_spotify_playlist(mock_sp, "pl_1234567890")
            self.assertEqual(deduped["original_count"], 3)
            self.assertEqual(deduped["kept_count"], 2)
            self.assertEqual(deduped["removed_count"], 1)
            mock_sp._put.assert_called_once()

    def test_clusterer_era_filter_and_refinement_parser(self):
        track_ids = ["t_old", "t_new", "t_nodate"]
        meta = {
            "t_old": {"name": "Old", "artists": "A", "alt_ids": []},
            "t_new": {"name": "New", "artists": "B", "alt_ids": []},
            "t_nodate": {"name": "NoDate", "artists": "C", "alt_ids": []},
        }
        hist_id = {
            "t_old": {"stream_timestamps": ["2019-05-01T10:00:00Z"]},
            "t_new": {"stream_timestamps": ["2025-07-12T10:00:00Z"]},
        }
        kept, excluded = taste_clusterer._filter_by_era([0, 1, 2], track_ids, meta, hist_id, {}, 2024, 2026)
        self.assertEqual(kept, [1])
        self.assertEqual(excluded, {"t_old", "t_nodate"})

        plan = llm_parser.parse_refinement_heuristic(
            "ramp up energy, remove Drake, more synthwave",
            [{"id": "d1", "name": "One Dance", "artist": "Drake"}],
        )
        self.assertIn("Drake", plan["remove_artists"])
        self.assertEqual(plan["flow_mode"], "energy_ramp")
        self.assertIn("synthwave", (plan["direction_prompt"] or "").lower())

    def test_warm_session_edit_search_export_and_refine(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            last_p = os.path.join(tmpdir, "last.json")
            arch_dir = os.path.join(tmpdir, "archive")
            meta_p = os.path.join(tmpdir, "track_metadata.json")
            utils.atomic_write_json(
                meta_p,
                {
                    "trk1": {"name": "Midnight City", "artists": "M83", "genres": "synthpop", "recency_weight": 12.0},
                    "trk2": {"name": "Sunrise Drive", "artists": "FM-84", "genres": "synthwave", "recency_weight": 8.0},
                },
            )
            with (
                patch.object(discovery_engine, "LAST_PL_PATH", last_p),
                patch.object(discovery_engine, "STAGE2_DIR", tmpdir),
                patch.object(config, "PLAYLIST_HISTORY_DIR", arch_dir),
            ):
                ws = discovery_engine.WarmDiscoverySession()
                results = ws.search_local_library("midnight")
                self.assertEqual(len(results), 1)
                self.assertEqual(results[0]["name"], "Midnight City")

                ws.last_playlist = {
                    "playlist_name": "Initial",
                    "playlist_description": "Desc",
                    "flow_mode": "mmr",
                    "tracks": [
                        {"id": "1", "name": "Keep Song", "artist": "Good Art", "bpm": 120.0, "energy": 0.6, "source": "Local Library"},
                        {"id": "2", "name": "Drop Song", "artist": "Drake", "bpm": 100.0, "energy": 0.4, "source": "Local Library"},
                        {"id": "3", "name": "Move Song", "artist": "Other Art", "bpm": 135.0, "energy": 0.9, "source": "Spotify Discovery"},
                    ],
                }
                ws.last_candidates = [
                    {"id": "4", "name": "Fast Replacement", "artist": "Synth Art", "bpm": 142.0, "energy": 0.92, "source": "Spotify Discovery"}
                ]

                edited = ws.update_current_playlist(
                    playlist_name="Renamed Mix",
                    playlist_description="Updated Desc",
                    order_indices=[2, 0, 1],
                )
                self.assertEqual(edited["playlist_name"], "Renamed Mix")
                self.assertEqual(edited["playlist_description"], "Updated Desc")
                self.assertEqual([t["id"] for t in edited["tracks"]], ["3", "1", "2"])

                removed_pl = ws.update_current_playlist(remove_indices=[2])
                self.assertEqual([t["id"] for t in removed_pl["tracks"]], ["3", "1"])

                # Put track #2 (Drake) back so refine_playlist tests removal + pool refill
                ws.last_playlist["tracks"].append(
                    {"id": "2", "name": "Drop Song", "artist": "Drake", "bpm": 100.0, "energy": 0.4, "source": "Local Library"}
                )
                refined = ws.refine_playlist("remove Drake, make it faster")
                ref_ids = [t["id"] for t in refined["playlist"]["tracks"]]
                self.assertNotIn("2", ref_ids)
                self.assertIn("4", ref_ids)
                self.assertIn("Drake", refined["playlist"]["avoid_artists"])

                mock_sp = MagicMock()
                ws.sp = mock_sp
                url = ws.export_playlist(target_playlist_id="existing12345")
                self.assertEqual(url, "https://open.spotify.com/playlist/existing12345")
                mock_sp._put.assert_called()

    def test_automation_runner_refresh_and_scheduler_helpers(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            prefs_file = os.path.join(tmpdir, "prefs.json")
            with patch.object(config, "USER_PREFS_FILE", prefs_file):
                utils.save_user_preferences(
                    {
                        "avoid_artists": ["Artist X"],
                        "avoid_genres": ["country"],
                        "automation": {
                            "prompt": "Weekly high-signal test",
                            "size": 15,
                            "flow_mode": "harmonic",
                            "fast": True,
                            "run_stage1_sync": False,
                            "run_stage2_refresh": True,
                            "target_playlist_id": "",
                        },
                    }
                )

                recorded_cmds = []

                def fake_subproc(cmd, **kwargs):
                    self.assertFalse(kwargs.get("shell", True))
                    recorded_cmds.append(cmd)
                    return subprocess.CompletedProcess(cmd, 0, stdout="OK", stderr="")

                class FakeSession:
                    def generate(self, **kwargs):
                        return {
                            "playlist": {
                                "playlist_name": "Weekly Mix",
                                "tracks": [{"id": "t1", "name": "Track 1", "artist": "A"}],
                            },
                            "candidate_pool_size": 10,
                        }

                    def export_playlist(self, **kwargs):
                        return "https://open.spotify.com/playlist/AUTOPL123456"

                res = automation_runner.run_weekly_refresh(
                    log_cb=lambda _m: None,
                    session_cls=FakeSession,
                    subprocess_run=fake_subproc,
                )
                self.assertTrue(res["ok"])
                self.assertEqual(res["track_count"], 1)
                self.assertEqual(res["target_playlist_id"], "AUTOPL123456")

                # Verify taste_clusterer received plural --avoid-artists / --avoid-genres
                cluster_cmd = [c for c in recorded_cmds if any("taste_clusterer.py" in part for part in c)][0]
                self.assertIn("--avoid-artists", cluster_cmd)
                self.assertIn("Artist X", cluster_cmd)
                self.assertNotIn("--avoid-artist", cluster_cmd)

                # Persisted target_playlist_id
                saved_auto = utils.get_effective_preferences()["automation"]
                self.assertEqual(saved_auto["target_playlist_id"], "AUTOPL123456")

                # Task scheduler helpers with mocked runner
                schtasks_calls = []

                def fake_schtasks(cmd, **kwargs):
                    self.assertFalse(kwargs.get("shell", True))
                    schtasks_calls.append(cmd)
                    if "/Query" in cmd:
                        return subprocess.CompletedProcess(cmd, 0, stdout="Status: Ready\nNext Run Time: Sunday\n", stderr="")
                    return subprocess.CompletedProcess(cmd, 0, stdout="SUCCESS", stderr="")

                reg = automation_runner.register_windows_task("SUN", "09:00", runner_fn=fake_schtasks)
                self.assertTrue(reg["registered"])
                qry = automation_runner.query_windows_task(runner_fn=fake_schtasks)
                self.assertTrue(qry["exists"])
                self.assertEqual(qry["status"], "Ready")
                rem = automation_runner.remove_windows_task(runner_fn=fake_schtasks)
                self.assertTrue(rem["removed"])
                with self.assertRaises(ValueError):
                    automation_runner.register_windows_task("BAD", "09:00", runner_fn=fake_schtasks)


class TestWebServerIntegration(unittest.TestCase):
    def test_audio_preview_endpoint_returns_cached_audio_without_youtube_embedding(self):
        entry = {
            "found": True, "v": 2,
            "preview_url": "https://example.com/preview.mp3",
            "title": "The Devil Went Down to Georgia", "artist": "The Charlie Daniels Band",
        }
        fp = utils.get_track_fingerprint(entry["title"], entry["artist"])
        with patch("deezer_audio_client.load_deezer_cache", return_value={fp: entry}):
            status, _, body = self._req(
                "/api/preview-audio?title=The%20Devil%20Went%20Down%20to%20Georgia&artist=The%20Charlie%20Daniels%20Band"
            )
        self.assertEqual(status, 200)
        self.assertTrue(json.loads(body)["found"])
        self.assertEqual(json.loads(body)["preview_url"], entry["preview_url"])

    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.prefs_file = os.path.join(self.tmpdir.name, "user_preferences.json")
        self.fb_file = os.path.join(self.tmpdir.name, "user_feedback.json")
        self.arch_dir = os.path.join(self.tmpdir.name, "playlist_history")
        self.logs_dir = os.path.join(self.tmpdir.name, "logs")

        self.patches = [
            patch.object(config, "USER_PREFS_FILE", self.prefs_file),
            patch.object(config, "USER_FEEDBACK_FILE", self.fb_file),
            patch.object(config, "PLAYLIST_HISTORY_DIR", self.arch_dir),
            patch.object(config, "LOGS_DIR", self.logs_dir),
        ]
        for p in self.patches:
            p.start()

        self.server, self.app_ctx = web_app.create_server(host="127.0.0.1", port=0)
        self.port = self.server.server_address[1]
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def tearDown(self):
        self.app_ctx["stop_event"].set()
        self.app_ctx["job_manager"].cancel_job()
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)
        for p in self.patches:
            p.stop()
        self.tmpdir.cleanup()

    def _req(self, path, method="GET", body=None, raw_body=None, headers=None):
        url = f"http://127.0.0.1:{self.port}{path}"
        hdrs = dict(headers or {})
        if raw_body is not None:
            data = raw_body
        elif body is not None:
            data = json.dumps(body).encode("utf-8")
            hdrs.setdefault("Content-Type", "application/json")
        else:
            data = None
        req = urllib.request.Request(url, data=data, headers=hdrs, method=method)
        with urllib.request.urlopen(req, timeout=5) as resp:
            return resp.status, resp.headers.get("Content-Type", ""), resp.read().decode("utf-8")

    def test_static_and_api_endpoints(self):
        st, ct, html = self._req("/")
        self.assertEqual(st, 200)
        self.assertIn("text/html", ct)
        self.assertIn("Music Taste Analyzer Studio", html)

        st_js, ct_js, js = self._req("/app.js")
        self.assertEqual(st_js, 200)
        self.assertIn("javascript", ct_js)
        self.assertIn("escapeHtml", js)

        # Path traversal blocked
        with self.assertRaises(urllib.error.HTTPError) as ctx:
            self._req("/../config.py")
        try:
            self.assertIn(ctx.exception.code, (403, 404))
        finally:
            ctx.exception.close()

        # Status & Doctor
        st_s, _, status_raw = self._req("/api/status")
        self.assertEqual(st_s, 200)
        status_obj = json.loads(status_raw)
        self.assertIn("pipeline", status_obj)
        self.assertIn("studio", status_obj)

        st_d, _, doc_raw = self._req("/api/doctor")
        self.assertEqual(st_d, 200)
        self.assertIn("python_version", json.loads(doc_raw))

        # Session exclusions POST & GET
        st_ex, _, ex_raw = self._req(
            "/api/session/exclusions",
            method="POST",
            body={"avoid_artists": ["Drake"], "avoid_genres": ["country"], "persist": True},
        )
        self.assertEqual(st_ex, 200)
        ex_obj = json.loads(ex_raw)
        self.assertEqual(ex_obj["avoid_artists"], ["Drake"])
        self.assertEqual(ex_obj["avoid_genres"], ["country"])

        _, _, ex_get = self._req("/api/session/exclusions")
        self.assertEqual(json.loads(ex_get)["avoid_artists"], ["Drake"])

        # Preferences round-trip
        _, _, prefs_get = self._req("/api/preferences")
        self.assertEqual(json.loads(prefs_get)["preferences"]["avoid_genres"], ["country"])

        # Archive & CSV/M3U/MD download
        pid = utils.save_playlist_to_archive(
            {
                "playlist_name": "HTTP Test List",
                "tracks": [{"id": "4cOdK2wGLETKBW3PvgPWqT", "name": "Song", "artist": "Art", "source": "Local Library"}],
            },
            archive_dir=self.arch_dir,
        )
        _, _, pl_list = self._req("/api/playlists")
        self.assertEqual(len(json.loads(pl_list)["archived"]), 1)

        _, ct_csv, csv_body = self._req(f"/api/playlists/{pid}/download?format=csv")
        self.assertIn("text/csv", ct_csv)
        self.assertIn("Song", csv_body)

    def test_new_endpoints_feedback_presets_hygiene_upload_automation_and_edit(self):
        # /api/feedback GET & POST
        st, _, fb_raw = self._req(
            "/api/feedback",
            method="POST",
            body={"track": {"id": "4cOdK2wGLETKBW3PvgPWqT", "name": "Track 1", "artist": "Art 1"}, "rating": "dislike"},
        )
        self.assertEqual(st, 200)
        self.assertIn("4cOdK2wGLETKBW3PvgPWqT", json.loads(fb_raw)["feedback"]["disliked"])

        _, _, fb_get = self._req("/api/feedback")
        self.assertIn("4cOdK2wGLETKBW3PvgPWqT", json.loads(fb_get)["disliked"])

        # /api/presets GET, POST, DELETE
        st_p, _, pr_raw = self._req("/api/presets")
        self.assertEqual(st_p, 200)
        self.assertIn("presets", json.loads(pr_raw))

        _, _, pr_post = self._req(
            "/api/presets",
            method="POST",
            body={"name": "GymMode", "preset": {"avoid_artists": ["Sleepy"], "size": 25}},
        )
        self.assertIn("GymMode", json.loads(pr_post)["presets"])

        _, _, pr_del = self._req("/api/presets/GymMode", method="DELETE")
        self.assertNotIn("GymMode", json.loads(pr_del)["presets"])

        # /api/hygiene & /api/library/search
        st_h, _, hy_raw = self._req("/api/hygiene")
        self.assertEqual(st_h, 200)
        self.assertIn("summary", json.loads(hy_raw))

        st_ls, _, ls_raw = self._req("/api/library/search?q=test")
        self.assertEqual(st_ls, 200)
        self.assertIn("results", json.loads(ls_raw))

        # /api/studio/edit
        self.app_ctx["warm_session"].last_playlist = {
            "playlist_name": "Before Edit",
            "playlist_description": "Old",
            "tracks": [
                {"id": "1111111111111", "name": "A", "artist": "X"},
                {"id": "2222222222222", "name": "B", "artist": "Y"},
            ],
        }
        with patch.object(discovery_engine, "LAST_PL_PATH", os.path.join(self.tmpdir.name, "last.json")):
            st_ed, _, ed_raw = self._req(
                "/api/studio/edit",
                method="POST",
                body={"playlist_name": "After Edit", "order_indices": [1, 0]},
            )
        self.assertEqual(st_ed, 200)
        ed_pl = json.loads(ed_raw)["playlist"]
        self.assertEqual(ed_pl["playlist_name"], "After Edit")
        self.assertEqual([t["id"] for t in ed_pl["tracks"]], ["2222222222222", "1111111111111"])

        # /api/automation GET & POST save
        st_a, _, a_raw = self._req("/api/automation")
        self.assertEqual(st_a, 200)
        self.assertIn("automation", json.loads(a_raw))
        _, _, a_post = self._req(
            "/api/automation",
            method="POST",
            body={"action": "save", "automation": {"day_of_week": "MON", "time": "10:30", "size": 22}},
        )
        self.assertEqual(json.loads(a_post)["automation"]["day_of_week"], "MON")

        # /vendor/plotly.min.js serving when present
        vendor_js = os.path.join(self.tmpdir.name, "plotly.min.js")
        with open(vendor_js, "w", encoding="utf-8") as vf:
            vf.write("window.Plotly = { version: 'local-test' };")
        with patch.object(web_app, "find_local_plotly_js", return_value=vendor_js):
            st_v, ct_v, v_body = self._req("/vendor/plotly.min.js")
            self.assertEqual(st_v, 200)
            self.assertIn("javascript", ct_v)
            self.assertIn("local-test", v_body)

        # /api/taste/export-card & /api/clusters/rename using fake BASE_DIR artifacts
        s2_dir = os.path.join(self.tmpdir.name, "02_Vector_Core")
        os.makedirs(s2_dir, exist_ok=True)
        utils.atomic_write_json(
            os.path.join(s2_dir, "taste_profile_summary.json"),
            {
                "generated_at": "2026-06-01T00:00:00Z",
                "master_taste_centroid": [0.5, 0.5],
                "clusters": [
                    {
                        "cluster_id": 0,
                        "vibe_label": "Auto Label",
                        "track_count": 10,
                        "weighted_share_pct": 100.0,
                        "top_genres": ["indie"],
                        "centroid_vector": [0.5, 0.5],
                    }
                ],
            },
        )
        utils.atomic_write_json(
            os.path.join(s2_dir, "taste_map_data.json"),
            {"clusters": [{"cluster_id": 0, "vibe_label": "Auto Label"}], "tracks": []},
        )
        orig_base = self.app_ctx["base_dir"]
        self.app_ctx["base_dir"] = self.tmpdir.name
        try:
            st_c, _, c_raw = self._req("/api/taste/export-card")
            self.assertEqual(st_c, 200)
            card = json.loads(c_raw)
            self.assertEqual(card["clusters"][0]["vibe_label"], "Auto Label")

            st_rn, _, rn_raw = self._req(
                "/api/clusters/rename",
                method="POST",
                body={"cluster_id": 0, "label": "Night Drive"},
            )
            self.assertEqual(st_rn, 200)
            self.assertEqual(json.loads(rn_raw)["cluster_labels"]["0"], "Night Drive")
            updated_sum = utils.load_json_safe(os.path.join(s2_dir, "taste_profile_summary.json"))
            self.assertEqual(updated_sum["clusters"][0]["vibe_label"], "Night Drive")

            # /api/history/upload (valid .json and Zip-Slip .zip)
            st_up, _, up_raw = self._req(
                "/api/history/upload",
                method="POST",
                raw_body=b"[]",
                headers={"X-Filename": "Streaming_History_Audio_2026.json"},
            )
            self.assertEqual(st_up, 200)
            self.assertEqual(json.loads(up_raw)["saved_files"], ["Streaming_History_Audio_2026.json"])

            evil_buf = io.BytesIO()
            with zipfile.ZipFile(evil_buf, "w") as zf:
                zf.writestr("../escape.json", b"[]")
            with self.assertRaises(urllib.error.HTTPError) as ctx_slip:
                self._req(
                    "/api/history/upload",
                    method="POST",
                    raw_body=evil_buf.getvalue(),
                    headers={"X-Filename": "evil.zip"},
                )
            try:
                self.assertEqual(ctx_slip.exception.code, 400)
            finally:
                ctx_slip.exception.close()
        finally:
            self.app_ctx["base_dir"] = orig_base


class TestSpotifyAndReccoBeatsCompliance(unittest.TestCase):
    def test_playlist_limit_clamped_and_fallback_uris_payload(self):
        sp = MagicMock()
        sp._get.return_value = {"items": [], "next": None}
        utils.fetch_playlist_items_2026(sp, "pl123", limit=100)
        sp._get.assert_called_once_with("playlists/pl123/items", limit=50)

        sp.playlist_add_items.side_effect = AttributeError("missing")
        utils.add_items_to_spotify_playlist(sp, "pl123", ["spotify:track:4cOdK2wGLETKBW3PvgPWqT"])
        sp._post.assert_called_once_with(
            "playlists/pl123/items",
            payload={"uris": ["spotify:track:4cOdK2wGLETKBW3PvgPWqT"]},
        )

    def test_popularity_omitted_in_dev_mode_does_not_drop_tracks(self):
        # Feb 2026 Dev Mode strips popularity -> None; tracks must NOT be filtered out
        self.assertTrue(
            discovery_engine.passes_popularity_filter(None, deep_cuts=False, raw_track_count=50)
        )
        self.assertTrue(
            discovery_engine.passes_popularity_filter(None, deep_cuts=True, raw_track_count=50)
        )
        # Numeric popularity values still filter when actually present
        self.assertFalse(
            discovery_engine.passes_popularity_filter(5, deep_cuts=False, raw_track_count=50)
        )
        self.assertTrue(
            discovery_engine.passes_popularity_filter(40, deep_cuts=True, relaxed_af=False, raw_track_count=50)
        )
        self.assertFalse(
            discovery_engine.passes_popularity_filter(90, deep_cuts=True, relaxed_af=False, raw_track_count=50)
        )

        # Stage 1 normalize helpers preserve None instead of inventing 0 popularity
        norm_t = stage1_extractor.normalize_track(
            {"id": "4cOdK2wGLETKBW3PvgPWqT", "name": "Song", "artists": [{"id": "0gxyHStUsqpMadRV0Di1Qt", "name": "Art"}]}
        )
        self.assertIsNone(norm_t["popularity"])

    def test_single_artist_genre_lookup_avoids_removed_batch_endpoint_and_bails_early(self):
        sp = MagicMock()
        # Simulate Dev Mode stripping artist genres -> empty list returned on first 3 probes
        sp.artist.return_value = {"id": "0gxyHStUsqpMadRV0Di1Qt", "genres": []}
        raw_tracks = [
            {"artists": [{"id": f"0gxyHStUsqpMadRV0Di1Q{i}", "name": f"Art {i}"}]}
            for i in range(10)
        ]
        ws = discovery_engine.WarmDiscoverySession()
        res = ws._enrich_artist_genres_for_discovery(raw_tracks, sp)
        sp.artists.assert_not_called()
        self.assertEqual(sp.artist.call_count, 3)
        self.assertEqual(len(res), 3)

    def test_reccobeats_handles_non_json_200_gracefully(self):
        bad_resp = MagicMock(status_code=200)
        bad_resp.json.side_effect = ValueError("Invalid JSON")
        session = MagicMock()
        session.get.return_value = bad_resp

        with patch.object(reccobeats_client.time, "sleep", return_value=None):
            items, ok = reccobeats_client._fetch_batch_with_retry(
                session, ["4cOdK2wGLETKBW3PvgPWqT"], max_retries=1
            )
        self.assertEqual(items, [])
        self.assertFalse(ok)

    def test_reccobeats_track_metadata_and_recommendations(self):
        track_resp = MagicMock(status_code=200)
        track_resp.json.return_value = {
            "content": [
                {
                    "id": "uuid-t1",
                    "href": "https://open.spotify.com/track/4cOdK2wGLETKBW3PvgPWqT",
                    "trackTitle": "Never Gonna Give You Up",
                    "durationMs": 213000,
                    "isrc": "GBARL9300135",
                    "popularity": 74,
                    "artists": [{"name": "Rick Astley"}],
                }
            ]
        }
        rec_resp = MagicMock(status_code=200)
        rec_resp.json.return_value = {
            "content": [
                {
                    "id": "uuid-rec1",
                    "href": "https://open.spotify.com/track/0VjIjW4GlUZAMYd2vXMi3b",
                    "trackTitle": "Blinding Lights",
                    "durationMs": 200040,
                    "isrc": "USUG11904206",
                    "popularity": 42,
                    "artists": [
                        {
                            "id": "uuid-a1",
                            "name": "The Weeknd",
                            "href": "https://open.spotify.com/artist/1Xyo4u8uXC1ZmMpatF05PJ",
                        }
                    ],
                }
            ]
        }
        session = MagicMock()
        session.get.side_effect = [track_resp, rec_resp]

        meta = reccobeats_client.fetch_track_metadata_batch(
            ["4cOdK2wGLETKBW3PvgPWqT", "bad_id"], session=session
        )
        self.assertEqual(meta["4cOdK2wGLETKBW3PvgPWqT"]["popularity"], 74)
        self.assertEqual(meta["4cOdK2wGLETKBW3PvgPWqT"]["isrc"], "GBARL9300135")

        recs = reccobeats_client.fetch_reccobeats_recommendations(
            seed_ids=["4cOdK2wGLETKBW3PvgPWqT"],
            negative_seed_ids=["4u7EnebtmKWzUH433cf5Qv"],
            size=15,
            popularity=40,
            target_bpm=130.0,
            target_energy=0.8,
            session=session,
        )
        self.assertEqual(len(recs), 1)
        self.assertEqual(recs[0]["id"], "0VjIjW4GlUZAMYd2vXMi3b")
        self.assertEqual(recs[0]["artists"][0]["id"], "1Xyo4u8uXC1ZmMpatF05PJ")
        self.assertEqual(recs[0]["popularity"], 42)
        self.assertTrue(recs[0]["_from_reccobeats_rec"])
        rec_params = session.get.call_args.kwargs["params"]
        self.assertEqual(rec_params["negativeSeeds"], "4u7EnebtmKWzUH433cf5Qv")
        self.assertEqual(rec_params["popularity"], 40)

    def test_unified_library_queue_playback_and_release_radar(self):
        sp = MagicMock()
        uris = [f"4cOdK2wGLETKBW3Pvg{idx:04d}" for idx in range(45)]
        save_res = utils.save_to_spotify_library(sp, uris)
        self.assertEqual(save_res["count"], 45)
        self.assertEqual(sp._put.call_count, 2)  # 40 + 5 chunking
        first_uris_arg = sp._put.call_args_list[0].kwargs["uris"]
        self.assertTrue(first_uris_arg.startswith("spotify:track:"))
        self.assertEqual(len(first_uris_arg.split(",")), 40)

        # 404 fallback on DELETE /me/library -> current_user_saved_tracks_delete
        sp._delete.side_effect = RuntimeError("404 Not Found")
        utils.remove_from_spotify_library(sp, ["spotify:track:4cOdK2wGLETKBW3PvgPWqT"])
        sp.current_user_saved_tracks_delete.assert_called_once_with(["4cOdK2wGLETKBW3PvgPWqT"])

        # Queue & Start playback
        q_res = utils.queue_tracks_on_spotify(sp, ["spotify:track:4cOdK2wGLETKBW3PvgPWqT"], max_items=10)
        self.assertEqual(q_res["queued"], 1)
        sp.add_to_queue.assert_called_once()

        p_res = utils.start_spotify_playback(sp, ["4cOdK2wGLETKBW3PvgPWqT"])
        self.assertEqual(p_res, {"started": True, "count": 1})
        sp.start_playback.assert_called_once()

        sp.queue.return_value = {
            "currently_playing": {"id": "trkNow1234567890123456", "name": "Now Song", "artists": [{"name": "A"}]},
            "queue": [{"id": "trkQ112345678901234567", "name": "Queued Song", "artists": [{"name": "B"}]}],
        }
        cur_q = utils.get_user_playback_queue(sp)
        self.assertEqual(cur_q["currently_playing"]["name"], "Now Song")
        self.assertEqual(len(cur_q["queue"]), 1)

        # Personal Release Radar
        now = datetime(2026, 11, 20, tzinfo=timezone.utc)
        sp.artist_albums.return_value = {
            "items": [
                {"id": "albRecent", "name": "Fresh Drop", "album_type": "single", "release_date": "2026-11-10", "artists": [{"name": "Band"}]},
                {"id": "albOld", "name": "Old Album", "album_type": "album", "release_date": "2020-01-01", "artists": [{"name": "Band"}]},
            ]
        }
        sp.album_tracks.return_value = {
            "items": [{"id": "4cOdK2wGLETKBW3PvgPWqT", "name": "Fresh Single", "artists": [{"name": "Band"}]}]
        }
        radar = utils.fetch_personal_release_radar(sp, ["0gxyHStUsqpMadRV0Di1Qt"], days=45, now=now)
        self.assertEqual(len(radar["releases"]), 1)
        self.assertEqual(radar["tracks"][0]["album"], "Fresh Drop")

    def test_stage1_and_vectorizer_include_saved_albums_and_followed_artists(self):
        raw_data = {
            "artists": {"short_term": [], "medium_term": [], "long_term": []},
            "tracks": {"short_term": [], "medium_term": [], "long_term": []},
            "recently_played": [],
            "short_term_artists_top_tracks": [],
            "liked_songs": [],
            "followed_artists": [{"id": "0gxyHStUsqpMadRV0Di1Qt", "name": "Rick Astley", "genres": ["synthpop"]}],
            "saved_albums": [
                {
                    "id": "alb123",
                    "name": "Whenever You Need Somebody",
                    "release_date": "1987-11-16",
                    "tracks": [
                        {
                            "id": "4cOdK2wGLETKBW3PvgPWqT",
                            "name": "Never Gonna Give You Up",
                            "artists": [{"id": "0gxyHStUsqpMadRV0Di1Qt", "name": "Rick Astley"}],
                            "album": {"id": "alb123", "name": "Whenever You Need Somebody", "release_date": "1987-11-16"},
                        }
                    ],
                }
            ],
            "playlists": [],
            "artist_genre_map": {"0gxyHStUsqpMadRV0Di1Qt": ["synthpop"]},
        }
        prof = stage1_extractor.synthesize_profile({"id": "u1", "display_name": "User"}, raw_data)
        self.assertEqual(prof["taste_summary"]["total_followed_artists"], 1)
        self.assertEqual(prof["taste_summary"]["total_saved_albums"], 1)
        self.assertEqual(prof["taste_summary"]["total_unique_tracks_analyzed"], 1)

        docs, track_ids, meta = npu_vectorizer.build_semantic_documents(prof)
        self.assertEqual(track_ids, ["4cOdK2wGLETKBW3PvgPWqT"])
        self.assertIn("synthpop", docs[0])
        self.assertEqual(meta["4cOdK2wGLETKBW3PvgPWqT"]["year"], "1987")

    def test_new_spotify_2026_http_endpoints(self):
        tmpdir = tempfile.TemporaryDirectory()
        try:
            prefs_file = os.path.join(tmpdir.name, "prefs.json")
            fb_file = os.path.join(tmpdir.name, "fb.json")
            with (
                patch.object(config, "USER_PREFS_FILE", prefs_file),
                patch.object(config, "USER_FEEDBACK_FILE", fb_file),
            ):
                server, app_ctx = web_app.create_server(host="127.0.0.1", port=0)
                port = server.server_address[1]
                th = threading.Thread(target=server.serve_forever, daemon=True)
                th.start()
                try:
                    mock_sp = MagicMock()
                    mock_sp.queue.return_value = {"currently_playing": None, "queue": []}
                    app_ctx["warm_session"].sp = mock_sp
                    app_ctx["warm_session"].last_playlist = {
                        "playlist_name": "Playback Test",
                        "tracks": [{"id": "4cOdK2wGLETKBW3PvgPWqT", "name": "Song", "artist": "Art"}],
                    }

                    def call(path, method="GET", body=None):
                        data = json.dumps(body).encode("utf-8") if body is not None else None
                        hdrs = {"Content-Type": "application/json"} if body is not None else {}
                        req = urllib.request.Request(f"http://127.0.0.1:{port}{path}", data=data, headers=hdrs, method=method)
                        with urllib.request.urlopen(req, timeout=5) as resp:
                            return resp.status, json.loads(resp.read().decode("utf-8"))

                    st_q, q_body = call("/api/spotify/queue")
                    self.assertEqual(st_q, 200)
                    self.assertEqual(q_body["queue"], [])

                    st_sq, sq_body = call("/api/studio/queue", method="POST", body={"max_items": 5})
                    self.assertEqual(st_sq, 200)
                    self.assertEqual(sq_body["queued"], 1)

                    st_sp, sp_body = call("/api/studio/play", method="POST", body={})
                    self.assertEqual(st_sp, 200)
                    self.assertTrue(sp_body["started"])

                    st_lib, lib_body = call(
                        "/api/spotify/library",
                        method="POST",
                        body={"action": "save", "uris": ["4cOdK2wGLETKBW3PvgPWqT"]},
                    )
                    self.assertEqual(st_lib, 200)
                    self.assertEqual(lib_body["count"], 1)

                    # Pref sync_likes_to_spotify triggers PUT /me/library on 👍
                    call("/api/preferences", method="POST", body={"preferences": {"sync_likes_to_spotify": True}})
                    st_fb, fb_body = call(
                        "/api/feedback",
                        method="POST",
                        body={"track": {"id": "4cOdK2wGLETKBW3PvgPWqT", "name": "Song", "artist": "Art"}, "rating": "like"},
                    )
                    self.assertEqual(st_fb, 200)
                    self.assertTrue(fb_body["spotify_synced"])
                finally:
                    app_ctx["stop_event"].set()
                    server.shutdown()
                    server.server_close()
                    th.join(timeout=2)
        finally:
            tmpdir.cleanup()


class TestYouTubeMusicAndUnifiedStudio(unittest.TestCase):
    def test_ytm_id_helpers_and_provider_detection(self):
        self.assertTrue(utils.is_valid_ytm_video_id("dQw4w9WgXcQ"))
        self.assertFalse(utils.is_valid_ytm_video_id("too_short"))
        self.assertTrue(utils.is_valid_ytm_id("yt:dQw4w9WgXcQ"))
        self.assertFalse(utils.is_valid_ytm_id("dQw4w9WgXcQ"))
        self.assertEqual(utils.extract_ytm_video_id("yt:dQw4w9WgXcQ"), "dQw4w9WgXcQ")
        self.assertEqual(utils.extract_ytm_video_id("https://music.youtube.com/watch?v=dQw4w9WgXcQ&list=RD"), "dQw4w9WgXcQ")
        self.assertEqual(utils.extract_ytm_video_id("https://youtu.be/dQw4w9WgXcQ"), "dQw4w9WgXcQ")
        self.assertEqual(utils.get_track_provider("yt:dQw4w9WgXcQ"), "ytmusic")
        self.assertEqual(utils.get_track_provider("4cOdK2wGLETKBW3PvgPWqT"), "spotify")
        self.assertEqual(
            utils.get_track_provider({"id": "4cOdK2wGLETKBW3PvgPWqT", "video_id": "dQw4w9WgXcQ"}),
            "both",
        )

    def test_ytmusic_client_auth_normalize_search_and_playlists(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            b_file = os.path.join(tmpdir, "ytm_browser.json")
            o_file = os.path.join(tmpdir, "ytm_oauth.json")
            oc_file = os.path.join(tmpdir, ".ytm_oauth_client.json")
            with (
                patch.object(config, "YTMUSIC_BROWSER_AUTH_FILE", b_file),
                patch.object(config, "YTMUSIC_OAUTH_FILE", o_file),
                patch.object(config, "YTMUSIC_OAUTH_CLIENT_FILE", oc_file),
            ):
                st0 = ytmusic_client.get_ytm_auth_status()
                self.assertFalse(st0["configured"])

                ytmusic_client.save_ytm_browser_headers("Cookie: SAPISID=abc\nAuthorization: SAPISIDHASH 123_abc\nX-Goog-AuthUser: 0")
                st1 = ytmusic_client.get_ytm_auth_status()
                self.assertTrue(st1["configured"])
                self.assertEqual(st1["method"], "browser")

                ytmusic_client.save_ytm_oauth_client("cid-123", "sec-456")
                mock_http = MagicMock()
                r_dev = MagicMock(status_code=200)
                r_dev.json.return_value = {
                    "device_code": "dev1",
                    "user_code": "ABCD-1234",
                    "verification_url": "https://www.google.com/device",
                    "expires_in": 1800,
                    "interval": 5,
                }
                r_tok = MagicMock(status_code=200)
                r_tok.json.return_value = {
                    "access_token": "tok-abc",
                    "refresh_token": "ref-xyz",
                    "expires_in": 3600,
                    "token_type": "Bearer",
                }
                mock_http.post.side_effect = [r_dev, r_tok]

                dev = ytmusic_client.start_ytm_oauth_device_flow(http_session=mock_http)
                self.assertEqual(dev["user_code"], "ABCD-1234")
                done = ytmusic_client.complete_ytm_oauth_device_flow("dev1", http_session=mock_http)
                self.assertTrue(done["ok"])
                self.assertTrue(os.path.exists(o_file))

                norm = ytmusic_client.normalize_ytm_track(
                    {
                        "videoId": "abc123DEF45",
                        "title": "Hidden Track",
                        "artists": [{"name": "Artist One"}],
                        "album": {"name": "Rare Tape"},
                        "year": "2021",
                        "duration_seconds": 210,
                    }
                )
                self.assertEqual(norm["id"], "yt:abc123DEF45")
                self.assertEqual(norm["provider"], "ytmusic")
                self.assertEqual(norm["album"]["release_date"], "2021")

                mock_ytm = MagicMock()
                mock_ytm.search.return_value = [
                    {
                        "videoId": "remx9999999",
                        "title": "Hidden Track (Club Remix)",
                        "artists": [{"name": "Artist One"}],
                    },
                    {
                        "videoId": "orig1111111",
                        "title": "Hidden Track",
                        "artists": [{"name": "Artist One"}],
                    },
                ]
                hit = ytmusic_client.search_ytm_track(mock_ytm, "Hidden Track", "Artist One")
                self.assertIsNotNone(hit)
                self.assertEqual(hit["video_id"], "orig1111111")

                mock_ytm.create_playlist.return_value = "PL_YTM_123"
                created = ytmusic_client.create_or_update_ytm_playlist(
                    mock_ytm,
                    name="My Mix",
                    description="Test",
                    video_ids=["orig1111111"],
                )
                self.assertEqual(created["playlist_id"], "PL_YTM_123")
                self.assertEqual(created["count"], 1)

                mock_ytm.get_playlist.return_value = {"tracks": [{"videoId": "orig1111111", "setVideoId": "sv1"}]}
                updated = ytmusic_client.create_or_update_ytm_playlist(
                    mock_ytm,
                    name="My Mix",
                    video_ids=["orig1111111", "newVid22222"],
                    target_playlist_id="PL_YTM_123",
                )
                self.assertEqual(updated["count"], 2)
                mock_ytm.remove_playlist_items.assert_called_once_with(
                    "PL_YTM_123", [{"videoId": "orig1111111", "setVideoId": "sv1"}]
                )
                mock_ytm.add_playlist_items.assert_called_once_with(
                    "PL_YTM_123", ["orig1111111", "newVid22222"], duplicates=False
                )

                rate_res = ytmusic_client.rate_ytm_song(mock_ytm, "orig1111111", "LIKE")
                self.assertTrue(rate_res["ok"])
                mock_ytm.rate_song.assert_called_once_with("orig1111111", "LIKE")

    def test_reccobeats_search_enrichment_and_cache(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_path = os.path.join(tmpdir, ".reccobeats_cache.json")
            mock_sess = MagicMock()
            r_search = MagicMock(status_code=200)
            r_search.json.return_value = {
                "content": [
                    {
                        "id": "rb-uuid-1",
                        "trackTitle": "Midnight City",
                        "artists": [{"name": "M83"}],
                        "href": "https://open.spotify.com/track/6GyFP1nfCDB8lbD2bG0Hq9",
                    }
                ]
            }
            r_batch = MagicMock(status_code=200)
            r_batch.json.return_value = {
                "content": [
                    {
                        "id": "rb-uuid-1",
                        "href": "https://open.spotify.com/track/6GyFP1nfCDB8lbD2bG0Hq9",
                        "tempo": 105.0,
                        "energy": 0.72,
                        "valence": 0.45,
                        "danceability": 0.61,
                        "acousticness": 0.04,
                        "key": 11,
                        "mode": 1,
                    }
                ]
            }
            mock_sess.get.side_effect = [r_search, r_batch]

            tracks = [{"id": "yt:dX3k_QDnzHE", "name": "Midnight City", "artists": [{"name": "M83"}]}]
            af_map = reccobeats_client.fetch_audio_features_by_search(tracks, cache_path=cache_path, session=mock_sess)
            self.assertIn("yt:dX3k_QDnzHE", af_map)
            self.assertEqual(af_map["yt:dX3k_QDnzHE"]["bpm"], 105.0)
            self.assertTrue(utils.is_real_audio_features(af_map["yt:dX3k_QDnzHE"]))

            # Second call hits local cache with 0 network calls
            mock_sess.get.reset_mock()
            af_map2 = reccobeats_client.fetch_audio_features_by_search(tracks, cache_path=cache_path, session=mock_sess)
            self.assertEqual(af_map2["yt:dX3k_QDnzHE"]["bpm"], 105.0)
            mock_sess.get.assert_not_called()

    def test_google_takeout_history_importer_and_zip_upload(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            watch_data = [
                {
                    "header": "YouTube Music",
                    "title": "Watched Midnight City",
                    "titleUrl": "https://music.youtube.com/watch?v=dX3k_QDnzHE",
                    "subtitles": [{"name": "M83 - Topic"}],
                    "time": "2026-03-10T22:15:00Z",
                },
                {
                    "header": "YouTube",
                    "title": "Watched Python Tutorial",
                    "titleUrl": "https://www.youtube.com/watch?v=aaaaaaaaaaa",
                    "subtitles": [{"name": "Dev Channel"}],
                    "time": "2026-03-10T23:00:00Z",
                },
            ]
            watch_path = os.path.join(tmpdir, "watch-history.json")
            with open(watch_path, "w", encoding="utf-8") as f:
                json.dump(watch_data, f)

            stats, summary = history_importer.import_history_with_stats(tmpdir)
            self.assertEqual(summary["valid_streams"], 1)
            self.assertIn("yt:dX3k_QDnzHE", stats)
            self.assertEqual(stats["yt:dX3k_QDnzHE"]["track_name"], "Midnight City")
            self.assertEqual(stats["yt:dX3k_QDnzHE"]["artist_name"], "M83")
            self.assertEqual(stats["yt:dX3k_QDnzHE"]["provider"], "ytmusic")
            self.assertNotIn("yt:aaaaaaaaaaa", stats)

            buf = io.BytesIO()
            with zipfile.ZipFile(buf, "w") as zf:
                zf.writestr("Takeout/YouTube and YouTube Music/history/watch-history.json", json.dumps(watch_data))
                zf.writestr("Takeout/YouTube/other.txt", "skip")
            ext_dest = os.path.join(tmpdir, "ext")
            res = utils.extract_uploaded_history_bytes("takeout.zip", buf.getvalue(), dest_dir=ext_dest)
            self.assertEqual(len(res["saved_files"]), 1)
            self.assertTrue(res["saved_files"][0].endswith("watch-history.json"))

    def test_merged_profile_and_vectorizer_unified_dedup(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            sp_profile = {
                "tracks": {
                    "short_term": [
                        {
                            "id": "6GyFP1nfCDB8lbD2bG0Hq9",
                            "name": "Midnight City",
                            "artists": [{"id": "a1", "name": "M83", "genres": ["synthpop"]}],
                            "album": {"release_date": "2011-10-18"},
                        }
                    ],
                    "medium_term": [],
                    "long_term": [],
                },
                "playlists": [],
                "saved_albums": [],
                "recently_played": [],
                "short_term_artists_top_tracks": [],
                "liked_songs_sample": [],
                "audio_features": {
                    "6GyFP1nfCDB8lbD2bG0Hq9": {
                        "bpm": 105.0,
                        "energy": 0.72,
                        "valence": 0.45,
                        "danceability": 0.61,
                        "acousticness": 0.04,
                    }
                },
            }
            ytm_profile = {
                "tracks": {
                    "short_term": [
                        {
                            "id": "yt:dX3k_QDnzHE",
                            "video_id": "dX3k_QDnzHE",
                            "provider": "ytmusic",
                            "name": "Midnight City",
                            "artists": [{"name": "M83"}],
                            "album": {"release_date": "2011-01-01"},
                        },
                        {
                            "id": "yt:rareExcl123",
                            "video_id": "rareExcl123",
                            "provider": "ytmusic",
                            "name": "Unlisted Demo (2009)",
                            "artists": [{"name": "Secret Artist"}],
                            "album": {"release_date": "2009-01-01"},
                        },
                    ],
                    "medium_term": [],
                    "long_term": [],
                },
                "playlists": [],
                "saved_albums": [],
                "recently_played": [],
                "short_term_artists_top_tracks": [],
                "liked_songs_sample": [],
                "audio_features": {},
            }
            utils.atomic_write_json(os.path.join(tmpdir, "user_music_profile.json"), sp_profile)
            utils.atomic_write_json(os.path.join(tmpdir, "ytmusic_profile.json"), ytm_profile)

            merged = utils.load_merged_music_profile(stage1_dir=tmpdir, provider="unified")
            self.assertEqual(merged["providers"], ["spotify", "ytmusic"])

            docs, track_ids, meta = npu_vectorizer.build_semantic_documents(merged)
            self.assertEqual(len(docs), 2)
            self.assertIn("6GyFP1nfCDB8lbD2bG0Hq9", track_ids)
            self.assertIn("yt:rareExcl123", track_ids)
            self.assertNotIn("yt:dX3k_QDnzHE", track_ids)
            self.assertEqual(meta["6GyFP1nfCDB8lbD2bG0Hq9"]["provider"], "both")
            self.assertEqual(meta["6GyFP1nfCDB8lbD2bG0Hq9"]["video_id"], "dX3k_QDnzHE")
            self.assertEqual(meta["yt:rareExcl123"]["provider"], "ytmusic")

    def test_cross_platform_resolution_and_transfer(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            prefs_file = os.path.join(tmpdir, "prefs.json")
            last_pl_file = os.path.join(tmpdir, "last.json")
            arch_dir = os.path.join(tmpdir, "history")
            with (
                patch.object(config, "USER_PREFS_FILE", prefs_file),
                patch.object(config, "PLAYLIST_HISTORY_DIR", arch_dir),
                patch.object(discovery_engine, "LAST_PL_PATH", last_pl_file),
            ):
                ws = discovery_engine.WarmDiscoverySession()
                ws.last_playlist = {
                    "playlist_name": "Unified Cross Mix",
                    "playlist_description": "Test transfer",
                    "tracks": [
                        {"id": "6GyFP1nfCDB8lbD2bG0Hq9", "name": "Midnight City", "artist": "M83", "provider": "spotify"},
                        {"id": "yt:rareExcl123", "video_id": "rareExcl123", "name": "Unlisted Demo", "artist": "Secret Artist", "provider": "ytmusic"},
                    ],
                }

                mock_sp = MagicMock()
                mock_sp.current_user.return_value = {"id": "u1"}
                mock_sp_pl = {"id": "sppl999", "external_urls": {"spotify": "https://open.spotify.com/playlist/sppl999"}}
                mock_sp.current_user_playlist_create.return_value = mock_sp_pl
                mock_sp.user_playlist_create.return_value = mock_sp_pl
                # Searching "Unlisted Demo" returns empty -> reported unmatched
                mock_sp.search.return_value = {"tracks": {"items": []}}
                ws.sp = mock_sp

                mock_ytm = MagicMock()
                mock_ytm.search.side_effect = [
                    [{"videoId": "dX3k_QDnzHE", "title": "Midnight City", "artists": [{"name": "M83"}]}],
                ]
                mock_ytm.create_playlist.return_value = "PL_YTM_OUT"
                ws.ytm = mock_ytm

                to_sp = ws.transfer_playlist(target_provider="spotify")
                self.assertEqual(to_sp["matched_count"], 1)
                self.assertEqual(to_sp["unmatched_count"], 1)
                self.assertEqual(to_sp["unmatched"][0]["name"], "Unlisted Demo")
                self.assertEqual(to_sp["url"], "https://open.spotify.com/playlist/sppl999")

                # export_to_spotify skips yt:<vid> track and keeps valid Spotify ID
                ok_sp = discovery_engine.export_to_spotify(ws.last_playlist, sp=mock_sp)
                self.assertEqual(ok_sp, "https://open.spotify.com/playlist/sppl999")

                to_ytm = ws.transfer_playlist(target_provider="ytmusic")
                self.assertEqual(to_ytm["matched_count"], 2)
                self.assertEqual(to_ytm["unmatched_count"], 0)
                self.assertEqual(to_ytm["url"], "https://music.youtube.com/playlist?list=PL_YTM_OUT")

    def test_ytm_and_unified_web_endpoints(self):
        tmpdir = tempfile.TemporaryDirectory()
        try:
            prefs_file = os.path.join(tmpdir.name, "prefs.json")
            fb_file = os.path.join(tmpdir.name, "fb.json")
            b_file = os.path.join(tmpdir.name, "ytm_browser.json")
            o_file = os.path.join(tmpdir.name, "ytm_oauth.json")
            oc_file = os.path.join(tmpdir.name, ".ytm_oauth_client.json")
            with (
                patch.object(config, "USER_PREFS_FILE", prefs_file),
                patch.object(config, "USER_FEEDBACK_FILE", fb_file),
                patch.object(config, "YTMUSIC_BROWSER_AUTH_FILE", b_file),
                patch.object(config, "YTMUSIC_OAUTH_FILE", o_file),
                patch.object(config, "YTMUSIC_OAUTH_CLIENT_FILE", oc_file),
            ):
                server, app_ctx = web_app.create_server(host="127.0.0.1", port=0)
                port = server.server_address[1]
                th = threading.Thread(target=server.serve_forever, daemon=True)
                th.start()
                try:
                    def call(path, method="GET", body=None):
                        data = json.dumps(body).encode("utf-8") if body is not None else None
                        hdrs = {"Content-Type": "application/json"} if body is not None else {}
                        req = urllib.request.Request(f"http://127.0.0.1:{port}{path}", data=data, headers=hdrs, method=method)
                        with urllib.request.urlopen(req, timeout=5) as resp:
                            return resp.status, json.loads(resp.read().decode("utf-8"))

                    st0, a0 = call("/api/ytmusic/auth-status")
                    self.assertEqual(st0, 200)
                    self.assertFalse(a0["configured"])

                    st_br, br = call(
                        "/api/ytmusic/auth/browser",
                        method="POST",
                        body={"headers": "Cookie: SAPISID=xyz\nAuthorization: SAPISIDHASH 1_2\nX-Goog-AuthUser: 0"},
                    )
                    self.assertEqual(st_br, 200)
                    self.assertTrue(br["ok"])
                    self.assertEqual(br["auth"]["method"], "browser")

                    st_pr, _ = call("/api/preferences", method="POST", body={"preferences": {"active_provider": "unified", "sync_likes_to_spotify": True}})
                    self.assertEqual(st_pr, 200)
                    st_stat, stat = call("/api/status")
                    self.assertEqual(st_stat, 200)
                    self.assertEqual(stat["active_provider"], "unified")
                    self.assertTrue(stat["auth"]["ytmusic"]["configured"])

                    mock_ytm = MagicMock()
                    mock_ytm.get_library_playlists.return_value = [{"playlistId": "PL_99", "title": "YTM Picks", "count": 12}]
                    mock_ytm.create_playlist.return_value = "PL_NEW_YTM"
                    app_ctx["warm_session"].ytm = mock_ytm
                    app_ctx["warm_session"].last_playlist = {
                        "playlist_name": "YTM Studio Test",
                        "tracks": [
                            {"id": "yt:rareExcl123", "video_id": "rareExcl123", "name": "Exclusive Track", "artist": "Indies", "provider": "ytmusic"},
                            {"id": "6GyFP1nfCDB8lbD2bG0Hq9", "name": "Missing On YTM", "artist": "NoMatch", "provider": "spotify"},
                        ],
                    }

                    st_pl, pl_body = call("/api/ytmusic/playlists")
                    self.assertEqual(st_pl, 200)
                    self.assertEqual(pl_body["playlists"][0]["id"], "PL_99")

                    with (
                        patch.object(ytmusic_client, "start_ytm_oauth_device_flow", return_value={"device_code": "dc1", "user_code": "XY-99", "verification_url": "https://www.google.com/device"}),
                        patch.object(ytmusic_client, "complete_ytm_oauth_device_flow", return_value={"ok": True, "auth_file": o_file}),
                    ):
                        st_os, os_body = call("/api/ytmusic/auth/oauth/start", method="POST", body={"client_id": "cid", "client_secret": "sec"})
                        self.assertEqual(st_os, 200)
                        self.assertEqual(os_body["user_code"], "XY-99")
                        st_oc, oc_body = call("/api/ytmusic/auth/oauth/complete", method="POST", body={"device_code": "dc1"})
                        self.assertEqual(st_oc, 200)
                        self.assertTrue(oc_body["ok"])

                    app_ctx["warm_session"].ytm = mock_ytm
                    mock_ytm.search.return_value = []
                    st_ey, ey_body = call("/api/playlists/last/export-ytmusic", method="POST", body={"privacy": "PRIVATE"})
                    self.assertEqual(st_ey, 200)
                    self.assertTrue(ey_body["ok"])
                    self.assertEqual(ey_body["matched_count"], 1)

                    mock_ytm.search.return_value = []
                    st_tr, tr_body = call(
                        "/api/playlists/last/transfer",
                        method="POST",
                        body={"target_provider": "ytmusic"},
                    )
                    self.assertEqual(st_tr, 200)
                    self.assertTrue(tr_body["ok"])
                    self.assertEqual(tr_body["matched_count"], 1)
                    self.assertEqual(tr_body["unmatched_count"], 1)

                    st_fb, fb_body = call(
                        "/api/feedback",
                        method="POST",
                        body={"track": {"id": "yt:rareExcl123", "video_id": "rareExcl123", "name": "Exclusive Track", "artist": "Indies"}, "rating": "like"},
                    )
                    self.assertEqual(st_fb, 200)
                    self.assertTrue(fb_body["ytmusic_synced"])
                finally:
                    app_ctx["stop_event"].set()
                    server.shutdown()
                    server.server_close()
                    th.join(timeout=2)
        finally:
            tmpdir.cleanup()

    def test_reuse_spotify_af_and_reccobeats_search_circuit_breaker(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            sp_prof = {
                "tracks": {
                    "short_term": [
                        {
                            "id": "6GyFP1nfCDB8lbD2bG0Hq9",
                            "name": "Midnight City",
                            "artists": [{"name": "M83"}],
                        }
                    ]
                },
                "audio_features": {
                    "6GyFP1nfCDB8lbD2bG0Hq9": {
                        "bpm": 105.0,
                        "energy": 0.72,
                        "valence": 0.45,
                        "danceability": 0.61,
                        "acousticness": 0.04,
                    }
                },
            }
            utils.atomic_write_json(os.path.join(tmpdir, "user_music_profile.json"), sp_prof)
            cands = [
                {"id": "yt:dX3k_QDnzHE", "name": "Midnight City", "artist": "M83"},
                {"id": "yt:rareExcl123", "name": "Unlisted Demo", "artist": "Secret Artist"},
            ]
            reused, remaining = ytmusic_extractor._reuse_spotify_audio_features(cands, stage1_dir=tmpdir)
            self.assertIn("yt:dX3k_QDnzHE", reused)
            self.assertEqual(reused["yt:dX3k_QDnzHE"]["bpm"], 105.0)
            self.assertEqual([c["id"] for c in remaining], ["yt:rareExcl123"])

            # Circuit breaker halts after 8 consecutive failed HTTP responses instead of making all 20 lookups
            cache_path = os.path.join(tmpdir, ".reccobeats_cache.json")
            mock_sess = MagicMock()
            mock_sess.get.return_value = MagicMock(status_code=400, text="Bad Request")
            many_tracks = [{"id": f"yt:vid00000{i:02d}", "name": f"Song {i}", "artist": "Art"} for i in range(20)]
            res = reccobeats_client.fetch_audio_features_by_search(
                many_tracks, cache_path=cache_path, session=mock_sess, max_lookups=20
            )
            self.assertEqual(res, {})
            self.assertEqual(mock_sess.get.call_count, 8)

    def test_npz_context_manager_allows_immediate_replace(self):
        try:
            import numpy as np
        except ImportError:
            self.skipTest("numpy not installed in current interpreter")

        with tempfile.TemporaryDirectory() as tmpdir:
            s2_dir = os.path.join(tmpdir, "02_Vector_Core")
            os.makedirs(s2_dir, exist_ok=True)
            npz_path = os.path.join(s2_dir, "taste_vectors.npz")
            meta_path = os.path.join(s2_dir, "track_metadata.json")
            embs = np.ones((2, 384), dtype=np.float32)
            tids = np.array(["4cOdK2wGLETKBW3PvgPWqT", "yt:dX3k_QDnzHE"])
            np.savez(npz_path, embeddings=embs, track_ids=tids)
            utils.atomic_write_json(meta_path, {"4cOdK2wGLETKBW3PvgPWqT": {"name": "A"}, "yt:dX3k_QDnzHE": {"name": "B"}})

            with patch.object(discovery_engine, "STAGE2_DIR", s2_dir):
                loaded_embs, loaded_ids, _ = discovery_engine.load_user_vectors()
                self.assertEqual(loaded_embs.shape, (2, 384))
                self.assertEqual(len(loaded_ids), 2)

                with np.load(npz_path) as old_data:
                    cached = np.array(old_data["embeddings"])
                tmp_npz = npz_path + ".tmp"
                with open(tmp_npz, "wb") as f:
                    np.savez(f, embeddings=cached, track_ids=tids)
                os.replace(tmp_npz, npz_path)
                self.assertTrue(os.path.exists(npz_path))


class TestTieredLLMAndTasteCurator(unittest.TestCase):
    def test_recommend_tier_for_specs_multi_device(self):
        t1 = utils.recommend_tier_for_specs(8.0, openvino_devices=["CPU"])
        self.assertEqual(t1["recommended_tier"], 1)
        self.assertEqual(t1["recommended_model_key"], "qwen3-1.7b-ov")

        t2 = utils.recommend_tier_for_specs(15.4, openvino_devices=["CPU", "GPU", "NPU"], cpu_name="Intel Core Ultra 7 155U")
        self.assertEqual(t2["recommended_tier"], 2)
        self.assertEqual(t2["recommended_model_key"], "qwen3-4b-ov")
        self.assertEqual(t2["recommended_ov_device"], "GPU")
        self.assertEqual(t2["recommended_backend"], "openvino")

        t3_rtx4050 = utils.recommend_tier_for_specs(
            32.0,
            nvidia_gpu_name="NVIDIA GeForce RTX 4050 Laptop GPU",
            nvidia_vram_gb=6.0,
            openvino_devices=["CPU"],
            cpu_name="AMD Ryzen AI HX",
            ollama_running=True,
        )
        self.assertEqual(t3_rtx4050["recommended_tier"], 3)
        self.assertEqual(t3_rtx4050["recommended_model_key"], "qwen3-8b-ov")
        self.assertEqual(t3_rtx4050["recommended_backend"], "ollama")
        self.assertEqual(t3_rtx4050["recommended_ollama_model"], "qwen3:8b")

        t4_rtx4060 = utils.recommend_tier_for_specs(
            32.0,
            nvidia_gpu_name="NVIDIA GeForce RTX 4060",
            nvidia_vram_gb=8.0,
            openvino_devices=["CPU"],
            ollama_running=False,
        )
        self.assertEqual(t4_rtx4060["recommended_tier"], 4)
        self.assertEqual(t4_rtx4060["recommended_model_key"], "qwen3-8b-ov")
        self.assertIn("RTX 4060", t4_rtx4060["advice"])

    def test_local_llm_client_openvino_ollama_and_think_stripping(self):
        # <think> reasoning tags stripped cleanly before parsing JSON
        extracted = llm_parser.extract_json_object('<think>Considering genres...\n{"bad": 1}</think>\n{"search_queries": ["genre:synthwave"]}')
        self.assertEqual(extracted["search_queries"], ["genre:synthwave"])

        # OpenVINO LocalLLMClient builds chat prompt and attaches StructuredOutputConfig
        fake_tok = MagicMock()
        fake_tok.apply_chat_template.return_value = "<chat>templated</chat>"
        fake_ov_pipe = MagicMock()
        fake_ov_pipe.get_tokenizer.return_value = fake_tok
        fake_ov_pipe.generate.return_value = '{"search_queries": ["genre:synthpop"], "playlist_name": "Neon", "playlist_description": "Test"}'

        client_ov = llm_parser.LocalLLMClient(
            backend="openvino",
            model_key="qwen3-4b-ov",
            label="Qwen3 4B",
            device="GPU",
            pipe=fake_ov_pipe,
            is_qwen3=True,
        )
        out = client_ov.generate_json(
            "System", "User", json_schema=llm_parser._PARSE_PROMPT_SCHEMA, max_new_tokens=150, temperature=0.1
        )
        self.assertIn("genre:synthpop", out)
        fake_tok.apply_chat_template.assert_called_once()
        msgs_arg = fake_tok.apply_chat_template.call_args.args[0]
        self.assertIn("/no_think", msgs_arg[1]["content"])
        gen_cfg_arg = fake_ov_pipe.generate.call_args.args[1]
        if hasattr(gen_cfg_arg, "structured_output_config"):
            self.assertIsNotNone(gen_cfg_arg.structured_output_config)

        # Ollama backend posts /api/chat with think=False and JSON schema format
        client_ol = llm_parser.LocalLLMClient(
            backend="ollama",
            model_key="qwen3:8b",
            label="Ollama: qwen3:8b",
            base_url="http://127.0.0.1:11434",
            ollama_model="qwen3:8b",
        )
        mock_resp = MagicMock(status_code=200)
        mock_resp.json.return_value = {"message": {"content": '{"search_queries": ["genre:ambient"]}'}}
        with patch("requests.post", return_value=mock_resp) as mock_post:
            res_ol = client_ol.generate_json("Sys", "Usr", json_schema={"type": "object"}, max_new_tokens=120, temperature=0.1)
            self.assertIn("genre:ambient", res_ol)
            call_kw = mock_post.call_args.kwargs["json"]
            self.assertEqual(call_kw["model"], "qwen3:8b")
            self.assertIs(call_kw["think"], False)
            self.assertEqual(call_kw["format"], {"type": "object"})

    def test_parse_prompt_taste_context_and_vibe_check_telemetry(self):
        client = llm_parser.LocalLLMClient(backend="ollama", ollama_model="qwen3:8b")
        seen_calls = []

        def fake_gen_json(sys_msg, usr_msg, json_schema=None, max_new_tokens=380, temperature=0.1):
            seen_calls.append((sys_msg, usr_msg))
            if "recommendation" in sys_msg.lower() or "recommended_tracks" in sys_msg:
                return json.dumps(
                    {
                        "search_queries": ["genre:synthwave", "genre:darksynth"],
                        "recommended_tracks": [
                            {"name": "Nightcall", "artist": "Kavinsky"},
                            {"name": "Nightcall", "artist": "Kavinsky"},  # dup filtered
                            {"name": "Blocked Song", "artist": "Drake"},  # avoided filtered
                        ],
                        "playlist_name": "Midnight Drive",
                        "playlist_description": "Analog synth pulse.",
                    }
                )
            return json.dumps(
                {
                    "selected_indices": [1, 0],
                    "playlist_name": "Neon Pulse",
                    "playlist_description": "Tight synth curation.",
                }
            )

        with patch.object(client, "generate_json", side_effect=fake_gen_json):
            parsed = llm_parser.parse_prompt(
                client,
                "late night synth drive",
                avoid_artists=["Drake"],
                taste_context="Top Weighted Artists: Kavinsky, M83",
            )
            self.assertIn("Kavinsky, M83", seen_calls[0][0])
            self.assertEqual(len(parsed["recommended_tracks"]), 1)
            self.assertEqual(parsed["recommended_tracks"][0]["name"], "Nightcall")

            cands = [
                {"name": "Track A", "artist": "Art A", "source": "Local Library", "bpm": 118, "energy": 0.7, "valence": 0.4, "genres": "synthpop"},
                {"name": "Track B", "artist": "Art B", "source": "AI Curator Pick", "bpm": 124, "energy": 0.85, "valence": 0.5, "genres": "synthwave"},
            ]
            vc = llm_parser.vibe_check_playlist(
                client, "late night synth drive", cands, target_size=2, taste_context="Platform: unified"
            )
            self.assertEqual(vc["selected_indices"], [1, 0])
            self.assertIn("124BPM", seen_calls[1][0])
            self.assertIn("Platform: unified", seen_calls[1][0])

    def test_warm_session_taste_context_and_llm_pick_resolution(self):
        try:
            import numpy as np
        except ImportError:
            self.skipTest("numpy not installed")

        with tempfile.TemporaryDirectory() as tmpdir:
            prefs_file = os.path.join(tmpdir, "prefs.json")
            fb_file = os.path.join(tmpdir, "fb.json")
            last_pl = os.path.join(tmpdir, "last.json")
            arch_dir = os.path.join(tmpdir, "history")
            with (
                patch.object(config, "USER_PREFS_FILE", prefs_file),
                patch.object(config, "USER_FEEDBACK_FILE", fb_file),
                patch.object(config, "PLAYLIST_HISTORY_DIR", arch_dir),
                patch.object(discovery_engine, "LAST_PL_PATH", last_pl),
            ):
                ws = discovery_engine.WarmDiscoverySession()
                ws.local_ids = ["4cOdK2wGLETKBW3PvgPWqT"]
                ws.local_embeddings = np.ones((1, 4), dtype=np.float32)
                ws.local_meta = {
                    "4cOdK2wGLETKBW3PvgPWqT": {
                        "name": "Local Anchor",
                        "artists": "M83",
                        "genres": "synthpop",
                        "cluster_id": 0,
                        "recency_weight": 12.0,
                        "audio_features": {"bpm": 105.0, "energy": 0.72, "valence": 0.45, "source": "reccobeats"},
                    }
                }
                ws.local_id_set = {"4cOdK2wGLETKBW3PvgPWqT"}
                ws.local_fps = {utils.get_track_fingerprint("Local Anchor", "M83")}
                ws.master_centroid = np.ones(4, dtype=float)
                ws.clusters = [{"cluster_id": 0, "vibe_label": "Dream Synth", "weighted_share_pct": 100.0, "top_genres": ["synthpop"]}]
                ws._vectors_sig = ws._stage2_sig()

                ctx = ws._build_taste_context(provider="unified")
                self.assertIn("Dream Synth", ctx)
                self.assertIn("M83", ctx)

                # Mock smart LLM pitching 1 real new track + Spotify resolving it as AI Curator Pick
                mock_client = llm_parser.LocalLLMClient(backend="ollama", ollama_model="qwen3:8b", label="Ollama: qwen3:8b")
                ws.pipe = mock_client
                ws.llm_config_sig = ws._current_llm_pref_sig()
                ws.embed_tokenizer = MagicMock()
                ws.embed_model = MagicMock()

                mock_sp = MagicMock()
                picked_sp_track = {
                    "id": "6GyFP1nfCDB8lbD2bG0Hq9",
                    "name": "Midnight City",
                    "artists": [{"id": "a1", "name": "M83"}],
                    "album": {"release_date": "2011-10-18"},
                    "explicit": False,
                }
                mock_sp.search.return_value = {"tracks": {"items": [picked_sp_track]}}
                ws.sp = mock_sp

                def _schema_aware_gen(sys_msg, usr_msg, json_schema=None, max_new_tokens=380, temperature=0.1):
                    if json_schema == llm_parser._VIBE_CHECK_SCHEMA:
                        return json.dumps({"selected_indices": [1, 0], "playlist_name": "Synth Picks", "playlist_description": "Curated."})
                    return json.dumps(
                        {
                            "search_queries": ["genre:synthpop"],
                            "recommended_tracks": [{"name": "Midnight City", "artist": "M83"}],
                            "known_tracks_ratio": 0.3,
                            "playlist_size": 5,
                            "playlist_name": "Synth Picks",
                            "playlist_description": "Curated.",
                        }
                    )

                with (
                    patch.object(ws, "_ensure_vectors_loaded"),
                    patch.object(ws, "warm_up"),
                    patch.object(
                        discovery_engine,
                        "embed_texts",
                        side_effect=lambda texts, *_: np.ones((len(texts), 4), dtype=float),
                    ),
                    patch.object(reccobeats_client, "fetch_reccobeats_recommendations", return_value=[]),
                    patch.object(reccobeats_client, "fetch_track_metadata_batch", return_value={}),
                    patch.object(reccobeats_client, "fetch_audio_features_batch", return_value={"6GyFP1nfCDB8lbD2bG0Hq9": {"bpm": 105.0, "energy": 0.72, "valence": 0.45, "source": "reccobeats"}}),
                    patch.object(mock_client, "generate_json", side_effect=_schema_aware_gen),
                ):
                    res = ws.generate("synth drive", fast=False, size=5, provider="spotify")
                    sources = [t["source"] for t in res["playlist"]["tracks"]]
                    self.assertIn("AI Curator Pick", sources)

    def test_llm_models_probe_ollama_and_hot_unload_http(self):
        tmpdir = tempfile.TemporaryDirectory()
        try:
            prefs_file = os.path.join(tmpdir.name, "prefs.json")
            with patch.object(config, "USER_PREFS_FILE", prefs_file):
                server, app_ctx = web_app.create_server(host="127.0.0.1", port=0)
                port = server.server_address[1]
                th = threading.Thread(target=server.serve_forever, daemon=True)
                th.start()
                try:
                    def call(path, method="GET", body=None):
                        data = json.dumps(body).encode("utf-8") if body is not None else None
                        hdrs = {"Content-Type": "application/json"} if body is not None else {}
                        req = urllib.request.Request(f"http://127.0.0.1:{port}{path}", data=data, headers=hdrs, method=method)
                        with urllib.request.urlopen(req, timeout=5) as resp:
                            return resp.status, json.loads(resp.read().decode("utf-8"))

                    st_m, m_body = call("/api/llm/models?probe_ollama=0")
                    self.assertEqual(st_m, 200)
                    self.assertIn("catalog", m_body)
                    self.assertTrue(any(c["key"] == "qwen3-8b-ov" for c in m_body["catalog"]))

                    with patch.object(web_app, "probe_ollama_server", return_value={"running": True, "models": ["qwen3:8b"], "base_url": "http://127.0.0.1:11434"}):
                        st_ol, ol_body = call("/api/llm/probe-ollama", method="POST", body={"ollama_url": "http://127.0.0.1:11434"})
                        self.assertEqual(st_ol, 200)
                        self.assertTrue(ol_body["running"])
                        self.assertIn("qwen3:8b", ol_body["models"])

                    app_ctx["warm_session"].pipe = MagicMock()
                    st_p, p_body = call(
                        "/api/preferences",
                        method="POST",
                        body={"preferences": {"llm_backend": "ollama", "ollama_model": "qwen3:8b", "llm_model_key": "qwen3-8b-ov"}},
                    )
                    self.assertEqual(st_p, 200)
                    self.assertEqual(p_body["preferences"]["llm_backend"], "ollama")
                    self.assertIsNone(app_ctx["warm_session"].pipe)
                finally:
                    app_ctx["stop_event"].set()
                    server.shutdown()
                    server.server_close()
                    th.join(timeout=2)
        finally:
            tmpdir.cleanup()


class TestPublicRepoHardeningAndSecurity(unittest.TestCase):
    def test_localhost_csrf_host_and_origin_protection(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            prefs_file = os.path.join(tmpdir, "prefs.json")
            with patch.object(config, "USER_PREFS_FILE", prefs_file):
                server, app_ctx = web_app.create_server(host="127.0.0.1", port=0)
                port = server.server_address[1]
                th = threading.Thread(target=server.serve_forever, daemon=True)
                th.start()
                try:
                    # 1. Valid local POST (no Origin) succeeds and carries security headers
                    req_ok = urllib.request.Request(
                        f"http://127.0.0.1:{port}/api/preferences",
                        data=b'{"default_playlist_size": 25}',
                        headers={"Content-Type": "application/json"},
                        method="POST",
                    )
                    with urllib.request.urlopen(req_ok, timeout=5) as resp:
                        self.assertEqual(resp.status, 200)
                        self.assertEqual(resp.headers.get("X-Frame-Options"), "DENY")
                        self.assertEqual(resp.headers.get("X-Content-Type-Options"), "nosniff")

                    # 2. Valid same-loopback Origin succeeds
                    req_loop = urllib.request.Request(
                        f"http://127.0.0.1:{port}/api/preferences",
                        data=b'{"default_playlist_size": 26}',
                        headers={
                            "Content-Type": "application/json",
                            "Origin": f"http://127.0.0.1:{port}",
                        },
                        method="POST",
                    )
                    with urllib.request.urlopen(req_loop, timeout=5) as resp:
                        self.assertEqual(resp.status, 200)

                    # 3. Cross-origin POST blocked with 403
                    req_evil_orig = urllib.request.Request(
                        f"http://127.0.0.1:{port}/api/preferences",
                        data=b'{"default_playlist_size": 99}',
                        headers={
                            "Content-Type": "application/json",
                            "Origin": "https://evil.example",
                        },
                        method="POST",
                    )
                    with self.assertRaises(urllib.error.HTTPError) as ctx_orig:
                        urllib.request.urlopen(req_evil_orig, timeout=5)
                    self.assertEqual(ctx_orig.exception.code, 403)
                    ctx_orig.exception.close()

                    # 4. Sec-Fetch-Site: cross-site blocked with 403
                    req_sfs = urllib.request.Request(
                        f"http://127.0.0.1:{port}/api/preferences",
                        data=b'{"default_playlist_size": 99}',
                        headers={
                            "Content-Type": "application/json",
                            "Sec-Fetch-Site": "cross-site",
                        },
                        method="POST",
                    )
                    with self.assertRaises(urllib.error.HTTPError) as ctx_sfs:
                        urllib.request.urlopen(req_sfs, timeout=5)
                    self.assertEqual(ctx_sfs.exception.code, 403)
                    ctx_sfs.exception.close()

                    # 5. DNS-rebinding / non-loopback Host header blocked with 403
                    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
                    try:
                        conn.putrequest("POST", "/api/preferences", skip_host=True)
                        conn.putheader("Host", "evil.example")
                        conn.putheader("Content-Type", "application/json")
                        conn.putheader("Content-Length", "2")
                        conn.endheaders()
                        conn.send(b"{}")
                        host_resp = conn.getresponse()
                        host_resp.read()
                        self.assertEqual(host_resp.status, 403)
                    finally:
                        conn.close()
                finally:
                    app_ctx["stop_event"].set()
                    server.shutdown()
                    server.server_close()
                    th.join(timeout=2)

    def test_sanitize_llm_url_blocks_ssrf_allows_local_and_lan(self):
        default_url = "http://127.0.0.1:11434"
        self.assertEqual(utils._sanitize_llm_url("http://127.0.0.1:11434"), "http://127.0.0.1:11434")
        self.assertEqual(utils._sanitize_llm_url("http://localhost:1234"), "http://localhost:1234")
        self.assertEqual(utils._sanitize_llm_url("http://192.168.1.42:11434"), "http://192.168.1.42:11434")

        self.assertEqual(utils._sanitize_llm_url("http://169.254.169.254/latest/meta-data"), default_url)
        self.assertEqual(utils._sanitize_llm_url("https://evil.com"), default_url)
        self.assertEqual(utils._sanitize_llm_url("http://user:pass@127.0.0.1:11434"), default_url)
        self.assertEqual(utils._sanitize_llm_url("file:///etc/passwd"), default_url)

    def test_zip_bomb_member_limit_protection(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            buf = io.BytesIO()
            with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
                for i in range(205):
                    zf.writestr(f"Streaming_History_Audio_{i}.json", b"[]")
            with self.assertRaises(ValueError):
                utils.extract_uploaded_history_bytes("bomb.zip", buf.getvalue(), dest_dir=tmpdir)

    def test_ytm_export_urls_in_csv_m3u_markdown(self):
        playlist = {
            "playlist_name": "YTM Exclusives",
            "playlist_description": "Cross platform export check",
            "prompt": "ytm mix",
            "tracks": [
                {
                    "id": "yt:dQw4w9WgXcQ",
                    "video_id": "dQw4w9WgXcQ",
                    "name": "YTM Song",
                    "artist": "YTM Artist",
                    "source": "YT Music Discovery",
                    "bpm": 113.0,
                    "energy": 0.8,
                    "valence": 0.9,
                }
            ],
        }
        expected_url = "https://music.youtube.com/watch?v=dQw4w9WgXcQ"
        csv_txt = utils.format_playlist_csv(playlist)
        m3u_txt = utils.format_playlist_m3u(playlist)
        md_txt = utils.format_playlist_markdown(playlist)

        self.assertIn(expected_url, csv_txt)
        self.assertNotIn("open.spotify.com/track/yt:", csv_txt)
        self.assertIn(expected_url, m3u_txt)
        self.assertIn(f"[YT Music]({expected_url})", md_txt)

    def test_ytm_playlist_error_dict_and_rollback_and_curl_headers(self):
        # cURL header parsing (bash & Cookie)
        curl_cmd = (
            "curl 'https://music.youtube.com/youtubei/v1/browse' \\\n"
            "  -H 'Cookie: SAPISID=xyz123; SID=abc' \\\n"
            "  -H 'X-Goog-AuthUser: 0' \\\n"
            "  --header 'Authorization: SAPISIDHASH 999_xyz'"
        )
        parsed = ytmusic_client._parse_browser_headers_text(curl_cmd)
        self.assertEqual(parsed["Cookie"], "SAPISID=xyz123; SID=abc")
        self.assertEqual(parsed["Authorization"], "SAPISIDHASH 999_xyz")
        self.assertIsNone(ytmusic_client.check_browser_headers_warning(parsed))

        # create_playlist returning error dict raises RuntimeError
        mock_ytm = MagicMock()
        mock_ytm.create_playlist.return_value = {"status": "STATUS_FAILED", "error": "Unauthorized"}
        with self.assertRaises(RuntimeError):
            ytmusic_client.create_or_update_ytm_playlist(
                mock_ytm, name="Bad Mix", video_ids=["dQw4w9WgXcQ"]
            )

        # Update failure after removing old items attempts rollback of removed videoIds
        mock_ytm_upd = MagicMock()
        mock_ytm_upd.get_playlist.return_value = {
            "tracks": [{"videoId": "oldVid11111", "setVideoId": "set_1"}]
        }
        mock_ytm_upd.remove_playlist_items.return_value = {"status": "STATUS_SUCCEEDED"}
        mock_ytm_upd.add_playlist_items.side_effect = [
            {"status": "STATUS_FAILED"},
            {"status": "STATUS_SUCCEEDED"},
        ]
        with self.assertRaises(RuntimeError):
            ytmusic_client.create_or_update_ytm_playlist(
                mock_ytm_upd,
                name="Update Fail Rollback",
                video_ids=["newVid22222"],
                target_playlist_id="PL_ROLLBACK",
            )
        self.assertEqual(mock_ytm_upd.add_playlist_items.call_count, 2)
        rollback_call = mock_ytm_upd.add_playlist_items.call_args_list[1]
        self.assertEqual(rollback_call.args, ("PL_ROLLBACK", ["oldVid11111"]))

    def test_freshness_detects_provider_mismatch(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            s2 = os.path.join(tmpdir, "02_Vector_Core")
            os.makedirs(s2, exist_ok=True)
            with open(os.path.join(s2, "taste_vectors.npz"), "wb") as f:
                f.write(b"v")
            utils.atomic_write_json(
                os.path.join(s2, "taste_profile_summary.json"),
                {"session_filters": {"active_provider": "spotify"}, "clusters": []},
            )
            prefs_file = os.path.join(tmpdir, "prefs.json")
            utils.atomic_write_json(prefs_file, {"active_provider": "unified"})
            with patch.object(config, "USER_PREFS_FILE", prefs_file):
                fresh = utils.compute_pipeline_freshness(tmpdir)
                self.assertTrue(fresh["stage2_exists"])
                self.assertTrue(fresh["stage2_stale"])
                self.assertTrue(any("unified" in r and "spotify" in r for r in fresh["stale_reasons"]))

    def test_has_complete_llm_files_rejects_tokenizer_only(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tok_dir = os.path.join(tmpdir, "partial_tok")
            os.makedirs(tok_dir, exist_ok=True)
            with open(os.path.join(tok_dir, "openvino_tokenizer.xml"), "w", encoding="utf-8") as f:
                f.write("<net/>")
            with open(os.path.join(tok_dir, "openvino_tokenizer.bin"), "wb") as f:
                f.write(b"123")
            self.assertFalse(llm_parser._has_complete_llm_files(tok_dir))

            full_dir = os.path.join(tmpdir, "complete_ov")
            os.makedirs(full_dir, exist_ok=True)
            with open(os.path.join(full_dir, "openvino_model.xml"), "w", encoding="utf-8") as f:
                f.write("<net/>")
            with open(os.path.join(full_dir, "openvino_model.bin"), "wb") as f:
                f.write(b"weights")
            self.assertTrue(llm_parser._has_complete_llm_files(full_dir))

    def test_build_status_payload_mtime_cache_invalidation(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            s1_dir = os.path.join(tmpdir, "01_Data_Extractor")
            os.makedirs(s1_dir, exist_ok=True)
            prof_path = os.path.join(s1_dir, "user_music_profile.json")
            utils.atomic_write_json(
                prof_path,
                {"taste_summary": {"total_unique_tracks_analyzed": 10}, "audio_features": {}},
            )
            mock_ws = MagicMock()
            mock_ws.get_status.return_value = {
                "embed_warm": False,
                "llm_warm": False,
                "clusters": [],
                "session_avoid_artists": [],
                "session_avoid_genres": [],
            }
            app_ctx = {
                "base_dir": tmpdir,
                "warm_session": mock_ws,
                "job_manager": web_app.JobManager(tmpdir),
            }

            with patch("web_app.load_json_safe", wraps=utils.load_json_safe) as spy_load:
                snap1 = web_app.build_status_payload(app_ctx)
                snap2 = web_app.build_status_payload(app_ctx)
                self.assertEqual(snap1["pipeline"]["stage1"]["tracks"], 10)
                self.assertEqual(snap2["pipeline"]["stage1"]["tracks"], 10)
                self.assertEqual(spy_load.call_count, 1)

                # Update profile on disk (different size & mtime) -> invalidates cache cleanly
                time.sleep(0.02)
                utils.atomic_write_json(
                    prof_path,
                    {
                        "taste_summary": {"total_unique_tracks_analyzed": 42},
                        "audio_features": {"t1": {"bpm": 120, "source": "reccobeats"}},
                    },
                )
                snap3 = web_app.build_status_payload(app_ctx)
                self.assertEqual(snap3["pipeline"]["stage1"]["tracks"], 42)
                self.assertEqual(snap3["pipeline"]["stage1"]["audio_features"], 1)
                self.assertEqual(spy_load.call_count, 2)


if __name__ == "__main__":
    unittest.main()
