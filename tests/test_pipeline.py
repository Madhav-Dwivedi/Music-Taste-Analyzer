import io
import json
import os
import sys
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from queue import Queue
from unittest.mock import MagicMock, patch

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
STAGE1_DIR = os.path.join(ROOT_DIR, "01_Data_Extractor")
STAGE2_DIR = os.path.join(ROOT_DIR, "02_Vector_Core")
STAGE3_DIR = os.path.join(ROOT_DIR, "03_Agentic_Engine")
for path in (ROOT_DIR, STAGE1_DIR, STAGE2_DIR, STAGE3_DIR):
    if path not in sys.path:
        sys.path.insert(0, path)

import config
import discovery_engine
import history_importer
import llm_parser
import main_orchestrator
import npu_vectorizer
import reccobeats_client
import stage1_extractor
import taste_clusterer
import utils
import view_taste_map


class TestUtils(unittest.TestCase):
    def test_track_fingerprint_normalization_and_unicode(self):
        fp1 = utils.get_track_fingerprint("Numb (2003 Remaster)", "Linkin Park, Jay-Z")
        fp2 = utils.get_track_fingerprint("Numb - Live at Milton Keynes", [{"name": "Linkin Park"}])
        self.assertEqual(fp1, fp2)
        self.assertEqual(fp1, "numb::linkin park")

        # Preserve non-Latin scripts while normalizing diacritics
        fp_jp = utils.get_track_fingerprint("夜に駆ける (feat. Guest)", ["YOASOBI"])
        self.assertEqual(fp_jp, "夜に駆ける::yoasobi")

        fp_cafe = utils.get_track_fingerprint("Café del Mar", ["Energy 52"])
        self.assertEqual(fp_cafe, "cafe del mar::energy 52")

    def test_spotify_id_validation(self):
        self.assertTrue(utils.is_valid_spotify_id("4cOdK2wGLETKBW3PvgPWqT"))
        self.assertFalse(utils.is_valid_spotify_id("short_id"))
        self.assertFalse(utils.is_valid_spotify_id("4cOdK2wGLETKBW3PvgPWqT_extra"))
        self.assertFalse(utils.is_valid_spotify_id("4cOdK2wGLETKBW3PvgPW!T"))
        self.assertFalse(utils.is_valid_spotify_id(None))

    def test_timestamp_parsing(self):
        dt_z = utils.parse_spotify_timestamp("2025-05-01T14:30:00Z")
        self.assertEqual(dt_z, datetime(2025, 5, 1, 14, 30, 0, tzinfo=timezone.utc))

        dt_space = utils.parse_spotify_timestamp("2024-03-15 14:32")
        self.assertEqual(dt_space, datetime(2024, 3, 15, 14, 32, 0, tzinfo=timezone.utc))

        self.assertIsNone(utils.parse_spotify_timestamp("invalid-date"))
        self.assertIsNone(utils.parse_spotify_timestamp(""))

    def test_audio_feature_validation_and_formatting(self):
        poisoned = {"bpm": 120.0, "energy": 0.5, "danceability": 0.5, "valence": 0.5, "acousticness": 0.5}
        self.assertFalse(utils.is_real_audio_features(poisoned))
        self.assertEqual(utils.format_audio_feature_text(poisoned), "")

        not_found = {"source": "not_found", "fetched_at": "2026-01-01T00:00:00Z"}
        self.assertFalse(utils.is_real_audio_features(not_found))

        real_af = {
            "bpm": 140.0,
            "energy": 0.86,
            "valence": 0.22,
            "danceability": 0.78,
            "acousticness": 0.05,
            "source": "reccobeats",
        }
        self.assertTrue(utils.is_real_audio_features(real_af))
        formatted = utils.format_audio_feature_text(real_af)
        self.assertIn("140.0 BPM", formatted)
        self.assertIn("High Energy", formatted)
        self.assertIn("Sad Dark Melancholy", formatted)
        self.assertIn("Danceable", formatted)

    def test_atomic_write_json(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            target = os.path.join(tmpdir, "nested", "data.json")
            utils.atomic_write_json(target, {"emoji": "🎵", "count": 42})
            loaded = utils.load_json_safe(target)
            self.assertEqual(loaded, {"emoji": "🎵", "count": 42})
            self.assertEqual([f for f in os.listdir(os.path.dirname(target)) if f.endswith(".tmp")], [])

    def test_spotify_2026_playlist_helpers(self):
        sp = MagicMock()
        sp.current_user_playlist_create.return_value = {"id": "pl123", "external_urls": {"spotify": "https://open.spotify.com/playlist/pl123"}}
        res = utils.create_spotify_playlist(sp, name="My\nPlaylist\t" + "x" * 120, description="Line1\nLine2")
        self.assertEqual(res["id"], "pl123")
        call_kwargs = sp.current_user_playlist_create.call_args.kwargs
        self.assertNotIn("\n", call_kwargs["name"])
        self.assertLessEqual(len(call_kwargs["name"]), 100)
        self.assertNotIn("\n", call_kwargs["description"])

        # Test 2026 /items pagination skipping local tracks and episodes
        sp._get.return_value = {
            "items": [
                {"item": {"id": "trk1", "name": "Song 1", "type": "track", "is_local": False}},
                {"item": {"id": "ep1", "name": "Podcast", "type": "episode", "is_local": False}},
                {"track": {"id": "loc1", "name": "Local MP3", "type": "track", "is_local": True}},
            ],
            "next": None,
        }
        items = utils.fetch_playlist_items_2026(sp, "pl123")
        self.assertEqual(len(items), 1)
        self.assertEqual(items[0]["id"], "trk1")


class TestReccoBeatsClient(unittest.TestCase):
    def test_purge_poisoned_cache_batching_retries_and_ttl(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_path = os.path.join(tmpdir, "rb_cache.json")
            valid_real_id = "0vbtURX4qv1l7besfwmnD8"
            poisoned_id = "3HfKlhohNNTLIv2t9uvmzz"
            expired_nf_id = "5n4ZzdO7KtpxsDpoDTa3D6"
            fresh_nf_id = "7rK0iM6jwuo1oizdx8suQA"

            now = datetime.now(timezone.utc)
            initial_cache = {
                poisoned_id: {"bpm": 120.0, "energy": 0.5, "danceability": 0.5, "valence": 0.5, "acousticness": 0.5},
                valid_real_id: {"bpm": 132.0, "energy": 0.81, "valence": 0.4, "source": "reccobeats"},
                expired_nf_id: {
                    "source": "not_found",
                    "fetched_at": (now - timedelta(days=20)).isoformat().replace("+00:00", "Z"),
                },
                fresh_nf_id: {
                    "source": "not_found",
                    "fetched_at": (now - timedelta(days=1)).isoformat().replace("+00:00", "Z"),
                },
            }
            utils.atomic_write_json(cache_path, initial_cache)

            # Generate 42 extra valid 22-char IDs so missing count > 40 (tests 40-ID chunking)
            extra_ids = [f"TrackId{idx:015d}" for idx in range(42)]
            all_query_ids = [valid_real_id, poisoned_id, expired_nf_id, fresh_nf_id] + extra_ids

            # Mock session: first batch gets 429 then 200; second batch gets 500s (must NOT poison cache!)
            resp_429 = MagicMock(status_code=429, headers={"Retry-After": "0.01"}, text="rate limit")
            resp_ok_1 = MagicMock(status_code=200)
            resp_ok_1.json.return_value = {
                "content": [
                    {
                        "id": "uuid-1111",
                        "href": f"https://open.spotify.com/track/{poisoned_id}",
                        "tempo": 144.5,
                        "energy": 0.77,
                        "danceability": 0.65,
                        "valence": 0.42,
                        "acousticness": 0.12,
                    }
                ]
            }
            resp_500 = MagicMock(status_code=500, headers={}, text="server error")

            session = MagicMock()
            session.get.side_effect = [resp_429, resp_ok_1, resp_500, resp_500, resp_500, resp_500]

            with patch.object(reccobeats_client.time, "sleep", return_value=None):
                result = reccobeats_client.fetch_audio_features_batch(all_query_ids, cache_path=cache_path, session=session)

            # Verify batch sizes never exceeded 40
            batch_sizes = [len(call.kwargs["params"]["ids"].split(",")) for call in session.get.call_args_list]
            self.assertTrue(all(bs <= 40 for bs in batch_sizes))
            self.assertEqual(batch_sizes[0], 40)
            self.assertEqual(batch_sizes[1], 40)  # retried first batch
            self.assertEqual(batch_sizes[2], 4)   # second batch (poisoned + expired_nf + 42 extra - 40 = 4; fresh_nf skipped)

            # Real data returned, poisoned replaced with real data, failed batch NOT poisoned
            self.assertIn(valid_real_id, result)
            self.assertIn(poisoned_id, result)
            self.assertEqual(result[poisoned_id]["bpm"], 144.5)
            self.assertNotIn(expired_nf_id, result)

            saved_cache = utils.load_json_safe(cache_path)
            self.assertEqual(saved_cache[expired_nf_id]["source"], "not_found")
            # Tracks in failed 500 batch must remain uncached so they can be retried later
            for failed_tid in extra_ids[-4:]:
                self.assertNotIn(failed_tid, saved_cache)


class TestHistoryImporter(unittest.TestCase):
    def test_import_formats_skips_video_and_sorting(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            # Format A (endsong)
            fmt_a = [
                {
                    "ts": "2024-02-01T10:00:00Z",
                    "ms_played": 180000,
                    "master_metadata_track_name": "Track A",
                    "master_metadata_album_artist_name": "Artist A",
                    "spotify_track_uri": "spotify:track:4cOdK2wGLETKBW3PvgPWqT",
                },
                {
                    "ts": "2024-02-02T10:00:00Z",
                    "ms_played": 12000,  # skip (<30s)
                    "master_metadata_track_name": "Track A",
                    "master_metadata_album_artist_name": "Artist A",
                    "spotify_track_uri": "spotify:track:4cOdK2wGLETKBW3PvgPWqT",
                },
                {
                    "ts": "2024-02-03T10:00:00Z",
                    "ms_played": 5000,  # skip only -> should NOT appear in aggregated_data
                    "master_metadata_track_name": "Skipped Only",
                    "master_metadata_album_artist_name": "Artist X",
                    "spotify_track_uri": "spotify:track:0VjIjW4GlUZAMYd2vXMi3b",
                },
                {
                    "ts": "2024-02-03T10:00:00Z",
                    "ms_played": 90000,
                    "master_metadata_track_name": None,
                    "master_metadata_album_artist_name": None,
                },
            ]
            # Format B (old Streaming_History_Audio)
            fmt_b = [
                {
                    "endTime": "2023-05-10 08:15",
                    "msPlayed": 120000,
                    "trackName": "Track A",
                    "artistName": "Artist A",
                    "spotifyTrackUri": "spotify:track:4cOdK2wGLETKBW3PvgPWqT",
                }
            ]
            # Video file (must be skipped) & malformed non-list JSON (must not crash)
            with open(os.path.join(tmpdir, "Streaming_History_Audio_2024.json"), "w", encoding="utf-8") as f:
                json.dump(fmt_a, f)
            with open(os.path.join(tmpdir, "Streaming_History_Audio_2023.json"), "w", encoding="utf-8") as f:
                json.dump(fmt_b, f)
            with open(os.path.join(tmpdir, "Streaming_History_Video_2024.json"), "w", encoding="utf-8") as f:
                json.dump([{"ts": "2024-01-01T00:00:00Z", "ms_played": 99999}], f)
            with open(os.path.join(tmpdir, "corrupt_object.json"), "w", encoding="utf-8") as f:
                json.dump({"unexpected": "object"}, f)

            agg, summary = history_importer.import_history_with_stats(tmpdir)
            self.assertEqual(summary["files_processed"], 2)
            self.assertEqual(summary["video_files_skipped"], 1)
            self.assertEqual(summary["invalid_files"], 1)
            self.assertEqual(summary["skips_filtered"], 2)
            self.assertEqual(len(agg), 1)
            rec = agg["4cOdK2wGLETKBW3PvgPWqT"]
            self.assertEqual(rec["play_count"], 2)
            self.assertEqual(rec["skip_count"], 1)
            self.assertEqual(rec["first_streamed_date"], "2023-05-10T08:15:00Z")
            self.assertEqual(rec["last_streamed_date"], "2024-02-01T10:00:00Z")


class TestStage1AndVectorizer(unittest.TestCase):
    def test_normalize_track_explicit_local_episode_and_genre_enrichment(self):
        self.assertIsNone(stage1_extractor.normalize_track({"id": "abc", "name": "Local", "is_local": True}))
        self.assertIsNone(stage1_extractor.normalize_track({"id": "ep1", "name": "Ep", "type": "episode"}))

        raw_track = {
            "id": "4cOdK2wGLETKBW3PvgPWqT",
            "name": "Never Gonna Give You Up",
            "explicit": True,
            "popularity": 80,
            "duration_ms": 213000,
            "artists": [{"id": "0gxyHStUsqpMadRV0Di1Qt", "name": "Rick Astley"}],
            "album": {"id": "alb1", "name": "Whenever You Need Somebody", "release_date": "1987-11-16", "images": []},
        }
        norm = stage1_extractor.normalize_track(raw_track, genre_lookup={"0gxyHStUsqpMadRV0Di1Qt": ["synthpop", "dance pop"]})
        self.assertTrue(norm["explicit"])
        self.assertEqual(norm["artists"][0]["genres"], ["synthpop", "dance pop"])

        # Verify synthesize_profile includes short_term_artists_top_tracks in unique track count
        raw_data = {
            "artists": {"short_term": [], "medium_term": [], "long_term": []},
            "tracks": {"short_term": [], "medium_term": [], "long_term": []},
            "recently_played": [],
            "short_term_artists_top_tracks": [raw_track],
            "liked_songs": [],
            "playlists": [],
            "artist_genre_map": {"0gxyHStUsqpMadRV0Di1Qt": ["synthpop"]},
        }
        prof = stage1_extractor.synthesize_profile({"id": "u1", "display_name": "User"}, raw_data)
        self.assertEqual(prof["taste_summary"]["total_unique_tracks_analyzed"], 1)

    def test_vectorizer_merges_duplicates_and_ignores_poisoned_af(self):
        profile = {
            "artists": {
                "short_term": [{"id": "art1", "name": "Tame Impala", "genres": ["psychedelic pop"]}],
                "medium_term": [],
                "long_term": [],
            },
            "tracks": {
                "short_term": [
                    {
                        "id": "tid_orig_1111111111111",
                        "name": "Let It Happen",
                        "explicit": False,
                        "artists": [{"id": "art1", "name": "Tame Impala", "genres": []}],
                        "album": {"release_date": "2015-07-17"},
                    }
                ],
                "medium_term": [],
                "long_term": [],
            },
            "recently_played": [],
            "short_term_artists_top_tracks": [],
            "liked_songs_sample": [],
            "playlists": [
                {
                    "name": "Late Night Drives",
                    "tracks": [
                        {
                            "id": "tid_remaster_222222222",
                            "name": "Let It Happen - Radio Edit",
                            "explicit": True,
                            "artists": [{"id": "art1", "name": "Tame Impala", "genres": ["indie"]}],
                            "album": {"release_date": ""},
                        }
                    ],
                }
            ],
            "audio_features": {
                "tid_orig_1111111111111": {"bpm": 120.0, "energy": 0.5, "danceability": 0.5, "valence": 0.5, "acousticness": 0.5},
                "tid_remaster_222222222": {"bpm": 125.0, "energy": 0.88, "valence": 0.58, "source": "reccobeats"},
            },
        }
        docs, tids, meta = npu_vectorizer.build_semantic_documents(profile)
        self.assertEqual(len(docs), 1)
        self.assertEqual(len(tids), 1)
        canon_id = tids[0]
        self.assertIn("tid_remaster_222222222", meta[canon_id]["alt_ids"])
        self.assertIn("Late Night Drives", meta[canon_id]["playlist_list"])
        self.assertIn("psychedelic pop", meta[canon_id]["genres"])
        self.assertIn("indie", meta[canon_id]["genres"])
        self.assertIn("125.0 BPM", docs[0])
        self.assertEqual(meta[canon_id]["explicit"], "Explicit")


class TestLLMParserAndDiscoveryEngine(unittest.TestCase):
    def test_extract_json_object_edge_cases(self):
        raw = 'Sure! Here is the JSON:\n```json\n{\n  "playlist_name": "Midnight {Glow}",\n  "playlist_size": 25,\n}\n```'
        parsed = llm_parser.extract_json_object(raw)
        self.assertEqual(parsed["playlist_name"], "Midnight {Glow}")
        self.assertEqual(parsed["playlist_size"], 25)
        self.assertIsNone(llm_parser.extract_json_object("No JSON at all"))

    def test_heuristic_parser_split_bpm_excludes_and_single_artist(self):
        c1 = llm_parser.parse_prompt_heuristic(
            "Make a 50 song playlist with sad linkin park vibes. 10 songs from my library, 40 brand new."
        )
        self.assertEqual(c1["playlist_size"], 50)
        self.assertAlmostEqual(c1["known_tracks_ratio"], 0.2)
        self.assertFalse(c1["local_only"])
        self.assertIsNone(c1["target_artist"])
        self.assertLessEqual(c1["target_valence"], 0.3)
        self.assertIn("artist:linkin park", [q.lower() for q in c1["search_queries"]])
        self.assertEqual(c1["playlist_name"], "Sad Linkin Park")

        c2 = llm_parser.parse_prompt_heuristic("135+ BPM gym workout no Drake", past_queries=["genre:dance pop"])
        self.assertEqual(c2["min_bpm"], 135)
        self.assertIn("Drake", c2["exclude_artists"])
        self.assertNotIn("genre:dance pop", c2["search_queries"])

        c3 = llm_parser.parse_prompt_heuristic("35 songs only by Tame Impala")
        self.assertEqual(c3["playlist_size"], 35)
        self.assertEqual(c3["target_artist"], "Tame Impala")

    def test_parse_prompt_mistral_format_and_vibe_check_partial(self):
        mock_pipe = MagicMock()
        mock_pipe.generate.return_value = '{"search_queries": ["feel good", "genre:indie pop"], "playlist_size": 200}'
        res = llm_parser.parse_prompt(mock_pipe, "happy indie playlist")
        prompt_sent = mock_pipe.generate.call_args.args[0]
        self.assertIn("[INST]", prompt_sent)
        self.assertNotIn("<|system|>", prompt_sent)
        self.assertNotIn("feel good", res["search_queries"])
        self.assertIn("genre:indie pop", res["search_queries"])
        self.assertEqual(res["playlist_size"], 100)  # clamped

        cands = [{"name": f"Track {i}", "artist": f"Artist {i}"} for i in range(5)]
        mock_pipe.generate.return_value = '{"selected_indices": [2, true, 99, 0, "bad", 2], "playlist_name": "Late Vibes"}'
        vc = llm_parser.vibe_check_playlist(mock_pipe, "chill", cands, target_size=4)
        self.assertEqual(vc["selected_indices"], [2, 0])
        self.assertEqual(vc["playlist_name"], "Late Vibes")

    def test_discovery_helpers(self):
        # determine_single_artist_mode must NOT lock on vibe inspiration
        is_single, art = discovery_engine.determine_single_artist_mode(
            "sad linkin park vibes",
            {"search_queries": ["artist:Linkin Park"], "target_artist": None},
        )
        self.assertFalse(is_single)
        self.assertIsNone(art)

        is_single2, art2 = discovery_engine.determine_single_artist_mode(
            "only songs by Tame Impala",
            {"search_queries": ["artist:Tame Impala"], "target_artist": "Tame Impala"},
        )
        self.assertTrue(is_single2)
        self.assertEqual(art2, "Tame Impala")

        # format_spotify_query preserves subsequent operators
        self.assertEqual(
            discovery_engine.format_spotify_query("artist:Linkin Park year:2000-2007"),
            'artist:"Linkin Park" year:2000-2007',
        )
        self.assertEqual(
            discovery_engine.format_spotify_query('artist:"Adele"'),
            'artist:"Adele"',
        )
        self.assertEqual(
            discovery_engine.format_spotify_query("genre:indie pop"),
            'genre:"indie pop"',
        )

        # check_audio_features skips missing/poisoned, checks real strict vs relaxed
        self.assertTrue(discovery_engine.check_audio_features({}, {"min_bpm": 130}))
        self.assertFalse(discovery_engine.check_audio_features({"bpm": 115, "source": "reccobeats"}, {"min_bpm": 130}, relaxed=False))
        self.assertTrue(discovery_engine.check_audio_features({"bpm": 115, "source": "reccobeats"}, {"min_bpm": 130}, relaxed=True))

        # select_balanced_indices honors ratio & artist caps & deduplication
        candidates = [
            {"id": "l1", "name": "Song A", "artist": "Art 1", "source": "Local Library"},
            {"id": "l2", "name": "Song A (Remaster)", "artist": "Art 1", "source": "Local Library"},  # dup fp
            {"id": "l3", "name": "Song B", "artist": "Art 2", "source": "Local Library"},
            {"id": "n1", "name": "New 1", "artist": "Art 3", "source": "New Discovery"},
            {"id": "n2", "name": "New 2", "artist": "Art 4", "source": "New Discovery"},
            {"id": "n3", "name": "New 3", "artist": "Art 5", "source": "New Discovery"},
        ]
        chosen = discovery_engine.select_balanced_indices(
            candidates,
            llm_indices=[0, 1, 2, 3, 4],
            target_size=4,
            target_local_count=1,
            max_per_artist=2,
        )
        self.assertEqual(len(chosen), 4)
        self.assertEqual(sum(1 for c in chosen if c["source"] == "Local Library"), 1)
        self.assertEqual(sum(1 for c in chosen if c["source"] == "New Discovery"), 3)

    def test_export_to_spotify_empty_and_2026(self):
        self.assertFalse(discovery_engine.export_to_spotify({"tracks": []}))
        sp = MagicMock()
        sp.current_user_playlist_create.return_value = {"id": "newpl", "external_urls": {"spotify": "https://open.spotify.com/playlist/newpl"}}
        ok = discovery_engine.export_to_spotify(
            {
                "playlist_name": "Test List",
                "playlist_description": "Desc",
                "tracks": [{"id": "4cOdK2wGLETKBW3PvgPWqT", "name": "Track", "artist": "Artist"}],
            },
            sp=sp,
        )
        self.assertTrue(ok)
        sp.current_user_playlist_create.assert_called_once()
        sp.playlist_add_items.assert_called_once_with("newpl", ["spotify:track:4cOdK2wGLETKBW3PvgPWqT"])


class TestClustererAndVisualizer(unittest.TestCase):
    def test_plotly_text_preserves_quotes_wraps_titles_and_escapes_markup(self):
        title = 'KSI "Thick of It" PARODY I\'ve Got a Little Di.. ~ Rucka Rucka Ali'
        formatted = view_taste_map._plotly_text(title)
        self.assertIn('"Thick of It"', formatted)
        self.assertIn("I've", formatted)
        self.assertIn("<br>", formatted)
        self.assertNotIn("&quot;", formatted)
        self.assertNotIn("&#x27;", formatted)
        self.assertEqual(view_taste_map._plotly_text('<script>alert("x")</script>'), '&lt;script&gt;alert("x")&lt;/script&gt;')

    def test_exact_artist_penalty_and_recency_weights(self):
        penalties = {"nf": 0.05}
        self.assertAlmostEqual(taste_clusterer.apply_artist_penalties("NF", 10.0, penalties), 0.5)
        self.assertAlmostEqual(taste_clusterer.apply_artist_penalties("NF, Hopsin", 10.0, penalties), 0.5)
        self.assertAlmostEqual(taste_clusterer.apply_artist_penalties("Infamous", 10.0, penalties), 10.0)
        self.assertAlmostEqual(taste_clusterer.apply_artist_penalties("Confetti", 10.0, penalties), 10.0)

        now = datetime(2026, 6, 1, tzinfo=timezone.utc)
        track_ids = ["tid_short", "tid_hist"]
        metadata = {
            "tid_short": {
                "name": "Recent Hit",
                "artists": "Artist X",
                "playlist_list": ["Gym", "Favorites"],
                "alt_ids": ["tid_short_alt"],
            },
            "tid_hist": {
                "name": "Old Track",
                "artists": "Infamous",
                "playlist_list": ["Archive"],
                "alt_ids": [],
            },
        }
        profile = {
            "tracks": {"short_term": [{"id": "tid_short_alt"}], "medium_term": [], "long_term": []},
            "recently_played": [{"track": {"id": "tid_short"}} for _ in range(10)],
            "short_term_artists_top_tracks": [],
        }
        history_by_id = {
            "tid_hist": {"stream_timestamps": ["2020-01-01T00:00:00Z"]},
        }
        weights = taste_clusterer.compute_recency_weights(
            track_ids,
            metadata,
            profile,
            history_by_id=history_by_id,
            penalized_artists=penalties,
            now=now,
        )
        # Short term (0.15 + 15 + 5*12 = 75.15) * (1 + 0.4*1 = 1.4)
        self.assertAlmostEqual(weights[0], 75.15 * 1.4, places=3)
        # Old track not penalized because "Infamous" != "NF"
        self.assertGreater(weights[1], 0.14)

    def test_vibe_label_uniqueness_and_distinct_second_artist(self):
        used_labels = set()
        used_artists = set()
        anchors = [
            {"name": "S1", "artist": "NF"},
            {"name": "S2", "artist": "NF, Feature"},
            {"name": "S3", "artist": "Eminem"},
        ]
        lbl1 = taste_clusterer.build_vibe_label(0, ["Hip Hop"], anchors, 100, used_labels, used_artists)
        lbl2 = taste_clusterer.build_vibe_label(1, ["Hip Hop"], anchors, 100, used_labels, used_artists)
        self.assertNotEqual(lbl1, lbl2)
        self.assertIn("Eminem", lbl2)
        self.assertNotIn("NF & NF", lbl2)

    def test_html_visualizer_escapes_xss_and_renders_telemetry(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            html_out = os.path.join(tmpdir, "map.html")
            track_coords = [[0.1, 0.2]]
            track_clusters = [0]
            track_ids = ["tid1"]
            metadata = {
                "tid1": {
                    "name": "<script>alert('xss')</script>",
                    "artists": "Artist </script> Evil",
                    "genres": "indie",
                    "playlists": "Mix & Chill",
                    "year": "2024",
                    "recency_weight": 12.0,
                    "audio_features": {"bpm": 128.0, "energy": 0.85, "valence": 0.72, "source": "reccobeats"},
                }
            }
            clusters = [
                {
                    "cluster_id": 0,
                    "vibe_label": "Indie <Test>",
                    "weighted_share_pct": 100.0,
                    "track_count": 1,
                    "anchor_tracks": [{"name": "Song", "artist": "Artist"}],
                    "telemetry": {"avg_bpm": 128.0, "avg_energy": 0.85, "avg_valence": 0.72},
                }
            ]
            view_taste_map.generate_html_visualizer(
                track_coords,
                track_clusters,
                track_ids,
                metadata,
                clusters,
                center_coords=[[0.1, 0.2]],
                html_path=html_out,
                explained_variance=[0.45, 0.25],
            )
            with open(html_out, "r", encoding="utf-8") as f:
                content = f.read()
            self.assertNotIn("<script>alert('xss')</script>", content)
            self.assertNotIn("Artist </script> Evil", content)
            self.assertIn("&lt;script&gt;", content)
            self.assertIn("128.0 BPM", content)
            self.assertIn("45.0% variance", content)


class TestMainOrchestrator(unittest.TestCase):
    def test_enqueue_output_preserves_split_multibyte_utf8(self):
        payload = "Hello 🎵🚀 Café\n".encode("utf-8")

        class ByteByByteStream:
            def __init__(self, raw):
                self._raw = raw
                self._pos = 0
                self.closed = False

            def read1(self, n=1):
                if self._pos >= len(self._raw):
                    return b""
                chunk = self._raw[self._pos : self._pos + 1]
                self._pos += 1
                return chunk

            def close(self):
                self.closed = True

        q = Queue()
        main_orchestrator.enqueue_output(ByteByByteStream(payload), q)
        pieces = []
        while not q.empty():
            pieces.append(q.get_nowait())
        decoded = "".join(pieces)
        self.assertEqual(decoded, "Hello 🎵🚀 Café\n")
        self.assertNotIn("\ufffd", decoded)

    def test_pipeline_status_and_cleaner(self):
        import web_app

        with tempfile.TemporaryDirectory() as tmpdir:
            os.makedirs(os.path.join(tmpdir, "01_Data_Extractor"), exist_ok=True)
            os.makedirs(os.path.join(tmpdir, "02_Vector_Core"), exist_ok=True)
            os.makedirs(os.path.join(tmpdir, "03_Agentic_Engine"), exist_ok=True)
            os.makedirs(os.path.join(tmpdir, "extended_history"), exist_ok=True)

            with open(os.path.join(tmpdir, ".cache"), "w", encoding="utf-8") as f:
                f.write('{"access_token": "tok"}')
            with open(os.path.join(tmpdir, "extended_history", "Streaming_History_Audio_2025.json"), "w", encoding="utf-8") as f:
                f.write("[]")
            ext_stats = os.path.join(tmpdir, "01_Data_Extractor", "extended_history_stats.json")
            with open(ext_stats, "w", encoding="utf-8") as f:
                json.dump({"track1": {}}, f)

            mock_ws = MagicMock()
            mock_ws.get_status.return_value = {"embed_warm": False, "llm_warm": False, "clusters": [], "session_avoid_artists": [], "session_avoid_genres": []}
            app_ctx = {"base_dir": tmpdir, "warm_session": mock_ws, "job_manager": web_app.JobManager(tmpdir)}
            with patch.object(web_app, "get_spotify_client_id", return_value="cid123"):
                snap = web_app.build_status_payload(app_ctx)
                self.assertTrue(snap["auth"]["token_cached"])
                self.assertTrue(snap["auth"]["client_id_configured"])
                self.assertEqual(snap["pipeline"]["extended_history"]["unique_tracks"], 1)

            removed, errors = main_orchestrator.clean_system_files(tmpdir)
            self.assertEqual(errors, [])
            self.assertIn(".cache", removed)
            self.assertTrue(os.path.exists(ext_stats))
            self.assertTrue(os.path.exists(os.path.join(tmpdir, "extended_history", "Streaming_History_Audio_2025.json")))


if __name__ == "__main__":
    unittest.main()
