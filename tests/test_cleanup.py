import copy
import io
import os
from pathlib import Path
import sys
import tempfile
import threading
import unittest
from unittest.mock import patch

import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[1]
for path in (ROOT_DIR, ROOT_DIR / "02_Vector_Core"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import config
import npu_vectorizer
import utils
import web_app


class TestVectorCache(unittest.TestCase):
    def test_refresh_reuses_unblended_vectors_and_applies_current_playlist_weight(self):
        track_ids = [f"t{i}" for i in range(5)]
        base = np.eye(5, 512, dtype=np.float32)
        metadata = {
            tid: {"doc_hash": tid, "playlist_list": ["vibe"] if i < 2 else []}
            for i, tid in enumerate(track_ids)
        }
        prefs = {"embed_backend": "clap", "playlist_cooccurrence_weight": 0.12}
        with tempfile.TemporaryDirectory() as tmpdir:
            npz_path = Path(tmpdir) / "taste_vectors.npz"
            with (
                patch.object(npu_vectorizer, "__file__", str(Path(tmpdir) / "npu_vectorizer.py")),
                patch.object(npu_vectorizer, "get_effective_preferences", return_value=prefs),
                patch.object(npu_vectorizer, "load_merged_music_profile", return_value={}),
                patch.object(npu_vectorizer, "build_semantic_documents", side_effect=lambda _: (
                    track_ids[:], track_ids[:], copy.deepcopy(metadata)
                )),
                patch.object(npu_vectorizer, "_enrich_and_embed_clap", side_effect=lambda indices, *_: (
                    {i: base[i] for i in indices}, {}
                )) as embed,
                patch.object(sys, "argv", ["npu_vectorizer.py"]),
                patch("sys.stdout", new=io.StringIO()),
            ):
                npu_vectorizer.main()
                with np.load(npz_path) as data:
                    first = data["embeddings"].copy()
                npu_vectorizer.main()
                with np.load(npz_path) as data:
                    np.testing.assert_allclose(data["embeddings"], first, atol=1e-7)
                    np.testing.assert_array_equal(data["base_embeddings"], base)
                prefs["playlist_cooccurrence_weight"] = 0.0
                npu_vectorizer.main()
                with np.load(npz_path) as data:
                    np.testing.assert_array_equal(data["embeddings"], base)
                embed.assert_called_once()

                # Old archives contain only playlist-blended vectors and must be rebuilt once.
                np.savez(npz_path, embeddings=first, track_ids=track_ids, embed_backend="clap")
                npu_vectorizer.main()
                self.assertEqual(embed.call_count, 2)
                with np.load(npz_path) as data:
                    np.testing.assert_array_equal(data["base_embeddings"], base)

                metadata["t0"]["doc_hash"] = "updated"
                npu_vectorizer.main()
                self.assertEqual(embed.call_count, 3)
                self.assertEqual(embed.call_args.args[0], list(range(5)))

                track_ids.pop()
                npu_vectorizer.main()
                self.assertEqual(embed.call_count, 4)
                self.assertEqual(embed.call_args.args[0], list(range(4)))

                with patch.object(config, "CLAP_TEXT_BLEND_WEIGHT", config.CLAP_TEXT_BLEND_WEIGHT + 0.1):
                    npu_vectorizer.main()
                self.assertEqual(embed.call_count, 5)


class TestAtomicOutput(unittest.TestCase):
    def test_failed_write_or_replace_preserves_destination_and_removes_temp(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            target = Path(tmpdir) / "data.bin"
            target.write_bytes(b"original")
            with self.assertRaisesRegex(ValueError, "interrupted"):
                with utils.atomic_output_file(target, binary=True) as stream:
                    stream.write(b"partial")
                    raise ValueError("interrupted")
            self.assertEqual(target.read_bytes(), b"original")
            with patch.object(utils, "_replace_with_retry", side_effect=PermissionError):
                with self.assertRaises(PermissionError):
                    utils.atomic_write_text(target, "replacement")
            self.assertEqual(target.read_bytes(), b"original")
            self.assertEqual(list(Path(tmpdir).iterdir()), [target])
            with utils.atomic_output_file(target, binary=True) as stream:
                stream.write(b"complete")
            self.assertEqual(target.read_bytes(), b"complete")


class TestPreferences(unittest.TestCase):
    def test_partial_updates_keep_saved_values_and_use_identical_load_validation(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.object(config, "USER_PREFS_FILE", str(Path(tmpdir) / "prefs.json")):
                saved = utils.save_user_preferences({
                    "active_provider": " YTMUSIC ",
                    "embed_backend": "CLAP",
                    "default_playlist_size": 200,
                    "half_life_days": 90,
                    "avoid_artists": "A, A, B",
                    "ollama_url": "http://127.0.0.1:11434",
                    "cluster_labels": {"0": " Chill "},
                    "presets": {"Custom": {"size": 20}},
                    "automation": {"size": 25, "prompt": "Keep this prompt"},
                })
                self.assertEqual(saved, utils.get_effective_preferences())
                self.assertEqual(saved["default_playlist_size"], 100)
                self.assertEqual(saved["active_provider"], "ytmusic")
                self.assertEqual(saved["avoid_artists"], ["A", "B"])
                self.assertEqual(saved["cluster_labels"], {"0": "Chill"})
                updated = utils.save_user_preferences({
                    "active_provider": "invalid",
                    "default_playlist_size": False,
                    "half_life_days": "invalid",
                    "ollama_url": "https://example.com",
                    "presets": None,
                    "automation": {"size": 10},
                })
                self.assertEqual(updated, utils.get_effective_preferences())
                for key in ("active_provider", "default_playlist_size", "half_life_days", "ollama_url", "presets"):
                    self.assertEqual(updated[key], saved[key])
                self.assertEqual(updated["automation"]["size"], 10)
                self.assertEqual(updated["automation"]["prompt"], "Keep this prompt")


class TestJobManager(unittest.TestCase):
    def test_spawn_failure_finishes_job_and_allows_next_run(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.object(config, "LOGS_DIR", tmpdir):
                manager = web_app.JobManager(tmpdir)
            manager.state["status"] = "running"
            steps = [([sys.executable, "missing.py"], tmpdir, "Test")]
            with patch.object(web_app.subprocess, "Popen", side_effect=OSError("launch failed")):
                manager._run_steps_thread(steps, os.path.join(tmpdir, "failed.log"))
            self.assertEqual(manager.state["status"], "failed")
            self.assertNotEqual(manager.state["exit_code"], 0)
            self.assertIsNotNone(manager.state["finished_at"])
            self.assertTrue(any("launch failed" in line for line in manager.log_buffer))
            steps = [([sys.executable, "-c", "print('Hello 🎵'); print('tail', end='')"], tmpdir, "Test")]
            manager._run_steps_thread(steps, os.path.join(tmpdir, "success.log"))
            self.assertEqual(manager.state["status"], "completed")
            self.assertIn("Hello 🎵", manager.log_buffer)
            self.assertIn("tail", manager.log_buffer)

    def test_failed_step_preserves_exit_code_and_skips_remaining_steps(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.object(config, "LOGS_DIR", tmpdir):
                manager = web_app.JobManager(tmpdir)
            steps = [
                ([sys.executable, "-c", "import sys; print('failure', file=sys.stderr); sys.exit(7)"], tmpdir, "Fails"),
                ([sys.executable, "-c", "print('must not run')"], tmpdir, "Skipped"),
            ]
            manager._run_steps_thread(steps, os.path.join(tmpdir, "failed.log"))
            self.assertEqual(manager.state["status"], "failed")
            self.assertEqual(manager.state["exit_code"], 7)
            self.assertIn("failure", manager.log_buffer)
            self.assertNotIn("must not run", manager.log_buffer)

    def test_cancel_stops_process_and_finishes_job(self):
        ready = threading.Event()
        finished = threading.Event()
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.object(config, "LOGS_DIR", tmpdir):
                manager = web_app.JobManager(tmpdir)

            def observe(event, payload):
                if event == "log" and payload["line"] == "ready":
                    ready.set()
                if event == "state" and payload["status"] == "cancelled":
                    finished.set()

            steps = [([sys.executable, "-c", "import time; print('ready', flush=True); time.sleep(30)"], tmpdir, "Test")]
            with (
                patch.object(manager, "_build_steps", return_value=("Test", steps)),
                patch.object(manager, "_broadcast", side_effect=observe),
            ):
                manager.start_job("test")
                try:
                    self.assertTrue(ready.wait(5), "Subprocess never produced output")
                finally:
                    manager.cancel_job()
                    finished.wait(5)
                self.assertTrue(finished.is_set(), "Cancelled job never finished")
                self.assertEqual(manager.state["exit_code"], 130)
                self.assertIsNone(manager._proc)


if __name__ == "__main__":
    unittest.main()
