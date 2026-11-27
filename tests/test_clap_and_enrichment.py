import os
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
for sub in (
    ROOT_DIR,
    os.path.join(ROOT_DIR, "01_Data_Extractor"),
    os.path.join(ROOT_DIR, "02_Vector_Core"),
    os.path.join(ROOT_DIR, "03_Agentic_Engine"),
):
    if sub not in sys.path:
        sys.path.insert(0, sub)

import deezer_audio_client as dz
import discovery_engine
import npu_vectorizer
import taste_clusterer


class TestClapAndEnrichment(unittest.TestCase):
    def test_extract_dsp_audio_features_synthetic_waveform(self):
        sr = 22050
        dur = 4.0
        t = np.linspace(0, dur, int(sr * dur), endpoint=False, dtype=np.float32)
        # 440Hz + 554.37Hz + 659.25Hz (A major triad) with 2 Hz (120 BPM) amplitude pulse
        pulse = 0.5 + 0.5 * np.maximum(0.0, np.sin(2.0 * np.pi * 2.0 * t))
        wave = 0.25 * pulse * (
            np.sin(2.0 * np.pi * 440.0 * t)
            + 0.7 * np.sin(2.0 * np.pi * 554.37 * t)
            + 0.6 * np.sin(2.0 * np.pi * 659.25 * t)
        )
        af = dz.extract_dsp_audio_features(wave, sr=sr, deezer_bpm=None)
        self.assertEqual(af["source"], "dsp")
        self.assertGreaterEqual(af["bpm"], 45.0)
        self.assertLessEqual(af["bpm"], 210.0)
        self.assertGreaterEqual(af["energy"], 0.05)
        self.assertLessEqual(af["energy"], 0.98)
        self.assertGreaterEqual(af["valence"], 0.10)
        self.assertLessEqual(af["valence"], 0.95)
        self.assertGreaterEqual(af["acousticness"], 0.05)
        self.assertLessEqual(af["acousticness"], 0.95)
        self.assertIn(af["key"], range(12))
        self.assertIn(af["mode"], (0, 1))

        af_dz = dz.extract_dsp_audio_features(wave, sr=sr, deezer_bpm=128.0)
        self.assertEqual(af_dz["bpm"], 128.0)

    def test_build_semantic_documents_omits_playlist_leakage(self):
        profile = {
            "tracks": {
                "short_term": [
                    {
                        "id": "yt:abc123",
                        "name": "Stay With Me",
                        "artists": [{"name": "Miki Matsubara", "genres": ["city pop"]}],
                        "album": {"release_date": "1979"},
                        "explicit": False,
                    }
                ]
            },
            "playlists": [
                {
                    "id": "pl1",
                    "name": "the pan",
                    "tracks": [
                        {
                            "id": "yt:abc123",
                            "name": "Stay With Me",
                            "artists": [{"name": "Miki Matsubara"}],
                        }
                    ],
                }
            ],
            "audio_features": {},
        }
        docs, tids, meta = npu_vectorizer.build_semantic_documents(profile)
        self.assertEqual(len(docs), 1)
        self.assertNotIn("Playlists:", docs[0])
        self.assertNotIn("the pan", docs[0])
        self.assertEqual(meta[tids[0]]["playlist_list"], ["the pan"])

    def test_soft_playlist_cooccurrence_ignores_catchall_and_normalizes(self):
        # 5 tracks; 'basic' has all 5 (100% > 40% catch-all), 'vibe' has t0 and t1 (40% <= 40% limit)
        tids = ["t0", "t1", "t2", "t3", "t4"]
        embs = np.eye(5, 8, dtype=np.float32)
        meta = {
            "t0": {"playlist_list": ["basic", "vibe"]},
            "t1": {"playlist_list": ["basic", "vibe"]},
            "t2": {"playlist_list": ["basic"]},
            "t3": {"playlist_list": ["basic"]},
            "t4": {"playlist_list": ["basic"]},
        }
        before_sim = float(np.dot(embs[0], embs[1]))
        out = npu_vectorizer.apply_soft_playlist_cooccurrence(embs, tids, meta, weight=0.15, catchall_ratio=0.40)
        after_sim = float(np.dot(out[0], out[1]))

        self.assertGreater(after_sim, before_sim)
        # t2 only belonged to catch-all 'basic', so it should remain unchanged
        np.testing.assert_allclose(out[2], embs[2], atol=1e-6)
        # All rows remain L2-normalized
        norms = np.linalg.norm(out, axis=1)
        np.testing.assert_allclose(norms, np.ones(5), atol=1e-5)

    def test_embed_texts_dispatches_to_clap(self):
        fake_model = SimpleNamespace(is_clap=True, _clap_device="cpu")
        fake_proc = object()
        empty = discovery_engine.embed_texts([], fake_proc, fake_model)
        self.assertEqual(empty.shape, (0, 512))

        fake_vecs = np.ones((2, 512), dtype=np.float32)
        with patch("clap_embedder.embed_texts_clap", return_value=fake_vecs) as mock_fn:
            out = discovery_engine.embed_texts(["chill synth", "jazz piano"], fake_proc, fake_model)
            mock_fn.assert_called_once()
            self.assertEqual(out.shape, (2, 512))

    def test_fit_gmm_clusters_recovers_synthetic_modes_and_soft_probs(self):
        rng = np.random.RandomState(7)
        dim = 64
        n_per = 30
        base_dirs = np.eye(4, dim, dtype=float)
        pts = []
        true_labels = []
        for cid in range(4):
            noise = rng.normal(0.0, 0.04, size=(n_per, dim))
            cluster_pts = base_dirs[cid] + noise
            cluster_pts /= np.linalg.norm(cluster_pts, axis=1, keepdims=True)
            pts.append(cluster_pts)
            true_labels.extend([cid] * n_per)

        embs = np.vstack(pts)
        weights = np.ones(len(embs), dtype=float)
        labels, centers, probs, bic_scores = taste_clusterer.fit_gmm_clusters(
            embs, weights, min_k=2, max_k=7, pca_dims=12
        )
        self.assertEqual(len(centers), 4)
        self.assertEqual(probs.shape, (120, 4))
        np.testing.assert_allclose(probs.sum(axis=1), np.ones(120), atol=1e-5)
        np.testing.assert_allclose(np.linalg.norm(centers, axis=1), np.ones(4), atol=1e-5)
        self.assertEqual(len(set(labels.tolist())), 4)
        self.assertIn(4, bic_scores)
        # Every true cluster maps cleanly to one unique GMM cluster
        for cid in range(4):
            slice_labels = labels[cid * n_per : (cid + 1) * n_per]
            self.assertEqual(len(set(slice_labels.tolist())), 1)


if __name__ == "__main__":
    unittest.main()
