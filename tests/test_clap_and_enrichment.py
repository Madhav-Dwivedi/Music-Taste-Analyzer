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

    def test_name_clusters_with_ollama_updates_labels(self):
        clusters = [
            {
                "cluster_id": 0,
                "vibe_label": "Old Label 0",
                "auto_vibe_label": "Old Label 0",
                "top_genres": ["Classical"],
                "track_count": 20,
                "anchor_tracks": [{"name": "Merry-Go-Round", "artist": "Joe Hisaishi"}],
                "telemetry": {"avg_bpm": 130.0, "avg_energy": 0.46, "avg_valence": 0.37},
            },
            {
                "cluster_id": 1,
                "vibe_label": "Old Label 1",
                "auto_vibe_label": "Old Label 1",
                "top_genres": ["Vocal Jazz"],
                "track_count": 100,
                "anchor_tracks": [{"name": "Fly Me To The Moon", "artist": "Frank Sinatra"}],
                "telemetry": {"avg_bpm": 120.0, "avg_energy": 0.65, "avg_valence": 0.52},
            },
        ]
        fake_resp = SimpleNamespace(
            status_code=200,
            json=lambda: {
                "message": {
                    "content": '{"0": "Orchestral & Film Scores", "1": "Vintage Vocal Jazz"}'
                }
            },
        )
        prefs = {
            "llm_backend": "ollama",
            "ollama_url": "http://127.0.0.1:11434",
            "ollama_model": "qwen3.5:9b",
        }
        with patch("requests.post", return_value=fake_resp) as mock_post:
            ok = taste_clusterer.name_clusters_with_ollama(
                clusters, prefs=prefs, custom_labels={"1": "Pinned Jazz"}
            )
            mock_post.assert_called_once()
        self.assertTrue(ok)
        self.assertEqual(clusters[0]["vibe_label"], "Orchestral & Film Scores")
        self.assertEqual(clusters[1]["auto_vibe_label"], "Vintage Vocal Jazz")
        self.assertEqual(clusters[1]["vibe_label"], "Pinned Jazz")

    def test_deezer_build_search_candidates_cleans_ytm_noise(self):
        anri_cands = dz._build_search_candidates(
            "悲しみがとまらない\u3000I CAN'T STOP THE LONELINESS - Kanashimi ga Tomaranai I CAN'T STOP THE LONELINESS",
            "Anri",
        )
        titles_only = [t.lower() for t, _ in anri_cands]
        self.assertTrue(any("can't stop the loneliness" in t and "悲しみ" not in t for t in titles_only))
        self.assertTrue(all(a == "Anri" for _, a in anri_cands if a))

        reupload_cands = dz._build_search_candidates(
            "The Handsome Family - Far From Any Road [Official HD Video] | HBO",
            "Beatriz, 1.2M views",
        )
        self.assertIn(("Far From Any Road", "The Handsome Family"), reupload_cands)

    def test_fit_gmm_clusters_splits_oversized_cluster_and_handles_anisotropy(self):
        rng = np.random.RandomState(11)
        dim = 64
        # Inject strong shared cone offset + 5 modes where 1 super-mode has two distinct sub-pockets (50% total)
        cone = np.ones(dim, dtype=float) / np.sqrt(dim) * 3.0
        sub_a = np.eye(6, dim, dtype=float)[0] * 0.6 + cone
        sub_b = np.eye(6, dim, dtype=float)[1] * 0.6 + cone
        other_modes = [np.eye(6, dim, dtype=float)[k] * 0.7 + cone for k in range(2, 6)]

        pts = []
        for center, count in [(sub_a, 40), (sub_b, 40)] + [(m, 20) for m in other_modes]:
            block = center + rng.normal(0.0, 0.02, size=(count, dim))
            block /= np.linalg.norm(block, axis=1, keepdims=True)
            pts.append(block)

        embs = np.vstack(pts)
        weights = np.ones(len(embs), dtype=float)
        labels, centers, _, _ = taste_clusterer.fit_gmm_clusters(
            embs, weights, min_k=3, max_k=5, pca_dims=12, max_cluster_share=0.22
        )
        # Giant 50% cluster splits so final cluster count >= 5 and max share <= 0.30
        unique_cnt = len(set(labels.tolist()))
        self.assertGreaterEqual(unique_cnt, 5)
        counts = np.bincount(labels)
        self.assertLessEqual(counts.max() / float(len(embs)), 0.30)

    def test_align_text_to_audio_manifold_projects_rotated_space(self):
        rng = np.random.RandomState(19)
        n = 60
        dim = 32
        audio = rng.normal(0.0, 1.0, size=(n, dim)).astype(np.float32)
        audio /= np.linalg.norm(audio, axis=1, keepdims=True)
        # Apply fixed orthogonal rotation + small noise to simulate CLAP text subspace offset
        q_mat, _ = np.linalg.qr(rng.normal(0.0, 1.0, size=(dim, dim)))
        text_audio = (audio @ q_mat.T + rng.normal(0.0, 0.05, size=(n, dim))).astype(np.float32)
        text_audio /= np.linalg.norm(text_audio, axis=1, keepdims=True)

        aligned = npu_vectorizer.align_text_to_audio_manifold(audio, text_audio, text_audio, alpha=1.0)
        before_cos = float(np.mean(np.sum(audio * text_audio, axis=1)))
        after_cos = float(np.mean(np.sum(audio * aligned, axis=1)))
        self.assertGreater(after_cos, before_cos + 0.35)
        np.testing.assert_allclose(np.linalg.norm(aligned, axis=1), np.ones(n), atol=1e-5)

    def test_fit_hdbscan_clusters_separates_modes_and_marks_outliers(self):
        rng = np.random.RandomState(23)
        dim = 32
        pts = []
        # 3 tight dense modes (25 each) + 8 isolated scatter outliers
        for k in range(3):
            center = np.eye(5, dim, dtype=float)[k]
            blk = center + rng.normal(0.0, 0.03, size=(25, dim))
            blk /= np.linalg.norm(blk, axis=1, keepdims=True)
            pts.append(blk)
        outliers = rng.normal(0.0, 1.0, size=(8, dim))
        outliers /= np.linalg.norm(outliers, axis=1, keepdims=True)
        pts.append(outliers)

        embs = np.vstack(pts)
        weights = np.ones(len(embs), dtype=float)
        labels, centers, probs, _ = taste_clusterer.fit_hdbscan_clusters(
            embs, weights, umap_dims=6, min_cluster_ratio=0.10, max_cluster_share=0.50
        )
        self.assertGreaterEqual(len(centers), 2)
        self.assertEqual(probs.shape, (len(embs), len(centers)))
        self.assertTrue(np.any(labels == -1))


if __name__ == "__main__":
    unittest.main()
