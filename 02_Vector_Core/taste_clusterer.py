import argparse
import json
import math
import os
import sys
from datetime import datetime, timezone

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import config
from utils import (
    atomic_write_json,
    configure_utf8_stdio,
    get_effective_preferences,
    get_track_fingerprint,
    is_real_audio_features,
    load_json_safe,
    load_merged_music_profile,
    load_user_preferences,
    matches_avoided_artist,
    matches_avoided_genre,
    normalize_avoid_list,
    parse_spotify_timestamp,
)


def load_extended_history(profile, stage1_dir=None):
    """
    Load extended streaming history directly from Stage 1's extended_history_stats.json
    (or legacy embedded profile['extended_history']), returning (by_id, by_fingerprint).
    """
    s1_dir = stage1_dir or os.path.join(ROOT_DIR, "01_Data_Extractor")
    stats_path = os.path.join(s1_dir, "extended_history_stats.json")
    raw_history = load_json_safe(stats_path)
    if not isinstance(raw_history, dict):
        raw_history = (profile or {}).get("extended_history")
    if not isinstance(raw_history, dict):
        return {}, {}

    by_id = {}
    by_fp = {}

    for key, record in raw_history.items():
        if not isinstance(record, dict):
            continue
        key_str = str(key)
        if ":::" not in key_str:
            by_id[key_str] = record

        fp = record.get("fingerprint") or get_track_fingerprint(
            record.get("track_name", ""),
            record.get("artist_name", ""),
        )
        if fp:
            if fp not in by_fp:
                by_fp[fp] = {
                    "stream_timestamps": list(record.get("stream_timestamps") or []),
                    "play_count": int(record.get("play_count", 0) or 0),
                    "skip_count": int(record.get("skip_count", 0) or 0),
                }
            else:
                by_fp[fp]["stream_timestamps"].extend(record.get("stream_timestamps") or [])
                by_fp[fp]["play_count"] += int(record.get("play_count", 0) or 0)
                by_fp[fp]["skip_count"] += int(record.get("skip_count", 0) or 0)

    return by_id, by_fp


def filter_excluded_tracks(track_ids, metadata, avoid_artists=None, avoid_genres=None):
    """Filter out tracks matching avoided artists or avoided genres for taste model construction."""
    norm_artists = normalize_avoid_list(avoid_artists)
    norm_genres = normalize_avoid_list(avoid_genres)
    if not norm_artists and not norm_genres:
        return list(range(len(track_ids))), set()

    kept_indices = []
    excluded_ids = set()
    for idx, raw_tid in enumerate(track_ids):
        tid = str(raw_tid)
        info = metadata.get(tid, {})
        if matches_avoided_artist(info.get("artists", ""), norm_artists) or matches_avoided_genre(
            info.get("genres", ""), norm_genres
        ):
            excluded_ids.add(tid)
        else:
            kept_indices.append(idx)
    return kept_indices, excluded_ids


def load_penalized_artists():
    """Merge artist weight penalties from config.py and optional user_preferences.json."""
    merged = {}
    for source in (
        getattr(config, "PENALIZED_ARTISTS", {}) or {},
        (load_user_preferences().get("penalized_artists") or {}),
    ):
        if isinstance(source, dict):
            for name, mult in source.items():
                clean_name = str(name or "").strip().lower()
                if not clean_name:
                    continue
                try:
                    merged[clean_name] = max(0.0, float(mult))
                except (TypeError, ValueError):
                    continue
    return merged


def apply_artist_penalties(artists_str, weight, penalized_artists):
    """Apply exact (case-insensitive) artist weight multipliers without substring false positives."""
    if not penalized_artists or not artists_str:
        return weight
    track_artists = [a.strip().lower() for a in str(artists_str).split(",") if a.strip()]
    adjusted = weight
    seen = set()
    for art in track_artists:
        if art in penalized_artists and art not in seen:
            seen.add(art)
            adjusted *= penalized_artists[art]
    return adjusted


def compute_recency_weights(
    track_ids,
    metadata,
    profile,
    history_by_id=None,
    history_by_fp=None,
    penalized_artists=None,
    half_life_days=None,
    now=None,
    skip_penalty_enabled=True,
    skip_ratios_out=None,
    era_start=None,
    era_end=None,
):
    """Calculate recency-biased, time-decayed clustering weights for each vectorized track."""
    if half_life_days is None:
        half_life_days = float(getattr(config, "EXTENDED_HISTORY_HALF_LIFE_DAYS", 120.0))
    now_dt = now or datetime.now(timezone.utc)
    if era_end is not None and era_end < now_dt.year:
        now_dt = datetime(int(era_end), 12, 31, 23, 59, 59, tzinfo=timezone.utc)
    history_by_id = history_by_id or {}
    history_by_fp = history_by_fp or {}
    penalized_artists = penalized_artists or {}

    track_weights = {}
    id_to_canon = {}
    for raw_tid in track_ids:
        tid_str = str(raw_tid)
        track_weights[tid_str] = 0.15
        id_to_canon[tid_str] = tid_str
        meta = metadata.get(tid_str, {})
        for alt_id in meta.get("alt_ids") or []:
            id_to_canon.setdefault(str(alt_id), tid_str)

    tracks_section = (profile or {}).get("tracks") or {}
    for range_name, boost in (("long_term", 1.5), ("medium_term", 5.0), ("short_term", 15.0)):
        seen_in_range = set()
        for t in tracks_section.get(range_name) or []:
            if not isinstance(t, dict) or not t.get("id"):
                continue
            canon = id_to_canon.get(str(t["id"]))
            if canon and canon not in seen_in_range:
                seen_in_range.add(canon)
                track_weights[canon] += boost

    seen_artist_top = set()
    for t in (profile or {}).get("short_term_artists_top_tracks") or []:
        if not isinstance(t, dict) or not t.get("id"):
            continue
        canon = id_to_canon.get(str(t["id"]))
        if canon and canon not in seen_artist_top:
            seen_artist_top.add(canon)
            track_weights[canon] += 15.0

    recent_hits = {}
    for item in (profile or {}).get("recently_played") or []:
        if not isinstance(item, dict) or not isinstance(item.get("track"), dict):
            continue
        raw_id = item["track"].get("id")
        if not raw_id:
            continue
        canon = id_to_canon.get(str(raw_id))
        if canon:
            recent_hits[canon] = recent_hits.get(canon, 0) + 1

    for canon, hits in recent_hits.items():
        track_weights[canon] += min(hits, 5) * 12.0

    if history_by_id or history_by_fp:
        ln2_over_hl = math.log(2.0) / max(1.0, float(half_life_days))
        for raw_tid in track_ids:
            tid_str = str(raw_tid)
            meta = metadata.get(tid_str, {})
            candidate_ids = [tid_str] + [str(a) for a in (meta.get("alt_ids") or [])]

            collected_timestamps = []
            matched_plays = 0
            matched_skips = 0
            for cid in candidate_ids:
                rec = history_by_id.get(cid)
                if rec:
                    if rec.get("stream_timestamps"):
                        collected_timestamps.extend(rec["stream_timestamps"])
                    matched_plays += int(rec.get("play_count", 0) or 0)
                    matched_skips += int(rec.get("skip_count", 0) or 0)

            if not collected_timestamps and matched_plays == 0 and matched_skips == 0:
                fp = get_track_fingerprint(meta.get("name", ""), meta.get("artists", ""))
                fp_rec = history_by_fp.get(fp)
                if fp_rec:
                    if fp_rec.get("stream_timestamps"):
                        collected_timestamps.extend(fp_rec["stream_timestamps"])
                    matched_plays += int(fp_rec.get("play_count", 0) or 0)
                    matched_skips += int(fp_rec.get("skip_count", 0) or 0)

            if collected_timestamps:
                unique_timestamps = set(collected_timestamps)
                decay_sum = 0.0
                for ts_str in unique_timestamps:
                    stream_dt = parse_spotify_timestamp(ts_str)
                    if stream_dt is None:
                        continue
                    if era_start is not None and stream_dt.year < int(era_start):
                        continue
                    if era_end is not None and stream_dt.year > int(era_end):
                        continue
                    delta_days = max(0.0, (now_dt - stream_dt).total_seconds() / 86400.0)
                    decay_sum += math.exp(-ln2_over_hl * delta_days)

                if decay_sum > 25.0:
                    decay_sum = 25.0 + 5.0 * math.log1p(decay_sum - 25.0)
                track_weights[tid_str] += decay_sum

            total_events = matched_plays + matched_skips
            if total_events > 0:
                skip_ratio = matched_skips / float(total_events)
                if skip_ratios_out is not None:
                    skip_ratios_out[tid_str] = round(skip_ratio, 3)
                if skip_penalty_enabled and matched_skips >= 3 and skip_ratio >= 0.60:
                    track_weights[tid_str] *= max(0.15, 1.0 - skip_ratio)

    for raw_tid in track_ids:
        tid_str = str(raw_tid)
        meta = metadata.get(tid_str, {})
        pl_list = meta.get("playlist_list")
        if isinstance(pl_list, list):
            pl_count = len(pl_list)
        else:
            pl_string = meta.get("playlists", "")
            pl_count = len([p for p in str(pl_string).split(",") if p.strip()])

        if pl_count > 1:
            track_weights[tid_str] *= 1.0 + 0.4 * min(pl_count - 1, 5)

        track_weights[tid_str] = apply_artist_penalties(
            meta.get("artists", ""),
            track_weights[tid_str],
            penalized_artists,
        )

    return [track_weights[str(tid)] for tid in track_ids]


def choose_optimal_k(embeddings, weights, min_k=3, max_k=7):
    """Adaptively choose number of clusters using silhouette score while avoiding empty clusters."""
    import numpy as np
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score

    n_samples = len(embeddings)
    if n_samples <= 1:
        return 1
    if n_samples < 15:
        return max(1, min(min_k, n_samples))

    k_low = max(2, min(min_k, n_samples - 1))
    k_high = max(k_low, min(max_k, n_samples - 1))
    if k_low == k_high:
        return k_low

    if n_samples > 1000:
        rng = np.random.RandomState(42)
        sample_idx = rng.choice(n_samples, size=1000, replace=False)
    else:
        sample_idx = np.arange(n_samples)

    best_k = k_low
    best_score = -2.0

    for k in range(k_low, k_high + 1):
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(embeddings, sample_weight=weights)
        labels = km.labels_
        if len(np.unique(labels)) < k:
            continue
        sampled_labels = labels[sample_idx]
        if len(np.unique(sampled_labels)) < 2:
            continue
        score = silhouette_score(embeddings[sample_idx], sampled_labels, metric="cosine")
        if score > best_score:
            best_score = score
            best_k = k

    return best_k


def _reduce_for_clustering(embeddings, max_dims=20):
    """Mean-center, compress with PCA, and L2-normalize before GMM to remove CLAP cone anisotropy."""
    import numpy as np
    from sklearn.decomposition import PCA

    mat = np.asarray(embeddings, dtype=float)
    n_samples, dim = mat.shape
    if n_samples >= 3:
        mat = mat - np.mean(mat, axis=0, keepdims=True)
    if dim <= max_dims or n_samples <= max_dims + 2:
        reduced = mat
    else:
        n_comp = min(int(max_dims), n_samples - 1, dim)
        pca = PCA(n_components=n_comp, random_state=42)
        reduced = pca.fit_transform(mat)

    norms = np.linalg.norm(reduced, axis=1, keepdims=True)
    norms = np.where(norms > 1e-9, norms, 1.0)
    return reduced / norms


def fit_gmm_clusters(embeddings, weights, min_k=3, max_k=14, fixed_k=None, pca_dims=20, max_cluster_share=0.22):
    """Fit diagonal Gaussian Mixture Models on centered PCA vectors, split bloated clusters, and tighten bounds."""
    import numpy as np
    from sklearn.metrics.pairwise import cosine_similarity
    from sklearn.mixture import GaussianMixture

    mat = np.asarray(embeddings, dtype=float)
    n_samples = len(mat)
    w_arr = np.asarray(weights, dtype=float)
    if len(w_arr) != n_samples or float(w_arr.sum()) <= 0.0:
        w_arr = np.ones(n_samples, dtype=float)

    if n_samples <= 1:
        center = mat[0].copy() if n_samples == 1 else np.zeros(mat.shape[1], dtype=float)
        c_norm = float(np.linalg.norm(center))
        if c_norm > 0:
            center = center / c_norm
        return np.zeros(n_samples, dtype=int), np.array([center]), np.ones((n_samples, 1), dtype=float), {}

    # Mean-centered full-D unit vectors prevent isotropic hubs from swallowing fringe tracks
    if n_samples >= 3:
        mat_centered = mat - np.mean(mat, axis=0, keepdims=True)
    else:
        mat_centered = mat.copy()
    c_norms = np.linalg.norm(mat_centered, axis=1, keepdims=True)
    c_norms = np.where(c_norms > 1e-9, c_norms, 1.0)
    mat_centered = mat_centered / c_norms

    x_pca = _reduce_for_clustering(mat, max_dims=pca_dims)
    k_low = max(2, min(int(min_k), n_samples - 1))
    k_high = max(k_low, min(int(max_k), n_samples - 1))

    if fixed_k is not None:
        ks_to_try = [max(1, min(int(fixed_k), n_samples - 1))]
    elif n_samples < 15:
        ks_to_try = [max(1, min(k_low, n_samples))]
    else:
        ks_to_try = list(range(k_low, k_high + 1))

    min_cluster_size = max(3, int(0.015 * n_samples)) if n_samples >= 60 else 1
    best_gmm = None
    best_bic = float("inf")
    bic_scores = {}

    for k in ks_to_try:
        gmm = GaussianMixture(
            n_components=k,
            covariance_type="diag",
            reg_covar=1e-4,
            n_init=5,
            random_state=42,
        )
        gmm.fit(x_pca)
        bic = float(gmm.bic(x_pca))
        bic_scores[k] = round(bic, 1)

        hard_labels = gmm.predict(x_pca)
        counts = np.bincount(hard_labels, minlength=k)
        non_empty = int(np.sum(counts > 0))
        smallest = int(np.min(counts)) if non_empty == k else 0

        is_degenerate = fixed_k is None and k > k_low and (non_empty < k or smallest < min_cluster_size)
        if is_degenerate:
            continue
        if bic < best_bic or best_gmm is None:
            best_bic = bic
            best_gmm = gmm

    if best_gmm is None:
        best_gmm = GaussianMixture(
            n_components=ks_to_try[0],
            covariance_type="diag",
            reg_covar=1e-4,
            n_init=5,
            random_state=42,
        ).fit(x_pca)

    probs = best_gmm.predict_proba(x_pca)
    labels = np.argmax(probs, axis=1)
    n_comp = probs.shape[1]

    # Split abnormally oversized clusters (e.g. >22% of library when k is already high)
    if fixed_k is None and n_samples >= 80 and float(max_cluster_share or 0.0) > 0.0:
        for _ in range(2):
            counts = np.bincount(labels, minlength=n_comp)
            biggest = int(np.argmax(counts))
            share_limit = max(float(max_cluster_share), 1.6 / max(1, n_comp))
            if (counts[biggest] / float(n_samples)) <= share_limit or n_comp >= k_high + 2:
                break
            mask = labels == biggest
            sub_x = x_pca[mask]
            if len(sub_x) < max(2 * min_cluster_size, 8):
                break
            sub_gmm = GaussianMixture(
                n_components=2,
                covariance_type="diag",
                reg_covar=1e-4,
                n_init=5,
                random_state=42,
            ).fit(sub_x)
            sub_probs = sub_gmm.predict_proba(sub_x)
            sub_labels = np.argmax(sub_probs, axis=1)
            if int(np.sum(sub_labels == 0)) < min_cluster_size or int(np.sum(sub_labels == 1)) < min_cluster_size:
                break

            new_cid = n_comp
            new_probs = np.zeros((n_samples, n_comp + 1), dtype=float)
            new_probs[:, :n_comp] = probs
            parent_p = probs[mask, biggest]
            new_probs[mask, biggest] = parent_p * sub_probs[:, 0]
            new_probs[mask, new_cid] = parent_p * sub_probs[:, 1]
            probs = new_probs
            labels[mask] = np.where(sub_labels == 0, biggest, new_cid)
            n_comp += 1

    c_centers = []
    for c in range(n_comp):
        soft_w = w_arr * probs[:, c]
        if float(soft_w.sum()) <= 1e-9:
            soft_w = probs[:, c]
        if float(soft_w.sum()) <= 1e-9:
            c_vec = np.mean(mat_centered, axis=0)
        else:
            c_vec = np.average(mat_centered, axis=0, weights=soft_w)
        c_norm = float(np.linalg.norm(c_vec))
        if c_norm > 0:
            c_vec = c_vec / c_norm
        c_centers.append(c_vec)
    c_centers_mat = np.vstack(c_centers)

    # Tighten cluster bounds in mean-centered full-D cosine space
    sims = cosine_similarity(mat_centered, c_centers_mat)
    for i in range(len(labels)):
        best_c = int(np.argmax(sims[i]))
        if sims[i, best_c] > sims[i, labels[i]] + 0.03:
            labels[i] = best_c

    final_centers = []
    for c in range(n_comp):
        mask = labels == c
        if np.any(mask):
            cw = w_arr[mask]
            c_vec = np.average(mat[mask], axis=0, weights=cw) if float(cw.sum()) > 1e-9 else np.mean(mat[mask], axis=0)
        else:
            pw = w_arr * probs[:, c]
            c_vec = np.average(mat, axis=0, weights=pw) if float(pw.sum()) > 1e-9 else np.mean(mat, axis=0)
        c_norm = float(np.linalg.norm(c_vec))
        if c_norm > 0:
            c_vec = c_vec / c_norm
        final_centers.append(c_vec)

    return labels, np.vstack(final_centers), probs, bic_scores


def fit_hdbscan_clusters(embeddings, weights, umap_dims=10, min_cluster_ratio=0.018, max_cluster_share=0.22):
    """Density-based clustering in cosine-UMAP manifold via HDBSCAN, allowing -1 unclustered outliers."""
    import numpy as np
    from sklearn.cluster import HDBSCAN
    from sklearn.mixture import GaussianMixture

    mat = np.asarray(embeddings, dtype=float)
    n_samples = len(mat)
    w_arr = np.asarray(weights, dtype=float)
    if len(w_arr) != n_samples or float(w_arr.sum()) <= 0.0:
        w_arr = np.ones(n_samples, dtype=float)

    if n_samples < 30:
        return fit_gmm_clusters(mat, w_arr, min_k=2, max_k=min(6, max(2, n_samples - 1)), max_cluster_share=max_cluster_share)

    mat_centered = mat - np.mean(mat, axis=0, keepdims=True)
    c_norms = np.linalg.norm(mat_centered, axis=1, keepdims=True)
    c_norms = np.where(c_norms > 1e-9, c_norms, 1.0)
    mat_centered = mat_centered / c_norms

    raw_min = int(float(min_cluster_ratio) * n_samples)
    min_c_size = max(12, raw_min) if n_samples >= 120 else max(5, min(12, n_samples // 6))
    min_samples = max(4, min_c_size // 3)

    n_comp = max(2, min(int(umap_dims), n_samples - 2, mat.shape[1]))
    try:
        import umap

        reducer = umap.UMAP(
            n_components=n_comp,
            n_neighbors=min(15, max(5, min_c_size)),
            min_dist=0.0,
            metric="cosine",
            random_state=42,
        )
        x_embed = reducer.fit_transform(mat_centered)
    except Exception:
        x_embed = _reduce_for_clustering(mat, max_dims=n_comp)

    hdb = HDBSCAN(
        min_cluster_size=min_c_size,
        min_samples=min_samples,
        metric="euclidean",
        cluster_selection_method="leaf",
        copy=True,
    )
    hdb.fit(x_embed)
    raw_labels = hdb.labels_.astype(int)
    raw_probs = getattr(hdb, "probabilities_", np.where(raw_labels >= 0, 1.0, 0.0))

    found_ids = sorted(c for c in set(raw_labels.tolist()) if c >= 0)
    if len(found_ids) < 2:
        return fit_gmm_clusters(mat, w_arr, min_k=3, max_k=14, max_cluster_share=max_cluster_share)

    # Split any oversized dense continent cleanly while preserving density outliers (-1)
    if n_samples >= 80 and float(max_cluster_share or 0.0) > 0.0:
        min_half = max(4, min_c_size // 2)
        for _ in range(2):
            cur_ids = sorted(c for c in set(raw_labels.tolist()) if c >= 0)
            share_limit = max(float(max_cluster_share), 1.6 / max(1, len(cur_ids)))
            biggest = max(cur_ids, key=lambda cid: int(np.sum(raw_labels == cid)))
            big_mask = raw_labels == biggest
            if (int(np.sum(big_mask)) / float(n_samples)) <= share_limit or len(cur_ids) >= 18:
                break
            sub_x = x_embed[big_mask]
            if len(sub_x) < max(2 * min_half, 8):
                break
            sub_gmm = GaussianMixture(
                n_components=2, covariance_type="diag", reg_covar=1e-4, n_init=5, random_state=42
            ).fit(sub_x)
            sub_lbls = np.argmax(sub_gmm.predict_proba(sub_x), axis=1)
            if int(np.sum(sub_lbls == 0)) < min_half or int(np.sum(sub_lbls == 1)) < min_half:
                break
            new_cid = max(cur_ids) + 1
            raw_labels[big_mask] = np.where(sub_lbls == 0, biggest, new_cid)

    final_ids = sorted(c for c in set(raw_labels.tolist()) if c >= 0)
    id_map = {old_cid: new_idx for new_idx, old_cid in enumerate(final_ids)}
    labels = np.array([id_map[c] if c >= 0 else -1 for c in raw_labels], dtype=int)
    n_out = len(final_ids)

    # Reject orthogonal border tracks dragged into clusters by UMAP k-NN edges
    c_centers = []
    for cid in range(n_out):
        mask = labels == cid
        cw = w_arr[mask]
        cv = np.average(mat_centered[mask], axis=0, weights=cw) if float(cw.sum()) > 1e-9 else np.mean(mat_centered[mask], axis=0)
        cn = float(np.linalg.norm(cv))
        c_centers.append(cv / cn if cn > 0 else cv)
    c_centers_mat = np.vstack(c_centers)

    for i in range(n_samples):
        if labels[i] >= 0 and float(np.dot(mat_centered[i], c_centers_mat[labels[i]])) < 0.20:
            labels[i] = -1

    kept_cids = sorted(c for c in set(labels.tolist()) if c >= 0 and int(np.sum(labels == c)) > 0)
    if len(kept_cids) < 2:
        return fit_gmm_clusters(mat, w_arr, min_k=3, max_k=14, max_cluster_share=max_cluster_share)

    re_map = {old_c: new_c for new_c, old_c in enumerate(kept_cids)}
    labels = np.array([re_map[c] if c in re_map else -1 for c in labels], dtype=int)
    n_out = len(kept_cids)

    centers = []
    probs = np.zeros((n_samples, n_out), dtype=float)
    for new_cid in range(n_out):
        mask = labels == new_cid
        cw = w_arr[mask]
        c_vec = np.average(mat[mask], axis=0, weights=cw) if float(cw.sum()) > 1e-9 else np.mean(mat[mask], axis=0)
        c_norm = float(np.linalg.norm(c_vec))
        if c_norm > 0:
            c_vec = c_vec / c_norm
        centers.append(c_vec)

    for i in range(n_samples):
        if labels[i] >= 0:
            probs[i, labels[i]] = float(raw_probs[i]) if i < len(raw_probs) else 1.0

    return labels, np.vstack(centers), probs, {}


def build_vibe_label(cluster_id, top_genre_names, anchor_tracks, cluster_size, used_labels, used_artists):
    """Construct a human-readable, guaranteed-unique vibe title for a cluster."""
    artist_counts = {}
    for tr in anchor_tracks or []:
        art_raw = str(tr.get("artist") or "").strip()
        primary = art_raw.split(",")[0].strip() if art_raw else ""
        if primary and primary.lower() != "unknown":
            artist_counts[primary] = artist_counts.get(primary, 0) + 1

    top_artist = sorted(artist_counts.items(), key=lambda x: x[1], reverse=True)[0][0] if artist_counts else ""
    second_artist = ""
    for tr in anchor_tracks or []:
        art_raw = str(tr.get("artist") or "").strip()
        primary = art_raw.split(",")[0].strip() if art_raw else ""
        if primary and primary.lower() != "unknown" and primary.lower() != top_artist.lower():
            second_artist = primary
            break

    if top_genre_names and top_artist:
        vibe_title = f"{', '.join(top_genre_names[:2])} ({top_artist} Style)"
    elif top_artist:
        vibe_title = f"{top_artist} Sound"
    elif top_genre_names:
        vibe_title = f"{', '.join(top_genre_names[:2])}"
    else:
        vibe_title = f"Vibe Cluster {cluster_id}"

    artist_already_used = bool(top_artist) and (top_artist.lower() in used_artists)
    if vibe_title in used_labels or artist_already_used:
        if top_artist:
            if cluster_size > 500:
                candidate = f"Historical {top_artist} Archive"
            elif second_artist:
                candidate = f"Active {top_artist} & {second_artist} Rotation"
            else:
                candidate = f"Active {top_artist} Mix"
            if candidate in used_labels and top_genre_names:
                candidate = f"{candidate} ({top_genre_names[0]})"
            vibe_title = candidate

    base_title = vibe_title
    suffix = 2
    while vibe_title in used_labels:
        vibe_title = f"{base_title} #{suffix}"
        suffix += 1

    used_labels.add(vibe_title)
    if top_artist:
        used_artists.add(top_artist.lower())
    return vibe_title


def name_clusters_with_ollama(clusters_out, prefs=None, custom_labels=None, timeout=25):
    """Batch-name discovered music clusters in one Ollama /api/chat call when configured."""
    import requests

    if not clusters_out:
        return False

    eff_prefs = prefs or get_effective_preferences()
    if str(eff_prefs.get("llm_backend") or "").strip().lower() != "ollama":
        return False

    base_url = str(eff_prefs.get("ollama_url") or getattr(config, "DEFAULT_OLLAMA_URL", "http://127.0.0.1:11434")).rstrip("/")
    model = str(eff_prefs.get("ollama_model") or getattr(config, "DEFAULT_OLLAMA_MODEL", "qwen3:8b")).strip()
    if not model:
        return False

    cluster_briefs = []
    for c in clusters_out:
        anchors = [f"{a.get('name', '')} - {a.get('artist', '').split(',')[0].strip()}" for a in (c.get("anchor_tracks") or [])[:5]]
        t = c.get("telemetry") or {}
        cluster_briefs.append(
            {
                "id": str(c["cluster_id"]),
                "tracks": c.get("track_count", 0),
                "top_tags": c.get("top_genres") or [],
                "avg_bpm": t.get("avg_bpm"),
                "avg_energy": t.get("avg_energy"),
                "avg_valence": t.get("avg_valence"),
                "sample_songs": anchors,
            }
        )

    sys_prompt = (
        "You name music taste clusters from anchor songs and acoustic metrics. "
        "Return ONLY a JSON object mapping each cluster id string to a concise, natural 2-4 word musical genre or mood title "
        "(e.g. 'Orchestral & Film Scores', 'Vintage Vocal Jazz', '70s Classic Rock', 'Modern Pop & R&B'). "
        "Rules: no emojis, no 'Style' or 'Sound' suffixes, no artist-only names unless all samples are covers/comedy, "
        "and make every cluster name distinct."
    )
    user_prompt = json.dumps({"clusters": cluster_briefs}, ensure_ascii=False)

    try:
        resp = requests.post(
            f"{base_url}/api/chat",
            json={
                "model": model,
                "messages": [
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "stream": False,
                "think": False,
                "format": "json",
                "options": {"temperature": 0.2, "num_predict": 400},
            },
            timeout=timeout,
        )
        if resp.status_code != 200:
            return False
        content = ((resp.json().get("message") or {}).get("content") or "").strip()
        if not content:
            return False
        parsed = json.loads(content)
        if not isinstance(parsed, dict):
            return False
    except Exception as e:
        print(f"      ⚠️ Ollama cluster naming skipped ({e}); keeping heuristic titles.")
        return False

    custom_map = custom_labels or {}
    used_names = set()
    updated = 0
    for c in clusters_out:
        cid_str = str(c["cluster_id"])
        raw_name = parsed.get(cid_str)
        if not isinstance(raw_name, str):
            continue
        clean = " ".join(raw_name.strip().split())[:45]
        if not clean:
            continue
        cand = clean
        n_idx = 2
        while cand.lower() in used_names:
            cand = f"{clean} ({n_idx})"
            n_idx += 1
        used_names.add(cand.lower())
        c["auto_vibe_label"] = cand
        c["vibe_label"] = custom_map.get(cid_str) or cand
        updated += 1

    return updated > 0


def _compute_cluster_telemetry(cluster_idx, track_ids, metadata):
    bpms, energies, valences, dances, acoustics = [], [], [], [], []
    real_count = 0
    for idx in cluster_idx:
        tid = str(track_ids[idx])
        af = metadata.get(tid, {}).get("audio_features")
        if is_real_audio_features(af):
            real_count += 1
            if isinstance(af.get("bpm"), (int, float)):
                bpms.append(float(af["bpm"]))
            if isinstance(af.get("energy"), (int, float)):
                energies.append(float(af["energy"]))
            if isinstance(af.get("valence"), (int, float)):
                valences.append(float(af["valence"]))
            if isinstance(af.get("danceability"), (int, float)):
                dances.append(float(af["danceability"]))
            if isinstance(af.get("acousticness"), (int, float)):
                acoustics.append(float(af["acousticness"]))

    total = max(1, len(cluster_idx))
    return {
        "avg_bpm": round(sum(bpms) / len(bpms), 1) if bpms else None,
        "avg_energy": round(sum(energies) / len(energies), 2) if energies else None,
        "avg_valence": round(sum(valences) / len(valences), 2) if valences else None,
        "avg_danceability": round(sum(dances) / len(dances), 2) if dances else None,
        "avg_acousticness": round(sum(acoustics) / len(acoustics), 2) if acoustics else None,
        "telemetry_coverage_pct": round((real_count / total) * 100.0, 1),
    }


def compute_yearly_centroids(track_ids, embeddings, metadata, history_by_id=None, history_by_fp=None):
    """Compute play-weighted unit-normalized vector centroids per calendar year for era trajectory."""
    import numpy as np

    if len(track_ids) == 0 or len(embeddings) == 0 or (not history_by_id and not history_by_fp):
        return []

    history_by_id = history_by_id or {}
    history_by_fp = history_by_fp or {}
    dim = embeddings.shape[1]
    year_sums = {}
    year_plays = {}
    year_tracks = {}

    for idx, raw_tid in enumerate(track_ids):
        tid = str(raw_tid)
        meta = metadata.get(tid, {})
        candidate_ids = [tid] + [str(a) for a in (meta.get("alt_ids") or [])]

        timestamps = []
        for cid in candidate_ids:
            rec = history_by_id.get(cid)
            if rec and rec.get("stream_timestamps"):
                timestamps.extend(rec["stream_timestamps"])
        if not timestamps:
            fp = get_track_fingerprint(meta.get("name", ""), meta.get("artists", ""))
            fp_rec = history_by_fp.get(fp)
            if fp_rec and fp_rec.get("stream_timestamps"):
                timestamps.extend(fp_rec["stream_timestamps"])

        if not timestamps:
            continue

        yr_counts = {}
        for ts in set(timestamps):
            dt = parse_spotify_timestamp(ts)
            if dt is not None:
                yr_counts[dt.year] = yr_counts.get(dt.year, 0) + 1

        vec = embeddings[idx]
        for yr, cnt in yr_counts.items():
            if yr not in year_sums:
                year_sums[yr] = np.zeros(dim, dtype=float)
                year_plays[yr] = 0
                year_tracks[yr] = set()
            year_sums[yr] += vec * float(cnt)
            year_plays[yr] += cnt
            year_tracks[yr].add(tid)

    out = []
    for yr in sorted(year_sums):
        v = year_sums[yr]
        norm = float(np.linalg.norm(v))
        if norm > 0:
            v = v / norm
        out.append(
            {
                "year": int(yr),
                "plays": int(year_plays[yr]),
                "unique_tracks": len(year_tracks[yr]),
                "centroid_vector": v.tolist(),
            }
        )
    return out


def _filter_by_era(kept_indices, track_ids, metadata, history_by_id, history_by_fp, era_start, era_end):
    if era_start is None and era_end is None:
        return kept_indices, set()
    lo = int(era_start) if era_start is not None else 1900
    hi = int(era_end) if era_end is not None else 2100

    era_kept = []
    era_excluded = set()
    for idx in kept_indices:
        tid = str(track_ids[idx])
        meta = metadata.get(tid, {})
        candidate_ids = [tid] + [str(a) for a in (meta.get("alt_ids") or [])]

        timestamps = []
        for cid in candidate_ids:
            rec = history_by_id.get(cid)
            if rec and rec.get("stream_timestamps"):
                timestamps.extend(rec["stream_timestamps"])
        if not timestamps:
            fp = get_track_fingerprint(meta.get("name", ""), meta.get("artists", ""))
            fp_rec = history_by_fp.get(fp)
            if fp_rec and fp_rec.get("stream_timestamps"):
                timestamps.extend(fp_rec["stream_timestamps"])

        in_era = False
        if timestamps:
            for ts in timestamps:
                dt = parse_spotify_timestamp(ts)
                if dt is not None and lo <= dt.year <= hi:
                    in_era = True
                    break
        else:
            yr_str = str(meta.get("year") or "")[:4]
            if yr_str.isdigit() and lo <= int(yr_str) <= hi:
                in_era = True

        if in_era:
            era_kept.append(idx)
        else:
            era_excluded.add(tid)
    return era_kept, era_excluded


def main():
    configure_utf8_stdio()
    parser = argparse.ArgumentParser(description="Music Taste Analyzer - Stage 2 Taste Clusterer")
    parser.add_argument("--avoid-artists", type=str, default=None, help="Comma-separated artists to exclude from taste modeling")
    parser.add_argument("--avoid-genres", type=str, default=None, help="Comma-separated genres to exclude from taste modeling")
    parser.add_argument("--half-life", type=float, default=None, help="Time-decay half-life in days")
    parser.add_argument("--no-skip-penalty", action="store_true", help="Disable skip-rate negative weighting")
    parser.add_argument("--era-start", type=int, default=None, help="Optional start year (YYYY) for era taste modeling")
    parser.add_argument("--era-end", type=int, default=None, help="Optional end year (YYYY) for era taste modeling")
    parser.add_argument(
        "--provider",
        choices=["spotify", "ytmusic", "unified"],
        default=None,
        help="Music provider profile to load (default: active preference)",
    )
    parser.add_argument(
        "--method",
        choices=["auto", "hdbscan", "gmm", "kmeans"],
        default="auto",
        help="Clustering algorithm: hdbscan (UMAP+HDBSCAN with outliers), gmm (PCA+GMM+BIC), or kmeans (default: auto)",
    )
    parser.add_argument("--k", type=int, default=None, help="Optional fixed number of clusters")
    parser.add_argument("--no-llm-names", action="store_true", help="Skip batched Ollama cluster naming")
    args = parser.parse_args()

    import numpy as np
    from sklearn.cluster import KMeans
    from sklearn.metrics.pairwise import cosine_similarity

    prefs = get_effective_preferences()
    avoid_artists = (
        normalize_avoid_list(args.avoid_artists) if args.avoid_artists is not None else prefs["avoid_artists"]
    )
    avoid_genres = (
        normalize_avoid_list(args.avoid_genres) if args.avoid_genres is not None else prefs["avoid_genres"]
    )
    half_life_days = float(args.half_life) if args.half_life is not None else float(prefs["half_life_days"])
    skip_penalty_enabled = False if args.no_skip_penalty else bool(prefs["skip_penalty_enabled"])
    custom_labels = prefs.get("cluster_labels") or {}

    print("==================================================")
    print(" Taste Clustering Engine - Stage 2")
    print("==================================================")

    print("[1/3] Loading Vector Matrix...")
    base_dir = os.path.dirname(os.path.abspath(__file__))
    npz_path = os.path.join(base_dir, "taste_vectors.npz")
    meta_path = os.path.join(base_dir, "track_metadata.json")

    if not os.path.exists(npz_path) or not os.path.exists(meta_path):
        print("❌ Error: Missing vector or metadata files. Run npu_vectorizer.py first.")
        sys.exit(1)

    with np.load(npz_path) as data:
        embeddings = np.array(data["embeddings"])
        track_ids = [str(tid) for tid in data["track_ids"]]

    metadata = load_json_safe(meta_path)
    if not isinstance(metadata, dict) or len(embeddings) == 0:
        print("❌ Error: Vector matrix or metadata is empty.")
        sys.exit(1)

    print(f"      ✅ Loaded {embeddings.shape[0]} tracks of {embeddings.shape[1]}-dimensions.")

    eff_provider = args.provider or prefs.get("active_provider", "spotify")
    profile = (
        load_merged_music_profile(
            stage1_dir=os.path.join(ROOT_DIR, "01_Data_Extractor"),
            provider=eff_provider,
        )
        or {}
    )
    history_by_id, history_by_fp = load_extended_history(profile)

    kept_indices, excluded_ids = filter_excluded_tracks(
        track_ids,
        metadata,
        avoid_artists=avoid_artists,
        avoid_genres=avoid_genres,
    )
    if excluded_ids:
        print(
            f"      🚫 Session Taste Exclusions active: excluded {len(excluded_ids)} tracks "
            f"(Artists: {avoid_artists or 'None'} | Genres: {avoid_genres or 'None'})"
        )
    if not kept_indices:
        print("      ⚠️ Exclusions matched every track; keeping full library for clustering.")
        kept_indices = list(range(len(track_ids)))
        excluded_ids = set()

    if args.era_start is not None or args.era_end is not None:
        era_kept, era_excl = _filter_by_era(
            kept_indices, track_ids, metadata, history_by_id, history_by_fp, args.era_start, args.era_end
        )
        if era_kept:
            kept_indices = era_kept
            excluded_ids |= era_excl
            print(f"      🕰️ Era Filter ({args.era_start or 'start'}–{args.era_end or 'now'}): {len(kept_indices)} matching tracks.")
        else:
            print("      ⚠️ Era filter matched 0 tracks; keeping full non-excluded library.")

    kept_track_ids = [track_ids[i] for i in kept_indices]
    kept_embeddings = embeddings[kept_indices]

    print("\n[2/3] Calculating Recency Bias & Skip-Adjusted Weights...")
    penalized_artists = load_penalized_artists()
    skip_ratios = {}

    weights_list = compute_recency_weights(
        kept_track_ids,
        metadata,
        profile,
        history_by_id=history_by_id,
        history_by_fp=history_by_fp,
        penalized_artists=penalized_artists,
        half_life_days=half_life_days,
        skip_penalty_enabled=skip_penalty_enabled,
        skip_ratios_out=skip_ratios,
        era_start=args.era_start,
        era_end=args.era_end,
    )
    weights = np.array(weights_list, dtype=float)
    total_weight_sum = float(weights.sum())
    if total_weight_sum <= 0.0:
        weights = np.ones(len(kept_track_ids), dtype=float)
        total_weight_sum = float(weights.sum())

    print(
        f"      ✅ Applied recency bias (Half-life: {half_life_days:.0f}d | Max weight: {weights.max():.2f} | "
        f"Skip penalty: {'ON' if skip_penalty_enabled else 'OFF'})"
    )

    cluster_method = (
        getattr(config, "DEFAULT_CLUSTER_METHOD", "hdbscan") if args.method == "auto" else args.method
    )
    if args.k is not None and cluster_method == "hdbscan":
        cluster_method = "gmm"
    min_k = int(getattr(config, "MIN_CLUSTERS", 3))
    max_k = int(getattr(config, "MAX_CLUSTERS", 14))
    max_cluster_share = float(getattr(config, "MAX_CLUSTER_SHARE", 0.22))
    pca_dims = int(getattr(config, "GMM_PCA_DIMS", 20))

    probs = None
    if cluster_method == "hdbscan":
        umap_dims = int(getattr(config, "HDBSCAN_UMAP_DIMS", 10))
        min_ratio = float(getattr(config, "HDBSCAN_MIN_CLUSTER_RATIO", 0.018))
        print(f"\n[3/3] Analyzing Taste Sub-Clusters (UMAP-{umap_dims} + HDBSCAN Density)...")
        labels, cluster_centers, probs, bic_scores = fit_hdbscan_clusters(
            kept_embeddings,
            weights,
            umap_dims=umap_dims,
            min_cluster_ratio=min_ratio,
            max_cluster_share=max_cluster_share,
        )
        n_clusters = len(cluster_centers)
        noise_cnt = int(np.sum(labels < 0))
        print(f"      -> Discovered {n_clusters} dense clusters ({noise_cnt} unclustered / eclectic outlier tracks)")
    elif cluster_method == "gmm":
        print(f"\n[3/3] Analyzing Taste Sub-Clusters (Centered PCA-{pca_dims} + GMM + BIC)...")
        labels, cluster_centers, probs, bic_scores = fit_gmm_clusters(
            kept_embeddings,
            weights,
            min_k=min_k,
            max_k=max_k,
            fixed_k=args.k,
            pca_dims=pca_dims,
            max_cluster_share=max_cluster_share,
        )
        n_clusters = len(cluster_centers)
        if bic_scores:
            bic_line = " | ".join(f"k={kk}:{vv:.0f}" for kk, vv in sorted(bic_scores.items()))
            print(f"      -> Selected k={n_clusters} via BIC ({bic_line})")
    else:
        print("\n[3/3] Analyzing Taste Sub-Clusters (Adaptive Time-Weighted K-Means)...")
        n_clusters = (
            max(1, min(int(args.k), len(kept_embeddings) - 1))
            if args.k is not None
            else choose_optimal_k(kept_embeddings, weights, min_k=min_k, max_k=max_k)
        )
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        kmeans.fit(kept_embeddings, sample_weight=weights)
        labels = kmeans.labels_
        cluster_centers = kmeans.cluster_centers_

    master_centroid = np.average(kept_embeddings, axis=0, weights=weights)
    mc_norm = np.linalg.norm(master_centroid)
    if mc_norm > 0:
        master_centroid = master_centroid / mc_norm

    global_genre_counts = {}
    for tid in kept_track_ids:
        g_str = metadata[tid].get("genres", "")
        if g_str:
            for g in {g.strip().lower() for g in g_str.split(",") if g.strip()}:
                global_genre_counts[g] = global_genre_counts.get(g, 0) + 1

    total_tracks_count = max(1, len(kept_track_ids))
    used_vibe_names = set()
    used_artists = set()
    clusters_out = []
    raw_to_seq_id = {}

    for raw_cid in range(n_clusters):
        cluster_idx = np.where(labels == raw_cid)[0]
        cluster_size = len(cluster_idx)
        if cluster_size == 0:
            continue

        seq_cid = len(clusters_out)
        raw_to_seq_id[raw_cid] = seq_cid

        center = cluster_centers[raw_cid]
        c_norm = np.linalg.norm(center)
        if c_norm > 0:
            center = center / c_norm

        cluster_weight_sum = float(weights[cluster_idx].sum())
        weighted_share_pct = round((cluster_weight_sum / total_weight_sum) * 100.0, 1)
        raw_pct = round((cluster_size / len(kept_embeddings)) * 100.0, 1)

        genre_freq = {}
        for idx in cluster_idx:
            tid = kept_track_ids[idx]
            t_weight = float(weights[idx])
            g_str = metadata[tid].get("genres", "")
            if g_str:
                for g in (g.strip().lower() for g in g_str.split(",") if g.strip()):
                    genre_freq[g] = genre_freq.get(g, 0.0) + t_weight

        tfidf_genres = {}
        for g, freq in genre_freq.items():
            idf = math.log((total_tracks_count + 1.0) / (global_genre_counts.get(g, 1) + 1.0))
            tfidf_genres[g] = freq * (max(0.0, idf) ** 0.7)

        top_genres = sorted(tfidf_genres.items(), key=lambda x: x[1], reverse=True)
        top_genre_names = [g[0].title() for g in top_genres[:3]]

        similarities = cosine_similarity([center], kept_embeddings[cluster_idx])[0]
        top_indices = np.argsort(similarities)[::-1][:10]

        anchor_tracks = []
        for idx in top_indices:
            global_idx = cluster_idx[idx]
            tid = kept_track_ids[global_idx]
            anchor_tracks.append(
                {
                    "name": metadata[tid]["name"],
                    "artist": metadata[tid]["artists"],
                    "similarity": round(float(similarities[idx]), 3),
                }
            )

        auto_vibe_title = build_vibe_label(
            seq_cid,
            top_genre_names,
            anchor_tracks,
            cluster_size,
            used_vibe_names,
            used_artists,
        )
        vibe_title = custom_labels.get(str(seq_cid)) or auto_vibe_title
        telemetry = _compute_cluster_telemetry(cluster_idx, kept_track_ids, metadata)

        clusters_out.append(
            {
                "cluster_id": seq_cid,
                "vibe_label": vibe_title,
                "auto_vibe_label": auto_vibe_title,
                "top_genres": top_genre_names,
                "track_count": cluster_size,
                "raw_percentage": raw_pct,
                "weighted_share_pct": weighted_share_pct,
                "anchor_tracks": anchor_tracks,
                "centroid_vector": center.tolist(),
                "telemetry": telemetry,
            }
        )

    if not args.no_llm_names:
        if name_clusters_with_ollama(clusters_out, prefs=prefs, custom_labels=custom_labels):
            print("      -> Refined cluster names with local Ollama LLM")

    for c in clusters_out:
        telemetry = c.get("telemetry") or {}
        bpm_str = f" | Avg BPM: {telemetry['avg_bpm']}" if telemetry.get("avg_bpm") is not None else ""
        print(
            f"      -> Cluster {c['cluster_id']}: [{c['vibe_label']}] ({c['track_count']} tracks | "
            f"{c['weighted_share_pct']}% Recency Share | Raw: {c['raw_percentage']}%{bpm_str})"
        )
        if c.get("anchor_tracks"):
            print(f"         Anchor: '{c['anchor_tracks'][0]['name']}' by {c['anchor_tracks'][0]['artist']}")

    for kept_pos, tid in enumerate(kept_track_ids):
        metadata[tid]["recency_weight"] = float(weights[kept_pos])
        raw_cid = int(labels[kept_pos])
        metadata[tid]["cluster_id"] = -1 if raw_cid < 0 else int(raw_to_seq_id.get(raw_cid, -1))
        metadata[tid]["excluded_from_taste"] = False
        if probs is not None and raw_cid >= 0:
            row_p = probs[kept_pos]
            top2_idx = np.argsort(row_p)[::-1][:2]
            metadata[tid]["cluster_probs"] = [
                {"cluster_id": int(raw_to_seq_id.get(int(ci), 0)), "prob": round(float(row_p[ci]), 3)}
                for ci in top2_idx
                if float(row_p[ci]) >= 0.05 and int(ci) in raw_to_seq_id
            ]
        else:
            metadata[tid].pop("cluster_probs", None)
        if tid in skip_ratios:
            metadata[tid]["skip_ratio"] = skip_ratios[tid]

    for tid in excluded_ids:
        metadata[tid]["recency_weight"] = 0.0
        metadata[tid]["cluster_id"] = -1
        metadata[tid]["excluded_from_taste"] = True

    atomic_write_json(meta_path, metadata, indent=2)

    yearly_centroids = compute_yearly_centroids(
        kept_track_ids,
        kept_embeddings,
        metadata,
        history_by_id=history_by_id,
        history_by_fp=history_by_fp,
    )

    profile_summary = {
        "cluster_method": cluster_method,
        "master_taste_centroid": master_centroid.tolist(),
        "clusters": clusters_out,
        "unclustered_track_count": int(np.sum(labels < 0)),
        "yearly_centroids": yearly_centroids,
        "session_filters": {
            "active_provider": eff_provider,
            "avoid_artists": avoid_artists,
            "avoid_genres": avoid_genres,
            "excluded_track_count": len(excluded_ids),
            "skip_penalty_enabled": skip_penalty_enabled,
            "half_life_days": half_life_days,
            "era_start": args.era_start,
            "era_end": args.era_end,
        },
    }
    print("\n[3/3] Exporting Taste Profile Summary & Metadata Weights...")
    summary_out_path = os.path.join(base_dir, "taste_profile_summary.json")
    atomic_write_json(summary_out_path, profile_summary, indent=2)

    print("      ✅ Saved 'taste_profile_summary.json' and updated 'track_metadata.json'")
    print("\n🚀 Stage 2 Complete! Your musical identity is now fully vectorized.")


if __name__ == "__main__":
    main()
