"""Stage 3 CLAP prompt-to-audio scoring, lazy sub-vibe centroids, and Deezer preview verification."""

import os
import re
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
STAGE1_DIR = os.path.join(ROOT_DIR, "01_Data_Extractor")
STAGE2_DIR = os.path.join(ROOT_DIR, "02_Vector_Core")
for _p in (ROOT_DIR, STAGE1_DIR, STAGE2_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from utils import (
    atomic_output_file,
    format_audio_feature_text,
    format_camelot_label,
    get_track_fingerprint,
    is_real_audio_features,
)


def is_real_clap_model(model):
    """Return True only when model is an actual CLAP model (avoids MagicMock truthiness)."""
    return getattr(model, "is_clap", False) is True


def _l2_normalize_vec(vec):
    arr = np.asarray(vec, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(arr))
    if norm > 1e-8:
        return (arr / norm).astype(np.float32)
    return arr


def _l2_normalize_rows(mat):
    arr = np.asarray(mat, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if len(arr) == 0:
        return arr
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms = np.where(norms > 1e-8, norms, 1.0)
    return (arr / norms).astype(np.float32)


def _standardize(values):
    arr = np.asarray(values, dtype=np.float32)
    n = len(arr)
    if n == 0:
        return arr
    mean = float(np.mean(arr))
    if n >= 3:
        std = float(np.std(arr))
        if std > 1e-6:
            return ((arr - mean) / std).astype(np.float32)
    return (arr - mean).astype(np.float32)


def load_raw_clap_audio_cache(stage2_dir=None):
    """Load Stage 2's raw 512-D CLAP audio vectors keyed by normalized track fingerprint."""
    cache_path = os.path.join(stage2_dir or STAGE2_DIR, "clap_audio_cache.npz")
    if not os.path.exists(cache_path):
        return {}
    out = {}
    try:
        with np.load(cache_path, allow_pickle=False) as d:
            fps = d["fingerprints"]
            vecs = d["vectors"]
            if len(fps) == len(vecs):
                for fp, vec in zip(fps, vecs):
                    out[str(fp)] = np.asarray(vec, dtype=np.float32)
    except Exception:
        return {}
    return out


def save_raw_clap_audio_cache(cache_dict, stage2_dir=None):
    """Atomically persist raw 512-D CLAP audio vectors to clap_audio_cache.npz."""
    if not cache_dict:
        return
    cache_path = os.path.join(stage2_dir or STAGE2_DIR, "clap_audio_cache.npz")
    fps = np.array(list(cache_dict.keys()))
    vecs = np.vstack([np.asarray(cache_dict[k], dtype=np.float32) for k in cache_dict]).astype(np.float32)
    with atomic_output_file(cache_path, binary=True) as f:
        np.savez(f, fingerprints=fps, vectors=vecs)


def compute_subcluster_centroid(local_embeddings, local_ids, local_meta, cluster_id, sub_cluster_id):
    """Lazily compute a recency-weighted centroid for a Level-2 sub-vibe from Stage 2 artifacts."""
    if local_embeddings is None or not local_ids or not isinstance(local_meta, dict):
        return None, []
    try:
        cid_target = int(cluster_id)
        sid_target = int(sub_cluster_id)
    except (TypeError, ValueError):
        return None, []

    matching_indices = []
    weights = []
    for idx, tid in enumerate(local_ids):
        info = local_meta.get(str(tid)) or {}
        if info.get("excluded_from_taste"):
            continue
        if info.get("cluster_id") == cid_target and info.get("sub_cluster_id") == sid_target:
            matching_indices.append(idx)
            weights.append(max(0.05, float(info.get("recency_weight") or 1.0)))

    if not matching_indices:
        return None, []

    sub_vecs = np.asarray(local_embeddings[matching_indices], dtype=np.float64)
    w_arr = np.asarray(weights, dtype=np.float64)
    centroid = np.average(sub_vecs, axis=0, weights=w_arr)
    norm = float(np.linalg.norm(centroid))
    if norm > 1e-8:
        centroid = centroid / norm
    return centroid.astype(np.float32), matching_indices


def build_clap_candidate_text(name, artists, genres="", year="", audio_features=None):
    """Format a genre-first acoustic text description matching Stage 2's CLAP encoder."""
    raw_name = str(name or "Unknown")
    clean_name = re.sub(r"\s*\|.*$", "", raw_name)
    clean_name = re.sub(r"\s+", " ", clean_name).strip() or raw_name

    raw_artist = str(artists or "Unknown")
    clean_artist = re.sub(r"(?i)(?:,\s*)?\b\d+(?:\.\d+)?\s*[kmb]?\s+views?\b", "", raw_artist)
    clean_artist = re.sub(r"\s+", " ", clean_artist).strip(" ,") or raw_artist

    parts = []
    genres_clean = str(genres or "").strip()
    if genres_clean:
        parts.append(f"Music genres: {genres_clean}")
    parts.append(f"Track: {clean_name} by {clean_artist}")
    year_clean = str(year or "").strip()
    if year_clean:
        parts.append(f"Year: {year_clean}")
    af_text = format_audio_feature_text(audio_features or {})
    if af_text:
        parts.append(af_text)
    return ". ".join(parts)


def score_local_library_candidates(
    prompt_vec,
    taste_vec,
    local_embeddings,
    local_ids,
    local_meta,
    valid_indices,
    raw_audio_cache=None,
    is_clap=False,
    taste_weight=0.30,
):
    """
    Score local library candidates against prompt and taste centroid.
    When CLAP + raw audio vectors are available, scores prompts directly against raw
    CLAP audio vectors while scoring taste similarity in Stage 2's centered space.
    """
    if not valid_indices:
        return np.empty(0, dtype=np.float32), np.empty(0, dtype=np.float32)

    tw = max(0.0, min(1.0, float(taste_weight)))
    pw = 1.0 - tw
    p_norm = _l2_normalize_vec(prompt_vec)
    t_norm = _l2_normalize_vec(taste_vec)
    sub_embs = _l2_normalize_rows(local_embeddings[valid_indices])
    dim = sub_embs.shape[1]

    if not is_clap or not raw_audio_cache or len(p_norm) != dim:
        target = _l2_normalize_vec((pw * p_norm) + (tw * t_norm))
        combined = (sub_embs @ target).astype(np.float32)
        raw_prompt_sims = (sub_embs @ p_norm).astype(np.float32)
        return combined, raw_prompt_sims

    taste_sims = (sub_embs @ t_norm).astype(np.float32)
    prompt_sims = np.empty(len(valid_indices), dtype=np.float32)
    valid_prompt_vals = []
    missing_positions = []

    for pos, orig_idx in enumerate(valid_indices):
        tid = str(local_ids[orig_idx])
        info = local_meta.get(tid) or {}
        fp = get_track_fingerprint(info.get("name", ""), info.get("artists", ""))
        raw_vec = raw_audio_cache.get(fp)
        if raw_vec is not None and len(raw_vec) == dim:
            rv_norm = _l2_normalize_vec(raw_vec)
            sim = float(np.dot(p_norm, rv_norm))
            prompt_sims[pos] = sim
            valid_prompt_vals.append(sim)
        else:
            missing_positions.append(pos)

    if not valid_prompt_vals:
        target = _l2_normalize_vec((pw * p_norm) + (tw * t_norm))
        combined = (sub_embs @ target).astype(np.float32)
        raw_prompt_sims = (sub_embs @ p_norm).astype(np.float32)
        return combined, raw_prompt_sims

    if missing_positions:
        fill_val = float(np.median(valid_prompt_vals))
        for pos in missing_positions:
            prompt_sims[pos] = fill_val

    z_prompt = _standardize(prompt_sims)
    z_taste = _standardize(taste_sims)
    combined = ((pw * z_prompt) + (tw * z_taste)).astype(np.float32)
    return combined, prompt_sims


def compute_scored_mmr(relevance_scores, candidate_embeddings, top_k, lambda_param=0.65):
    """Maximal Marginal Relevance using explicit pre-computed relevance scores."""
    from sklearn.metrics.pairwise import cosine_similarity

    n_items = len(candidate_embeddings)
    if top_k <= 0 or n_items == 0:
        return []

    rel_raw = np.asarray(relevance_scores, dtype=np.float64).reshape(-1)
    if len(rel_raw) != n_items:
        raise ValueError("relevance_scores length must match candidate_embeddings.")

    if n_items <= top_k:
        return np.argsort(rel_raw)[::-1].tolist()

    r_min = float(np.min(rel_raw))
    r_max = float(np.max(rel_raw))
    if (r_max - r_min) > 1e-9:
        rel = (rel_raw - r_min) / (r_max - r_min)
    else:
        rel = np.full(n_items, 0.5, dtype=np.float64)

    selected = []
    unselected = list(range(n_items))

    best_first = int(np.argmax(rel))
    selected.append(best_first)
    unselected.remove(best_first)

    embs = np.asarray(candidate_embeddings, dtype=np.float64)
    while len(selected) < top_k and unselected:
        sub_vecs = embs[unselected]
        sel_vecs = embs[selected]
        sims_to_selected = cosine_similarity(sub_vecs, sel_vecs)
        max_sims_to_sel = np.max(sims_to_selected, axis=1)
        target_rels = rel[unselected]

        mmr_scores = (lambda_param * target_rels) - ((1.0 - lambda_param) * max_sims_to_sel)
        best_pos = int(np.argmax(mmr_scores))
        selected.append(unselected[best_pos])
        unselected.pop(best_pos)

    return selected


def _passes_relaxed_constraints(af, constraints):
    if not constraints or not is_real_audio_features(af):
        return True
    min_bpm = constraints.get("min_bpm")
    max_bpm = constraints.get("max_bpm")
    target_energy = constraints.get("target_energy")
    target_valence = constraints.get("target_valence")
    bpm = af.get("bpm")
    energy = af.get("energy")
    valence = af.get("valence")

    if min_bpm and isinstance(bpm, (int, float)) and float(bpm) < (float(min_bpm) * 0.85):
        return False
    if max_bpm and isinstance(bpm, (int, float)) and float(bpm) > (float(max_bpm) * 1.15):
        return False
    if target_energy is not None and isinstance(energy, (int, float)):
        te = float(target_energy)
        ev = float(energy)
        if te >= 0.7 and ev < 0.40:
            return False
        if te <= 0.4 and ev > 0.65:
            return False
    if target_valence is not None and isinstance(valence, (int, float)):
        tv = float(target_valence)
        vv = float(valence)
        if tv >= 0.7 and vv < 0.35:
            return False
        if tv <= 0.4 and vv > 0.65:
            return False
    return True


def _apply_dsp_and_genres_to_candidate(cand, dsp_af, dz_genres):
    changed_dsp = False
    changed_genres = False
    if is_real_audio_features(dsp_af):
        if cand.get("bpm") is None and dsp_af.get("bpm") is not None:
            cand["bpm"] = dsp_af["bpm"]
            changed_dsp = True
        if cand.get("energy") is None and dsp_af.get("energy") is not None:
            cand["energy"] = dsp_af["energy"]
            changed_dsp = True
        if cand.get("valence") is None and dsp_af.get("valence") is not None:
            cand["valence"] = dsp_af["valence"]
            changed_dsp = True
        if cand.get("key") is None and dsp_af.get("key") is not None:
            cand["key"] = dsp_af["key"]
        if cand.get("mode") is None and dsp_af.get("mode") is not None:
            cand["mode"] = dsp_af["mode"]
        if not cand.get("camelot") and cand.get("key") is not None and cand.get("mode") is not None:
            cand["camelot"] = format_camelot_label(cand["key"], cand["mode"])
    if dz_genres and not (cand.get("genres") or "").strip():
        cand["genres"] = ", ".join(str(g).strip().lower() for g in dz_genres[:3] if str(g).strip())
        if cand["genres"]:
            changed_genres = True
    return changed_dsp, changed_genres


def verify_and_rerank_discovery_candidates(
    candidates,
    prompt_vec,
    taste_vec,
    embed_tokenizer,
    embed_model,
    stage2_dir=None,
    max_verify=30,
    taste_weight=0.30,
    constraints=None,
    log_cb=None,
):
    """
    Fetch 30s Deezer previews + DSP telemetry for top discovery finalists, run CLAP audio
    inference (caching to clap_audio_cache.npz), and re-rank by audio+taste alignment.
    """
    stats = {
        "checked": 0,
        "audio_verified": 0,
        "cache_hits": 0,
        "dsp_enriched": 0,
        "genres_enriched": 0,
    }
    if not candidates or not is_real_clap_model(embed_model):
        return list(candidates or []), stats

    import clap_embedder
    import deezer_audio_client as dz

    cap = max(1, int(max_verify or 30))
    pick_indices = [i for i, c in enumerate(candidates) if c.get("source") == "AI Curator Pick"]
    other_indices = [i for i, c in enumerate(candidates) if c.get("source") != "AI Curator Pick"]
    checked_order = (pick_indices + other_indices)[:cap]
    checked_set = set(checked_order)
    unchecked_tail = [dict(c) for i, c in enumerate(candidates) if i not in checked_set]
    checked_cands = [dict(candidates[i]) for i in checked_order]
    stats["checked"] = len(checked_cands)

    deezer_cache = dz.load_deezer_cache()
    album_genres_cache = dz.load_deezer_album_genres_cache()
    dsp_cache = dz.load_dsp_cache()
    raw_audio_cache = load_raw_clap_audio_cache(stage2_dir=stage2_dir)

    dz_dirty = False
    alb_dirty = False
    dsp_dirty = False
    raw_dirty = False
    cache_lock = threading.Lock()

    p_norm = _l2_normalize_vec(prompt_vec)
    t_norm = _l2_normalize_vec(taste_vec)
    dim = len(p_norm)

    def _resolve_one(pos):
        nonlocal dz_dirty, alb_dirty, dsp_dirty
        cand = checked_cands[pos]
        name = cand.get("name", "")
        artist = cand.get("artist", "")
        fp = get_track_fingerprint(name, artist)

        with cache_lock:
            cached_raw = raw_audio_cache.get(fp)
            cached_dz = deezer_cache.get(fp)
            cached_dsp = dsp_cache.get(fp)

        has_raw = cached_raw is not None and len(cached_raw) == dim
        if has_raw:
            dz_genres = []
            aid = (cached_dz or {}).get("album_id") if isinstance(cached_dz, dict) else None
            if aid and not (cand.get("genres") or "").strip():
                with cache_lock:
                    dz_genres = list(album_genres_cache.get(str(aid)) or [])
                if not dz_genres:
                    dz_genres = dz.fetch_deezer_album_genres(aid, cache=None)
                    with cache_lock:
                        album_genres_cache[str(aid)] = dz_genres
                        alb_dirty = True
            return pos, fp, cached_raw, None, cached_dsp if is_real_audio_features(cached_dsp) else None, dz_genres, True

        if cached_dz is None or (isinstance(cached_dz, dict) and not cached_dz.get("found") and cached_dz.get("v") != 2):
            cached_dz = dz.search_deezer_track(name, artist, cache=None)
            with cache_lock:
                deezer_cache[fp] = cached_dz
                dz_dirty = True

        wav = None
        if isinstance(cached_dz, dict) and cached_dz.get("found"):
            wav, fresh_url = dz.download_preview_waveform(
                cached_dz.get("preview_url", ""),
                deezer_id=cached_dz.get("deezer_id"),
                target_sr=48000,
            )
            if fresh_url and fresh_url != cached_dz.get("preview_url"):
                with cache_lock:
                    cached_dz = dict(cached_dz, preview_url=fresh_url)
                    deezer_cache[fp] = cached_dz
                    dz_dirty = True

        dsp_af = cached_dsp if is_real_audio_features(cached_dsp) else None
        needs_af = (cand.get("bpm") is None or cand.get("energy") is None or cand.get("valence") is None)
        if dsp_af is None and wav is not None and needs_af:
            try:
                dsp_af = dz.extract_dsp_audio_features(
                    wav, sr=48000, deezer_bpm=(cached_dz or {}).get("bpm")
                )
                with cache_lock:
                    dsp_cache[fp] = dsp_af
                    dsp_dirty = True
            except Exception:
                dsp_af = None

        dz_genres = []
        aid = (cached_dz or {}).get("album_id") if isinstance(cached_dz, dict) else None
        if aid and not (cand.get("genres") or "").strip():
            with cache_lock:
                dz_genres = list(album_genres_cache.get(str(aid)) or [])
            if not dz_genres:
                dz_genres = dz.fetch_deezer_album_genres(aid, cache=None)
                with cache_lock:
                    album_genres_cache[str(aid)] = dz_genres
                    alb_dirty = True

        return pos, fp, None, wav, dsp_af, dz_genres, False

    fetched_rows = [None] * len(checked_cands)
    with ThreadPoolExecutor(max_workers=4) as pool:
        futs = [pool.submit(_resolve_one, pos) for pos in range(len(checked_cands))]
        for fut in as_completed(futs):
            row = fut.result()
            fetched_rows[row[0]] = row

    device = getattr(embed_model, "_clap_device", None) or "cpu"
    audio_vecs = {}
    dsp_constraint_ok = [True] * len(checked_cands)
    text_fallback_positions = []

    for pos, fp, cached_raw, wav, dsp_af, dz_genres, is_hit in fetched_rows:
        cand = checked_cands[pos]
        had_missing_af = cand.get("bpm") is None or cand.get("energy") is None or cand.get("valence") is None
        changed_dsp, changed_genres = _apply_dsp_and_genres_to_candidate(cand, dsp_af, dz_genres)
        if changed_dsp:
            stats["dsp_enriched"] += 1
        if changed_genres:
            stats["genres_enriched"] += 1
        if had_missing_af and is_real_audio_features(dsp_af):
            dsp_constraint_ok[pos] = _passes_relaxed_constraints(dsp_af, constraints)

        if cached_raw is not None and len(cached_raw) == dim:
            audio_vecs[pos] = _l2_normalize_vec(cached_raw)
            stats["audio_verified"] += 1
            if is_hit:
                stats["cache_hits"] += 1
        elif wav is not None:
            try:
                vec = clap_embedder.embed_audio_waveform(
                    wav, embed_tokenizer, embed_model, device=device, max_windows=3, sr=48000
                )
                vec_norm = _l2_normalize_vec(vec)
                if len(vec_norm) == dim:
                    audio_vecs[pos] = vec_norm
                    raw_audio_cache[fp] = vec_norm
                    raw_dirty = True
                    stats["audio_verified"] += 1
                else:
                    text_fallback_positions.append(pos)
            except Exception:
                text_fallback_positions.append(pos)
        else:
            text_fallback_positions.append(pos)

    if dz_dirty:
        dz.save_deezer_cache(deezer_cache)
    if alb_dirty:
        dz.save_deezer_album_genres_cache(album_genres_cache)
    if dsp_dirty:
        dz.save_dsp_cache(dsp_cache)
    if raw_dirty:
        save_raw_clap_audio_cache(raw_audio_cache, stage2_dir=stage2_dir)

    valid_cache_vecs = [np.asarray(v, dtype=np.float32) for v in raw_audio_cache.values() if len(v) == dim]
    audio_mean = None
    if len(valid_cache_vecs) >= 5:
        audio_mean = np.mean(np.vstack(valid_cache_vecs), axis=0).astype(np.float32)

    text_vecs = {}
    if text_fallback_positions:
        try:
            fb_texts = [
                build_clap_candidate_text(
                    checked_cands[p].get("name", ""),
                    checked_cands[p].get("artist", ""),
                    genres=checked_cands[p].get("genres", ""),
                    audio_features={
                        "bpm": checked_cands[p].get("bpm"),
                        "energy": checked_cands[p].get("energy"),
                        "valence": checked_cands[p].get("valence"),
                        "source": "reccobeats",
                    }
                    if checked_cands[p].get("bpm") is not None
                    else None,
                )
                for p in text_fallback_positions
            ]
            t_mat = clap_embedder.embed_texts_clap(fb_texts, embed_tokenizer, embed_model, device=device)
            for idx_p, pos in enumerate(text_fallback_positions):
                if idx_p < len(t_mat) and len(t_mat[idx_p]) == dim:
                    text_vecs[pos] = _l2_normalize_vec(t_mat[idx_p])
        except Exception:
            pass

    prompt_sims = np.zeros(len(checked_cands), dtype=np.float32)
    taste_sims = np.zeros(len(checked_cands), dtype=np.float32)

    for pos, cand in enumerate(checked_cands):
        if pos in audio_vecs:
            avec = audio_vecs[pos]
            p_sim = float(np.dot(p_norm, avec))
            centered_avec = _l2_normalize_vec(avec - audio_mean) if audio_mean is not None else avec
            t_sim = float(np.dot(t_norm, centered_avec))
            cand["audio_verified"] = True
            cand["prompt_match"] = round(p_sim, 3)
            prompt_sims[pos] = p_sim
            taste_sims[pos] = t_sim
        elif pos in text_vecs:
            tvec = text_vecs[pos]
            p_sim = float(np.dot(p_norm, tvec))
            t_sim = float(np.dot(t_norm, tvec))
            cand["audio_verified"] = False
            cand["prompt_match"] = round(p_sim, 3)
            prompt_sims[pos] = p_sim
            taste_sims[pos] = t_sim
        else:
            cand["audio_verified"] = False

    tw = max(0.0, min(1.0, float(taste_weight)))
    pw = 1.0 - tw
    z_prompt = _standardize(prompt_sims)
    z_taste = _standardize(taste_sims)
    combined = (pw * z_prompt) + (tw * z_taste)

    for pos, cand in enumerate(checked_cands):
        if not cand.get("audio_verified"):
            combined[pos] -= 0.15
        if not dsp_constraint_ok[pos]:
            combined[pos] -= 0.35
        if cand.get("source") == "AI Curator Pick":
            combined[pos] += 0.10

    ranked_pos = np.argsort(combined)[::-1].tolist()
    reranked = [checked_cands[p] for p in ranked_pos] + unchecked_tail

    if log_cb is not None:
        msg = (
            f"      🎧 Live CLAP audio verification: {stats['audio_verified']}/{stats['checked']} tracks verified "
            f"({stats['cache_hits']} cached, {stats['dsp_enriched']} DSP, {stats['genres_enriched']} genres)."
        )
        try:
            log_cb(msg)
        except Exception:
            pass

    return reranked, stats


def build_stratified_curator_shortlist(
    local_tracks,
    discovered_tracks,
    target_size,
    target_local_count,
    max_candidates=45,
):
    """
    Build a balanced shortlist capped at max_candidates so abundant local tracks never starve
    discovered tracks or resolved AI Curator Picks in vibe_check_playlist, returning
    (shortlist, overflow_pool).
    """
    cap = max(1, int(max_candidates or 45))
    t_size = max(1, int(target_size or 30))
    t_local = max(0, min(t_size, int(target_local_count)))
    t_new = max(0, t_size - t_local)

    seen_ids = set()
    seen_fps = set()

    def _dedupe(tracks):
        out = []
        for tr in tracks or []:
            if not isinstance(tr, dict):
                continue
            tid = tr.get("id")
            fp = get_track_fingerprint(tr.get("name", ""), tr.get("artist", ""))
            if (tid and tid in seen_ids) or fp in seen_fps:
                continue
            if tid:
                seen_ids.add(tid)
            seen_fps.add(fp)
            out.append(tr)
        return out

    clean_local = _dedupe(local_tracks)
    clean_new = _dedupe(discovered_tracks)

    if t_new == 0 or not clean_new:
        local_slots = min(cap, len(clean_local))
        new_slots = min(len(clean_new), max(0, cap - local_slots))
    elif t_local == 0 or not clean_local:
        new_slots = min(cap, len(clean_new))
        local_slots = min(len(clean_local), max(0, cap - new_slots))
    else:
        desired_new = min(len(clean_new), max(min(len(clean_new), t_new + 2), round(cap * (t_new / float(t_size)))))
        desired_local = min(len(clean_local), max(min(len(clean_local), t_local + 2), round(cap * (t_local / float(t_size)))))
        if desired_local + desired_new > cap:
            local_slots = min(desired_local, max(0, cap - desired_new))
            new_slots = min(len(clean_new), cap - local_slots)
        else:
            local_slots = desired_local
            new_slots = desired_new
            rem = cap - (local_slots + new_slots)
            if rem > 0 and len(clean_new) > new_slots:
                add_new = min(rem, len(clean_new) - new_slots)
                new_slots += add_new
                rem -= add_new
            if rem > 0 and len(clean_local) > local_slots:
                add_loc = min(rem, len(clean_local) - local_slots)
                local_slots += add_loc

    picked_local = clean_local[:local_slots]
    rem_local = clean_local[local_slots:]

    ai_picks = [tr for tr in clean_new if tr.get("source") == "AI Curator Pick"]
    other_new = [tr for tr in clean_new if tr.get("source") != "AI Curator Pick"]
    guaranteed_picks = ai_picks[: min(10, new_slots)]
    extra_picks = ai_picks[len(guaranteed_picks) :]
    remaining_new_ordered = other_new + extra_picks

    needed_other = max(0, new_slots - len(guaranteed_picks))
    picked_new = guaranteed_picks + remaining_new_ordered[:needed_other]
    rem_new = remaining_new_ordered[needed_other:]

    shortlist = picked_local + picked_new
    overflow_pool = rem_local + rem_new
    return shortlist, overflow_pool
