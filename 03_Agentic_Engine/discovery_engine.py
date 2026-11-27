import argparse
import datetime
import gc
import json
import os
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
STAGE1_DIR = os.path.join(ROOT_DIR, "01_Data_Extractor")
STAGE2_DIR = os.path.join(ROOT_DIR, "02_Vector_Core")
for p in (ROOT_DIR, STAGE1_DIR, STAGE2_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

import config
import ytmusic_client
from llm_parser import get_llm, parse_prompt, parse_refinement_instruction, vibe_check_playlist
from stage1_extractor import authenticate_spotify
from utils import (
    add_items_to_spotify_playlist,
    atomic_write_json,
    configure_utf8_stdio,
    create_spotify_playlist,
    enrich_artist_genres_2026,
    extract_ytm_video_id,
    fetch_personal_release_radar,
    format_audio_feature_text,
    format_camelot_label,
    get_currently_playing_track,
    get_disliked_sets,
    get_effective_preferences,
    get_liked_ids,
    get_track_fingerprint,
    get_track_provider,
    get_user_playback_queue,
    is_real_audio_features,
    is_valid_spotify_id,
    is_valid_ytm_id,
    is_valid_ytm_video_id,
    load_archived_playlist,
    load_json_safe,
    matches_avoided_artist,
    matches_avoided_genre,
    normalize_avoid_list,
    queue_tracks_on_spotify,
    replace_spotify_playlist_items,
    save_playlist_to_archive,
    sequence_playlist_tracks,
    start_spotify_playback,
)

_OPERATOR_STOP_RE = re.compile(
    r"\b(artist|genre):(?!['\"])\s*(.+?)(?=\s+(?:artist|genre|year|album|track|tag):|$)",
    re.IGNORECASE,
)

LAST_PL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "last_generated_playlist.json")


def load_user_vectors():
    import numpy as np

    npz_path = os.path.join(STAGE2_DIR, "taste_vectors.npz")
    meta_path = os.path.join(STAGE2_DIR, "track_metadata.json")
    if not os.path.exists(npz_path) or not os.path.exists(meta_path):
        raise FileNotFoundError("Could not find taste_vectors.npz or track_metadata.json. Run Stage 2 first.")

    with np.load(npz_path) as data:
        embeddings = np.array(data["embeddings"])
        track_ids = [str(tid) for tid in data["track_ids"]]
    meta = load_json_safe(meta_path)
    if not isinstance(meta, dict):
        raise ValueError("track_metadata.json is empty or corrupted. Re-run Stage 2.")
    return embeddings, track_ids, meta


def get_master_centroid():
    import numpy as np

    summary_path = os.path.join(STAGE2_DIR, "taste_profile_summary.json")
    summary = load_json_safe(summary_path)
    if not isinstance(summary, dict) or "master_taste_centroid" not in summary:
        raise FileNotFoundError("Missing master_taste_centroid in taste_profile_summary.json. Run Stage 2 first.")
    return np.array(summary["master_taste_centroid"], dtype=float)


def get_vectors_backend():
    """Detect whether Stage 2 vectors use 'clap' (512-D) or 'minilm' (384-D)."""
    import numpy as np

    npz_path = os.path.join(STAGE2_DIR, "taste_vectors.npz")
    if os.path.exists(npz_path):
        try:
            with np.load(npz_path) as data:
                if "embed_backend" in data:
                    val = str(data["embed_backend"])
                    if val in ("clap", "minilm"):
                        return val
                embs = data["embeddings"]
                if len(embs.shape) == 2 and embs.shape[1] == 512:
                    return "clap"
                return "minilm"
        except Exception:
            pass
    return get_effective_preferences().get("embed_backend", "minilm")


def embed_texts(texts, tokenizer, model):
    import numpy as np
    import torch
    from npu_vectorizer import mean_pooling

    is_clap = bool(getattr(model, "is_clap", False))
    if not texts:
        return np.empty((0, 512 if is_clap else 384), dtype=float)

    if is_clap:
        from clap_embedder import embed_texts_clap

        return embed_texts_clap(texts, tokenizer, model, device=getattr(model, "_clap_device", None))

    with torch.no_grad():
        if getattr(model, "is_static_npu", False):
            all_embeddings = []
            for text in texts:
                encoded_input = tokenizer(
                    text,
                    padding="max_length",
                    max_length=128,
                    truncation=True,
                    return_tensors="pt",
                )
                model_output = model(**encoded_input)
                sentence_embeddings = mean_pooling(model_output, encoded_input["attention_mask"])
                norm_emb = torch.nn.functional.normalize(sentence_embeddings, p=2, dim=1).detach().cpu().numpy()
                all_embeddings.append(norm_emb[0])
            return np.array(all_embeddings)

        encoded_input = tokenizer(
            texts,
            padding=True,
            max_length=256,
            truncation=True,
            return_tensors="pt",
        )
        model_output = model(**encoded_input)
        sentence_embeddings = mean_pooling(model_output, encoded_input["attention_mask"])
        return torch.nn.functional.normalize(sentence_embeddings, p=2, dim=1).detach().cpu().numpy()


def compute_mmr(target_vec, candidate_embeddings, top_k, lambda_param=0.65):
    """Maximal Marginal Relevance (MMR) for diverse track selection."""
    import numpy as np
    from sklearn.metrics.pairwise import cosine_similarity

    n_items = len(candidate_embeddings)
    if top_k <= 0 or n_items == 0:
        return []
    if n_items <= top_k:
        return list(range(n_items))

    sims_to_target = cosine_similarity([target_vec], candidate_embeddings)[0]
    selected = []
    unselected = list(range(n_items))

    best_first = int(np.argmax(sims_to_target))
    selected.append(best_first)
    unselected.remove(best_first)

    while len(selected) < top_k and unselected:
        sub_vecs = candidate_embeddings[unselected]
        sel_vecs = candidate_embeddings[selected]
        sims_to_selected = cosine_similarity(sub_vecs, sel_vecs)
        max_sims_to_sel = np.max(sims_to_selected, axis=1)
        target_sims = sims_to_target[unselected]

        mmr_scores = (lambda_param * target_sims) - ((1.0 - lambda_param) * max_sims_to_sel)
        best_pos = int(np.argmax(mmr_scores))
        selected.append(unselected[best_pos])
        unselected.pop(best_pos)

    return selected


def format_spotify_query(q_str):
    """Quote multi-word artist: and genre: operators while preserving subsequent search operators."""
    if not q_str or not isinstance(q_str, str):
        return ""
    cleaned = q_str.strip()
    return _OPERATOR_STOP_RE.sub(lambda m: f'{m.group(1).lower()}:"{m.group(2).strip()}"', cleaned)


def format_ytm_query(q_str):
    """Strip Spotify-specific search operators so YouTube Music gets clean free-text keywords."""
    if not q_str or not isinstance(q_str, str):
        return ""
    cleaned = re.sub(r'\b(artist|genre|year|album|track|tag):(["\']?)', "", q_str, flags=re.IGNORECASE)
    cleaned = cleaned.replace('"', "").replace("'", "")
    return re.sub(r"\s+", " ", cleaned).strip()


def determine_single_artist_mode(prompt, constraints):
    """Only lock onto a single artist when explicitly requested, never from vibe inspiration queries."""
    target_artist = constraints.get("target_artist")
    if isinstance(target_artist, str) and target_artist.strip():
        return True, target_artist.strip()

    prompt_low = str(prompt or "").lower()
    explicit_single_markers = (
        "only",
        "discography",
        "all songs",
        "just ",
        "songs by ",
        "songs from ",
        "tracks by ",
        "tracks from ",
    )
    has_explicit_marker = any(m in prompt_low for m in explicit_single_markers)

    search_qs = [str(q).strip() for q in (constraints.get("search_queries") or []) if str(q).strip()]
    if has_explicit_marker and search_qs and all(q.lower().startswith("artist:") for q in search_qs):
        artists = {
            q.split(":", 1)[1].strip().strip('"').strip("'").lower(): q.split(":", 1)[1].strip().strip('"').strip("'")
            for q in search_qs
        }
        if len(artists) == 1:
            only_art = next(iter(artists.values()))
            if only_art:
                return True, only_art

    return False, None


def check_audio_features(af, constraints, relaxed=False):
    """Filter tracks by verified audio features; gracefully skips tracks without real telemetry."""
    if not is_real_audio_features(af):
        return True

    min_bpm = constraints.get("min_bpm")
    max_bpm = constraints.get("max_bpm")
    target_energy = constraints.get("target_energy")
    target_valence = constraints.get("target_valence")

    bpm = af.get("bpm")
    energy = af.get("energy")
    valence = af.get("valence")

    bpm_low_mult = 0.85 if relaxed else 0.95
    bpm_high_mult = 1.15 if relaxed else 1.05

    if min_bpm and isinstance(bpm, (int, float)) and float(bpm) < (float(min_bpm) * bpm_low_mult):
        return False
    if max_bpm and isinstance(bpm, (int, float)) and float(bpm) > (float(max_bpm) * bpm_high_mult):
        return False

    if target_energy is not None and isinstance(energy, (int, float)):
        e_val = float(energy)
        te = float(target_energy)
        if te >= 0.7 and e_val < (0.40 if relaxed else 0.55):
            return False
        if te <= 0.4 and e_val > (0.65 if relaxed else 0.55):
            return False

    if target_valence is not None and isinstance(valence, (int, float)):
        v_val = float(valence)
        tv = float(target_valence)
        if tv >= 0.7 and v_val < (0.35 if relaxed else 0.45):
            return False
        if tv <= 0.4 and v_val > (0.65 if relaxed else 0.55):
            return False

    return True


def passes_popularity_filter(raw_popularity, deep_cuts=False, relaxed_af=False, raw_track_count=0):
    """Apply popularity thresholds only when Spotify returns popularity (omitted in Feb 2026 Dev Mode)."""
    if raw_popularity is None or isinstance(raw_popularity, bool):
        return True
    try:
        pop = int(raw_popularity)
    except (TypeError, ValueError):
        return True

    if deep_cuts:
        return 15 <= pop <= (70 if relaxed_af else 58)
    if raw_track_count > 10 and pop < 15:
        return False
    return True


def _is_excluded_artist(artists_str, exclude_artists):
    return matches_avoided_artist(artists_str, exclude_artists)


def select_balanced_indices(
    candidates,
    llm_indices,
    target_size,
    target_local_count,
    max_per_artist,
    get_fp_fn=get_track_fingerprint,
):
    """Combine LLM-curated selections with deduplication, artist caps, and Local/New ratio targets."""
    if not candidates or target_size <= 0:
        return []

    target_local = max(0, min(target_size, int(target_local_count)))
    target_new = max(0, target_size - target_local)

    avail_local = sum(1 for c in candidates if c.get("source") == "Local Library")
    avail_new = len(candidates) - avail_local
    local_cap = target_local if avail_new >= target_new else target_size
    new_cap = target_new if avail_local >= target_local else target_size

    chosen = []
    seen_ids = set()
    seen_fps = set()
    artist_counts = {}
    picked_local = 0
    picked_new = 0

    def try_add(tr, enforce_source_cap=True):
        nonlocal picked_local, picked_new
        tid = tr.get("id")
        name = tr.get("name", "")
        artist = tr.get("artist", "")
        fp = get_fp_fn(name, artist)
        pa = artist.split(",")[0].strip().lower() if artist else "unknown"
        is_local = tr.get("source") == "Local Library"

        if tid in seen_ids or fp in seen_fps:
            return False
        if artist_counts.get(pa, 0) >= max_per_artist:
            return False
        if enforce_source_cap:
            if is_local and picked_local >= local_cap:
                return False
            if not is_local and picked_new >= new_cap:
                return False

        seen_ids.add(tid)
        seen_fps.add(fp)
        artist_counts[pa] = artist_counts.get(pa, 0) + 1
        if is_local:
            picked_local += 1
        else:
            picked_new += 1
        chosen.append(tr)
        return True

    for idx in llm_indices or []:
        if len(chosen) >= target_size:
            break
        if isinstance(idx, int) and not isinstance(idx, bool) and 0 <= idx < len(candidates):
            try_add(candidates[idx], enforce_source_cap=True)

    if len(chosen) < target_size:
        for tr in candidates:
            if len(chosen) >= target_size:
                break
            try_add(tr, enforce_source_cap=True)

    if len(chosen) < target_size:
        for tr in candidates:
            if len(chosen) >= target_size:
                break
            try_add(tr, enforce_source_cap=False)

    return chosen


def resolve_tracks_for_spotify(tracks, sp):
    """Resolve tracks (including YTM-exclusives) into Spotify track URIs and report unmatched items."""
    uris = []
    unmatched = []
    for t in tracks or []:
        if not isinstance(t, dict):
            continue
        sp_id = t.get("spotify_id") if is_valid_spotify_id(t.get("spotify_id")) else None
        if not sp_id and is_valid_spotify_id(t.get("id")):
            sp_id = t.get("id")
        if sp_id:
            uris.append(f"spotify:track:{sp_id}")
            continue

        name = str(t.get("name") or "").strip()
        artist = str(t.get("artist") or "").strip()
        primary_artist = artist.split(",")[0].strip() if artist else ""
        clean_title = re.sub(r"\s*\([^)]*\)", "", name).strip() or name
        query = f"{clean_title} {primary_artist}".strip()

        matched_id = None
        if sp is not None and query:
            target_fp = get_track_fingerprint(name, artist)
            target_clean = re.sub(r"[^a-z0-9]+", " ", clean_title.lower()).strip()
            target_art_low = primary_artist.lower()
            try:
                res = sp.search(q=query, type="track", limit=5)
                items = ((res or {}).get("tracks") or {}).get("items") or []
                for cand in items:
                    if not isinstance(cand, dict) or not is_valid_spotify_id(cand.get("id")):
                        continue
                    c_name = str(cand.get("name") or "")
                    c_artists = [a.get("name", "") for a in (cand.get("artists") or []) if isinstance(a, dict)]
                    c_fp = get_track_fingerprint(c_name, c_artists)
                    if c_fp == target_fp:
                        matched_id = cand["id"]
                        break
                    c_clean = re.sub(r"\s*\([^)]*\)", "", c_name).lower()
                    c_clean = re.sub(r"[^a-z0-9]+", " ", c_clean).strip()
                    c_art_join = " ".join(c_artists).lower()
                    if target_clean and c_clean == target_clean and (not target_art_low or target_art_low in c_art_join):
                        matched_id = cand["id"]
                        break
            except Exception:
                matched_id = None

        if matched_id:
            uris.append(f"spotify:track:{matched_id}")
        else:
            unmatched.append(
                {
                    "id": t.get("id"),
                    "name": name or "Unknown",
                    "artist": artist or "Unknown",
                    "reason": "No Spotify match (YTM-exclusive or unavailable)",
                }
            )

    return {
        "uris": uris,
        "matched_count": len(uris),
        "unmatched_count": len(unmatched),
        "unmatched": unmatched,
        "total_tracks": len([t for t in (tracks or []) if isinstance(t, dict)]),
    }


def resolve_tracks_for_ytmusic(tracks, ytm):
    """Resolve tracks (including Spotify-only tracks) into YouTube Music videoIds and report unmatched items."""
    video_ids = []
    unmatched = []
    for t in tracks or []:
        if not isinstance(t, dict):
            continue
        vid = extract_ytm_video_id(t.get("video_id") or t.get("id"))
        if vid:
            video_ids.append(vid)
            continue

        name = str(t.get("name") or "").strip()
        artist = str(t.get("artist") or "").strip()
        matched_vid = None
        if ytm is not None and name:
            try:
                hit = ytmusic_client.search_ytm_track(ytm, name, artist)
                if isinstance(hit, dict):
                    matched_vid = extract_ytm_video_id(hit.get("video_id") or hit.get("id"))
            except Exception:
                matched_vid = None

        if matched_vid:
            video_ids.append(matched_vid)
        else:
            unmatched.append(
                {
                    "id": t.get("id"),
                    "name": name or "Unknown",
                    "artist": artist or "Unknown",
                    "reason": "No confident YouTube Music match found",
                }
            )

    return {
        "video_ids": video_ids,
        "matched_count": len(video_ids),
        "unmatched_count": len(unmatched),
        "unmatched": unmatched,
        "total_tracks": len([t for t in (tracks or []) if isinstance(t, dict)]),
    }


def export_to_ytmusic(playlist_data, ytm=None, privacy="PRIVATE", target_playlist_id=None):
    """Export or update a generated playlist on YouTube Music, returning URL + match report."""
    tracks = playlist_data.get("tracks") or []
    if not tracks:
        raise ValueError("Cannot export an empty playlist (0 tracks).")

    pl_name = playlist_data.get("playlist_name", "AI Playlist")
    pl_desc = playlist_data.get("playlist_description", "Generated by Music Taste Analyzer")
    if ytm is None:
        ytm = ytmusic_client.authenticate_ytmusic()

    resolved = resolve_tracks_for_ytmusic(tracks, ytm)
    if not resolved["video_ids"]:
        raise ValueError("Playlist contains no tracks that could be matched on YouTube Music.")

    res = ytmusic_client.create_or_update_ytm_playlist(
        ytm,
        name=pl_name,
        description=pl_desc,
        video_ids=resolved["video_ids"],
        target_playlist_id=target_playlist_id,
        privacy=privacy,
    )
    return {
        "playlist_id": res["playlist_id"],
        "url": res["url"],
        "matched_count": resolved["matched_count"],
        "unmatched_count": resolved["unmatched_count"],
        "unmatched": resolved["unmatched"],
        "total_tracks": resolved["total_tracks"],
    }


def export_to_spotify(playlist_data, sp=None, public=False, target_playlist_id=None):
    tracks = playlist_data.get("tracks") or []
    if not tracks:
        print("❌ Cannot export an empty playlist (0 tracks).")
        return False

    pl_name = playlist_data.get("playlist_name", "AI Playlist")
    pl_desc = playlist_data.get("playlist_description", "Generated by Music Taste Analyzer")
    track_uris = []
    for t in tracks:
        if not isinstance(t, dict):
            continue
        sp_id = t.get("spotify_id")
        if sp_id and not is_valid_ytm_id(str(sp_id)):
            track_uris.append(f"spotify:track:{sp_id}")
        elif t.get("id") and not is_valid_ytm_id(str(t["id"])):
            track_uris.append(f"spotify:track:{t['id']}")
    if not track_uris:
        print("❌ Cannot export: playlist contains no valid Spotify track IDs (use Cross-Platform Transfer for YTM tracks).")
        return False

    if target_playlist_id:
        print(f"\n[Export] Updating existing Spotify playlist '{target_playlist_id}' with '{pl_name}'...")
    else:
        print(f"\n[Export] Creating {'public' if public else 'private'} playlist '{pl_name}' on Spotify...")

    cache_file = os.path.join(ROOT_DIR, ".cache")
    for attempt in range(2):
        try:
            if sp is None:
                sp, _ = authenticate_spotify()
            if target_playlist_id:
                replace_spotify_playlist_items(
                    sp, target_playlist_id, track_uris, name=pl_name, description=pl_desc
                )
                pl_url = f"https://open.spotify.com/playlist/{target_playlist_id}"
            else:
                new_pl = create_spotify_playlist(sp, name=pl_name, description=pl_desc, public=public)
                add_items_to_spotify_playlist(sp, new_pl["id"], track_uris, batch_size=100)
                pl_url = (new_pl.get("external_urls") or {}).get(
                    "spotify", f"https://open.spotify.com/playlist/{new_pl['id']}"
                )
            print(f"✅ Success! Check your Spotify app. Link: {pl_url}")
            return pl_url
        except Exception as e:
            err_str = str(e)
            if attempt == 0 and ("403" in err_str or "Forbidden" in err_str):
                print("⚠️ Cache token missing playlist write scopes. Clearing cache and re-authenticating...")
                if os.path.exists(cache_file):
                    try:
                        os.remove(cache_file)
                    except OSError:
                        pass
                sp = None
                continue
            print(f"❌ Failed to export: {e}")
            return False
    return False


class WarmDiscoverySession:
    """
    Stateful Stage 3 Discovery engine that keeps embeddings, MiniLM, and optionally
    the configured Local LLM warm across multiple playlist generations, track swaps, and re-sequencing.
    """

    def __init__(self):
        self._lock = threading.RLock()
        self.embed_tokenizer = None
        self.embed_model = None
        self.pipe = None
        self.llm_config_sig = None
        self.llm_info = {}
        self.sp = None
        self.ytm = None
        self._vectors_sig = None

        self.local_embeddings = None
        self.local_ids = None
        self.local_meta = None
        self.local_id_set = set()
        self.local_fps = set()
        self.master_centroid = None
        self.clusters = []

        prefs = get_effective_preferences()
        self.session_avoid_artists = list(prefs["avoid_artists"])
        self.session_avoid_genres = list(prefs["avoid_genres"])

        self.last_candidates = []
        self.last_playlist = load_json_safe(LAST_PL_PATH)

    def _log(self, msg, log_cb=None):
        print(msg)
        if log_cb is not None:
            try:
                log_cb(msg)
            except Exception:
                pass

    def _stage2_sig(self):
        paths = (
            os.path.join(STAGE2_DIR, "taste_vectors.npz"),
            os.path.join(STAGE2_DIR, "track_metadata.json"),
            os.path.join(STAGE2_DIR, "taste_profile_summary.json"),
        )
        sig = []
        for p in paths:
            try:
                st = os.stat(p)
                sig.append((True, st.st_mtime_ns, st.st_size))
            except OSError:
                sig.append((False, 0, 0))
        return tuple(sig)

    def _ensure_vectors_loaded(self, log_cb=None):
        import numpy as np

        sig = self._stage2_sig()
        if not sig[0][0] or not sig[1][0] or not sig[2][0]:
            raise FileNotFoundError("Missing Stage 2 Vector Core files. Run Stage 2 first.")

        if self._vectors_sig == sig and self.local_embeddings is not None:
            return

        self._log("[WarmSession] Loading Stage 2 Vector Matrices...", log_cb)
        self.local_embeddings, self.local_ids, self.local_meta = load_user_vectors()
        summary = load_json_safe(os.path.join(STAGE2_DIR, "taste_profile_summary.json")) or {}
        self.master_centroid = np.array(summary["master_taste_centroid"], dtype=float)
        self.clusters = summary.get("clusters") or []

        self.local_id_set = set(self.local_ids)
        self.local_fps = set()
        for tid, info in self.local_meta.items():
            self.local_id_set.add(str(tid))
            for alt_id in info.get("alt_ids") or []:
                self.local_id_set.add(str(alt_id))
            self.local_fps.add(get_track_fingerprint(info.get("name", ""), info.get("artists", "")))
        self._vectors_sig = sig

    def get_status(self):
        with self._lock:
            sig = self._stage2_sig()
            vectors_ready = bool(sig[0][0] and sig[1][0] and sig[2][0])
            if vectors_ready and self._vectors_sig != sig:
                summary = load_json_safe(os.path.join(STAGE2_DIR, "taste_profile_summary.json")) or {}
                clusters_summary = [
                    {
                        "cluster_id": c.get("cluster_id"),
                        "vibe_label": c.get("vibe_label"),
                        "track_count": c.get("track_count"),
                        "weighted_share_pct": c.get("weighted_share_pct"),
                    }
                    for c in (summary.get("clusters") or [])
                ]
                track_count = sum(c.get("track_count", 0) for c in clusters_summary)
            else:
                clusters_summary = [
                    {
                        "cluster_id": c.get("cluster_id"),
                        "vibe_label": c.get("vibe_label"),
                        "track_count": c.get("track_count"),
                        "weighted_share_pct": c.get("weighted_share_pct"),
                    }
                    for c in (self.clusters or [])
                ]
                track_count = len(self.local_ids) if self.local_ids else 0

            prefs = get_effective_preferences()
            if self.pipe is not None and self.llm_info:
                llm_label = self.llm_info.get("label") or "Local LLM"
                llm_backend = self.llm_info.get("backend") or prefs.get("llm_backend", "openvino")
                llm_model_key = self.llm_info.get("model_key") or prefs.get("llm_model_key", "auto")
                llm_device = self.llm_info.get("device") or prefs.get("llm_device", "auto")
            else:
                llm_backend = prefs.get("llm_backend", "openvino")
                llm_model_key = prefs.get("llm_model_key", "auto")
                llm_device = prefs.get("llm_device", "auto")
                if llm_backend in ("ollama", "openai_local"):
                    prefix = "Ollama" if llm_backend == "ollama" else "Local API"
                    llm_label = f"{prefix}: {prefs.get('ollama_model')}"
                elif llm_model_key == "auto":
                    llm_label = "Auto (Hardware Tier)"
                else:
                    from llm_parser import get_catalog_entry

                    llm_label = get_catalog_entry(llm_model_key).get("short_label", llm_model_key)

            return {
                "embed_warm": self.embed_model is not None,
                "llm_warm": self.pipe is not None,
                "llm_label": llm_label,
                "llm_backend": llm_backend,
                "llm_model_key": llm_model_key,
                "llm_device": llm_device,
                "spotify_warm": self.sp is not None,
                "ytmusic_warm": self.ytm is not None,
                "active_provider": prefs.get("active_provider", "spotify"),
                "vectors_loaded": self.local_embeddings is not None,
                "vectors_ready": vectors_ready,
                "track_count": track_count,
                "clusters": clusters_summary,
                "session_avoid_artists": list(self.session_avoid_artists),
                "session_avoid_genres": list(self.session_avoid_genres),
                "has_last_playlist": isinstance(self.last_playlist, dict) and bool(self.last_playlist.get("tracks")),
            }

    def set_session_exclusions(self, avoid_artists=None, avoid_genres=None):
        with self._lock:
            if avoid_artists is not None:
                self.session_avoid_artists = normalize_avoid_list(avoid_artists)
            if avoid_genres is not None:
                self.session_avoid_genres = normalize_avoid_list(avoid_genres)
            return {
                "avoid_artists": list(self.session_avoid_artists),
                "avoid_genres": list(self.session_avoid_genres),
            }

    def _current_llm_pref_sig(self):
        prefs = get_effective_preferences()
        return (
            prefs.get("llm_backend", "openvino"),
            prefs.get("llm_model_key", "auto"),
            prefs.get("llm_device", "auto"),
            prefs.get("ollama_url", ""),
            prefs.get("ollama_model", ""),
        )

    def _build_taste_context(self, selected_cluster=None, provider="spotify"):
        """Build a concise Taste DNA summary for grounding Stage 3 LLM query planning & curation."""
        lines = [f"Platform: {provider}"]
        target_cid = selected_cluster.get("cluster_id") if isinstance(selected_cluster, dict) else None

        if isinstance(selected_cluster, dict):
            top_g = ", ".join((selected_cluster.get("top_genres") or [])[:4])
            lines.append(
                f"Focused Cluster: C{selected_cluster.get('cluster_id')} {selected_cluster.get('vibe_label', '')}"
                + (f" [{top_g}]" if top_g else "")
            )
        elif self.clusters:
            top_clusters = sorted(
                self.clusters,
                key=lambda c: float(c.get("weighted_share_pct") or 0.0),
                reverse=True,
            )[:5]
            c_strs = []
            for c in top_clusters:
                gs = "/".join((c.get("top_genres") or [])[:2])
                c_strs.append(
                    f"{c.get('vibe_label', 'Cluster')} ({c.get('weighted_share_pct', 0)}%)"
                    + (f"[{gs}]" if gs else "")
                )
            lines.append("Top Taste Clusters: " + "; ".join(c_strs))

        meta = self.local_meta or {}
        if meta:
            art_weights = {}
            track_rows = []
            for info in meta.values():
                if not isinstance(info, dict) or info.get("excluded_from_taste"):
                    continue
                if target_cid is not None and info.get("cluster_id") != target_cid:
                    continue
                rw = float(info.get("recency_weight") or 1.0)
                prim_art = str(info.get("artists") or "").split(",")[0].strip()
                if prim_art and prim_art.lower() != "unknown":
                    art_weights[prim_art] = art_weights.get(prim_art, 0.0) + rw
                track_rows.append((rw, str(info.get("name") or ""), str(info.get("artists") or "")))

            if target_cid is not None and not track_rows:
                for info in meta.values():
                    if not isinstance(info, dict) or info.get("excluded_from_taste"):
                        continue
                    rw = float(info.get("recency_weight") or 1.0)
                    prim_art = str(info.get("artists") or "").split(",")[0].strip()
                    if prim_art and prim_art.lower() != "unknown":
                        art_weights[prim_art] = art_weights.get(prim_art, 0.0) + rw
                    track_rows.append((rw, str(info.get("name") or ""), str(info.get("artists") or "")))

            top_arts = [a for a, _ in sorted(art_weights.items(), key=lambda x: x[1], reverse=True)[:15]]
            if top_arts:
                lines.append("Top Weighted Artists: " + ", ".join(top_arts))
            track_rows.sort(key=lambda x: x[0], reverse=True)
            anchors = [f"{n} - {a.split(',')[0].strip()}" for _, n, a in track_rows[:6] if n]
            if anchors:
                lines.append("Top Anchor Tracks: " + " | ".join(anchors))

        return "\n".join(lines)

    def warm_up(self, load_llm=False, log_cb=None):
        from npu_vectorizer import get_openvino_model

        with self._lock:
            target_backend = get_vectors_backend()
            cur_is_clap = bool(getattr(self.embed_model, "is_clap", False)) if self.embed_model is not None else None
            need_reload_embedder = (
                self.embed_model is None
                or self.embed_tokenizer is None
                or (target_backend == "clap" and not cur_is_clap)
                or (target_backend != "clap" and cur_is_clap)
            )
            if need_reload_embedder:
                if target_backend == "clap":
                    from clap_embedder import get_clap_model

                    self._log("[WarmSession] Loading CLAP (512-D) Audio+Text Embedding Engine...", log_cb)
                    self.embed_tokenizer, self.embed_model, _ = get_clap_model()
                else:
                    self._log("[WarmSession] Loading MiniLM Embedding Engine...", log_cb)
                    self.embed_tokenizer, self.embed_model = get_openvino_model()

            cur_sig = self._current_llm_pref_sig()
            if load_llm and (self.pipe is None or self.llm_config_sig != cur_sig):
                try:
                    self.pipe = None
                    gc.collect()
                    self._log(f"[WarmSession] Loading Stage 3 Local LLM ({cur_sig[0]} / {cur_sig[1]})...", log_cb)
                    self.pipe = get_llm()
                    self.llm_config_sig = cur_sig
                    self.llm_info = {
                        "backend": getattr(self.pipe, "backend", cur_sig[0]),
                        "model_key": getattr(self.pipe, "model_key", cur_sig[1]),
                        "label": getattr(self.pipe, "label", str(cur_sig[1])),
                        "device": getattr(self.pipe, "device", cur_sig[2]),
                    }
                except Exception as e:
                    self._log(f"⚠️ Could not load Local LLM ({e}); continuing in Fast Vector Mode.", log_cb)
                    self.pipe = None
                    self.llm_config_sig = None
                    self.llm_info = {}

            if self._stage2_sig()[0][0]:
                try:
                    self._ensure_vectors_loaded(log_cb=log_cb)
                except Exception as e:
                    self._log(f"⚠️ Could not preload Stage 2 vectors: {e}", log_cb)

            return self.get_status()

    def unload_llm(self):
        with self._lock:
            self.pipe = None
            self.llm_config_sig = None
            self.llm_info = {}
            gc.collect()
            return self.get_status()

    def _ensure_spotify(self, log_cb=None):
        if self.sp is None:
            self._log("[WarmSession] Connecting to Spotify API...", log_cb)
            self.sp, _ = authenticate_spotify()
        return self.sp

    def _ensure_ytmusic(self, log_cb=None):
        if self.ytm is None:
            self._log("[WarmSession] Connecting to YouTube Music...", log_cb)
            self.ytm = ytmusic_client.authenticate_ytmusic()
        return self.ytm

    def _enrich_artist_genres_for_discovery(self, raw_tracks, sp, log_cb=None):
        artist_ids = []
        seen = set()
        for t in raw_tracks:
            if not isinstance(t, dict):
                continue
            for a in t.get("artists") or []:
                if isinstance(a, dict) and a.get("id"):
                    aid = str(a["id"])
                    if aid not in seen:
                        seen.add(aid)
                        artist_ids.append(aid)

        return enrich_artist_genres_2026(
            sp,
            artist_ids,
            max_lookups=25,
            max_empty_streak=3,
        )

    def _save_payload(self, payload, update_archive=True):
        atomic_write_json(LAST_PL_PATH, payload, indent=2)
        if update_archive:
            arch_dir = getattr(config, "PLAYLIST_HISTORY_DIR", os.path.join(ROOT_DIR, "03_Agentic_Engine", "playlist_history"))
            pid = payload.get("id")
            if pid and os.path.exists(os.path.join(arch_dir, f"{pid}.json")):
                atomic_write_json(os.path.join(arch_dir, f"{pid}.json"), payload, indent=2)
            else:
                payload["id"] = save_playlist_to_archive(payload, archive_dir=arch_dir)
                atomic_write_json(LAST_PL_PATH, payload, indent=2)

    def _format_local_track(self, tid):
        info = self.local_meta[tid]
        af = info.get("audio_features") if is_real_audio_features(info.get("audio_features")) else {}
        sp_id = info.get("spotify_id") if is_valid_spotify_id(info.get("spotify_id")) else (tid if is_valid_spotify_id(tid) else None)
        vid = extract_ytm_video_id(info.get("video_id") or tid)
        prov = get_track_provider({"id": tid, "provider": info.get("provider"), "spotify_id": sp_id, "video_id": vid})
        return {
            "id": tid,
            "name": info["name"],
            "artist": info["artists"],
            "genres": info.get("genres", ""),
            "explicit": info.get("explicit") == "Explicit",
            "provider": prov,
            "spotify_id": sp_id,
            "video_id": vid,
            "source": "Local Library",
            "bpm": af.get("bpm"),
            "energy": af.get("energy"),
            "valence": af.get("valence"),
            "key": af.get("key"),
            "mode": af.get("mode"),
            "camelot": format_camelot_label(af.get("key"), af.get("mode")),
            "spotify_url": f"https://open.spotify.com/track/{sp_id}" if sp_id else None,
            "youtube_url": f"https://music.youtube.com/watch?v={vid}" if vid else None,
        }

    def generate(
        self,
        prompt="",
        fast=False,
        size=None,
        local_only=False,
        cluster_id=None,
        flow_mode="mmr",
        clean_only=False,
        deep_cuts=False,
        avoid_artists=None,
        avoid_genres=None,
        save_to_archive=True,
        custom_centroid=None,
        seed_text=None,
        provider=None,
        log_cb=None,
    ):
        import numpy as np
        from reccobeats_client import (
            fetch_audio_features_batch,
            fetch_audio_features_by_search,
            fetch_reccobeats_recommendations,
            fetch_track_metadata_batch,
        )

        with self._lock:
            self._ensure_vectors_loaded(log_cb=log_cb)

            prompt_clean = str(prompt or "").strip()
            selected_cluster = None
            if cluster_id is not None:
                for c in self.clusters:
                    if c.get("cluster_id") == int(cluster_id):
                        selected_cluster = c
                        break
                if selected_cluster is None:
                    raise ValueError(f"Cluster C{cluster_id} was not found in Stage 2 summary.")
                if not prompt_clean:
                    prompt_clean = f"Songs matching {selected_cluster.get('vibe_label', f'Cluster {cluster_id}')}"

            if not prompt_clean:
                raise ValueError("Prompt cannot be empty.")

            self.warm_up(load_llm=not fast, log_cb=log_cb)
            active_pipe = None if fast else self.pipe

            req_avoid_artists = normalize_avoid_list(
                avoid_artists if avoid_artists is not None else self.session_avoid_artists
            )
            req_avoid_genres = normalize_avoid_list(
                avoid_genres if avoid_genres is not None else self.session_avoid_genres
            )

            eff_provider = str(provider or get_effective_preferences().get("active_provider", "spotify")).strip().lower()
            if eff_provider not in ("spotify", "ytmusic", "unified"):
                eff_provider = "spotify"

            taste_ctx = self._build_taste_context(selected_cluster=selected_cluster, provider=eff_provider)

            self._log(f"\n[3/5] Parsing intent for: '{prompt_clean}'", log_cb)
            constraints = parse_prompt(
                active_pipe,
                prompt_clean,
                avoid_artists=req_avoid_artists,
                avoid_genres=req_avoid_genres,
                taste_context=taste_ctx,
            )

            if size is not None:
                constraints["playlist_size"] = max(5, min(100, int(size)))
            if local_only:
                constraints["local_only"] = True
                constraints["known_tracks_ratio"] = 1.0
            if clean_only:
                constraints["clean_only"] = True
            if deep_cuts:
                constraints["deep_cuts"] = True

            eff_avoid_artists = normalize_avoid_list(req_avoid_artists + (constraints.get("exclude_artists") or []))
            eff_avoid_genres = normalize_avoid_list(req_avoid_genres + (constraints.get("exclude_genres") or []))
            constraints["exclude_artists"] = eff_avoid_artists
            constraints["exclude_genres"] = eff_avoid_genres

            self._log(f"      Parsed constraints: {json.dumps(constraints, indent=2)}", log_cb)

            disliked_ids, disliked_fps = get_disliked_sets()
            liked_ids = get_liked_ids()

            prompt_vec = embed_texts([seed_text or prompt_clean], self.embed_tokenizer, self.embed_model)[0]
            if custom_centroid is not None:
                c_vec = np.array(custom_centroid, dtype=float)
                blended = (0.35 * prompt_vec) + (0.65 * c_vec) if prompt_clean else c_vec
            elif selected_cluster is not None and selected_cluster.get("centroid_vector"):
                cluster_vec = np.array(selected_cluster["centroid_vector"], dtype=float)
                blended = (0.45 * prompt_vec) + (0.55 * cluster_vec)
            else:
                blended = (0.70 * prompt_vec) + (0.30 * self.master_centroid)

            # Gently bias toward user's liked tracks when not targeting a fixed custom/cluster vector
            if liked_ids and custom_centroid is None and selected_cluster is None:
                liked_indices = [i for i, tid in enumerate(self.local_ids) if tid in liked_ids]
                if liked_indices:
                    liked_mean = np.mean(self.local_embeddings[liked_indices], axis=0)
                    blended = blended + (0.08 * liked_mean)

            norm = float(np.linalg.norm(blended))
            target_vec = (blended / norm) if norm > 0 else prompt_vec

            target_size = int(constraints.get("playlist_size", 30))
            known_ratio = float(constraints.get("known_tracks_ratio", 0.3))
            eff_local_only = bool(constraints.get("local_only", False))
            eff_clean_only = bool(constraints.get("clean_only", False))
            eff_deep_cuts = bool(constraints.get("deep_cuts", False))

            sp = None
            ytm = None
            if not eff_local_only:
                if eff_provider in ("spotify", "unified"):
                    try:
                        sp = self._ensure_spotify(log_cb=log_cb)
                    except Exception as e:
                        self._log(f"      ⚠️ Spotify authentication unavailable ({e}).", log_cb)
                if eff_provider in ("ytmusic", "unified"):
                    try:
                        ytm = self._ensure_ytmusic(log_cb=log_cb)
                    except Exception as e:
                        self._log(f"      ⚠️ YouTube Music authentication unavailable ({e}).", log_cb)
                if sp is None and ytm is None:
                    self._log("      ⚠️ No remote provider available; switching to local-only mode.", log_cb)
                    eff_local_only = True
                    known_ratio = 1.0

            fetch_size = int(target_size * 1.5) + 15
            target_local_count = target_size if eff_local_only else round(target_size * known_ratio)
            n_known_fetch = fetch_size if eff_local_only else max(target_local_count + 5, int(fetch_size * known_ratio))
            n_new_fetch = 0 if eff_local_only else max(0, fetch_size - n_known_fetch)

            is_single_artist, target_artist = determine_single_artist_mode(prompt_clean, constraints)
            max_per_artist = target_size if is_single_artist else max(3, target_size // 10)
            prefs = get_effective_preferences()

            self._log(f"\n[4/5] Reranking Local Database via MMR Diversity (target {n_known_fetch} candidates)...", log_cb)

            def filter_local_candidates(relaxed_af):
                matched = []
                for i, tid in enumerate(self.local_ids):
                    info = self.local_meta.get(tid, {})
                    art_str = info.get("artists", "")
                    gen_str = info.get("genres", "")
                    fp = get_track_fingerprint(info.get("name", ""), art_str)
                    if tid in disliked_ids or fp in disliked_fps:
                        continue
                    if matches_avoided_artist(art_str, eff_avoid_artists):
                        continue
                    if matches_avoided_genre(gen_str, eff_avoid_genres):
                        continue
                    if eff_clean_only and str(info.get("explicit", "")).lower() == "explicit":
                        continue
                    if prefs.get("skip_penalty_enabled") and not is_single_artist:
                        if float(info.get("skip_ratio") or 0.0) >= 0.75:
                            continue
                    if is_single_artist and target_artist and target_artist.lower() not in art_str.lower():
                        continue
                    if not check_audio_features(info.get("audio_features", {}), constraints, relaxed=relaxed_af):
                        continue
                    matched.append(i)
                return matched

            valid_local_indices = filter_local_candidates(relaxed_af=False)
            if len(valid_local_indices) < max(3, target_local_count // 2):
                relaxed_local = filter_local_candidates(relaxed_af=True)
                if len(relaxed_local) > len(valid_local_indices):
                    self._log("      ℹ️ Broadening local acoustic tolerance to recover enough matching tracks...", log_cb)
                    valid_local_indices = relaxed_local

            final_playlist = []
            if valid_local_indices and n_known_fetch > 0:
                sub_vecs = self.local_embeddings[valid_local_indices]
                top_sub_idx = compute_mmr(
                    target_vec,
                    sub_vecs,
                    top_k=min(n_known_fetch, len(valid_local_indices)),
                    lambda_param=0.65,
                )
                for sub_i in top_sub_idx:
                    tid = self.local_ids[valid_local_indices[sub_i]]
                    final_playlist.append(self._format_local_track(tid))

            if not eff_local_only and n_new_fetch > 0 and (sp is not None or ytm is not None):
                discovered_tracks = []
                discovered_ids = set()
                discovered_fps = set()
                past_queries = []
                agent_loops = 0
                max_agent_loops = 10
                sp_lock = threading.Lock()
                spotify_rate_limited = threading.Event()

                artist_freq = {}
                for ft in final_playlist:
                    pa = ft["artist"].split(",")[0].strip().lower()
                    artist_freq[pa] = artist_freq.get(pa, 0) + 1

                def fetch_spotify_tracks(q_str):
                    items = []
                    formatted_q = format_spotify_query(q_str)
                    if not formatted_q or sp is None or spotify_rate_limited.is_set():
                        return items
                    for offset in (0, 10, 20, 30, 40):
                        if spotify_rate_limited.is_set():
                            return items
                        for attempt in range(3):
                            if spotify_rate_limited.is_set():
                                return items
                            try:
                                with sp_lock:
                                    res = sp.search(q=formatted_q, type="track", limit=10, offset=offset)
                                batch = ((res or {}).get("tracks") or {}).get("items") or []
                                if not batch:
                                    return items
                                for bt in batch:
                                    if isinstance(bt, dict):
                                        bt["_provider"] = "spotify"
                                items.extend(batch)
                                break
                            except Exception as exc:
                                msg = str(exc)
                                status = getattr(exc, "http_status", None)
                                headers = getattr(exc, "headers", None) or {}
                                is_429 = (status == 429) or ("429" in msg)
                                if is_429:
                                    raw_ra = headers.get("Retry-After") if hasattr(headers, "get") else None
                                    try:
                                        retry_after = float(raw_ra) if raw_ra is not None else 2.0
                                    except (TypeError, ValueError):
                                        retry_after = 2.0
                                    if "QUOTA_EXCEEDED" in msg.upper() or retry_after > 20.0:
                                        spotify_rate_limited.set()
                                        self._log(
                                            f"      ⚠️ Spotify rate/quota limit reached (Retry-After={retry_after:.0f}s); stopping Spotify search and using collected tracks.",
                                            log_cb,
                                        )
                                        return items
                                    if attempt < 2:
                                        time.sleep(min(10.0, max(1.0, retry_after)))
                                        continue
                                self._log(f"      ⚠️ Search query '{formatted_q}' (offset={offset}) warning: {exc}", log_cb)
                                return items
                    return items

                def fetch_ytm_tracks(q_str):
                    items = []
                    cleaned_q = format_ytm_query(q_str)
                    if not cleaned_q or ytm is None:
                        return items
                    try:
                        res = ytm.search(cleaned_q, filter="songs", limit=15)
                    except Exception as exc:
                        self._log(f"      ⚠️ YT Music search '{cleaned_q}' warning: {exc}", log_cb)
                        return items
                    for raw_yt in res or []:
                        norm = ytmusic_client.normalize_ytm_track(raw_yt)
                        if norm:
                            norm["_provider"] = "ytmusic"
                            items.append(norm)
                    return items

                while len(discovered_tracks) < n_new_fetch and agent_loops < max_agent_loops:
                    if agent_loops == 0:
                        queries = constraints.get("search_queries", [])
                    else:
                        tracks_needed = n_new_fetch - len(discovered_tracks)
                        self._log(f"      -> Asking parser for deeper queries (Need {tracks_needed} more tracks)...", log_cb)
                        new_constraints = parse_prompt(
                            active_pipe,
                            prompt_clean,
                            past_queries=past_queries,
                            avoid_artists=eff_avoid_artists,
                            avoid_genres=eff_avoid_genres,
                            taste_context=taste_ctx,
                        )
                        queries = new_constraints.get("search_queries", [])
                        if not queries:
                            break

                    agent_loops += 1
                    past_low = {pq.lower() for pq in past_queries}
                    active_queries = [q for q in queries if q and q.lower() not in past_low]
                    past_queries.extend(active_queries)
                    if not active_queries:
                        continue

                    raw_tracks = []
                    if sp is not None and not spotify_rate_limited.is_set():
                        self._log(f"      ⚡ Querying {len(active_queries)} Spotify search targets...", log_cb)
                        with ThreadPoolExecutor(max_workers=min(3, len(active_queries))) as executor:
                            futures = {executor.submit(fetch_spotify_tracks, q): q for q in active_queries}
                            for future in as_completed(futures):
                                raw_tracks.extend(future.result())

                    if ytm is not None:
                        ytm_qs = active_queries[:3]
                        self._log(f"      ⚡ Querying {len(ytm_qs)} YouTube Music search targets...", log_cb)
                        for q in ytm_qs:
                            raw_tracks.extend(fetch_ytm_tracks(q))

                        if agent_loops == 1 and not is_single_artist:
                            ytm_seeds = []
                            for tr in final_playlist:
                                svid = extract_ytm_video_id(tr.get("video_id") or tr.get("id"))
                                if svid and svid not in ytm_seeds:
                                    ytm_seeds.append(svid)
                                if len(ytm_seeds) >= 3:
                                    break
                            added_watch = 0
                            seen_raw_ids = {t.get("id") for t in raw_tracks if isinstance(t, dict) and t.get("id")}
                            for seed_vid in ytm_seeds:
                                try:
                                    watch = ytm.get_watch_playlist(videoId=seed_vid, limit=15)
                                    for wtr in (watch or {}).get("tracks") or []:
                                        norm = ytmusic_client.normalize_ytm_track(wtr)
                                        if norm and norm.get("id") not in seen_raw_ids:
                                            norm["_provider"] = "ytmusic"
                                            norm["_from_ytm_radio"] = True
                                            seen_raw_ids.add(norm["id"])
                                            raw_tracks.append(norm)
                                            added_watch += 1
                                except Exception as exc:
                                    self._log(f"      ⚠️ YT Music watch radio warning: {exc}", log_cb)
                            if added_watch:
                                self._log(f"      ✨ Added {added_watch} YouTube Music Watch Radio recommendations.", log_cb)

                    # Supplement first discovery wave with ReccoBeats /v1/track/recommendation (with negativeSeeds)
                    if sp is not None and agent_loops == 1 and not is_single_artist:
                        seed_pool = [
                            tr.get("id")
                            for tr in final_playlist
                            if is_valid_spotify_id(tr.get("id"))
                        ]
                        for lid in liked_ids:
                            if is_valid_spotify_id(lid) and lid not in seed_pool:
                                seed_pool.append(lid)
                        neg_pool = [did for did in disliked_ids if is_valid_spotify_id(did)][:5]
                        if seed_pool:
                            min_b = constraints.get("min_bpm")
                            max_b = constraints.get("max_bpm")
                            t_bpm = None
                            if isinstance(min_b, (int, float)) and isinstance(max_b, (int, float)):
                                t_bpm = (float(min_b) + float(max_b)) / 2.0
                            elif isinstance(min_b, (int, float)):
                                t_bpm = float(min_b) + 10.0
                            elif isinstance(max_b, (int, float)):
                                t_bpm = max(60.0, float(max_b) - 10.0)
                            try:
                                rb_recs = fetch_reccobeats_recommendations(
                                    seed_ids=seed_pool[:5],
                                    negative_seed_ids=neg_pool,
                                    size=min(30, max(10, n_new_fetch)),
                                    popularity=38 if eff_deep_cuts else None,
                                    target_bpm=t_bpm,
                                    target_energy=constraints.get("target_energy"),
                                    target_valence=constraints.get("target_valence"),
                                )
                                if rb_recs:
                                    seen_rb = {t.get("id") for t in raw_tracks if isinstance(t, dict) and t.get("id")}
                                    added_rb = 0
                                    for rtrack in rb_recs:
                                        rtid = rtrack.get("id")
                                        if rtid and rtid not in seen_rb:
                                            seen_rb.add(rtid)
                                            rtrack["_provider"] = "spotify"
                                            raw_tracks.append(rtrack)
                                            added_rb += 1
                                    if added_rb:
                                        self._log(
                                            f"      ✨ Added {added_rb} ReccoBeats recommendations (negativeSeeds={len(neg_pool)}).",
                                            log_cb,
                                        )
                            except Exception as e:
                                self._log(f"      ⚠️ ReccoBeats recommendation warning: {e}", log_cb)

                    # Resolve direct AI Curator track suggestions (recommended_tracks) across active providers
                    if agent_loops == 1 and not is_single_artist and (constraints.get("recommended_tracks") or []):
                        rec_picks = [
                            rt
                            for rt in (constraints.get("recommended_tracks") or [])
                            if isinstance(rt, dict) and rt.get("name") and rt.get("artist")
                        ][:10]
                        added_picks = 0
                        seen_pick_ids = {t.get("id") for t in raw_tracks if isinstance(t, dict) and t.get("id")}
                        for pick in rec_picks:
                            p_name = str(pick["name"]).strip()
                            p_art = str(pick["artist"]).strip()
                            p_fp = get_track_fingerprint(p_name, p_art)
                            resolved_sp = False
                            if sp is not None and not spotify_rate_limited.is_set():
                                try:
                                    with sp_lock:
                                        sres = sp.search(q=f"{p_name} {p_art}".strip(), type="track", limit=3)
                                    items = ((sres or {}).get("tracks") or {}).get("items") or []
                                    chosen_sp = None
                                    for cand in items:
                                        if not isinstance(cand, dict) or not cand.get("id"):
                                            continue
                                        c_arts = [a.get("name", "") for a in (cand.get("artists") or []) if isinstance(a, dict)]
                                        if get_track_fingerprint(cand.get("name", ""), c_arts) == p_fp:
                                            chosen_sp = cand
                                            break
                                        if chosen_sp is None:
                                            chosen_sp = cand
                                    if chosen_sp:
                                        if chosen_sp["id"] not in seen_pick_ids:
                                            chosen_sp["_provider"] = "spotify"
                                            chosen_sp["_from_llm_pick"] = True
                                            seen_pick_ids.add(chosen_sp["id"])
                                            raw_tracks.append(chosen_sp)
                                            added_picks += 1
                                        else:
                                            for existing in raw_tracks:
                                                if isinstance(existing, dict) and existing.get("id") == chosen_sp["id"]:
                                                    existing["_from_llm_pick"] = True
                                        resolved_sp = True
                                except Exception:
                                    pass
                            if ytm is not None and (not resolved_sp or eff_provider in ("ytmusic", "unified")):
                                try:
                                    hit = ytmusic_client.search_ytm_track(ytm, p_name, p_art)
                                    if hit:
                                        norm_hit = ytmusic_client.normalize_ytm_track(
                                            {"videoId": hit["video_id"], "title": hit["name"], "artists": [{"name": hit["artist"]}]}
                                        )
                                        if norm_hit and norm_hit["id"] not in seen_pick_ids:
                                            norm_hit["_provider"] = "ytmusic"
                                            norm_hit["_from_llm_pick"] = True
                                            seen_pick_ids.add(norm_hit["id"])
                                            raw_tracks.append(norm_hit)
                                            if not resolved_sp:
                                                added_picks += 1
                                except Exception:
                                    pass
                        if added_picks:
                            self._log(f"      🧠 Resolved {added_picks} AI Curator Pick track(s).", log_cb)

                    if not raw_tracks:
                        continue

                    # Deduplicate cross-provider hits in unified mode by (title, primary_artist)
                    deduped_raw = []
                    fp_to_idx = {}
                    for t in raw_tracks:
                        if not isinstance(t, dict) or not t.get("id") or not t.get("name"):
                            continue
                        arts = [a.get("name", "") for a in (t.get("artists") or []) if isinstance(a, dict)]
                        fp = get_track_fingerprint(t["name"], arts)
                        if fp in fp_to_idx:
                            prev = deduped_raw[fp_to_idx[fp]]
                            if is_valid_spotify_id(t.get("id")) and is_valid_ytm_id(prev.get("id")):
                                t["video_id"] = prev.get("video_id") or extract_ytm_video_id(prev.get("id"))
                                t["_provider"] = "both"
                                deduped_raw[fp_to_idx[fp]] = t
                            elif is_valid_ytm_id(t.get("id")) and is_valid_spotify_id(prev.get("id")):
                                prev["video_id"] = t.get("video_id") or extract_ytm_video_id(t.get("id"))
                                prev["_provider"] = "both"
                        else:
                            fp_to_idx[fp] = len(deduped_raw)
                            deduped_raw.append(t)
                    raw_tracks = deduped_raw

                    sp_raw_tracks = [t for t in raw_tracks if is_valid_spotify_id(t.get("id"))]
                    ytm_raw_tracks = [t for t in raw_tracks if is_valid_ytm_id(t.get("id"))]
                    sp_raw_ids = [t["id"] for t in sp_raw_tracks]

                    # Backfill stripped Feb 2026 Dev Mode track popularity via ReccoBeats /v1/track when filtering deep cuts
                    missing_pop_ids = [
                        t["id"]
                        for t in sp_raw_tracks
                        if t.get("popularity") is None
                    ]
                    if missing_pop_ids and (eff_deep_cuts or len(sp_raw_tracks) > 10):
                        try:
                            meta_pop_map = fetch_track_metadata_batch(missing_pop_ids)
                            for t in sp_raw_tracks:
                                if t.get("id") in meta_pop_map and t.get("popularity") is None:
                                    mpop = meta_pop_map[t["id"]].get("popularity")
                                    if mpop is not None:
                                        t["popularity"] = mpop
                        except Exception as e:
                            self._log(f"      ⚠️ ReccoBeats metadata backfill warning: {e}", log_cb)

                    af_map = {}
                    if sp_raw_ids:
                        try:
                            af_map.update(fetch_audio_features_batch(sp_raw_ids))
                        except Exception as e:
                            self._log(f"      ⚠️ ReccoBeats warning during discovery: {e}", log_cb)
                    if ytm_raw_tracks:
                        try:
                            ytm_af_map = fetch_audio_features_by_search(
                                ytm_raw_tracks,
                                cache_path=os.path.join(STAGE1_DIR, ".reccobeats_cache.json"),
                                max_lookups=40,
                            )
                            af_map.update(ytm_af_map)
                        except Exception as e:
                            self._log(f"      ⚠️ ReccoBeats YTM lookup warning: {e}", log_cb)

                    artist_genre_map = (
                        self._enrich_artist_genres_for_discovery(sp_raw_tracks, sp, log_cb=log_cb)
                        if (eff_avoid_genres and sp is not None and sp_raw_tracks)
                        else {}
                    )

                    def collect_discovery_candidates(relaxed_af):
                        docs = []
                        meta_list = []
                        batch_seen_ids = set()
                        batch_seen_fps = set()

                        for t in raw_tracks:
                            if not isinstance(t, dict) or not t.get("id") or not t.get("name"):
                                continue
                            if t.get("is_local") or t.get("type") == "episode":
                                continue

                            tid = str(t["id"])
                            raw_artists = [a for a in (t.get("artists") or []) if isinstance(a, dict) and a.get("name")]
                            artists = [a["name"] for a in raw_artists]
                            artists_str = ", ".join(artists) if artists else "Unknown"
                            primary_art = artists[0].strip().lower() if artists else "unknown"
                            t_fp = get_track_fingerprint(t["name"], artists_str)

                            track_genres = []
                            for a in raw_artists:
                                aid = str(a.get("id") or "")
                                if aid and aid in artist_genre_map:
                                    track_genres.extend(artist_genre_map[aid])
                            genres_str = ", ".join(dict.fromkeys(track_genres))

                            if tid in disliked_ids or t_fp in disliked_fps:
                                continue
                            if matches_avoided_artist(artists_str, eff_avoid_artists):
                                continue
                            if eff_avoid_genres and genres_str and matches_avoided_genre(genres_str, eff_avoid_genres):
                                continue
                            if eff_clean_only and t.get("explicit"):
                                continue
                            if is_single_artist and target_artist and target_artist.lower() not in artists_str.lower():
                                continue
                            if tid in self.local_id_set or t_fp in self.local_fps:
                                continue
                            if tid in discovered_ids or t_fp in discovered_fps:
                                continue
                            if tid in batch_seen_ids or t_fp in batch_seen_fps:
                                continue

                            raw_pop = t.get("popularity")
                            if not passes_popularity_filter(
                                raw_pop,
                                deep_cuts=eff_deep_cuts,
                                relaxed_af=relaxed_af,
                                raw_track_count=len(raw_tracks),
                            ):
                                continue
                            pop = (
                                int(raw_pop)
                                if isinstance(raw_pop, (int, float)) and not isinstance(raw_pop, bool)
                                else None
                            )

                            if artist_freq.get(primary_art, 0) >= max_per_artist:
                                continue

                            af = af_map.get(tid, {})
                            if not check_audio_features(af, constraints, relaxed=relaxed_af):
                                continue

                            batch_seen_ids.add(tid)
                            batch_seen_fps.add(t_fp)

                            release_year = ""
                            album = t.get("album") or {}
                            if isinstance(album, dict) and album.get("release_date"):
                                release_year = str(album["release_date"])[:4]
                            explicit_tag = "Explicit" if t.get("explicit") else "Clean"

                            doc = f"Track: {t['name']} | Artist: {artists_str}"
                            if release_year:
                                doc += f" | Year: {release_year}"
                            doc += f" | Content: {explicit_tag}"
                            if genres_str:
                                doc += f" | Genres: {genres_str}"
                            af_text = format_audio_feature_text(af)
                            if af_text:
                                doc += f" | {af_text}"

                            real_af = af if is_real_audio_features(af) else {}
                            sp_id = t.get("spotify_id") if is_valid_spotify_id(t.get("spotify_id")) else (tid if is_valid_spotify_id(tid) else None)
                            vid = extract_ytm_video_id(t.get("video_id") or tid)
                            prov = get_track_provider({"id": tid, "provider": t.get("_provider") or t.get("provider"), "spotify_id": sp_id, "video_id": vid})
                            if t.get("_from_llm_pick"):
                                src_label = "AI Curator Pick"
                            elif t.get("_from_reccobeats_rec"):
                                src_label = "ReccoBeats Discovery"
                            elif t.get("_from_ytm_radio"):
                                src_label = "YT Music Radio"
                            elif prov == "ytmusic":
                                src_label = "YT Music Discovery"
                            else:
                                src_label = "New Discovery"
                            docs.append(doc)
                            meta_list.append(
                                {
                                    "id": tid,
                                    "name": t["name"],
                                    "artist": artists_str,
                                    "genres": genres_str,
                                    "primary_artist": primary_art,
                                    "fingerprint": t_fp,
                                    "explicit": bool(t.get("explicit")),
                                    "popularity": pop,
                                    "provider": prov,
                                    "spotify_id": sp_id,
                                    "video_id": vid,
                                    "bpm": real_af.get("bpm"),
                                    "energy": real_af.get("energy"),
                                    "valence": real_af.get("valence"),
                                    "key": real_af.get("key"),
                                    "mode": real_af.get("mode"),
                                    "camelot": format_camelot_label(real_af.get("key"), real_af.get("mode")),
                                    "spotify_url": f"https://open.spotify.com/track/{sp_id}" if sp_id else None,
                                    "youtube_url": f"https://music.youtube.com/watch?v={vid}" if vid else None,
                                    "source_label": src_label,
                                }
                            )
                        return docs, meta_list

                    scraped_docs, scraped_meta = collect_discovery_candidates(relaxed_af=False)
                    if not scraped_docs:
                        scraped_docs, scraped_meta = collect_discovery_candidates(relaxed_af=True)

                    if scraped_docs:
                        scraped_vecs = embed_texts(scraped_docs, self.embed_tokenizer, self.embed_model)
                        needed = n_new_fetch - len(discovered_tracks)
                        mmr_ranks = compute_mmr(target_vec, scraped_vecs, top_k=needed, lambda_param=0.65)
                        for idx in mmr_ranks:
                            cand = scraped_meta[idx]
                            pa = cand["primary_artist"]
                            if artist_freq.get(pa, 0) < max_per_artist:
                                discovered_ids.add(cand["id"])
                                discovered_fps.add(cand["fingerprint"])
                                discovered_tracks.append(
                                    {
                                        "id": cand["id"],
                                        "name": cand["name"],
                                        "artist": cand["artist"],
                                        "genres": cand["genres"],
                                        "explicit": cand["explicit"],
                                        "popularity": cand["popularity"],
                                        "provider": cand["provider"],
                                        "spotify_id": cand.get("spotify_id"),
                                        "video_id": cand.get("video_id"),
                                        "source": cand.get("source_label") or "New Discovery",
                                        "bpm": cand["bpm"],
                                        "energy": cand["energy"],
                                        "valence": cand["valence"],
                                        "key": cand["key"],
                                        "mode": cand["mode"],
                                        "camelot": cand["camelot"],
                                        "spotify_url": cand["spotify_url"],
                                        "youtube_url": cand["youtube_url"],
                                    }
                                )
                                artist_freq[pa] = artist_freq.get(pa, 0) + 1

                    if spotify_rate_limited.is_set() and ytm is None:
                        break

                final_playlist.extend(discovered_tracks)

            if not final_playlist:
                raise ValueError("No matching tracks could be found for this prompt and filters.")

            self._log(f"\n[5/5] 🎧 Curating final {target_size} tracks from {len(final_playlist)} candidates...", log_cb)
            vibe_res = vibe_check_playlist(
                active_pipe,
                prompt_clean,
                final_playlist,
                target_size,
                avoid_artists=eff_avoid_artists,
                avoid_genres=eff_avoid_genres,
                taste_context=taste_ctx,
            )
            if not isinstance(vibe_res, dict):
                vibe_res = {}

            curated_playlist = select_balanced_indices(
                final_playlist,
                vibe_res.get("selected_indices"),
                target_size=target_size,
                target_local_count=target_local_count,
                max_per_artist=max_per_artist,
                get_fp_fn=get_track_fingerprint,
            )

            eff_flow = flow_mode or "mmr"
            curated_playlist = sequence_playlist_tracks(curated_playlist, flow_mode=eff_flow)

            self.last_candidates = list(final_playlist)
            pl_name = vibe_res.get("playlist_name") or constraints.get("playlist_name") or f"AI: {prompt_clean[:30]}"
            pl_desc = (
                vibe_res.get("playlist_description")
                or constraints.get("playlist_description")
                or "Generated by Music Taste Analyzer"
            )
            playlist_payload = {
                "prompt": prompt_clean,
                "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat().replace("+00:00", "Z"),
                "playlist_name": pl_name,
                "playlist_description": pl_desc,
                "flow_mode": eff_flow,
                "provider": eff_provider,
                "cluster_id": cluster_id,
                "avoid_artists": eff_avoid_artists,
                "avoid_genres": eff_avoid_genres,
                "tracks": curated_playlist,
            }
            self._save_payload(playlist_payload, update_archive=save_to_archive)
            self.last_playlist = playlist_payload

            return {
                "playlist": playlist_payload,
                "constraints": constraints,
                "candidate_pool_size": len(final_playlist),
            }

    def swap_track(self, drop_index, flow_mode=None):
        """Replace track at drop_index with the next unused candidate respecting artist/avoid rules."""
        with self._lock:
            payload = self.last_playlist or load_json_safe(LAST_PL_PATH)
            if not isinstance(payload, dict) or not payload.get("tracks"):
                raise ValueError("No active playlist loaded to edit.")

            tracks = list(payload["tracks"])
            idx = int(drop_index)
            if idx < 0 or idx >= len(tracks):
                raise IndexError("Track index out of range.")

            dropped = tracks[idx]
            remaining_ids = {t.get("id") for i, t in enumerate(tracks) if i != idx}
            remaining_fps = {get_track_fingerprint(t.get("name", ""), t.get("artist", "")) for i, t in enumerate(tracks) if i != idx}
            remaining_ids.add(dropped.get("id"))
            remaining_fps.add(get_track_fingerprint(dropped.get("name", ""), dropped.get("artist", "")))

            artist_counts = {}
            for i, t in enumerate(tracks):
                if i == idx:
                    continue
                pa = str(t.get("artist") or "").split(",")[0].strip().lower()
                artist_counts[pa] = artist_counts.get(pa, 0) + 1

            eff_avoid_art = normalize_avoid_list(payload.get("avoid_artists") or self.session_avoid_artists)
            eff_avoid_gen = normalize_avoid_list(payload.get("avoid_genres") or self.session_avoid_genres)
            disliked_ids, disliked_fps = get_disliked_sets()
            max_per_artist = max(3, len(tracks) // 10)

            pool = self.last_candidates or []
            replacement = None
            for enforce_cap in (True, False):
                for cand in pool:
                    cid = cand.get("id")
                    cfp = get_track_fingerprint(cand.get("name", ""), cand.get("artist", ""))
                    pa = str(cand.get("artist") or "").split(",")[0].strip().lower()
                    if cid in remaining_ids or cfp in remaining_fps:
                        continue
                    if cid in disliked_ids or cfp in disliked_fps:
                        continue
                    if matches_avoided_artist(cand.get("artist", ""), eff_avoid_art):
                        continue
                    if matches_avoided_genre(cand.get("genres", ""), eff_avoid_gen):
                        continue
                    if enforce_cap and artist_counts.get(pa, 0) >= max_per_artist:
                        continue
                    replacement = dict(cand)
                    break
                if replacement is not None:
                    break

            if replacement is None:
                raise ValueError("No additional unused candidate tracks remain in the current pool.")

            tracks[idx] = replacement
            eff_flow = flow_mode or payload.get("flow_mode") or "mmr"
            if flow_mode:
                tracks = sequence_playlist_tracks(tracks, flow_mode=eff_flow)
                payload["flow_mode"] = eff_flow
            payload["tracks"] = tracks

            self._save_payload(payload, update_archive=True)
            self.last_playlist = payload
            return {"playlist": payload, "dropped": dropped, "replacement": replacement}

    def resequence_current(self, flow_mode):
        """Re-order the current playlist tracks using a new DJ flow mode instantaneously."""
        with self._lock:
            payload = self.last_playlist or load_json_safe(LAST_PL_PATH)
            if not isinstance(payload, dict) or not payload.get("tracks"):
                raise ValueError("No active playlist loaded to sequence.")
            payload["tracks"] = sequence_playlist_tracks(payload["tracks"], flow_mode=flow_mode)
            payload["flow_mode"] = flow_mode
            self._save_payload(payload, update_archive=True)
            self.last_playlist = payload
            return payload

    def _resolve_playlist_payload(self, playlist_id=None):
        if playlist_id and playlist_id != "last":
            data = load_archived_playlist(playlist_id)
            if not isinstance(data, dict):
                raise FileNotFoundError(f"Archived playlist '{playlist_id}' not found.")
            return data
        data = self.last_playlist or load_json_safe(LAST_PL_PATH)
        if not isinstance(data, dict):
            raise FileNotFoundError("No previously generated playlist found.")
        return data

    def export_playlist(self, playlist_id=None, public=False, target_playlist_id=None):
        with self._lock:
            data = self._resolve_playlist_payload(playlist_id=playlist_id)
            sp = self._ensure_spotify()
            return export_to_spotify(data, sp=sp, public=public, target_playlist_id=target_playlist_id)

    def export_ytmusic_playlist(self, playlist_id=None, privacy="PRIVATE", target_playlist_id=None):
        with self._lock:
            data = self._resolve_playlist_payload(playlist_id=playlist_id)
            ytm = self._ensure_ytmusic()
            return export_to_ytmusic(data, ytm=ytm, privacy=privacy, target_playlist_id=target_playlist_id)

    def transfer_playlist(
        self,
        playlist_id=None,
        target_provider="ytmusic",
        public=False,
        privacy="PRIVATE",
        target_playlist_id=None,
    ):
        target = str(target_provider or "").strip().lower()
        if target not in ("spotify", "ytmusic"):
            raise ValueError("target_provider must be 'spotify' or 'ytmusic'.")
        with self._lock:
            data = self._resolve_playlist_payload(playlist_id=playlist_id)
            tracks = data.get("tracks") or []
            if not tracks:
                raise ValueError("Cannot transfer an empty playlist.")

            if target == "ytmusic":
                res = self.export_ytmusic_playlist(
                    playlist_id=playlist_id,
                    privacy=privacy,
                    target_playlist_id=target_playlist_id,
                )
                return {**res, "target_provider": "ytmusic"}

            sp = self._ensure_spotify()
            resolved = resolve_tracks_for_spotify(tracks, sp)
            if not resolved["uris"]:
                return {
                    "url": None,
                    "playlist_id": None,
                    "target_provider": "spotify",
                    "matched_count": 0,
                    "unmatched_count": resolved["unmatched_count"],
                    "unmatched": resolved["unmatched"],
                    "total_tracks": resolved["total_tracks"],
                    "error": "No tracks could be matched on Spotify.",
                }

            cache_file = os.path.join(ROOT_DIR, ".cache")
            target_uris = list(dict.fromkeys(resolved["uris"]))
            pl_name = data.get("playlist_name", "AI Playlist")
            pl_desc = data.get("playlist_description", "Generated by Music Taste Analyzer")
            for attempt in range(2):
                try:
                    if sp is None:
                        sp, _ = authenticate_spotify()
                        self.sp = sp
                    if target_playlist_id:
                        replace_spotify_playlist_items(
                            sp, target_playlist_id, target_uris, name=pl_name, description=pl_desc
                        )
                        pl_url = f"https://open.spotify.com/playlist/{target_playlist_id}"
                        out_pid = target_playlist_id
                    else:
                        new_pl = create_spotify_playlist(sp, name=pl_name, description=pl_desc, public=public)
                        add_items_to_spotify_playlist(sp, new_pl["id"], target_uris, batch_size=100)
                        out_pid = new_pl["id"]
                        pl_url = (new_pl.get("external_urls") or {}).get(
                            "spotify", f"https://open.spotify.com/playlist/{out_pid}"
                        )
                    return {
                        "playlist_id": out_pid,
                        "url": pl_url,
                        "target_provider": "spotify",
                        "matched_count": resolved["matched_count"],
                        "unmatched_count": resolved["unmatched_count"],
                        "unmatched": resolved["unmatched"],
                        "total_tracks": resolved["total_tracks"],
                    }
                except Exception as exc:
                    err_str = str(exc)
                    if attempt == 0 and ("403" in err_str or "Forbidden" in err_str):
                        if os.path.exists(cache_file):
                            try:
                                os.remove(cache_file)
                            except OSError:
                                pass
                        sp = None
                        self.sp = None
                        continue
                    raise

    def search_local_library(self, query, limit=15):
        """Search vectorized local track metadata for autocomplete pickers."""
        q = str(query or "").strip().lower()
        meta = self.local_meta or load_json_safe(os.path.join(STAGE2_DIR, "track_metadata.json"), default={}) or {}
        if not meta:
            return []

        scored = []
        for tid, info in meta.items():
            if not isinstance(info, dict):
                continue
            name = str(info.get("name") or "")
            artists = str(info.get("artists") or "")
            hay = f"{name} {artists}".lower()
            if q and q not in hay:
                continue
            score = float(info.get("recency_weight", 1.0))
            if q and name.lower().startswith(q):
                score += 50.0
            af = info.get("audio_features") if is_real_audio_features(info.get("audio_features")) else {}
            scored.append(
                (
                    score,
                    {
                        "id": str(tid),
                        "name": name,
                        "artist": artists,
                        "genres": info.get("genres", ""),
                        "bpm": af.get("bpm"),
                        "energy": af.get("energy"),
                        "valence": af.get("valence"),
                    },
                )
            )
        scored.sort(key=lambda x: x[0], reverse=True)
        return [item for _, item in scored[: max(1, min(50, int(limit)))]]

    def continue_from_now_playing(
        self, size=25, fast=False, flow_mode="harmonic", clean_only=False, deep_cuts=False, log_cb=None
    ):
        """Read currently playing Spotify track and generate a continuation mix anchored to its vibe."""
        from reccobeats_client import fetch_audio_features_batch

        with self._lock:
            sp = self._ensure_spotify(log_cb=log_cb)
            cur = get_currently_playing_track(sp)
            if not cur:
                raise ValueError("No active Spotify track detected. Play a song on Spotify and try again.")
            af_map = fetch_audio_features_batch([cur["id"]])
            af = af_map.get(cur["id"]) or {}
            af_text = format_audio_feature_text(af)
            prompt = f"Songs matching the vibe of {cur['name']} by {cur['artist']}"
            seed_text = f"Track: {cur['name']} | Artist: {cur['artist']}"
            if af_text:
                seed_text += f" | {af_text}"

            res = self.generate(
                prompt=prompt,
                fast=fast,
                size=size,
                flow_mode=flow_mode,
                clean_only=clean_only,
                deep_cuts=deep_cuts,
                seed_text=seed_text,
                log_cb=log_cb,
            )
            res["seed_track"] = cur
            return res

    def _resolve_bridge_endpoint(self, query_or_id):
        import numpy as np

        raw = str(query_or_id or "").strip()
        if not raw:
            raise ValueError("Both Start and End track/vibe inputs are required for Vibe Bridge.")

        if raw in self.local_meta:
            idx = self.local_ids.index(raw)
            info = self.local_meta[raw]
            af = info.get("audio_features") if is_real_audio_features(info.get("audio_features")) else {}
            return (
                self.local_embeddings[idx],
                float(af.get("energy", 0.5)),
                float(af.get("bpm", 120.0)),
                f"{info['name']} ({info['artists'].split(',')[0].strip()})",
                raw,
            )

        q_low = raw.lower()
        best_match = None
        for idx, tid in enumerate(self.local_ids):
            info = self.local_meta[tid]
            full = f"{info.get('name', '')} - {info.get('artists', '')}".lower()
            if q_low == info.get("name", "").lower() or q_low in full:
                best_match = (idx, tid, info)
                break

        if best_match is not None:
            idx, tid, info = best_match
            af = info.get("audio_features") if is_real_audio_features(info.get("audio_features")) else {}
            return (
                self.local_embeddings[idx],
                float(af.get("energy", 0.5)),
                float(af.get("bpm", 120.0)),
                f"{info['name']} ({info['artists'].split(',')[0].strip()})",
                tid,
            )

        vec = embed_texts([raw], self.embed_tokenizer, self.embed_model)[0]
        return vec, 0.5, 120.0, raw[:35], None

    def generate_vibe_bridge(self, start_query, end_query, steps=20, clean_only=False, log_cb=None):
        """Interpolate smoothly across 384-D vector space and BPM/energy from Start to End."""
        import numpy as np
        from sklearn.metrics.pairwise import cosine_similarity

        with self._lock:
            self._ensure_vectors_loaded(log_cb=log_cb)
            self.warm_up(load_llm=False, log_cb=log_cb)

            n_steps = max(4, min(60, int(steps or 20)))
            v_start, e_start, b_start, lbl_start, start_tid = self._resolve_bridge_endpoint(start_query)
            v_end, e_end, b_end, lbl_end, end_tid = self._resolve_bridge_endpoint(end_query)

            disliked_ids, disliked_fps = get_disliked_sets()
            avoid_art = self.session_avoid_artists
            avoid_gen = self.session_avoid_genres

            eligible_indices = []
            for idx, tid in enumerate(self.local_ids):
                info = self.local_meta.get(tid, {})
                fp = get_track_fingerprint(info.get("name", ""), info.get("artists", ""))
                if tid in disliked_ids or fp in disliked_fps:
                    continue
                if matches_avoided_artist(info.get("artists", ""), avoid_art):
                    continue
                if matches_avoided_genre(info.get("genres", ""), avoid_gen):
                    continue
                if clean_only and str(info.get("explicit", "")).lower() == "explicit":
                    continue
                eligible_indices.append(idx)

            if not eligible_indices:
                raise ValueError("No eligible tracks remain after applying session guardrails/filters.")

            sub_embs = self.local_embeddings[eligible_indices]
            max_per_artist = max(2, n_steps // 6)
            used_ids = set()
            used_fps = set()
            art_counts = {}
            bridge_tracks = [None] * n_steps

            def _claim_tid(tid):
                tr = self._format_local_track(tid)
                pa = tr["artist"].split(",")[0].strip().lower()
                used_ids.add(tid)
                used_fps.add(get_track_fingerprint(tr["name"], tr["artist"]))
                art_counts[pa] = art_counts.get(pa, 0) + 1
                return tr

            if start_tid and start_tid in self.local_meta and start_tid not in disliked_ids:
                bridge_tracks[0] = _claim_tid(start_tid)
            if end_tid and end_tid in self.local_meta and end_tid != start_tid and end_tid not in disliked_ids:
                bridge_tracks[-1] = _claim_tid(end_tid)

            for step_i in range(n_steps):
                if bridge_tracks[step_i] is not None:
                    continue
                t = step_i / float(max(1, n_steps - 1))
                interp = ((1.0 - t) * v_start) + (t * v_end)
                norm = float(np.linalg.norm(interp))
                if norm > 0:
                    interp = interp / norm
                target_e = ((1.0 - t) * e_start) + (t * e_end)
                target_b = ((1.0 - t) * b_start) + (t * b_end)

                sims = cosine_similarity([interp], sub_embs)[0]
                best_tid = None
                best_score = -1e9
                for pos, orig_idx in enumerate(eligible_indices):
                    tid = self.local_ids[orig_idx]
                    info = self.local_meta[tid]
                    fp = get_track_fingerprint(info.get("name", ""), info.get("artists", ""))
                    pa = str(info.get("artists") or "").split(",")[0].strip().lower()
                    if tid in used_ids or fp in used_fps:
                        continue
                    if art_counts.get(pa, 0) >= max_per_artist:
                        continue

                    af = info.get("audio_features") if is_real_audio_features(info.get("audio_features")) else {}
                    e_diff = abs(float(af.get("energy", target_e)) - target_e) if "energy" in af else 0.0
                    b_diff = abs(float(af.get("bpm", target_b)) - target_b) if "bpm" in af else 0.0
                    score = float(sims[pos]) - (0.15 * e_diff) - (0.002 * b_diff)
                    if score > best_score:
                        best_score = score
                        best_tid = tid

                if best_tid:
                    bridge_tracks[step_i] = _claim_tid(best_tid)

            final_tracks = [tr for tr in bridge_tracks if tr is not None]
            if not final_tracks:
                raise ValueError("Could not assemble a Vibe Bridge playlist.")

            payload = {
                "prompt": f"Vibe Bridge: {lbl_start} -> {lbl_end}",
                "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat().replace("+00:00", "Z"),
                "playlist_name": f"Bridge: {lbl_start} → {lbl_end}"[:100],
                "playlist_description": f"Smooth {len(final_tracks)}-track acoustic & vector transition from {lbl_start} to {lbl_end}.",
                "flow_mode": "vibe_bridge",
                "avoid_artists": list(avoid_art),
                "avoid_genres": list(avoid_gen),
                "tracks": final_tracks,
            }
            self._save_payload(payload, update_archive=True)
            self.last_playlist = payload
            return {"playlist": payload}

    def refine_playlist(self, instruction, fast=False, log_cb=None):
        """Apply conversational refinement edits to the active playlist."""
        with self._lock:
            payload = self.last_playlist or load_json_safe(LAST_PL_PATH)
            if not isinstance(payload, dict) or not payload.get("tracks"):
                raise ValueError("No active playlist loaded to refine.")

            tracks = list(payload["tracks"])
            active_pipe = None if fast else self.pipe
            plan = parse_refinement_instruction(active_pipe, instruction, tracks)

            drop_set = set(plan.get("drop_indices") or [])
            rem_artists = normalize_avoid_list(plan.get("remove_artists") or [])
            cond = plan.get("filter_condition")

            kept = []
            dropped_ids = set()
            dropped_fps = set()
            for idx, tr in enumerate(tracks):
                remove = idx in drop_set
                if not remove and rem_artists and matches_avoided_artist(tr.get("artist", ""), rem_artists):
                    remove = True
                if not remove and cond:
                    bpm = tr.get("bpm")
                    eng = tr.get("energy")
                    if cond == "drop_slow" and isinstance(bpm, (int, float)) and bpm < 105:
                        remove = True
                    elif cond == "drop_fast" and isinstance(bpm, (int, float)) and bpm > 135:
                        remove = True
                    elif cond == "drop_low_energy" and isinstance(eng, (int, float)) and eng < 0.45:
                        remove = True
                    elif cond == "drop_high_energy" and isinstance(eng, (int, float)) and eng > 0.75:
                        remove = True
                    elif cond == "drop_explicit" and tr.get("explicit"):
                        remove = True

                if remove:
                    dropped_ids.add(tr.get("id"))
                    dropped_fps.add(get_track_fingerprint(tr.get("name", ""), tr.get("artist", "")))
                else:
                    kept.append(tr)

            orig_size = len(tracks)
            if plan.get("target_size") is not None:
                desired_size = max(1, min(100, int(plan["target_size"])))
            else:
                desired_size = max(1, min(100, orig_size + int(plan.get("delta_tracks") or 0)))

            if len(kept) > desired_size:
                kept = kept[:desired_size]
            elif len(kept) < desired_size:
                needed = desired_size - len(kept)
                used_ids = {t.get("id") for t in kept} | dropped_ids
                used_fps = {get_track_fingerprint(t.get("name", ""), t.get("artist", "")) for t in kept} | dropped_fps
                disliked_ids, disliked_fps = get_disliked_sets()
                eff_avoid_art = normalize_avoid_list((payload.get("avoid_artists") or []) + rem_artists + self.session_avoid_artists)
                eff_avoid_gen = normalize_avoid_list((payload.get("avoid_genres") or []) + self.session_avoid_genres)

                direction = plan.get("direction_prompt")
                pool = []
                if not direction and self.last_candidates:
                    pool = list(self.last_candidates)
                elif self._stage2_sig()[0][0]:
                    import numpy as np

                    self._ensure_vectors_loaded(log_cb=log_cb)
                    self.warm_up(load_llm=False, log_cb=log_cb)
                    dir_query = direction or payload.get("prompt") or "popular favorites"
                    q_vec = embed_texts([dir_query], self.embed_tokenizer, self.embed_model)[0]
                    blended = (0.75 * q_vec) + (0.25 * self.master_centroid)
                    nrm = float(np.linalg.norm(blended))
                    t_vec = (blended / nrm) if nrm > 0 else q_vec
                    mmr_idx = compute_mmr(t_vec, self.local_embeddings, top_k=min(needed + 30, len(self.local_ids)))
                    pool = [self._format_local_track(self.local_ids[i]) for i in mmr_idx]
                elif self.last_candidates:
                    pool = list(self.last_candidates)

                for cand in pool:
                    if len(kept) >= desired_size:
                        break
                    cid = cand.get("id")
                    cfp = get_track_fingerprint(cand.get("name", ""), cand.get("artist", ""))
                    if cid in used_ids or cfp in used_fps or cid in disliked_ids or cfp in disliked_fps:
                        continue
                    if matches_avoided_artist(cand.get("artist", ""), eff_avoid_art):
                        continue
                    if matches_avoided_genre(cand.get("genres", ""), eff_avoid_gen):
                        continue
                    used_ids.add(cid)
                    used_fps.add(cfp)
                    kept.append(dict(cand))

            if plan.get("flow_mode"):
                kept = sequence_playlist_tracks(kept, flow_mode=plan["flow_mode"])
                payload["flow_mode"] = plan["flow_mode"]

            payload["tracks"] = kept
            if rem_artists:
                payload["avoid_artists"] = normalize_avoid_list((payload.get("avoid_artists") or []) + rem_artists)
            self._save_payload(payload, update_archive=True)
            self.last_playlist = payload
            return {"playlist": payload, "plan": plan}

    def update_current_playlist(
        self, playlist_name=None, playlist_description=None, order_indices=None, remove_indices=None
    ):
        """Apply inline title/description edits, track removals, or drag-and-drop reordering."""
        with self._lock:
            payload = self.last_playlist or load_json_safe(LAST_PL_PATH)
            if not isinstance(payload, dict) or not payload.get("tracks"):
                raise ValueError("No active playlist loaded to edit.")

            tracks = list(payload["tracks"])
            if order_indices is not None:
                if len(order_indices) != len(tracks) or set(order_indices) != set(range(len(tracks))):
                    raise ValueError("Invalid order_indices permutation.")
                tracks = [tracks[i] for i in order_indices]

            if remove_indices:
                rm_set = set(remove_indices)
                tracks = [tr for idx, tr in enumerate(tracks) if idx not in rm_set]

            if playlist_name is not None and str(playlist_name).strip():
                payload["playlist_name"] = str(playlist_name).strip()[:100]
            if playlist_description is not None:
                payload["playlist_description"] = str(playlist_description).strip()[:300]
            payload["tracks"] = tracks

            self._save_payload(payload, update_archive=True)
            self.last_playlist = payload
            return payload

    def generate_blend(
        self,
        friend_summary,
        blend_ratio=0.5,
        prompt="",
        size=30,
        fast=True,
        flow_mode="mmr",
        clean_only=False,
        log_cb=None,
    ):
        """Compute Taste Compatibility % and generate a shared playlist from two taste profiles."""
        import numpy as np
        from sklearn.metrics.pairwise import cosine_similarity

        if not isinstance(friend_summary, dict) or not friend_summary.get("master_taste_centroid"):
            raise ValueError("Friend JSON must contain a valid 'master_taste_centroid' vector.")

        with self._lock:
            self._ensure_vectors_loaded(log_cb=log_cb)
            friend_vec = np.array(friend_summary["master_taste_centroid"], dtype=float)
            if friend_vec.shape != self.master_centroid.shape:
                raise ValueError(
                    f"Vector dimensionality mismatch: expected {self.master_centroid.shape}, got {friend_vec.shape}."
                )

            f_norm = float(np.linalg.norm(friend_vec))
            if f_norm > 0:
                friend_vec = friend_vec / f_norm

            cos_sim = float(cosine_similarity([self.master_centroid], [friend_vec])[0][0])
            compatibility_pct = round(max(0.0, min(100.0, ((cos_sim + 0.15) / 1.15) * 100.0)), 1)

            shared_vibes = []
            for mc in self.clusters:
                mv = np.array(mc.get("centroid_vector") or [], dtype=float)
                if mv.shape != friend_vec.shape:
                    continue
                for fc in friend_summary.get("clusters") or []:
                    fv = np.array(fc.get("centroid_vector") or [], dtype=float)
                    if fv.shape != friend_vec.shape:
                        continue
                    sim = float(cosine_similarity([mv], [fv])[0][0])
                    shared_vibes.append(
                        {
                            "my_cluster": mc.get("vibe_label"),
                            "friend_cluster": fc.get("vibe_label"),
                            "similarity_pct": round(max(0.0, min(100.0, sim * 100.0)), 1),
                        }
                    )
            shared_vibes.sort(key=lambda x: x["similarity_pct"], reverse=True)

            ratio = max(0.0, min(1.0, float(blend_ratio)))
            shared_centroid = ((1.0 - ratio) * self.master_centroid) + (ratio * friend_vec)
            s_norm = float(np.linalg.norm(shared_centroid))
            if s_norm > 0:
                shared_centroid = shared_centroid / s_norm

            blend_prompt = str(prompt or "").strip() or "Shared taste blend favorites and fresh discoveries"
            res = self.generate(
                prompt=blend_prompt,
                fast=fast,
                size=size,
                flow_mode=flow_mode,
                clean_only=clean_only,
                custom_centroid=shared_centroid,
                log_cb=log_cb,
            )
            res["compatibility_pct"] = compatibility_pct
            res["shared_vibes"] = shared_vibes[:3]
            return res

    def _resolve_playlist_uris(self, playlist_id=None):
        with self._lock:
            if playlist_id and playlist_id != "last":
                data = load_archived_playlist(playlist_id)
                if not isinstance(data, dict):
                    raise FileNotFoundError(f"Archived playlist '{playlist_id}' not found.")
            else:
                data = self.last_playlist or load_json_safe(LAST_PL_PATH)
                if not isinstance(data, dict) or not data.get("tracks"):
                    raise ValueError("No active playlist loaded.")
            uris = [
                f"spotify:track:{t['id']}"
                for t in (data.get("tracks") or [])
                if isinstance(t, dict) and is_valid_spotify_id(t.get("id"))
            ]
            if not uris:
                raise ValueError("Playlist contains no valid Spotify track IDs.")
            return data, uris

    def queue_current_playlist(self, playlist_id=None, max_items=30, log_cb=None):
        _, uris = self._resolve_playlist_uris(playlist_id=playlist_id)
        sp = self._ensure_spotify(log_cb=log_cb)
        return queue_tracks_on_spotify(sp, uris, max_items=max_items)

    def play_current_playlist(self, playlist_id=None, device_id=None, log_cb=None):
        _, uris = self._resolve_playlist_uris(playlist_id=playlist_id)
        sp = self._ensure_spotify(log_cb=log_cb)
        return start_spotify_playback(sp, uris, device_id=device_id)

    def get_playback_queue(self, log_cb=None):
        sp = self._ensure_spotify(log_cb=log_cb)
        return get_user_playback_queue(sp)

    def get_release_radar(
        self, days=45, max_artists=20, save_as_playlist=False, flow_mode="mmr", log_cb=None
    ):
        """Scan recent albums & singles from the user's Top + Followed artists (replaces removed /browse/new-releases)."""
        from reccobeats_client import fetch_audio_features_batch

        profile = load_json_safe(os.path.join(STAGE1_DIR, "user_music_profile.json"))
        if not isinstance(profile, dict):
            raise FileNotFoundError("user_music_profile.json not found. Run Stage 1 Spotify Sync first.")

        artist_ids = []
        seen = set()
        for src in (
            (profile.get("artists") or {}).get("short_term") or [],
            profile.get("followed_artists") or [],
            (profile.get("artists") or {}).get("medium_term") or [],
        ):
            for art in src:
                if isinstance(art, dict) and art.get("id") and art["id"] not in seen:
                    seen.add(art["id"])
                    artist_ids.append(art["id"])

        if not artist_ids:
            raise ValueError("No top or followed artist IDs found in Stage 1 profile.")

        sp = self._ensure_spotify(log_cb=log_cb)
        raw_radar = fetch_personal_release_radar(sp, artist_ids, days=days, max_artists=max_artists)

        disliked_ids, disliked_fps = get_disliked_sets()
        avoid_art = self.session_avoid_artists
        filtered = []
        for tr in raw_radar.get("tracks") or []:
            fp = get_track_fingerprint(tr.get("name", ""), tr.get("artist", ""))
            if tr.get("id") in disliked_ids or fp in disliked_fps:
                continue
            if matches_avoided_artist(tr.get("artist", ""), avoid_art):
                continue
            filtered.append(tr)

        tids = [t["id"] for t in filtered if is_valid_spotify_id(t.get("id"))]
        af_map = {}
        if tids:
            try:
                af_map = fetch_audio_features_batch(tids)
            except Exception:
                af_map = {}

        enriched = []
        for tr in filtered:
            item = dict(tr)
            af = af_map.get(tr["id"], {})
            real_af = af if is_real_audio_features(af) else {}
            item["bpm"] = real_af.get("bpm")
            item["energy"] = real_af.get("energy")
            item["valence"] = real_af.get("valence")
            item["key"] = real_af.get("key")
            item["mode"] = real_af.get("mode")
            item["camelot"] = format_camelot_label(real_af.get("key"), real_af.get("mode"))
            enriched.append(item)

        sequenced = sequence_playlist_tracks(enriched, flow_mode=flow_mode or "mmr")
        saved_payload = None
        if save_as_playlist and sequenced:
            saved_payload = {
                "prompt": f"Personal Release Radar (last {days} days across {min(len(artist_ids), int(max_artists))} artists)",
                "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat().replace("+00:00", "Z"),
                "playlist_name": f"Release Radar ({days}d)",
                "playlist_description": "Fresh albums and singles from your Top & Followed Spotify artists.",
                "flow_mode": flow_mode or "mmr",
                "avoid_artists": list(avoid_art),
                "avoid_genres": list(self.session_avoid_genres),
                "tracks": sequenced,
            }
            self._save_payload(saved_payload, update_archive=True)
            self.last_playlist = saved_payload

        return {
            "releases": raw_radar.get("releases") or [],
            "tracks": sequenced,
            "playlist": saved_payload,
        }


def _view_last_playlist(last_pl_path):
    if not os.path.exists(last_pl_path):
        print("❌ Error: No previously generated playlist found.")
        print("Please generate a playlist first using Stage 3!")
        sys.exit(1)

    data = load_json_safe(last_pl_path)
    if not isinstance(data, dict):
        print(f"❌ Error: {last_pl_path} is corrupted.")
        sys.exit(1)

    tracks = data.get("tracks") or []
    print("==================================================")
    print(f" 🎵 LAST GENERATED PLAYLIST: '{data.get('playlist_name', 'AI Playlist')}'")
    print("==================================================")
    if data.get("prompt"):
        print(f" Prompt          : {data['prompt']}")
    if data.get("generated_at"):
        print(f" Generated At    : {data['generated_at']}")
    if data.get("flow_mode"):
        print(f" DJ Flow Mode    : {data['flow_mode']}")
    print(f" Vibe Description: {data.get('playlist_description', 'N/A')}")
    print(f" Total Tracks    : {len(tracks)}")

    bpms = [t["bpm"] for t in tracks if isinstance(t.get("bpm"), (int, float))]
    energies = [t["energy"] for t in tracks if isinstance(t.get("energy"), (int, float))]
    valences = [t["valence"] for t in tracks if isinstance(t.get("valence"), (int, float))]
    if bpms:
        avg_bpm = round(sum(bpms) / len(bpms), 1)
        avg_eng = round(sum(energies) / len(energies), 2) if energies else "N/A"
        avg_val = round(sum(valences) / len(valences), 2) if valences else "N/A"
        print(f" Acoustic Avg    : {avg_bpm} BPM | Energy {avg_eng} | Valence {avg_val}")
    print("-" * 50)

    for i, track in enumerate(tracks, start=1):
        badge = "🔵 [LIBRARY]" if track.get("source") == "Local Library" else "✨ [NEW]"
        meta_bits = []
        if track.get("bpm") is not None:
            meta_bits.append(f"{track['bpm']} BPM")
        if track.get("energy") is not None:
            meta_bits.append(f"E:{track['energy']}")
        if track.get("valence") is not None:
            meta_bits.append(f"V:{track['valence']}")
        if track.get("camelot"):
            meta_bits.append(track["camelot"])
        suffix = f" ({' | '.join(meta_bits)})" if meta_bits else ""
        print(f"  {i:2d}. {badge} {track.get('name', 'Unknown')} - {track.get('artist', 'Unknown')}{suffix}")

    print("\n" + "-" * 50)
    print("💾 File: 03_Agentic_Engine/last_generated_playlist.json")


def main():
    configure_utf8_stdio()

    parser = argparse.ArgumentParser(description="Music Taste Analyzer - Stage 3 Discovery Engine")
    parser.add_argument("prompt", nargs="?", help="Playlist prompt / vibe request")
    parser.add_argument("--no-export", action="store_true", help="Keep generated playlist local only")
    parser.add_argument("--auto-export", action="store_true", help="Automatically export playlist to Spotify")
    parser.add_argument("--view-last", action="store_true", help="Display last generated playlist")
    parser.add_argument("--export-last", action="store_true", help="Export last generated playlist to Spotify")
    parser.add_argument("--fast", action="store_true", help="Skip Local LLM and use fast heuristic + vector mode")
    parser.add_argument("--size", type=int, default=None, help="Override playlist size (5-100)")
    parser.add_argument("--local-only", action="store_true", help="Restrict recommendations to your local library")
    parser.add_argument("--public", action="store_true", help="Create exported Spotify playlist as public")
    parser.add_argument("--cluster", type=int, default=None, help="Seed discovery around a specific Stage 2 cluster ID")
    parser.add_argument(
        "--flow",
        choices=["mmr", "energy_ramp", "chill_descent", "warmup_peak_cooldown", "harmonic", "shuffle"],
        default="mmr",
        help="DJ track sequencing mode",
    )
    parser.add_argument("--clean", action="store_true", help="Exclude explicit tracks")
    parser.add_argument("--deep-cuts", action="store_true", help="Prefer lesser-known deep cuts (popularity 15-58)")
    parser.add_argument("--avoid-artists", type=str, default=None, help="Comma-separated artists to avoid")
    parser.add_argument("--avoid-genres", type=str, default=None, help="Comma-separated genres to avoid")
    parser.add_argument("--bridge-start", type=str, default=None, help="Start track/vibe for Vibe Bridge mode")
    parser.add_argument("--bridge-end", type=str, default=None, help="End track/vibe for Vibe Bridge mode")
    parser.add_argument("--bridge-steps", type=int, default=20, help="Number of tracks in Vibe Bridge")
    parser.add_argument("--target-playlist", type=str, default=None, help="Existing Spotify playlist ID to update in-place")
    parser.add_argument(
        "--provider",
        choices=["spotify", "ytmusic", "unified"],
        default=None,
        help="Music provider source mode (default: active preference)",
    )
    parser.add_argument("--interactive", action="store_true", help="Keep models warm in an interactive prompt loop")

    args = parser.parse_args()

    if args.export_last:
        if not os.path.exists(LAST_PL_PATH):
            print("❌ Error: No previously generated playlist found.")
            sys.exit(1)
        data = load_json_safe(LAST_PL_PATH)
        if not isinstance(data, dict):
            print("❌ Error: last_generated_playlist.json is corrupted.")
            sys.exit(1)
        ok = export_to_spotify(data, public=args.public, target_playlist_id=args.target_playlist)
        sys.exit(0 if ok else 1)

    if args.view_last:
        _view_last_playlist(LAST_PL_PATH)
        return

    is_bridge = bool(args.bridge_start and args.bridge_end)
    if not args.interactive and not is_bridge and (not args.prompt or not args.prompt.strip()) and args.cluster is None:
        print("❌ Error: No prompt provided.")
        sys.exit(1)

    summary_path = os.path.join(STAGE2_DIR, "taste_profile_summary.json")
    vectors_path = os.path.join(STAGE2_DIR, "taste_vectors.npz")
    if not os.path.exists(summary_path) or not os.path.exists(vectors_path):
        print("❌ Error: Missing Vector Core data.")
        print("Please ensure you run Stage 2 first to generate your taste clusters!")
        sys.exit(1)

    print("==================================================")
    print(" NPU Discovery Engine - Stage 3 (Warm Session)")
    print("==================================================")

    session = WarmDiscoverySession()
    if args.avoid_artists is not None or args.avoid_genres is not None:
        session.set_session_exclusions(avoid_artists=args.avoid_artists, avoid_genres=args.avoid_genres)

    def run_once(prompt_text):
        if is_bridge:
            result = session.generate_vibe_bridge(
                args.bridge_start,
                args.bridge_end,
                steps=args.bridge_steps,
                clean_only=args.clean,
            )
        else:
            result = session.generate(
                prompt=prompt_text,
                fast=args.fast,
                size=args.size,
                local_only=args.local_only,
                cluster_id=args.cluster,
                flow_mode=args.flow,
                clean_only=args.clean,
                deep_cuts=args.deep_cuts,
                provider=args.provider,
            )
        payload = result["playlist"]
        tracks = payload["tracks"]
        print(f"\n🎉 Generated '{payload['playlist_name']}' ({len(tracks)} tracks | Flow: {payload['flow_mode']}):")
        for i, track in enumerate(tracks, start=1):
            badge = "🔵 [LIBRARY]" if track["source"] == "Local Library" else "✨ [NEW]"
            bpm_txt = f" ({track['bpm']} BPM)" if track.get("bpm") is not None else ""
            print(f"  {i:2d}. {badge} {track['name']} - {track['artist']}{bpm_txt}")
        print("\n" + "-" * 50)
        return payload

    if args.interactive:
        session.warm_up(load_llm=not args.fast)
        print("\n🔥 Warm Stage 3 Session Ready! Type prompts, or 'exit' to quit.")
        while True:
            try:
                user_in = input("\nPrompt > ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not user_in or user_in.lower() in ("exit", "quit", "q"):
                break
            try:
                payload = run_once(user_in)
                if args.auto_export:
                    export_to_spotify(payload, sp=session.sp, public=args.public)
            except Exception as e:
                print(f"❌ Generation error: {e}")
        return

    try:
        payload = run_once(args.prompt or "")
    except Exception as e:
        print(f"\n❌ Pipeline Failed: {e}")
        sys.exit(1)

    if args.auto_export:
        ok = export_to_spotify(payload, sp=session.sp, public=args.public, target_playlist_id=args.target_playlist)
        sys.exit(0 if ok else 1)
    elif args.no_export:
        print("\n💾 Playlist saved locally to '03_Agentic_Engine/last_generated_playlist.json'.")
    elif sys.stdin and sys.stdin.isatty():
        confirm = input("\n💾 Would you like to export this playlist to Spotify right now? (y/N): ").strip().lower()
        if confirm in ("y", "yes"):
            ok = export_to_spotify(payload, sp=session.sp, public=args.public, target_playlist_id=args.target_playlist)
            sys.exit(0 if ok else 1)
        else:
            print("\n👍 Playlist kept local. You can export it later anytime!")
    else:
        print("\n💾 Playlist saved locally to '03_Agentic_Engine/last_generated_playlist.json'.")


if __name__ == "__main__":
    main()
