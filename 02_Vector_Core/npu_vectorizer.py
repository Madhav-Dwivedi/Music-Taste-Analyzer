import argparse
import hashlib
import json
import os
import sys
import time

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import config
from utils import (
    atomic_write_json,
    configure_utf8_stdio,
    extract_ytm_video_id,
    format_audio_feature_text,
    get_effective_preferences,
    get_openvino_devices,
    get_track_fingerprint,
    get_track_provider,
    is_real_audio_features,
    is_valid_spotify_id,
    load_json_safe,
    load_merged_music_profile,
)

MODEL_ID = config.EMBED_MODEL_ID
EMBED_MODEL_DIR = getattr(config, "EMBED_MODEL_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "embed_ov_model"))


def mean_pooling(model_output, attention_mask):
    """Mean Pooling - Take attention mask into account for correct averaging."""
    import torch

    token_embeddings = model_output[0] if not hasattr(model_output, "last_hidden_state") else model_output.last_hidden_state
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


def _has_cached_openvino_ir(model_dir):
    return (
        os.path.isdir(model_dir)
        and os.path.exists(os.path.join(model_dir, "openvino_model.xml"))
        and os.path.exists(os.path.join(model_dir, "openvino_model.bin"))
    )


def get_openvino_model():
    from transformers import AutoTokenizer

    print(f"[1/4] Loading {MODEL_ID} optimized for OpenVINO...")
    dev_info = get_openvino_devices()

    tokenizer_src = EMBED_MODEL_DIR if os.path.exists(os.path.join(EMBED_MODEL_DIR, "tokenizer_config.json")) else MODEL_ID
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_src)

    model = None
    try:
        from optimum.intel import OVModelForFeatureExtraction

        if not _has_cached_openvino_ir(EMBED_MODEL_DIR):
            print(f"      * Exporting & caching OpenVINO IR to {EMBED_MODEL_DIR}...")
            os.makedirs(EMBED_MODEL_DIR, exist_ok=True)
            exported_model = OVModelForFeatureExtraction.from_pretrained(MODEL_ID, export=True, compile=False)
            # Save BEFORE any NPU static reshape so the cached IR stays dynamic for GPU/CPU.
            exported_model.save_pretrained(EMBED_MODEL_DIR)
            tokenizer.save_pretrained(EMBED_MODEL_DIR)
            model_source = EMBED_MODEL_DIR
            export_flag = False
        else:
            model_source = EMBED_MODEL_DIR
            export_flag = False

        if dev_info["npu"]:
            try:
                model_temp = OVModelForFeatureExtraction.from_pretrained(model_source, export=export_flag, compile=False)
                model_temp.reshape(1, 128)
                model_temp.to(dev_info["npu"])
                model_temp.compile()
                model_temp.is_static_npu = True
                model_temp.is_pytorch_fallback = False
                print(f"      ✅ OpenVINO Model compiled successfully for Neural Processing Unit ({dev_info['npu']})!")
                model = model_temp
            except Exception as e:
                print(f"      ⚠️ NPU compilation failed: {e}")

        if model is None and dev_info["gpu"]:
            print(f"      * Found Intel GPU ({dev_info['gpu']}), attempting hardware acceleration...")
            try:
                model_temp = OVModelForFeatureExtraction.from_pretrained(model_source, export=export_flag, compile=False)
                model_temp.to(dev_info["gpu"])
                model_temp.compile()
                model_temp.is_static_npu = False
                model_temp.is_pytorch_fallback = False
                print(f"      ✅ OpenVINO Model compiled successfully for System GPU ({dev_info['gpu']})!")
                model = model_temp
            except Exception as e:
                print(f"      ⚠️ GPU compilation failed: {e}")

        if model is None:
            model_temp = OVModelForFeatureExtraction.from_pretrained(model_source, export=export_flag, compile=False)
            model_temp.to(dev_info["cpu"])
            model_temp.compile()
            model_temp.is_static_npu = False
            model_temp.is_pytorch_fallback = False
            print(f"      ✅ OpenVINO Model compiled successfully for {dev_info['cpu']}!")
            model = model_temp
    except Exception as e:
        print(f"      ⚠️ OpenVINO embedding runtime failed ({e}); falling back to PyTorch CPU...")
        from transformers import AutoModel

        model = AutoModel.from_pretrained(MODEL_ID)
        model.eval()
        model.is_static_npu = False
        model.is_pytorch_fallback = True
        print("      ✅ Loaded PyTorch CPU fallback embedding model!")

    return tokenizer, model


def _build_artist_genre_lookup(profile):
    by_id = {}
    by_name = {}
    artist_sources = list((profile.get("artists") or {}).values()) + [profile.get("followed_artists") or []]
    for group in artist_sources:
        for art in group or []:
            if not isinstance(art, dict):
                continue
            genres = [g.strip() for g in (art.get("genres") or []) if isinstance(g, str) and g.strip()]
            if not genres:
                continue
            if art.get("id"):
                by_id[str(art["id"])] = genres
            if art.get("name"):
                by_name[art["name"].strip().lower()] = genres
    return by_id, by_name


def _merge_unique(existing_list, new_items):
    seen = set(existing_list)
    for item in new_items:
        if item and item not in seen:
            seen.add(item)
            existing_list.append(item)


def build_semantic_documents(profile):
    print("\n[2/4] Synthesizing Semantic Text Representations for all tracks...")

    artist_genres_by_id, artist_genres_by_name = _build_artist_genre_lookup(profile)

    # Map every track ID to the ordered unique playlists it appears in
    track_to_playlists = {}
    for pl in profile.get("playlists") or []:
        if not isinstance(pl, dict):
            continue
        pl_name = (pl.get("name") or "").strip()
        if not pl_name:
            continue
        for track in pl.get("tracks") or []:
            if not isinstance(track, dict):
                continue
            tid = track.get("id")
            if tid:
                tid_str = str(tid)
                pl_list = track_to_playlists.setdefault(tid_str, [])
                if pl_name not in pl_list:
                    pl_list.append(pl_name)

    unique_tracks = {}
    fp_to_canonical_tid = {}

    def extract_tracks(track_list):
        for t in track_list or []:
            if not isinstance(t, dict) or not t.get("id"):
                continue
            tid = str(t["id"])
            name = t.get("name") or "Unknown"

            raw_artists = [a for a in (t.get("artists") or []) if isinstance(a, dict) and a.get("name")]
            artist_names = [a["name"] for a in raw_artists]
            fp = get_track_fingerprint(name, artist_names)

            track_genres = []
            for a in raw_artists:
                _merge_unique(track_genres, [g.strip() for g in (a.get("genres") or []) if isinstance(g, str) and g.strip()])
                aid = a.get("id")
                if aid and str(aid) in artist_genres_by_id:
                    _merge_unique(track_genres, artist_genres_by_id[str(aid)])
                aname = a.get("name", "").strip().lower()
                if aname in artist_genres_by_name:
                    _merge_unique(track_genres, artist_genres_by_name[aname])

            release_year = ""
            album = t.get("album")
            if isinstance(album, dict) and album.get("release_date"):
                release_year = str(album["release_date"])[:4]

            explicit_val = t.get("explicit")
            explicit_tag = "Explicit" if explicit_val is True else ("Clean" if explicit_val is False else "")

            playlists_for_tid = track_to_playlists.get(tid, [])
            t_prov = get_track_provider(t)
            sp_id = t.get("spotify_id") if is_valid_spotify_id(t.get("spotify_id")) else (tid if is_valid_spotify_id(tid) else None)
            yt_vid = extract_ytm_video_id(t.get("video_id") or tid)

            if fp not in fp_to_canonical_tid:
                fp_to_canonical_tid[fp] = tid
                pl_list = list(playlists_for_tid)
                prov_set = {"spotify", "ytmusic"} if t_prov == "both" else {t_prov}
                unique_tracks[tid] = {
                    "name": name,
                    "artists": ", ".join(artist_names) if artist_names else "Unknown",
                    "genres_list": track_genres,
                    "playlist_list": pl_list,
                    "year": release_year,
                    "explicit": explicit_tag or "Clean",
                    "alt_ids": [],
                    "_providers": prov_set,
                    "spotify_id": sp_id,
                    "video_id": yt_vid,
                }
            else:
                canon_tid = fp_to_canonical_tid[fp]
                canon = unique_tracks[canon_tid]
                if tid != canon_tid and tid not in canon["alt_ids"]:
                    canon["alt_ids"].append(tid)
                if t_prov == "both":
                    canon["_providers"].update(("spotify", "ytmusic"))
                else:
                    canon["_providers"].add(t_prov)
                if not canon.get("spotify_id") and sp_id:
                    canon["spotify_id"] = sp_id
                if not canon.get("video_id") and yt_vid:
                    canon["video_id"] = yt_vid
                _merge_unique(canon["playlist_list"], playlists_for_tid)
                _merge_unique(canon["genres_list"], track_genres)
                if not canon["year"] and release_year:
                    canon["year"] = release_year
                if explicit_tag == "Explicit":
                    canon["explicit"] = "Explicit"

    tracks_section = profile.get("tracks") or {}
    extract_tracks(tracks_section.get("short_term", []))
    extract_tracks(tracks_section.get("medium_term", []))
    extract_tracks(tracks_section.get("long_term", []))

    for item in profile.get("recently_played") or []:
        if isinstance(item, dict) and item.get("track"):
            extract_tracks([item["track"]])

    extract_tracks(profile.get("short_term_artists_top_tracks", []))
    extract_tracks(profile.get("liked_songs_sample", []))

    for pl in profile.get("playlists") or []:
        if isinstance(pl, dict):
            extract_tracks(pl.get("tracks", []))

    for alb in profile.get("saved_albums") or []:
        if isinstance(alb, dict):
            extract_tracks(alb.get("tracks", []))

    af_map = profile.get("audio_features") or {}
    docs = []
    track_ids = []

    for tid, t in unique_tracks.items():
        genres_list = t.pop("genres_list")
        prov_set = t.pop("_providers", {"spotify"})
        if "spotify" in prov_set and "ytmusic" in prov_set:
            t["provider"] = "both"
        elif "ytmusic" in prov_set:
            t["provider"] = "ytmusic"
        else:
            t["provider"] = "spotify"
        pl_list = t["playlist_list"]
        t["genres"] = ", ".join(genres_list)
        t["playlists"] = ", ".join(pl_list)

        af = {}
        for candidate_id in [tid] + t["alt_ids"]:
            cand_af = af_map.get(candidate_id)
            if is_real_audio_features(cand_af):
                af = cand_af
                break
        t["audio_features"] = af

        doc = f"Track: {t['name']} | Artist: {t['artists']}"
        if t.get("year"):
            doc += f" | Year: {t['year']}"
        if t.get("explicit"):
            doc += f" | Content: {t['explicit']}"
        if t["genres"]:
            doc += f" | Genres: {t['genres']}"

        af_text = format_audio_feature_text(af)
        if af_text:
            doc += f" | {af_text}"

        # Keep playlist membership in metadata for soft co-occurrence, not raw text leakage
        t["doc_hash"] = hashlib.sha256(doc.encode("utf-8")).hexdigest()[:16]
        docs.append(doc)
        track_ids.append(tid)

    return docs, track_ids, unique_tracks


def apply_soft_playlist_cooccurrence(embeddings, track_ids, metadata, weight=0.12, catchall_ratio=0.40):
    """Gently blend track vectors toward curated (non-catch-all) playlist centroids."""
    import numpy as np

    mat = np.asarray(embeddings, dtype=np.float32).copy()
    n_tracks = len(track_ids)
    if n_tracks == 0 or len(mat) != n_tracks or weight <= 0.0:
        return mat

    w = float(max(0.0, min(0.5, weight)))
    tid_to_idx = {str(tid): i for i, tid in enumerate(track_ids)}
    playlist_members = {}
    for tid in track_ids:
        info = metadata.get(str(tid), {})
        for pl_name in info.get("playlist_list") or []:
            clean_pl = str(pl_name or "").strip()
            if clean_pl:
                playlist_members.setdefault(clean_pl, []).append(tid_to_idx[str(tid)])

    max_members = max(2, int(n_tracks * float(catchall_ratio)))
    playlist_centroids = {}
    for pl_name, idxs in playlist_members.items():
        if len(idxs) < 2 or len(idxs) > max_members:
            continue
        centroid = np.mean(mat[idxs], axis=0)
        norm = float(np.linalg.norm(centroid))
        if norm > 1e-8:
            playlist_centroids[pl_name] = centroid / norm

    if not playlist_centroids:
        return mat

    for i, tid in enumerate(track_ids):
        info = metadata.get(str(tid), {})
        member_vecs = [
            playlist_centroids[str(pl).strip()]
            for pl in (info.get("playlist_list") or [])
            if str(pl).strip() in playlist_centroids
        ]
        if not member_vecs:
            continue
        pl_pull = np.mean(member_vecs, axis=0)
        p_norm = float(np.linalg.norm(pl_pull))
        if p_norm > 1e-8:
            pl_pull = pl_pull / p_norm
        blended = ((1.0 - w) * mat[i]) + (w * pl_pull)
        b_norm = float(np.linalg.norm(blended))
        if b_norm > 1e-8:
            mat[i] = blended / b_norm

    return mat


def _enrich_and_embed_clap(to_embed_indices, docs, track_ids, metadata):
    """Enrich missing audio/DSP telemetry from Deezer 30s previews and embed via CLAP."""
    import threading
    from concurrent.futures import ThreadPoolExecutor, as_completed
    import numpy as np

    stage1_dir = os.path.join(ROOT_DIR, "01_Data_Extractor")
    if stage1_dir not in sys.path:
        sys.path.insert(0, stage1_dir)

    import deezer_audio_client as dz
    from clap_embedder import CLAP_DIM, embed_audio_waveform, embed_texts_clap, get_clap_model, tag_audio_embeddings

    deezer_cache = dz.load_deezer_cache()
    dsp_cache = dz.load_dsp_cache()
    cache_lock = threading.Lock()

    processor, model, device = get_clap_model()
    print(f"\n[3/4] Fetching Deezer 30s previews & running CLAP + DSP on {len(to_embed_indices)} tracks...")

    audio_vectors = {}
    audio_indices = []
    text_fallback_indices = []

    def _fetch_track_audio(orig_idx):
        tid = track_ids[orig_idx]
        info = metadata[tid]
        name = info.get("name", "")
        artist = info.get("artists", "")
        fp = get_track_fingerprint(name, artist)

        with cache_lock:
            cached_dz = deezer_cache.get(fp)
        if cached_dz is None:
            cached_dz = dz.search_deezer_track(name, artist, cache=None)
            with cache_lock:
                deezer_cache[fp] = cached_dz

        wav = None
        if cached_dz.get("found"):
            wav, fresh_url = dz.download_preview_waveform(
                cached_dz.get("preview_url", ""),
                deezer_id=cached_dz.get("deezer_id"),
                target_sr=48000,
            )
            if fresh_url and fresh_url != cached_dz.get("preview_url"):
                with cache_lock:
                    cached_dz = dict(cached_dz, preview_url=fresh_url)
                    deezer_cache[fp] = cached_dz

        dsp_af = None
        with cache_lock:
            cached_dsp = dsp_cache.get(fp)
        if is_real_audio_features(cached_dsp):
            dsp_af = cached_dsp
        elif wav is not None:
            try:
                dsp_af = dz.extract_dsp_audio_features(wav, sr=48000, deezer_bpm=cached_dz.get("bpm"))
                with cache_lock:
                    dsp_cache[fp] = dsp_af
            except Exception:
                dsp_af = None

        return orig_idx, cached_dz, wav, dsp_af

    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(_fetch_track_audio, orig_idx) for orig_idx in to_embed_indices]
        done_cnt = 0
        total_cnt = len(to_embed_indices)
        for fut in as_completed(futures):
            done_cnt += 1
            orig_idx, dz_info, wav, dsp_af = fut.result()
            tid = track_ids[orig_idx]
            info = metadata[tid]

            if dz_info.get("found"):
                rel_date = str(dz_info.get("release_date") or "")[:4]
                if not info.get("year") and rel_date.isdigit():
                    info["year"] = rel_date

            if not is_real_audio_features(info.get("audio_features")) and is_real_audio_features(dsp_af):
                info["audio_features"] = dsp_af

            if wav is not None:
                try:
                    vec = embed_audio_waveform(wav, processor, model, device=device, max_windows=3, sr=48000)
                    audio_vectors[orig_idx] = vec
                    audio_indices.append(orig_idx)
                    info["embed_source"] = "clap_audio"
                except Exception:
                    text_fallback_indices.append(orig_idx)
            else:
                text_fallback_indices.append(orig_idx)

            if done_cnt % 50 == 0 or done_cnt == total_cnt:
                print(
                    f"      ... processed {done_cnt}/{total_cnt} tracks "
                    f"(audio: {len(audio_indices)}, text fallback: {len(text_fallback_indices)})",
                    flush=True,
                )

    dz.save_deezer_cache(deezer_cache)
    dz.save_dsp_cache(dsp_cache)

    # Zero-shot tag tracks missing genres using CLAP audio vectors
    if audio_indices:
        need_tag_idx = [i for i in audio_indices if not (metadata[track_ids[i]].get("genres") or "").strip()]
        if need_tag_idx:
            mat = np.vstack([audio_vectors[i] for i in need_tag_idx])
            tag_lists = tag_audio_embeddings(mat, processor, model, device=device, top_k=3)
            for idx_pos, orig_idx in enumerate(need_tag_idx):
                metadata[track_ids[orig_idx]]["genres"] = ", ".join(tag_lists[idx_pos])

    # Embed tracks without previews using CLAP text encoder in the same 512-D space
    text_vectors = {}
    if text_fallback_indices:
        fallback_docs = []
        for orig_idx in text_fallback_indices:
            info = metadata[track_ids[orig_idx]]
            af_text = format_audio_feature_text(info.get("audio_features") or {})
            doc = f"Track: {info['name']} | Artist: {info['artists']}"
            if info.get("year"):
                doc += f" | Year: {info['year']}"
            if info.get("genres"):
                doc += f" | Genres: {info['genres']}"
            if af_text:
                doc += f" | {af_text}"
            fallback_docs.append(doc)
        t_mat = embed_texts_clap(fallback_docs, processor, model, device=device)
        if audio_vectors and len(t_mat) > 0:
            audio_mat = np.vstack(list(audio_vectors.values()))
            delta = np.mean(audio_mat, axis=0) - np.mean(t_mat, axis=0)
            shifted = t_mat + delta
            norms = np.linalg.norm(shifted, axis=1, keepdims=True)
            norms = np.where(norms > 1e-8, norms, 1.0)
            t_mat = (shifted / norms).astype(np.float32)
            print(f"      🌉 Bridged CLAP modality gap for {len(text_fallback_indices)} text-fallback tracks.")
        for pos, orig_idx in enumerate(text_fallback_indices):
            text_vectors[orig_idx] = t_mat[pos]
            metadata[track_ids[orig_idx]]["embed_source"] = "clap_text"

    result_vecs = {}
    result_vecs.update(audio_vectors)
    result_vecs.update(text_vectors)
    return result_vecs, CLAP_DIM


def main():
    configure_utf8_stdio()
    parser = argparse.ArgumentParser(description="Music Taste Analyzer - Stage 2 Incremental Vectorizer")
    parser.add_argument("--force", action="store_true", help="Re-embed all tracks ignoring incremental hash cache")
    parser.add_argument(
        "--provider",
        choices=["spotify", "ytmusic", "unified"],
        default=None,
        help="Music provider profile to vectorize (default: active preference)",
    )
    parser.add_argument(
        "--backend",
        choices=["auto", "clap", "minilm"],
        default="auto",
        help="Embedding backend: clap (audio+text 512-D) or minilm (OpenVINO text 384-D)",
    )
    args = parser.parse_args()

    import numpy as np

    prefs = get_effective_preferences()
    backend = prefs.get("embed_backend", "minilm") if args.backend == "auto" else args.backend
    if backend not in ("clap", "minilm"):
        backend = "minilm"
    coocc_weight = float(prefs.get("playlist_cooccurrence_weight", 0.12))
    catchall_ratio = float(getattr(config, "CATCHALL_PLAYLIST_RATIO", 0.40))

    print("==================================================")
    print(f" Music Taste Analyzer - Stage 2 Vectorizer ({backend.upper()})")
    print("==================================================")

    eff_provider = args.provider or prefs.get("active_provider", "spotify")
    stage1_dir = os.path.join(ROOT_DIR, "01_Data_Extractor")
    profile = load_merged_music_profile(stage1_dir=stage1_dir, provider=eff_provider)
    if not isinstance(profile, dict):
        print(f"❌ Error: Could not load a valid Stage 1 profile for provider '{eff_provider}' in {stage1_dir}")
        print("Please ensure you run Stage 1 first!")
        sys.exit(1)

    docs, track_ids, metadata = build_semantic_documents(profile)
    print(f"      🎵 Prepared {len(docs)} unique track records.")
    if not docs:
        print("❌ Error: No tracks available to vectorize. Check Stage 1 output.")
        sys.exit(1)

    clap_id = getattr(config, "CLAP_MODEL_ID", "laion/clap-htsat-unfused") if backend == "clap" else ""
    for tid in track_ids:
        metadata[tid]["embed_backend"] = backend
        metadata[tid]["doc_hash"] = hashlib.sha256(
            f"{backend}:{clap_id}::{metadata[tid]['doc_hash']}".encode("utf-8")
        ).hexdigest()[:16]

    base_dir = os.path.dirname(os.path.abspath(__file__))
    npz_path = os.path.join(base_dir, "taste_vectors.npz")
    meta_path = os.path.join(base_dir, "track_metadata.json")

    cached_vectors = {}
    expected_dim = 512 if backend == "clap" else 384
    if not args.force and os.path.exists(npz_path) and os.path.exists(meta_path):
        try:
            with np.load(npz_path) as old_data:
                old_embs = np.array(old_data["embeddings"])
                old_tids = [str(t) for t in old_data["track_ids"]]
                old_backend = (
                    str(old_data["embed_backend"])
                    if "embed_backend" in old_data
                    else ("clap" if old_embs.shape[1] == 512 else "minilm")
                )
            old_meta = load_json_safe(meta_path, default={}) or {}
            if old_backend == backend and len(old_embs) == len(old_tids) and len(old_embs) > 0 and old_embs.shape[1] == expected_dim:
                for idx, old_tid in enumerate(old_tids):
                    old_hash = (old_meta.get(old_tid) or {}).get("doc_hash")
                    if old_hash:
                        cached_vectors[old_tid] = (old_hash, old_embs[idx], old_meta.get(old_tid) or {})
        except Exception as e:
            print(f"      ⚠️ Could not read previous vector cache ({e}); re-embedding all tracks.")
            cached_vectors = {}

    final_rows = [None] * len(track_ids)
    to_embed_indices = []
    reused_count = 0

    for idx, tid in enumerate(track_ids):
        cur_hash = metadata[tid]["doc_hash"]
        cached = cached_vectors.get(tid)
        if cached and cached[0] == cur_hash and len(cached[1]) == expected_dim:
            final_rows[idx] = cached[1]
            prev_info = cached[2]
            if not is_real_audio_features(metadata[tid].get("audio_features")) and is_real_audio_features(prev_info.get("audio_features")):
                metadata[tid]["audio_features"] = prev_info["audio_features"]
            if not (metadata[tid].get("genres") or "").strip() and (prev_info.get("genres") or "").strip():
                metadata[tid]["genres"] = prev_info["genres"]
            if prev_info.get("embed_source"):
                metadata[tid]["embed_source"] = prev_info["embed_source"]
            reused_count += 1
        else:
            to_embed_indices.append(idx)

    if reused_count > 0:
        print(f"      ⚡ Incremental Cache: Reused {reused_count} unchanged track vectors ({len(to_embed_indices)} new/updated).")

    if to_embed_indices:
        if backend == "clap":
            clap_vecs, _ = _enrich_and_embed_clap(to_embed_indices, docs, track_ids, metadata)
            for orig_idx, vec in clap_vecs.items():
                final_rows[orig_idx] = vec
        else:
            import torch

            tokenizer, model = get_openvino_model()
            print(f"\n[3/4] Vectorizing {len(to_embed_indices)} tracks via Intel OpenVINO Engine...")
            is_static = getattr(model, "is_static_npu", False)
            batch_size = 1 if is_static else 16
            embedded_chunks = []
            to_embed_docs = [docs[i] for i in to_embed_indices]

            with torch.no_grad():
                for i in range(0, len(to_embed_docs), batch_size):
                    batch = to_embed_docs[i : i + batch_size]

                    if is_static:
                        encoded_input = tokenizer(
                            batch,
                            padding="max_length",
                            max_length=128,
                            truncation=True,
                            return_tensors="pt",
                        )
                    else:
                        encoded_input = tokenizer(
                            batch,
                            padding=True,
                            max_length=256,
                            truncation=True,
                            return_tensors="pt",
                        )

                    model_output = model(**encoded_input)
                    sentence_embeddings = mean_pooling(model_output, encoded_input["attention_mask"])
                    sentence_embeddings = torch.nn.functional.normalize(sentence_embeddings, p=2, dim=1)
                    embedded_chunks.append(sentence_embeddings.detach().cpu().numpy())

                    if (i > 0 and i % 160 == 0) or (i + batch_size >= len(to_embed_docs)):
                        print(f"      ... embedded {min(i + batch_size, len(to_embed_docs))} / {len(to_embed_docs)} tracks")

            new_matrix = np.vstack(embedded_chunks)
            for pos, orig_idx in enumerate(to_embed_indices):
                final_rows[orig_idx] = new_matrix[pos]
                metadata[track_ids[orig_idx]]["embed_source"] = "minilm_text"
    else:
        print(f"\n[3/4] ⚡ All track vectors unchanged — skipping {backend.upper()} inference!")

    embeddings_matrix = np.vstack(final_rows)
    if coocc_weight > 0.0:
        print(f"      🎚️ Blending soft playlist co-occurrence (weight={coocc_weight:.2f}, catch-all>{int(catchall_ratio*100)}%)...")
        embeddings_matrix = apply_soft_playlist_cooccurrence(
            embeddings_matrix,
            track_ids,
            metadata,
            weight=coocc_weight,
            catchall_ratio=catchall_ratio,
        )

    print("\n[4/4] Generating Output Artifacts...")
    print(f"      ✅ Generated Taste Vector Matrix of shape: {embeddings_matrix.shape}")

    npz_tmp = f"{npz_path}.{os.getpid()}.tmp"
    try:
        with open(npz_tmp, "wb") as f:
            np.savez(
                f,
                embeddings=embeddings_matrix,
                track_ids=np.array(track_ids),
                embed_backend=np.array(backend),
            )
            f.flush()
            os.fsync(f.fileno())
        for attempt in range(5):
            try:
                os.replace(npz_tmp, npz_path)
                break
            except PermissionError:
                if attempt == 4:
                    raise
                time.sleep(0.15 * (attempt + 1))
    except Exception:
        if os.path.exists(npz_tmp):
            try:
                os.remove(npz_tmp)
            except OSError:
                pass
        raise

    atomic_write_json(meta_path, metadata, indent=2)

    print("      ✅ Saved 'taste_vectors.npz' and 'track_metadata.json'")
    print("\n🚀 Vectorizer Complete! You are ready for the Clustering Engine.")


if __name__ == "__main__":
    main()
