import json
import os
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone

import requests

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import config
from utils import (
    LEGACY_FALLBACK_AF,
    atomic_write_json,
    configure_utf8_stdio,
    get_track_fingerprint,
    is_real_audio_features,
    is_valid_spotify_id,
    load_json_safe,
    parse_spotify_timestamp,
)

CACHE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".reccobeats_cache.json")
RECCOBEATS_ENDPOINT = "https://api.reccobeats.com/v1/audio-features"
RECCOBEATS_TRACK_ENDPOINT = "https://api.reccobeats.com/v1/track"
RECCOBEATS_RECOMMEND_ENDPOINT = "https://api.reccobeats.com/v1/track/recommendation"
RECCOBEATS_SEARCH_ENDPOINT = "https://api.reccobeats.com/v1/track/search"
_HREF_TRACK_ID_RE = re.compile(r"/track/([A-Za-z0-9]{22})")
_HREF_ARTIST_ID_RE = re.compile(r"/artist/([A-Za-z0-9]{22})")


def is_poisoned_legacy_entry(entry):
    """Detect legacy fake-default cache entries written when ReccoBeats requests failed."""
    if not isinstance(entry, dict):
        return True
    source = entry.get("source")
    if source in ("fallback", "estimated"):
        return True
    if source == "not_found":
        return False
    if not source:
        try:
            if all(k in entry for k in LEGACY_FALLBACK_AF) and all(
                abs(float(entry[k]) - val) < 1e-6 for k, val in LEGACY_FALLBACK_AF.items()
            ):
                return True
        except (TypeError, ValueError):
            return True
    return not is_real_audio_features(entry)


def is_not_found_expired(entry, ttl_days=None, now=None):
    """Return True if a cached 'not_found' record has expired and should be retried."""
    if not isinstance(entry, dict) or entry.get("source") != "not_found":
        return False
    if ttl_days is None:
        ttl_days = getattr(config, "RECCOBEATS_NOT_FOUND_TTL_DAYS", 14)
    fetched_dt = parse_spotify_timestamp(entry.get("fetched_at"))
    if fetched_dt is None:
        return True
    now_dt = now or datetime.now(timezone.utc)
    return (now_dt - fetched_dt).total_seconds() >= (float(ttl_days) * 86400.0)


def extract_spotify_id_from_item(item, requested_set):
    """Extract the Spotify track ID from ReccoBeats item['href'] or fallback to item['id'] if requested."""
    if not isinstance(item, dict):
        return None
    href = item.get("href")
    if isinstance(href, str):
        match = _HREF_TRACK_ID_RE.search(href)
        if match:
            return match.group(1)
    raw_id = item.get("id")
    if raw_id is not None:
        raw_id_str = str(raw_id)
        if raw_id_str in requested_set and is_valid_spotify_id(raw_id_str):
            return raw_id_str
    return None


def _safe_round(val, digits):
    if val is None:
        return None
    try:
        return round(float(val), digits)
    except (TypeError, ValueError):
        return None


def _safe_int(val):
    if val is None or isinstance(val, bool):
        return None
    try:
        return int(val)
    except (TypeError, ValueError):
        return None


def _normalize_reccobeats_item(item, fetched_at_iso):
    tempo_val = item.get("tempo", item.get("bpm"))
    record = {
        "bpm": _safe_round(tempo_val, 1),
        "energy": _safe_round(item.get("energy"), 3),
        "danceability": _safe_round(item.get("danceability"), 3),
        "valence": _safe_round(item.get("valence"), 3),
        "acousticness": _safe_round(item.get("acousticness"), 3),
        "instrumentalness": _safe_round(item.get("instrumentalness"), 4),
        "speechiness": _safe_round(item.get("speechiness"), 4),
        "liveness": _safe_round(item.get("liveness"), 3),
        "loudness": _safe_round(item.get("loudness"), 2),
        "key": _safe_int(item.get("key")),
        "mode": _safe_int(item.get("mode")),
        "isrc": item.get("isrc"),
        "source": "reccobeats",
        "fetched_at": fetched_at_iso,
    }
    return {k: v for k, v in record.items() if v is not None}


def load_reccobeats_cache(cache_path=None):
    """Load cache from disk and automatically purge poisoned legacy default entries."""
    target = cache_path or CACHE_FILE
    raw_cache = load_json_safe(target, default={})
    if not isinstance(raw_cache, dict):
        return {}

    cleaned = {}
    purged_count = 0
    for tid, entry in raw_cache.items():
        if is_poisoned_legacy_entry(entry):
            purged_count += 1
        else:
            cleaned[str(tid)] = entry

    if purged_count > 0:
        print(f"      🧹 Purged {purged_count} poisoned placeholder entries from ReccoBeats cache.")
        save_reccobeats_cache(cleaned, cache_path=target)

    return cleaned


def save_reccobeats_cache(cache, cache_path=None):
    """Atomically persist the ReccoBeats cache to disk."""
    target = cache_path or CACHE_FILE
    try:
        atomic_write_json(target, cache, indent=2)
    except Exception as e:
        print(f"      [!] Cache save error: {e}")


def _extract_response_items(payload):
    if isinstance(payload, dict):
        for key in ("content", "audio_features", "data"):
            val = payload.get(key)
            if isinstance(val, list):
                return val
        return []
    if isinstance(payload, list):
        return payload
    return []


def _get_json_with_retry(http_session, url, params, max_retries=3, timeout=15):
    """Execute a ReccoBeats GET request with 429/5xx backoff and JSON parsing safety."""
    headers = {"Accept": "application/json"}

    for attempt in range(max_retries + 1):
        try:
            resp = http_session.get(url, params=params, headers=headers, timeout=timeout)
            if resp.status_code == 200:
                return resp.json(), True
            if resp.status_code == 429:
                if attempt < max_retries:
                    retry_after = resp.headers.get("Retry-After")
                    try:
                        delay = min(15.0, max(0.5, float(retry_after)))
                    except (TypeError, ValueError):
                        delay = 1.0 * (2 ** attempt)
                    time.sleep(delay)
                    continue
                return None, False
            if 500 <= resp.status_code < 600:
                if attempt < max_retries:
                    time.sleep(0.5 * (2 ** attempt))
                    continue
                return None, False
            # Non-retryable 4xx (e.g., 400 validation error)
            print(f"      [!] ReccoBeats HTTP {resp.status_code}: {resp.text[:160]}")
            return None, False
        except (requests.RequestException, ValueError) as exc:
            if attempt < max_retries:
                time.sleep(0.5 * (2 ** attempt))
                continue
            print(f"      [!] ReccoBeats network/JSON error: {exc}")
            return None, False
    return None, False


def _fetch_batch_with_retry(http_session, batch, max_retries=3, timeout=15):
    payload, ok = _get_json_with_retry(
        http_session,
        RECCOBEATS_ENDPOINT,
        {"ids": ",".join(batch)},
        max_retries=max_retries,
        timeout=timeout,
    )
    if not ok:
        return [], False
    return _extract_response_items(payload), True


def get_cache_health(cache_path=None):
    """Inspect ReccoBeats cache health without modifying files."""
    target = cache_path or CACHE_FILE
    raw_cache = load_json_safe(target, default={})
    if not isinstance(raw_cache, dict):
        raw_cache = {}
    real = 0
    not_found = 0
    poisoned = 0
    for entry in raw_cache.values():
        if isinstance(entry, dict) and entry.get("source") == "not_found":
            not_found += 1
        elif is_poisoned_legacy_entry(entry):
            poisoned += 1
        else:
            real += 1
    size_kb = round(os.path.getsize(target) / 1024.0, 1) if os.path.exists(target) else 0.0
    return {
        "total": len(raw_cache),
        "real": real,
        "not_found": not_found,
        "poisoned": poisoned,
        "file_size_kb": size_kb,
    }


def purge_poisoned_and_not_found(cache_path=None, include_not_found=False):
    """Purge poisoned entries (and optionally 'not_found' records) from the ReccoBeats cache."""
    target = cache_path or CACHE_FILE
    raw_cache = load_json_safe(target, default={})
    if not isinstance(raw_cache, dict):
        return 0, get_cache_health(cache_path=target)

    cleaned = {}
    removed = 0
    for tid, entry in raw_cache.items():
        if is_poisoned_legacy_entry(entry):
            removed += 1
        elif include_not_found and isinstance(entry, dict) and entry.get("source") == "not_found":
            removed += 1
        else:
            cleaned[str(tid)] = entry

    if removed > 0:
        save_reccobeats_cache(cleaned, cache_path=target)
    return removed, get_cache_health(cache_path=target)


def fetch_audio_features_batch(track_ids, cache_path=None, session=None, progress_cb=None):
    """
    Fetch real audio features (BPM, Energy, Danceability, Valence, Acousticness, etc.)
    for a list of Spotify Track IDs using ReccoBeats API with validated local caching.
    """
    target_cache = cache_path or CACHE_FILE
    cache = load_reccobeats_cache(cache_path=target_cache)

    deduped_ids = []
    seen_ids = set()
    for tid in track_ids or []:
        tid_str = str(tid).strip() if tid is not None else ""
        if tid_str and tid_str not in seen_ids and is_valid_spotify_id(tid_str):
            seen_ids.add(tid_str)
            deduped_ids.append(tid_str)

    if not deduped_ids:
        return {}

    ttl_days = getattr(config, "RECCOBEATS_NOT_FOUND_TTL_DAYS", 14)
    now_dt = datetime.now(timezone.utc)

    missing_ids = []
    for tid in deduped_ids:
        entry = cache.get(tid)
        if entry is None:
            missing_ids.append(tid)
        elif entry.get("source") == "not_found" and is_not_found_expired(entry, ttl_days=ttl_days, now=now_dt):
            missing_ids.append(tid)

    if missing_ids:
        print(f"      🎵 Querying ReccoBeats API for {len(missing_ids)} track audio features...")
        batch_size = min(40, max(1, int(getattr(config, "RECCOBEATS_BATCH_SIZE", 40))))
        total_batches = (len(missing_ids) + batch_size - 1) // batch_size
        own_session = session is None
        http_session = session or requests.Session()
        dirty = False
        batches_since_save = 0
        total_matched = 0
        start_time = time.monotonic()

        try:
            for batch_idx, i in enumerate(range(0, len(missing_ids), batch_size), start=1):
                batch = missing_ids[i : i + batch_size]
                batch_set = set(batch)
                items, ok = _fetch_batch_with_retry(http_session, batch)

                if ok:
                    fetched_at_iso = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
                    matched_ids = set()
                    for item in items:
                        sid = extract_spotify_id_from_item(item, batch_set)
                        if not sid:
                            continue
                        norm_record = _normalize_reccobeats_item(item, fetched_at_iso)
                        if is_real_audio_features(norm_record):
                            cache[sid] = norm_record
                            matched_ids.add(sid)
                            dirty = True

                    total_matched += len(matched_ids)
                    for tid in batch:
                        if tid not in matched_ids:
                            cache[tid] = {"source": "not_found", "fetched_at": fetched_at_iso}
                            dirty = True

                    batches_since_save += 1
                    if dirty and batches_since_save >= 10:
                        save_reccobeats_cache(cache, cache_path=target_cache)
                        dirty = False
                        batches_since_save = 0

                processed = min(i + batch_size, len(missing_ids))
                elapsed = max(0.01, time.monotonic() - start_time)
                eta_sec = max(0, int((elapsed / batch_idx) * (total_batches - batch_idx)))
                if total_batches <= 10 or batch_idx == 1 or batch_idx % 2 == 0 or batch_idx == total_batches:
                    print(
                        f"      ... Batch {batch_idx}/{total_batches} "
                        f"({processed}/{len(missing_ids)} tracks) | {total_matched} matched | ETA ~{eta_sec}s"
                    )
                if progress_cb is not None:
                    try:
                        progress_cb(batch_idx, total_batches, processed, len(missing_ids), total_matched, eta_sec)
                    except Exception:
                        pass

                if i + batch_size < len(missing_ids):
                    time.sleep(0.12)
        finally:
            if dirty:
                save_reccobeats_cache(cache, cache_path=target_cache)
            if own_session:
                http_session.close()

    return {
        tid: cache[tid]
        for tid in deduped_ids
        if tid in cache and is_real_audio_features(cache[tid])
    }


def _dedupe_valid_spotify_ids(track_ids):
    out = []
    seen = set()
    for tid in track_ids or []:
        tid_str = str(tid).strip() if tid is not None else ""
        if tid_str and tid_str not in seen and is_valid_spotify_id(tid_str):
            seen.add(tid_str)
            out.append(tid_str)
    return out


def fetch_track_metadata_batch(track_ids, session=None):
    """
    Backfill Spotify track metadata (real 0-100 popularity, ISRC, durationMs) via
    ReccoBeats GET /v1/track?ids=... (up to 40 IDs per request).
    """
    deduped = _dedupe_valid_spotify_ids(track_ids)
    if not deduped:
        return {}

    batch_size = min(40, max(1, int(getattr(config, "RECCOBEATS_BATCH_SIZE", 40))))
    own_session = session is None
    http_session = session or requests.Session()
    meta_out = {}

    try:
        for i in range(0, len(deduped), batch_size):
            batch = deduped[i : i + batch_size]
            batch_set = set(batch)
            payload, ok = _get_json_with_retry(
                http_session, RECCOBEATS_TRACK_ENDPOINT, {"ids": ",".join(batch)}
            )
            if ok:
                for item in _extract_response_items(payload):
                    sid = extract_spotify_id_from_item(item, batch_set)
                    if not sid:
                        continue
                    meta_out[sid] = {
                        "popularity": _safe_int(item.get("popularity")),
                        "isrc": item.get("isrc"),
                        "duration_ms": _safe_int(item.get("durationMs", item.get("duration_ms"))),
                        "title": str(item.get("trackTitle") or item.get("name") or ""),
                        "artists": item.get("artists") or [],
                    }
            if i + batch_size < len(deduped):
                time.sleep(0.1)
    finally:
        if own_session:
            http_session.close()

    return meta_out


def _normalize_reccobeats_track(item):
    if not isinstance(item, dict):
        return None
    href = item.get("href")
    if not isinstance(href, str):
        return None
    m = _HREF_TRACK_ID_RE.search(href)
    if not m:
        return None
    tid = m.group(1)
    name = str(item.get("trackTitle") or item.get("name") or "").strip()
    if not name:
        return None

    norm_artists = []
    for art in item.get("artists") or []:
        if not isinstance(art, dict) or not art.get("name"):
            continue
        aid = None
        art_href = art.get("href")
        if isinstance(art_href, str):
            am = _HREF_ARTIST_ID_RE.search(art_href)
            if am:
                aid = am.group(1)
        norm_artists.append({"id": aid, "name": str(art["name"]).strip()})

    return {
        "id": tid,
        "name": name,
        "artists": norm_artists or [{"id": None, "name": "Unknown"}],
        "popularity": _safe_int(item.get("popularity")),
        "duration_ms": _safe_int(item.get("durationMs", item.get("duration_ms"))),
        "isrc": item.get("isrc"),
        "explicit": bool(item.get("explicit", False)),
        "album": {},
        "type": "track",
        "is_local": False,
        "_from_reccobeats_rec": True,
    }


def fetch_reccobeats_recommendations(
    seed_ids,
    negative_seed_ids=None,
    size=25,
    popularity=None,
    target_bpm=None,
    target_energy=None,
    target_valence=None,
    target_danceability=None,
    target_acousticness=None,
    feature_weight=None,
    session=None,
):
    """
    Fetch Spotify-ID recommendations from ReccoBeats GET /v1/track/recommendation,
    supporting positive seeds, negativeSeeds, popularity, and acoustic target constraints.
    """
    clean_seeds = _dedupe_valid_spotify_ids(seed_ids)[:5]
    if not clean_seeds:
        return []
    clean_neg = _dedupe_valid_spotify_ids(negative_seed_ids)[:5]

    params = {
        "seeds": ",".join(clean_seeds),
        "size": max(1, min(100, int(size or 25))),
    }
    if clean_neg:
        params["negativeSeeds"] = ",".join(clean_neg)
    if popularity is not None and not isinstance(popularity, bool):
        params["popularity"] = max(0, min(100, int(popularity)))
    if target_bpm is not None and not isinstance(target_bpm, bool):
        params["tempo"] = round(float(target_bpm), 1)
    if target_energy is not None and not isinstance(target_energy, bool):
        params["energy"] = round(max(0.0, min(1.0, float(target_energy))), 3)
    if target_valence is not None and not isinstance(target_valence, bool):
        params["valence"] = round(max(0.0, min(1.0, float(target_valence))), 3)
    if target_danceability is not None and not isinstance(target_danceability, bool):
        params["danceability"] = round(max(0.0, min(1.0, float(target_danceability))), 3)
    if target_acousticness is not None and not isinstance(target_acousticness, bool):
        params["acousticness"] = round(max(0.0, min(1.0, float(target_acousticness))), 3)
    if feature_weight is not None and not isinstance(feature_weight, bool):
        params["featureWeight"] = round(float(feature_weight), 3)

    own_session = session is None
    http_session = session or requests.Session()
    try:
        payload, ok = _get_json_with_retry(http_session, RECCOBEATS_RECOMMEND_ENDPOINT, params)
    finally:
        if own_session:
            http_session.close()

    if not ok:
        return []

    out = []
    for item in _extract_response_items(payload):
        norm = _normalize_reccobeats_track(item)
        if norm:
            out.append(norm)
    return out


def _pick_reccobeats_search_match(items, track_name, artist_name):
    target_fp = get_track_fingerprint(track_name, artist_name)
    target_title_fp = get_track_fingerprint(track_name, "")
    target_title_part = target_title_fp.split("::")[0]
    primary_artist = str(artist_name or "").split(",")[0].strip().lower()

    for item in items or []:
        if not isinstance(item, dict) or not item.get("id"):
            continue
        cand_title = str(item.get("trackTitle") or item.get("name") or "").strip()
        cand_artists = [
            str(a.get("name") or "").strip()
            for a in (item.get("artists") or [])
            if isinstance(a, dict) and a.get("name")
        ]
        if not cand_title or not cand_artists:
            continue
        cand_fp = get_track_fingerprint(cand_title, cand_artists)
        cand_title_part = get_track_fingerprint(cand_title, "").split("::")[0]
        cand_art_joined = ", ".join(cand_artists).lower()

        exact_fp = cand_fp == target_fp
        strong_fuzzy = (
            target_title_part
            and cand_title_part
            and (target_title_part == cand_title_part)
            and primary_artist
            and primary_artist != "unknown"
            and primary_artist in cand_art_joined
        )
        if exact_fp or strong_fuzzy:
            sp_id = None
            href = item.get("href")
            if isinstance(href, str):
                m = _HREF_TRACK_ID_RE.search(href)
                if m:
                    sp_id = m.group(1)
            return str(item["id"]), sp_id
    return None, None


def fetch_audio_features_by_search(tracks, cache_path=None, session=None, max_lookups=150, workers=4):
    """
    Search ReccoBeats (/v1/track/search?searchText=...) by title + artist and batch-fetch
    audio features for non-Spotify tracks (e.g., YouTube Music yt:<videoId> items).
    """
    target_cache = cache_path or CACHE_FILE
    cache = load_reccobeats_cache(cache_path=target_cache)

    deduped = []
    seen_tids = set()
    for tr in tracks or []:
        if not isinstance(tr, dict):
            continue
        tid = str(tr.get("id") or "").strip()
        name = str(tr.get("name") or "").strip()
        raw_artist = tr.get("artist")
        if not raw_artist and isinstance(tr.get("artists"), list):
            raw_artist = ", ".join(
                a.get("name", "") if isinstance(a, dict) else str(a)
                for a in tr["artists"]
                if (a.get("name") if isinstance(a, dict) else a)
            )
        artist = str(raw_artist or "").strip()
        if not tid or not name or tid in seen_tids:
            continue
        seen_tids.add(tid)
        deduped.append({"id": tid, "name": name, "artist": artist})

    if not deduped:
        return {}

    ttl_days = getattr(config, "RECCOBEATS_NOT_FOUND_TTL_DAYS", 14)
    now_dt = datetime.now(timezone.utc)

    to_lookup = []
    for tr in deduped:
        entry = cache.get(tr["id"])
        if entry is None:
            to_lookup.append(tr)
        elif entry.get("source") == "not_found" and is_not_found_expired(entry, ttl_days=ttl_days, now=now_dt):
            to_lookup.append(tr)

    if to_lookup and max_lookups > 0:
        cap_list = to_lookup[: int(max_lookups)]
        total_cap = len(cap_list)
        print(f"      🎵 Searching ReccoBeats for {total_cap} YouTube Music track features...", flush=True)
        own_session = session is None
        dirty = False
        rb_to_tracks = {}
        matched_tids = set()
        fetched_at_iso = now_dt.isoformat().replace("+00:00", "Z")
        start_time = time.monotonic()
        completed = 0
        consec_failures = 0

        def _search_single_track(tr, http_sess):
            primary_art = tr["artist"].split(",")[0].strip() if tr["artist"] else ""
            query = f"{tr['name']} {primary_art}".strip()
            payload, ok = _get_json_with_retry(
                http_sess,
                RECCOBEATS_SEARCH_ENDPOINT,
                {"searchText": query},
                max_retries=1,
                timeout=8,
            )
            if not ok:
                return tr, False, None, None
            items = _extract_response_items(payload)
            rb_id, sp_id = _pick_reccobeats_search_match(items, tr["name"], tr["artist"])
            return tr, True, rb_id, sp_id

        def _record_result(tr, ok, rb_id, sp_id):
            nonlocal completed, consec_failures, dirty
            completed += 1
            if not ok:
                consec_failures += 1
            else:
                consec_failures = 0
                if rb_id:
                    rb_to_tracks.setdefault(rb_id, []).append((tr["id"], sp_id))
                    matched_tids.add(tr["id"])
                else:
                    cache[tr["id"]] = {"source": "not_found", "fetched_at": fetched_at_iso}
                    dirty = True

            if completed % 15 == 0 or completed == total_cap:
                elapsed = max(0.01, time.monotonic() - start_time)
                rate = completed / elapsed
                eta = int(round((total_cap - completed) / rate)) if rate > 0 else 0
                print(
                    f"      ... Searched {completed}/{total_cap} tracks | {len(matched_tids)} matched | ETA ~{eta}s",
                    flush=True,
                )
            if completed % 30 == 0 and dirty:
                save_reccobeats_cache(cache, cache_path=target_cache)

        batch_session = session if session is not None else requests.Session()
        created_sessions = []
        stop_event = threading.Event()
        try:
            if session is not None:
                for tr in cap_list:
                    r_tr, ok, rb_id, sp_id = _search_single_track(tr, session)
                    _record_result(r_tr, ok, rb_id, sp_id)
                    if consec_failures >= 8:
                        print(
                            "      ⚠️ ReccoBeats search is rate-limited/slow; stopping early and using matches collected so far.",
                            flush=True,
                        )
                        break
            else:
                thread_local = threading.local()
                sessions_lock = threading.Lock()

                def _worker_lookup(tr):
                    if stop_event.is_set():
                        return tr, False, None, None
                    time.sleep(0.08)
                    if stop_event.is_set():
                        return tr, False, None, None
                    sess = getattr(thread_local, "sess", None)
                    if sess is None:
                        sess = requests.Session()
                        thread_local.sess = sess
                        with sessions_lock:
                            created_sessions.append(sess)
                    return _search_single_track(tr, sess)

                max_w = max(1, min(6, int(workers or 4)))
                with ThreadPoolExecutor(max_workers=max_w) as executor:
                    fut_map = {executor.submit(_worker_lookup, tr): tr for tr in cap_list}
                    for fut in as_completed(fut_map):
                        orig_tr = fut_map[fut]
                        try:
                            r_tr, ok, rb_id, sp_id = fut.result()
                        except Exception:
                            r_tr, ok, rb_id, sp_id = orig_tr, False, None, None
                        _record_result(r_tr, ok, rb_id, sp_id)
                        if consec_failures >= 8:
                            stop_event.set()
                            print(
                                "      ⚠️ ReccoBeats search is rate-limited/slow; stopping early and using matches collected so far.",
                                flush=True,
                            )
                            for pending in fut_map:
                                pending.cancel()
                            break

            rb_ids = list(rb_to_tracks.keys())
            batch_size = min(40, max(1, int(getattr(config, "RECCOBEATS_BATCH_SIZE", 40))))
            resolved_count = 0
            for i in range(0, len(rb_ids), batch_size):
                batch_rb = rb_ids[i : i + batch_size]
                items, ok = _fetch_batch_with_retry(batch_session, batch_rb)
                if ok:
                    resolved_in_batch = set()
                    for item in items:
                        item_rb_id = str(item.get("id") or "")
                        targets = rb_to_tracks.get(item_rb_id)
                        if not targets:
                            # Fallback match via href Spotify ID
                            item_sp = extract_spotify_id_from_item(item, set())
                            if item_sp:
                                for cand_rb, pairs in rb_to_tracks.items():
                                    if any(sp == item_sp for _, sp in pairs):
                                        targets = pairs
                                        break
                        if not targets:
                            continue
                        norm_rec = _normalize_reccobeats_item(item, fetched_at_iso)
                        if is_real_audio_features(norm_rec):
                            for tid_dest, sp_dest in targets:
                                rec_copy = dict(norm_rec)
                                if sp_dest:
                                    rec_copy["matched_spotify_id"] = sp_dest
                                cache[tid_dest] = rec_copy
                                resolved_in_batch.add(tid_dest)
                                dirty = True
                                resolved_count += 1
                    for rb_id in batch_rb:
                        for tid_dest, _ in rb_to_tracks.get(rb_id, []):
                            if tid_dest not in resolved_in_batch:
                                cache[tid_dest] = {"source": "not_found", "fetched_at": fetched_at_iso}
                                dirty = True
            print(f"      -> Matched ReccoBeats audio telemetry for {resolved_count} / {total_cap} YTM tracks.", flush=True)
        finally:
            if dirty:
                save_reccobeats_cache(cache, cache_path=target_cache)
            if own_session:
                batch_session.close()
            for ws in created_sessions:
                try:
                    ws.close()
                except Exception:
                    pass

    return {
        tr["id"]: cache[tr["id"]]
        for tr in deduped
        if tr["id"] in cache and is_real_audio_features(cache[tr["id"]])
    }


if __name__ == "__main__":
    configure_utf8_stdio()
    test_ids = ["4cOdK2wGLETKBW3PvgPWqT", "0VjIjW4GlUZAMYd2vXMi3b"]
    features = fetch_audio_features_batch(test_ids)
    print("Test ReccoBeats Response:", json.dumps(features, indent=2))
