import math
import os
import sys
from datetime import datetime, timezone

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from utils import (
    get_disliked_sets,
    get_track_fingerprint,
    is_real_audio_features,
    is_valid_spotify_id,
    load_json_safe,
    matches_avoided_artist,
    matches_avoided_genre,
    normalize_avoid_list,
    parse_spotify_timestamp,
)

_ANALYTICS_CACHE = {
    "sig": None,
    "data": None,
}


def _file_sig(path):
    try:
        st = os.stat(path)
        return (True, st.st_mtime_ns, st.st_size)
    except OSError:
        return (False, 0, 0)


def _fast_year_hour(ts_str):
    """Fast-path year/hour extraction for canonical ISO timestamps with fallback parser."""
    if isinstance(ts_str, str) and len(ts_str) >= 13 and ts_str[4] == "-" and ts_str[10] in ("T", " "):
        try:
            year = int(ts_str[:4])
            hour = int(ts_str[11:13])
            if 1990 <= year <= 2100 and 0 <= hour <= 23:
                return year, hour
        except ValueError:
            pass
    dt = parse_spotify_timestamp(ts_str)
    if dt is not None:
        return dt.year, dt.hour
    return None, None


def _load_telemetry_and_genres(s1_dir, s2_dir):
    rb_cache = load_json_safe(os.path.join(s1_dir, ".reccobeats_cache.json"), default={}) or {}
    meta = load_json_safe(os.path.join(s2_dir, "track_metadata.json"), default={}) or {}
    profile = load_json_safe(os.path.join(s1_dir, "user_music_profile.json"), default={}) or {}

    af_lookup = {}
    for tid, af in rb_cache.items():
        if is_real_audio_features(af):
            af_lookup[str(tid)] = af

    genres_by_tid = {}
    for tid, info in meta.items():
        if not isinstance(info, dict):
            continue
        if info.get("genres"):
            genres_by_tid[str(tid)] = info["genres"]
        af = info.get("audio_features")
        if str(tid) not in af_lookup and is_real_audio_features(af):
            af_lookup[str(tid)] = af
        for alt in info.get("alt_ids") or []:
            genres_by_tid.setdefault(str(alt), info.get("genres", ""))
            if str(alt) not in af_lookup and is_real_audio_features(af):
                af_lookup[str(alt)] = af

    genres_by_artist = {}
    for group in (profile.get("artists") or {}).values():
        for art in group or []:
            if isinstance(art, dict) and art.get("name") and art.get("genres"):
                genres_by_artist[art["name"].strip().lower()] = ", ".join(art["genres"])

    return af_lookup, genres_by_tid, genres_by_artist


def compute_history_analytics(stage1_dir=None, stage2_dir=None, force=False):
    """Compute yearly evolution, 24h listening distribution, top artists/tracks, and skip analytics."""
    s1_dir = stage1_dir or os.path.dirname(os.path.abspath(__file__))
    s2_dir = stage2_dir or os.path.join(ROOT_DIR, "02_Vector_Core")

    stats_path = os.path.join(s1_dir, "extended_history_stats.json")
    rb_path = os.path.join(s1_dir, ".reccobeats_cache.json")
    meta_path = os.path.join(s2_dir, "track_metadata.json")

    use_global_cache = stage1_dir is None and stage2_dir is None
    sig = (_file_sig(stats_path), _file_sig(rb_path), _file_sig(meta_path))
    if use_global_cache and not force and _ANALYTICS_CACHE["sig"] == sig and _ANALYTICS_CACHE["data"] is not None:
        return _ANALYTICS_CACHE["data"]

    history = load_json_safe(stats_path)
    if not isinstance(history, dict) or not history:
        empty_res = {
            "available": False,
            "overview": {
                "unique_tracks": 0,
                "total_plays": 0,
                "total_skips": 0,
                "total_hours": 0.0,
                "earliest_stream": None,
                "latest_stream": None,
                "avg_skip_rate_pct": 0.0,
            },
            "yearly_evolution": [],
            "hourly_distribution": [{"hour": h, "plays": 0, "avg_energy": None, "avg_valence": None} for h in range(24)],
            "top_tracks": [],
            "top_artists": [],
            "most_skipped": [],
        }
        if use_global_cache:
            _ANALYTICS_CACHE["sig"] = sig
            _ANALYTICS_CACHE["data"] = empty_res
        return empty_res

    af_lookup, _, _ = _load_telemetry_and_genres(s1_dir, s2_dir)

    total_plays = 0
    total_skips = 0
    total_ms = 0
    earliest = None
    latest = None

    year_buckets = {}
    hour_buckets = [
        {"plays": 0, "eng_sum": 0.0, "eng_cnt": 0, "val_sum": 0.0, "val_cnt": 0}
        for _ in range(24)
    ]
    artist_agg = {}
    track_rows = []
    skipped_rows = []

    for key, rec in history.items():
        if not isinstance(rec, dict):
            continue
        t_name = rec.get("track_name") or "Unknown"
        a_name = rec.get("artist_name") or "Unknown"
        plays = int(rec.get("play_count", 0) or 0)
        skips = int(rec.get("skip_count", 0) or 0)
        ms = int(rec.get("total_ms_listened", 0) or 0)
        hours = round(ms / (1000.0 * 60.0 * 60.0), 2)

        total_plays += plays
        total_skips += skips
        total_ms += ms

        f_dt = rec.get("first_streamed_date")
        l_dt = rec.get("last_streamed_date")
        if f_dt and (earliest is None or f_dt < earliest):
            earliest = f_dt
        if l_dt and (latest is None or l_dt > latest):
            latest = l_dt

        tid_str = str(key)
        valid_tid = tid_str if is_valid_spotify_id(tid_str) else None
        af = af_lookup.get(tid_str) or {}
        bpm = af.get("bpm") if is_real_audio_features(af) else None
        energy = af.get("energy") if is_real_audio_features(af) else None
        valence = af.get("valence") if is_real_audio_features(af) else None

        art_entry = artist_agg.setdefault(a_name, {"artist": a_name, "plays": 0, "ms": 0, "unique_tracks": 0})
        art_entry["plays"] += plays
        art_entry["ms"] += ms
        art_entry["unique_tracks"] += 1

        track_rows.append(
            {
                "id": valid_tid or tid_str,
                "name": t_name,
                "artist": a_name,
                "play_count": plays,
                "skip_count": skips,
                "hours": hours,
                "bpm": bpm,
                "energy": energy,
                "valence": valence,
                "last_streamed_date": l_dt,
                "spotify_url": f"https://open.spotify.com/track/{valid_tid}" if valid_tid else None,
            }
        )

        if skips >= 5 and (plays + skips) > 0:
            skip_rate = round((skips / float(plays + skips)) * 100.0, 1)
            skipped_rows.append(
                {
                    "id": valid_tid or tid_str,
                    "name": t_name,
                    "artist": a_name,
                    "play_count": plays,
                    "skip_count": skips,
                    "skip_rate_pct": skip_rate,
                    "spotify_url": f"https://open.spotify.com/track/{valid_tid}" if valid_tid else None,
                }
            )

        avg_ms_per_play = (ms / plays) if plays > 0 else 0.0
        for ts in rec.get("stream_timestamps") or []:
            yr, hr = _fast_year_hour(ts)
            if yr is not None:
                yb = year_buckets.get(yr)
                if yb is None:
                    yb = {
                        "year": yr,
                        "plays": 0,
                        "ms": 0.0,
                        "tracks": set(),
                        "artists": {},
                        "bpm_sum": 0.0,
                        "bpm_cnt": 0,
                        "eng_sum": 0.0,
                        "eng_cnt": 0,
                        "val_sum": 0.0,
                        "val_cnt": 0,
                    }
                    year_buckets[yr] = yb
                yb["plays"] += 1
                yb["ms"] += avg_ms_per_play
                yb["tracks"].add(tid_str)
                yb["artists"][a_name] = yb["artists"].get(a_name, 0) + 1
                if isinstance(bpm, (int, float)):
                    yb["bpm_sum"] += float(bpm)
                    yb["bpm_cnt"] += 1
                if isinstance(energy, (int, float)):
                    yb["eng_sum"] += float(energy)
                    yb["eng_cnt"] += 1
                if isinstance(valence, (int, float)):
                    yb["val_sum"] += float(valence)
                    yb["val_cnt"] += 1

            if hr is not None:
                hb = hour_buckets[hr]
                hb["plays"] += 1
                if isinstance(energy, (int, float)):
                    hb["eng_sum"] += float(energy)
                    hb["eng_cnt"] += 1
                if isinstance(valence, (int, float)):
                    hb["val_sum"] += float(valence)
                    hb["val_cnt"] += 1

    yearly_evolution = []
    for yr in sorted(year_buckets):
        yb = year_buckets[yr]
        top_art = max(yb["artists"].items(), key=lambda x: x[1])[0] if yb["artists"] else None
        yearly_evolution.append(
            {
                "year": yr,
                "plays": yb["plays"],
                "hours": round(yb["ms"] / (1000.0 * 60.0 * 60.0), 1),
                "unique_tracks": len(yb["tracks"]),
                "top_artist": top_art,
                "avg_bpm": round(yb["bpm_sum"] / yb["bpm_cnt"], 1) if yb["bpm_cnt"] else None,
                "avg_energy": round(yb["eng_sum"] / yb["eng_cnt"], 3) if yb["eng_cnt"] else None,
                "avg_valence": round(yb["val_sum"] / yb["val_cnt"], 3) if yb["val_cnt"] else None,
            }
        )

    hourly_distribution = []
    for hr, hb in enumerate(hour_buckets):
        hourly_distribution.append(
            {
                "hour": hr,
                "plays": hb["plays"],
                "avg_energy": round(hb["eng_sum"] / hb["eng_cnt"], 3) if hb["eng_cnt"] else None,
                "avg_valence": round(hb["val_sum"] / hb["val_cnt"], 3) if hb["val_cnt"] else None,
            }
        )

    track_rows.sort(key=lambda r: (r["play_count"], r["hours"]), reverse=True)
    top_artists = sorted(
        (
            {
                "artist": a["artist"],
                "plays": a["plays"],
                "hours": round(a["ms"] / (1000.0 * 60.0 * 60.0), 1),
                "unique_tracks": a["unique_tracks"],
            }
            for a in artist_agg.values()
        ),
        key=lambda x: (x["hours"], x["plays"]),
        reverse=True,
    )[:15]
    skipped_rows.sort(key=lambda r: (r["skip_rate_pct"], r["skip_count"]), reverse=True)

    total_events = total_plays + total_skips
    result = {
        "available": True,
        "overview": {
            "unique_tracks": len(history),
            "total_plays": total_plays,
            "total_skips": total_skips,
            "total_hours": round(total_ms / (1000.0 * 60.0 * 60.0), 1),
            "earliest_stream": earliest,
            "latest_stream": latest,
            "avg_skip_rate_pct": round((total_skips / float(total_events)) * 100.0, 1) if total_events > 0 else 0.0,
        },
        "yearly_evolution": yearly_evolution,
        "hourly_distribution": hourly_distribution,
        "top_tracks": track_rows[:20],
        "top_artists": top_artists,
        "most_skipped": skipped_rows[:15],
    }

    if use_global_cache:
        _ANALYTICS_CACHE["sig"] = sig
        _ANALYTICS_CACHE["data"] = result
    return result


def find_forgotten_favorites(
    min_plays=5,
    dormant_days=180,
    limit=30,
    avoid_artists=None,
    avoid_genres=None,
    stage1_dir=None,
    stage2_dir=None,
    now=None,
):
    """Find historically loved tracks with 0 plays in the recent dormant_days window."""
    s1_dir = stage1_dir or os.path.dirname(os.path.abspath(__file__))
    s2_dir = stage2_dir or os.path.join(ROOT_DIR, "02_Vector_Core")
    history = load_json_safe(os.path.join(s1_dir, "extended_history_stats.json"))
    if not isinstance(history, dict) or not history:
        return []

    af_lookup, genres_by_tid, genres_by_artist = _load_telemetry_and_genres(s1_dir, s2_dir)
    disliked_ids, disliked_fps = get_disliked_sets()
    avoid_art_list = normalize_avoid_list(avoid_artists)
    avoid_gen_list = normalize_avoid_list(avoid_genres)
    now_dt = now or datetime.now(timezone.utc)
    min_p = max(1, int(min_plays))
    dormant_sec = max(1, int(dormant_days)) * 86400.0
    cap = max(1, min(100, int(limit)))

    candidates = []
    for key, rec in history.items():
        if not isinstance(rec, dict):
            continue
        plays = int(rec.get("play_count", 0) or 0)
        if plays < min_p:
            continue

        last_str = rec.get("last_streamed_date")
        last_dt = parse_spotify_timestamp(last_str)
        if last_dt is None:
            continue
        if (now_dt - last_dt).total_seconds() < dormant_sec:
            continue

        t_name = rec.get("track_name") or "Unknown"
        a_name = rec.get("artist_name") or "Unknown"
        tid_str = str(key)
        if tid_str in disliked_ids or get_track_fingerprint(t_name, a_name) in disliked_fps:
            continue
        if matches_avoided_artist(a_name, avoid_art_list):
            continue
        genres_str = genres_by_tid.get(tid_str) or genres_by_artist.get(a_name.strip().lower(), "")
        if avoid_gen_list and genres_str and matches_avoided_genre(genres_str, avoid_gen_list):
            continue

        ms = int(rec.get("total_ms_listened", 0) or 0)
        hours = ms / (1000.0 * 60.0 * 60.0)
        score = hours * math.log1p(plays)
        valid_tid = tid_str if is_valid_spotify_id(tid_str) else None
        af = af_lookup.get(tid_str) or {}
        real_af = af if is_real_audio_features(af) else {}

        candidates.append(
            {
                "_score": score,
                "id": valid_tid or tid_str,
                "has_spotify_id": valid_tid is not None,
                "name": t_name,
                "artist": a_name,
                "source": "Forgotten Favorite",
                "play_count": plays,
                "hours": round(hours, 2),
                "last_played": last_str,
                "bpm": real_af.get("bpm"),
                "energy": real_af.get("energy"),
                "valence": real_af.get("valence"),
                "key": real_af.get("key"),
                "mode": real_af.get("mode"),
                "spotify_url": f"https://open.spotify.com/track/{valid_tid}" if valid_tid else None,
            }
        )

    candidates.sort(key=lambda item: item["_score"], reverse=True)
    out = []
    for item in candidates[:cap]:
        item.pop("_score", None)
        out.append(item)
    return out
