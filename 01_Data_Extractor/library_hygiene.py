import os
import sys

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from utils import (
    fetch_playlist_items_2026,
    get_disliked_sets,
    get_track_fingerprint,
    is_real_audio_features,
    load_json_safe,
    replace_spotify_playlist_items,
)


def analyze_library_hygiene(stage1_dir=None, stage2_dir=None):
    """
    Scan personal Spotify playlists for duplicate fingerprints (e.g. Single vs Album/Remaster),
    chronically skipped tracks (>=70% skip ratio), and user-disliked tracks still in playlists.
    """
    s1_dir = stage1_dir or os.path.dirname(os.path.abspath(__file__))
    profile = load_json_safe(os.path.join(s1_dir, "user_music_profile.json"))
    if not isinstance(profile, dict):
        return {
            "available": False,
            "playlist_duplicates": [],
            "chronically_skipped": [],
            "disliked_in_playlists": [],
            "summary": {
                "playlists_scanned": 0,
                "duplicate_groups": 0,
                "duplicate_extra_tracks": 0,
                "chronically_skipped_count": 0,
                "disliked_in_playlists_count": 0,
                "telemetry_coverage_pct": 0.0,
            },
        }

    history = load_json_safe(os.path.join(s1_dir, "extended_history_stats.json"), default={}) or {}
    rb_cache = load_json_safe(os.path.join(s1_dir, ".reccobeats_cache.json"), default={}) or {}
    profile_af = profile.get("audio_features") or {}
    disliked_ids, disliked_fps = get_disliked_sets()

    history_by_fp = {}
    for key, rec in history.items():
        if isinstance(rec, dict):
            fp = rec.get("fingerprint") or get_track_fingerprint(rec.get("track_name", ""), rec.get("artist_name", ""))
            if fp:
                history_by_fp[fp] = rec

    playlist_duplicates = []
    chronically_skipped = []
    disliked_in_playlists = []
    unique_tids = set()
    telemetry_tids = set()
    seen_skipped = set()
    seen_disliked = set()
    total_extra_dups = 0

    playlists = profile.get("playlists") or []
    for pl in playlists:
        if not isinstance(pl, dict):
            continue
        pl_id = pl.get("id")
        pl_name = pl.get("name") or "Untitled Playlist"
        tracks = pl.get("tracks") or []

        fp_groups = {}
        for pos, t in enumerate(tracks, start=1):
            if not isinstance(t, dict) or not t.get("id"):
                continue
            tid = str(t["id"])
            t_name = t.get("name") or "Unknown"
            artists_list = [a.get("name") for a in (t.get("artists") or []) if isinstance(a, dict) and a.get("name")]
            artists_str = ", ".join(artists_list) if artists_list else "Unknown"
            fp = get_track_fingerprint(t_name, artists_str)

            unique_tids.add(tid)
            if is_real_audio_features(profile_af.get(tid)) or is_real_audio_features(rb_cache.get(tid)):
                telemetry_tids.add(tid)

            fp_groups.setdefault(fp, []).append(
                {
                    "id": tid,
                    "name": t_name,
                    "artist": artists_str,
                    "position": pos,
                }
            )

            h_rec = history.get(tid) or history_by_fp.get(fp)
            if isinstance(h_rec, dict):
                plays = int(h_rec.get("play_count", 0) or 0)
                skips = int(h_rec.get("skip_count", 0) or 0)
                total = plays + skips
                if skips >= 5 and total > 0:
                    ratio = skips / float(total)
                    if ratio >= 0.70 and (pl_name, tid) not in seen_skipped:
                        seen_skipped.add((pl_name, tid))
                        chronically_skipped.append(
                            {
                                "playlist_id": pl_id,
                                "playlist_name": pl_name,
                                "id": tid,
                                "name": t_name,
                                "artist": artists_str,
                                "play_count": plays,
                                "skip_count": skips,
                                "skip_rate_pct": round(ratio * 100.0, 1),
                            }
                        )

            if (tid in disliked_ids or fp in disliked_fps) and (pl_name, tid) not in seen_disliked:
                seen_disliked.add((pl_name, tid))
                disliked_in_playlists.append(
                    {
                        "playlist_id": pl_id,
                        "playlist_name": pl_name,
                        "id": tid,
                        "name": t_name,
                        "artist": artists_str,
                    }
                )

        for fp, versions in fp_groups.items():
            if len(versions) > 1:
                total_extra_dups += len(versions) - 1
                playlist_duplicates.append(
                    {
                        "playlist_id": pl_id,
                        "playlist_name": pl_name,
                        "track_name": versions[0]["name"],
                        "artist": versions[0]["artist"],
                        "duplicate_count": len(versions),
                        "versions": versions,
                    }
                )

    chronically_skipped.sort(key=lambda r: (r["skip_rate_pct"], r["skip_count"]), reverse=True)
    coverage = round((len(telemetry_tids) / float(len(unique_tids))) * 100.0, 1) if unique_tids else 0.0

    return {
        "available": True,
        "playlist_duplicates": playlist_duplicates,
        "chronically_skipped": chronically_skipped[:40],
        "disliked_in_playlists": disliked_in_playlists,
        "summary": {
            "playlists_scanned": len(playlists),
            "duplicate_groups": len(playlist_duplicates),
            "duplicate_extra_tracks": total_extra_dups,
            "chronically_skipped_count": len(chronically_skipped),
            "disliked_in_playlists_count": len(disliked_in_playlists),
            "telemetry_coverage_pct": coverage,
        },
    }


def dedupe_spotify_playlist(sp, playlist_id):
    """Remove duplicate track fingerprints from a user-owned Spotify playlist while preserving order."""
    if not playlist_id:
        raise ValueError("playlist_id is required.")
    items = fetch_playlist_items_2026(sp, playlist_id, limit=50)
    seen_fps = set()
    kept_uris = []
    for t in items:
        tid = t.get("id")
        if not tid:
            continue
        artists = [a.get("name") for a in (t.get("artists") or []) if isinstance(a, dict) and a.get("name")]
        fp = get_track_fingerprint(t.get("name", ""), artists)
        if fp in seen_fps:
            continue
        seen_fps.add(fp)
        kept_uris.append(f"spotify:track:{tid}")

    removed_count = len(items) - len(kept_uris)
    if removed_count > 0:
        replace_spotify_playlist_items(sp, playlist_id, kept_uris)

    return {
        "playlist_id": playlist_id,
        "original_count": len(items),
        "kept_count": len(kept_uris),
        "removed_count": removed_count,
    }
