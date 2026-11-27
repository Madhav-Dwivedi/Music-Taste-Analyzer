"""
Stage 1 YouTube Music Library Extractor -> `01_Data_Extractor/ytmusic_profile.json`.

Harvests Liked Songs, Library Songs, Saved Albums, Subscribed Artists, Playlists,
Watch History, and Uploaded Songs via `ytmusicapi`, enriches BPM/Energy/Valence
via ReccoBeats `/v1/track/search`, and outputs the unified Stage 2 schema.
"""

import argparse
import datetime
import os
import sys

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import config
from reccobeats_client import fetch_audio_features_by_search
from utils import (
    atomic_write_json,
    configure_utf8_stdio,
    get_track_fingerprint,
    is_real_audio_features,
    load_json_safe,
)
from ytmusic_client import authenticate_ytmusic, normalize_ytm_track


def _normalize_ytm_artist(raw_art):
    if not isinstance(raw_art, dict):
        return None
    name = str(raw_art.get("name") or raw_art.get("artist") or "").strip()
    if not name:
        return None
    aid = raw_art.get("browseId") or raw_art.get("id")
    thumbs = raw_art.get("thumbnails") or []
    img = thumbs[-1].get("url") if thumbs and isinstance(thumbs[-1], dict) else None
    return {
        "id": str(aid) if aid else None,
        "name": name,
        "genres": [],
        "popularity": None,
        "image": img,
        "provider": "ytmusic",
    }


def fetch_ytm_data(ytm):
    """Fetch library, liked songs, albums, subscriptions, playlists, history, and uploads."""
    print("\n[2/4] Harvesting listening data from YouTube Music...")
    data = {
        "liked_songs": [],
        "library_songs": [],
        "history": [],
        "followed_artists": [],
        "saved_albums": [],
        "playlists": [],
        "uploaded_songs": [],
    }

    try:
        print("      -> Fetching YouTube Music Liked Songs...")
        liked_resp = ytm.get_liked_songs(limit=500) or {}
        data["liked_songs"] = list(liked_resp.get("tracks") or [])
    except Exception as e:
        print(f"      ⚠️ Liked songs warning: {e}")

    try:
        print("      -> Fetching Library Songs...")
        data["library_songs"] = list(ytm.get_library_songs(limit=500) or [])
    except Exception as e:
        print(f"      ⚠️ Library songs warning: {e}")

    try:
        print("      -> Fetching Subscribed Artists...")
        data["followed_artists"] = list(ytm.get_library_subscriptions(limit=250) or [])
    except Exception as e:
        print(f"      ⚠️ Subscriptions warning: {e}")

    try:
        print("      -> Fetching Recent History...")
        data["history"] = list(ytm.get_history() or [])
    except Exception as e:
        print(f"      ⚠️ History warning: {e}")

    try:
        print("      -> Fetching Saved Albums (up to 30 detailed tracklists)...")
        raw_albums = list(ytm.get_library_albums(limit=100) or [])
        for alb in raw_albums[:30]:
            if not isinstance(alb, dict) or not alb.get("browseId"):
                continue
            try:
                full_alb = ytm.get_album(alb["browseId"]) or {}
                thumbs = full_alb.get("thumbnails") or alb.get("thumbnails") or []
                img = thumbs[-1].get("url") if thumbs and isinstance(thumbs[-1], dict) else None
                data["saved_albums"].append(
                    {
                        "id": alb["browseId"],
                        "name": full_alb.get("title") or alb.get("title") or "Untitled Album",
                        "release_date": full_alb.get("year") or alb.get("year"),
                        "image": img,
                        "tracks": list(full_alb.get("tracks") or []),
                    }
                )
            except Exception:
                continue
    except Exception as e:
        print(f"      ⚠️ Saved albums warning: {e}")

    try:
        print("      -> Fetching Personal Playlists...")
        excl_raw = load_json_safe(os.path.join(ROOT_DIR, "ytm_excluded_playlists.json"), default=[])
        excluded_set = {str(x).strip().lower() for x in (excl_raw or []) if str(x).strip()}
        raw_pls = list(ytm.get_library_playlists(limit=50) or [])
        for pl in raw_pls:
            if not isinstance(pl, dict):
                continue
            pid = pl.get("playlistId")
            if not pid or pid in ("LM", "SE"):
                continue
            pl_title = pl.get("title") or "Untitled Playlist"
            if pl_title.strip().lower() in excluded_set or str(pid).strip().lower() in excluded_set:
                print(f"         * Skipping excluded YTM playlist: '{pl_title}'")
                continue
            try:
                full_pl = ytm.get_playlist(pid, limit=500) or {}
                tracks = list(full_pl.get("tracks") or [])
                print(f"         * Extracting YTM playlist: '{pl_title}'... [{len(tracks)} tracks]")
                data["playlists"].append({"id": str(pid), "name": pl_title, "tracks": tracks})
            except Exception as e:
                print(f"         * Warning on YTM playlist '{pl_title}': {e}")
    except Exception as e:
        print(f"      ⚠️ Playlists warning: {e}")

    if hasattr(ytm, "get_library_upload_songs"):
        try:
            data["uploaded_songs"] = list(ytm.get_library_upload_songs(limit=200) or [])
            if data["uploaded_songs"]:
                print(f"      -> Found {len(data['uploaded_songs'])} uploaded personal songs.")
        except Exception:
            # Uploaded songs are only supported under Browser auth; fail silently under OAuth
            pass

    return data


def synthesize_ytm_profile(raw_data):
    """Normalize fetched YouTube Music data into Music Taste Analyzer's standard Stage 1 schema."""
    print("\n[3/4] Synthesizing & Normalizing YouTube Music Profile JSON...", flush=True)

    max_pl_track_ms = 20 * 60 * 1000
    skipped_long_videos = 0

    def norm_list(items, default_album=None, filter_long=False):
        nonlocal skipped_long_videos
        out = []
        for item in items or []:
            nt = normalize_ytm_track(item, default_album=default_album)
            if nt:
                if filter_long and (nt.get("duration_ms") or 0) > max_pl_track_ms:
                    skipped_long_videos += 1
                    continue
                out.append(nt)
        return out

    liked_norm = norm_list(raw_data.get("liked_songs"))
    lib_norm = norm_list(raw_data.get("library_songs"))
    hist_norm = norm_list(raw_data.get("history"))
    upload_norm = norm_list(raw_data.get("uploaded_songs"))

    # Combine Library + Uploaded into liked_songs_sample; Liked + History feed top/recent slots
    seen_liked_ids = {x["id"] for x in liked_norm}
    merged_liked = list(liked_norm)
    for tr in (lib_norm + upload_norm):
        if tr["id"] not in seen_liked_ids:
            seen_liked_ids.add(tr["id"])
            merged_liked.append(tr)
    recent_wrapped = [{"track": tr, "played_at": None} for tr in hist_norm]

    followed = [
        na
        for na in (_normalize_ytm_artist(a) for a in (raw_data.get("followed_artists") or []))
        if na
    ]

    saved_albums_norm = []
    for alb in raw_data.get("saved_albums") or []:
        if not isinstance(alb, dict):
            continue
        def_alb = {
            "id": alb.get("id"),
            "name": alb.get("name") or "Untitled Album",
            "release_date": alb.get("release_date"),
            "image": alb.get("image"),
        }
        saved_albums_norm.append({**def_alb, "tracks": norm_list(alb.get("tracks"), default_album=def_alb)})

    playlists_norm = [
        {
            "id": pl.get("id"),
            "name": pl.get("name", "Untitled"),
            "provider": "ytmusic",
            "tracks": norm_list(pl.get("tracks"), filter_long=True),
        }
        for pl in (raw_data.get("playlists") or [])
        if isinstance(pl, dict)
    ]
    if skipped_long_videos:
        print(
            f"      🧹 Filtered {skipped_long_videos} non-music long video(s) (>20 min) from YTM playlists.",
            flush=True,
        )

    unique_ids = set()
    for grp in (liked_norm, merged_liked, hist_norm):
        unique_ids.update(t["id"] for t in grp)
    for pl in playlists_norm:
        unique_ids.update(t["id"] for t in pl["tracks"])
    for alb in saved_albums_norm:
        unique_ids.update(t["id"] for t in alb["tracks"])

    return {
        "schema_version": "1.2-ytmusic",
        "provider": "ytmusic",
        "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat().replace("+00:00", "Z"),
        "user_info": {"provider": "ytmusic"},
        "artists": {
            "short_term": followed[:30],
            "medium_term": followed[30:80],
            "long_term": [],
        },
        "tracks": {
            "short_term": hist_norm[:50],
            "medium_term": liked_norm[:50],
            "long_term": [],
        },
        "recently_played": recent_wrapped,
        "short_term_artists_top_tracks": [],
        "liked_songs_sample": merged_liked,
        "followed_artists": followed,
        "saved_albums": saved_albums_norm,
        "playlists": playlists_norm,
        "audio_features": {},
        "taste_summary": {
            "top_genres_weighted": [],
            "total_unique_artists_analyzed": len(followed),
            "total_followed_artists": len(followed),
            "total_unique_tracks_analyzed": len(unique_ids),
            "total_saved_albums": len(saved_albums_norm),
            "total_personal_playlists_analyzed": len(playlists_norm),
        },
    }


def _collect_search_candidates(profile):
    candidates = []
    seen = set()
    max_search_ms = 12 * 60 * 1000

    def scan(items):
        for tr in items or []:
            if not isinstance(tr, dict) or not tr.get("id") or not tr.get("name"):
                continue
            if (tr.get("duration_ms") or 0) > max_search_ms:
                continue
            tid = tr["id"]
            if tid in seen:
                continue
            seen.add(tid)
            arts = [a.get("name") for a in (tr.get("artists") or []) if isinstance(a, dict) and a.get("name")]
            candidates.append(
                {
                    "id": tid,
                    "name": tr["name"],
                    "artist": ", ".join(arts) if arts else "",
                }
            )

    scan(profile.get("liked_songs_sample"))
    for entry in profile.get("recently_played") or []:
        if isinstance(entry, dict):
            scan([entry.get("track")])
    for pl in profile.get("playlists") or []:
        scan(pl.get("tracks"))
    for alb in profile.get("saved_albums") or []:
        scan(alb.get("tracks"))
    return candidates


def _reuse_spotify_audio_features(candidates, stage1_dir=None):
    """Reuse real audio_features already fetched in Spotify Stage 1 by normalized (title, artist) fingerprint."""
    s1_dir = stage1_dir or os.path.dirname(os.path.abspath(__file__))
    sp_prof = load_json_safe(os.path.join(s1_dir, "user_music_profile.json"))
    if not isinstance(sp_prof, dict):
        return {}, list(candidates or [])

    af_map = sp_prof.get("audio_features") or {}
    if not isinstance(af_map, dict) or not af_map:
        return {}, list(candidates or [])

    fp_to_af = {}

    def index_tracks(track_list):
        for tr in track_list or []:
            if not isinstance(tr, dict):
                continue
            tid = str(tr.get("id") or "")
            af = af_map.get(tid)
            if not is_real_audio_features(af):
                continue
            arts = [
                a.get("name", "")
                for a in (tr.get("artists") or [])
                if isinstance(a, dict) and a.get("name")
            ]
            fp = get_track_fingerprint(tr.get("name", ""), arts)
            if fp and fp not in fp_to_af:
                rec = dict(af)
                rec.setdefault("matched_spotify_id", tid)
                fp_to_af[fp] = rec

    tracks_sec = sp_prof.get("tracks") or {}
    for tr_key in ("short_term", "medium_term", "long_term"):
        index_tracks(tracks_sec.get(tr_key))
    for entry in sp_prof.get("recently_played") or []:
        if isinstance(entry, dict) and isinstance(entry.get("track"), dict):
            index_tracks([entry["track"]])
    index_tracks(sp_prof.get("short_term_artists_top_tracks"))
    index_tracks(sp_prof.get("liked_songs_sample"))
    for pl in sp_prof.get("playlists") or []:
        if isinstance(pl, dict):
            index_tracks(pl.get("tracks"))
    for alb in sp_prof.get("saved_albums") or []:
        if isinstance(alb, dict):
            index_tracks(alb.get("tracks"))

    reused = {}
    remaining = []
    for cand in candidates or []:
        if not isinstance(cand, dict) or not cand.get("id"):
            continue
        fp = get_track_fingerprint(cand.get("name", ""), cand.get("artist", ""))
        if fp and fp in fp_to_af:
            reused[cand["id"]] = dict(fp_to_af[fp])
        else:
            remaining.append(cand)
    return reused, remaining


def main():
    configure_utf8_stdio()
    parser = argparse.ArgumentParser(description="Music Taste Analyzer - Stage 1 YouTube Music Extractor")
    parser.add_argument(
        "--auth-method",
        choices=["auto", "browser", "oauth"],
        default="auto",
        help="YouTube Music auth method",
    )
    parser.add_argument(
        "--skip-reccobeats",
        action="store_true",
        help="Skip ReccoBeats audio-feature title search lookup",
    )
    args = parser.parse_args()

    print("========================================", flush=True)
    print(" YouTube Music Taste Extractor - Stage 1", flush=True)
    print("========================================", flush=True)

    try:
        ytm = authenticate_ytmusic(method=args.auth_method)
        raw = fetch_ytm_data(ytm)
        profile = synthesize_ytm_profile(raw)

        out_path = getattr(
            config,
            "YTMUSIC_PROFILE_FILE",
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "ytmusic_profile.json"),
        )
        # Persist base library profile immediately so tracks aren't lost if telemetry lookup is interrupted
        atomic_write_json(out_path, profile, indent=2)

        if not args.skip_reccobeats:
            print("\n[3.5/4] Enriching Audio Telemetry via ReccoBeats Track Search...", flush=True)
            try:
                cands = _collect_search_candidates(profile)
                reused_af, remaining_cands = _reuse_spotify_audio_features(cands)
                if reused_af:
                    print(
                        f"      ⚡ Reused local Spotify audio telemetry for {len(reused_af)} tracks (skipping ReccoBeats search).",
                        flush=True,
                    )
                searched_af = (
                    fetch_audio_features_by_search(remaining_cands, max_lookups=150) if remaining_cands else {}
                )
                merged_af = {**reused_af, **searched_af}
                profile["audio_features"] = {
                    k: v for k, v in merged_af.items() if is_real_audio_features(v)
                }
                atomic_write_json(out_path, profile, indent=2)
            except Exception as e:
                print(f"      ⚠️ ReccoBeats YTM search enrichment warning: {e}", flush=True)

        ts = profile["taste_summary"]
        print(f"\n[4/4] ✅ Saved YouTube Music profile to {out_path}", flush=True)
        print(
            f"      🎵 {ts['total_unique_tracks_analyzed']} unique tracks | "
            f"{ts['total_personal_playlists_analyzed']} playlists | "
            f"{len(profile['audio_features'])} ReccoBeats audio profiles matched",
            flush=True,
        )
    except Exception as exc:
        print(f"\n❌ YouTube Music Sync Failed: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
