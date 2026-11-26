import datetime
import json
import os
import sys

import spotipy
from spotipy.oauth2 import SpotifyPKCE

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import config
from utils import (
    atomic_write_json,
    configure_utf8_stdio,
    enrich_artist_genres_2026,
    fetch_playlist_items_2026,
    get_effective_preferences,
    get_spotify_client_id,
    is_real_audio_features,
    is_valid_spotify_id,
    load_json_safe,
)

SCOPES = [
    "user-read-private",
    "user-top-read",
    "user-library-read",
    "user-library-modify",
    "user-follow-read",
    "user-read-recently-played",
    "user-read-currently-playing",
    "user-read-playback-state",
    "user-modify-playback-state",
    "playlist-read-private",
    "playlist-read-collaborative",
    "playlist-modify-private",
    "playlist-modify-public",
]


def authenticate_spotify():
    print("\n[1/4] Authenticating with Spotify via PKCE...")
    client_id = get_spotify_client_id()
    if not client_id:
        raise ValueError(
            "Missing Spotify Client ID. Set SPOTIFY_CLIENT_ID in config.py / env or create .spotify_client_id."
        )

    cache_file = os.path.join(ROOT_DIR, ".cache")
    auth_manager = SpotifyPKCE(
        client_id=client_id,
        redirect_uri=config.SPOTIFY_REDIRECT_URI,
        scope=" ".join(SCOPES),
        cache_path=cache_file,
        open_browser=True,
        requests_timeout=15,
    )

    sp = spotipy.Spotify(
        auth_manager=auth_manager,
        retries=3,
        status_retries=3,
        backoff_factor=0.5,
        requests_timeout=15,
    )

    user = sp.current_user()
    print(f"✅ Successfully authenticated as: {user.get('display_name') or user['id']}")
    return sp, user


def normalize_artist(artist, genre_lookup=None):
    if not isinstance(artist, dict) or not artist.get("name"):
        return None
    aid = artist.get("id")
    images = artist.get("images") or []
    genres = list(artist.get("genres") or [])
    if not genres and genre_lookup and aid and aid in genre_lookup:
        genres = list(genre_lookup[aid])

    raw_pop = artist.get("popularity")
    return {
        "id": aid,
        "name": artist["name"],
        "genres": genres,
        "popularity": int(raw_pop) if isinstance(raw_pop, (int, float)) and not isinstance(raw_pop, bool) else None,
        "image": images[0].get("url") if images and isinstance(images[0], dict) else None,
    }


def normalize_track(track, genre_lookup=None):
    if not isinstance(track, dict):
        return None
    if track.get("is_local") or track.get("type") == "episode":
        return None
    tid = track.get("id")
    name = track.get("name")
    if not tid or not name:
        return None

    album = track.get("album") or {}
    images = album.get("images") or []
    raw_artists = track.get("artists") or []
    norm_artists = [na for na in (normalize_artist(a, genre_lookup=genre_lookup) for a in raw_artists) if na]
    raw_pop = track.get("popularity")

    return {
        "id": tid,
        "uri": track.get("uri") or f"spotify:track:{tid}",
        "name": name,
        "artists": norm_artists,
        "album": {
            "id": album.get("id"),
            "name": album.get("name"),
            "release_date": album.get("release_date"),
            "image": images[0].get("url") if images and isinstance(images[0], dict) else None,
        },
        "explicit": bool(track.get("explicit", False)),
        "popularity": int(raw_pop) if isinstance(raw_pop, (int, float)) and not isinstance(raw_pop, bool) else None,
        "duration_ms": track.get("duration_ms", 0),
    }


def _collect_track_artist_ids(track_list, seen_set):
    for t in track_list:
        if not isinstance(t, dict) or t.get("is_local") or t.get("type") == "episode":
            continue
        for a in t.get("artists") or []:
            if isinstance(a, dict):
                aid = a.get("id")
                if is_valid_spotify_id(aid):
                    seen_set.add(aid)


def fetch_data(sp, user):
    print("\n[2/4] Harvesting listening data from Spotify APIs...")

    data = {
        "artists": {"short_term": [], "medium_term": [], "long_term": []},
        "tracks": {"short_term": [], "medium_term": [], "long_term": []},
        "recently_played": [],
        "short_term_artists_top_tracks": [],
        "liked_songs": [],
        "followed_artists": [],
        "saved_albums": [],
        "playlists": [],
        "artist_genre_map": {},
    }

    artist_genre_map = {}

    print("      -> Fetching Top Artists (short, medium, long)...")
    for tr in ("short_term", "medium_term", "long_term"):
        try:
            items = sp.current_user_top_artists(time_range=tr, limit=50).get("items", [])
            data["artists"][tr] = items
            for art in items:
                if isinstance(art, dict) and art.get("id") and art.get("genres"):
                    artist_genre_map[art["id"]] = list(art["genres"])
        except Exception as e:
            print(f"      ⚠️ Could not fetch top artists ({tr}): {e}")

    print("      -> Fetching Top Tracks (short, medium, long)...")
    for tr in ("short_term", "medium_term", "long_term"):
        try:
            data["tracks"][tr] = sp.current_user_top_tracks(time_range=tr, limit=50).get("items", [])
        except Exception as e:
            print(f"      ⚠️ Could not fetch top tracks ({tr}): {e}")

    print("      -> Fetching Recently Played...")
    try:
        data["recently_played"] = sp.current_user_recently_played(limit=50).get("items", [])
    except Exception as e:
        print(f"      ⚠️ Could not fetch recently played: {e}")

    print("      -> Harvesting Top Tracks for Short-Term Top Artists (Album/Artist Streaming)...")
    short_term_artist_tracks = []
    for artist in data["artists"]["short_term"][:12]:
        if not isinstance(artist, dict) or not artist.get("name"):
            continue
        try:
            safe_art = artist["name"].replace('"', "").strip()
            if not safe_art:
                continue
            s_res = sp.search(q=f'artist:"{safe_art}"', type="track", limit=6)
            items = (s_res or {}).get("tracks", {}).get("items", [])
            for item in items:
                if isinstance(item, dict) and not item.get("is_local") and item.get("type") != "episode":
                    short_term_artist_tracks.append(item)
        except Exception as e:
            print(f"      ⚠️ Artist track search warning for '{artist.get('name')}': {e}")
    data["short_term_artists_top_tracks"] = short_term_artist_tracks

    max_liked = int(get_effective_preferences().get("max_liked_songs", getattr(config, "MAX_LIKED_SONGS", 500)))
    print(f"      -> Fetching Liked Songs (up to {max_liked})...")
    liked_songs = []
    try:
        saved_resp = sp.current_user_saved_tracks(limit=50)
        while saved_resp and len(liked_songs) < max_liked:
            for item in saved_resp.get("items", []):
                if not isinstance(item, dict):
                    continue
                t = item.get("track")
                if isinstance(t, dict) and not t.get("is_local") and t.get("type") != "episode" and t.get("id"):
                    liked_songs.append(t)
                    if len(liked_songs) >= max_liked:
                        break
            if saved_resp.get("next") and len(liked_songs) < max_liked:
                saved_resp = sp.next(saved_resp)
            else:
                break
    except Exception as e:
        print(f"      ⚠️ Could not fetch liked songs: {e}")
    data["liked_songs"] = liked_songs

    print("      -> Fetching Followed Artists...")
    followed_artists = []
    after_cursor = None
    try:
        while len(followed_artists) < 250:
            f_resp = sp.current_user_followed_artists(limit=50, after=after_cursor)
            art_page = (f_resp or {}).get("artists") or {}
            items = art_page.get("items") or []
            if not items:
                break
            for art in items:
                if isinstance(art, dict) and art.get("name"):
                    followed_artists.append(art)
                    if art.get("id") and art.get("genres"):
                        artist_genre_map[art["id"]] = list(art["genres"])
            after_cursor = ((art_page.get("cursors") or {}).get("after")) or None
            if not after_cursor or not art_page.get("next"):
                break
    except Exception as e:
        print(f"      ⚠️ Could not fetch followed artists: {e}")
    data["followed_artists"] = followed_artists[:250]

    print("      -> Fetching Saved Albums (up to 100)...")
    saved_albums = []
    try:
        alb_resp = sp.current_user_saved_albums(limit=50)
        while alb_resp and len(saved_albums) < 100:
            for item in alb_resp.get("items") or []:
                alb = item.get("album") if isinstance(item, dict) else None
                if not isinstance(alb, dict) or not alb.get("id"):
                    continue
                alb_meta = {
                    "id": alb.get("id"),
                    "name": alb.get("name", "Untitled Album"),
                    "release_date": alb.get("release_date"),
                    "images": alb.get("images") or [],
                }
                alb_tracks = []
                for tr in (alb.get("tracks") or {}).get("items") or []:
                    if isinstance(tr, dict) and not tr.get("is_local") and tr.get("type") != "episode" and tr.get("id"):
                        enriched_tr = dict(tr)
                        enriched_tr.setdefault("album", alb_meta)
                        alb_tracks.append(enriched_tr)
                saved_albums.append({**alb_meta, "tracks": alb_tracks})
                if len(saved_albums) >= 100:
                    break
            if alb_resp.get("next") and len(saved_albums) < 100:
                alb_resp = sp.next(alb_resp)
            else:
                break
    except Exception as e:
        print(f"      ⚠️ Could not fetch saved albums: {e}")
    data["saved_albums"] = saved_albums

    print("      -> Fetching Personal Playlists...")
    user_id = user.get("id")
    try:
        playlists_response = sp.current_user_playlists(limit=50)
        while playlists_response:
            for pl in playlists_response.get("items", []):
                if not isinstance(pl, dict):
                    continue
                owner_id = (pl.get("owner") or {}).get("id")
                if owner_id == user_id and pl.get("id"):
                    pl_name = pl.get("name", "Untitled Playlist")
                    pl_data = {"name": pl_name, "id": pl["id"], "tracks": []}
                    print(f"         * Extracting playlist: '{pl_name}'... ", end="", flush=True)
                    try:
                        pl_data["tracks"] = fetch_playlist_items_2026(sp, pl["id"], limit=50)
                        print(f"[{len(pl_data['tracks'])} tracks grabbed]")
                        data["playlists"].append(pl_data)
                    except Exception as e:
                        print(f"[Error: {e}]")
            if playlists_response.get("next"):
                playlists_response = sp.next(playlists_response)
            else:
                break
    except Exception as e:
        print(f"      ⚠️ Could not fetch playlists: {e}")

    # Single-artist genre enrichment (batch GET /artists?ids=... was removed in Feb 2026)
    print("      -> Enriching Artist Genres for Discovered Tracks...")
    all_artist_ids = set()
    for tr in ("short_term", "medium_term", "long_term"):
        _collect_track_artist_ids(data["tracks"][tr], all_artist_ids)
    _collect_track_artist_ids(
        [item.get("track") for item in data["recently_played"] if isinstance(item, dict)],
        all_artist_ids,
    )
    _collect_track_artist_ids(data["short_term_artists_top_tracks"], all_artist_ids)
    _collect_track_artist_ids(data["liked_songs"], all_artist_ids)
    for pl in data["playlists"]:
        _collect_track_artist_ids(pl.get("tracks", []), all_artist_ids)
    for alb in data["saved_albums"]:
        _collect_track_artist_ids(alb.get("tracks", []), all_artist_ids)

    missing_artist_ids = [aid for aid in sorted(all_artist_ids) if aid not in artist_genre_map]
    if missing_artist_ids:
        extra_genres = enrich_artist_genres_2026(
            sp,
            missing_artist_ids,
            existing_map=artist_genre_map,
            max_lookups=30,
            max_empty_streak=5,
        )
        artist_genre_map.update(extra_genres)
        enriched = sum(1 for g in extra_genres.values() if g)
        print(f"      -> Enriched genres for {enriched} additional artists (Total mapped: {len(artist_genre_map)}).")

    data["artist_genre_map"] = artist_genre_map
    return data


def _summarize_extended_history(history_data):
    if not isinstance(history_data, dict) or not history_data:
        return None
    total_plays = 0
    total_ms = 0
    earliest = None
    latest = None
    for stats in history_data.values():
        if not isinstance(stats, dict):
            continue
        total_plays += int(stats.get("play_count", 0) or 0)
        total_ms += int(stats.get("total_ms_listened", 0) or 0)
        f_date = stats.get("first_streamed_date")
        l_date = stats.get("last_streamed_date")
        if f_date and (earliest is None or f_date < earliest):
            earliest = f_date
        if l_date and (latest is None or l_date > latest):
            latest = l_date
    return {
        "unique_tracks": len(history_data),
        "total_plays": total_plays,
        "total_hours_listened": round(total_ms / (1000 * 60 * 60), 2),
        "earliest_stream": earliest,
        "latest_stream": latest,
    }


def synthesize_profile(user, raw_data):
    print("\n[3/4] Synthesizing & Normalizing Taste Profile JSON...")
    genre_lookup = raw_data.get("artist_genre_map") or {}

    def norm_tracks(items):
        out = []
        for t in items or []:
            nt = normalize_track(t, genre_lookup=genre_lookup)
            if nt:
                out.append(nt)
        return out

    def norm_artists(items):
        out = []
        for a in items or []:
            na = normalize_artist(a, genre_lookup=genre_lookup)
            if na:
                out.append(na)
        return out

    recently_played = []
    for item in raw_data.get("recently_played") or []:
        if not isinstance(item, dict):
            continue
        nt = normalize_track(item.get("track"), genre_lookup=genre_lookup)
        if nt:
            recently_played.append({"track": nt, "played_at": item.get("played_at")})

    profile = {
        "schema_version": "1.2",
        "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat().replace("+00:00", "Z"),
        "user_info": {
            "id": user.get("id"),
            "display_name": user.get("display_name") or user.get("id"),
            "country": user.get("country"),
            "is_premium": user.get("product") == "premium",
        },
        "taste_summary": {},
        "artists": {
            "short_term": norm_artists(raw_data.get("artists", {}).get("short_term")),
            "medium_term": norm_artists(raw_data.get("artists", {}).get("medium_term")),
            "long_term": norm_artists(raw_data.get("artists", {}).get("long_term")),
        },
        "tracks": {
            "short_term": norm_tracks(raw_data.get("tracks", {}).get("short_term")),
            "medium_term": norm_tracks(raw_data.get("tracks", {}).get("medium_term")),
            "long_term": norm_tracks(raw_data.get("tracks", {}).get("long_term")),
        },
        "recently_played": recently_played,
        "short_term_artists_top_tracks": norm_tracks(raw_data.get("short_term_artists_top_tracks")),
        "liked_songs_sample": norm_tracks(raw_data.get("liked_songs")),
        "followed_artists": norm_artists(raw_data.get("followed_artists")),
        "saved_albums": [
            {
                "id": alb.get("id"),
                "name": alb.get("name", "Untitled Album"),
                "release_date": alb.get("release_date"),
                "tracks": norm_tracks(alb.get("tracks")),
            }
            for alb in (raw_data.get("saved_albums") or [])
            if isinstance(alb, dict)
        ],
        "playlists": [
            {
                "id": pl.get("id"),
                "name": pl.get("name", "Untitled"),
                "tracks": norm_tracks(pl.get("tracks")),
            }
            for pl in (raw_data.get("playlists") or [])
            if isinstance(pl, dict)
        ],
    }

    # Summarize Extended History without bloating user_music_profile.json with raw timestamps
    history_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "extended_history_stats.json")
    history_data = load_json_safe(history_file)
    ext_summary = _summarize_extended_history(history_data)
    if ext_summary:
        profile["extended_history_summary"] = ext_summary
        print(f"      -> Linked Extended History ({ext_summary['unique_tracks']} unique tracks, {ext_summary['total_plays']:,} plays)!")

    # --- Genre Weighting Algorithm ---
    genre_scores = {}

    def tally_genres(artist_list, multiplier):
        for index, artist in enumerate(artist_list):
            if not artist or not artist.get("genres"):
                continue
            rank_bonus = max(0.0, (50 - index) / 50.0)
            for genre in artist["genres"]:
                genre_scores[genre] = genre_scores.get(genre, 0.0) + (1.0 * multiplier * (1.0 + rank_bonus))

    tally_genres(profile["artists"]["short_term"], 1.5)
    tally_genres(profile["artists"]["medium_term"], 1.0)
    tally_genres(profile["artists"]["long_term"], 0.7)

    sorted_genres = [
        {"genre": k, "score": round(v, 2)}
        for k, v in sorted(genre_scores.items(), key=lambda item: item[1], reverse=True)
    ]

    unique_artists = set()
    for tr in ("short_term", "medium_term", "long_term"):
        unique_artists.update(a["id"] for a in profile["artists"][tr] if a and a.get("id"))
    unique_artists.update(a["id"] for a in profile["followed_artists"] if a and a.get("id"))

    unique_tracks = set()
    for tr in ("short_term", "medium_term", "long_term"):
        unique_tracks.update(t["id"] for t in profile["tracks"][tr] if t and t.get("id"))
    unique_tracks.update(
        item["track"]["id"] for item in profile["recently_played"] if item and item.get("track") and item["track"].get("id")
    )
    unique_tracks.update(t["id"] for t in profile["short_term_artists_top_tracks"] if t and t.get("id"))
    unique_tracks.update(t["id"] for t in profile["liked_songs_sample"] if t and t.get("id"))

    for pl in profile["playlists"]:
        for t in pl["tracks"]:
            if t and t.get("id"):
                unique_tracks.add(t["id"])
    for alb in profile["saved_albums"]:
        for t in alb["tracks"]:
            if t and t.get("id"):
                unique_tracks.add(t["id"])

    profile["taste_summary"] = {
        "top_genres_weighted": sorted_genres[:30],
        "total_unique_artists_analyzed": len(unique_artists),
        "total_followed_artists": len(profile["followed_artists"]),
        "total_unique_tracks_analyzed": len(unique_tracks),
        "total_saved_albums": len(profile["saved_albums"]),
        "total_personal_playlists_analyzed": len(profile["playlists"]),
        "has_extended_history": ext_summary is not None,
    }

    return profile


def collect_profile_track_ids(profile):
    """Collect all unique valid Spotify track IDs from a synthesized profile."""
    ids = []
    seen = set()

    def add_tid(tid):
        if is_valid_spotify_id(tid) and tid not in seen:
            seen.add(tid)
            ids.append(tid)

    for group in profile.get("tracks", {}).values():
        for t in group or []:
            if isinstance(t, dict):
                add_tid(t.get("id"))
    for item in profile.get("recently_played", []):
        if isinstance(item, dict) and isinstance(item.get("track"), dict):
            add_tid(item["track"].get("id"))
    for t in profile.get("short_term_artists_top_tracks", []):
        if isinstance(t, dict):
            add_tid(t.get("id"))
    for t in profile.get("liked_songs_sample", []):
        if isinstance(t, dict):
            add_tid(t.get("id"))
    for pl in profile.get("playlists", []):
        if isinstance(pl, dict):
            for t in pl.get("tracks", []):
                if isinstance(t, dict):
                    add_tid(t.get("id"))
    for alb in profile.get("saved_albums", []):
        if isinstance(alb, dict):
            for t in alb.get("tracks", []):
                if isinstance(t, dict):
                    add_tid(t.get("id"))
    return ids


def main():
    configure_utf8_stdio()
    print("========================================", flush=True)
    print(" Music Taste Analyzer - Stage 1 Spotify Extractor (v1.2)", flush=True)
    print("========================================", flush=True)

    if not get_spotify_client_id():
        print("❌ Error: Spotify Client ID cannot be empty.", flush=True)
        sys.exit(1)

    try:
        sp, user = authenticate_spotify()
        raw_data = fetch_data(sp, user)
        profile = synthesize_profile(user, raw_data)
        profile.setdefault("audio_features", {})
        output_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "user_music_profile.json")
        # Persist base Spotify profile immediately so tracks aren't lost if telemetry lookup is interrupted
        atomic_write_json(output_file, profile, indent=2)

        print("\n[3.5/4] Fetching Acoustic Telemetry via ReccoBeats API...", flush=True)
        try:
            from reccobeats_client import fetch_audio_features_batch

            all_tids = collect_profile_track_ids(profile)
            af_map = fetch_audio_features_batch(all_tids)
            real_af_map = {tid: af for tid, af in af_map.items() if is_real_audio_features(af)}
            profile["audio_features"] = real_af_map
            print(
                f"      ✅ Successfully attached real audio features (BPM, Energy, Valence) for "
                f"{len(real_af_map)} / {len(all_tids)} tracks!",
                flush=True,
            )
        except Exception as e:
            print(f"      ⚠️ ReccoBeats fetching warning: {e}", flush=True)

        atomic_write_json(output_file, profile, indent=2)

        print(f"\n[4/4] ✅ Success! Taste profile saved to {output_file}", flush=True)
        print(
            f"      🎵 Discovered {profile['taste_summary']['total_unique_tracks_analyzed']} unique tracks "
            f"across {profile['taste_summary']['total_personal_playlists_analyzed']} personal playlists!",
            flush=True,
        )

    except Exception as e:
        print(f"\n❌ Pipeline Failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
