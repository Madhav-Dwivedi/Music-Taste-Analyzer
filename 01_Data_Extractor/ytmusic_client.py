"""
YouTube Music Client & Dual-Authentication Helpers (`ytmusicapi`).

Supports both Browser Request Headers (`ytm_browser.json`) and Google Cloud
TV/Limited-Input OAuth (`ytm_oauth.json` + `.ytm_oauth_client.json`), while
keeping `ytmusicapi` lazily imported so stdlib tests run cleanly without it.
"""

import importlib.util
import json
import os
import re
import sys
import time

import requests

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import config
from utils import (
    atomic_write_json,
    extract_ytm_video_id,
    get_track_fingerprint,
    is_valid_ytm_video_id,
    load_json_safe,
    sanitize_playlist_meta,
)

_GOOGLE_DEVICE_CODE_URL = "https://oauth2.googleapis.com/device/code"
_GOOGLE_TOKEN_URL = "https://oauth2.googleapis.com/token"
_YTM_OAUTH_SCOPE = "https://www.googleapis.com/auth/youtube"


def _browser_auth_path():
    return getattr(config, "YTMUSIC_BROWSER_AUTH_FILE", os.path.join(config.BASE_DIR, "ytm_browser.json"))


def _oauth_token_path():
    return getattr(config, "YTMUSIC_OAUTH_FILE", os.path.join(config.BASE_DIR, "ytm_oauth.json"))


def _oauth_client_path():
    return getattr(config, "YTMUSIC_OAUTH_CLIENT_FILE", os.path.join(config.BASE_DIR, ".ytm_oauth_client.json"))


def get_ytm_auth_status():
    """Return non-sensitive status of YouTube Music credentials and ytmusicapi availability."""
    b_path = _browser_auth_path()
    o_path = _oauth_token_path()
    c_path = _oauth_client_path()
    b_ok = os.path.isfile(b_path) and os.path.getsize(b_path) > 10
    o_ok = os.path.isfile(o_path) and os.path.getsize(o_path) > 10
    c_ok = os.path.isfile(c_path) and os.path.getsize(c_path) > 10

    active = "browser" if b_ok else ("oauth" if o_ok else None)
    return {
        "configured": bool(active),
        "method": active,
        "has_oauth_client": c_ok,
        "ytmusicapi_installed": importlib.util.find_spec("ytmusicapi") is not None,
        "browser_configured": b_ok,
        "oauth_configured": o_ok,
        "oauth_client_configured": c_ok,
        "active_method": active,
    }


def _extract_headers_from_curl(text):
    """Parse pasted 'Copy as cURL' command text into an HTTP headers dictionary."""
    flat = re.sub(r"[\\^]\s*\r?\n\s*", " ", text)
    if '^"' in flat:
        flat = flat.replace('^"', '"')
    headers = {}
    try:
        import shlex

        tokens = shlex.split(flat, posix=True)
    except Exception:
        tokens = re.findall(r"""'(?:\\.|[^'])*'|"(?:\\.|[^"])*"|\S+""", flat)
        tokens = [
            t[1:-1] if len(t) >= 2 and t[0] == t[-1] and t[0] in ("'", '"') else t
            for t in tokens
        ]
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        low = tok.lower()
        if low in ("-h", "--header") and i + 1 < len(tokens):
            hdr = tokens[i + 1]
            if ":" in hdr:
                k, v = hdr.split(":", 1)
                if k.strip() and v.strip():
                    headers[k.strip()] = v.strip()
            i += 2
            continue
        if low.startswith("--header=") and ":" in tok[9:]:
            k, v = tok[9:].split(":", 1)
            if k.strip() and v.strip():
                headers[k.strip()] = v.strip()
        elif low in ("-b", "--cookie") and i + 1 < len(tokens):
            headers["Cookie"] = tokens[i + 1].strip()
            i += 2
            continue
        i += 1
    return headers


def _parse_browser_headers_text(raw_text):
    """Parse pasted HTTP request headers, cURL commands, or JSON dict into ytmusicapi browser.json schema."""
    text = str(raw_text or "").strip()
    if not text:
        raise ValueError("Browser headers cannot be empty.")

    parsed_dict = {}
    if text.startswith("{"):
        try:
            obj = json.loads(text)
            if isinstance(obj, dict):
                parsed_dict = {str(k).strip(): str(v).strip() for k, v in obj.items() if k and v is not None}
        except Exception as exc:
            raise ValueError(f"Malformed JSON headers: {exc}") from exc
    elif text.lower().lstrip().startswith("curl "):
        parsed_dict = _extract_headers_from_curl(text)
    else:
        lines = [ln.rstrip("\r") for ln in text.splitlines()]
        ignore_prefixes = (":method", ":path", ":scheme", ":authority", "get ", "post ", "options ")
        i = 0
        while i < len(lines):
            ln = lines[i].strip()
            i += 1
            if not ln or any(ln.lower().startswith(p) for p in ignore_prefixes):
                continue
            if ":" in ln:
                k, v = ln.split(":", 1)
                key_clean = k.strip()
                val_clean = v.strip()
                if not val_clean and i < len(lines) and ":" not in lines[i]:
                    # Firefox / Edge vertical copy format where header value is on next line
                    val_clean = lines[i].strip()
                    i += 1
                if key_clean and val_clean:
                    parsed_dict[key_clean] = val_clean

    lower_keys = {k.lower(): (k, v) for k, v in parsed_dict.items()}
    if "cookie" not in lower_keys or not lower_keys["cookie"][1].strip():
        raise ValueError(
            "Missing required 'Cookie' header. Copy full request headers (or Copy as cURL) from a logged-in "
            "music.youtube.com/youtubei/v1/browse request in DevTools Network tab."
        )

    cookie_val = lower_keys["cookie"][1].strip()
    authuser = (lower_keys.get("x-goog-authuser") or ("", "0"))[1].strip() or "0"
    ua = (lower_keys.get("user-agent") or (
        "",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130.0.0.0 Safari/537.36",
    ))[1].strip()

    normalized = {
        "Accept": "*/*",
        "Accept-Language": (lower_keys.get("accept-language") or ("", "en-US,en;q=0.9"))[1],
        "Content-Type": "application/json",
        "Origin": "https://music.youtube.com",
        "Referer": "https://music.youtube.com/",
        "User-Agent": ua,
        "X-Goog-AuthUser": authuser,
        "X-Origin": "https://music.youtube.com",
        "Cookie": cookie_val,
    }
    if "authorization" in lower_keys and lower_keys["authorization"][1]:
        normalized["Authorization"] = lower_keys["authorization"][1].strip()
    return normalized


def check_browser_headers_warning(normalized):
    """Return a non-fatal warning if browser headers appear to lack YouTube auth cookies."""
    if not isinstance(normalized, dict):
        return None
    cookie = str(normalized.get("Cookie") or "")
    auth = str(normalized.get("Authorization") or "")
    if ("SAPISID=" in cookie) or ("__Secure-3PAPISID=" in cookie) or ("SAPISIDHASH" in auth):
        return None
    return (
        "Saved headers are missing SAPISID / __Secure-3PAPISID auth cookies; YouTube Music may treat "
        "requests as signed-out unless copied from an authenticated music.youtube.com request."
    )


def save_ytm_browser_headers(headers_raw):
    """Validate pasted YouTube Music browser headers and atomically write ytm_browser.json."""
    normalized = _parse_browser_headers_text(headers_raw)
    atomic_write_json(_browser_auth_path(), normalized, indent=2)
    status = get_ytm_auth_status()
    warn = check_browser_headers_warning(normalized)
    if warn:
        status["warning"] = warn
    return status


def save_ytm_oauth_client(client_id, client_secret):
    """Persist Google Cloud OAuth TV/Limited-Input client credentials locally."""
    cid = str(client_id or "").strip()
    csec = str(client_secret or "").strip()
    if not cid or not csec:
        raise ValueError("Both Google OAuth client_id and client_secret are required.")
    atomic_write_json(_oauth_client_path(), {"client_id": cid, "client_secret": csec}, indent=2)
    return {"client_id": cid}


def start_ytm_oauth_device_flow(client_id=None, client_secret=None, http_session=None):
    """Initiate Google's OAuth2 TV/Limited-Input Device Authorization flow for YouTube Music."""
    saved_client = load_json_safe(_oauth_client_path(), default={}) or {}
    cid = str(client_id or saved_client.get("client_id") or "").strip()
    csec = str(client_secret or saved_client.get("client_secret") or "").strip()
    if not cid or not csec:
        raise ValueError("Google Cloud OAuth client_id and client_secret (TV & Limited Input type) are required.")
    save_ytm_oauth_client(cid, csec)

    sess = http_session or requests.Session()
    own_sess = http_session is None
    try:
        resp = sess.post(
            _GOOGLE_DEVICE_CODE_URL,
            data={"client_id": cid, "scope": _YTM_OAUTH_SCOPE},
            timeout=15,
        )
        data = resp.json()
        if resp.status_code != 200 or not data.get("device_code") or not data.get("user_code"):
            err = data.get("error_description") or data.get("error") or f"HTTP {resp.status_code}"
            raise ValueError(f"Failed to start Google device OAuth: {err}")
        return {
            "device_code": data["device_code"],
            "user_code": data["user_code"],
            "verification_url": data.get("verification_url") or data.get("verification_uri") or "https://www.google.com/device",
            "expires_in": int(data.get("expires_in", 1800)),
            "interval": int(data.get("interval", 5)),
        }
    finally:
        if own_sess:
            sess.close()


def complete_ytm_oauth_device_flow(device_code, client_id=None, client_secret=None, http_session=None):
    """Exchange authorized Google device code for OAuth tokens and save ytm_oauth.json."""
    code = str(device_code or "").strip()
    if not code:
        raise ValueError("device_code is required.")
    saved_client = load_json_safe(_oauth_client_path(), default={}) or {}
    cid = str(client_id or saved_client.get("client_id") or "").strip()
    csec = str(client_secret or saved_client.get("client_secret") or "").strip()
    if not cid or not csec:
        raise ValueError("Missing Google OAuth client credentials. Enter client_id and client_secret first.")

    sess = http_session or requests.Session()
    own_sess = http_session is None
    try:
        resp = sess.post(
            _GOOGLE_TOKEN_URL,
            data={
                "client_id": cid,
                "client_secret": csec,
                "device_code": code,
                "grant_type": "urn:ietf:params:oauth:grant-type:device_code",
            },
            timeout=15,
        )
        data = resp.json()
        if resp.status_code != 200 or data.get("error"):
            err = data.get("error_description") or data.get("error") or f"HTTP {resp.status_code}"
            raise ValueError(f"Google OAuth authorization not complete: {err}")

        expires_in = int(data.get("expires_in", 3600))
        token_payload = {
            "access_token": data["access_token"],
            "refresh_token": data.get("refresh_token", ""),
            "scope": data.get("scope", _YTM_OAUTH_SCOPE),
            "token_type": data.get("token_type", "Bearer"),
            "expires_in": expires_in,
            "expires_at": int(time.time()) + expires_in,
        }
        atomic_write_json(_oauth_token_path(), token_payload, indent=2)
        return {"ok": True, **get_ytm_auth_status()}
    finally:
        if own_sess:
            sess.close()


def authenticate_ytmusic(method=None):
    """Instantiate an authenticated ytmusicapi.YTMusic instance using browser headers or OAuth."""
    try:
        from ytmusicapi import OAuthCredentials, YTMusic  # type: ignore
    except ImportError as exc:
        raise RuntimeError("ytmusicapi is not installed. Run: pip install ytmusicapi") from exc

    status = get_ytm_auth_status()
    chosen = str(method or "auto").strip().lower()
    if chosen == "auto":
        chosen = status["active_method"] or ""

    if chosen == "browser":
        b_path = _browser_auth_path()
        if not os.path.isfile(b_path):
            raise FileNotFoundError("YouTube Music browser auth file (ytm_browser.json) not found.")
        return YTMusic(b_path)

    if chosen == "oauth":
        o_path = _oauth_token_path()
        if not os.path.isfile(o_path):
            raise FileNotFoundError("YouTube Music OAuth token file (ytm_oauth.json) not found.")
        saved_client = load_json_safe(_oauth_client_path(), default={}) or {}
        cid = str(saved_client.get("client_id") or "").strip()
        csec = str(saved_client.get("client_secret") or "").strip()
        if not cid or not csec:
            raise ValueError(
                "Missing Google Cloud OAuth client credentials (.ytm_oauth_client.json). "
                "Enter your TV/Limited-Input client_id and client_secret in Preferences."
            )
        return YTMusic(o_path, oauth_credentials=OAuthCredentials(client_id=cid, client_secret=csec))

    raise ValueError(
        "No YouTube Music credentials configured. Save Browser Headers or complete Google OAuth in Preferences."
    )


def _parse_duration_ms(raw_track):
    secs = raw_track.get("duration_seconds")
    if isinstance(secs, (int, float)) and not isinstance(secs, bool) and secs >= 0:
        return int(secs * 1000)
    dur_str = str(raw_track.get("duration") or "").strip()
    if ":" in dur_str:
        parts = dur_str.split(":")
        try:
            nums = [int(p) for p in parts]
            if len(nums) == 2:
                return (nums[0] * 60 + nums[1]) * 1000
            if len(nums) == 3:
                return (nums[0] * 3600 + nums[1] * 60 + nums[2]) * 1000
        except ValueError:
            pass
    return 0


def _pick_thumbnail_url(thumbs):
    if isinstance(thumbs, list) and thumbs:
        last = thumbs[-1]
        if isinstance(last, dict):
            return last.get("url")
    return None


def normalize_ytm_track(raw_track, default_album=None):
    """Convert a ytmusicapi track object into the normalized profile track schema."""
    if not isinstance(raw_track, dict):
        return None
    vid = extract_ytm_video_id(raw_track.get("videoId") or raw_track.get("id"))
    title = str(raw_track.get("title") or raw_track.get("name") or "").strip()
    if not vid or not title:
        return None

    raw_artists = raw_track.get("artists") or []
    norm_artists = []
    for a in raw_artists:
        if isinstance(a, dict) and a.get("name"):
            norm_artists.append({"id": a.get("id"), "name": str(a["name"]).strip(), "genres": []})
        elif isinstance(a, str) and a.strip():
            norm_artists.append({"id": None, "name": a.strip(), "genres": []})
    if not norm_artists:
        norm_artists = [{"id": None, "name": "Unknown", "genres": []}]

    alb_raw = raw_track.get("album")
    alb_obj = {}
    if isinstance(alb_raw, dict):
        alb_obj = {
            "id": alb_raw.get("id"),
            "name": alb_raw.get("name") or "",
            "release_date": alb_raw.get("year") or raw_track.get("year") or (default_album or {}).get("release_date"),
            "image": _pick_thumbnail_url(raw_track.get("thumbnails")) or (default_album or {}).get("image"),
        }
    elif isinstance(default_album, dict):
        alb_obj = {
            "id": default_album.get("id"),
            "name": default_album.get("name") or "",
            "release_date": default_album.get("release_date"),
            "image": _pick_thumbnail_url(raw_track.get("thumbnails")) or default_album.get("image"),
        }
    else:
        alb_obj = {
            "id": None,
            "name": "",
            "release_date": raw_track.get("year"),
            "image": _pick_thumbnail_url(raw_track.get("thumbnails")),
        }

    tid = f"yt:{vid}"
    out = {
        "id": tid,
        "video_id": vid,
        "uri": f"https://music.youtube.com/watch?v={vid}",
        "youtube_url": f"https://music.youtube.com/watch?v={vid}",
        "provider": "ytmusic",
        "name": title,
        "artists": norm_artists,
        "album": alb_obj,
        "explicit": bool(raw_track.get("isExplicit", False)),
        "popularity": None,
        "duration_ms": _parse_duration_ms(raw_track),
    }
    if raw_track.get("setVideoId"):
        out["set_video_id"] = str(raw_track["setVideoId"])
    return out


def list_ytm_playlists(ytm, limit=50):
    """List personal/library YouTube Music playlists."""
    raw = ytm.get_library_playlists(limit=max(1, min(100, int(limit or 50))))
    out = []
    for pl in raw or []:
        if not isinstance(pl, dict):
            continue
        pid = pl.get("playlistId")
        if not pid or pid in ("LM", "SE"):
            continue
        cnt = pl.get("count")
        try:
            total = int(str(cnt).split()[0]) if cnt is not None else 0
        except (ValueError, IndexError):
            total = 0
        out.append(
            {
                "id": str(pid),
                "name": str(pl.get("title") or "Untitled Playlist"),
                "tracks_total": total,
                "url": f"https://music.youtube.com/playlist?list={pid}",
            }
        )
    return out


def _check_ytm_write_ok(resp, op_name):
    if isinstance(resp, dict) and resp.get("status"):
        st = str(resp["status"]).upper()
        if st not in ("STATUS_SUCCEEDED", "OK"):
            raise RuntimeError(f"YouTube Music {op_name} failed: {resp}")


def create_or_update_ytm_playlist(
    ytm,
    name,
    description="",
    video_ids=None,
    target_playlist_id=None,
    privacy="PRIVATE",
):
    """Create a new YouTube Music playlist or replace an existing playlist's tracks in-place."""
    vids = []
    seen = set()
    for raw in video_ids or []:
        vid = extract_ytm_video_id(raw)
        if vid and vid not in seen:
            seen.add(vid)
            vids.append(vid)
    if not vids:
        raise ValueError("No valid YouTube Music videoIds provided.")

    clean_name, clean_desc = sanitize_playlist_meta(name or "AI Playlist", description or "")
    priv = str(privacy or "PRIVATE").upper()
    if priv not in ("PRIVATE", "PUBLIC", "UNLISTED"):
        priv = "PRIVATE"

    pid = str(target_playlist_id or "").strip()
    if pid:
        if hasattr(ytm, "edit_playlist"):
            try:
                ytm.edit_playlist(pid, title=clean_name, description=clean_desc)
            except Exception:
                pass
        existing = ytm.get_playlist(pid, limit=5000) or {}
        existing_items = [
            {"videoId": tr["videoId"], "setVideoId": tr["setVideoId"]}
            for tr in (existing.get("tracks") or [])
            if isinstance(tr, dict) and tr.get("videoId") and tr.get("setVideoId")
        ]
        if existing_items:
            for i in range(0, len(existing_items), 100):
                rem_res = ytm.remove_playlist_items(pid, existing_items[i : i + 100])
                _check_ytm_write_ok(rem_res, "remove_playlist_items")
        try:
            for i in range(0, len(vids), 100):
                add_res = ytm.add_playlist_items(pid, vids[i : i + 100], duplicates=False)
                _check_ytm_write_ok(add_res, "add_playlist_items")
        except Exception:
            if existing_items:
                restore_vids = [it["videoId"] for it in existing_items if it.get("videoId")]
                for i in range(0, len(restore_vids), 100):
                    try:
                        ytm.add_playlist_items(pid, restore_vids[i : i + 100], duplicates=False)
                    except Exception:
                        pass
            raise
    else:
        first_batch = vids[:100]
        rest = vids[100:]
        pid = ytm.create_playlist(
            title=clean_name,
            description=clean_desc,
            privacy_status=priv,
            video_ids=first_batch,
        )
        if not isinstance(pid, str) or not pid.strip():
            raise RuntimeError(f"YouTube Music failed to create playlist: {pid}")
        pid = pid.strip()
        for i in range(0, len(rest), 100):
            add_res = ytm.add_playlist_items(pid, rest[i : i + 100], duplicates=False)
            _check_ytm_write_ok(add_res, "add_playlist_items")

    return {
        "playlist_id": pid,
        "url": f"https://music.youtube.com/playlist?list={pid}",
        "count": len(vids),
    }


def rate_ytm_song(ytm, video_id, rating="LIKE"):
    """Apply LIKE, DISLIKE, or INDIFFERENT rating to a YouTube Music track."""
    vid = extract_ytm_video_id(video_id)
    if not vid:
        raise ValueError("Invalid YouTube Music videoId.")
    clean_rating = str(rating or "LIKE").strip().upper()
    if clean_rating not in ("LIKE", "DISLIKE", "INDIFFERENT"):
        raise ValueError("rating must be LIKE, DISLIKE, or INDIFFERENT.")
    res = ytm.rate_song(vid, clean_rating)
    return {"ok": True, "video_id": vid, "rating": clean_rating, "result": res}


def search_ytm_track(ytm, name, artist):
    """Search YouTube Music songs and return the strongest normalized track match or None."""
    t_name = str(name or "").strip()
    t_art = str(artist or "").strip()
    if not t_name:
        return None
    primary_art = t_art.split(",")[0].strip() if t_art and t_art.lower() != "unknown" else ""
    query = f"{t_name} {primary_art}".strip()

    try:
        candidates = ytm.search(query, filter="songs", limit=5) or []
    except Exception:
        candidates = []
    if not candidates:
        try:
            candidates = ytm.search(query, limit=5) or []
        except Exception:
            candidates = []

    target_fp = get_track_fingerprint(t_name, t_art)
    target_title_base = get_track_fingerprint(t_name, "").split("::")[0]

    best_item = None
    best_score = -1
    for item in candidates:
        if not isinstance(item, dict):
            continue
        vid = extract_ytm_video_id(item.get("videoId"))
        cand_title = str(item.get("title") or "").strip()
        if not vid or not cand_title:
            continue
        cand_artists = [
            str(a.get("name") or "").strip()
            for a in (item.get("artists") or [])
            if isinstance(a, dict) and a.get("name")
        ]
        cand_fp = get_track_fingerprint(cand_title, cand_artists)
        cand_title_base = get_track_fingerprint(cand_title, "").split("::")[0]
        cand_art_joined = ", ".join(cand_artists).lower()

        score = 0
        if cand_fp == target_fp:
            score += 100
        elif target_title_base and target_title_base == cand_title_base:
            score += 60
        elif target_title_base and target_title_base in cand_title_base:
            score += 35

        if primary_art and primary_art.lower() in cand_art_joined:
            score += 40
        elif primary_art and primary_art.lower() in cand_title.lower():
            score += 20

        if str(item.get("resultType") or "").lower() == "song":
            score += 10

        if score > best_score:
            best_score = score
            best_item = (vid, cand_title, ", ".join(cand_artists) or t_art or "Unknown")

    if best_item and best_score >= 30:
        vid, cand_title, art_str = best_item
        return {
            "id": f"yt:{vid}",
            "video_id": vid,
            "name": cand_title,
            "artist": art_str,
            "provider": "ytmusic",
            "youtube_url": f"https://music.youtube.com/watch?v={vid}",
        }
    return None
