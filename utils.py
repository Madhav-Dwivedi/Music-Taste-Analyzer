import csv
import importlib.util
import io
import ipaddress
import json
import os
import random
import re
import subprocess
import sys
import threading
import time
import unicodedata
import urllib.parse
import uuid
import zipfile
from datetime import datetime, timezone

import config

_SPOTIFY_ID_RE = re.compile(r"^[A-Za-z0-9]{22}$")
_SUFFIX_DASH_RE = re.compile(
    r"(?i)\s*-\s*(?:\d{4}\s+)?(?:live|remaster(?:ed)?|radio\s*edit|acoustic|version|mix|edit|mono|stereo|deluxe|instrumental|sped\s*up|slowed|bonus\s*track|single\s*version|from\b).*"
)
_SUFFIX_PAREN_RE = re.compile(
    r"(?i)\s*\((?:\d{4}\s+)?(?:live|remaster(?:ed)?|radio\s*edit|acoustic|version|mix|edit|feat\.|ft\.|with\b|instrumental|sped\s*up|slowed|bonus\s*track|single\s*version|from\b).*?\)"
)
_SUFFIX_BRACKET_RE = re.compile(
    r"(?i)\s*\[(?:\d{4}\s+)?(?:live|remaster(?:ed)?|radio\s*edit|acoustic|version|mix|edit|feat\.|ft\.|with\b|instrumental|sped\s*up|slowed|bonus\s*track|single\s*version|from\b).*?\]"
)

LEGACY_FALLBACK_AF = {
    "bpm": 120.0,
    "energy": 0.5,
    "danceability": 0.5,
    "valence": 0.5,
    "acousticness": 0.5,
}


def configure_utf8_stdio():
    """Safely configure stdout/stderr to UTF-8 with replacement on unsupported consoles."""
    for stream in (sys.stdout, sys.stderr):
        if stream is not None and hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8", errors="replace")
            except Exception:
                pass


def _replace_with_retry(src, dst, max_attempts=5):
    """Replace dst with src, retrying briefly if Windows Defender/Indexers hold a transient lock."""
    for attempt in range(max_attempts):
        try:
            os.replace(src, dst)
            return
        except PermissionError:
            if attempt == max_attempts - 1:
                raise
            time.sleep(0.08 * (attempt + 1))


def atomic_write_text(path, text):
    """Atomically write UTF-8 text to path using a unique same-directory temporary file."""
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    tmp_path = f"{path}.{os.getpid()}.{threading.get_ident()}.{uuid.uuid4().hex[:6]}.tmp"
    try:
        with open(tmp_path, "w", encoding="utf-8", newline="\n") as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        _replace_with_retry(tmp_path, path)
    except Exception:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass
        raise


def atomic_write_json(path, data, indent=2):
    """Atomically serialize JSON data to path in UTF-8."""
    payload = json.dumps(data, indent=indent, ensure_ascii=False)
    atomic_write_text(path, payload)


def load_json_safe(path, default=None):
    """Load JSON file safely, returning default if missing or malformed."""
    if not path or not os.path.exists(path):
        return default
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default


def load_user_preferences():
    """Load optional user_preferences.json overrides if present."""
    prefs_path = getattr(config, "USER_PREFS_FILE", os.path.join(config.BASE_DIR, "user_preferences.json"))
    data = load_json_safe(prefs_path, default={})
    return data if isinstance(data, dict) else {}


def get_spotify_client_id():
    """Resolve Spotify Client ID from environment, local .spotify_client_id file, or config."""
    env_cid = os.environ.get("SPOTIFY_CLIENT_ID", "").strip()
    if env_cid:
        return env_cid

    cid_file = os.path.join(config.BASE_DIR, ".spotify_client_id")
    if os.path.exists(cid_file):
        try:
            with open(cid_file, "r", encoding="utf-8") as f:
                file_cid = f.read().strip()
            if file_cid:
                return file_cid
        except OSError:
            pass

    cfg_cid = (getattr(config, "SPOTIFY_CLIENT_ID", None) or "").strip()
    return cfg_cid or None


def is_valid_spotify_id(tid):
    """Return True if tid is a standard 22-character base62 Spotify ID."""
    return isinstance(tid, str) and bool(_SPOTIFY_ID_RE.match(tid))


def _normalize_text_for_fp(text):
    if not text:
        return ""
    decomposed = unicodedata.normalize("NFKD", str(text))
    without_marks = "".join(ch for ch in decomposed if unicodedata.category(ch) != "Mn")
    cleaned = re.sub(r"[^\w\s]", " ", without_marks)
    return re.sub(r"\s+", " ", cleaned).lower().strip()


def _extract_primary_artist(artists):
    if not artists:
        return "unknown"
    if isinstance(artists, str):
        parts = [p.strip() for p in artists.split(",") if p.strip()]
        return parts[0] if parts else "unknown"
    if isinstance(artists, (list, tuple)):
        for item in artists:
            if isinstance(item, dict):
                name = (item.get("name") or "").strip()
                if name:
                    return name
            elif isinstance(item, str) and item.strip():
                return item.strip()
    return "unknown"


def get_track_fingerprint(name, artists):
    """Build a canonical deduplication key across extraction, vectorization, and discovery."""
    raw_name = str(name or "")
    clean_name = _SUFFIX_DASH_RE.sub("", raw_name)
    clean_name = _SUFFIX_PAREN_RE.sub("", clean_name)
    clean_name = _SUFFIX_BRACKET_RE.sub("", clean_name)

    norm_name = _normalize_text_for_fp(clean_name) or _normalize_text_for_fp(raw_name) or "unknown"
    primary_artist = _normalize_text_for_fp(_extract_primary_artist(artists)) or "unknown"
    return f"{norm_name}::{primary_artist}"


def parse_spotify_timestamp(ts_str):
    """Parse Spotify ISO-8601 or space-delimited UTC timestamps into a UTC-aware datetime."""
    if not ts_str or not isinstance(ts_str, str):
        return None
    s = ts_str.strip()
    if not s:
        return None
    try:
        if s.endswith("Z"):
            s = s[:-1] + "+00:00"
        if "T" in s:
            dt = datetime.fromisoformat(s)
        else:
            for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M"):
                try:
                    dt = datetime.strptime(s, fmt)
                    break
                except ValueError:
                    dt = None
            if dt is None:
                dt = datetime.fromisoformat(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        else:
            dt = dt.astimezone(timezone.utc)
        return dt
    except Exception:
        return None


def is_real_audio_features(af):
    """Return True if af contains genuine acoustic telemetry rather than fallback/not_found placeholders."""
    if not isinstance(af, dict) or not af:
        return False
    source = af.get("source")
    if source in ("not_found", "fallback", "estimated"):
        return False

    has_metric = any(isinstance(af.get(k), (int, float)) for k in ("bpm", "tempo", "energy", "valence"))
    if not has_metric:
        return False

    if not source:
        try:
            matches_poisoned = all(
                abs(float(af.get(k)) - v) < 1e-6
                for k, v in LEGACY_FALLBACK_AF.items()
                if k in af
            )
            if matches_poisoned and all(k in af for k in LEGACY_FALLBACK_AF):
                return False
        except (TypeError, ValueError):
            return False

    return True


def format_audio_feature_text(af):
    """Format verified audio features into semantic document text for embeddings."""
    if not is_real_audio_features(af):
        return ""
    try:
        bpm = float(af.get("bpm", af.get("tempo", 120.0)))
        eng = float(af.get("energy", 0.5))
        val = float(af.get("valence", 0.5))
    except (TypeError, ValueError):
        return ""

    e_label = "High Energy" if eng >= 0.7 else ("Low Energy Chill" if eng <= 0.35 else "Medium Energy")
    v_label = "Upbeat Happy" if val >= 0.65 else ("Sad Dark Melancholy" if val <= 0.35 else "Neutral Vibe")

    parts = [f"Tempo: {bpm:.1f} BPM", f"{e_label} ({eng:.2f})", f"{v_label} ({val:.2f})"]

    dance = af.get("danceability")
    if isinstance(dance, (int, float)) and dance >= 0.72:
        parts.append("Danceable")
    acoustic = af.get("acousticness")
    if isinstance(acoustic, (int, float)) and acoustic >= 0.70:
        parts.append("Acoustic")
    instrumental = af.get("instrumentalness")
    if isinstance(instrumental, (int, float)) and instrumental >= 0.50:
        parts.append("Instrumental")
    speech = af.get("speechiness")
    if isinstance(speech, (int, float)) and speech >= 0.33:
        parts.append("Spoken/Rap")

    return " | ".join(parts)


_OPENVINO_DEVICES_CACHE = None


def get_openvino_devices(refresh=False):
    """Detect available OpenVINO devices safely and cache result for low-overhead telemetry polling."""
    global _OPENVINO_DEVICES_CACHE
    if _OPENVINO_DEVICES_CACHE is not None and not refresh:
        return {
            **_OPENVINO_DEVICES_CACHE,
            "raw": list(_OPENVINO_DEVICES_CACHE["raw"]),
            "descriptions": list(_OPENVINO_DEVICES_CACHE["descriptions"]),
        }

    try:
        from openvino import Core

        core = Core()
        raw_devices = list(core.available_devices)
        descriptions = []
        for dev in raw_devices:
            try:
                descriptions.append(core.get_property(dev, "FULL_DEVICE_NAME"))
            except Exception:
                descriptions.append(dev)

        npu_dev = next((d for d in raw_devices if str(d).upper().startswith("NPU")), None)
        gpu_dev = next((d for d in raw_devices if str(d).upper().startswith("GPU")), None)
        cpu_dev = next((d for d in raw_devices if str(d).upper() == "CPU"), "CPU")
        info = {
            "raw": raw_devices,
            "npu": npu_dev,
            "gpu": gpu_dev,
            "cpu": cpu_dev,
            "descriptions": descriptions if descriptions else ["CPU"],
        }
    except Exception:
        info = {
            "raw": ["CPU"],
            "npu": None,
            "gpu": None,
            "cpu": "CPU",
            "descriptions": ["CPU"],
        }

    _OPENVINO_DEVICES_CACHE = {
        **info,
        "raw": list(info["raw"]),
        "descriptions": list(info["descriptions"]),
    }
    return info


def sanitize_playlist_meta(name, description=""):
    """Normalize playlist title and description for Spotify API limits."""
    clean_name = re.sub(r"[\r\n\t]+", " ", str(name or "")).strip()
    clean_name = re.sub(r"\s+", " ", clean_name)[:100] or "AI Playlist"

    clean_desc = re.sub(r"[\r\n\t]+", " ", str(description or "")).strip()
    clean_desc = re.sub(r"\s+", " ", clean_desc)[:300]
    return clean_name, clean_desc


def create_spotify_playlist(sp, name, description="", public=False, collaborative=False):
    """Create a Spotify playlist using 2026 /me/playlists endpoint with legacy fallback."""
    clean_name, clean_desc = sanitize_playlist_meta(name, description)
    payload = {
        "name": clean_name,
        "public": bool(public),
        "collaborative": bool(collaborative),
        "description": clean_desc,
    }
    try:
        if hasattr(sp, "current_user_playlist_create"):
            return sp.current_user_playlist_create(
                name=clean_name,
                public=bool(public),
                collaborative=bool(collaborative),
                description=clean_desc,
            )
        return sp._post("me/playlists", payload=payload)
    except Exception as exc:
        msg = str(exc)
        status = getattr(exc, "http_status", None)
        if status in (404, 405) or "404" in msg or "405" in msg:
            user_id = sp.current_user()["id"]
            return sp.user_playlist_create(
                user=user_id,
                name=clean_name,
                public=bool(public),
                collaborative=bool(collaborative),
                description=clean_desc,
            )
        raise


def add_items_to_spotify_playlist(sp, playlist_id, track_uris, batch_size=100):
    """Add track URIs to a playlist in chunks using Spotify 2026 /items endpoint."""
    uris = [u for u in track_uris if u]
    if not uris:
        return
    step = max(1, min(100, int(batch_size)))
    for i in range(0, len(uris), step):
        batch = uris[i : i + step]
        try:
            sp.playlist_add_items(playlist_id, batch)
        except Exception as exc:
            msg = str(exc)
            status = getattr(exc, "http_status", None)
            if status in (400, 404, 405) or any(code in msg for code in ("400", "404", "405")) or isinstance(exc, AttributeError):
                sp._post(f"playlists/{playlist_id}/items", payload={"uris": batch})
            else:
                raise


def fetch_playlist_items_2026(sp, playlist_id, limit=50):
    """Fetch playlist track objects via Spotify 2026 /playlists/{id}/items endpoint (max limit 50)."""
    tracks = []
    eff_limit = max(1, min(50, int(limit or 50)))
    resp = sp._get(f"playlists/{playlist_id}/items", limit=eff_limit)
    while resp:
        for entry in resp.get("items", []):
            if not isinstance(entry, dict):
                continue
            track_obj = entry.get("track") or entry.get("item") or (entry if "name" in entry else None)
            if not isinstance(track_obj, dict):
                continue
            if track_obj.get("is_local") or track_obj.get("type") == "episode":
                continue
            if not track_obj.get("id") or not track_obj.get("name"):
                continue
            tracks.append(track_obj)

        if resp.get("next"):
            resp = sp.next(resp)
        else:
            break
    return tracks


def enrich_artist_genres_2026(sp, artist_ids, existing_map=None, max_lookups=25, max_empty_streak=5):
    """
    Safely look up artist genres one-by-one using GET /artists/{id} (2026 compliant;
    batch GET /artists?ids=... was removed in Feb 2026). Aborts early if Dev Mode
    returns empty/missing genres or restricts the endpoint.
    """
    existing = existing_map if isinstance(existing_map, dict) else {}
    out = {}
    targets = []
    seen = set()
    for raw_aid in artist_ids or []:
        aid = str(raw_aid or "").strip()
        if not aid or aid in seen or aid in existing or not is_valid_spotify_id(aid):
            continue
        seen.add(aid)
        targets.append(aid)
        if len(targets) >= max(1, int(max_lookups)):
            break

    empty_streak = 0
    for aid in targets:
        try:
            art = sp.artist(aid)
            if not isinstance(art, dict):
                empty_streak += 1
            else:
                genres = list(art.get("genres") or [])
                out[aid] = genres
                empty_streak = 0 if genres else empty_streak + 1
            if empty_streak >= max(1, int(max_empty_streak)):
                break
        except Exception as exc:
            status = getattr(exc, "http_status", None)
            msg = str(exc)
            if status in (400, 403, 404) or any(code in msg for code in ("400", "403", "404")):
                break
            break
    return out


# =======================================================
# PREFERENCES, EXCLUSIONS, SEQUENCING & ARCHIVE HELPERS
# =======================================================

_ARCHIVE_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,120}$")

DEFAULT_BUILTIN_PRESETS = {
    "Gym Mode": {
        "builtin": True,
        "avoid_artists": [],
        "avoid_genres": ["ambient", "sleep", "classical"],
        "flow_mode": "warmup_peak_cooldown",
        "size": 35,
        "fast": True,
        "local_only": False,
        "clean_only": False,
        "deep_cuts": False,
        "prompt_hint": "135+ BPM high energy workout hype",
    },
    "Deep Focus": {
        "builtin": True,
        "avoid_artists": [],
        "avoid_genres": ["metal", "punk", "trap"],
        "flow_mode": "chill_descent",
        "size": 40,
        "fast": True,
        "local_only": False,
        "clean_only": True,
        "deep_cuts": False,
        "prompt_hint": "calm ambient lo-fi electronic study focus",
    },
    "Pure Discovery": {
        "builtin": True,
        "avoid_artists": [],
        "avoid_genres": [],
        "flow_mode": "harmonic",
        "size": 30,
        "fast": False,
        "local_only": False,
        "clean_only": False,
        "deep_cuts": True,
        "prompt_hint": "all new underground hidden gems matching my taste",
    },
    "Nostalgia Mode": {
        "builtin": True,
        "avoid_artists": [],
        "avoid_genres": [],
        "flow_mode": "mmr",
        "size": 30,
        "fast": True,
        "local_only": True,
        "clean_only": False,
        "deep_cuts": False,
        "prompt_hint": "2000s 2010s nostalgic favorites from my library",
    },
}

DEFAULT_AUTOMATION_SETTINGS = {
    "enabled": False,
    "day_of_week": "FRI",
    "time": "08:00",
    "prompt": "Fresh weekly discovery matching my active taste",
    "size": 30,
    "flow_mode": "harmonic",
    "fast": True,
    "clean_only": False,
    "deep_cuts": False,
    "target_playlist_id": "",
    "target_playlist_name": "Weekly AI Discovery",
    "run_stage1_sync": True,
    "run_stage2_refresh": True,
    "last_run_at": None,
    "last_status": None,
}

DEFAULT_USER_PREFERENCES = {
    "penalized_artists": {},
    "avoid_artists": [],
    "avoid_genres": [],
    "half_life_days": 120.0,
    "skip_penalty_enabled": True,
    "max_liked_songs": 500,
    "default_playlist_size": 30,
    "default_fast_mode": False,
    "default_public_export": False,
    "sync_likes_to_spotify": False,
    "active_provider": getattr(config, "DEFAULT_ACTIVE_PROVIDER", "spotify"),
    "llm_backend": "openvino",
    "llm_model_key": "auto",
    "llm_device": "auto",
    "ollama_url": getattr(config, "DEFAULT_OLLAMA_URL", "http://127.0.0.1:11434"),
    "ollama_model": getattr(config, "DEFAULT_OLLAMA_MODEL", "qwen3:8b"),
    "cluster_labels": {},
    "presets": dict(DEFAULT_BUILTIN_PRESETS),
    "automation": dict(DEFAULT_AUTOMATION_SETTINGS),
}

_CAMELOT_MAJOR = {0: 8, 1: 3, 2: 10, 3: 5, 4: 12, 5: 7, 6: 2, 7: 9, 8: 4, 9: 11, 10: 6, 11: 1}
_CAMELOT_MINOR = {0: 5, 1: 12, 2: 7, 3: 2, 4: 9, 5: 4, 6: 11, 7: 6, 8: 1, 9: 8, 10: 3, 11: 10}


def normalize_avoid_list(items):
    """Normalize a list or comma-separated string into deduplicated clean phrases."""
    if not items:
        return []
    raw_items = items.split(",") if isinstance(items, str) else list(items)
    cleaned = []
    seen = set()
    for item in raw_items:
        val = str(item or "").strip()
        val_low = val.lower()
        if val and val_low not in seen:
            seen.add(val_low)
            cleaned.append(val)
    return cleaned


def _clamp_pref_float(val, low, high, default):
    if isinstance(val, bool) or val is None:
        return default
    try:
        return round(max(low, min(high, float(val))), 4)
    except (TypeError, ValueError):
        return default


def _clamp_pref_int(val, low, high, default):
    if isinstance(val, bool) or val is None:
        return default
    try:
        return max(low, min(high, int(val)))
    except (TypeError, ValueError):
        return default


def _sanitize_presets(raw_presets):
    merged = {k: dict(v) for k, v in DEFAULT_BUILTIN_PRESETS.items()}
    if not isinstance(raw_presets, dict):
        return merged
    for name, cfg in raw_presets.items():
        clean_name = str(name or "").strip()[:60]
        if not clean_name or not isinstance(cfg, dict):
            continue
        merged[clean_name] = {
            "builtin": bool(clean_name in DEFAULT_BUILTIN_PRESETS and cfg.get("builtin", True)),
            "avoid_artists": normalize_avoid_list(cfg.get("avoid_artists")),
            "avoid_genres": normalize_avoid_list(cfg.get("avoid_genres")),
            "flow_mode": str(cfg.get("flow_mode") or "mmr"),
            "size": _clamp_pref_int(cfg.get("size"), 5, 100, 30),
            "fast": bool(cfg.get("fast", True)),
            "local_only": bool(cfg.get("local_only", False)),
            "clean_only": bool(cfg.get("clean_only", False)),
            "deep_cuts": bool(cfg.get("deep_cuts", False)),
            "prompt_hint": str(cfg.get("prompt_hint") or "").strip()[:240],
        }
    return merged


def _sanitize_automation(raw_auto, base=None):
    cur = dict(base or DEFAULT_AUTOMATION_SETTINGS)
    if not isinstance(raw_auto, dict):
        return cur
    if "enabled" in raw_auto:
        cur["enabled"] = bool(raw_auto["enabled"])
    valid_days = {"MON", "TUE", "WED", "THU", "FRI", "SAT", "SUN"}
    day_val = str(raw_auto.get("day_of_week") or cur["day_of_week"]).upper().strip()
    if day_val in valid_days:
        cur["day_of_week"] = day_val
    time_val = str(raw_auto.get("time") or cur["time"]).strip()
    if re.match(r"^(?:[01]\d|2[0-3]):[0-5]\d$", time_val):
        cur["time"] = time_val
    if "prompt" in raw_auto and isinstance(raw_auto["prompt"], str) and raw_auto["prompt"].strip():
        cur["prompt"] = raw_auto["prompt"].strip()[:300]
    if "size" in raw_auto:
        cur["size"] = _clamp_pref_int(raw_auto.get("size"), 5, 100, cur["size"])
    if "flow_mode" in raw_auto and isinstance(raw_auto["flow_mode"], str):
        cur["flow_mode"] = raw_auto["flow_mode"].strip() or "harmonic"
    for bkey in ("fast", "clean_only", "deep_cuts", "run_stage1_sync", "run_stage2_refresh"):
        if bkey in raw_auto and isinstance(raw_auto[bkey], bool):
            cur[bkey] = raw_auto[bkey]
    if "target_playlist_id" in raw_auto and raw_auto["target_playlist_id"] is not None:
        cur["target_playlist_id"] = str(raw_auto["target_playlist_id"]).strip()[:64]
    if "target_playlist_name" in raw_auto and isinstance(raw_auto["target_playlist_name"], str):
        cur["target_playlist_name"] = raw_auto["target_playlist_name"].strip()[:100] or "Weekly AI Discovery"
    if "last_run_at" in raw_auto:
        cur["last_run_at"] = raw_auto["last_run_at"]
    if "last_status" in raw_auto:
        cur["last_status"] = raw_auto["last_status"]
    return cur


def _sanitize_llm_url(raw_url, default=None):
    """Restrict local LLM base URLs to loopback or private RFC1918 LAN addresses (SSRF guard)."""
    fallback = str(default or getattr(config, "DEFAULT_OLLAMA_URL", "http://127.0.0.1:11434")).rstrip("/")
    val = str(raw_url or "").strip().rstrip("/")
    if not val or len(val) > 200:
        return fallback
    try:
        parsed = urllib.parse.urlparse(val)
    except ValueError:
        return fallback
    if parsed.scheme.lower() not in ("http", "https"):
        return fallback
    if parsed.username or parsed.password:
        return fallback
    host = (parsed.hostname or "").strip().lower()
    if not host:
        return fallback
    if host not in ("localhost", "::1", "127.0.0.1"):
        try:
            ip = ipaddress.ip_address(host)
        except ValueError:
            return fallback
        if not (
            ip.is_loopback
            or (ip.is_private and not ip.is_link_local and not ip.is_unspecified and not ip.is_multicast)
        ):
            return fallback
    return val


def _valid_llm_model_keys():
    keys = {"auto"}
    for item in getattr(config, "LLM_MODEL_CATALOG", []) or []:
        if isinstance(item, dict) and item.get("key"):
            keys.add(str(item["key"]))
    return keys


def get_effective_preferences():
    """Return validated user preferences merged over config.py and module defaults."""
    prefs = {
        "penalized_artists": dict(getattr(config, "PENALIZED_ARTISTS", {}) or {}),
        "avoid_artists": normalize_avoid_list(getattr(config, "DEFAULT_AVOID_ARTISTS", [])),
        "avoid_genres": normalize_avoid_list(getattr(config, "DEFAULT_AVOID_GENRES", [])),
        "half_life_days": float(getattr(config, "EXTENDED_HISTORY_HALF_LIFE_DAYS", 120.0)),
        "skip_penalty_enabled": bool(getattr(config, "SKIP_PENALTY_ENABLED", True)),
        "max_liked_songs": int(getattr(config, "MAX_LIKED_SONGS", 500)),
        "default_playlist_size": int(getattr(config, "DEFAULT_PLAYLIST_SIZE", 30)),
        "default_fast_mode": False,
        "default_public_export": False,
        "sync_likes_to_spotify": False,
        "active_provider": getattr(config, "DEFAULT_ACTIVE_PROVIDER", "spotify"),
        "embed_backend": getattr(config, "DEFAULT_EMBED_BACKEND", "minilm"),
        "playlist_cooccurrence_weight": float(getattr(config, "DEFAULT_PLAYLIST_COOCCURRENCE_WEIGHT", 0.12)),
        "llm_backend": "openvino",
        "llm_model_key": "auto",
        "llm_device": "auto",
        "ollama_url": getattr(config, "DEFAULT_OLLAMA_URL", "http://127.0.0.1:11434"),
        "ollama_model": getattr(config, "DEFAULT_OLLAMA_MODEL", "qwen3:8b"),
        "cluster_labels": {},
        "presets": _sanitize_presets(None),
        "automation": dict(DEFAULT_AUTOMATION_SETTINGS),
    }
    raw = load_user_preferences()
    if not isinstance(raw, dict):
        return prefs

    if isinstance(raw.get("penalized_artists"), dict):
        merged_pen = {}
        for k, v in raw["penalized_artists"].items():
            clean_k = str(k or "").strip()
            if clean_k:
                merged_pen[clean_k] = _clamp_pref_float(v, 0.0, 2.0, 0.05)
        prefs["penalized_artists"] = merged_pen

    if "avoid_artists" in raw:
        prefs["avoid_artists"] = normalize_avoid_list(raw.get("avoid_artists"))
    if "avoid_genres" in raw:
        prefs["avoid_genres"] = normalize_avoid_list(raw.get("avoid_genres"))
    prefs["half_life_days"] = _clamp_pref_float(raw.get("half_life_days"), 7.0, 1825.0, prefs["half_life_days"])
    if isinstance(raw.get("skip_penalty_enabled"), bool):
        prefs["skip_penalty_enabled"] = raw["skip_penalty_enabled"]
    prefs["max_liked_songs"] = _clamp_pref_int(raw.get("max_liked_songs"), 50, 5000, prefs["max_liked_songs"])
    prefs["default_playlist_size"] = _clamp_pref_int(
        raw.get("default_playlist_size"), 5, 100, prefs["default_playlist_size"]
    )
    if isinstance(raw.get("default_fast_mode"), bool):
        prefs["default_fast_mode"] = raw["default_fast_mode"]
    if isinstance(raw.get("default_public_export"), bool):
        prefs["default_public_export"] = raw["default_public_export"]
    if isinstance(raw.get("sync_likes_to_spotify"), bool):
        prefs["sync_likes_to_spotify"] = raw["sync_likes_to_spotify"]
    if str(raw.get("active_provider") or "").strip().lower() in ("spotify", "ytmusic", "unified"):
        prefs["active_provider"] = str(raw["active_provider"]).strip().lower()
    if str(raw.get("embed_backend") or "").strip().lower() in ("minilm", "clap"):
        prefs["embed_backend"] = str(raw["embed_backend"]).strip().lower()
    if "playlist_cooccurrence_weight" in raw:
        prefs["playlist_cooccurrence_weight"] = _clamp_pref_float(
            raw.get("playlist_cooccurrence_weight"), 0.0, 0.35, prefs["playlist_cooccurrence_weight"]
        )
    if str(raw.get("llm_backend") or "").strip().lower() in ("openvino", "ollama", "openai_local"):
        prefs["llm_backend"] = str(raw["llm_backend"]).strip().lower()
    mkey = str(raw.get("llm_model_key") or "").strip()
    if mkey in _valid_llm_model_keys():
        prefs["llm_model_key"] = mkey
    dev_val = str(raw.get("llm_device") or "").strip()
    if dev_val.lower() == "auto":
        prefs["llm_device"] = "auto"
    elif dev_val.upper() in ("GPU", "CPU", "NPU"):
        prefs["llm_device"] = dev_val.upper()
    if "ollama_url" in raw:
        prefs["ollama_url"] = _sanitize_llm_url(raw.get("ollama_url"), default=prefs["ollama_url"])
    omodel = str(raw.get("ollama_model") or "").strip()[:120]
    if omodel:
        prefs["ollama_model"] = omodel
    if isinstance(raw.get("cluster_labels"), dict):
        prefs["cluster_labels"] = {
            str(k): str(v).strip()[:80]
            for k, v in raw["cluster_labels"].items()
            if str(v or "").strip()
        }
    if "presets" in raw:
        prefs["presets"] = _sanitize_presets(raw.get("presets"))
    if "automation" in raw:
        prefs["automation"] = _sanitize_automation(raw.get("automation"), base=prefs["automation"])
    return prefs


def save_user_preferences(updates):
    """Validate, merge, and atomically save user preferences to user_preferences.json."""
    current = get_effective_preferences()
    if isinstance(updates, dict):
        if "penalized_artists" in updates and isinstance(updates["penalized_artists"], dict):
            clean_pen = {}
            for k, v in updates["penalized_artists"].items():
                ck = str(k or "").strip()
                if ck:
                    clean_pen[ck] = _clamp_pref_float(v, 0.0, 2.0, 0.05)
            current["penalized_artists"] = clean_pen
        if "avoid_artists" in updates:
            current["avoid_artists"] = normalize_avoid_list(updates["avoid_artists"])
        if "avoid_genres" in updates:
            current["avoid_genres"] = normalize_avoid_list(updates["avoid_genres"])
        if "half_life_days" in updates:
            current["half_life_days"] = _clamp_pref_float(
                updates["half_life_days"], 7.0, 1825.0, current["half_life_days"]
            )
        if "skip_penalty_enabled" in updates and isinstance(updates["skip_penalty_enabled"], bool):
            current["skip_penalty_enabled"] = updates["skip_penalty_enabled"]
        if "max_liked_songs" in updates:
            current["max_liked_songs"] = _clamp_pref_int(
                updates["max_liked_songs"], 50, 5000, current["max_liked_songs"]
            )
        if "default_playlist_size" in updates:
            current["default_playlist_size"] = _clamp_pref_int(
                updates["default_playlist_size"], 5, 100, current["default_playlist_size"]
            )
        if "default_fast_mode" in updates and isinstance(updates["default_fast_mode"], bool):
            current["default_fast_mode"] = updates["default_fast_mode"]
        if "default_public_export" in updates and isinstance(updates["default_public_export"], bool):
            current["default_public_export"] = updates["default_public_export"]
        if "sync_likes_to_spotify" in updates and isinstance(updates["sync_likes_to_spotify"], bool):
            current["sync_likes_to_spotify"] = updates["sync_likes_to_spotify"]
        if "active_provider" in updates:
            prov = str(updates["active_provider"] or "").strip().lower()
            if prov in ("spotify", "ytmusic", "unified"):
                current["active_provider"] = prov
        if "embed_backend" in updates:
            eb = str(updates["embed_backend"] or "").strip().lower()
            if eb in ("minilm", "clap"):
                current["embed_backend"] = eb
        if "playlist_cooccurrence_weight" in updates:
            current["playlist_cooccurrence_weight"] = _clamp_pref_float(
                updates["playlist_cooccurrence_weight"], 0.0, 0.35, current["playlist_cooccurrence_weight"]
            )
        if "llm_backend" in updates:
            lb = str(updates["llm_backend"] or "").strip().lower()
            if lb in ("openvino", "ollama", "openai_local"):
                current["llm_backend"] = lb
        if "llm_model_key" in updates:
            mk = str(updates["llm_model_key"] or "").strip()
            if mk in _valid_llm_model_keys():
                current["llm_model_key"] = mk
        if "llm_device" in updates:
            dv = str(updates["llm_device"] or "").strip()
            if dv.lower() == "auto":
                current["llm_device"] = "auto"
            elif dv.upper() in ("GPU", "CPU", "NPU"):
                current["llm_device"] = dv.upper()
        if "ollama_url" in updates:
            current["ollama_url"] = _sanitize_llm_url(updates["ollama_url"], default=current["ollama_url"])
        if "ollama_model" in updates:
            om = str(updates["ollama_model"] or "").strip()[:120]
            if om:
                current["ollama_model"] = om
        if "cluster_labels" in updates and isinstance(updates["cluster_labels"], dict):
            current["cluster_labels"] = {
                str(k): str(v).strip()[:80]
                for k, v in updates["cluster_labels"].items()
                if str(v or "").strip()
            }
        if "presets" in updates and isinstance(updates["presets"], dict):
            current["presets"] = _sanitize_presets(updates["presets"])
        if "automation" in updates and isinstance(updates["automation"], dict):
            current["automation"] = _sanitize_automation(updates["automation"], base=current["automation"])

    prefs_path = getattr(config, "USER_PREFS_FILE", os.path.join(config.BASE_DIR, "user_preferences.json"))
    atomic_write_json(prefs_path, current, indent=2)
    return current


def _extract_artist_names_lower(artists_value):
    if not artists_value:
        return set()
    if isinstance(artists_value, str):
        return {p.strip().lower() for p in artists_value.split(",") if p.strip()}
    if isinstance(artists_value, (list, tuple, set)):
        out = set()
        for item in artists_value:
            if isinstance(item, dict):
                name = str(item.get("name") or "").strip().lower()
                if name:
                    out.add(name)
            elif item is not None:
                name = str(item).strip().lower()
                if name:
                    out.add(name)
        return out
    return set()


def matches_avoided_artist(artists_value, avoid_artists):
    """Exact case-insensitive artist match (no substring false positives)."""
    avoid_low = {a.lower() for a in normalize_avoid_list(avoid_artists)}
    if not avoid_low:
        return False
    track_artists = _extract_artist_names_lower(artists_value)
    return bool(avoid_low & track_artists)


def matches_avoided_genre(genres_value, avoid_genres):
    """Word/phrase-boundary genre matching across comma-separated strings or genre lists."""
    avoid_list = normalize_avoid_list(avoid_genres)
    if not avoid_list or not genres_value:
        return False

    if isinstance(genres_value, str):
        track_genres = [g.strip().lower() for g in genres_value.split(",") if g.strip()]
    elif isinstance(genres_value, (list, tuple, set)):
        track_genres = [str(g).strip().lower() for g in genres_value if str(g or "").strip()]
    else:
        return False

    if not track_genres:
        return False

    patterns = [
        re.compile(rf"(?<![A-Za-z0-9]){re.escape(ag.lower())}(?![A-Za-z0-9])", re.IGNORECASE)
        for ag in avoid_list
    ]
    for tg in track_genres:
        for pat in patterns:
            if pat.search(tg):
                return True
    return False


def spotify_key_to_camelot(key, mode):
    """Convert Spotify pitch class (0-11) and mode (0=minor, 1=major) to Camelot (num, letter)."""
    if isinstance(key, bool) or isinstance(mode, bool) or key is None or mode is None:
        return None
    try:
        k = int(key)
        m = int(mode)
    except (TypeError, ValueError):
        return None
    if k < 0 or k > 11 or m not in (0, 1):
        return None
    num = _CAMELOT_MAJOR[k] if m == 1 else _CAMELOT_MINOR[k]
    letter = "B" if m == 1 else "A"
    return (num, letter)


def format_camelot_label(key, mode):
    cam = spotify_key_to_camelot(key, mode)
    return f"{cam[0]}{cam[1]}" if cam else None


def camelot_distance(c1, c2):
    """Compute harmonic wheel distance between two Camelot tuples."""
    if c1 is None or c2 is None:
        return 3
    n1, l1 = c1
    n2, l2 = c2
    wheel_diff = min((n1 - n2) % 12, (n2 - n1) % 12)
    if wheel_diff == 0:
        return 0 if l1 == l2 else 1
    if wheel_diff == 1 and l1 == l2:
        return 1
    return wheel_diff + (1 if l1 != l2 else 0)


def _track_energy(tr):
    val = tr.get("energy")
    return float(val) if isinstance(val, (int, float)) and not isinstance(val, bool) else 0.5


def _track_bpm(tr):
    val = tr.get("bpm")
    return float(val) if isinstance(val, (int, float)) and not isinstance(val, bool) else 120.0


def _track_primary_artist(tr):
    art = str(tr.get("artist") or tr.get("artists") or "")
    return art.split(",")[0].strip().lower() if art else "unknown"


def sequence_playlist_tracks(tracks, flow_mode="mmr", seed=None):
    """Re-order playlist tracks using acoustic energy, BPM, or Camelot harmonic transitions."""
    items = [dict(t) for t in (tracks or [])]
    n = len(items)
    if n <= 1 or not flow_mode or flow_mode == "mmr":
        return items

    if flow_mode == "energy_ramp":
        return sorted(items, key=lambda t: (_track_energy(t), _track_bpm(t)))

    if flow_mode == "chill_descent":
        return sorted(items, key=lambda t: (_track_energy(t), _track_bpm(t)), reverse=True)

    if flow_mode == "warmup_peak_cooldown":
        def intensity(t):
            norm_bpm = min(1.0, max(0.0, (_track_bpm(t) - 60.0) / 120.0))
            return 0.65 * _track_energy(t) + 0.35 * norm_bpm

        ranked = sorted(items, key=intensity)
        if n < 3:
            return ranked
        n_warm = max(1, round(n * 0.30))
        n_cool = max(1, round(n * 0.30))
        n_peak = max(1, n - n_warm - n_cool)
        lower_pool = ranked[: n - n_peak]
        peak_pool = ranked[n - n_peak :]

        warmup = lower_pool[0::2]
        cooldown_asc = lower_pool[1::2]
        cooldown = list(reversed(cooldown_asc))

        # Shape peak as rising to climax then easing slightly into cooldown
        peak_left = peak_pool[0::2]
        peak_right = list(reversed(peak_pool[1::2]))
        return warmup + peak_left + peak_right + cooldown

    if flow_mode == "harmonic":
        remaining = list(items)
        ordered = [remaining.pop(0)]
        while remaining:
            cur = ordered[-1]
            c_cur = spotify_key_to_camelot(cur.get("key"), cur.get("mode"))
            b_cur = _track_bpm(cur)
            e_cur = _track_energy(cur)
            pa_cur = _track_primary_artist(cur)

            def cost(cand):
                c_cand = spotify_key_to_camelot(cand.get("key"), cand.get("mode"))
                same_art = 25.0 if _track_primary_artist(cand) == pa_cur else 0.0
                return (
                    camelot_distance(c_cur, c_cand) * 15.0
                    + abs(b_cur - _track_bpm(cand)) * 0.5
                    + abs(e_cur - _track_energy(cand)) * 20.0
                    + same_art
                )

            best_idx = min(range(len(remaining)), key=lambda idx: cost(remaining[idx]))
            ordered.append(remaining.pop(best_idx))
        return ordered

    if flow_mode == "shuffle":
        rng = random.Random(seed)
        pool = list(items)
        rng.shuffle(pool)
        ordered = []
        while pool:
            prev_art = _track_primary_artist(ordered[-1]) if ordered else None
            pick_idx = 0
            if prev_art is not None:
                for idx, cand in enumerate(pool):
                    if _track_primary_artist(cand) != prev_art:
                        pick_idx = idx
                        break
            ordered.append(pool.pop(pick_idx))
        return ordered

    return items


def _slugify_playlist_id(name):
    slug = re.sub(r"[^A-Za-z0-9]+", "-", str(name or "playlist")).strip("-").lower()
    return (slug[:40] or "playlist")


def save_playlist_to_archive(playlist_payload, archive_dir=None):
    """Save a generated playlist into playlist_history/ with a unique ID and return its ID."""
    target_dir = archive_dir or getattr(
        config, "PLAYLIST_HISTORY_DIR", os.path.join(config.BASE_DIR, "03_Agentic_Engine", "playlist_history")
    )
    os.makedirs(target_dir, exist_ok=True)
    ts_part = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    slug = _slugify_playlist_id(playlist_payload.get("playlist_name"))
    candidate_id = f"{ts_part}_{slug}"
    suffix = 2
    while os.path.exists(os.path.join(target_dir, f"{candidate_id}.json")):
        candidate_id = f"{ts_part}_{slug}_{suffix}"
        suffix += 1

    payload = dict(playlist_payload)
    payload["id"] = candidate_id
    atomic_write_json(os.path.join(target_dir, f"{candidate_id}.json"), payload, indent=2)
    return candidate_id


def _summarize_playlist_entry(pid, data):
    tracks = data.get("tracks") or []
    bpms = [float(t["bpm"]) for t in tracks if isinstance(t.get("bpm"), (int, float))]
    local_count = sum(1 for t in tracks if t.get("source") == "Local Library")
    return {
        "id": pid,
        "playlist_name": data.get("playlist_name", "AI Playlist"),
        "playlist_description": data.get("playlist_description", ""),
        "prompt": data.get("prompt", ""),
        "generated_at": data.get("generated_at", ""),
        "track_count": len(tracks),
        "avg_bpm": round(sum(bpms) / len(bpms), 1) if bpms else None,
        "local_count": local_count,
        "new_count": len(tracks) - local_count,
        "flow_mode": data.get("flow_mode", "mmr"),
    }


def list_archived_playlists(archive_dir=None):
    """Return metadata summaries for archived playlists, sorted newest first."""
    target_dir = archive_dir or getattr(
        config, "PLAYLIST_HISTORY_DIR", os.path.join(config.BASE_DIR, "03_Agentic_Engine", "playlist_history")
    )
    if not os.path.isdir(target_dir):
        return []
    summaries = []
    for fn in os.listdir(target_dir):
        if not fn.endswith(".json"):
            continue
        pid = fn[:-5]
        if not _ARCHIVE_ID_RE.match(pid):
            continue
        data = load_json_safe(os.path.join(target_dir, fn))
        if isinstance(data, dict):
            summaries.append(_summarize_playlist_entry(pid, data))
    summaries.sort(key=lambda item: (item.get("generated_at") or "", item["id"]), reverse=True)
    return summaries


def load_archived_playlist(playlist_id, archive_dir=None):
    """Load an archived playlist safely after validating its ID."""
    if not isinstance(playlist_id, str) or not _ARCHIVE_ID_RE.match(playlist_id):
        return None
    target_dir = archive_dir or getattr(
        config, "PLAYLIST_HISTORY_DIR", os.path.join(config.BASE_DIR, "03_Agentic_Engine", "playlist_history")
    )
    path = os.path.join(target_dir, f"{playlist_id}.json")
    data = load_json_safe(path)
    return data if isinstance(data, dict) else None


def delete_archived_playlist(playlist_id, archive_dir=None):
    """Delete an archived playlist safely after validating its ID."""
    if not isinstance(playlist_id, str) or not _ARCHIVE_ID_RE.match(playlist_id):
        return False
    target_dir = archive_dir or getattr(
        config, "PLAYLIST_HISTORY_DIR", os.path.join(config.BASE_DIR, "03_Agentic_Engine", "playlist_history")
    )
    path = os.path.join(target_dir, f"{playlist_id}.json")
    if os.path.exists(path):
        os.remove(path)
        return True
    return False


def _resolve_track_export_url(t):
    """Resolve the best Spotify or YouTube Music URL for exporting a track row."""
    if not isinstance(t, dict):
        return ""
    sp_url = str(t.get("spotify_url") or "").strip()
    if sp_url.startswith("https://") or sp_url.startswith("http://"):
        return sp_url
    yt_url = str(t.get("youtube_url") or "").strip()
    if yt_url.startswith("https://") or yt_url.startswith("http://"):
        return yt_url

    sp_id = t.get("spotify_id") if is_valid_spotify_id(t.get("spotify_id")) else None
    if sp_id:
        return f"https://open.spotify.com/track/{sp_id}"

    tid = str(t.get("id") or "").strip()
    vid = extract_ytm_video_id(t.get("video_id") or tid)
    if vid:
        return f"https://music.youtube.com/watch?v={vid}"
    if is_valid_spotify_id(tid):
        return f"https://open.spotify.com/track/{tid}"
    return f"https://open.spotify.com/track/{tid}" if tid else ""


def format_playlist_csv(playlist_data):
    """Serialize a playlist payload into CSV text."""
    buf = io.StringIO()
    writer = csv.writer(buf, lineterminator="\n")
    writer.writerow(["#", "Name", "Artist", "Source", "BPM", "Energy", "Valence", "Key", "Track URL", "Track ID"])
    for idx, t in enumerate(playlist_data.get("tracks") or [], start=1):
        tid = t.get("id", "")
        cam = format_camelot_label(t.get("key"), t.get("mode")) or ""
        url = _resolve_track_export_url(t)
        writer.writerow(
            [
                idx,
                t.get("name", ""),
                t.get("artist", ""),
                t.get("source", ""),
                t.get("bpm", ""),
                t.get("energy", ""),
                t.get("valence", ""),
                cam,
                url,
                tid,
            ]
        )
    return buf.getvalue()


def format_playlist_m3u(playlist_data):
    """Serialize a playlist payload into Extended M3U format."""
    name = str(playlist_data.get("playlist_name") or "AI Playlist").replace("\n", " ")
    lines = ["#EXTM3U", f"#PLAYLIST:{name}"]
    for t in playlist_data.get("tracks") or []:
        artist = str(t.get("artist") or "Unknown").replace("\n", " ")
        title = str(t.get("name") or "Unknown").replace("\n", " ")
        url = _resolve_track_export_url(t)
        lines.append(f"#EXTINF:-1,{artist} - {title}")
        if url:
            lines.append(url)
    return "\n".join(lines) + "\n"


def format_playlist_markdown(playlist_data):
    """Serialize a playlist payload into Markdown."""
    name = playlist_data.get("playlist_name") or "AI Playlist"
    desc = playlist_data.get("playlist_description") or ""
    prompt = playlist_data.get("prompt") or ""
    tracks = playlist_data.get("tracks") or []
    lines = [f"# {name}", ""]
    if prompt:
        lines.append(f"**Prompt:** {prompt}  ")
    if desc:
        lines.append(f"**Vibe:** {desc}  ")
    lines.append(f"**Total Tracks:** {len(tracks)}")
    lines.extend(["", "| # | Track | Artist | Source | BPM | Energy | Valence | Key | Link |", "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"])
    for idx, t in enumerate(tracks, start=1):
        url = _resolve_track_export_url(t)
        link_lbl = "YT Music" if "music.youtube.com" in url else "Spotify"
        link = f"[{link_lbl}]({url})" if url else ""
        cam = format_camelot_label(t.get("key"), t.get("mode")) or "-"
        t_name = str(t.get("name") or "").replace("|", "\\|")
        t_art = str(t.get("artist") or "").replace("|", "\\|")
        lines.append(
            f"| {idx} | {t_name} | {t_art} | {t.get('source', '')} | "
            f"{t.get('bpm', '-')} | {t.get('energy', '-')} | {t.get('valence', '-')} | {cam} | {link} |"
        )
    return "\n".join(lines) + "\n"


def _file_mtime(path):
    try:
        return os.stat(path).st_mtime
    except OSError:
        return None


def compute_pipeline_freshness(base_dir=None):
    """Detect missing or stale artifacts across Stage 1 (Spotify/YTM), Stage 2, and Taste Map."""
    root = base_dir or config.BASE_DIR
    ext_stats = os.path.join(root, "01_Data_Extractor", "extended_history_stats.json")
    s1_profile = os.path.join(root, "01_Data_Extractor", "user_music_profile.json")
    ytm_profile = getattr(
        config, "YTMUSIC_PROFILE_FILE", os.path.join(root, "01_Data_Extractor", "ytmusic_profile.json")
    )
    if base_dir and not ytm_profile.startswith(os.path.abspath(root)):
        ytm_profile = os.path.join(root, "01_Data_Extractor", "ytmusic_profile.json")
    s2_vecs = os.path.join(root, "02_Vector_Core", "taste_vectors.npz")
    s2_sum = os.path.join(root, "02_Vector_Core", "taste_profile_summary.json")
    map_data = os.path.join(root, "02_Vector_Core", "taste_map_data.json")
    map_html = os.path.join(root, "taste_map.html")

    ext_t = _file_mtime(ext_stats)
    s1_t = _file_mtime(s1_profile)
    ytm_t = _file_mtime(ytm_profile)
    s2_vec_t = _file_mtime(s2_vecs)
    s2_sum_t = _file_mtime(s2_sum)
    map_data_t = _file_mtime(map_data)
    map_html_t = _file_mtime(map_html)

    stage1_spotify_exists = s1_t is not None
    stage1_ytm_exists = ytm_t is not None
    stage1_exists = stage1_spotify_exists or stage1_ytm_exists
    stage2_exists = (s2_vec_t is not None) and (s2_sum_t is not None)
    map_exists = (map_data_t is not None) or (map_html_t is not None)

    stale_reasons = []
    stage2_stale = False
    if stage2_exists:
        newest_s1 = max([t for t in (s1_t, ytm_t) if t is not None], default=None)
        if newest_s1 is not None and (newest_s1 > s2_vec_t or newest_s1 > s2_sum_t):
            stage2_stale = True
            stale_reasons.append("Stage 1 profile updated more recently than Stage 2 vectors/clusters.")
        if ext_t is not None and ext_t > s2_sum_t:
            stage2_stale = True
            stale_reasons.append("Extended History stats updated more recently than Stage 2 clusters.")
        s2_sum_data = load_json_safe(s2_sum)
        if isinstance(s2_sum_data, dict):
            built_prov = str(((s2_sum_data.get("session_filters") or {}).get("active_provider")) or "").strip().lower()
            cur_prov = str(get_effective_preferences().get("active_provider", "spotify")).strip().lower()
            if (
                built_prov in ("spotify", "ytmusic", "unified")
                and cur_prov in ("spotify", "ytmusic", "unified")
                and built_prov != cur_prov
            ):
                stage2_stale = True
                stale_reasons.append(f"Active provider is '{cur_prov}', but Stage 2 vectors were built for '{built_prov}'.")

    map_stale = False
    if map_exists and s2_sum_t is not None:
        newest_map = max([t for t in (map_data_t, map_html_t) if t is not None], default=0.0)
        if s2_sum_t > newest_map or map_data_t is None:
            map_stale = True
            stale_reasons.append("Stage 2 taste clusters updated more recently than Taste Map data.")

    return {
        "stage1_exists": stage1_exists,
        "stage1_spotify_exists": stage1_spotify_exists,
        "stage1_ytm_exists": stage1_ytm_exists,
        "stage2_exists": stage2_exists,
        "map_exists": map_exists,
        "stage2_stale": stage2_stale,
        "map_stale": map_stale,
        "stale_reasons": stale_reasons,
    }


# =======================================================
# FEEDBACK MEMORY, PLAYBACK, PLAYLIST REPLACE & UPLOAD
# =======================================================


def _feedback_path():
    return getattr(config, "USER_FEEDBACK_FILE", os.path.join(config.BASE_DIR, "user_feedback.json"))


def load_user_feedback():
    """Load liked and disliked track feedback dictionary."""
    raw = load_json_safe(_feedback_path(), default={})
    if not isinstance(raw, dict):
        raw = {}
    liked = raw.get("liked") if isinstance(raw.get("liked"), dict) else {}
    disliked = raw.get("disliked") if isinstance(raw.get("disliked"), dict) else {}
    return {"liked": liked, "disliked": disliked}


def save_user_feedback(data):
    """Atomically persist user track feedback dictionary."""
    clean = {
        "liked": data.get("liked") if isinstance(data.get("liked"), dict) else {},
        "disliked": data.get("disliked") if isinstance(data.get("disliked"), dict) else {},
    }
    atomic_write_json(_feedback_path(), clean, indent=2)
    return clean


def record_track_feedback(track, rating):
    """Record 'like', 'dislike', or 'clear' feedback for a track."""
    if not isinstance(track, dict):
        raise ValueError("Track payload must be an object.")
    tid = str(track.get("id") or "").strip()
    name = str(track.get("name") or "").strip()
    artist = str(track.get("artist") or track.get("artists") or "").strip()
    fp = get_track_fingerprint(name, artist)
    key = tid or fp
    if not key:
        raise ValueError("Track requires an id or name/artist.")

    fb = load_user_feedback()
    fb["liked"].pop(key, None)
    fb["disliked"].pop(key, None)
    if tid:
        fb["liked"].pop(tid, None)
        fb["disliked"].pop(tid, None)

    now_iso = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    entry = {
        "id": tid,
        "name": name or "Unknown",
        "artist": artist or "Unknown",
        "fingerprint": fp,
        "updated_at": now_iso,
    }

    if rating == "like":
        fb["liked"][key] = entry
    elif rating == "dislike":
        fb["disliked"][key] = entry
    elif rating != "clear":
        raise ValueError(f"Invalid rating '{rating}'. Expected like, dislike, or clear.")

    return save_user_feedback(fb)


def get_disliked_sets():
    """Return (disliked_ids, disliked_fingerprints) sets for instant filtering."""
    fb = load_user_feedback()
    ids = set()
    fps = set()
    for key, rec in (fb.get("disliked") or {}).items():
        if not isinstance(rec, dict):
            continue
        if rec.get("id"):
            ids.add(str(rec["id"]))
        elif is_valid_spotify_id(str(key)):
            ids.add(str(key))
        fp = rec.get("fingerprint") or get_track_fingerprint(rec.get("name", ""), rec.get("artist", ""))
        if fp:
            fps.add(fp)
    return ids, fps


def get_liked_ids():
    """Return set of liked track IDs."""
    fb = load_user_feedback()
    ids = set()
    for key, rec in (fb.get("liked") or {}).items():
        if isinstance(rec, dict) and rec.get("id"):
            ids.add(str(rec["id"]))
        elif is_valid_spotify_id(str(key)):
            ids.add(str(key))
    return ids


def get_currently_playing_track(sp):
    """Fetch the user's currently playing Spotify track safely."""
    resp = None
    try:
        if hasattr(sp, "current_user_playing_track"):
            resp = sp.current_user_playing_track()
        elif hasattr(sp, "currently_playing"):
            resp = sp.currently_playing()
        else:
            resp = sp._get("me/player/currently-playing")
    except Exception as exc:
        msg = str(exc)
        if "404" in msg or "204" in msg:
            return None
        raise

    if not isinstance(resp, dict):
        return None
    item = resp.get("item")
    if not isinstance(item, dict) or item.get("type") == "episode" or item.get("is_local"):
        return None

    tid = item.get("id")
    name = item.get("name")
    if not tid or not name:
        return None

    artists = [a.get("name") for a in (item.get("artists") or []) if isinstance(a, dict) and a.get("name")]
    album = item.get("album") or {}
    images = album.get("images") or []
    return {
        "id": str(tid),
        "name": name,
        "artist": ", ".join(artists) if artists else "Unknown",
        "album": album.get("name", ""),
        "image": images[0].get("url") if images and isinstance(images[0], dict) else None,
        "explicit": bool(item.get("explicit")),
        "is_playing": bool(resp.get("is_playing", False)),
        "progress_ms": resp.get("progress_ms", 0),
        "duration_ms": item.get("duration_ms", 0),
        "spotify_url": f"https://open.spotify.com/track/{tid}",
    }


def list_user_owned_playlists(sp, limit=50):
    """List personal playlists owned by the authenticated user."""
    user = sp.current_user()
    uid = user.get("id")
    owned = []
    resp = sp.current_user_playlists(limit=min(50, int(limit)))
    while resp:
        for pl in resp.get("items") or []:
            if not isinstance(pl, dict):
                continue
            if (pl.get("owner") or {}).get("id") == uid and pl.get("id"):
                items_meta = pl.get("items") or pl.get("tracks") or {}
                owned.append(
                    {
                        "id": pl["id"],
                        "name": pl.get("name", "Untitled"),
                        "tracks_total": items_meta.get("total", 0) if isinstance(items_meta, dict) else 0,
                        "public": bool(pl.get("public", False)),
                        "url": (pl.get("external_urls") or {}).get("spotify", f"https://open.spotify.com/playlist/{pl['id']}"),
                    }
                )
        if resp.get("next") and len(owned) < 200:
            resp = sp.next(resp)
        else:
            break
    return owned


def replace_spotify_playlist_items(sp, playlist_id, track_uris, name=None, description=None):
    """Replace all items in an existing Spotify playlist via 2026 /items endpoint."""
    uris = [u for u in track_uris if u]
    first_chunk = uris[:100]
    rest = uris[100:]

    if name is not None or description is not None:
        clean_name, clean_desc = sanitize_playlist_meta(name or "AI Playlist", description or "")
        try:
            if hasattr(sp, "playlist_change_details"):
                sp.playlist_change_details(playlist_id, name=clean_name, description=clean_desc)
            else:
                sp._put(f"playlists/{playlist_id}", payload={"name": clean_name, "description": clean_desc})
        except Exception:
            pass

    try:
        sp._put(f"playlists/{playlist_id}/items", payload={"uris": first_chunk})
    except Exception as exc:
        msg = str(exc)
        status = getattr(exc, "http_status", None)
        if status in (404, 405) or "404" in msg or "405" in msg or isinstance(exc, AttributeError):
            if hasattr(sp, "playlist_replace_items"):
                sp.playlist_replace_items(playlist_id, first_chunk)
            else:
                sp.user_playlist_replace_tracks(sp.current_user()["id"], playlist_id, first_chunk)
        else:
            raise

    if rest:
        add_items_to_spotify_playlist(sp, playlist_id, rest, batch_size=100)


def find_local_plotly_js():
    """Locate a local plotly.min.js bundle from web_ui/vendor or the installed plotly package."""
    vendor_path = os.path.join(config.BASE_DIR, "web_ui", "vendor", "plotly.min.js")
    if os.path.isfile(vendor_path):
        return vendor_path
    spec = importlib.util.find_spec("plotly")
    if spec and spec.origin:
        pkg_dir = os.path.dirname(spec.origin)
        pkg_plotly = os.path.join(pkg_dir, "package_data", "plotly.min.js")
        if os.path.isfile(pkg_plotly):
            return pkg_plotly
    return None


def extract_uploaded_history_bytes(filename, raw_bytes, dest_dir=None):
    """
    Safely extract Spotify Extended Streaming History .zip or .json uploads into dest_dir.
    Neutralizes Zip-Slip paths, ignores video JSONs, and verifies audio JSON array structure.
    """
    target_dir = dest_dir or os.path.join(config.BASE_DIR, "extended_history")
    os.makedirs(target_dir, exist_ok=True)
    clean_upload_name = os.path.basename(str(filename or "")).strip()
    lower = clean_upload_name.lower()

    saved_files = []
    skipped_files = []

    MAX_MEMBER_BYTES = 150 * 1024 * 1024
    MAX_TOTAL_BYTES = 500 * 1024 * 1024
    MAX_MEMBERS = 200

    def _safe_save_json_member(member_name, data_bytes):
        base = os.path.basename(str(member_name or "").replace("\\", "/")).strip()
        base_low = base.lower()
        if not base or not base_low.endswith(".json") or base.startswith("."):
            if base:
                skipped_files.append(base)
            return
        if base_low.startswith("streaming_history_video"):
            skipped_files.append(base)
            return
        is_spotify_hist = "streaming_history_audio" in base_low or "endsong" in base_low
        is_takeout_hist = base_low in ("watch-history.json", "watch_history.json")
        if not (is_spotify_hist or is_takeout_hist):
            # Allow generic .json upload if explicitly uploaded directly, skip random zip metadata JSONs
            if lower.endswith(".zip"):
                skipped_files.append(base)
                return

        try:
            parsed = json.loads(data_bytes.decode("utf-8"))
            if not isinstance(parsed, list):
                skipped_files.append(base)
                return
        except Exception:
            skipped_files.append(base)
            return

        safe_name = re.sub(r"[^A-Za-z0-9._-]+", "_", base)
        out_path = os.path.join(target_dir, safe_name)
        tmp_path = f"{out_path}.{os.getpid()}.{threading.get_ident()}.{uuid.uuid4().hex[:6]}.tmp"
        try:
            with open(tmp_path, "wb") as f:
                f.write(data_bytes)
                f.flush()
                os.fsync(f.fileno())
            _replace_with_retry(tmp_path, out_path)
        except Exception:
            if os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass
            raise
        saved_files.append(safe_name)

    if lower.endswith(".zip"):
        try:
            cumulative_bytes = 0
            processed_members = 0
            with zipfile.ZipFile(io.BytesIO(raw_bytes)) as zf:
                for info in zf.infolist():
                    posix_name = str(info.filename or "").replace("\\", "/")
                    parts = [p for p in posix_name.split("/") if p]
                    if (
                        posix_name.startswith("/")
                        or ".." in parts
                        or (parts and ":" in parts[0])
                    ):
                        raise ValueError(f"Unsafe path in ZIP archive rejected: {info.filename}")
                    if info.is_dir():
                        continue
                    processed_members += 1
                    if processed_members > MAX_MEMBERS:
                        raise ValueError("ZIP archive exceeds safe extraction limits (too many entries).")
                    # Cap individual uncompressed member at 150 MB (verify actual stream bytes, not just header)
                    if info.file_size > MAX_MEMBER_BYTES:
                        skipped_files.append(os.path.basename(posix_name))
                        continue
                    with zf.open(info, "r") as member_fp:
                        chunk = member_fp.read(MAX_MEMBER_BYTES + 1)
                    if len(chunk) > MAX_MEMBER_BYTES:
                        skipped_files.append(os.path.basename(posix_name))
                        continue
                    cumulative_bytes += len(chunk)
                    if cumulative_bytes > MAX_TOTAL_BYTES:
                        raise ValueError("ZIP archive exceeds safe extraction limits (total uncompressed size).")
                    _safe_save_json_member(info.filename, chunk)
        except zipfile.BadZipFile as exc:
            raise ValueError(f"Invalid ZIP archive: {exc}") from exc
    elif lower.endswith(".json"):
        _safe_save_json_member(clean_upload_name, raw_bytes)
    else:
        raise ValueError("Unsupported file type. Upload a Spotify Extended History .zip or .json file.")

    return {"saved_files": saved_files, "skipped_files": skipped_files}


_SPOTIFY_TYPED_URI_RE = re.compile(
    r"^spotify:(track|album|episode|show|audiobook|user|playlist):([A-Za-z0-9]{10,40})$"
)


def _normalize_spotify_uri(uri_or_id, default_type="track"):
    """Normalize a 22-char base62 ID, Spotify URL, or URI into a canonical spotify:<type>:<id> URI."""
    raw = str(uri_or_id or "").strip()
    if not raw:
        return None
    if _SPOTIFY_TYPED_URI_RE.match(raw):
        return raw
    m = re.search(
        r"open\.spotify\.com/(track|album|episode|show|playlist)/([A-Za-z0-9]{10,40})",
        raw,
    )
    if m:
        return f"spotify:{m.group(1)}:{m.group(2)}"
    if is_valid_spotify_id(raw):
        return f"spotify:{default_type}:{raw}"
    return None


def _mutate_spotify_library(sp, uris, remove=False):
    clean_uris = []
    seen = set()
    for item in uris or []:
        norm = _normalize_spotify_uri(item, default_type="track")
        if norm and norm not in seen:
            seen.add(norm)
            clean_uris.append(norm)
    if not clean_uris:
        return {"count": 0, "uris": []}

    for i in range(0, len(clean_uris), 40):
        chunk = clean_uris[i : i + 40]
        uris_param = ",".join(chunk)
        try:
            if remove:
                sp._delete("me/library", uris=uris_param)
            else:
                sp._put("me/library", uris=uris_param)
        except Exception as exc:
            msg = str(exc)
            status = getattr(exc, "http_status", None)
            if status in (404, 405) or "404" in msg or "405" in msg:
                track_ids = [u.split(":")[-1] for u in chunk if u.startswith("spotify:track:")]
                if track_ids:
                    if remove and hasattr(sp, "current_user_saved_tracks_delete"):
                        sp.current_user_saved_tracks_delete(track_ids)
                    elif not remove and hasattr(sp, "current_user_saved_tracks_add"):
                        sp.current_user_saved_tracks_add(track_ids)
                    else:
                        raise
                else:
                    raise
            else:
                raise
    return {"count": len(clean_uris), "uris": clean_uris}


def save_to_spotify_library(sp, uris):
    """Save up to 40 Spotify URIs per request using Feb 2026 PUT /me/library?uris=..."""
    return _mutate_spotify_library(sp, uris, remove=False)


def remove_from_spotify_library(sp, uris):
    """Remove up to 40 Spotify URIs per request using Feb 2026 DELETE /me/library?uris=..."""
    return _mutate_spotify_library(sp, uris, remove=True)


def _format_playback_item(item):
    if not isinstance(item, dict) or item.get("type") == "episode" or item.get("is_local"):
        return None
    tid = item.get("id")
    name = item.get("name")
    if not tid or not name:
        return None
    artists = [a.get("name") for a in (item.get("artists") or []) if isinstance(a, dict) and a.get("name")]
    album = item.get("album") or {}
    return {
        "id": str(tid),
        "name": name,
        "artist": ", ".join(artists) if artists else "Unknown",
        "album": album.get("name", ""),
        "spotify_url": f"https://open.spotify.com/track/{tid}",
    }


def get_user_playback_queue(sp):
    """Fetch currently playing track + upcoming queue via GET /me/player/queue."""
    resp = sp.queue() if hasattr(sp, "queue") else sp._get("me/player/queue")
    if not isinstance(resp, dict):
        return {"currently_playing": None, "queue": []}
    cur = _format_playback_item(resp.get("currently_playing"))
    q_items = []
    for raw in resp.get("queue") or []:
        formatted = _format_playback_item(raw)
        if formatted:
            q_items.append(formatted)
    return {"currently_playing": cur, "queue": q_items}


def queue_tracks_on_spotify(sp, track_uris, max_items=30):
    """Append tracks to the active Spotify playback queue via POST /me/player/queue."""
    cap = max(1, min(50, int(max_items or 30)))
    clean_uris = []
    seen = set()
    for item in track_uris or []:
        norm = _normalize_spotify_uri(item, default_type="track")
        if norm and norm.startswith("spotify:track:") and norm not in seen:
            seen.add(norm)
            clean_uris.append(norm)
            if len(clean_uris) >= cap:
                break

    queued = 0
    errors = []
    for uri in clean_uris:
        try:
            sp.add_to_queue(uri)
            queued += 1
        except Exception as exc:
            errors.append(str(exc))
            if "404" in str(exc) or "NO_ACTIVE_DEVICE" in str(exc) or "403" in str(exc):
                break
    return {"queued": queued, "requested": len(clean_uris), "errors": errors}


def start_spotify_playback(sp, track_uris, device_id=None):
    """Start immediate playback of a track URI list via PUT /me/player/play."""
    clean_uris = []
    seen = set()
    for item in track_uris or []:
        norm = _normalize_spotify_uri(item, default_type="track")
        if norm and norm.startswith("spotify:track:") and norm not in seen:
            seen.add(norm)
            clean_uris.append(norm)
            if len(clean_uris) >= 100:
                break
    if not clean_uris:
        raise ValueError("No valid Spotify track URIs to play.")
    sp.start_playback(uris=clean_uris, device_id=device_id)
    return {"started": True, "count": len(clean_uris)}


def _parse_release_date(val):
    s = str(val or "").strip()
    if not s:
        return None
    try:
        if len(s) >= 10:
            dt = datetime.strptime(s[:10], "%Y-%m-%d")
        elif len(s) == 7:
            dt = datetime.strptime(s, "%Y-%m")
        elif len(s) == 4 and s.isdigit():
            dt = datetime.strptime(s, "%Y")
        else:
            return None
        return dt.replace(tzinfo=timezone.utc)
    except ValueError:
        return None


def fetch_personal_release_radar(sp, artist_ids, days=45, max_artists=25, now=None):
    """
    Build a Personal Release Radar from recent albums & singles of the user's Top/Followed
    artists using GET /artists/{id}/albums and GET /albums/{id}/tracks (2026 compliant).
    """
    now_dt = now or datetime.now(timezone.utc)
    cutoff_sec = max(1, int(days or 45)) * 86400.0
    cap_artists = max(1, min(50, int(max_artists or 25)))

    targets = []
    seen_aids = set()
    for raw_aid in artist_ids or []:
        aid = str(raw_aid or "").strip()
        if aid and aid not in seen_aids and is_valid_spotify_id(aid):
            seen_aids.add(aid)
            targets.append(aid)
            if len(targets) >= cap_artists:
                break

    albums_by_id = {}
    for aid in targets:
        try:
            resp = sp.artist_albums(aid, include_groups="album,single", limit=10)
        except Exception:
            continue
        for alb in (resp or {}).get("items") or []:
            if not isinstance(alb, dict) or not alb.get("id"):
                continue
            r_date_str = str(alb.get("release_date") or "")
            r_dt = _parse_release_date(r_date_str)
            if r_dt is None:
                continue
            age_sec = (now_dt - r_dt).total_seconds()
            if age_sec < -86400.0 or age_sec > cutoff_sec:
                continue
            alb_id = str(alb["id"])
            if alb_id not in albums_by_id:
                images = alb.get("images") or []
                albums_by_id[alb_id] = {
                    "id": alb_id,
                    "name": alb.get("name") or "Untitled Release",
                    "album_type": alb.get("album_type") or "single",
                    "release_date": r_date_str,
                    "image": images[0].get("url") if images and isinstance(images[0], dict) else None,
                    "artists": [
                        a.get("name")
                        for a in (alb.get("artists") or [])
                        if isinstance(a, dict) and a.get("name")
                    ],
                }

    sorted_albums = sorted(albums_by_id.values(), key=lambda a: a["release_date"], reverse=True)
    tracks = []
    seen_tids = set()
    for alb in sorted_albums[:40]:
        try:
            tr_resp = sp.album_tracks(alb["id"], limit=50)
        except Exception:
            continue
        for raw_t in ((tr_resp or {}).get("items") or [])[:10]:
            if not isinstance(raw_t, dict) or not raw_t.get("id") or not raw_t.get("name"):
                continue
            if raw_t.get("is_local") or raw_t.get("type") == "episode":
                continue
            tid = str(raw_t["id"])
            if tid in seen_tids:
                continue
            seen_tids.add(tid)
            artists = [
                a.get("name")
                for a in (raw_t.get("artists") or [])
                if isinstance(a, dict) and a.get("name")
            ] or alb["artists"]
            tracks.append(
                {
                    "id": tid,
                    "name": raw_t["name"],
                    "artist": ", ".join(artists) if artists else "Unknown",
                    "album": alb["name"],
                    "album_type": alb["album_type"],
                    "release_date": alb["release_date"],
                    "image": alb["image"],
                    "explicit": bool(raw_t.get("explicit", False)),
                    "source": "Release Radar",
                    "spotify_url": f"https://open.spotify.com/track/{tid}",
                }
            )

    return {"releases": sorted_albums, "tracks": tracks}


# =======================================================
# YOUTUBE MUSIC IDS & UNIFIED PROFILE HELPERS
# =======================================================

_YTM_VIDEO_ID_RE = re.compile(r"^[A-Za-z0-9_-]{11}$")
_YTM_TYPED_ID_RE = re.compile(r"^yt:([A-Za-z0-9_-]{11})$")
_YTM_URL_RE = re.compile(r"(?:music\.youtube\.com|youtube\.com)/watch\?(?:[^#]*&)?v=([A-Za-z0-9_-]{11})")
_YOUTU_BE_RE = re.compile(r"youtu\.be/([A-Za-z0-9_-]{11})")


def is_valid_ytm_video_id(vid):
    """Return True if vid is an 11-character YouTube/YTM videoId."""
    return isinstance(vid, str) and bool(_YTM_VIDEO_ID_RE.match(vid))


def is_valid_ytm_id(tid):
    """Return True if tid is a prefixed 'yt:<11-char videoId>' track ID."""
    return isinstance(tid, str) and bool(_YTM_TYPED_ID_RE.match(tid))


def extract_ytm_video_id(val):
    """Extract an 11-char videoId from yt:<id>, URL, or bare videoId."""
    raw = str(val or "").strip()
    if not raw:
        return None
    m = _YTM_TYPED_ID_RE.match(raw)
    if m:
        return m.group(1)
    if _YTM_VIDEO_ID_RE.match(raw):
        return raw
    for pat in (_YTM_URL_RE, _YOUTU_BE_RE):
        um = pat.search(raw)
        if um:
            return um.group(1)
    return None


def get_track_provider(track_or_id):
    """Determine track provider: 'spotify', 'ytmusic', or 'both'."""
    if isinstance(track_or_id, dict):
        prov = str(track_or_id.get("provider") or "").strip().lower()
        if prov in ("spotify", "ytmusic", "both"):
            return prov
        has_sp = is_valid_spotify_id(track_or_id.get("spotify_id")) or is_valid_spotify_id(track_or_id.get("id"))
        has_yt = is_valid_ytm_video_id(track_or_id.get("video_id")) or is_valid_ytm_id(track_or_id.get("id"))
        if has_sp and has_yt:
            return "both"
        if has_yt:
            return "ytmusic"
        return "spotify"
    s = str(track_or_id or "").strip()
    if is_valid_ytm_id(s):
        return "ytmusic"
    return "spotify"


def _tag_profile_items(items, prov, is_playlist_wrap=False):
    out = []
    for item in items or []:
        if not isinstance(item, dict):
            continue
        cp = dict(item)
        if is_playlist_wrap:
            cp.setdefault("provider", prov)
            cp["tracks"] = _tag_profile_items(cp.get("tracks"), prov, is_playlist_wrap=False)
        else:
            cp.setdefault("provider", prov)
        out.append(cp)
    return out


def load_merged_music_profile(stage1_dir=None, provider=None):
    """
    Load Spotify profile, YouTube Music profile, or a Unified merged profile dict
    combining both platforms according to active_provider preference.
    """
    s1_dir = stage1_dir or os.path.join(config.BASE_DIR, "01_Data_Extractor")
    eff_provider = (
        str(provider or get_effective_preferences().get("active_provider", "spotify")).strip().lower()
    )
    if eff_provider not in ("spotify", "ytmusic", "unified"):
        eff_provider = "spotify"

    sp_path = os.path.join(s1_dir, "user_music_profile.json")
    ytm_path = os.path.join(s1_dir, "ytmusic_profile.json")

    sp_prof = load_json_safe(sp_path) if eff_provider in ("spotify", "unified") else None
    ytm_prof = load_json_safe(ytm_path) if eff_provider in ("ytmusic", "unified") else None
    sp_ok = isinstance(sp_prof, dict)
    ytm_ok = isinstance(ytm_prof, dict)

    if eff_provider == "spotify":
        return sp_prof if sp_ok else None
    if eff_provider == "ytmusic":
        return ytm_prof if ytm_ok else None

    if not sp_ok and not ytm_ok:
        return None
    if sp_ok and not ytm_ok:
        return sp_prof
    if ytm_ok and not sp_ok:
        return ytm_prof

    merged = {
        "schema_version": "1.2-unified",
        "generated_at": max(
            str(sp_prof.get("generated_at") or ""),
            str(ytm_prof.get("generated_at") or ""),
        ),
        "providers": ["spotify", "ytmusic"],
        "user_info": sp_prof.get("user_info") or ytm_prof.get("user_info") or {},
        "artists": {
            tr: _tag_profile_items((sp_prof.get("artists") or {}).get(tr), "spotify")
            + _tag_profile_items((ytm_prof.get("artists") or {}).get(tr), "ytmusic")
            for tr in ("short_term", "medium_term", "long_term")
        },
        "tracks": {
            tr: _tag_profile_items((sp_prof.get("tracks") or {}).get(tr), "spotify")
            + _tag_profile_items((ytm_prof.get("tracks") or {}).get(tr), "ytmusic")
            for tr in ("short_term", "medium_term", "long_term")
        },
        "recently_played": [
            {**entry, "track": {**entry["track"], "provider": "spotify"}}
            for entry in (sp_prof.get("recently_played") or [])
            if isinstance(entry, dict) and isinstance(entry.get("track"), dict)
        ]
        + [
            {**entry, "track": {**entry["track"], "provider": "ytmusic"}}
            for entry in (ytm_prof.get("recently_played") or [])
            if isinstance(entry, dict) and isinstance(entry.get("track"), dict)
        ],
        "short_term_artists_top_tracks": _tag_profile_items(
            sp_prof.get("short_term_artists_top_tracks"), "spotify"
        )
        + _tag_profile_items(ytm_prof.get("short_term_artists_top_tracks"), "ytmusic"),
        "liked_songs_sample": _tag_profile_items(sp_prof.get("liked_songs_sample"), "spotify")
        + _tag_profile_items(ytm_prof.get("liked_songs_sample"), "ytmusic"),
        "followed_artists": _tag_profile_items(sp_prof.get("followed_artists"), "spotify")
        + _tag_profile_items(ytm_prof.get("followed_artists"), "ytmusic"),
        "saved_albums": _tag_profile_items(sp_prof.get("saved_albums"), "spotify", is_playlist_wrap=True)
        + _tag_profile_items(ytm_prof.get("saved_albums"), "ytmusic", is_playlist_wrap=True),
        "playlists": _tag_profile_items(sp_prof.get("playlists"), "spotify", is_playlist_wrap=True)
        + _tag_profile_items(ytm_prof.get("playlists"), "ytmusic", is_playlist_wrap=True),
        "audio_features": {
            **(ytm_prof.get("audio_features") or {}),
            **(sp_prof.get("audio_features") or {}),
        },
        "extended_history_summary": sp_prof.get("extended_history_summary")
        or ytm_prof.get("extended_history_summary"),
    }
    return merged


_NVIDIA_GPU_CACHE = None


def probe_nvidia_gpu():
    """Safely detect local NVIDIA GPU name & VRAM (GB) via nvidia-smi without shell=True."""
    global _NVIDIA_GPU_CACHE
    if _NVIDIA_GPU_CACHE is not None:
        return dict(_NVIDIA_GPU_CACHE)

    info = {"name": None, "vram_gb": None}
    try:
        proc = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=2,
            shell=False,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        if proc.returncode == 0 and proc.stdout.strip():
            first_line = proc.stdout.strip().splitlines()[0]
            parts = [p.strip() for p in first_line.split(",")]
            if parts and parts[0]:
                info["name"] = parts[0]
            if len(parts) > 1:
                try:
                    info["vram_gb"] = round(float(parts[1]) / 1024.0, 1)
                except ValueError:
                    pass
    except Exception:
        pass

    _NVIDIA_GPU_CACHE = dict(info)
    return info


def probe_ollama_server(base_url=None, timeout=0.8):
    """Check if a free local Ollama server is running and return available local model tags."""
    url = _sanitize_llm_url(base_url)
    try:
        import requests

        resp = requests.get(f"{url}/api/tags", timeout=timeout)
        if resp.status_code == 200:
            data = resp.json() if resp.content else {}
            models = [
                str(m.get("name") or "").strip()
                for m in (data.get("models") or [])
                if isinstance(m, dict) and m.get("name")
            ]
            return {"running": True, "models": models, "base_url": url}
    except Exception:
        pass
    return {"running": False, "models": [], "base_url": url}


def recommend_tier_for_specs(
    ram_total_gb,
    nvidia_gpu_name=None,
    nvidia_vram_gb=None,
    openvino_devices=None,
    cpu_name="",
    ollama_running=False,
):
    """Recommend hardware tier, OpenVINO model key, and local Ollama CUDA model from machine specs."""
    ram = float(ram_total_gb or 16.0)
    vram = float(nvidia_vram_gb) if isinstance(nvidia_vram_gb, (int, float)) else 0.0
    ov_devs = [str(d).upper() for d in (openvino_devices or ["CPU"])]
    has_ov_gpu = any(d.startswith("GPU") for d in ov_devs)

    if ram < 12.0:
        tier = 1
        tier_label = "Tier 1 — Compact (8 GB+ RAM)"
        ov_key = "qwen3-1.7b-ov"
        ov_dev = "GPU" if has_ov_gpu else "CPU"
        ol_model = "qwen3:4b"
        advice = "Compact Qwen3-1.7B fits smoothly in <=12 GB RAM."
    elif ram < 24.0:
        tier = 2
        tier_label = "Tier 2 — Balanced Laptop (16 GB RAM)"
        ov_key = "qwen3-4b-ov"
        ov_dev = "GPU" if has_ov_gpu else "CPU"
        ol_model = "qwen3:4b"
        if vram >= 5.5 and nvidia_gpu_name:
            advice = (
                f"{nvidia_gpu_name} ({vram:.0f} GB VRAM) + {ram:.0f} GB RAM detected: "
                f"Qwen3-4B runs well on OpenVINO, or run Ollama GPU ({ol_model}) for max speed."
            )
        else:
            advice = (
                f"{ram:.0f} GB RAM system ({ov_dev} acceleration): Qwen3-4B INT4 (2.1 GB) "
                "delivers strong JSON adherence without heavy Windows memory swapping."
            )
    elif vram >= 7.5:
        tier = 4 if ram >= 30.0 else 3
        tier_label = "Tier 4 — Enthusiast (8 GB+ RTX / 32 GB RAM)" if tier == 4 else "Tier 3 — Performance (24–32 GB RAM)"
        ov_key = "qwen3-8b-ov"
        ov_dev = "GPU" if has_ov_gpu else "CPU"
        ol_model = "qwen3:8b"
        advice = (
            f"{nvidia_gpu_name or 'NVIDIA RTX'} ({vram:.0f} GB VRAM) + {ram:.0f} GB RAM detected! "
            "OpenVINO Qwen3-8B runs on CPU/iGPU out of the box, or install free local Ollama (qwen3:8b or gemma3:12b) to unleash your RTX CUDA cores at 40+ tok/s."
        )
    elif vram >= 5.0:
        tier = 3
        tier_label = "Tier 3 — Performance (6 GB RTX / 32 GB RAM)"
        ov_key = "qwen3-8b-ov"
        ov_dev = "GPU" if has_ov_gpu else "CPU"
        ol_model = "qwen3:8b"
        advice = (
            f"{nvidia_gpu_name or 'NVIDIA RTX'} ({vram:.0f} GB VRAM) + {ram:.0f} GB RAM detected! "
            "OpenVINO Qwen3-8B runs cleanly on your CPU/32 GB RAM, or run free local Ollama (qwen3:8b / qwen3:4b) on RTX CUDA for ultra-fast curation."
        )
    else:
        tier = 3
        tier_label = "Tier 3 — Performance (24–32 GB RAM)"
        ov_key = "qwen3-8b-ov"
        ov_dev = "GPU" if has_ov_gpu else "CPU"
        ol_model = "qwen3:8b"
        advice = (
            f"{ram:.0f} GB RAM detected ({cpu_name or ov_dev}): Qwen3-8B INT4 (4.6 GB) fits comfortably "
            "and provides much stronger genre/artist depth."
        )

    return {
        "recommended_tier": tier,
        "recommended_tier_label": tier_label,
        "recommended_model_key": ov_key,
        "recommended_ov_device": ov_dev,
        "recommended_backend": "ollama" if (ollama_running and vram >= 5.0) else "openvino",
        "recommended_ollama_model": ol_model,
        "advice": advice,
    }


def detect_hardware_profile(probe_ollama=False, ollama_url=None):
    """Detect CPU, RAM, OpenVINO devices, NVIDIA RTX GPU, and recommended LLM tier."""
    dev_info = get_openvino_devices()
    descs = dev_info.get("descriptions") or ["CPU"]
    cpu_name = str(descs[0]) if descs else "CPU"

    ram_total_gb = None
    ram_used_gb = None
    ram_percent = None
    cpu_percent = None
    psutil_ok = False
    try:
        import psutil

        psutil_ok = True
        vm = psutil.virtual_memory()
        ram_total_gb = round(vm.total / (1024 ** 3), 1)
        ram_used_gb = round(vm.used / (1024 ** 3), 1)
        ram_percent = round(float(vm.percent), 1)
        cpu_percent = round(float(psutil.cpu_percent(interval=None)), 1)
    except ImportError:
        pass

    nv = probe_nvidia_gpu()
    eff_ollama_url = ollama_url or get_effective_preferences().get("ollama_url")
    ol = probe_ollama_server(eff_ollama_url) if probe_ollama else {
        "running": False,
        "models": [],
        "base_url": _sanitize_llm_url(eff_ollama_url),
    }

    rec = recommend_tier_for_specs(
        ram_total_gb=ram_total_gb if ram_total_gb is not None else 16.0,
        nvidia_gpu_name=nv.get("name"),
        nvidia_vram_gb=nv.get("vram_gb"),
        openvino_devices=dev_info.get("raw", ["CPU"]),
        cpu_name=cpu_name,
        ollama_running=bool(ol.get("running")),
    )

    return {
        "psutil_available": psutil_ok,
        "cpu_percent": cpu_percent,
        "cpu_cores": os.cpu_count() or 1,
        "cpu_name": cpu_name,
        "ram_used_gb": ram_used_gb,
        "ram_total_gb": ram_total_gb,
        "ram_percent": ram_percent,
        "silicon": ", ".join(descs),
        "openvino_devices": dev_info.get("raw", ["CPU"]),
        "nvidia_gpu_name": nv.get("name"),
        "nvidia_vram_gb": nv.get("vram_gb"),
        "ollama": ol,
        **rec,
    }


