"""Deezer public preview search, in-memory MP3 decoding (PyAV), and librosa DSP feature extraction."""

import io
import os
import re
import sys
import threading
import time
from difflib import SequenceMatcher

import numpy as np
import requests

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from utils import atomic_write_json, get_track_fingerprint, load_json_safe

DEEZER_CACHE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "deezer_cache.json")
DEEZER_ALBUM_GENRES_CACHE_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "deezer_album_genres_cache.json"
)
DSP_CACHE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "audio_dsp_cache.json")

_STRIP_PARENS_RE = re.compile(
    r"(?i)\s*[\(\[（【](?:official\s*(?:music\s*|hd\s*|4k\s*|lyric\s*)?video|official\s*audio|"
    r"official\s*lyric\s*video|lyric\s*video|lyrics?|remaster(?:ed)?(?:\s*\d{4})?|"
    r"\d{4}\s*remaster(?:ed)?|live[^)\]）】]*|audio|hd|4k|full(?:\s*version)?|完整版|"
    r"covered?\s*by[^)\]）】]*|from\s*[^)\]）】]*|feat\.?[^)\]）】]*|ft\.?[^)\]）】]*)[\)\]）】]"
)
_VIEW_COUNT_RE = re.compile(r"(?i)(?:,\s*)?\b\d+(?:\.\d+)?\s*[kmb]?\s+views?\b")
_PIPE_TAIL_RE = re.compile(r"\s*\|.*$")
_HASHTAG_RE = re.compile(r"#\S+")
_NOISE_PHRASES_RE = re.compile(
    r"(?i)\b(?:official\s*(?:hd|4k|music|lyric)?\s*video|official\s*audio|"
    r"4k\s*video\s*song|music\s*video|lyric\s*video|full\s*song|full\s*video)\b"
)
_EMOJI_RE = re.compile(r"[\U00010000-\U0010ffff\u2600-\u27bf\ufe0f]+")
_TOPIC_SUFFIX_RE = re.compile(r"(?i)\s*-\s*topic$")
_RATE_LOCK = threading.Lock()
_LAST_DEEZER_CALL = 0.0

# Krumhansl-Schmuckler key profiles (C .. B)
_MAJOR_PROFILE = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88], dtype=float)
_MINOR_PROFILE = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17], dtype=float)


def _strip_wrapping_quotes(text):
    return str(text or "").strip().strip('"\'“”‘’').strip()


def _clean_title_text(title):
    s = str(title or "").replace("\u3000", " ").strip()
    s = _PIPE_TAIL_RE.sub("", s)
    s = _HASHTAG_RE.sub(" ", s)
    s = _EMOJI_RE.sub(" ", s)
    s = _STRIP_PARENS_RE.sub(" ", s)
    s = _NOISE_PHRASES_RE.sub(" ", s)
    s = re.sub(r"\s*[:\-–—]\s*$", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    return _strip_wrapping_quotes(s) or _strip_wrapping_quotes(str(title or ""))


def _clean_artist_text(artist):
    raw = _VIEW_COUNT_RE.sub("", str(artist or ""))
    primary_artist = raw.split(",")[0].strip()
    primary_artist = _TOPIC_SUFFIX_RE.sub("", primary_artist).strip()
    if primary_artist.lower() in ("unknown", "various artists"):
        primary_artist = ""
    return primary_artist


def _clean_query_parts(title, artist):
    return _clean_title_text(title), _clean_artist_text(artist)


def _extract_ascii_phrase(text):
    """Extract a clean, deduplicated ASCII title phrase from mixed-script titles."""
    ascii_parts = re.findall(r"[A-Za-z0-9][A-Za-z0-9'’&,.\- ]{3,}", str(text or ""))
    best_words = []
    for part in ascii_parts:
        # Strip inline repeats like 'I CAN'T STOP THE LONELINESS - ... I CAN'T STOP THE LONELINESS'
        segments = [seg.strip(" -–—") for seg in re.split(r"\s+[-–—]\s+", part) if seg.strip(" -–—")]
        for seg in segments:
            alpha_cnt = sum(1 for ch in seg if ch.isalpha())
            if alpha_cnt >= 5:
                norm = re.sub(r"\s+", " ", seg).strip()
                if norm.lower() not in [w.lower() for w in best_words]:
                    best_words.append(norm)
    if not best_words:
        return ""
    # Prefer the first English/ASCII phrase if shorter than full concatenated duplicate
    candidate = best_words[0]
    for w in best_words:
        if any(ch.isupper() for ch in w) and len(w.split()) >= 2:
            candidate = w
            break
    return _strip_wrapping_quotes(candidate)


def _build_search_candidates(title, artist):
    """Build ordered (clean_title, artist) search candidates for noisy YouTube Music metadata."""
    clean_title, primary_artist = _clean_query_parts(title, artist)
    if not clean_title:
        return []

    candidates = []
    seen = set()

    def _add(cand_t, cand_a):
        ct = _strip_wrapping_quotes(re.sub(r"\s+", " ", str(cand_t or "")).strip())
        ca = str(cand_a or "").strip()
        if not ct:
            return
        key = (ct.lower(), ca.lower())
        if key not in seen:
            seen.add(key)
            candidates.append((ct, ca))

    has_non_ascii = any(ord(ch) > 127 for ch in clean_title)
    ascii_phrase = _extract_ascii_phrase(clean_title) if has_non_ascii else ""

    if ascii_phrase and ascii_phrase.lower() != clean_title.lower():
        _add(ascii_phrase, primary_artist)

    dash_match = re.split(r"\s+[-–—]\s+", clean_title, maxsplit=1)
    if len(dash_match) == 2:
        left = _clean_title_text(dash_match[0])
        right = _clean_title_text(dash_match[1])
        if primary_artist and left.lower() == primary_artist.lower():
            # e.g. 'Zak Azoury - Familiar Feelings' by 'Zak Azoury'
            _add(right, primary_artist)
        elif has_non_ascii and ascii_phrase:
            # Bilingual title like '悲しみがとまらない I CAN'T STOP THE LONELINESS - Kanashimi...'
            if left:
                _add(left, primary_artist)
            if right:
                _add(right, primary_artist)
        else:
            if right and left:
                # Rescues re-uploads like 'The Handsome Family - Far From Any Road' by 'Beatriz'
                _add(right, left)
            if right and primary_artist:
                _add(right, primary_artist)
            if left and primary_artist:
                _add(left, primary_artist)

    _add(clean_title, primary_artist)
    return candidates[:4]


def _similarity(a, b):
    return SequenceMatcher(None, str(a or "").lower().strip(), str(b or "").lower().strip()).ratio()


def _polite_deezer_get(url, params=None, session=None, timeout=10):
    global _LAST_DEEZER_CALL
    client = session or requests
    for attempt in range(3):
        with _RATE_LOCK:
            elapsed = time.monotonic() - _LAST_DEEZER_CALL
            if elapsed < 0.15:
                time.sleep(0.15 - elapsed)
            _LAST_DEEZER_CALL = time.monotonic()
        try:
            resp = client.get(url, params=params, timeout=timeout)
            if resp.status_code == 429 or resp.status_code >= 500:
                time.sleep(0.5 * (attempt + 1))
                continue
            if resp.status_code == 200:
                data = resp.json()
                if isinstance(data, dict) and data.get("error", {}).get("code") == 4:
                    # Deezer Quota limit exceeded
                    time.sleep(1.0 * (attempt + 1))
                    continue
                return data
            return None
        except Exception:
            time.sleep(0.4 * (attempt + 1))
    return None


def _pick_best_deezer_match(items, clean_title, primary_artist):
    best_item = None
    best_score = -1.0
    for item in items or []:
        if not isinstance(item, dict):
            continue
        it_title = item.get("title_short") or item.get("title") or ""
        it_artist = (item.get("artist") or {}).get("name") or ""
        t_sim = _similarity(clean_title, it_title)
        a_sim = _similarity(primary_artist, it_artist) if primary_artist else 0.7
        if primary_artist and primary_artist.lower() in it_artist.lower():
            a_sim = max(a_sim, 0.9)
        if clean_title.lower() in it_title.lower():
            t_sim = max(t_sim, 0.85)
        score = (0.62 * t_sim) + (0.38 * a_sim)
        if item.get("preview"):
            score += 0.05
        if score > best_score:
            best_score = score
            best_item = item
    if best_item is not None and best_score >= 0.55:
        return best_item
    return None


def search_deezer_track(title, artist, session=None, cache=None):
    """Search Deezer public API for a track's metadata, BPM, and 30s MP3 preview URL."""
    fp = get_track_fingerprint(title, artist)
    if cache is not None and fp in cache:
        cached = cache[fp]
        if isinstance(cached, dict) and (cached.get("found") or cached.get("v") == 2):
            return cached

    candidates = _build_search_candidates(title, artist)
    if not candidates:
        res = {"found": False, "v": 2}
        if cache is not None:
            cache[fp] = res
        return res

    matched = None
    matched_title, matched_artist = candidates[0]
    for cand_title, cand_artist in candidates:
        queries = []
        if cand_artist:
            queries.append(f'artist:"{cand_artist}" track:"{cand_title}"')
            queries.append(f"{cand_title} {cand_artist}")
        else:
            queries.append(cand_title)

        for q in queries:
            data = _polite_deezer_get("https://api.deezer.com/search", params={"q": q, "limit": 5}, session=session)
            items = (data or {}).get("data") or []
            matched = _pick_best_deezer_match(items, cand_title, cand_artist)
            if matched is not None:
                matched_title, matched_artist = cand_title, cand_artist
                break
        if matched is not None:
            break

    if matched is None:
        res = {"found": False, "v": 2}
    else:
        album = matched.get("album") or {}
        artist_obj = matched.get("artist") or {}
        bpm_val = matched.get("bpm")
        try:
            bpm_float = float(bpm_val) if bpm_val else None
            if bpm_float is not None and not (35.0 <= bpm_float <= 240.0):
                bpm_float = None
        except (TypeError, ValueError):
            bpm_float = None

        res = {
            "found": True,
            "v": 2,
            "deezer_id": matched.get("id"),
            "title": matched.get("title") or matched_title,
            "artist": artist_obj.get("name") or matched_artist,
            "album": album.get("title") or "",
            "album_id": album.get("id"),
            "preview_url": matched.get("preview") or "",
            "bpm": bpm_float,
            "release_date": matched.get("release_date") or "",
        }

    if cache is not None:
        cache[fp] = res
    return res


def refresh_deezer_preview_url(deezer_id, session=None):
    """Fetch a fresh signed preview URL for an existing Deezer track ID."""
    if not deezer_id:
        return ""
    data = _polite_deezer_get(f"https://api.deezer.com/track/{deezer_id}", session=session)
    if isinstance(data, dict):
        return data.get("preview") or ""
    return ""


def fetch_deezer_album_genres(album_id, session=None, cache=None):
    """Fetch and cache Deezer's editorial genre names for an album ID."""
    if not album_id:
        return []
    key = str(album_id)
    if cache is not None and key in cache:
        cached = cache[key]
        return list(cached) if isinstance(cached, list) else []

    data = _polite_deezer_get(f"https://api.deezer.com/album/{key}", session=session)
    genres = []
    if isinstance(data, dict):
        raw_list = ((data.get("genres") or {}).get("data")) or []
        seen = set()
        for item in raw_list:
            name = str((item or {}).get("name") or "").strip()
            if name and name.lower() != "all" and name.lower() not in seen:
                seen.add(name.lower())
                genres.append(name)
    if cache is not None:
        cache[key] = genres
    return genres


def decode_mp3_bytes(mp3_bytes, target_sr=48000):
    """Decode raw MP3/audio bytes in memory into a mono float32 waveform at target_sr."""
    if not mp3_bytes:
        raise ValueError("Empty audio byte stream.")

    try:
        import av

        container = av.open(io.BytesIO(mp3_bytes))
        resampler = av.AudioResampler(format="fltp", layout="mono", rate=int(target_sr))
        chunks = []
        for frame in container.decode(audio=0):
            resampled = resampler.resample(frame)
            if not isinstance(resampled, list):
                resampled = [resampled]
            for rf in resampled:
                arr = rf.to_ndarray()
                if arr.ndim == 2:
                    arr = arr[0]
                chunks.append(arr.astype(np.float32, copy=False))
        # Flush resampler
        for rf in resampler.resample(None) or []:
            arr = rf.to_ndarray()
            if arr.ndim == 2:
                arr = arr[0]
            chunks.append(arr.astype(np.float32, copy=False))
        container.close()
        if not chunks:
            raise ValueError("Decoded 0 audio frames.")
        return np.concatenate(chunks)
    except ImportError:
        import librosa
        import soundfile as sf

        data, sr = sf.read(io.BytesIO(mp3_bytes), dtype="float32", always_2d=False)
        if data.ndim == 2:
            data = np.mean(data, axis=1)
        if sr != target_sr:
            data = librosa.resample(data, orig_sr=sr, target_sr=target_sr)
        return data.astype(np.float32, copy=False)


def download_preview_waveform(preview_url, deezer_id=None, target_sr=48000, session=None, timeout=15):
    """Download a 30s preview MP3 (refreshing signed URL once on 403) and decode in memory."""
    client = session or requests
    url = preview_url
    for attempt in range(2):
        if not url:
            if attempt == 0 and deezer_id:
                url = refresh_deezer_preview_url(deezer_id, session=session)
            if not url:
                return None, ""
        try:
            resp = client.get(url, timeout=timeout)
            if resp.status_code in (403, 410) and attempt == 0 and deezer_id:
                url = refresh_deezer_preview_url(deezer_id, session=session)
                continue
            if resp.status_code == 200 and resp.content:
                wav = decode_mp3_bytes(resp.content, target_sr=target_sr)
                return wav, url
            return None, url
        except Exception:
            if attempt == 0 and deezer_id:
                url = refresh_deezer_preview_url(deezer_id, session=session)
                continue
            return None, url
    return None, url


def _estimate_key_mode(chroma):
    chroma_mean = np.mean(chroma, axis=1)
    if np.allclose(chroma_mean, 0):
        return 0, 1
    best_corr = -2.0
    best_key = 0
    best_mode = 1
    for shift in range(12):
        maj_rot = np.roll(_MAJOR_PROFILE, shift)
        min_rot = np.roll(_MINOR_PROFILE, shift)
        c_maj = float(np.corrcoef(chroma_mean, maj_rot)[0, 1])
        c_min = float(np.corrcoef(chroma_mean, min_rot)[0, 1])
        if np.isfinite(c_maj) and c_maj > best_corr:
            best_corr, best_key, best_mode = c_maj, shift, 1
        if np.isfinite(c_min) and c_min > best_corr:
            best_corr, best_key, best_mode = c_min, shift, 0
    return int(best_key), int(best_mode)


def extract_dsp_audio_features(waveform, sr=48000, deezer_bpm=None):
    """Compute BPM, energy, danceability, acousticness, key, mode, and valence from audio."""
    import librosa

    y = np.asarray(waveform, dtype=np.float32)
    if y.ndim > 1:
        y = np.mean(y, axis=0)
    if len(y) < max(2048, int(sr * 0.5)):
        raise ValueError("Waveform too short for DSP analysis.")

    analysis_sr = 22050
    if int(sr) != analysis_sr:
        y = librosa.resample(y, orig_sr=int(sr), target_sr=analysis_sr)

    # Normalize peak softly for consistent RMS/onset behavior across mastered eras
    peak = float(np.max(np.abs(y)))
    if peak > 1e-6:
        y_norm = y / peak
    else:
        y_norm = y

    # 1. Tempo (BPM)
    bpm = None
    if deezer_bpm is not None:
        try:
            db = float(deezer_bpm)
            if 40.0 <= db <= 220.0:
                bpm = db
        except (TypeError, ValueError):
            bpm = None

    onset_env = librosa.onset.onset_strength(y=y_norm, sr=analysis_sr)
    if bpm is None:
        tempo_arr = librosa.beat.beat_track(onset_envelope=onset_env, sr=analysis_sr)[0]
        raw_tempo = float(np.atleast_1d(tempo_arr)[0]) if tempo_arr is not None else 120.0
        if not np.isfinite(raw_tempo) or raw_tempo <= 20.0:
            raw_tempo = 120.0
        bpm = float(np.clip(raw_tempo, 45.0, 210.0))

    # 2. Energy from perceptual dBFS RMS (computed on un-peak-normalized audio so quiet tracks read quieter)
    rms = float(np.sqrt(np.mean(y ** 2)))
    rms_db = 20.0 * float(np.log10(max(rms, 1e-5)))
    energy = float(np.clip((rms_db + 28.0) / 21.0, 0.05, 0.98))

    # 3. Acousticness from spectral flatness & centroid rolloff
    flatness = float(np.mean(librosa.feature.spectral_flatness(y=y_norm)))
    centroid = float(np.mean(librosa.feature.spectral_centroid(y=y_norm, sr=analysis_sr)))
    brightness = float(np.clip((centroid - 600.0) / 3600.0, 0.0, 1.0))
    # Pure acoustic instruments have lower spectral flatness & moderate centroid
    acousticness = float(np.clip(1.0 - (0.65 * min(1.0, flatness / 0.08) + 0.35 * brightness), 0.05, 0.95))

    # 4. Danceability from onset pulse prominence & steady tempo fit
    if len(onset_env) > 4 and float(np.mean(onset_env)) > 1e-6:
        pulse_cv = float(np.std(onset_env) / (np.mean(onset_env) + 1e-6))
        tempo_sweet = 1.0 - min(1.0, abs(bpm - 118.0) / 80.0)
        danceability = float(np.clip(0.32 * min(1.0, pulse_cv / 1.4) + 0.38 * tempo_sweet + 0.30 * energy, 0.10, 0.95))
    else:
        danceability = 0.45

    # 5. Key & Mode via chroma STFT + Krumhansl-Schmuckler correlation
    chroma = librosa.feature.chroma_stft(y=y_norm, sr=analysis_sr)
    key, mode = _estimate_key_mode(chroma)

    # 6. Valence from mode (major brighter than minor), brightness, and energy
    mode_boost = 0.20 if mode == 1 else -0.15
    valence = float(np.clip(0.42 + mode_boost + 0.22 * (brightness - 0.45) + 0.18 * (energy - 0.5), 0.10, 0.95))

    return {
        "bpm": round(bpm, 1),
        "energy": round(energy, 3),
        "danceability": round(danceability, 3),
        "valence": round(valence, 3),
        "acousticness": round(acousticness, 3),
        "key": key,
        "mode": mode,
        "source": "dsp",
    }


def load_deezer_cache():
    data = load_json_safe(DEEZER_CACHE_PATH, default={})
    return data if isinstance(data, dict) else {}


def save_deezer_cache(cache):
    atomic_write_json(DEEZER_CACHE_PATH, cache, indent=2)


def load_deezer_album_genres_cache():
    data = load_json_safe(DEEZER_ALBUM_GENRES_CACHE_PATH, default={})
    return data if isinstance(data, dict) else {}


def save_deezer_album_genres_cache(cache):
    atomic_write_json(DEEZER_ALBUM_GENRES_CACHE_PATH, cache, indent=2)


def load_dsp_cache():
    data = load_json_safe(DSP_CACHE_PATH, default={})
    return data if isinstance(data, dict) else {}


def save_dsp_cache(cache):
    atomic_write_json(DSP_CACHE_PATH, cache, indent=2)
