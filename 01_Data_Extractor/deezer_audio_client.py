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
DSP_CACHE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "audio_dsp_cache.json")

_STRIP_PARENS_RE = re.compile(
    r"(?i)\s*[\(\[](?:official\s*(?:music\s*)?video|official\s*audio|lyric\s*video|lyrics|"
    r"remaster(?:ed)?(?:\s*\d{4})?|\d{4}\s*remaster(?:ed)?|live[^)\]]*|audio|hd|4k|"
    r"feat\.?[^)\]]*|ft\.?[^)\]]*)[\)\]]"
)
_TOPIC_SUFFIX_RE = re.compile(r"(?i)\s*-\s*topic$")
_RATE_LOCK = threading.Lock()
_LAST_DEEZER_CALL = 0.0

# Krumhansl-Schmuckler key profiles (C .. B)
_MAJOR_PROFILE = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88], dtype=float)
_MINOR_PROFILE = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17], dtype=float)


def _clean_query_parts(title, artist):
    clean_title = _STRIP_PARENS_RE.sub("", str(title or "")).strip()
    clean_title = re.sub(r"\s+", " ", clean_title) or str(title or "").strip()
    primary_artist = str(artist or "").split(",")[0].strip()
    primary_artist = _TOPIC_SUFFIX_RE.sub("", primary_artist).strip()
    if primary_artist.lower() in ("unknown", "various artists"):
        primary_artist = ""
    return clean_title, primary_artist


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
        return cache[fp]

    clean_title, primary_artist = _clean_query_parts(title, artist)
    if not clean_title:
        res = {"found": False}
        if cache is not None:
            cache[fp] = res
        return res

    queries = []
    if primary_artist:
        queries.append(f'artist:"{primary_artist}" track:"{clean_title}"')
        queries.append(f"{clean_title} {primary_artist}")
    else:
        queries.append(clean_title)

    matched = None
    for q in queries:
        data = _polite_deezer_get("https://api.deezer.com/search", params={"q": q, "limit": 5}, session=session)
        items = (data or {}).get("data") or []
        matched = _pick_best_deezer_match(items, clean_title, primary_artist)
        if matched is not None:
            break

    if matched is None:
        res = {"found": False}
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
            "deezer_id": matched.get("id"),
            "title": matched.get("title") or clean_title,
            "artist": artist_obj.get("name") or primary_artist,
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


def load_dsp_cache():
    data = load_json_safe(DSP_CACHE_PATH, default={})
    return data if isinstance(data, dict) else {}


def save_dsp_cache(cache):
    atomic_write_json(DSP_CACHE_PATH, cache, indent=2)
