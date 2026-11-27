import argparse
import codecs
import datetime
import importlib.util
import json
import mimetypes
import os
import re
import subprocess
import sys
import threading
import time
import urllib.parse
import webbrowser
from collections import deque
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from queue import Empty, Queue

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STAGE1_DIR = os.path.join(BASE_DIR, "01_Data_Extractor")
STAGE2_DIR = os.path.join(BASE_DIR, "02_Vector_Core")
STAGE3_DIR = os.path.join(BASE_DIR, "03_Agentic_Engine")
WEB_UI_DIR = os.path.join(BASE_DIR, "web_ui")

for path in (BASE_DIR, STAGE1_DIR, STAGE2_DIR, STAGE3_DIR):
    if path not in sys.path:
        sys.path.insert(0, path)

import config
import ytmusic_client
from automation_runner import query_windows_task, register_windows_task, remove_windows_task
from discovery_engine import LAST_PL_PATH, WarmDiscoverySession
from history_analytics import compute_history_analytics, find_forgotten_favorites
from library_hygiene import analyze_library_hygiene, dedupe_spotify_playlist
from llm_parser import get_llm_catalog_status
from reccobeats_client import get_cache_health, purge_poisoned_and_not_found
from utils import (
    DEFAULT_AUTOMATION_SETTINGS,
    atomic_write_json,
    atomic_write_text,
    compute_pipeline_freshness,
    configure_utf8_stdio,
    delete_archived_playlist,
    detect_hardware_profile,
    extract_uploaded_history_bytes,
    extract_ytm_video_id,
    find_local_plotly_js,
    format_playlist_csv,
    format_playlist_m3u,
    format_playlist_markdown,
    get_currently_playing_track,
    get_effective_preferences,
    get_openvino_devices,
    get_spotify_client_id,
    get_track_fingerprint,
    is_valid_spotify_id,
    is_valid_ytm_id,
    is_valid_ytm_video_id,
    list_archived_playlists,
    list_user_owned_playlists,
    load_archived_playlist,
    load_json_safe,
    load_user_feedback,
    normalize_avoid_list,
    probe_ollama_server,
    record_track_feedback,
    remove_from_spotify_library,
    save_playlist_to_archive,
    save_to_spotify_library,
    save_user_preferences,
    sequence_playlist_tracks,
)

_SAFE_LOG_FILE_RE = re.compile(r"^[A-Za-z0-9_.-]+\.log$")
_ARCHIVE_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,120}$")
_SPOTIFY_ID_RE = re.compile(r"^[A-Za-z0-9]{10,40}$")
_SPOTIFY_CID_INPUT_RE = re.compile(r"^[A-Za-z0-9]{16,64}$")
_YTM_PLAYLIST_ID_RE = re.compile(r"^[A-Za-z0-9_-]{2,80}$")
_DATE_OR_YEAR_RE = re.compile(r"^\d{4}(-\d{2}-\d{2})?$")
_LOOPBACK_HOSTS = {"127.0.0.1", "localhost", "::1"}
MAX_HISTORY_UPLOAD_BYTES = 250 * 1024 * 1024


def _is_loopback_host(host_header):
    raw = str(host_header or "").strip()
    if not raw:
        return False
    try:
        parsed = urllib.parse.urlparse(f"http://{raw}")
        host = (parsed.hostname or "").strip().lower()
    except ValueError:
        return False
    return host in _LOOPBACK_HOSTS


def _parse_int_param(qs, key, default, lo, hi):
    raw = (qs.get(key) or [None])[0]
    if raw is None or str(raw).strip() == "":
        return default
    try:
        val = int(str(raw).strip())
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid integer parameter '{key}': {raw!r}") from exc
    return max(lo, min(hi, val))


def enqueue_output(out, queue):
    """Stream subprocess stdout using an incremental UTF-8 decoder so multi-byte emojis never split."""
    decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
    try:
        while True:
            if hasattr(out, "read1"):
                chunk = out.read1(256)
            else:
                chunk = out.read(1)
            if not chunk:
                tail = decoder.decode(b"", final=True)
                if tail:
                    queue.put(tail)
                break
            text = decoder.decode(chunk, final=False)
            if text:
                queue.put(text)
    finally:
        try:
            out.close()
        except Exception:
            pass


def clean_system_files(base_dir=BASE_DIR):
    """Safely remove cached tokens, profiles, vector artifacts, and HTML output."""
    targets = [
        ".cache",
        ".spotify_client_id",
        "taste_map.html",
        os.path.join("01_Data_Extractor", ".cache"),
        os.path.join("01_Data_Extractor", "user_music_profile.json"),
        os.path.join("01_Data_Extractor", "ytmusic_profile.json"),
        os.path.join("02_Vector_Core", "taste_vectors.npz"),
        os.path.join("02_Vector_Core", "track_metadata.json"),
        os.path.join("02_Vector_Core", "taste_profile_summary.json"),
        os.path.join("02_Vector_Core", "taste_map_data.json"),
        os.path.join("03_Agentic_Engine", "last_generated_playlist.json"),
    ]
    removed = []
    errors = []
    for rel_path in targets:
        full_path = os.path.join(base_dir, rel_path)
        if os.path.exists(full_path):
            try:
                os.remove(full_path)
                removed.append(rel_path)
            except OSError as e:
                errors.append(f"{rel_path}: {e}")
    return removed, errors


def _dir_size_mb(path):
    if not os.path.isdir(path):
        return 0.0
    total = 0
    for root, _, files in os.walk(path):
        for fn in files:
            try:
                total += os.path.getsize(os.path.join(root, fn))
            except OSError:
                pass
    return round(total / (1024 * 1024), 1)


def _get_hardware_telemetry():
    hw = detect_hardware_profile(probe_ollama=False)
    return {
        "psutil_available": hw.get("psutil_available", False),
        "cpu_percent": hw.get("cpu_percent"),
        "ram_used_gb": hw.get("ram_used_gb"),
        "ram_total_gb": hw.get("ram_total_gb"),
        "ram_percent": hw.get("ram_percent"),
        "silicon": hw.get("silicon", "CPU"),
        "openvino_devices": hw.get("openvino_devices", ["CPU"]),
        "nvidia_gpu_name": hw.get("nvidia_gpu_name"),
        "nvidia_vram_gb": hw.get("nvidia_vram_gb"),
        "recommended_tier": hw.get("recommended_tier"),
        "recommended_tier_label": hw.get("recommended_tier_label"),
        "recommended_model_key": hw.get("recommended_model_key"),
        "recommended_ollama_model": hw.get("recommended_ollama_model"),
    }


class JobManager:
    """Runs background pipeline subprocesses safely and streams logs over SSE + disk logs."""

    def __init__(self, base_dir=BASE_DIR):
        self.base_dir = base_dir
        self.logs_dir = getattr(config, "LOGS_DIR", os.path.join(base_dir, "logs"))
        self._lock = threading.RLock()
        self._proc = None
        self._cancel_requested = False
        self._subscribers = []
        self.log_buffer = deque(maxlen=4000)
        self.state = {
            "id": None,
            "type": None,
            "label": "Idle",
            "status": "idle",
            "exit_code": None,
            "started_at": None,
            "finished_at": None,
            "log_file": None,
            "step_index": 0,
            "total_steps": 0,
        }

    def get_snapshot(self):
        with self._lock:
            return {
                "job": dict(self.state),
                "logs": list(self.log_buffer),
            }

    def subscribe(self):
        q = Queue(maxsize=500)
        with self._lock:
            self._subscribers.append(q)
        return q

    def unsubscribe(self, q):
        with self._lock:
            if q in self._subscribers:
                self._subscribers.remove(q)

    def _broadcast(self, event_name, payload):
        with self._lock:
            subs = list(self._subscribers)
        msg = {"event": event_name, "data": payload}
        for q in subs:
            try:
                q.put_nowait(msg)
            except Exception:
                pass

    def _append_log(self, line, log_fh=None):
        with self._lock:
            self.log_buffer.append(line)
        if log_fh is not None:
            try:
                log_fh.write(line + "\n")
                log_fh.flush()
            except Exception:
                pass
        self._broadcast("log", {"line": line})

    def _build_steps(
        self,
        job_type,
        avoid_artists=None,
        avoid_genres=None,
        force_reembed=False,
        era_start=None,
        era_end=None,
        projection=None,
        provider=None,
    ):
        s1_dir = os.path.join(self.base_dir, "01_Data_Extractor")
        s2_dir = os.path.join(self.base_dir, "02_Vector_Core")

        if provider is not None and str(provider).strip():
            eff_provider = str(provider).strip().lower()
            if eff_provider not in ("spotify", "ytmusic", "unified"):
                raise ValueError(f"Unsupported provider: {eff_provider}")
        else:
            eff_provider = str(get_effective_preferences().get("active_provider", "spotify")).strip().lower()
            if eff_provider not in ("spotify", "ytmusic", "unified"):
                eff_provider = "spotify"

        vec_argv = [sys.executable, os.path.join(s2_dir, "npu_vectorizer.py"), "--provider", eff_provider]
        if force_reembed:
            vec_argv.append("--force")

        clus_argv = [sys.executable, os.path.join(s2_dir, "taste_clusterer.py"), "--provider", eff_provider]
        if avoid_artists is not None:
            clus_argv.extend(["--avoid-artists", ",".join(normalize_avoid_list(avoid_artists))])
        if avoid_genres is not None:
            clus_argv.extend(["--avoid-genres", ",".join(normalize_avoid_list(avoid_genres))])
        if era_start:
            val = str(era_start).strip()
            if not _DATE_OR_YEAR_RE.match(val):
                raise ValueError(f"Invalid era_start format: {val!r} (expected YYYY or YYYY-MM-DD)")
            yr = int(val[:4])
            if not (1900 <= yr <= 2100):
                raise ValueError(f"era_start year {yr} out of range (1900-2100)")
            clus_argv.extend(["--era-start", str(yr)])
        if era_end:
            val = str(era_end).strip()
            if not _DATE_OR_YEAR_RE.match(val):
                raise ValueError(f"Invalid era_end format: {val!r} (expected YYYY or YYYY-MM-DD)")
            yr = int(val[:4])
            if not (1900 <= yr <= 2100):
                raise ValueError(f"era_end year {yr} out of range (1900-2100)")
            clus_argv.extend(["--era-end", str(yr)])

        map_argv = [sys.executable, os.path.join(s2_dir, "view_taste_map.py")]
        if projection:
            proj = str(projection).strip().lower()
            if proj not in ("pca", "umap"):
                raise ValueError(f"Unsupported projection: {proj}")
            map_argv.extend(["--projection", proj])

        step_s1 = ([sys.executable, os.path.join(s1_dir, "stage1_extractor.py")], s1_dir, "Stage 1 (Spotify Extractor)")
        step_s1_ytm = ([sys.executable, os.path.join(s1_dir, "ytmusic_extractor.py")], s1_dir, "Stage 1 (YouTube Music Extractor)")
        step_s2a = (vec_argv, s2_dir, "Stage 2a (Incremental Vectorizer)")
        step_s2b = (clus_argv, s2_dir, "Stage 2b (Taste Clusterer)")
        step_map = (map_argv, s2_dir, "Interactive Taste Map Builder")
        step_hist = ([sys.executable, os.path.join(s1_dir, "history_importer.py")], s1_dir, "Extended History Importer")
        step_auto = ([sys.executable, os.path.join(self.base_dir, "automation_runner.py"), "--run-now"], self.base_dir, "Weekly Playlist Auto-Refresh")

        if eff_provider == "ytmusic":
            full_steps = [step_s1_ytm, step_s2a, step_s2b, step_map]
        elif eff_provider == "unified":
            full_steps = [step_s1, step_s1_ytm, step_s2a, step_s2b, step_map]
        else:
            full_steps = [step_s1, step_s2a, step_s2b, step_map]

        mapping = {
            "stage1": ("Stage 1 (Spotify Sync)", [step_s1]),
            "stage1_ytm": ("Stage 1 (YouTube Music Sync)", [step_s1_ytm]),
            "stage2": ("Stage 2 (Vector Core & Clusters)", [step_s2a, step_s2b, step_map]),
            "tastemap": ("Taste Map Rebuild", [step_map]),
            "history": ("Extended History Import", [step_hist]),
            "full_sync": ("Full Pipeline Sync", full_steps),
            "full_sync_unified": ("Full Unified Sync (Spotify + YT Music)", [step_s1, step_s1_ytm, step_s2a, step_s2b, step_map]),
            "weekly_automation": ("Weekly Auto-Refresh", [step_auto]),
        }
        return mapping.get(job_type)

    def start_job(
        self,
        job_type,
        avoid_artists=None,
        avoid_genres=None,
        force_reembed=False,
        era_start=None,
        era_end=None,
        projection=None,
        provider=None,
    ):
        built = self._build_steps(
            job_type,
            avoid_artists=avoid_artists,
            avoid_genres=avoid_genres,
            force_reembed=force_reembed,
            era_start=era_start,
            era_end=era_end,
            projection=projection,
            provider=provider,
        )
        if built is None:
            raise ValueError(f"Unknown job type: {job_type}")

        job_label, steps = built
        with self._lock:
            if self.state["status"] == "running":
                raise RuntimeError("Another pipeline job is already running.")

            os.makedirs(self.logs_dir, exist_ok=True)
            ts = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d_%H%M%S")
            log_filename = f"{ts}_{job_type}.log"
            log_path = os.path.join(self.logs_dir, log_filename)

            self.log_buffer.clear()
            self._cancel_requested = False
            self.state = {
                "id": f"{ts}_{job_type}",
                "type": job_type,
                "label": job_label,
                "status": "running",
                "exit_code": None,
                "started_at": datetime.datetime.now(datetime.timezone.utc).isoformat().replace("+00:00", "Z"),
                "finished_at": None,
                "log_file": log_filename,
                "step_index": 1,
                "total_steps": len(steps),
            }
            snapshot = dict(self.state)

        self._broadcast("state", snapshot)
        worker = threading.Thread(target=self._run_steps_thread, args=(steps, log_path), daemon=True)
        worker.start()
        return snapshot

    def _run_steps_thread(self, steps, log_path):
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        env["PYTHONIOENCODING"] = "utf-8"

        final_code = 0
        try:
            with open(log_path, "w", encoding="utf-8") as log_fh:
                for idx, (argv, cwd, label) in enumerate(steps, start=1):
                    with self._lock:
                        if self._cancel_requested:
                            break
                        self.state["step_index"] = idx
                        self.state["label"] = f"{label} ({idx}/{len(steps)})"
                        cur_state = dict(self.state)
                    self._broadcast("state", cur_state)
                    self._append_log(f"--- Starting {label} ({idx}/{len(steps)}) ---", log_fh)

                    with self._lock:
                        if self._cancel_requested:
                            break
                        proc = subprocess.Popen(
                            argv,
                            shell=False,
                            cwd=cwd,
                            stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT,
                            env=env,
                            text=True,
                            encoding="utf-8",
                            errors="replace",
                        )
                        self._proc = proc

                    try:
                        with proc.stdout:
                            for line in proc.stdout:
                                self._append_log(line.rstrip("\r\n"), log_fh)
                        final_code = proc.wait()
                    finally:
                        if proc.poll() is None:
                            proc.kill()
                            proc.wait()
                        with self._lock:
                            self._proc = None
                    if final_code != 0:
                        break
        except Exception as exc:
            final_code = 1
            self._append_log(f"❌ Could not run job: {exc}")
        finally:
            with self._lock:
                cancelled = self._cancel_requested
                self.state["exit_code"] = 130 if cancelled else final_code
                self.state["status"] = "cancelled" if cancelled else ("completed" if final_code == 0 else "failed")
                self.state["finished_at"] = datetime.datetime.now(datetime.timezone.utc).isoformat().replace("+00:00", "Z")
                done_state = dict(self.state)

        if done_state["status"] == "cancelled":
            self._append_log("⚠️ Job cancelled by user.", None)
        elif done_state["status"] == "completed":
            self._append_log("✅ Job completed successfully.", None)
        else:
            self._append_log(f"❌ Job failed with exit code {done_state['exit_code']}.", None)
        self._broadcast("state", done_state)

    def cancel_job(self):
        with self._lock:
            if self.state["status"] != "running":
                return False
            self._cancel_requested = True
            proc = self._proc

        if proc is not None:
            try:
                proc.terminate()
                proc.wait(timeout=3)
            except Exception:
                try:
                    proc.kill()
                except Exception:
                    pass
        return True


def _cached_json_summary(app_ctx, path, extractor_fn, default_val):
    """Cache compact metadata summaries keyed by (mtime_ns, size) to avoid re-parsing multi-MB JSONs."""
    if not path:
        return default_val
    try:
        st = os.stat(path)
        sig = (st.st_mtime_ns, st.st_size)
    except OSError:
        return default_val

    lock = app_ctx.setdefault("_status_cache_lock", threading.Lock())
    cache = app_ctx.setdefault("_status_file_cache", {})
    with lock:
        hit = cache.get(path)
        if hit is not None and hit[0] == sig:
            return hit[1]

    data = load_json_safe(path)
    summary = extractor_fn(data)
    with lock:
        cache[path] = (sig, summary)
    return summary


def _extract_spotify_stage1_summary(p_data):
    if not isinstance(p_data, dict):
        return {
            "ready": False,
            "tracks": 0,
            "saved_albums": 0,
            "followed_artists": 0,
            "audio_features": 0,
            "generated_at": None,
        }
    ts = p_data.get("taste_summary") or {}
    return {
        "ready": True,
        "tracks": ts.get("total_unique_tracks_analyzed", 0),
        "saved_albums": ts.get("total_saved_albums", len(p_data.get("saved_albums") or [])),
        "followed_artists": ts.get("total_followed_artists", len(p_data.get("followed_artists") or [])),
        "audio_features": len(p_data.get("audio_features") or {}),
        "generated_at": p_data.get("generated_at"),
    }


def _extract_ytm_stage1_summary(ytm_data):
    if not isinstance(ytm_data, dict):
        return {"ready": False, "tracks": 0, "audio_features": 0, "generated_at": None}
    ts = ytm_data.get("taste_summary") or {}
    return {
        "ready": True,
        "tracks": ts.get("total_unique_tracks_analyzed", 0),
        "audio_features": len(ytm_data.get("audio_features") or {}),
        "generated_at": ytm_data.get("generated_at"),
    }


def _extract_stage2_summary(sum_data):
    if not isinstance(sum_data, dict):
        return {"valid": False, "clusters": 0, "session_filters": None}
    return {
        "valid": True,
        "clusters": len(sum_data.get("clusters") or []),
        "session_filters": sum_data.get("session_filters"),
    }


def _extract_ext_history_summary(ext_stats):
    if not isinstance(ext_stats, dict):
        return {"imported": False, "unique_tracks": 0}
    return {"imported": bool(ext_stats), "unique_tracks": len(ext_stats)}


def _extract_last_pl_summary(last_pl):
    if not isinstance(last_pl, dict):
        return {"has_last": False, "last_name": None, "last_count": 0}
    tracks = last_pl.get("tracks") or []
    return {
        "has_last": bool(tracks),
        "last_name": last_pl.get("playlist_name"),
        "last_count": len(tracks),
    }


def build_status_payload(app_ctx):
    base_dir = app_ctx["base_dir"]
    warm_session = app_ctx["warm_session"]
    job_manager = app_ctx["job_manager"]

    freshness = compute_pipeline_freshness(base_dir)
    client_id = get_spotify_client_id()
    token_cache = os.path.join(base_dir, ".cache")
    token_cached = os.path.exists(token_cache) and os.path.getsize(token_cache) > 0

    s1_info = _cached_json_summary(
        app_ctx,
        os.path.join(base_dir, "01_Data_Extractor", "user_music_profile.json"),
        _extract_spotify_stage1_summary,
        {"ready": False, "tracks": 0, "saved_albums": 0, "followed_artists": 0, "audio_features": 0, "generated_at": None},
    )
    ytm_info = _cached_json_summary(
        app_ctx,
        os.path.join(base_dir, "01_Data_Extractor", "ytmusic_profile.json"),
        _extract_ytm_stage1_summary,
        {"ready": False, "tracks": 0, "audio_features": 0, "generated_at": None},
    )
    ytm_auth = ytmusic_client.get_ytm_auth_status()
    prefs = get_effective_preferences()
    sum_info = _cached_json_summary(
        app_ctx,
        os.path.join(base_dir, "02_Vector_Core", "taste_profile_summary.json"),
        _extract_stage2_summary,
        {"valid": False, "clusters": 0, "session_filters": None},
    )
    ext_info = _cached_json_summary(
        app_ctx,
        os.path.join(base_dir, "01_Data_Extractor", "extended_history_stats.json"),
        _extract_ext_history_summary,
        {"imported": False, "unique_tracks": 0},
    )
    last_info = _cached_json_summary(
        app_ctx,
        os.path.join(base_dir, "03_Agentic_Engine", "last_generated_playlist.json"),
        _extract_last_pl_summary,
        {"has_last": False, "last_name": None, "last_count": 0},
    )

    ext_dir = os.path.join(base_dir, "extended_history")
    raw_ext_files = 0
    if os.path.isdir(ext_dir):
        try:
            raw_ext_files = sum(
                1
                for fn in os.listdir(ext_dir)
                if fn.lower().endswith(".json") and not fn.lower().startswith("streaming_history_video")
            )
        except OSError:
            raw_ext_files = 0

    studio_status = warm_session.get_status()
    return {
        "active_provider": prefs.get("active_provider", "spotify"),
        "auth": {
            "client_id_configured": bool(client_id),
            "token_cached": token_cached,
            "ytmusic": ytm_auth,
        },
        "pipeline": {
            "freshness": freshness,
            "stage1": s1_info,
            "stage1_ytm": ytm_info,
            "stage2": {
                "ready": sum_info["valid"] and freshness["stage2_exists"],
                "stale": freshness["stage2_stale"],
                "clusters": sum_info["clusters"],
                "session_filters": sum_info["session_filters"],
            },
            "taste_map": {
                "ready": freshness["map_exists"],
                "stale": freshness["map_stale"],
            },
            "stage3": last_info,
            "extended_history": {
                **ext_info,
                "raw_files": raw_ext_files,
            },
        },
        "hardware": _get_hardware_telemetry(),
        "job": job_manager.get_snapshot()["job"],
        "studio": studio_status,
    }


def build_doctor_payload(app_ctx):
    base_dir = app_ctx["base_dir"]
    packages = [
        "requests",
        "spotipy",
        "ytmusicapi",
        "numpy",
        "torch",
        "sklearn",
        "transformers",
        "optimum",
        "openvino",
        "openvino_genai",
        "psutil",
        "plotly",
    ]
    pkg_status = {pkg: importlib.util.find_spec(pkg) is not None for pkg in packages}
    venv_exists = os.path.exists(os.path.join(base_dir, ".venv", "Scripts", "python.exe")) or os.path.exists(
        os.path.join(base_dir, ".venv", "bin", "python")
    )

    rb_cache_path = os.path.join(base_dir, "01_Data_Extractor", ".reccobeats_cache.json")
    rb_health = get_cache_health(cache_path=rb_cache_path)

    llm_cat = get_llm_catalog_status(base_dir=base_dir, probe_ollama=False)
    downloaded_models = [m for m in llm_cat.get("catalog", []) if m.get("downloaded")]
    total_llm_mb = round(sum(float(m.get("size_on_disk_mb") or 0.0) for m in downloaded_models), 1)
    emb_dir = getattr(config, "EMBED_MODEL_DIR", os.path.join(base_dir, "02_Vector_Core", "embed_ov_model"))

    logs_dir = getattr(config, "LOGS_DIR", os.path.join(base_dir, "logs"))
    log_files = []
    if os.path.isdir(logs_dir):
        for fn in sorted(os.listdir(logs_dir), reverse=True)[:20]:
            if _SAFE_LOG_FILE_RE.match(fn):
                fpath = os.path.join(logs_dir, fn)
                log_files.append({"name": fn, "size_kb": round(os.path.getsize(fpath) / 1024.0, 1)})

    return {
        "python_version": sys.version.split()[0],
        "python_executable": sys.executable,
        "venv_exists": venv_exists,
        "packages": pkg_status,
        "hardware": _get_hardware_telemetry(),
        "reccobeats_cache": rb_health,
        "models": {
            "llm_cached": bool(downloaded_models) or llm_cat.get("active", {}).get("llm_backend") != "openvino",
            "llm_size_mb": total_llm_mb,
            "llm_label": llm_cat.get("active", {}).get("label"),
            "llm_backend": llm_cat.get("active", {}).get("llm_backend"),
            "llm_downloaded_count": len(downloaded_models),
            "embed_cached": os.path.isdir(emb_dir) and os.path.exists(os.path.join(emb_dir, "openvino_model.xml")),
            "embed_size_mb": _dir_size_mb(emb_dir),
        },
        "auth": {
            "client_id_set": bool(get_spotify_client_id()),
            "token_cache_exists": os.path.exists(os.path.join(base_dir, ".cache")),
            "ytmusic": ytmusic_client.get_ytm_auth_status(),
        },
        "logs": log_files,
    }


def make_handler(app_ctx):
    class SpotMusHandler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, format, *args):
            pass

        def _validate_request_origin(self, require_same_origin=False):
            host_hdr = self.headers.get("Host")
            if host_hdr is not None and not _is_loopback_host(host_hdr):
                self._send_json(HTTPStatus.FORBIDDEN, {"error": "Forbidden host header."})
                return False
            if not require_same_origin:
                return True

            sec_fetch = str(self.headers.get("Sec-Fetch-Site") or "").strip().lower()
            if sec_fetch == "cross-site":
                self._send_json(HTTPStatus.FORBIDDEN, {"error": "Cross-site request blocked."})
                return False

            origin = self.headers.get("Origin")
            if origin is not None:
                try:
                    po = urllib.parse.urlparse(origin.strip())
                    o_host = (po.hostname or "").strip().lower()
                    if po.scheme.lower() not in ("http", "https") or o_host not in _LOOPBACK_HOSTS:
                        self._send_json(HTTPStatus.FORBIDDEN, {"error": "Cross-origin request blocked."})
                        return False
                except Exception:
                    self._send_json(HTTPStatus.FORBIDDEN, {"error": "Cross-origin request blocked."})
                    return False
            else:
                referer = self.headers.get("Referer")
                if referer is not None:
                    try:
                        pr = urllib.parse.urlparse(referer.strip())
                        r_host = (pr.hostname or "").strip().lower()
                        if pr.scheme.lower() not in ("http", "https") or r_host not in _LOOPBACK_HOSTS:
                            self._send_json(HTTPStatus.FORBIDDEN, {"error": "Cross-origin referer blocked."})
                            return False
                    except Exception:
                        self._send_json(HTTPStatus.FORBIDDEN, {"error": "Cross-origin referer blocked."})
                        return False
            return True

        def _send_security_headers(self):
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("X-Frame-Options", "DENY")
            self.send_header("Referrer-Policy", "strict-origin-when-cross-origin")

        def _send_json(self, status_code, payload):
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status_code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self._send_security_headers()
            self.end_headers()
            self.wfile.write(body)

        def _send_text(self, status_code, text, content_type, filename=None):
            body = text.encode("utf-8")
            self.send_response(status_code)
            self.send_header("Content-Type", f"{content_type}; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            if filename:
                self.send_header("Content-Disposition", f'attachment; filename="{filename}"')
            self.send_header("Cache-Control", "no-store")
            self._send_security_headers()
            self.end_headers()
            self.wfile.write(body)

        def _read_json_body(self):
            length = int(self.headers.get("Content-Length", "0") or 0)
            if length <= 0:
                return {}
            if length > 2 * 1024 * 1024:
                raise ValueError("Request body too large.")
            raw = self.rfile.read(length)
            if not raw:
                return {}
            parsed = json.loads(raw.decode("utf-8"))
            if not isinstance(parsed, dict):
                raise ValueError("JSON body must be an object.")
            return parsed

        def _serve_static(self, rel_path):
            ui_root = os.path.abspath(WEB_UI_DIR)
            clean_rel = "index.html" if rel_path in ("", "/") else rel_path.lstrip("/")
            resolved = os.path.abspath(os.path.join(ui_root, clean_rel))
            try:
                if os.path.commonpath([ui_root, resolved]) != ui_root:
                    self._send_json(HTTPStatus.FORBIDDEN, {"error": "Access denied."})
                    return
            except ValueError:
                self._send_json(HTTPStatus.FORBIDDEN, {"error": "Access denied."})
                return

            if not os.path.isfile(resolved):
                self._send_json(HTTPStatus.NOT_FOUND, {"error": "Not found."})
                return

            ctype, _ = mimetypes.guess_type(resolved)
            ctype = ctype or "application/octet-stream"
            with open(resolved, "rb") as f:
                data = f.read()
            self.send_response(HTTPStatus.OK)
            if ctype.startswith("text/") or ctype == "application/javascript":
                self.send_header("Content-Type", f"{ctype}; charset=utf-8")
            else:
                self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self._send_security_headers()
            self.end_headers()
            self.wfile.write(data)

        def _handle_sse_stream(self):
            jm = app_ctx["job_manager"]
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/event-stream; charset=utf-8")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "close")
            self.end_headers()

            q = jm.subscribe()
            try:
                snap = jm.get_snapshot()
                init_payload = f"event: snapshot\ndata: {json.dumps(snap, ensure_ascii=False)}\n\n".encode("utf-8")
                self.wfile.write(init_payload)
                self.wfile.flush()

                while not app_ctx["stop_event"].is_set():
                    try:
                        msg = q.get(timeout=5.0)
                        frame = f"event: {msg['event']}\ndata: {json.dumps(msg['data'], ensure_ascii=False)}\n\n".encode("utf-8")
                        self.wfile.write(frame)
                        self.wfile.flush()
                    except Empty:
                        self.wfile.write(b": keepalive\n\n")
                        self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError, OSError):
                pass
            finally:
                jm.unsubscribe(q)

        def do_GET(self):
            if not self._validate_request_origin(require_same_origin=False):
                return
            parsed = urllib.parse.urlparse(self.path)
            path = parsed.path
            qs = urllib.parse.parse_qs(parsed.query)

            try:
                if path == "/vendor/plotly.min.js":
                    local_js = find_local_plotly_js()
                    if not local_js or not os.path.isfile(local_js):
                        return self._send_json(HTTPStatus.NOT_FOUND, {"error": "Local plotly.min.js not found."})
                    with open(local_js, "rb") as pf:
                        js_bytes = pf.read()
                    self.send_response(HTTPStatus.OK)
                    self.send_header("Content-Type", "application/javascript; charset=utf-8")
                    self.send_header("Content-Length", str(len(js_bytes)))
                    self.send_header("Cache-Control", "public, max-age=86400")
                    self._send_security_headers()
                    self.end_headers()
                    self.wfile.write(js_bytes)
                    return

                if path == "/api/status":
                    return self._send_json(HTTPStatus.OK, build_status_payload(app_ctx))

                if path == "/api/spotify/now-playing":
                    try:
                        ws = app_ctx["warm_session"]
                        sp = ws._ensure_spotify()
                        track = get_currently_playing_track(sp)
                        return self._send_json(HTTPStatus.OK, {"playing": bool(track), "track": track})
                    except Exception as exc:
                        err_l = str(exc).lower()
                        status = HTTPStatus.FORBIDDEN if ("403" in err_l or "scope" in err_l) else HTTPStatus.BAD_REQUEST
                        return self._send_json(
                            status,
                            {
                                "error": f"Spotify playback unavailable: {exc}. "
                                "Re-run Stage 1 Sync in Dashboard to refresh Spotify permissions."
                            },
                        )

                if path == "/api/spotify/playlists":
                    limit = _parse_int_param(qs, "limit", 50, 1, 200)
                    try:
                        ws = app_ctx["warm_session"]
                        sp = ws._ensure_spotify()
                        playlists = list_user_owned_playlists(sp, limit=limit)
                        return self._send_json(HTTPStatus.OK, {"playlists": playlists})
                    except Exception as exc:
                        err_l = str(exc).lower()
                        status = HTTPStatus.FORBIDDEN if ("403" in err_l or "scope" in err_l) else HTTPStatus.BAD_REQUEST
                        return self._send_json(
                            status,
                            {
                                "error": f"Could not load Spotify playlists: {exc}. "
                                "Re-run Stage 1 Sync in Dashboard to refresh Spotify permissions."
                            },
                        )

                if path == "/api/ytmusic/auth-status":
                    return self._send_json(HTTPStatus.OK, ytmusic_client.get_ytm_auth_status())

                if path == "/api/llm/models":
                    probe_ol = (qs.get("probe_ollama") or ["1"])[0] != "0"
                    return self._send_json(
                        HTTPStatus.OK,
                        get_llm_catalog_status(base_dir=app_ctx["base_dir"], probe_ollama=probe_ol),
                    )

                if path == "/api/ytmusic/playlists":
                    limit = _parse_int_param(qs, "limit", 50, 1, 200)
                    try:
                        ws = app_ctx["warm_session"]
                        ytm = ws._ensure_ytmusic()
                        playlists = ytmusic_client.list_ytm_playlists(ytm, limit=limit)
                        return self._send_json(HTTPStatus.OK, {"playlists": playlists})
                    except Exception as exc:
                        return self._send_json(
                            HTTPStatus.BAD_REQUEST,
                            {"error": f"Could not load YouTube Music playlists: {exc}"},
                        )

                if path == "/api/spotify/queue":
                    try:
                        q_state = app_ctx["warm_session"].get_playback_queue()
                        return self._send_json(HTTPStatus.OK, q_state)
                    except Exception as exc:
                        err_l = str(exc).lower()
                        status = HTTPStatus.FORBIDDEN if ("403" in err_l or "scope" in err_l) else HTTPStatus.BAD_REQUEST
                        return self._send_json(
                            status,
                            {
                                "error": f"Could not read Spotify playback queue: {exc}. "
                                "Ensure Spotify Premium is active and re-run Stage 1 Sync for scopes."
                            },
                        )

                if path == "/api/library/search":
                    q = (qs.get("q") or [""])[0]
                    limit = _parse_int_param(qs, "limit", 15, 1, 50)
                    results = app_ctx["warm_session"].search_local_library(q, limit=limit)
                    return self._send_json(HTTPStatus.OK, {"results": results})

                if path == "/api/feedback":
                    return self._send_json(HTTPStatus.OK, load_user_feedback())

                if path == "/api/taste/export-card":
                    summary_path = os.path.join(app_ctx["base_dir"], "02_Vector_Core", "taste_profile_summary.json")
                    summary = load_json_safe(summary_path)
                    if not isinstance(summary, dict) or not summary.get("master_taste_centroid"):
                        return self._send_json(
                            HTTPStatus.NOT_FOUND,
                            {"error": "Stage 2 taste_profile_summary.json not found. Run Stage 2 first."},
                        )
                    safe_clusters = []
                    for c in summary.get("clusters") or []:
                        if not isinstance(c, dict):
                            continue
                        share = c.get("weighted_share_pct", c.get("weight_percentage", c.get("percentage")))
                        safe_clusters.append(
                            {
                                "cluster_id": c.get("cluster_id"),
                                "vibe_label": c.get("vibe_label"),
                                "track_count": c.get("track_count"),
                                "weighted_share_pct": share,
                                "percentage": share,
                                "top_genres": c.get("top_genres", []),
                                "centroid_vector": c.get("centroid_vector", []),
                            }
                        )
                    card = {
                        "schema_version": 1,
                        "generated_at": summary.get("generated_at"),
                        "master_taste_centroid": summary["master_taste_centroid"],
                        "clusters": safe_clusters,
                    }
                    if (qs.get("download") or ["0"])[0] == "1":
                        return self._send_text(
                            HTTPStatus.OK,
                            json.dumps(card, ensure_ascii=False, indent=2),
                            "application/json",
                            filename="music_taste_card.json",
                        )
                    return self._send_json(HTTPStatus.OK, card)

                if path == "/api/presets":
                    prefs = get_effective_preferences()
                    return self._send_json(HTTPStatus.OK, {"presets": prefs.get("presets", {})})

                if path == "/api/hygiene":
                    s1_dir = os.path.join(app_ctx["base_dir"], "01_Data_Extractor")
                    s2_dir = os.path.join(app_ctx["base_dir"], "02_Vector_Core")
                    return self._send_json(HTTPStatus.OK, analyze_library_hygiene(stage1_dir=s1_dir, stage2_dir=s2_dir))

                if path == "/api/automation":
                    prefs = get_effective_preferences()
                    return self._send_json(
                        HTTPStatus.OK,
                        {
                            "automation": prefs.get("automation") or DEFAULT_AUTOMATION_SETTINGS,
                            "task": query_windows_task(),
                        },
                    )

                if path == "/api/session/exclusions":
                    ws = app_ctx["warm_session"]
                    return self._send_json(
                        HTTPStatus.OK,
                        {
                            "avoid_artists": list(ws.session_avoid_artists),
                            "avoid_genres": list(ws.session_avoid_genres),
                        },
                    )

                if path == "/api/jobs/stream":
                    return self._handle_sse_stream()

                if path == "/api/jobs/snapshot":
                    return self._send_json(HTTPStatus.OK, app_ctx["job_manager"].get_snapshot())

                if path == "/api/jobs/logs":
                    logs_dir = getattr(config, "LOGS_DIR", os.path.join(app_ctx["base_dir"], "logs"))
                    req_file = (qs.get("file") or [None])[0]
                    if req_file:
                        if not _SAFE_LOG_FILE_RE.match(req_file):
                            return self._send_json(HTTPStatus.BAD_REQUEST, {"error": "Invalid log filename."})
                        target = os.path.join(logs_dir, req_file)
                        if not os.path.isfile(target):
                            return self._send_json(HTTPStatus.NOT_FOUND, {"error": "Log file not found."})
                        with open(target, "r", encoding="utf-8", errors="replace") as f:
                            content = f.read()
                        return self._send_json(HTTPStatus.OK, {"file": req_file, "content": content})

                    files = []
                    if os.path.isdir(logs_dir):
                        for fn in sorted(os.listdir(logs_dir), reverse=True)[:30]:
                            if _SAFE_LOG_FILE_RE.match(fn):
                                files.append({"name": fn, "size_kb": round(os.path.getsize(os.path.join(logs_dir, fn)) / 1024.0, 1)})
                    return self._send_json(HTTPStatus.OK, {"files": files})

                if path == "/api/taste-map":
                    map_path = os.path.join(app_ctx["base_dir"], "02_Vector_Core", "taste_map_data.json")
                    data = load_json_safe(map_path)
                    if not isinstance(data, dict):
                        return self._send_json(HTTPStatus.OK, {"available": False})
                    return self._send_json(HTTPStatus.OK, {"available": True, **data})

                if path == "/api/preview-audio":
                    import deezer_audio_client as dz

                    title = (qs.get("title") or [""])[0].strip()
                    artist = (qs.get("artist") or [""])[0].strip()
                    if not title:
                        return self._send_json(HTTPStatus.OK, {"found": False})

                    fp = get_track_fingerprint(title, artist)
                    dz_cache = dz.load_deezer_cache()
                    entry = dz_cache.get(fp)
                    dirty = False
                    if not isinstance(entry, dict) or (not entry.get("found") and entry.get("v") != 2):
                        entry = dz.search_deezer_track(title, artist, cache=None)
                        dz_cache[fp] = entry
                        dirty = True

                    if not entry.get("found"):
                        if dirty:
                            dz.save_deezer_cache(dz_cache)
                        return self._send_json(HTTPStatus.OK, {"found": False})

                    preview_url = entry.get("preview_url") or ""
                    exp_match = re.search(r"exp=(\d+)", preview_url)
                    is_expired = bool(exp_match and int(exp_match.group(1)) <= int(time.time()) + 60)
                    if not preview_url or is_expired:
                        fresh = dz.refresh_deezer_preview_url(entry.get("deezer_id"))
                        if fresh:
                            preview_url = fresh
                            dz_cache[fp] = dict(entry, preview_url=fresh)
                            dirty = True

                    if dirty:
                        dz.save_deezer_cache(dz_cache)

                    return self._send_json(
                        HTTPStatus.OK,
                        {
                            "found": bool(preview_url),
                            "preview_url": preview_url,
                            "title": entry.get("title") or title,
                            "artist": entry.get("artist") or artist,
                            "album": entry.get("album") or "",
                        },
                    )

                if path == "/api/playlists":
                    last_pl = load_json_safe(LAST_PL_PATH)
                    archived = list_archived_playlists()
                    return self._send_json(
                        HTTPStatus.OK,
                        {
                            "last": last_pl if isinstance(last_pl, dict) else None,
                            "archived": archived,
                        },
                    )

                if path.startswith("/api/playlists/"):
                    rest = path[len("/api/playlists/") :]
                    if rest.endswith("/download"):
                        pid = rest[: -len("/download")]
                        fmt = ((qs.get("format") or ["csv"])[0]).lower()
                        data = load_json_safe(LAST_PL_PATH) if pid == "last" else load_archived_playlist(pid)
                        if not isinstance(data, dict):
                            return self._send_json(HTTPStatus.NOT_FOUND, {"error": "Playlist not found."})
                        slug = re.sub(r"[^A-Za-z0-9_-]+", "_", str(data.get("playlist_name") or "playlist")).strip("_") or "playlist"
                        if fmt == "csv":
                            return self._send_text(HTTPStatus.OK, format_playlist_csv(data), "text/csv", filename=f"{slug}.csv")
                        if fmt == "m3u":
                            return self._send_text(HTTPStatus.OK, format_playlist_m3u(data), "audio/x-mpegurl", filename=f"{slug}.m3u")
                        if fmt in ("md", "markdown"):
                            return self._send_text(HTTPStatus.OK, format_playlist_markdown(data), "text/markdown", filename=f"{slug}.md")
                        return self._send_json(HTTPStatus.BAD_REQUEST, {"error": f"Unsupported format: {fmt}"})

                    pid = rest
                    data = load_json_safe(LAST_PL_PATH) if pid == "last" else load_archived_playlist(pid)
                    if not isinstance(data, dict):
                        return self._send_json(HTTPStatus.NOT_FOUND, {"error": "Playlist not found."})
                    return self._send_json(HTTPStatus.OK, data)

                if path == "/api/analytics":
                    return self._send_json(HTTPStatus.OK, compute_history_analytics())

                if path == "/api/preferences":
                    prefs = get_effective_preferences()
                    cid = get_spotify_client_id() or ""
                    masked = f"{cid[:4]}...{cid[-4:]}" if len(cid) > 8 else ("Set" if cid else "")
                    return self._send_json(
                        HTTPStatus.OK,
                        {
                            "preferences": prefs,
                            "spotify_client_id_masked": masked,
                            "has_client_id": bool(cid),
                        },
                    )

                if path == "/api/doctor":
                    return self._send_json(HTTPStatus.OK, build_doctor_payload(app_ctx))

                if not path.startswith("/api/"):
                    return self._serve_static(path)

                return self._send_json(HTTPStatus.NOT_FOUND, {"error": "Unknown endpoint."})
            except ValueError as exc:
                return self._send_json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
            except Exception as exc:
                return self._send_json(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": str(exc)})

        def do_POST(self):
            if not self._validate_request_origin(require_same_origin=True):
                return
            parsed = urllib.parse.urlparse(self.path)
            path = parsed.path

            if path == "/api/history/upload":
                try:
                    length = int(self.headers.get("Content-Length", "0") or 0)
                except ValueError:
                    return self._send_json(HTTPStatus.BAD_REQUEST, {"error": "Invalid Content-Length header."})
                if length <= 0:
                    return self._send_json(HTTPStatus.BAD_REQUEST, {"error": "Empty upload body."})
                if length > MAX_HISTORY_UPLOAD_BYTES:
                    return self._send_json(
                        getattr(HTTPStatus, "CONTENT_TOO_LARGE", 413),
                        {"error": f"Upload exceeds {MAX_HISTORY_UPLOAD_BYTES // (1024 * 1024)} MB limit."},
                    )
                try:
                    raw_bytes = self.rfile.read(length)
                    filename = urllib.parse.unquote(self.headers.get("X-Filename", "extended_history.zip") or "extended_history.zip")
                    ext_dest = os.path.join(app_ctx["base_dir"], "extended_history")
                    result = extract_uploaded_history_bytes(filename, raw_bytes, dest_dir=ext_dest)
                    return self._send_json(HTTPStatus.OK, {"ok": True, **result})
                except ValueError as exc:
                    return self._send_json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
                except Exception as exc:
                    return self._send_json(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": str(exc)})

            try:
                body = self._read_json_body()
            except ValueError as exc:
                return self._send_json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
            except Exception:
                return self._send_json(HTTPStatus.BAD_REQUEST, {"error": "Invalid JSON body."})

            try:
                if path == "/api/session/exclusions":
                    ws = app_ctx["warm_session"]
                    updated = ws.set_session_exclusions(
                        avoid_artists=body.get("avoid_artists"),
                        avoid_genres=body.get("avoid_genres"),
                    )
                    if body.get("persist"):
                        save_user_preferences(updated)
                    return self._send_json(HTTPStatus.OK, updated)

                if path == "/api/jobs/run":
                    job_type = body.get("job")
                    ws = app_ctx["warm_session"]
                    try:
                        job_snap = app_ctx["job_manager"].start_job(
                            job_type,
                            avoid_artists=ws.session_avoid_artists,
                            avoid_genres=ws.session_avoid_genres,
                            force_reembed=bool(body.get("force_reembed", False)),
                            era_start=body.get("era_start"),
                            era_end=body.get("era_end"),
                            projection=body.get("projection"),
                            provider=body.get("provider"),
                        )
                        return self._send_json(HTTPStatus.ACCEPTED, {"job": job_snap})
                    except RuntimeError as exc:
                        return self._send_json(HTTPStatus.CONFLICT, {"error": str(exc)})
                    except ValueError as exc:
                        return self._send_json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})

                if path == "/api/jobs/cancel":
                    cancelled = app_ctx["job_manager"].cancel_job()
                    return self._send_json(HTTPStatus.OK, {"cancelled": cancelled})

                if path == "/api/system/clean":
                    removed, errors = clean_system_files(app_ctx["base_dir"])
                    rb_removed = 0
                    if body.get("purge_reccobeats"):
                        rb_path = os.path.join(app_ctx["base_dir"], "01_Data_Extractor", ".reccobeats_cache.json")
                        rb_removed, _ = purge_poisoned_and_not_found(
                            cache_path=rb_path,
                            include_not_found=bool(body.get("include_not_found", False)),
                        )
                    ws = app_ctx["warm_session"]
                    with ws._lock:
                        ws.sp = None
                        ws.ytm = None
                        ws._vectors_sig = None
                        ws.last_playlist = None
                        ws.last_candidates = []
                    with app_ctx.setdefault("_status_cache_lock", threading.Lock()):
                        app_ctx.setdefault("_status_file_cache", {}).clear()
                    return self._send_json(
                        HTTPStatus.OK,
                        {"removed": removed, "errors": errors, "reccobeats_purged": rb_removed},
                    )

                if path == "/api/studio/warmup":
                    ws = app_ctx["warm_session"]
                    status = ws.warm_up(load_llm=bool(body.get("load_llm", False)))
                    return self._send_json(HTTPStatus.OK, status)

                if path == "/api/studio/unload-llm":
                    ws = app_ctx["warm_session"]
                    return self._send_json(HTTPStatus.OK, ws.unload_llm())

                if path == "/api/studio/generate":
                    ws = app_ctx["warm_session"]
                    res = ws.generate(
                        prompt=body.get("prompt", ""),
                        fast=bool(body.get("fast", False)),
                        size=body.get("size"),
                        local_only=bool(body.get("local_only", False)),
                        cluster_id=body.get("cluster_id"),
                        flow_mode=body.get("flow_mode") or "mmr",
                        clean_only=bool(body.get("clean_only", False)),
                        deep_cuts=bool(body.get("deep_cuts", False)),
                        avoid_artists=body.get("avoid_artists"),
                        avoid_genres=body.get("avoid_genres"),
                        save_to_archive=True,
                        provider=body.get("provider"),
                    )
                    return self._send_json(HTTPStatus.OK, res)

                if path == "/api/studio/swap":
                    ws = app_ctx["warm_session"]
                    res = ws.swap_track(
                        drop_index=body.get("drop_index"),
                        flow_mode=body.get("flow_mode"),
                    )
                    return self._send_json(HTTPStatus.OK, res)

                if path == "/api/studio/resequence":
                    ws = app_ctx["warm_session"]
                    payload = ws.resequence_current(body.get("flow_mode") or "mmr")
                    return self._send_json(HTTPStatus.OK, {"playlist": payload})

                if path == "/api/studio/continue-playing":
                    ws = app_ctx["warm_session"]
                    res = ws.continue_from_now_playing(
                        size=body.get("size") or 25,
                        fast=bool(body.get("fast", False)),
                        flow_mode=body.get("flow_mode") or "harmonic",
                        clean_only=bool(body.get("clean_only", False)),
                        deep_cuts=bool(body.get("deep_cuts", False)),
                    )
                    return self._send_json(HTTPStatus.OK, res)

                if path == "/api/studio/vibe-bridge":
                    ws = app_ctx["warm_session"]
                    res = ws.generate_vibe_bridge(
                        start_query=body.get("start_query") or "",
                        end_query=body.get("end_query") or "",
                        steps=int(body.get("steps") or 20),
                        clean_only=bool(body.get("clean_only", False)),
                    )
                    return self._send_json(HTTPStatus.OK, res)

                if path == "/api/studio/refine":
                    ws = app_ctx["warm_session"]
                    res = ws.refine_playlist(
                        instruction=body.get("instruction") or "",
                        fast=bool(body.get("fast", False)),
                    )
                    return self._send_json(HTTPStatus.OK, res)

                if path == "/api/studio/edit":
                    ws = app_ctx["warm_session"]
                    updated_pl = ws.update_current_playlist(
                        playlist_name=body.get("playlist_name"),
                        playlist_description=body.get("playlist_description"),
                        order_indices=body.get("order_indices"),
                        remove_indices=body.get("remove_indices"),
                    )
                    return self._send_json(HTTPStatus.OK, {"playlist": updated_pl})

                if path == "/api/studio/taste-blend":
                    ws = app_ctx["warm_session"]
                    friend_card = body.get("friend_card")
                    if not isinstance(friend_card, dict):
                        raise ValueError("friend_card must be a JSON object.")
                    res = ws.generate_blend(
                        friend_summary=friend_card,
                        blend_ratio=float(body.get("blend_ratio", 0.5)),
                        prompt=body.get("prompt") or "",
                        size=int(body.get("size") or 30),
                        fast=bool(body.get("fast", True)),
                        flow_mode=body.get("flow_mode") or "mmr",
                        clean_only=bool(body.get("clean_only", False)),
                    )
                    return self._send_json(HTTPStatus.OK, res)

                if path == "/api/ytmusic/auth/browser":
                    headers_raw = body.get("headers") or ""
                    res = ytmusic_client.save_ytm_browser_headers(headers_raw)
                    app_ctx["warm_session"].ytm = None
                    return self._send_json(
                        HTTPStatus.OK,
                        {"ok": True, "auth": ytmusic_client.get_ytm_auth_status(), **res},
                    )

                if path == "/api/ytmusic/auth/oauth/start":
                    client_id = str(body.get("client_id") or "").strip()
                    client_secret = str(body.get("client_secret") or "").strip()
                    if client_id and client_secret:
                        ytmusic_client.save_ytm_oauth_client(client_id, client_secret)
                    res = ytmusic_client.start_ytm_oauth_device_flow(
                        client_id=client_id or None,
                        client_secret=client_secret or None,
                    )
                    return self._send_json(HTTPStatus.OK, res)

                if path == "/api/ytmusic/auth/oauth/complete":
                    device_code = str(body.get("device_code") or "").strip()
                    client_id = str(body.get("client_id") or "").strip()
                    client_secret = str(body.get("client_secret") or "").strip()
                    res = ytmusic_client.complete_ytm_oauth_device_flow(
                        device_code=device_code,
                        client_id=client_id or None,
                        client_secret=client_secret or None,
                    )
                    app_ctx["warm_session"].ytm = None
                    return self._send_json(
                        HTTPStatus.OK,
                        {"ok": True, "auth": ytmusic_client.get_ytm_auth_status(), **res},
                    )

                if path == "/api/ytmusic/rate":
                    vid = extract_ytm_video_id(body.get("video_id"))
                    if not vid:
                        raise ValueError("Invalid YouTube Music video_id.")
                    rating = str(body.get("rating") or "LIKE").strip().upper()
                    if rating not in ("LIKE", "DISLIKE", "INDIFFERENT"):
                        raise ValueError("rating must be LIKE, DISLIKE, or INDIFFERENT.")
                    ws = app_ctx["warm_session"]
                    ytm = ws._ensure_ytmusic()
                    res = ytmusic_client.rate_ytm_song(ytm, vid, rating=rating)
                    return self._send_json(HTTPStatus.OK, res)

                if path.startswith("/api/playlists/") and path.endswith("/export-spotify"):
                    pid = path[len("/api/playlists/") : -len("/export-spotify")]
                    if pid != "last" and not _ARCHIVE_ID_RE.match(pid):
                        raise ValueError("Invalid playlist ID.")
                    ws = app_ctx["warm_session"]
                    target_id = body.get("target_playlist_id")
                    if target_id:
                        target_id = str(target_id).strip()
                        if not _SPOTIFY_ID_RE.match(target_id):
                            raise ValueError("Invalid target_playlist_id.")
                    res = ws.export_playlist(
                        playlist_id=pid,
                        public=bool(body.get("public", False)),
                        target_playlist_id=target_id or None,
                    )
                    if not res:
                        return self._send_json(HTTPStatus.BAD_REQUEST, {"error": "Spotify export failed. Check auth/logs."})
                    return self._send_json(HTTPStatus.OK, {"ok": True, "spotify_url": res if isinstance(res, str) else None})

                if path.startswith("/api/playlists/") and path.endswith("/export-ytmusic"):
                    pid = path[len("/api/playlists/") : -len("/export-ytmusic")]
                    if pid != "last" and not _ARCHIVE_ID_RE.match(pid):
                        raise ValueError("Invalid playlist ID.")
                    target_id = body.get("target_playlist_id")
                    if target_id:
                        target_id = str(target_id).strip()
                        if not _YTM_PLAYLIST_ID_RE.match(target_id):
                            raise ValueError("Invalid target_playlist_id.")
                    privacy = str(body.get("privacy") or "PRIVATE").strip().upper()
                    if privacy not in ("PRIVATE", "PUBLIC", "UNLISTED"):
                        raise ValueError("privacy must be PRIVATE, PUBLIC, or UNLISTED.")
                    ws = app_ctx["warm_session"]
                    res = ws.export_ytmusic_playlist(
                        playlist_id=pid,
                        privacy=privacy,
                        target_playlist_id=target_id or None,
                    )
                    return self._send_json(HTTPStatus.OK, {"ok": True, **res})

                if path.startswith("/api/playlists/") and path.endswith("/transfer"):
                    pid = path[len("/api/playlists/") : -len("/transfer")]
                    if pid != "last" and not _ARCHIVE_ID_RE.match(pid):
                        raise ValueError("Invalid playlist ID.")
                    target_provider = str(body.get("target_provider") or "").strip().lower()
                    if target_provider not in ("spotify", "ytmusic"):
                        raise ValueError("target_provider must be 'spotify' or 'ytmusic'.")
                    target_id = body.get("target_playlist_id")
                    if target_id:
                        target_id = str(target_id).strip()
                        if target_provider == "spotify" and not _SPOTIFY_ID_RE.match(target_id):
                            raise ValueError("Invalid Spotify target_playlist_id.")
                        if target_provider == "ytmusic" and not _YTM_PLAYLIST_ID_RE.match(target_id):
                            raise ValueError("Invalid YouTube Music target_playlist_id.")
                    privacy = str(body.get("privacy") or "PRIVATE").strip().upper()
                    if privacy not in ("PRIVATE", "PUBLIC", "UNLISTED"):
                        privacy = "PRIVATE"
                    ws = app_ctx["warm_session"]
                    res = ws.transfer_playlist(
                        playlist_id=pid,
                        target_provider=target_provider,
                        public=bool(body.get("public", False)),
                        privacy=privacy,
                        target_playlist_id=target_id or None,
                    )
                    return self._send_json(HTTPStatus.OK, {"ok": bool(res.get("url")), **res})

                if path == "/api/feedback":
                    track = body.get("track")
                    if not isinstance(track, dict):
                        raise ValueError("track must be an object.")
                    rating = str(body.get("rating", "")).strip().lower()
                    updated_fb = record_track_feedback(track, rating)
                    spotify_synced = False
                    ytmusic_synced = False
                    if rating == "like":
                        prefs = get_effective_preferences()
                        ws = app_ctx["warm_session"]
                        sp_cand = track.get("spotify_id") if (track.get("spotify_id") and not is_valid_ytm_id(str(track.get("spotify_id")))) else (
                            track.get("id") if (track.get("id") and not is_valid_ytm_id(str(track["id"]))) else None
                        )
                        if sp_cand and (body.get("sync_spotify") or prefs.get("sync_likes_to_spotify")):
                            try:
                                sp = ws._ensure_spotify()
                                sync_res = save_to_spotify_library(sp, [f"spotify:track:{sp_cand}"])
                                spotify_synced = sync_res.get("count", 0) > 0
                            except Exception:
                                spotify_synced = False
                        vid = extract_ytm_video_id(track.get("video_id") or track.get("id"))
                        if vid and (body.get("sync_ytmusic") or prefs.get("sync_likes_to_spotify")) and ytmusic_client.get_ytm_auth_status().get("configured"):
                            try:
                                ytm = ws._ensure_ytmusic()
                                ytm_res = ytmusic_client.rate_ytm_song(ytm, vid, rating="LIKE")
                                ytmusic_synced = bool(ytm_res.get("ok"))
                            except Exception:
                                ytmusic_synced = False
                    return self._send_json(
                        HTTPStatus.OK,
                        {"feedback": updated_fb, "spotify_synced": spotify_synced, "ytmusic_synced": ytmusic_synced},
                    )

                if path == "/api/studio/queue":
                    ws = app_ctx["warm_session"]
                    res = ws.queue_current_playlist(
                        playlist_id=body.get("playlist_id"),
                        max_items=int(body.get("max_items") or 30),
                    )
                    return self._send_json(HTTPStatus.OK, res)

                if path == "/api/studio/play":
                    ws = app_ctx["warm_session"]
                    res = ws.play_current_playlist(
                        playlist_id=body.get("playlist_id"),
                        device_id=body.get("device_id"),
                    )
                    return self._send_json(HTTPStatus.OK, res)

                if path == "/api/spotify/library":
                    action = str(body.get("action") or "save").strip().lower()
                    uris = body.get("uris")
                    if not isinstance(uris, list) or not uris:
                        raise ValueError("uris must be a non-empty list.")
                    ws = app_ctx["warm_session"]
                    sp = ws._ensure_spotify()
                    if action == "remove":
                        res = remove_from_spotify_library(sp, uris)
                    elif action == "save":
                        res = save_to_spotify_library(sp, uris)
                    else:
                        raise ValueError(f"Unsupported library action: {action}")
                    return self._send_json(HTTPStatus.OK, {"action": action, **res})

                if path == "/api/radar/releases":
                    ws = app_ctx["warm_session"]
                    res = ws.get_release_radar(
                        days=int(body.get("days") or 45),
                        max_artists=int(body.get("max_artists") or 20),
                        save_as_playlist=bool(body.get("save_as_playlist", False)),
                        flow_mode=body.get("flow_mode") or "mmr",
                    )
                    return self._send_json(HTTPStatus.OK, res)

                if path == "/api/clusters/rename":
                    raw_cid = body.get("cluster_id")
                    if raw_cid is None or raw_cid == "":
                        raise ValueError("cluster_id is required.")
                    cid_int = int(raw_cid)
                    cid_str = str(cid_int)
                    label = str(body.get("label") or "").strip()[:80]
                    prefs = get_effective_preferences()
                    labels = dict(prefs.get("cluster_labels") or {})
                    if label:
                        labels[cid_str] = label
                    else:
                        labels.pop(cid_str, None)
                    updated_prefs = save_user_preferences({"cluster_labels": labels})

                    # Keep Stage 2 artifacts and warm session clusters synchronized immediately
                    s2_dir = os.path.join(app_ctx["base_dir"], "02_Vector_Core")
                    sum_path = os.path.join(s2_dir, "taste_profile_summary.json")
                    summary = load_json_safe(sum_path)
                    if isinstance(summary, dict) and isinstance(summary.get("clusters"), list):
                        for c in summary["clusters"]:
                            if isinstance(c, dict) and int(c.get("cluster_id", -999)) == cid_int:
                                c.setdefault("auto_vibe_label", c.get("vibe_label", f"Cluster {cid_int}"))
                                c["vibe_label"] = label or c.get("auto_vibe_label", f"Cluster {cid_int}")
                        atomic_write_json(sum_path, summary, indent=2)

                    map_path = os.path.join(s2_dir, "taste_map_data.json")
                    map_data = load_json_safe(map_path)
                    if isinstance(map_data, dict) and isinstance(map_data.get("clusters"), list):
                        for c in map_data["clusters"]:
                            if isinstance(c, dict) and int(c.get("cluster_id", -999)) == cid_int:
                                c.setdefault("auto_vibe_label", c.get("vibe_label", f"Cluster {cid_int}"))
                                c["vibe_label"] = label or c.get("auto_vibe_label", f"Cluster {cid_int}")
                        atomic_write_json(map_path, map_data, indent=None)

                    ws = app_ctx["warm_session"]
                    with ws._lock:
                        for c in ws.clusters or []:
                            if isinstance(c, dict) and int(c.get("cluster_id", -999)) == cid_int:
                                c.setdefault("auto_vibe_label", c.get("vibe_label", f"Cluster {cid_int}"))
                                c["vibe_label"] = label or c.get("auto_vibe_label", f"Cluster {cid_int}")

                    return self._send_json(HTTPStatus.OK, {"cluster_labels": updated_prefs["cluster_labels"]})

                if path == "/api/presets":
                    name = str(body.get("name") or "").strip()[:64]
                    preset = body.get("preset")
                    if not name or not isinstance(preset, dict):
                        raise ValueError("Preset 'name' and object 'preset' are required.")
                    prefs = get_effective_preferences()
                    merged = dict(prefs.get("presets") or {})
                    merged[name] = preset
                    updated_prefs = save_user_preferences({"presets": merged})
                    return self._send_json(HTTPStatus.OK, {"presets": updated_prefs["presets"]})

                if path == "/api/hygiene/dedupe-playlist":
                    playlist_id = str(body.get("playlist_id") or "").strip()
                    if not _SPOTIFY_ID_RE.match(playlist_id):
                        raise ValueError("Invalid Spotify playlist_id.")
                    ws = app_ctx["warm_session"]
                    sp = ws._ensure_spotify()
                    res = dedupe_spotify_playlist(sp, playlist_id)
                    return self._send_json(HTTPStatus.OK, res)

                if path == "/api/automation":
                    action = str(body.get("action") or "save").strip().lower()
                    raw_auto = body.get("automation") if isinstance(body.get("automation"), dict) else body
                    prefs = get_effective_preferences()
                    cur_auto = dict(prefs.get("automation") or DEFAULT_AUTOMATION_SETTINGS)
                    for k in DEFAULT_AUTOMATION_SETTINGS:
                        if k in raw_auto:
                            cur_auto[k] = raw_auto[k]
                    updated_prefs = save_user_preferences({"automation": cur_auto})
                    saved_auto = updated_prefs["automation"]

                    if action == "install_task":
                        task_res = register_windows_task(
                            day_of_week=saved_auto.get("day_of_week", "SUN"),
                            time_str=saved_auto.get("time", "09:00"),
                        )
                        return self._send_json(
                            HTTPStatus.OK,
                            {
                                "automation": get_effective_preferences()["automation"],
                                "task_result": task_res,
                                "task": query_windows_task(),
                            },
                        )
                    if action == "remove_task":
                        task_res = remove_windows_task()
                        return self._send_json(
                            HTTPStatus.OK,
                            {
                                "automation": get_effective_preferences()["automation"],
                                "task_result": task_res,
                                "task": query_windows_task(),
                            },
                        )
                    if action == "run_now":
                        try:
                            job_snap = app_ctx["job_manager"].start_job("weekly_automation")
                            return self._send_json(HTTPStatus.ACCEPTED, {"automation": saved_auto, "job": job_snap})
                        except RuntimeError as exc:
                            return self._send_json(HTTPStatus.CONFLICT, {"error": str(exc)})

                    return self._send_json(
                        HTTPStatus.OK,
                        {"automation": saved_auto, "task": query_windows_task()},
                    )

                if path == "/api/analytics/forgotten-favorites":
                    ws = app_ctx["warm_session"]
                    use_ex = body.get("use_session_exclusions", True)
                    avoid_art = ws.session_avoid_artists if use_ex else []
                    avoid_gen = ws.session_avoid_genres if use_ex else []
                    tracks = find_forgotten_favorites(
                        min_plays=int(body.get("min_plays", 5) or 5),
                        dormant_days=int(body.get("dormant_days", 180) or 180),
                        limit=int(body.get("limit", 30) or 30),
                        avoid_artists=avoid_art,
                        avoid_genres=avoid_gen,
                    )
                    flow = body.get("flow_mode") or "mmr"
                    tracks = sequence_playlist_tracks(tracks, flow_mode=flow)
                    saved_payload = None
                    if body.get("save_as_playlist") and tracks:
                        saved_payload = {
                            "prompt": f"Forgotten Favorites (>= {body.get('min_plays', 5)} plays, dormant >= {body.get('dormant_days', 180)}d)",
                            "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat().replace("+00:00", "Z"),
                            "playlist_name": "Forgotten Favorites",
                            "playlist_description": "Once-obsessed tracks from your listening archive you haven't played recently.",
                            "flow_mode": flow,
                            "avoid_artists": avoid_art,
                            "avoid_genres": avoid_gen,
                            "tracks": tracks,
                        }
                        saved_payload["id"] = save_playlist_to_archive(saved_payload)
                        atomic_write_json(LAST_PL_PATH, saved_payload, indent=2)
                        ws.last_playlist = saved_payload
                    return self._send_json(HTTPStatus.OK, {"tracks": tracks, "playlist": saved_payload})

                if path == "/api/llm/probe-ollama":
                    target_url = body.get("ollama_url") or get_effective_preferences().get("ollama_url")
                    return self._send_json(HTTPStatus.OK, probe_ollama_server(target_url, timeout=1.5))

                if path == "/api/preferences":
                    cid_val = body.get("spotify_client_id")
                    clean_cid = cid_val.strip() if isinstance(cid_val, str) else ""
                    if clean_cid and not _SPOTIFY_CID_INPUT_RE.match(clean_cid):
                        raise ValueError("Invalid Spotify Client ID format (expected 16-64 alphanumeric characters).")

                    prev_prefs = get_effective_preferences()
                    llm_keys = ("llm_backend", "llm_model_key", "llm_device", "ollama_url", "ollama_model")
                    prev_sig = tuple(prev_prefs.get(k) for k in llm_keys)
                    updated = save_user_preferences(body.get("preferences") or body)
                    new_sig = tuple(updated.get(k) for k in llm_keys)
                    if prev_sig != new_sig:
                        app_ctx["warm_session"].unload_llm()
                    if clean_cid:
                        atomic_write_text(os.path.join(app_ctx["base_dir"], ".spotify_client_id"), clean_cid + "\n")
                        app_ctx["warm_session"].sp = None
                    if body.get("sync_session", True):
                        app_ctx["warm_session"].set_session_exclusions(
                            avoid_artists=updated["avoid_artists"],
                            avoid_genres=updated["avoid_genres"],
                        )
                    return self._send_json(HTTPStatus.OK, {"preferences": updated})

                if path == "/api/doctor/purge-reccobeats":
                    rb_path = os.path.join(app_ctx["base_dir"], "01_Data_Extractor", ".reccobeats_cache.json")
                    removed, health = purge_poisoned_and_not_found(
                        cache_path=rb_path,
                        include_not_found=bool(body.get("include_not_found", False)),
                    )
                    return self._send_json(HTTPStatus.OK, {"removed": removed, "health": health})

                return self._send_json(HTTPStatus.NOT_FOUND, {"error": "Unknown endpoint."})
            except FileNotFoundError as exc:
                return self._send_json(HTTPStatus.NOT_FOUND, {"error": str(exc)})
            except (ValueError, IndexError) as exc:
                return self._send_json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
            except Exception as exc:
                return self._send_json(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": str(exc)})

        def do_DELETE(self):
            if not self._validate_request_origin(require_same_origin=True):
                return
            parsed = urllib.parse.urlparse(self.path)
            path = parsed.path
            if path.startswith("/api/playlists/"):
                pid = path[len("/api/playlists/") :]
                if not _ARCHIVE_ID_RE.match(pid):
                    return self._send_json(HTTPStatus.BAD_REQUEST, {"error": "Invalid playlist ID."})
                deleted = delete_archived_playlist(pid)
                if not deleted:
                    return self._send_json(HTTPStatus.NOT_FOUND, {"error": "Playlist not found."})
                return self._send_json(HTTPStatus.OK, {"deleted": True, "id": pid})

            if path.startswith("/api/presets/"):
                preset_name = urllib.parse.unquote(path[len("/api/presets/") :]).strip()
                prefs = get_effective_preferences()
                presets = dict(prefs.get("presets") or {})
                if preset_name not in presets:
                    return self._send_json(HTTPStatus.NOT_FOUND, {"error": "Preset not found."})
                presets.pop(preset_name, None)
                updated_prefs = save_user_preferences({"presets": presets})
                return self._send_json(
                    HTTPStatus.OK,
                    {"deleted": True, "name": preset_name, "presets": updated_prefs["presets"]},
                )

            return self._send_json(HTTPStatus.NOT_FOUND, {"error": "Unknown endpoint."})

    return SpotMusHandler


def create_server(host=None, port=None):
    bind_host = host or getattr(config, "WEB_HOST", "127.0.0.1")
    bind_port = int(port if port is not None else getattr(config, "WEB_PORT", 8765))

    app_ctx = {
        "base_dir": BASE_DIR,
        "job_manager": JobManager(BASE_DIR),
        "warm_session": WarmDiscoverySession(),
        "stop_event": threading.Event(),
    }

    class ReusableThreadingServer(ThreadingHTTPServer):
        allow_reuse_address = True
        daemon_threads = True

    server = ReusableThreadingServer((bind_host, bind_port), make_handler(app_ctx))
    return server, app_ctx


def main():
    configure_utf8_stdio()
    parser = argparse.ArgumentParser(description="Music Taste Analyzer - Local Web Studio")
    parser.add_argument("--host", default=getattr(config, "WEB_HOST", "127.0.0.1"), help="Bind address (default 127.0.0.1)")
    parser.add_argument("--port", type=int, default=getattr(config, "WEB_PORT", 8765), help="Port (default 8765)")
    parser.add_argument("--no-browser", action="store_true", help="Do not automatically open the web browser")
    args = parser.parse_args()

    server, app_ctx = create_server(host=args.host, port=args.port)
    actual_port = server.server_address[1]
    url = f"http://{args.host}:{actual_port}"

    print("==================================================")
    print(" 🌌 Music Taste Analyzer - Local Web Studio")
    print("==================================================")
    print(f" 🚀 Studio URL : {url}")
    print(" ⌨️  Press Ctrl+C to shut down cleanly.")

    if not args.no_browser:
        threading.Timer(0.4, lambda: webbrowser.open(url)).start()

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n🛑 Shutting down Music Taste Analyzer Web Studio...")
    finally:
        app_ctx["stop_event"].set()
        app_ctx["job_manager"].cancel_job()
        server.shutdown()
        server.server_close()


if __name__ == "__main__":
    main()
