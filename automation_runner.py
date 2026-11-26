"""
Headless Weekly Auto-Updating Playlist Runner & Windows Task Scheduler Helper.

Can be invoked standalone via CLI (`python automation_runner.py --run-now`),
scheduled weekly via `schtasks.exe` (`--install-task`), or triggered from the
Local Web Studio (`/api/automation`, `/api/jobs/run` with `weekly_automation`).
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import os
import re
import subprocess
import sys

from config import BASE_DIR
from utils import (
    DEFAULT_AUTOMATION_SETTINGS,
    get_effective_preferences,
    save_user_preferences,
)

TASK_NAME = "MusicTasteAnalyzer_WeeklyRefresh"
VALID_DAYS = ("MON", "TUE", "WED", "THU", "FRI", "SAT", "SUN")
_TIME_RE = re.compile(r"^([01]\d|2[0-3]):([0-5]\d)$")
_PLAYLIST_URL_RE = re.compile(r"playlist/([A-Za-z0-9]{10,40})")
_PLAIN_ID_RE = re.compile(r"^[A-Za-z0-9]{10,40}$")


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _extract_playlist_id(url_or_id: str | None) -> str:
    if not url_or_id:
        return ""
    text = str(url_or_id).strip()
    match = _PLAYLIST_URL_RE.search(text)
    if match:
        return match.group(1)
    if _PLAIN_ID_RE.match(text):
        return text
    return ""


def _import_discovery_engine():
    engine_dir = os.path.join(BASE_DIR, "03_Agentic_Engine")
    if engine_dir not in sys.path:
        sys.path.insert(0, engine_dir)
    import discovery_engine  # type: ignore

    return discovery_engine


def _run_stage_script(
    script_rel_path: str,
    extra_args: list[str] | None = None,
    log_cb=None,
    subprocess_run=None,
) -> None:
    runner = subprocess_run or subprocess.run
    script_abs = os.path.join(BASE_DIR, script_rel_path)
    cmd = [sys.executable, script_abs, *(extra_args or [])]
    if log_cb:
        log_cb(f"Running automation step: {script_rel_path}")
    proc = runner(
        cmd,
        cwd=BASE_DIR,
        capture_output=True,
        text=True,
        shell=False,
    )
    if proc.returncode != 0:
        stderr_tail = (proc.stderr or proc.stdout or "").strip()[-400:]
        raise RuntimeError(
            f"{script_rel_path} exited with code {proc.returncode}: {stderr_tail}"
        )


def _persist_automation_status(updates: dict) -> dict:
    prefs = get_effective_preferences()
    current = dict(prefs.get("automation") or DEFAULT_AUTOMATION_SETTINGS)
    current.update(updates)
    saved = save_user_preferences({"automation": current})
    return dict(saved.get("automation") or current)


def run_weekly_refresh(
    log_cb=None,
    session_cls=None,
    subprocess_run=None,
) -> dict:
    """Execute the configured weekly playlist refresh end-to-end."""
    log = log_cb or (lambda msg: print(f"[automation] {msg}"))
    prefs = get_effective_preferences()
    auto = dict(prefs.get("automation") or DEFAULT_AUTOMATION_SETTINGS)

    try:
        active_provider = str(prefs.get("active_provider", "spotify")).strip().lower()
        if active_provider not in ("spotify", "ytmusic", "unified"):
            active_provider = "spotify"

        run_s1 = bool(auto.get("run_stage1_sync", auto.get("refresh_stage1", False)))
        run_s2 = bool(auto.get("run_stage2_refresh", auto.get("refresh_stage2", False)))

        if run_s1:
            if active_provider in ("spotify", "unified"):
                _run_stage_script(
                    os.path.join("01_Data_Extractor", "stage1_extractor.py"),
                    log_cb=log,
                    subprocess_run=subprocess_run,
                )
            if active_provider in ("ytmusic", "unified"):
                _run_stage_script(
                    os.path.join("01_Data_Extractor", "ytmusic_extractor.py"),
                    log_cb=log,
                    subprocess_run=subprocess_run,
                )

        if run_s2:
            _run_stage_script(
                os.path.join("02_Vector_Core", "npu_vectorizer.py"),
                extra_args=["--provider", active_provider],
                log_cb=log,
                subprocess_run=subprocess_run,
            )
            cluster_args: list[str] = ["--provider", active_provider]
            avoid_art = prefs.get("avoid_artists") or []
            avoid_gen = prefs.get("avoid_genres") or []
            if avoid_art:
                cluster_args.extend(["--avoid-artists", ",".join(avoid_art)])
            if avoid_gen:
                cluster_args.extend(["--avoid-genres", ",".join(avoid_gen)])
            _run_stage_script(
                os.path.join("02_Vector_Core", "taste_clusterer.py"),
                extra_args=cluster_args,
                log_cb=log,
                subprocess_run=subprocess_run,
            )
            _run_stage_script(
                os.path.join("02_Vector_Core", "view_taste_map.py"),
                extra_args=["--offline"],
                log_cb=log,
                subprocess_run=subprocess_run,
            )

        if session_cls is None:
            engine = _import_discovery_engine()
            session = engine.WarmDiscoverySession()
        else:
            session = session_cls()

        prompt = str(auto.get("prompt") or DEFAULT_AUTOMATION_SETTINGS["prompt"]).strip()
        size = int(auto.get("size") or DEFAULT_AUTOMATION_SETTINGS["size"])
        fast = bool(auto.get("fast", True))
        clean_only = bool(auto.get("clean_only", False))
        deep_cuts = bool(auto.get("deep_cuts", False))
        flow_mode = str(auto.get("flow_mode") or "harmonic").strip()
        target_id = _extract_playlist_id(auto.get("target_playlist_id"))

        log(f"Generating weekly playlist ({size} tracks, flow={flow_mode}, provider={active_provider})...")
        gen_res = session.generate(
            prompt=prompt,
            size=size,
            fast=fast,
            flow_mode=flow_mode,
            clean_only=clean_only,
            deep_cuts=deep_cuts,
            provider=active_provider,
            log_cb=log,
        )
        playlist = gen_res.get("playlist", gen_res) if isinstance(gen_res, dict) else {}

        public_export = bool(prefs.get("default_public_export", False))
        if active_provider == "ytmusic" and hasattr(session, "export_ytmusic_playlist"):
            yt_res = session.export_ytmusic_playlist(
                privacy="PUBLIC" if public_export else "PRIVATE",
                target_playlist_id=target_id or None,
            )
            exported_url = str((yt_res or {}).get("url") or "")
            resolved_target_id = str((yt_res or {}).get("playlist_id") or target_id)
        else:
            export_result = session.export_playlist(
                public=public_export,
                target_playlist_id=target_id or None,
            )
            if not export_result:
                raise RuntimeError(
                    "Spotify export failed or was cancelled. Ensure Stage 1 OAuth token is valid."
                )

            exported_url = export_result if isinstance(export_result, str) else ""
            resolved_target_id = _extract_playlist_id(exported_url) or target_id
        status_msg = (
            f"Updated {len(playlist.get('tracks', []))} tracks -> "
            f"{exported_url or resolved_target_id}"
        )
        updated_auto = _persist_automation_status(
            {
                "target_playlist_id": resolved_target_id,
                "last_run_at": _utc_now_iso(),
                "last_status": status_msg,
            }
        )
        log(f"Weekly refresh complete: {status_msg}")
        return {
            "ok": True,
            "playlist_name": playlist.get("playlist_name", ""),
            "track_count": len(playlist.get("tracks", [])),
            "url": exported_url,
            "target_playlist_id": resolved_target_id,
            "automation": updated_auto,
        }
    except Exception as exc:
        err_msg = f"error: {exc}"
        _persist_automation_status(
            {
                "last_run_at": _utc_now_iso(),
                "last_status": err_msg[:300],
            }
        )
        log(f"Weekly refresh failed: {exc}")
        return {
            "ok": False,
            "error": str(exc),
        }


def _validate_schedule(day_of_week: str, time_str: str) -> tuple[str, str]:
    day = str(day_of_week or "SUN").strip().upper()[:3]
    if day not in VALID_DAYS:
        raise ValueError(
            f"Invalid day_of_week {day_of_week!r}; expected one of {', '.join(VALID_DAYS)}"
        )
    cleaned_time = str(time_str or "09:00").strip()
    if not _TIME_RE.match(cleaned_time):
        raise ValueError(f"Invalid time {time_str!r}; expected HH:MM in 24-hour format")
    return day, cleaned_time


def _build_schtasks_command(day: str, time_str: str) -> list[str]:
    script_path = os.path.abspath(__file__)
    action = f'"{sys.executable}" "{script_path}" --run-now'
    return [
        "schtasks.exe",
        "/Create",
        "/SC",
        "WEEKLY",
        "/D",
        day,
        "/ST",
        time_str,
        "/TN",
        TASK_NAME,
        "/TR",
        action,
        "/F",
    ]


def register_windows_task(
    day_of_week: str = "SUN",
    time_str: str = "09:00",
    runner_fn=None,
) -> dict:
    """Register or update the weekly Windows Task Scheduler job."""
    day, cleaned_time = _validate_schedule(day_of_week, time_str)
    if os.name != "nt" and runner_fn is None:
        return {
            "supported": False,
            "registered": False,
            "task_name": TASK_NAME,
            "error": "Windows Task Scheduler (schtasks.exe) is only available on Windows.",
        }

    runner = runner_fn or subprocess.run
    cmd = _build_schtasks_command(day, cleaned_time)
    proc = runner(cmd, capture_output=True, text=True, shell=False)
    if proc.returncode != 0:
        detail = (proc.stderr or proc.stdout or "").strip()
        return {
            "supported": True,
            "registered": False,
            "task_name": TASK_NAME,
            "error": detail or f"schtasks exited with code {proc.returncode}",
        }

    _persist_automation_status(
        {
            "enabled": True,
            "day_of_week": day,
            "time": cleaned_time,
        }
    )
    return {
        "supported": True,
        "registered": True,
        "task_name": TASK_NAME,
        "day_of_week": day,
        "time": cleaned_time,
        "output": (proc.stdout or "").strip(),
    }


def remove_windows_task(runner_fn=None) -> dict:
    """Remove the weekly Windows Task Scheduler job if present."""
    if os.name != "nt" and runner_fn is None:
        return {
            "supported": False,
            "removed": False,
            "task_name": TASK_NAME,
            "error": "Windows Task Scheduler (schtasks.exe) is only available on Windows.",
        }

    runner = runner_fn or subprocess.run
    cmd = ["schtasks.exe", "/Delete", "/TN", TASK_NAME, "/F"]
    proc = runner(cmd, capture_output=True, text=True, shell=False)
    stderr_lower = (proc.stderr or proc.stdout or "").lower()
    not_found = "cannot find" in stderr_lower or "does not exist" in stderr_lower
    if proc.returncode != 0 and not not_found:
        return {
            "supported": True,
            "removed": False,
            "task_name": TASK_NAME,
            "error": (proc.stderr or proc.stdout or "").strip(),
        }

    _persist_automation_status({"enabled": False})
    return {
        "supported": True,
        "removed": True,
        "task_name": TASK_NAME,
        "was_missing": not_found,
    }


def query_windows_task(runner_fn=None) -> dict:
    """Query status of the weekly Windows Task Scheduler job."""
    if os.name != "nt" and runner_fn is None:
        return {
            "supported": False,
            "exists": False,
            "task_name": TASK_NAME,
        }

    runner = runner_fn or subprocess.run
    cmd = ["schtasks.exe", "/Query", "/TN", TASK_NAME, "/FO", "LIST"]
    proc = runner(cmd, capture_output=True, text=True, shell=False)
    if proc.returncode != 0:
        return {
            "supported": True,
            "exists": False,
            "task_name": TASK_NAME,
            "status": "Not scheduled",
        }

    stdout = proc.stdout or ""
    status_val = "Scheduled"
    next_run = ""
    for line in stdout.splitlines():
        if ":" not in line:
            continue
        key, val = line.split(":", 1)
        key_l = key.strip().lower()
        val_s = val.strip()
        if key_l == "status":
            status_val = val_s
        elif key_l == "next run time":
            next_run = val_s

    return {
        "supported": True,
        "exists": True,
        "task_name": TASK_NAME,
        "status": status_val,
        "next_run_time": next_run,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Music Taste Analyzer Weekly Auto-Updating Playlist Runner"
    )
    parser.add_argument(
        "--run-now",
        action="store_true",
        help="Execute the weekly playlist refresh right now.",
    )
    parser.add_argument(
        "--install-task",
        action="store_true",
        help="Register a weekly Windows Task Scheduler job.",
    )
    parser.add_argument(
        "--remove-task",
        action="store_true",
        help="Remove the weekly Windows Task Scheduler job.",
    )
    parser.add_argument(
        "--status",
        action="store_true",
        help="Print scheduled task and saved automation status.",
    )
    parser.add_argument(
        "--day",
        type=str,
        default=None,
        help="Day of week for --install-task (MON..SUN).",
    )
    parser.add_argument(
        "--time",
        type=str,
        default=None,
        help="24-hour HH:MM time for --install-task.",
    )
    args = parser.parse_args(argv)

    if args.install_task:
        auto = get_effective_preferences().get("automation") or DEFAULT_AUTOMATION_SETTINGS
        day = args.day or auto.get("day_of_week", "SUN")
        time_str = args.time or auto.get("time", "09:00")
        res = register_windows_task(day, time_str)
        print(res)
        return 0 if res.get("registered") else 1

    if args.remove_task:
        res = remove_windows_task()
        print(res)
        return 0 if res.get("removed") else 1

    if args.status:
        auto = get_effective_preferences().get("automation") or DEFAULT_AUTOMATION_SETTINGS
        task = query_windows_task()
        print({"automation": auto, "task": task})
        return 0

    if args.run_now:
        res = run_weekly_refresh()
        return 0 if res.get("ok") else 1

    parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
