import json
import os
import sys

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from utils import (
    atomic_write_json,
    configure_utf8_stdio,
    extract_ytm_video_id,
    get_track_fingerprint,
    parse_spotify_timestamp,
)


def parse_iso_time(ts_str):
    """Parse Spotify timestamp string into a UTC-aware datetime or None."""
    return parse_spotify_timestamp(ts_str)


def _parse_takeout_entry(item):
    """Parse a Google Takeout watch-history.json entry if it belongs to YouTube Music."""
    products = item.get("products") or []
    header = str(item.get("header") or "").strip()
    is_ytm = ("YouTube Music" in products) or (header == "YouTube Music")
    if not is_ytm:
        return None

    raw_title = str(item.get("title") or "").strip()
    if raw_title.startswith("Watched "):
        raw_title = raw_title[len("Watched ") :].strip()
    if raw_title.startswith("Topic - "):
        raw_title = raw_title[len("Topic - ") :].strip()

    subs = item.get("subtitles")
    artist_name = ""
    if isinstance(subs, list) and subs and isinstance(subs[0], dict):
        artist_name = str(subs[0].get("name") or "").strip()
    if artist_name.endswith(" - Topic"):
        artist_name = artist_name[: -len(" - Topic")].strip()

    if not raw_title or not artist_name:
        return None

    # Handle titles formatted as "Artist - Track Title"
    prefix = f"{artist_name} - "
    if raw_title.lower().startswith(prefix.lower()) and len(raw_title) > len(prefix):
        track_name = raw_title[len(prefix) :].strip()
    else:
        track_name = raw_title

    vid = extract_ytm_video_id(item.get("titleUrl"))
    key = f"yt:{vid}" if vid else f"{track_name}:::{artist_name}"
    ts_raw = item.get("time")
    parsed_dt = parse_spotify_timestamp(ts_raw)
    if parsed_dt is None:
        return None

    return {
        "key": key,
        "track_name": track_name,
        "artist_name": artist_name,
        "ms_played": 180000,
        "norm_ts": parsed_dt.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "provider": "ytmusic",
    }


def import_history_with_stats(folder_path="extended_history"):
    """
    Parse Spotify Extended Streaming History AND Google Takeout YouTube Music
    watch-history.json files from folder_path.
    Filters skips (< 30s), skips video history files, normalizes timestamps to UTC ISO,
    and returns (aggregated_data, stats_summary).
    """
    aggregated_data = {}
    summary = {
        "files_processed": 0,
        "video_files_skipped": 0,
        "invalid_files": 0,
        "total_records": 0,
        "valid_streams": 0,
        "skips_filtered": 0,
        "non_music_skipped": 0,
        "unique_tracks": 0,
        "total_hours_listened": 0.0,
        "earliest_stream": None,
        "latest_stream": None,
    }

    if not folder_path or not os.path.exists(folder_path):
        return aggregated_data, summary

    skip_counts_by_key = {}

    for filename in sorted(os.listdir(folder_path)):
        lower_name = filename.lower()
        if not lower_name.endswith(".json"):
            continue
        if lower_name.startswith("streaming_history_video"):
            summary["video_files_skipped"] += 1
            continue

        file_path = os.path.join(folder_path, filename)
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            summary["invalid_files"] += 1
            print(f"Error reading {filename}: {e}")
            continue

        if not isinstance(data, list):
            summary["invalid_files"] += 1
            print(f"Skipping {filename}: expected a JSON array, got {type(data).__name__}.")
            continue

        summary["files_processed"] += 1

        for item in data:
            if not isinstance(item, dict):
                continue
            summary["total_records"] += 1

            # Detect Google Takeout YouTube / YouTube Music history entries
            if "titleUrl" in item or "header" in item or "products" in item:
                ytm_parsed = _parse_takeout_entry(item)
                if ytm_parsed is None:
                    summary["non_music_skipped"] += 1
                    continue
                key = ytm_parsed["key"]
                track_name = ytm_parsed["track_name"]
                artist_name = ytm_parsed["artist_name"]
                ms_played = ytm_parsed["ms_played"]
                norm_ts = ytm_parsed["norm_ts"]
                provider = ytm_parsed["provider"]
            else:
                raw_ms = item.get("ms_played")
                if raw_ms is None:
                    raw_ms = item.get("msPlayed")
                try:
                    ms_played = int(raw_ms or 0)
                except (TypeError, ValueError):
                    ms_played = 0

                track_name = item.get("master_metadata_track_name") or item.get("trackName")
                artist_name = item.get("master_metadata_album_artist_name") or item.get("artistName")

                if not track_name or not artist_name:
                    summary["non_music_skipped"] += 1
                    continue

                uri = item.get("spotify_track_uri") or item.get("spotifyTrackUri")
                if isinstance(uri, str) and uri.startswith("spotify:track:"):
                    key = uri.split(":")[-1]
                else:
                    key = f"{track_name}:::{artist_name}"

                if ms_played < 30000:
                    summary["skips_filtered"] += 1
                    skip_counts_by_key[key] = skip_counts_by_key.get(key, 0) + 1
                    continue

                ts_raw = item.get("ts") or item.get("endTime")
                parsed_dt = parse_spotify_timestamp(ts_raw)
                if parsed_dt is None:
                    continue
                norm_ts = parsed_dt.strftime("%Y-%m-%dT%H:%M:%SZ")
                provider = "spotify"

            if key not in aggregated_data:
                aggregated_data[key] = {
                    "track_name": track_name,
                    "artist_name": artist_name,
                    "fingerprint": get_track_fingerprint(track_name, artist_name),
                    "provider": provider,
                    "play_count": 0,
                    "skip_count": 0,
                    "total_ms_listened": 0,
                    "stream_timestamps": [],
                }
            elif aggregated_data[key].get("provider") != provider:
                aggregated_data[key]["provider"] = "both"

            aggregated_data[key]["play_count"] += 1
            aggregated_data[key]["total_ms_listened"] += ms_played
            aggregated_data[key]["stream_timestamps"].append(norm_ts)
            summary["valid_streams"] += 1

    earliest = None
    latest = None
    total_ms = 0

    for key, stats in aggregated_data.items():
        stats["skip_count"] = skip_counts_by_key.get(key, 0)
        stats["total_hours_listened"] = round(stats["total_ms_listened"] / (1000 * 60 * 60), 4)
        total_ms += stats["total_ms_listened"]
        stats["stream_timestamps"].sort()
        if stats["stream_timestamps"]:
            first_ts = stats["stream_timestamps"][0]
            last_ts = stats["stream_timestamps"][-1]
            stats["first_streamed_date"] = first_ts
            stats["last_streamed_date"] = last_ts
            if earliest is None or first_ts < earliest:
                earliest = first_ts
            if latest is None or last_ts > latest:
                latest = last_ts

    summary["unique_tracks"] = len(aggregated_data)
    summary["total_hours_listened"] = round(total_ms / (1000 * 60 * 60), 2)
    summary["earliest_stream"] = earliest
    summary["latest_stream"] = latest
    return aggregated_data, summary


def import_history(folder_path="extended_history"):
    """Backward-compatible wrapper returning only the aggregated track dictionary."""
    aggregated_data, _ = import_history_with_stats(folder_path)
    return aggregated_data


def run_import(folder_arg=None):
    configure_utf8_stdio()
    print("========================================")
    print(" Spotify Extended History Importer")
    print("========================================")

    if folder_arg:
        history_dir = os.path.abspath(folder_arg)
    else:
        history_dir = os.path.join(ROOT_DIR, "extended_history")

    if not os.path.exists(history_dir):
        print(f"❌ Error: {history_dir} not found.")
        print("Please create an 'extended_history' folder in the root directory and place your .json files inside.")
        sys.exit(1)

    print(f"Scanning {history_dir}...")
    aggregated_data, summary = import_history_with_stats(history_dir)

    if not aggregated_data:
        print("❌ Error: No valid streaming history found or files could not be parsed.")
        sys.exit(1)

    print(
        f"\n📊 Import Summary:"
        f"\n   • Audio JSON files parsed : {summary['files_processed']} "
        f"(Video skipped: {summary['video_files_skipped']}, Invalid: {summary['invalid_files']})"
        f"\n   • Total records scanned   : {summary['total_records']:,}"
        f"\n   • Valid streams (>= 30s)  : {summary['valid_streams']:,}"
        f"\n   • Skips filtered (< 30s)  : {summary['skips_filtered']:,}"
        f"\n   • Unique music tracks     : {summary['unique_tracks']:,}"
        f"\n   • Total listening hours   : {summary['total_hours_listened']:,.1f} hrs"
        f"\n   • Date range (UTC)        : {summary['earliest_stream']} -> {summary['latest_stream']}"
    )

    top_tracks = sorted(aggregated_data.values(), key=lambda x: x["play_count"], reverse=True)[:5]
    if top_tracks:
        print("\n🏆 Top 5 Most Played Tracks:")
        for idx, t in enumerate(top_tracks, start=1):
            print(f"   {idx}. {t['track_name']} - {t['artist_name']} ({t['play_count']} plays / {t['total_hours_listened']:.1f}h)")

    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "extended_history_stats.json")
    atomic_write_json(out_path, aggregated_data, indent=2)

    print(f"\n✅ Extended history stats saved to {out_path}")
    print("You can now run Stage 1 or directly refresh Stage 2 (Vector Core) to apply time-decay weights.")


if __name__ == "__main__":
    cli_folder = sys.argv[1] if len(sys.argv) > 1 else None
    run_import(cli_folder)
