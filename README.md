# 🌌 Music Taste Analyzer — Local AI Music Discovery & Web Studio

> **Vectorize your musical identity across Spotify & YouTube Music, explore interactive 2D/3D taste maps, and generate DJ-sequenced AI playlists using 100% free local OpenVINO (NPU/iGPU/CPU) and optional Ollama (NVIDIA CUDA) models.**

![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue)
![Local AI](https://img.shields.io/badge/Local%20AI-OpenVINO%20%7C%20Ollama-brightgreen)
![Spotify API](https://img.shields.io/badge/Spotify-API%20v1-1DB954)
![YouTube Music](https://img.shields.io/badge/YouTube%20Music-ytmusicapi-FF0000)
![ReccoBeats](https://img.shields.io/badge/Acoustic%20Telemetry-ReccoBeats%20API-purple)

---

## 🌟 Key Features

* **🖥️ Local Web Studio (`http://127.0.0.1:8765`)**: Interactive browser-based workspace with a top-bar **Provider Switcher** (`Spotify`, `YouTube Music`, `Unified (Spotify + YTM)`), live Server-Sent Events (SSE) job logs, stale-artifact detection, **Named Session Presets ("Modes")**, and one-click **Full Sync**.
* **🌐 Unified Dual-Provider Engine (Spotify + YouTube Music + Cross-Platform Transfer)**:
  * Connect YouTube Music via **Browser Headers** or **Copy as cURL** (`ytm_browser.json`) or **Google Cloud TV/Limited-Input OAuth** (`ytm_oauth.json`) directly in **Preferences**.
  * Run **Stage 1 YT Music Sync** (`ytmusic_extractor.py`) or **Unified Mode** (`--provider unified`) so Spotify tracks and YouTube Music exclusives (`yt:<videoId>`) vectorize and cluster side by side — overlapping songs deduplicate by normalized title/artist while platform exclusives remain first-class citizens.
  * Bridges YTM tracks into acoustic feature space via ReccoBeats `GET /v1/track/search?searchText=...` + `GET /v1/audio-features` and supports **Google Takeout** `watch-history.json` drag-and-drop alongside Spotify Extended History ZIPs.
  * One-click **Export to YT Music** and **Cross-Platform Transfer** (`Spotify ↔ YT Music`) with an explicit unmatched-tracks report for platform exclusives.
* **🎧 Dual Embedded Preview Dock, Queue/Play & Now Playing**: Preview Spotify tracks inline (`open.spotify.com/embed/track/...`) or full YouTube Music tracks via the YouTube embed (`youtube.com/embed/<videoId>`), start or **Queue** generated mixes directly on your active Spotify device (`PUT /me/player/play`, `POST /me/player/queue`), save favorites with `♡` via Spotify's 2026 `PUT /me/library` endpoint, or seed mixes from what's **Playing Now**.
* **🔥 Warm Interactive Studio & Hardware-Tiered Local AI Curator (100% Free & Local)**:
  * Keeps `all-MiniLM-L6-v2` and a **Hardware-Tiered Local LLM** warm in memory (automatically recommends **Tier 1 Qwen3-1.7B** for 8 GB laptops, **Tier 2 Qwen3-4B / Phi-4 Mini** on Intel iGPU for 16 GB laptops, **Tier 3 Qwen3-8B / Qwen2.5-7B** for 32 GB desktops & AMD Ryzen AI systems, and **Tier 4 Qwen3-14B** for workstations — zero API fees).
  * **Optional Free Local Ollama (NVIDIA RTX CUDA)**: On PCs with NVIDIA GPUs (e.g., RTX 4050/4060), select **Local Ollama Server** in **Preferences** to run `qwen3:8b`, `gemma3:12b`, or `qwen3:14b` via CUDA at high token speeds.
  * **XGrammar Structured JSON + Taste-DNA Grounding**: Enforces strict JSON Schemas natively via OpenVINO GenAI `StructuredOutputConfig` / Ollama schema format, grounds query planning with your top Stage 2 clusters and weighted artists, resolves **AI Curator Picks**, and runs telemetry-aware vibe checks (`BPM`, `Energy`, `Valence`, `Genres`).
  * **Hybrid ReccoBeats Discovery**: Combines Spotify Search (`limit<=10`) with ReccoBeats `/v1/track/recommendation` (using 👍/liked seeds and 👎 `negativeSeeds`) plus `/v1/track` popularity backfill so Deep Cuts works reliably under Spotify 2026 Dev Mode.
  * **Personal Release Radar & Album Sync**: Stage 1 indexes your **Saved Albums** (`GET /me/albums`) and **Followed Artists** (`GET /me/following`), powering a Personal Release Radar (`GET /artists/{id}/albums` + `GET /albums/{id}/tracks`) in History & Archive.
  * **Track-to-Track Vibe Bridge**: Smoothly interpolates 384-D embeddings + BPM/energy from a Start track to a Destination track.
  * **Taste Blend ("Match With a Friend")**: Export a non-PII `music_taste_card.json` share card, compute **Taste Compatibility %**, and generate blended playlists.
  * **Live Curation**: Conversational refinement bar (`"more upbeat"`, `"remove Drake"`), drag-and-drop track reordering, inline title/description editing, 👍/👎 feedback memory (with optional Liked Songs sync), and updating **existing playlists** in place.
* **🚫 Session Taste Guardrails**: Exclude specific **artists** or **genres** for your session across Stage 2 clustering and Stage 3 discovery, or save persistent defaults & cluster labels in `user_preferences.json`.
* **⚡ Incremental Vectorization & NPU/CPU Acceleration**: Hash-based embedding cache reuses unchanged track vectors so Stage 2 refreshes finish in seconds.
* **🎛️ DJ Flow & Harmonic Sequencing**: Order playlists via *Warm-Up → Peak → Cooldown*, *Energy Ramp*, *Chill Descent*, *Camelot Wheel Harmonic Mixing*, *Artist-Spaced Shuffle*, or *MMR Relevance*.
* **🌌 Interactive 2D / 3D Taste Map, UMAP & Era Evolution**: Choose **PCA** or optional **UMAP** projection, filter by **Era years** (`--era-start`/`--era-end`), overlay your **2020 → 2026 Taste Evolution Path**, rename clusters, and click tracks to preview or expand.
* **🧹 Extended History Drag-and-Drop & Library Hygiene**: Upload Spotify `.zip`/`.json` or Google Takeout `.zip`/`watch-history.json` archives safely in-browser, inspect listening telemetry, resurrect Forgotten Favorites, and audit playlists for duplicates, chronically skipped songs, or disliked tracks (with one-click Spotify playlist deduplication).
* **📅 Offline-First Plotly & Weekly Auto-Updating Playlist**: Optional local `/vendor/plotly.min.js` bundle plus `automation_runner.py` (Windows Task Scheduler integration) to automatically refresh a pinned playlist using your configured `active_provider` (`Spotify`, `YouTube Music`, or `Unified`).
* **📦 Playlist Archive & Multi-Format Exports**: Automatically archives generated playlists and exports to **Spotify** or **YouTube Music** (New or Update Existing, Private/Public), **CSV**, **M3U**, and **Markdown**.

---

## 🚀 Quick Start (Local Web Studio)

See [SETUP_GUIDE.md](SETUP_GUIDE.md) for a condensed walkthrough, or follow the steps below:

### 1. Configure Your Spotify Client ID (Optional if Using YouTube Music Only)

No API secrets or paid subscriptions to external LLM providers are required. For Spotify library access, create a free Spotify Developer app:

1. Open the [Spotify Developer Dashboard](https://developer.spotify.com/dashboard) and click **Create App**.
2. Add Redirect URI: **`http://127.0.0.1:8080`** (select Web API).
3. Copy your **Client ID** and paste it into **Preferences -> Spotify Developer Client ID** in the Web Studio (saved locally to `.spotify_client_id`), or copy `.env.example` to `.env` and set `SPOTIFY_CLIENT_ID=...`.
   *(Under Spotify's 2026 Development Mode policy, the app owner needs Spotify Premium and can allowlist up to 5 users under Dashboard -> Settings -> User Management. If you only want YouTube Music, skip this step and connect YT Music in Preferences!)*

### 2. Launch the Web Studio

* **Windows**: Double-click **`RUN_MUSIC_TASTE_ANALYZER.bat`**
* **macOS / Linux**:

```bash
chmod +x run.sh
./run.sh
```

Or launch directly inside your virtual environment:

```bash
python web_app.py
# Or via compatibility entrypoint:
python main_orchestrator.py
```

Your browser opens **`http://127.0.0.1:8765`** automatically (note: Spotify OAuth PKCE uses port `8080`, while the Web Studio binds strictly to loopback `127.0.0.1:8765`).

### 3. Connect & Build Your Taste Core
1. Choose **Spotify**, **YouTube Music**, or **Unified (Spotify + YTM)** in the topbar **Provider Switcher** (configure YouTube Music **Browser Headers / Copy as cURL** or **Google OAuth** in **Preferences** if using YT Music).
2. In **Dashboard & Sync**, click **⚡ Full Sync (1 → 2 → Map)** (or run **1. Sync Spotify Library** / **1b. Sync YouTube Music**).
3. Add any **Session Taste Guardrails** (artists or genres to avoid) and click **Rebuild Taste Clusters Now** if you want clusters modeled without them.
4. Open **Playlist Studio**, choose your **Hardware-Tier Local LLM** (OpenVINO auto-tier or free local Ollama) in **Preferences**, warm the engine, choose a prompt/cluster/DJ flow, and generate!

---

## 🛠️ CLI Usage (Optional Script Flags)

Each stage can still be driven directly from the terminal:

```bash
# Stage 1 YouTube Music Sync
python 01_Data_Extractor/ytmusic_extractor.py --auth-method auto

# Unified (Spotify + YouTube Music) vectorization & clustering
python 02_Vector_Core/npu_vectorizer.py --provider unified
python 02_Vector_Core/taste_clusterer.py --provider unified

# Stage 3 Playlist Discovery with Warm Interactive Session & CLI options
python 03_Agentic_Engine/discovery_engine.py --interactive --fast
python 03_Agentic_Engine/discovery_engine.py "135+ BPM dark synthwave workout" \
  --fast --size 35 --flow warmup_peak_cooldown --clean --deep-cuts \
  --provider unified --avoid-artists "Drake" --avoid-genres "country"

# Expand a specific Stage 2 cluster
python 03_Agentic_Engine/discovery_engine.py --cluster 1 --flow harmonic --no-export

# Build Stage 2 clusters excluding specific artists/genres and filtering by era
python 02_Vector_Core/taste_clusterer.py --avoid-artists "Artist A" --avoid-genres "metal,country" --era-start 2022 --era-end 2026

# Build 2D/3D Taste Map with optional UMAP projection & offline Plotly bundle
python 02_Vector_Core/view_taste_map.py --projection umap --offline

# Generate a Track-to-Track Vibe Bridge or update an existing Spotify playlist
python 03_Agentic_Engine/discovery_engine.py --bridge-start "Track A" --bridge-end "Track B" --bridge-steps 20 --target-playlist YOUR_PLAYLIST_ID

# Headless Weekly Auto-Updating Playlist Runner & Windows Task Scheduler
python automation_runner.py --run-now
python automation_runner.py --install-task --day SUN --time 09:00
python automation_runner.py --status
python automation_runner.py --remove-task
```

Supported CLI flags:
* `ytmusic_extractor.py`: `--auth-method` (`auto`, `browser`, `oauth`), `--skip-reccobeats`.
* `npu_vectorizer.py`: `--force`, `--provider` (`spotify`, `ytmusic`, `unified`).
* `discovery_engine.py`: `--fast`, `--size`, `--local-only`, `--provider` (`spotify`, `ytmusic`, `unified`), `--cluster`, `--flow` (`mmr`, `warmup_peak_cooldown`, `energy_ramp`, `chill_descent`, `harmonic`, `shuffle`), `--clean`, `--deep-cuts`, `--avoid-artists`, `--avoid-genres`, `--bridge-start`, `--bridge-end`, `--bridge-steps`, `--target-playlist`, `--interactive`, `--no-export`, `--auto-export`, `--public`, `--view-last`, `--export-last`.
* `taste_clusterer.py`: `--provider` (`spotify`, `ytmusic`, `unified`), `--avoid-artists`, `--avoid-genres`, `--half-life`, `--no-skip-penalty`, `--era-start`, `--era-end`.
* `view_taste_map.py`: `--projection` (`pca`, `umap`), `--offline`.
* `automation_runner.py`: `--run-now`, `--install-task`, `--remove-task`, `--status`, `--day`, `--time`.

---

## 👥 Inviting Friends to Use Your App (For Host Developers)

Spotify Development Mode apps require an active host Spotify Premium account and support up to **5** allowlisted users under Spotify's 2026 policy:

1. Open the [Spotify Developer Dashboard](https://developer.spotify.com/dashboard).
2. Select your **Music Taste Analyzer** app -> **Settings** -> **User Management**.
3. Add your friends' Spotify email addresses (up to 5 accounts), or let each friend clone this repo and follow [SETUP_GUIDE.md](SETUP_GUIDE.md) to use their own Client ID (and/or connect YouTube Music).

---

## 📂 Importing Extended Streaming History (Spotify & YouTube Music)

1. **Spotify**: Request your **Extended streaming history** (`Streaming_History_Audio_*.json` / `.zip`) from your Spotify Privacy settings page.
2. **YouTube Music**: Request your **YouTube and YouTube Music -> history** (`watch-history.json` / `.zip`) via [Google Takeout](https://takeout.google.com).
3. Drag & drop the `.zip` or `.json` files directly into the **Upload Extended Streaming History** dropzone in the Web Studio Dashboard (or place them inside `extended_history/`).
4. Click **4. Import Extended History** in the Web Studio Dashboard (or run `python 01_Data_Extractor/history_importer.py`).
5. Rebuild **Stage 2** to incorporate time-decay weights, yearly evolution centroids, and skip-rate penalties, then inspect **History & Archive**.

---

## 🗂️ Project Structure

```text
├── 01_Data_Extractor/      # Spotify & YTM extractors, ReccoBeats client, history importer & hygiene
├── 02_Vector_Core/         # OpenVINO MiniLM vectorizer, weighted clusterer, 2D/3D taste map
├── 03_Agentic_Engine/      # Hardware-tiered LLM query parser (OpenVINO/Ollama) & discovery engine
├── web_ui/                 # Single-page Web Studio frontend (HTML/CSS/JS)
├── tests/                  # Unit & integration test suite (unittest)
├── web_app.py              # Loopback HTTP + SSE studio server & API
├── main_orchestrator.py    # Compatibility entrypoint delegating to web_app.py
├── automation_runner.py    # Headless weekly playlist refresh & Task Scheduler helper
├── config.py / utils.py    # Environment config, hardware tiering, atomic JSON I/O, security guards
├── RUN_MUSIC_TASTE_ANALYZER.bat / run.sh  # Cross-platform launchers
├── SETUP_GUIDE.md          # Quick 3-step setup guide
└── .env.example            # Sample local environment configuration
```

---

## 🧪 Running Tests

Run the unit & integration test suite (all network/model calls are cleanly mocked):

```bash
python -m unittest discover -s tests -v
```

---

## 🛡️ Privacy & Security

* **100% Local & Loopback-Isolated**: Binds strictly to `127.0.0.1`, enforces localhost `Host` / `Origin` / `Sec-Fetch-Site` CSRF protections on mutating requests, sets `X-Frame-Options: DENY` & `X-Content-Type-Options: nosniff`, and restricts optional Ollama/Local API URLs to loopback or private RFC1918 LAN addresses.
* **Private Credentials Stay Local**: Tokens, `.env`, `.spotify_client_id`, `ytm_browser.json`, `ytm_oauth.json`, `.ytm_oauth_client.json`, `user_music_profile.json`, `ytmusic_profile.json`, `user_preferences.json`, `user_feedback.json`, `03_Agentic_Engine/models/`, `extended_history/`, `logs/`, and `playlist_history/` are excluded by `.gitignore`.
* **PKCE OAuth Security**: Spotify authentication uses PKCE (Proof Key for Code Exchange), so Spotify passwords and Client Secrets are never requested or stored.

---

## 📄 License
Distributed under the [MIT License](LICENSE).
