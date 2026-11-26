# 🚀 Music Taste Analyzer: Quick Setup Guide

Follow these steps after cloning or downloading the repository to connect Spotify, YouTube Music, and your Local AI tier.

---

## 🔐 Step 1: Set Up Your Spotify Client ID (60 Seconds)

To keep credentials private on GitHub, no shared Client ID is hardcoded in `config.py`. Setting up your own free PKCE Client ID takes under a minute (and no Client Secret is needed):

1. Open the [Spotify Developer Dashboard](https://developer.spotify.com/dashboard) and click **Create App**.
2. Give it any name (e.g. `My Music Taste Analyzer`) and add this exact **Redirect URI**:
   `http://127.0.0.1:8080`
3. Under APIs used, check **Web API**, save, and copy your **Client ID**.
4. Configure it either by:
   * Copying `.env.example` to `.env` and setting `SPOTIFY_CLIENT_ID=<your-id>`, **or**
   * Launching the Web Studio and pasting it into **Preferences -> Spotify Developer Client ID** (stored locally in gitignored `.spotify_client_id`).

*(Prefer YouTube Music only? Skip Step 1 and connect YT Music directly in Preferences!)*

---

## 🎧 Step 2: Launch the Web Studio

* **Windows**: Double-click **`RUN_MUSIC_TASTE_ANALYZER.bat`**
* **macOS / Linux**:

```bash
chmod +x run.sh
./run.sh
```

The launcher creates `.venv`, installs dependencies from `requirements.txt`, and opens **`http://127.0.0.1:8765`** automatically.

---

## 🔄 Step 3: Connect Platforms, Pick Your AI Tier & Sync
1. **Spotify**: Open **Dashboard & Sync** and click **⚡ Full Sync (1 → 2 → Map)** (or **1. Sync Spotify Library**), then approve the Spotify PKCE login pop-up.
2. **YouTube Music (Optional / Unified Mode)**: In **Preferences**, paste Request Headers or **Copy as cURL** from `music.youtube.com` (`DevTools -> Network -> /browse`), or link **Google Cloud TV/Limited-Input OAuth**, then click **1b. Sync YouTube Music**. Switch between `Spotify`, `YouTube Music`, and `Unified` anytime in the topbar.
3. **Hardware-Tiered Local LLM (100% Free)**: In **Preferences -> Local AI Model & Hardware Tier Selector**, leave **Auto** selected (Qwen3-1.7B / Qwen3-4B / Qwen3-8B / Qwen3-14B based on RAM & GPU) or enable **Local Ollama Server** (`ollama pull qwen3:8b`) for fast NVIDIA RTX CUDA acceleration.

---

## 👥 Sharing With Friends (Spotify 2026 Dev Mode Policy)
Spotify Development Mode apps require an active Spotify Premium account on the app owner's account and support up to **5** allowlisted test users:

1. Log into the [Spotify Developer Dashboard](https://developer.spotify.com/dashboard).
2. Select your app and click **Settings** -> **User Management**.
3. Add your friends' Spotify email addresses (up to 5 accounts), or share your own public repo setup instructions so each friend creates their own free Client ID.

---

## 🧹 Troubleshooting
* **Auth or Cache Errors**: Click **🧹 Clean System Caches** on the Dashboard, or check **System Doctor** for ReccoBeats cache health, package status, and logs in `logs/`.
* **Cross-Machine Setup**: Both `.env` and `user_preferences.json` are gitignored, so hardware tier recommendations adapt automatically when friends clone onto different laptops or desktops.
