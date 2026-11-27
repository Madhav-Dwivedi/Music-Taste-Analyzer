import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

try:
    from dotenv import load_dotenv

    load_dotenv(os.path.join(BASE_DIR, ".env"))
except ImportError:
    pass

# =======================================================
# AI MODEL CONFIGURATION
# =======================================================

# 1. The Local Large Language Model (Tiered OpenVINO & Free Local Ollama)
LLM_MODEL_ID = "OpenVINO/Qwen3-4B-int4-ov"
LLM_MODEL_DIR = os.path.join(BASE_DIR, "03_Agentic_Engine", "mistral_ov_model")
LLM_MODELS_DIR = os.path.join(BASE_DIR, "03_Agentic_Engine", "models")
DEFAULT_OLLAMA_URL = "http://127.0.0.1:11434"
DEFAULT_OLLAMA_MODEL = "qwen3:8b"

LLM_MODEL_CATALOG = [
    {
        "key": "qwen3-1.7b-ov",
        "tier": 1,
        "tier_label": "Tier 1 — Compact (8 GB+ RAM)",
        "name": "Qwen3 1.7B INT4",
        "short_label": "Qwen3 1.7B",
        "repo_id": "OpenVINO/Qwen3-1.7B-int4-ov",
        "size_gb": 1.13,
        "min_ram_gb": 8,
        "recommended_hw": "8 GB RAM laptops & low-memory systems",
        "summary": "Fastest lightweight local model with clean structured JSON output.",
        "is_qwen3": True,
        "legacy": False,
    },
    {
        "key": "qwen3-4b-ov",
        "tier": 2,
        "tier_label": "Tier 2 — Balanced Laptop (16 GB RAM)",
        "name": "Qwen3 4B INT4",
        "short_label": "Qwen3 4B",
        "repo_id": "OpenVINO/Qwen3-4B-int4-ov",
        "size_gb": 2.13,
        "min_ram_gb": 12,
        "recommended_hw": "16 GB RAM (e.g. Intel Core Ultra iGPU / thin-and-light laptops)",
        "summary": "Best balance of music taste intelligence, JSON accuracy, and 16 GB RAM speed.",
        "is_qwen3": True,
        "legacy": False,
    },
    {
        "key": "phi4-mini-ov",
        "tier": 2,
        "tier_label": "Tier 2 — Balanced Laptop (16 GB RAM)",
        "name": "Phi-4 Mini 3.8B INT4",
        "short_label": "Phi-4 Mini",
        "repo_id": "OpenVINO/Phi-4-mini-instruct-int4-ov",
        "size_gb": 2.22,
        "min_ram_gb": 12,
        "recommended_hw": "16 GB RAM systems wanting crisp strict instruction following",
        "summary": "Microsoft 3.8B reasoning model; strong constraint following in a 2.2 GB footprint.",
        "is_qwen3": False,
        "legacy": False,
    },
    {
        "key": "qwen3-8b-ov",
        "tier": 3,
        "tier_label": "Tier 3 — Performance (24–32 GB RAM)",
        "name": "Qwen3 8B INT4",
        "short_label": "Qwen3 8B",
        "repo_id": "OpenVINO/Qwen3-8B-int4-ov",
        "size_gb": 4.55,
        "min_ram_gb": 24,
        "recommended_hw": "32 GB RAM (AMD Ryzen AI HX, desktops, high-memory laptops)",
        "summary": "Deep indie/subgenre music recall & smart curation; flagship everyday local model.",
        "is_qwen3": True,
        "legacy": False,
    },
    {
        "key": "qwen2.5-7b-ov",
        "tier": 3,
        "tier_label": "Tier 3 — Performance (24–32 GB RAM)",
        "name": "Qwen2.5 7B Instruct INT4",
        "short_label": "Qwen2.5 7B",
        "repo_id": "OpenVINO/Qwen2.5-7B-Instruct-int4-ov",
        "size_gb": 4.17,
        "min_ram_gb": 24,
        "recommended_hw": "24–32 GB RAM machines wanting ultra-fast 7B inference without thinking mode",
        "summary": "Battle-tested 7B workhorse with broad music catalog knowledge.",
        "is_qwen3": False,
        "legacy": False,
    },
    {
        "key": "qwen3-14b-ov",
        "tier": 4,
        "tier_label": "Tier 4 — Enthusiast (32 GB+ RAM)",
        "name": "Qwen3 14B INT4",
        "short_label": "Qwen3 14B",
        "repo_id": "OpenVINO/Qwen3-14B-int4-ov",
        "size_gb": 9.06,
        "min_ram_gb": 30,
        "recommended_hw": "32 GB+ high-bandwidth desktops & workstations",
        "summary": "Highest local OpenVINO taste nuance and rare-track recall (~9.1 GB download).",
        "is_qwen3": True,
        "legacy": False,
    },
    {
        "key": "mistral-7b-legacy-ov",
        "tier": 0,
        "tier_label": "Legacy — Compatibility",
        "name": "Mistral 7B v0.2 INT4 (Legacy)",
        "short_label": "Mistral 7B",
        "repo_id": "OpenVINO/Mistral-7B-Instruct-v0.2-int4-ov",
        "size_gb": 4.30,
        "min_ram_gb": 16,
        "recommended_hw": "Existing mistral_ov_model cache compatibility",
        "summary": "Original 2023 model kept only for compatibility with previously downloaded weights.",
        "is_qwen3": False,
        "legacy": True,
    },
]

OLLAMA_RECOMMENDED_TIERS = [
    {"tier": 1, "min_vram_gb": 0, "model": "qwen3:4b", "label": "4B Fast (<6 GB VRAM / iGPU)"},
    {"tier": 2, "min_vram_gb": 5.5, "model": "qwen3:8b", "label": "8B Balanced (RTX 4050 6GB / RTX 4060 8GB)"},
    {"tier": 3, "min_vram_gb": 7.5, "model": "gemma3:12b", "label": "12B Quality (RTX 4060 8GB+ / 32GB RAM)"},
    {"tier": 4, "min_vram_gb": 10.0, "model": "qwen3:14b", "label": "14B Enthusiast (32GB+ RAM / 12GB+ VRAM)"},
]

# 2. The Vector Embedding Model (CLAP Audio+Text & OpenVINO MiniLM)
EMBED_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
EMBED_MODEL_DIR = os.path.join(BASE_DIR, "02_Vector_Core", "embed_ov_model")
CLAP_MODEL_ID = "laion/clap-htsat-unfused"
CLAP_TEXT_BLEND_WEIGHT = 0.28
DEFAULT_EMBED_BACKEND = "minilm"
DEFAULT_PLAYLIST_COOCCURRENCE_WEIGHT = 0.12
CATCHALL_PLAYLIST_RATIO = 0.40

# =======================================================
# SPOTIFY API CONFIGURATION
# =======================================================
SPOTIFY_CLIENT_ID = os.environ.get("SPOTIFY_CLIENT_ID", "").strip()
SPOTIFY_REDIRECT_URI = os.environ.get("SPOTIFY_REDIRECT_URI", "http://127.0.0.1:8080")

# =======================================================
# YOUTUBE MUSIC CONFIGURATION
# =======================================================
YTMUSIC_BROWSER_AUTH_FILE = os.path.join(BASE_DIR, "ytm_browser.json")
YTMUSIC_OAUTH_FILE = os.path.join(BASE_DIR, "ytm_oauth.json")
YTMUSIC_OAUTH_CLIENT_FILE = os.path.join(BASE_DIR, ".ytm_oauth_client.json")
YTMUSIC_PROFILE_FILE = os.path.join(BASE_DIR, "01_Data_Extractor", "ytmusic_profile.json")
DEFAULT_ACTIVE_PROVIDER = "spotify"

# =======================================================
# PIPELINE & PREFERENCE SETTINGS
# =======================================================
USER_PREFS_FILE = os.path.join(BASE_DIR, "user_preferences.json")
USER_FEEDBACK_FILE = os.path.join(BASE_DIR, "user_feedback.json")
MAX_LIKED_SONGS = 500
RECCOBEATS_BATCH_SIZE = 40
RECCOBEATS_NOT_FOUND_TTL_DAYS = 14
EXTENDED_HISTORY_HALF_LIFE_DAYS = 120.0
MIN_CLUSTERS = 3
MAX_CLUSTERS = 14
MAX_CLUSTER_SHARE = 0.22
DEFAULT_CLUSTER_METHOD = "gmm"
GMM_PCA_DIMS = 20

# Optional per-artist clustering weight multipliers (exact artist names, case-insensitive).
# You can also define {"penalized_artists": {"Artist Name": 0.05}} in user_preferences.json.
PENALIZED_ARTISTS = {}

SKIP_PENALTY_ENABLED = True
DEFAULT_PLAYLIST_SIZE = 30
DEFAULT_AVOID_ARTISTS = []
DEFAULT_AVOID_GENRES = []

# =======================================================
# WEB STUDIO & STORAGE CONFIGURATION
# =======================================================
WEB_HOST = "127.0.0.1"
WEB_PORT = int(os.environ.get("MUSIC_ANALYZER_PORT") or os.environ.get("SPOTMUS_PORT") or "8765")
PLAYLIST_HISTORY_DIR = os.path.join(BASE_DIR, "03_Agentic_Engine", "playlist_history")
LOGS_DIR = os.path.join(BASE_DIR, "logs")

