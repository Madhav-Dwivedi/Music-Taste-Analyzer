import json
import os
import re
import sys

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import config
from utils import (
    detect_hardware_profile,
    get_effective_preferences,
    get_openvino_devices,
    matches_avoided_artist,
    matches_avoided_genre,
    normalize_avoid_list,
)

_KNOWN_GENRE_TERMS = {
    "country", "metal", "nu metal", "heavy metal", "pop", "indie pop", "dance pop",
    "rock", "indie rock", "alt rock", "alternative rock", "punk", "pop punk", "emo",
    "rap", "hip hop", "hip-hop", "trap", "r&b", "rnb", "soul", "neo soul",
    "edm", "electronic", "house", "deep house", "tropical house", "techno", "dubstep", "trance",
    "synthwave", "lofi", "lo-fi", "jazz", "blues", "folk", "acoustic", "classical",
    "ambient", "funk", "disco", "latin", "reggaeton", "k-pop", "kpop", "afrobeats", "indie", "alternative",
}

MODEL_ID = config.LLM_MODEL_ID
MODEL_DIR = config.LLM_MODEL_DIR

_GENERIC_QUERIES = {
    "feel good",
    "good vibes",
    "vibes",
    "songs",
    "song",
    "playlist",
    "music",
    "tracks",
    "mix",
    "best songs",
}

_GENRE_KEYWORDS = [
    (["synthwave", "retrowave", "outrun"], ["genre:synthwave", "genre:synthpop", "genre:electronic"]),
    (["lofi", "lo-fi", "chillhop", "study"], ["genre:lo-fi", "genre:chillhop", "genre:ambient"]),
    (["indie pop", "dream pop", "bedroom pop"], ["genre:indie pop", "genre:dream pop", "genre:indie"]),
    (["indie rock", "indie"], ["genre:indie rock", "genre:indie", "genre:alternative"]),
    (["alt rock", "alternative rock", "alternative"], ["genre:alternative rock", "genre:modern rock", "genre:rock"]),
    (["nu metal", "linkin park", "metal"], ["genre:nu metal", "genre:alternative metal", "genre:hard rock"]),
    (["punk", "pop punk", "emo"], ["genre:pop punk", "genre:emo", "genre:punk"]),
    (["hip hop", "hip-hop", "rap", "trap"], ["genre:hip hop", "genre:rap", "genre:pop rap"]),
    (["r&b", "rnb", "soul", "neo soul"], ["genre:r&b", "genre:neo soul", "genre:soul"]),
    (["tropical house", "deep house", "house"], ["genre:house", "genre:tropical house", "genre:dance"]),
    (["edm", "electronic", "techno", "dubstep", "trance"], ["genre:electronic", "genre:edm", "genre:dance"]),
    (["workout", "gym", "running", "cardio", "hiit", "hype", "pump"], ["genre:dance pop", "genre:pop rap", "genre:electronic"]),
    (["acoustic", "folk", "singer-songwriter"], ["genre:acoustic", "genre:folk", "genre:indie folk"]),
    (["jazz", "blues"], ["genre:jazz", "genre:smooth jazz", "genre:blues"]),
    (["funk", "disco", "groove"], ["genre:funk", "genre:disco", "genre:nu disco"]),
    (["latin", "reggaeton"], ["genre:latin", "genre:reggaeton", "genre:latin pop"]),
    (["country"], ["genre:country", "genre:contemporary country", "genre:americana"]),
    (["classical", "orchestral", "piano"], ["genre:classical", "genre:neoclassical", "genre:ambient"]),
    (["ambient", "sleep", "calm", "chill", "relax"], ["genre:ambient", "genre:chillout", "genre:downtempo"]),
    (["pop", "upbeat", "happy", "party", "summer", "feel good"], ["genre:pop", "genre:dance pop", "genre:indie pop"]),
    (["rock"], ["genre:rock", "genre:modern rock", "genre:alternative rock"]),
    (["sad", "melancholy", "heartbreak", "dark", "moody", "depressing"], ["genre:sad", "genre:indie", "genre:alternative r&b"]),
]

_FALLBACK_QUERY_POOL = [
    "genre:indie pop",
    "genre:alternative rock",
    "genre:pop",
    "genre:electronic",
    "genre:hip hop",
    "genre:r&b",
    "genre:chill",
    "genre:modern rock",
    "genre:synthpop",
    "genre:indie",
    "year:2018-2025",
    "year:2010-2019",
]


_PARSE_PROMPT_SCHEMA = {
    "type": "object",
    "properties": {
        "search_queries": {"type": "array", "items": {"type": "string"}},
        "target_artist": {"type": ["string", "null"]},
        "exclude_artists": {"type": "array", "items": {"type": "string"}},
        "exclude_genres": {"type": "array", "items": {"type": "string"}},
        "clean_only": {"type": "boolean"},
        "deep_cuts": {"type": "boolean"},
        "playlist_size": {"type": "integer"},
        "local_only": {"type": "boolean"},
        "known_tracks_ratio": {"type": "number"},
        "min_bpm": {"type": ["integer", "null"]},
        "max_bpm": {"type": ["integer", "null"]},
        "target_energy": {"type": ["number", "null"]},
        "target_valence": {"type": ["number", "null"]},
        "recommended_tracks": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "artist": {"type": "string"},
                },
                "required": ["name", "artist"],
            },
        },
        "playlist_name": {"type": "string"},
        "playlist_description": {"type": "string"},
    },
    "required": ["search_queries", "playlist_name", "playlist_description"],
}

_VIBE_CHECK_SCHEMA = {
    "type": "object",
    "properties": {
        "selected_indices": {"type": "array", "items": {"type": "integer"}},
        "playlist_name": {"type": "string"},
        "playlist_description": {"type": "string"},
    },
    "required": ["selected_indices", "playlist_name", "playlist_description"],
}

_REFINEMENT_SCHEMA = {
    "type": "object",
    "properties": {
        "drop_track_numbers": {"type": "array", "items": {"type": "integer"}},
        "remove_artists": {"type": "array", "items": {"type": "string"}},
        "filter_condition": {"type": ["string", "null"]},
        "delta_tracks": {"type": "integer"},
        "target_size": {"type": ["integer", "null"]},
        "flow_mode": {"type": ["string", "null"]},
        "direction_prompt": {"type": ["string", "null"]},
    },
    "required": ["drop_track_numbers", "remove_artists"],
}


def _has_complete_llm_files(model_dir):
    if not os.path.isdir(model_dir):
        return False
    xml_path = os.path.join(model_dir, "openvino_model.xml")
    bin_path = os.path.join(model_dir, "openvino_model.bin")
    if os.path.isfile(xml_path) and os.path.isfile(bin_path):
        try:
            return os.path.getsize(bin_path) > 0
        except OSError:
            return False
    return False


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


def get_catalog_entry(model_key):
    catalog = list(getattr(config, "LLM_MODEL_CATALOG", []) or [])
    for entry in catalog:
        if entry.get("key") == model_key:
            return dict(entry)
    for entry in catalog:
        if entry.get("repo_id") == MODEL_ID:
            return dict(entry)
    for entry in catalog:
        if entry.get("key") == "qwen3-4b-ov":
            return dict(entry)
    if catalog:
        return dict(catalog[0])
    return {
        "key": "default-ov",
        "tier": 2,
        "tier_label": "Tier 2 — Balanced",
        "name": MODEL_ID,
        "short_label": "Local LLM",
        "repo_id": MODEL_ID,
        "size_gb": 2.2,
        "min_ram_gb": 12,
        "is_qwen3": "qwen3" in str(MODEL_ID).lower(),
        "legacy": False,
    }


def resolve_model_dir(entry, base_dir=None):
    bdir = base_dir or ROOT_DIR
    key = str((entry or {}).get("key") or "")
    if key == "mistral-7b-legacy-ov":
        return os.path.join(bdir, "03_Agentic_Engine", "mistral_ov_model") if base_dir else MODEL_DIR
    models_root = os.path.join(bdir, "03_Agentic_Engine", "models") if base_dir else getattr(
        config, "LLM_MODELS_DIR", os.path.join(ROOT_DIR, "03_Agentic_Engine", "models")
    )
    return os.path.join(models_root, key)


def get_llm_catalog_status(base_dir=None, probe_ollama=False):
    prefs = get_effective_preferences()
    hw = detect_hardware_profile(probe_ollama=probe_ollama, ollama_url=prefs.get("ollama_url"))
    rec_key = hw.get("recommended_model_key", "qwen3-4b-ov")
    pref_key = prefs.get("llm_model_key", "auto")
    resolved_key = rec_key if pref_key == "auto" else pref_key
    backend = prefs.get("llm_backend", "openvino")
    ram_total = hw.get("ram_total_gb") or 16.0

    catalog_out = []
    for raw_item in getattr(config, "LLM_MODEL_CATALOG", []) or []:
        item = dict(raw_item)
        mdir = resolve_model_dir(item, base_dir=base_dir)
        dl = _has_complete_llm_files(mdir)
        item["downloaded"] = dl
        item["size_on_disk_mb"] = _dir_size_mb(mdir) if dl else 0.0
        item["model_dir"] = mdir
        item["recommended"] = item.get("key") == rec_key
        item["active"] = backend == "openvino" and item.get("key") == resolved_key
        item["fits_ram"] = ram_total >= float(item.get("min_ram_gb", 8))
        catalog_out.append(item)

    active_entry = get_catalog_entry(resolved_key)
    if backend == "openvino":
        active_label = active_entry.get("short_label") or active_entry.get("name")
    elif backend == "ollama":
        active_label = f"Ollama ({prefs.get('ollama_model')})"
    else:
        active_label = f"Local API ({prefs.get('ollama_model')})"

    return {
        "active": {
            "llm_backend": backend,
            "llm_model_key": pref_key,
            "resolved_model_key": resolved_key,
            "llm_device": prefs.get("llm_device", "auto"),
            "ollama_url": prefs.get("ollama_url"),
            "ollama_model": prefs.get("ollama_model"),
            "label": active_label,
        },
        "hardware": hw,
        "catalog": catalog_out,
        "ollama_tiers": list(getattr(config, "OLLAMA_RECOMMENDED_TIERS", []) or []),
    }


def _make_generation_config(max_new_tokens, temperature):
    try:
        import openvino_genai as ov_genai

        cfg = ov_genai.GenerationConfig()
        cfg.max_new_tokens = int(max_new_tokens)
        cfg.temperature = float(temperature)
        return cfg
    except ImportError:
        return None


class LocalLLMClient:
    """Unified wrapper across OpenVINO (with XGrammar JSON Schema), Ollama, and local OpenAI-compatible servers."""

    def __init__(
        self,
        backend="openvino",
        model_key="qwen3-4b-ov",
        label="Qwen3 4B",
        device="GPU",
        pipe=None,
        base_url=None,
        ollama_model=None,
        is_qwen3=False,
    ):
        self.backend = backend
        self.model_key = model_key
        self.label = label
        self.device = device
        self.pipe = pipe
        self.base_url = str(base_url or getattr(config, "DEFAULT_OLLAMA_URL", "http://127.0.0.1:11434")).rstrip("/")
        self.ollama_model = str(ollama_model or getattr(config, "DEFAULT_OLLAMA_MODEL", "qwen3:8b"))
        self.is_qwen3 = bool(is_qwen3)

    def _build_openvino_prompt(self, system_msg, user_msg):
        sys_txt = str(system_msg or "").strip()
        usr_txt = str(user_msg or "").strip()
        if self.is_qwen3 and "/no_think" not in usr_txt:
            usr_txt = f"{usr_txt}\n/no_think"
        messages = [
            {"role": "system", "content": sys_txt},
            {"role": "user", "content": usr_txt},
        ]
        try:
            tok = self.pipe.get_tokenizer()
            if tok is not None and hasattr(tok, "apply_chat_template"):
                if self.is_qwen3:
                    try:
                        return tok.apply_chat_template(
                            messages,
                            add_generation_prompt=True,
                            extra_context={"enable_thinking": False},
                        )
                    except TypeError:
                        pass
                return tok.apply_chat_template(messages, add_generation_prompt=True)
        except Exception:
            pass
        return f"System:\n{sys_txt}\n\nUser:\n{usr_txt}\n\nAssistant:\n"

    def _generate_openvino_json(self, system_msg, user_msg, json_schema=None, max_new_tokens=380, temperature=0.1):
        import openvino_genai as ov_genai

        prompt = self._build_openvino_prompt(system_msg, user_msg)
        cfg = ov_genai.GenerationConfig()
        cfg.max_new_tokens = int(max_new_tokens)
        cfg.temperature = float(temperature)
        cfg.do_sample = bool(temperature > 0.05)

        used_schema = False
        if json_schema is not None and hasattr(ov_genai, "StructuredOutputConfig"):
            try:
                cfg.structured_output_config = ov_genai.StructuredOutputConfig(json_schema=json.dumps(json_schema))
                used_schema = True
            except Exception:
                used_schema = False

        try:
            return self.pipe.generate(prompt, cfg)
        except Exception:
            if used_schema:
                cfg_plain = ov_genai.GenerationConfig()
                cfg_plain.max_new_tokens = int(max_new_tokens)
                cfg_plain.temperature = float(temperature)
                cfg_plain.do_sample = bool(temperature > 0.05)
                return self.pipe.generate(prompt, cfg_plain)
            raise

    def _generate_ollama_json(self, system_msg, user_msg, json_schema=None, max_new_tokens=380, temperature=0.1):
        import requests

        endpoint = f"{self.base_url}/api/chat"
        payload = {
            "model": self.ollama_model,
            "messages": [
                {"role": "system", "content": str(system_msg or "")},
                {"role": "user", "content": str(user_msg or "")},
            ],
            "stream": False,
            "think": False,
            "format": json_schema if isinstance(json_schema, dict) else "json",
            "options": {"temperature": float(temperature), "num_predict": int(max_new_tokens)},
        }
        resp = requests.post(endpoint, json=payload, timeout=90)
        if resp.status_code == 400 and "think" in resp.text.lower():
            payload.pop("think", None)
            resp = requests.post(endpoint, json=payload, timeout=90)
        if resp.status_code != 200:
            raise RuntimeError(f"Ollama HTTP {resp.status_code}: {resp.text[:200]}")
        data = resp.json()
        return str(((data.get("message") or {}).get("content")) or "")

    def _generate_openai_local_json(self, system_msg, user_msg, max_new_tokens=380, temperature=0.1):
        import requests

        if self.base_url.endswith("/chat/completions"):
            endpoint = self.base_url
        elif self.base_url.endswith("/v1"):
            endpoint = f"{self.base_url}/chat/completions"
        else:
            endpoint = f"{self.base_url}/v1/chat/completions"
        payload = {
            "model": self.ollama_model,
            "messages": [
                {"role": "system", "content": str(system_msg or "")},
                {"role": "user", "content": str(user_msg or "")},
            ],
            "temperature": float(temperature),
            "max_tokens": int(max_new_tokens),
            "response_format": {"type": "json_object"},
        }
        resp = requests.post(endpoint, json=payload, timeout=90)
        if resp.status_code == 400 and "response_format" in payload:
            payload.pop("response_format", None)
            resp = requests.post(endpoint, json=payload, timeout=90)
        if resp.status_code != 200:
            raise RuntimeError(f"Local API HTTP {resp.status_code}: {resp.text[:200]}")
        data = resp.json()
        choices = data.get("choices") or []
        if not choices:
            return ""
        return str(((choices[0].get("message") or {}).get("content")) or "")

    def generate_json(self, system_msg, user_msg, json_schema=None, max_new_tokens=380, temperature=0.1):
        if self.backend == "ollama":
            return self._generate_ollama_json(
                system_msg, user_msg, json_schema=json_schema, max_new_tokens=max_new_tokens, temperature=temperature
            )
        if self.backend == "openai_local":
            return self._generate_openai_local_json(
                system_msg, user_msg, max_new_tokens=max_new_tokens, temperature=temperature
            )
        return self._generate_openvino_json(
            system_msg, user_msg, json_schema=json_schema, max_new_tokens=max_new_tokens, temperature=temperature
        )


def get_llm(model_key=None, backend=None, device=None, ollama_url=None, ollama_model=None):
    prefs = get_effective_preferences()
    eff_backend = str(backend or prefs.get("llm_backend") or "openvino").strip().lower()
    eff_key = str(model_key or prefs.get("llm_model_key") or "auto").strip()
    eff_dev = str(device or prefs.get("llm_device") or "auto").strip()
    eff_url = str(ollama_url or prefs.get("ollama_url") or getattr(config, "DEFAULT_OLLAMA_URL", "http://127.0.0.1:11434"))
    eff_omodel = str(ollama_model or prefs.get("ollama_model") or getattr(config, "DEFAULT_OLLAMA_MODEL", "qwen3:8b"))

    if eff_backend in ("ollama", "openai_local"):
        label = f"Ollama: {eff_omodel}" if eff_backend == "ollama" else f"Local API: {eff_omodel}"
        print(f"[LLM] Using {label} at {eff_url}", flush=True)
        return LocalLLMClient(
            backend=eff_backend,
            model_key=eff_omodel,
            label=label,
            device="CUDA/Local GPU",
            base_url=eff_url,
            ollama_model=eff_omodel,
            is_qwen3="qwen3" in eff_omodel.lower(),
        )

    from huggingface_hub import snapshot_download
    import openvino_genai as ov_genai

    if eff_key == "auto":
        hw = detect_hardware_profile(probe_ollama=False)
        eff_key = hw.get("recommended_model_key", "qwen3-4b-ov")

    entry = get_catalog_entry(eff_key)
    target_dir = resolve_model_dir(entry)
    repo_id = entry.get("repo_id") or MODEL_ID
    short_label = entry.get("short_label") or entry.get("name") or eff_key

    if not _has_complete_llm_files(target_dir):
        print(f"\n[LLM] Downloading {entry['name']} ({entry['size_gb']} GB OpenVINO INT4) from {repo_id}...", flush=True)
        os.makedirs(target_dir, exist_ok=True)
        snapshot_download(repo_id=repo_id, local_dir=target_dir)

    print(f"[LLM] Loading {short_label} — detecting optimal accelerator...", flush=True)
    dev_info = get_openvino_devices()

    order = []
    if eff_dev.lower() != "auto":
        req_upper = eff_dev.upper()
        if req_upper == "GPU" and dev_info.get("gpu"):
            order.append(dev_info["gpu"])
        elif req_upper == "NPU" and dev_info.get("npu"):
            order.append(dev_info["npu"])
        elif req_upper == "CPU":
            order.append(dev_info.get("cpu", "CPU"))
    # Default OpenVINO LLM order: GPU -> CPU -> NPU (GPU/CPU outperform 1st-gen NPU for LLMs)
    if dev_info.get("gpu") and dev_info["gpu"] not in order:
        order.append(dev_info["gpu"])
    cpu_dev = dev_info.get("cpu", "CPU")
    if cpu_dev and cpu_dev not in order:
        order.append(cpu_dev)
    if dev_info.get("npu") and dev_info["npu"] not in order:
        order.append(dev_info["npu"])

    pipe = None
    chosen_dev = "CPU"
    last_err = None
    for cand_dev in order:
        try:
            print(f"      Trying {short_label} on {cand_dev}...", flush=True)
            pipe = ov_genai.LLMPipeline(target_dir, cand_dev)
            chosen_dev = cand_dev
            print(f"      ✅ Loaded {short_label} onto {cand_dev}!", flush=True)
            break
        except Exception as e:
            last_err = e
            print(f"      ⚠️ {cand_dev} load failed: {e}", flush=True)

    if pipe is None:
        raise RuntimeError(f"Failed to load {short_label} on any OpenVINO device: {last_err}")

    return LocalLLMClient(
        backend="openvino",
        model_key=entry["key"],
        label=short_label,
        device=chosen_dev,
        pipe=pipe,
        is_qwen3=bool(entry.get("is_qwen3")),
    )


def extract_json_object(raw_text):
    """Extract the first balanced JSON object from raw LLM output, stripping <think> blocks and markdown."""
    if not raw_text or not isinstance(raw_text, str):
        return None

    text = re.sub(r"<think>.*?</think>", "", raw_text, flags=re.IGNORECASE | re.DOTALL).strip()
    fence_match = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
    if fence_match:
        text = fence_match.group(1).strip()

    start_idx = text.find("{")
    if start_idx == -1:
        return None

    depth = 0
    in_string = False
    escape_next = False
    end_idx = -1

    for idx in range(start_idx, len(text)):
        ch = text[idx]
        if escape_next:
            escape_next = False
            continue
        if ch == "\\" and in_string:
            escape_next = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if not in_string:
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    end_idx = idx
                    break

    if end_idx == -1:
        return None

    candidate = text[start_idx : end_idx + 1]
    for attempt_str in (candidate, re.sub(r",\s*([}\]])", r"\1", candidate)):
        try:
            parsed = json.loads(attempt_str)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            continue
    return None


def _clamp_int(val, low, high, default):
    if isinstance(val, bool) or val is None:
        return default
    try:
        return max(low, min(high, int(val)))
    except (TypeError, ValueError):
        return default


def _clamp_float(val, low, high, default):
    if isinstance(val, bool) or val is None:
        return default
    try:
        return round(max(low, min(high, float(val))), 3)
    except (TypeError, ValueError):
        return default


def _clean_artist_phrase(raw):
    cleaned = re.sub(r"\s+(?:vibes?|style|type|music|songs?|tracks?|playlist|mix)$", "", str(raw or "").strip(), flags=re.IGNORECASE)
    cleaned = cleaned.strip(" .,!?\"'")
    if not cleaned or cleaned.lower() in {"my library", "library", "my playlist", "spotify", "me", "favorites", "liked songs"}:
        return None
    return cleaned


def _build_aesthetic_title(user_prompt):
    stop_words = {
        "make", "a", "an", "the", "me", "playlist", "song", "songs", "track", "tracks",
        "with", "for", "from", "my", "library", "brand", "new", "only", "give", "create",
        "mix", "vibe", "vibes", "and", "or", "some",
    }
    words = [
        w
        for w in re.findall(r"[A-Za-z0-9']+", str(user_prompt or ""))
        if not w.isdigit() and w.lower() not in stop_words
    ]
    if not words:
        return "Custom Taste Discovery"
    title_words = [w.capitalize() for w in words[:4]]
    return " ".join(title_words)[:60]


def _query_matches_exclusions(q_str, exclude_artists, exclude_genres):
    q_low = str(q_str or "").strip().lower()
    if not q_low:
        return True
    if q_low.startswith("artist:"):
        art_val = q_low.split(":", 1)[1].strip().strip('"').strip("'")
        if art_val in {a.lower() for a in exclude_artists}:
            return True
    if q_low.startswith("genre:"):
        gen_val = q_low.split(":", 1)[1].strip().strip('"').strip("'")
        if matches_avoided_genre(gen_val, exclude_genres):
            return True
    # Also guard plain queries containing avoided artists or genres
    for art in exclude_artists:
        if re.search(rf"(?<![A-Za-z0-9]){re.escape(art.lower())}(?![A-Za-z0-9])", q_low):
            return True
    if exclude_genres and matches_avoided_genre(q_low.replace("genre:", ""), exclude_genres):
        return True
    return False


def parse_prompt_heuristic(user_prompt, past_queries=None, avoid_artists=None, avoid_genres=None):
    """Deterministic prompt parser for --fast mode, offline fallback, and LLM safety merging."""
    text = str(user_prompt or "").strip()
    lower = text.lower()
    past_set = {str(q).strip().lower() for q in (past_queries or []) if q}
    merged_avoid_artists = normalize_avoid_list(avoid_artists)
    merged_avoid_genres = normalize_avoid_list(avoid_genres)

    # 1. Playlist size, ratio, clean_only, deep_cuts
    playlist_size = 30
    known_tracks_ratio = 0.3
    local_only = False
    has_explicit_new_split = False
    clean_only = any(
        p in lower
        for p in ("clean only", "no explicit", "without explicit", "no profanity", "family friendly", "radio friendly")
    )
    deep_cuts = any(
        p in lower
        for p in ("deep cuts", "deep-cuts", "underground", "hidden gems", "lesser known", "lesser-known", "obscure", "b-sides")
    )

    split_known_first = re.search(
        r"(\d+)\s*(?:songs?|tracks?)?\s*(?:from\s+(?:my\s+)?(?:library|saved|liked|local)|known).*?(\d+)\s*(?:brand\s+)?new",
        lower,
    )
    split_new_first = re.search(
        r"(\d+)\s*(?:brand\s+)?new(?:\s+songs?|\s+tracks?)?.*?(\d+)\s*(?:songs?|tracks?)?\s*(?:from\s+(?:my\s+)?(?:library|saved|liked|local)|known)",
        lower,
    )
    if split_known_first:
        k_cnt = int(split_known_first.group(1))
        n_cnt = int(split_known_first.group(2))
        total = max(1, k_cnt + n_cnt)
        playlist_size = _clamp_int(total, 5, 100, 30)
        known_tracks_ratio = round(k_cnt / total, 2)
        has_explicit_new_split = True
    elif split_new_first:
        n_cnt = int(split_new_first.group(1))
        k_cnt = int(split_new_first.group(2))
        total = max(1, k_cnt + n_cnt)
        playlist_size = _clamp_int(total, 5, 100, 30)
        known_tracks_ratio = round(k_cnt / total, 2)
        has_explicit_new_split = True
    else:
        size_match = re.search(r"(?:playlist\s+of|make\s+(?:a|me\s+a)?|give\s+me)?\s*\b(\d{1,3})\s*(?:song|track)s?\b", lower)
        if not size_match:
            size_match = re.search(r"\b(?:playlist\s+of|make\s+(?:a|me\s+a)?)\s*(\d{1,3})\b", lower)
        if size_match:
            playlist_size = _clamp_int(size_match.group(1), 5, 100, 30)

    if not has_explicit_new_split:
        if any(
            phrase in lower
            for phrase in (
                "only from my library",
                "from my library only",
                "my library only",
                "songs i know",
                "just my library",
                "saved songs only",
                "liked songs only",
                "local only",
            )
        ):
            local_only = True
            known_tracks_ratio = 1.0
        elif "from my library" in lower and "new" not in lower and "discover" not in lower:
            local_only = True
            known_tracks_ratio = 1.0
        elif any(phrase in lower for phrase in ("all new", "100% new", "brand new only", "only new songs")):
            known_tracks_ratio = 0.0

    # 2. Target artist (only when user asks specifically for tracks by/from an artist)
    target_artist = None
    explicit_artist_patterns = [
        r"(?i)\b(?:only|just)\s+(?:songs?\s+|tracks?\s+)?(?:by|from)\s+([A-Za-z0-9][A-Za-z0-9 &'’.\-]{1,40}?)(?=\s*(?:,|with\b|under\b|above\b|over\b|from\b|in\b|on\b|\d+\s*bpm|$|\.|!))",
        r"(?i)\b([A-Za-z0-9][A-Za-z0-9 &'’.\-]{1,40}?)\s+discography\b",
        r"(?i)\b(?:all|only)\s+([A-Za-z0-9][A-Za-z0-9 &'’.\-]{1,40}?)\s+(?:songs|tracks)\b",
        r"(?i)\b(?:songs|tracks)\s+(?:by|from)\s+([A-Za-z0-9][A-Za-z0-9 &'’.\-]{1,40}?)(?=\s*(?:,|with\b|only\b|under\b|above\b|over\b|\d+\s*bpm|$|\.|!))",
    ]
    for pat in explicit_artist_patterns:
        m = re.search(pat, text)
        if m:
            cand = _clean_artist_phrase(m.group(1))
            if cand and not any(w in m.group(0).lower() for w in ("vibes", "like ")):
                target_artist = cand
                break

    # Artist inspirations (e.g. "sad linkin park vibes", "artists like Tame Impala")
    inspired_artists = []
    for pat in (
        r"(?i)\blike\s+([A-Za-z0-9][A-Za-z0-9 &'’.\-]{1,35}?)(?=\s*(?:,|and\b|with\b|vibes?\b|songs?\b|tracks?\b|$|\.|!))",
        r"(?i)(?:\bwith\b|\bof\b|\bfor\b|\bplaylist\b|\bsongs?\b|\btracks?\b|,|^)\s*(?:sad|happy|dark|chill|upbeat|hype|heavy|mellow|old|new)?\s*([A-Za-z0-9][A-Za-z0-9'’&-]{1,25}(?:\s+[A-Za-z0-9'’&-]{1,20}){0,2})\s+vibes?\b",
    ):
        for m in re.finditer(pat, text):
            cand = _clean_artist_phrase(m.group(1))
            if cand and cand.lower() not in {"sad", "happy", "chill", "hype", "good", "dark", "late night", "summer", "workout"}:
                cand = re.sub(r"^(?:with|for|of|sad|happy|dark|chill|upbeat|hype|heavy|mellow|old|new)\s+", "", cand, flags=re.IGNORECASE).strip()
                if cand and cand.lower() not in {"good", "sad", "happy", "chill", "hype", "dark", "summer", "workout", "late night"}:
                    inspired_artists.append(cand)

    # 3. Excluded artists & excluded genres
    exclude_artists = list(merged_avoid_artists)
    exclude_genres = list(merged_avoid_genres)
    stop_excludes = {"lyrics", "vocals", "skips", "duplicates", "slow", "fast", "sad", "happy", "explicit", "profanity", "bad"}
    for m in re.finditer(
        r"(?i)\b(?:no|without|avoid|exclude|except)\s+([A-Za-z0-9][A-Za-z0-9 &'’.\-]{1,35}?)(?=\s*(?:,|and\b|or\b|with\b|songs?\b|tracks?\b|playlist\b|$|\.|!))",
        text,
    ):
        cand = _clean_artist_phrase(m.group(1))
        if not cand or cand.lower() in stop_excludes:
            continue
        cand_no_genre = re.sub(r"\s+(?:genre|genres|music)$", "", cand, flags=re.IGNORECASE).strip()
        if cand_no_genre.lower() in _KNOWN_GENRE_TERMS or cand.lower().endswith((" genre", " genres")):
            if cand_no_genre and cand_no_genre.lower() not in {g.lower() for g in exclude_genres}:
                exclude_genres.append(cand_no_genre)
        elif cand.lower() not in {a.lower() for a in exclude_artists}:
            exclude_artists.append(cand)

    # 4. Acoustic constraints (BPM, energy, valence)
    min_bpm = None
    max_bpm = None
    bpm_range = re.search(r"(\d{2,3})\s*-\s*(\d{2,3})\s*bpm", lower)
    if bpm_range:
        min_bpm = _clamp_int(bpm_range.group(1), 40, 240, None)
        max_bpm = _clamp_int(bpm_range.group(2), 40, 240, None)
    else:
        max_bpm_m = re.search(r"(?:under|below|max|less\s+than)\s*(\d{2,3})\s*bpm", lower)
        min_bpm_m = re.search(r"(?:above|over|min|at\s+least)?\s*(\d{2,3})\+?\s*bpm", lower)
        if max_bpm_m:
            max_bpm = _clamp_int(max_bpm_m.group(1), 40, 240, None)
        elif min_bpm_m:
            min_bpm = _clamp_int(min_bpm_m.group(1), 40, 240, None)

    target_energy = None
    target_valence = None

    if any(w in lower for w in ("workout", "gym", "running", "cardio", "hiit", "hype", "pump", "intense", "rage")):
        target_energy = 0.85
        if min_bpm is None and max_bpm is None:
            min_bpm = 125
    elif any(w in lower for w in ("fast", "uptempo", "energetic", "high energy")):
        target_energy = 0.78
        if min_bpm is None and max_bpm is None:
            min_bpm = 120
    elif any(w in lower for w in ("chill", "lofi", "lo-fi", "ambient", "sleep", "study", "calm", "relax", "mellow", "acoustic")):
        target_energy = 0.32
        if max_bpm is None and min_bpm is None:
            max_bpm = 115

    if any(w in lower for w in ("happy", "upbeat", "feel good", "feel-good", "joyful", "euphoric", "party", "summer")):
        target_valence = 0.8
    elif any(w in lower for w in ("sad", "melancholy", "heartbreak", "depressing", "gloomy", "cry", "lonely", "dark")):
        target_valence = 0.22

    # 5. Build search queries
    queries = []
    if target_artist:
        queries.append(f"artist:{target_artist}")
    for art in inspired_artists:
        q = f"artist:{art}"
        if q not in queries:
            queries.append(q)

    for keywords, mapped_queries in _GENRE_KEYWORDS:
        if any(kw in lower for kw in keywords):
            for mq in mapped_queries:
                if mq not in queries:
                    queries.append(mq)
        if len(queries) >= 5:
            break

    decade_map = {
        r"\b(?:19)?80s\b": "year:1980-1989",
        r"\b(?:19)?90s\b": "year:1990-1999",
        r"\b(?:20)?00s\b": "year:2000-2009",
        r"\b(?:20)?10s\b": "year:2010-2019",
        r"\b(?:20)?20s\b": "year:2020-2026",
    }
    for pat, yq in decade_map.items():
        if re.search(pat, lower) and yq not in queries:
            queries.append(yq)

    if not queries:
        queries = ["genre:indie pop", "genre:alternative", "genre:pop", "genre:electronic"]

    filtered_queries = [
        q
        for q in queries
        if q.lower() not in past_set and not _query_matches_exclusions(q, exclude_artists, exclude_genres)
    ]
    if not filtered_queries:
        for fallback_q in _FALLBACK_QUERY_POOL:
            if (
                fallback_q.lower() not in past_set
                and fallback_q not in filtered_queries
                and not _query_matches_exclusions(fallback_q, exclude_artists, exclude_genres)
            ):
                filtered_queries.append(fallback_q)
            if len(filtered_queries) >= 4:
                break

    if not filtered_queries:
        filtered_queries = ["year:2018-2026", "year:2010-2019"]

    return {
        "search_queries": filtered_queries[:5],
        "target_artist": target_artist,
        "exclude_artists": exclude_artists,
        "exclude_genres": exclude_genres,
        "clean_only": clean_only,
        "deep_cuts": deep_cuts,
        "playlist_size": playlist_size,
        "local_only": local_only,
        "known_tracks_ratio": known_tracks_ratio,
        "min_bpm": min_bpm,
        "max_bpm": max_bpm,
        "target_energy": target_energy,
        "target_valence": target_valence,
        "recommended_tracks": [],
        "playlist_name": _build_aesthetic_title(text),
        "playlist_description": f"Vector & acoustic discovery curated for: {text[:180]}",
    }


def sanitize_constraints(parsed, user_prompt, past_queries=None, avoid_artists=None, avoid_genres=None):
    """Validate and clamp LLM-parsed constraints, filling gaps with deterministic heuristics."""
    heuristic = parse_prompt_heuristic(
        user_prompt,
        past_queries=past_queries,
        avoid_artists=avoid_artists,
        avoid_genres=avoid_genres,
    )
    raw = parsed if isinstance(parsed, dict) else {}
    past_set = {str(q).strip().lower() for q in (past_queries or []) if q}

    raw_excludes = raw.get("exclude_artists") or []
    exclude_artists = []
    seen_ex_art = set()
    for src_item in (list(raw_excludes) if isinstance(raw_excludes, list) else []) + heuristic["exclude_artists"]:
        cleaned = _clean_artist_phrase(src_item) if isinstance(src_item, str) else None
        if cleaned and cleaned.lower() not in seen_ex_art:
            seen_ex_art.add(cleaned.lower())
            exclude_artists.append(cleaned)

    raw_ex_genres = raw.get("exclude_genres") or []
    exclude_genres = []
    seen_ex_gen = set()
    for src_item in (list(raw_ex_genres) if isinstance(raw_ex_genres, list) else []) + heuristic["exclude_genres"]:
        cleaned = str(src_item or "").strip()
        if cleaned and cleaned.lower() not in seen_ex_gen:
            seen_ex_gen.add(cleaned.lower())
            exclude_genres.append(cleaned)

    clean_queries = []
    seen_q = set()
    for q in raw.get("search_queries") or []:
        if not isinstance(q, str):
            continue
        q_clean = q.strip()
        q_low = q_clean.lower()
        if not q_clean or q_low in _GENERIC_QUERIES or q_low in seen_q or q_low in past_set:
            continue
        if _query_matches_exclusions(q_clean, exclude_artists, exclude_genres):
            continue
        seen_q.add(q_low)
        clean_queries.append(q_clean)

    if not clean_queries:
        clean_queries = heuristic["search_queries"]

    # Preserve heuristic explicit size/ratio/local_only when prompt clearly stated them
    prompt_low = str(user_prompt or "").lower()
    has_explicit_num = bool(re.search(r"\b\d+\b", prompt_low))
    playlist_size = (
        heuristic["playlist_size"]
        if has_explicit_num and heuristic["playlist_size"] != 30
        else _clamp_int(raw.get("playlist_size"), 5, 100, heuristic["playlist_size"])
    )

    local_only = raw.get("local_only")
    if not isinstance(local_only, bool):
        local_only = heuristic["local_only"]
    if heuristic["local_only"]:
        local_only = True

    clean_only = bool(raw.get("clean_only")) or heuristic["clean_only"]
    deep_cuts = bool(raw.get("deep_cuts")) or heuristic["deep_cuts"]

    known_ratio = _clamp_float(raw.get("known_tracks_ratio"), 0.0, 1.0, heuristic["known_tracks_ratio"])
    if local_only:
        known_ratio = 1.0
    elif re.search(r"(\d+)\s*(?:songs?|tracks?)?.*?(?:library|known).*?(\d+)\s*(?:brand\s+)?new", prompt_low):
        known_ratio = heuristic["known_tracks_ratio"]

    min_bpm = _clamp_int(raw.get("min_bpm"), 40, 240, None)
    if min_bpm is None:
        min_bpm = heuristic["min_bpm"]
    max_bpm = _clamp_int(raw.get("max_bpm"), 40, 240, None)
    if max_bpm is None:
        max_bpm = heuristic["max_bpm"]
    if min_bpm is not None and max_bpm is not None and max_bpm < min_bpm:
        max_bpm = None

    target_energy = _clamp_float(raw.get("target_energy"), 0.0, 1.0, heuristic["target_energy"])
    target_valence = _clamp_float(raw.get("target_valence"), 0.0, 1.0, heuristic["target_valence"])

    raw_target_artist = raw.get("target_artist")
    target_artist = _clean_artist_phrase(raw_target_artist) if isinstance(raw_target_artist, str) else None
    if target_artist is None:
        target_artist = heuristic["target_artist"]
    if target_artist and target_artist.lower() in seen_ex_art:
        target_artist = None

    pl_name = raw.get("playlist_name")
    if not isinstance(pl_name, str) or not pl_name.strip():
        pl_name = heuristic["playlist_name"]
    else:
        pl_name = pl_name.strip()[:100]

    pl_desc = raw.get("playlist_description")
    if not isinstance(pl_desc, str) or not pl_desc.strip():
        pl_desc = heuristic["playlist_description"]
    else:
        pl_desc = pl_desc.strip()[:300]

    raw_recs = raw.get("recommended_tracks") or []
    clean_recs = []
    seen_rec = set()
    if isinstance(raw_recs, list):
        for item in raw_recs:
            r_name = ""
            r_art = ""
            if isinstance(item, dict):
                r_name = str(item.get("name") or item.get("track") or "").strip()
                r_art = str(item.get("artist") or item.get("artists") or "").strip()
            elif isinstance(item, str) and " - " in item:
                parts = item.split(" - ", 1)
                r_name = parts[0].strip()
                r_art = parts[1].strip()
            if not r_name or not r_art:
                continue
            if matches_avoided_artist(r_art, exclude_artists):
                continue
            rec_key = (r_name.lower(), r_art.lower())
            if rec_key in seen_rec:
                continue
            seen_rec.add(rec_key)
            clean_recs.append({"name": r_name[:120], "artist": r_art[:120]})
            if len(clean_recs) >= 12:
                break

    return {
        "search_queries": clean_queries[:6],
        "target_artist": target_artist,
        "exclude_artists": exclude_artists,
        "exclude_genres": exclude_genres,
        "clean_only": clean_only,
        "deep_cuts": deep_cuts,
        "playlist_size": playlist_size,
        "local_only": local_only,
        "known_tracks_ratio": known_ratio,
        "min_bpm": min_bpm,
        "max_bpm": max_bpm,
        "target_energy": target_energy,
        "target_valence": target_valence,
        "recommended_tracks": clean_recs,
        "playlist_name": pl_name,
        "playlist_description": pl_desc,
    }


def _regex_partial_recovery(raw_output):
    parsed = {}
    sq = re.search(r'"search_queries"\s*:\s*\[(.*?)\]', raw_output, re.DOTALL)
    if sq:
        parsed["search_queries"] = re.findall(r'"([^"]*)"', sq.group(1))
    for int_field in ("playlist_size", "min_bpm", "max_bpm"):
        m = re.search(rf'"{int_field}"\s*:\s*(\d+)', raw_output)
        if m:
            parsed[int_field] = int(m.group(1))
    for float_field in ("known_tracks_ratio", "target_energy", "target_valence"):
        m = re.search(rf'"{float_field}"\s*:\s*([\d.]+)', raw_output)
        if m:
            parsed[float_field] = float(m.group(1))
    lo = re.search(r'"local_only"\s*:\s*(true|false)', raw_output, re.IGNORECASE)
    if lo:
        parsed["local_only"] = lo.group(1).lower() == "true"
    ta = re.search(r'"target_artist"\s*:\s*"([^"]+)"', raw_output)
    if ta:
        parsed["target_artist"] = ta.group(1)
    return parsed


def parse_prompt(pipe, user_prompt, past_queries=None, avoid_artists=None, avoid_genres=None, taste_context=None):
    av_artists = normalize_avoid_list(avoid_artists)
    av_genres = normalize_avoid_list(avoid_genres)
    if pipe is None:
        return parse_prompt_heuristic(
            user_prompt,
            past_queries=past_queries,
            avoid_artists=av_artists,
            avoid_genres=av_genres,
        )

    is_smart_client = isinstance(pipe, LocalLLMClient)

    if is_smart_client:
        taste_block = f"\n\nLISTENER TASTE DNA (use this to tailor artist picks & search flavor):\n{taste_context}" if taste_context else ""
        avoid_block = ""
        if av_artists or av_genres:
            avoid_block += "\n\nCRITICAL SESSION AVOIDANCE RULES:"
            if av_artists:
                avoid_block += f"\n- NEVER include or search for these avoided artists: {av_artists}."
            if av_genres:
                avoid_block += f"\n- NEVER include or search for these avoided genres: {av_genres}."
        past_block = ""
        if past_queries:
            past_block = (
                f"\n\nCRITICAL: Do NOT output any of these search_queries again: {past_queries}. "
                "Generate fresher, deeper queries!"
            )

        system_msg = (
            "You are an expert Music Taste AI Curator. Translate the listener's prompt into structured discovery "
            "search queries and real song recommendations aligned with their Taste DNA.\n"
            "Rules:\n"
            "1. Never output generic words like 'feel good', 'songs', 'playlist', or 'music' in search_queries.\n"
            "2. Put 3-5 concrete search queries in search_queries (use genre:<g>, artist:<a>, year:<y1-y2>, or descriptive subgenre terms).\n"
            "3. Set target_artist ONLY if user explicitly requests ONLY/just songs by that artist (NOT 'vibes' or 'like <artist>').\n"
            "4. Suggest 6-10 real, fitting tracks in recommended_tracks ({\"name\", \"artist\"}) — blend deep cuts and adjacent artists matching the Taste DNA."
            f"{taste_block}{avoid_block}{past_block}\n\nOutput ONLY strict JSON matching the schema."
        )
        user_msg = f"User request: {user_prompt}"
        print(f"      -> Asking {pipe.label} ({pipe.device}) to curate & parse intent...", flush=True)
        try:
            raw_output = pipe.generate_json(
                system_msg,
                user_msg,
                json_schema=_PARSE_PROMPT_SCHEMA,
                max_new_tokens=480,
                temperature=0.15,
            )
        except Exception as e:
            print(f"      [!] {pipe.label} generation failed ({e}); using heuristic parser.")
            return parse_prompt_heuristic(
                user_prompt,
                past_queries=past_queries,
                avoid_artists=av_artists,
                avoid_genres=av_genres,
            )
    else:
        system_msg = """You are a Music AI Configuration parser. Read the user's prompt and extract exact Spotify search constraints.
You MUST output ONLY a valid JSON object, nothing else. Do NOT use markdown code blocks.

CRITICAL SEARCH RULES:
1. Do NOT output generic prompt words like "feel good", "good vibes", "songs", "playlist", or "music" in search_queries.
2. Translate abstract vibes into concrete Spotify search operators:
   - "genre:<genre_name>" (e.g., "genre:indie pop", "genre:synthwave", "genre:funk", "genre:pop", "genre:rock", "genre:dance pop")
   - "artist:<artist_name>" (e.g., "artist:Gorillaz", "artist:OneRepublic", "artist:Dua Lipa")
   - "year:<year_range>" (e.g., "year:2018-2024")
3. Set "target_artist" ONLY if the user explicitly asks for songs ONLY or specifically by that artist (NOT for "<artist> vibes" or "like <artist>").
4. Output 3 to 5 diverse, high-quality search queries matching the requested mood.

Schema:
{
  "search_queries": ["genre:pop", "genre:indie pop", "artist:OneRepublic", "year:2020-2024"],
  "target_artist": null,
  "exclude_artists": [],
  "exclude_genres": [],
  "clean_only": false,
  "deep_cuts": false,
  "playlist_size": 30,
  "local_only": false,
  "known_tracks_ratio": 0.3,
  "min_bpm": null,
  "max_bpm": null,
  "target_energy": null,
  "target_valence": null,
  "recommended_tracks": [{"name": "Song Title", "artist": "Artist Name"}],
  "playlist_name": "Creative 3-5 word playlist name",
  "playlist_description": "Short 1-2 sentence aesthetic description of the vibe"
}"""

        if taste_context:
            system_msg += f"\n\nLISTENER TASTE DNA:\n{taste_context}"
        if av_artists or av_genres:
            system_msg += "\n\nCRITICAL SESSION AVOIDANCE RULES:"
            if av_artists:
                system_msg += f"\n- NEVER include or search for these avoided artists: {av_artists}."
            if av_genres:
                system_msg += f"\n- NEVER include or search for these avoided genres: {av_genres}."
        if past_queries:
            system_msg += (
                f"\n\nCRITICAL: Do NOT output any of these search_queries again: {past_queries}. "
                "Generate new, different, or more niche queries to find the remaining songs!"
            )

        prompt = f"<s>[INST] {system_msg}\n\nUser request: {user_prompt}\n\nReturn ONLY raw JSON. [/INST]"
        gen_cfg = _make_generation_config(max_new_tokens=360, temperature=0.1)
        print("      -> Asking Local LLM to parse intent...")
        try:
            raw_output = pipe.generate(prompt, gen_cfg) if gen_cfg is not None else pipe.generate(prompt)
        except Exception as e:
            print(f"      [!] Local LLM generation failed ({e}); using heuristic parser.")
            return parse_prompt_heuristic(
                user_prompt,
                past_queries=past_queries,
                avoid_artists=av_artists,
                avoid_genres=av_genres,
            )

    parsed = extract_json_object(raw_output)
    if parsed is None:
        print("      [!] Failed to parse LLM JSON cleanly. Applying regex + heuristic recovery...")
        parsed = _regex_partial_recovery(raw_output)

    return sanitize_constraints(
        parsed,
        user_prompt,
        past_queries=past_queries,
        avoid_artists=av_artists,
        avoid_genres=av_genres,
    )


def _format_candidate_for_curator(idx, t):
    name = t.get("name", "Unknown")
    artist = t.get("artist", "Unknown")
    src = "Lib" if t.get("source") == "Local Library" else "New"
    bits = [f"Src:{src}"]
    if t.get("bpm") is not None:
        bits.append(f"{t['bpm']}BPM")
    if t.get("energy") is not None:
        bits.append(f"E:{t['energy']}")
    if t.get("valence") is not None:
        bits.append(f"V:{t['valence']}")
    genres = str(t.get("genres") or "").strip()
    if genres:
        bits.append(f"G:{genres[:40]}")
    return f"[{idx}] {name} - {artist} ({' | '.join(bits)})"


def vibe_check_playlist(
    pipe, user_prompt, candidates, target_size, avoid_artists=None, avoid_genres=None, taste_context=None
):
    if not candidates:
        return {
            "selected_indices": [],
            "playlist_name": _build_aesthetic_title(user_prompt),
            "playlist_description": f"AI Generated playlist for: {user_prompt}",
        }

    is_smart_client = isinstance(pipe, LocalLLMClient)
    cap = 45 if is_smart_client else 80
    capped_candidates = list(candidates)[:cap]
    default_indices = list(range(min(target_size, len(capped_candidates))))

    if pipe is None:
        return {
            "selected_indices": default_indices,
            "playlist_name": _build_aesthetic_title(user_prompt),
            "playlist_description": f"Vector & acoustic discovery for: {user_prompt}",
        }

    if is_smart_client:
        track_list_str = "\n".join(_format_candidate_for_curator(i, t) for i, t in enumerate(capped_candidates))
    else:
        track_list_str = "\n".join(
            f"[{i}] {t.get('name', 'Unknown')} - {t.get('artist', 'Unknown')}"
            for i, t in enumerate(capped_candidates)
        )

    avoid_note = ""
    av_art = normalize_avoid_list(avoid_artists)
    av_gen = normalize_avoid_list(avoid_genres)
    if av_art or av_gen:
        avoid_note = f"\nAvoid artists: {av_art or 'None'} | Avoid genres: {av_gen or 'None'}."
    taste_note = f"\nListener Taste DNA:\n{taste_context}" if (is_smart_client and taste_context) else ""

    system_msg = f"""You are an elite DJ AI curator. A vector+acoustic engine shortlisted {len(capped_candidates)} candidate songs for: "{user_prompt}".
Select the best {min(target_size, len(capped_candidates))} tracks balancing mood cohesion, BPM/energy flow, and artist variety.{avoid_note}{taste_note}

Candidates:
{track_list_str}

Output ONLY a valid JSON object (no markdown):
{{
  "selected_indices": [array of integer indices from 0 to {len(capped_candidates) - 1}],
  "playlist_name": "Creative 3-5 word aesthetic playlist title",
  "playlist_description": "Short 1-2 sentence aesthetic description of the vibe."
}}"""

    max_tokens = max(450, min(target_size, len(capped_candidates)) * 10 + 150)

    try:
        if is_smart_client:
            output = pipe.generate_json(
                system_msg,
                f"Curate the final {min(target_size, len(capped_candidates))} tracks for '{user_prompt}'.",
                json_schema=_VIBE_CHECK_SCHEMA,
                max_new_tokens=max_tokens,
                temperature=0.2,
            )
        else:
            prompt = f"<s>[INST] {system_msg}\n\nReturn ONLY raw JSON. [/INST]"
            gen_cfg = _make_generation_config(max_new_tokens=max_tokens, temperature=0.2)
            output = pipe.generate(prompt, gen_cfg) if gen_cfg is not None else pipe.generate(prompt)
        parsed = extract_json_object(output)
        if not isinstance(parsed, dict):
            raise ValueError("No valid JSON object returned by Local LLM.")

        valid_indices = []
        seen_idx = set()
        for idx in parsed.get("selected_indices") or []:
            if isinstance(idx, int) and not isinstance(idx, bool) and 0 <= idx < len(capped_candidates) and idx not in seen_idx:
                seen_idx.add(idx)
                valid_indices.append(idx)

        if not valid_indices:
            valid_indices = default_indices

        pl_name = parsed.get("playlist_name")
        if not isinstance(pl_name, str) or not pl_name.strip():
            pl_name = _build_aesthetic_title(user_prompt)

        pl_desc = parsed.get("playlist_description")
        if not isinstance(pl_desc, str) or not pl_desc.strip():
            pl_desc = f"AI Generated playlist for: {user_prompt}"

        return {
            "selected_indices": valid_indices,
            "playlist_name": pl_name.strip()[:100],
            "playlist_description": pl_desc.strip()[:300],
        }
    except Exception as e:
        print(f"      [Vibe Check Failed] Fallback to raw math selection. Error: {e}")
        return {
            "selected_indices": default_indices,
            "playlist_name": _build_aesthetic_title(user_prompt),
            "playlist_description": f"AI Generated playlist for: {user_prompt}",
        }


def parse_refinement_heuristic(instruction, current_tracks):
    """Parse conversational playlist refinement commands into structured edit operations."""
    text = str(instruction or "").strip()
    lower = text.lower()
    total = len(current_tracks or [])

    drop_indices = []
    seen_idx = set()

    # Match 1-based track index drops/swaps/removals, including ranges like "4-6"
    idx_clause_re = re.compile(
        r"(?i)\b(?:drop|replace|remove|delete|swap)\s+(?:tracks?|songs?)?\s*(?:#?\d+(?:\s*-\s*#?\d+)?(?:\s*(?:,|and)\s*#?\d+(?:\s*-\s*#?\d+)?)*)"
    )
    for m in idx_clause_re.finditer(text):
        frag = m.group(0)
        # Skip "remove 5 songs" (delta count, not track numbers)
        if re.search(r"(?i)\bremove\s+\d+\s+(?:songs?|tracks?)\b", frag):
            continue
        for token in re.findall(r"(\d+)\s*-\s*(\d+)|(\d+)", frag):
            if token[0] and token[1]:
                a, b = int(token[0]), int(token[1])
                for one_based in range(min(a, b), max(a, b) + 1):
                    z = one_based - 1
                    if 0 <= z < total and z not in seen_idx:
                        seen_idx.add(z)
                        drop_indices.append(z)
            elif token[2]:
                z = int(token[2]) - 1
                if 0 <= z < total and z not in seen_idx:
                    seen_idx.add(z)
                    drop_indices.append(z)

    remove_artists = []
    seen_art = set()
    # Match artist names present in current_tracks when mentioned after cut/remove/drop/no/without
    for t in current_tracks or []:
        for art in str(t.get("artist") or "").split(","):
            ac = art.strip()
            if len(ac) >= 2 and ac.lower() not in seen_art:
                if re.search(rf"(?i)\b(?:cut|remove|drop|no|without|exclude)\b[^,.]*\b{re.escape(ac)}\b", text):
                    seen_art.add(ac.lower())
                    remove_artists.append(ac)

    filter_condition = None
    if any(p in lower for p in ("drop slow", "remove slow", "replace slow", "too slow", "no slow")):
        filter_condition = "drop_slow"
    elif any(p in lower for p in ("drop fast", "remove fast", "too fast")):
        filter_condition = "drop_fast"
    elif any(p in lower for p in ("low energy", "too chill", "boring", "higher energy")):
        filter_condition = "drop_low_energy"
    elif any(p in lower for p in ("too intense", "high energy", "calmer")):
        filter_condition = "drop_high_energy"
    elif any(p in lower for p in ("no explicit", "drop explicit", "clean only")):
        filter_condition = "drop_explicit"

    delta_tracks = 0
    target_size = None
    add_m = re.search(r"\badd\s+(\d{1,2})\s*(?:more\s*)?(?:songs?|tracks?)?\b", lower)
    rem_m = re.search(r"\bremove\s+(\d{1,2})\s+(?:songs?|tracks?)\b", lower)
    trim_m = re.search(r"\b(?:trim|reduce|keep)\s+(?:to\s+)?(\d{1,3})\b", lower)
    if add_m:
        delta_tracks = _clamp_int(add_m.group(1), 1, 50, 0)
    elif rem_m:
        delta_tracks = -_clamp_int(rem_m.group(1), 1, 50, 0)
    elif trim_m:
        target_size = _clamp_int(trim_m.group(1), 1, 100, None)

    flow_mode = None
    if any(w in lower for w in ("harmonic", "camelot")):
        flow_mode = "harmonic"
    elif any(w in lower for w in ("warmup", "warm-up", "workout arc", "peak")):
        flow_mode = "warmup_peak_cooldown"
    elif any(w in lower for w in ("energy ramp", "ramp up", "build energy")):
        flow_mode = "energy_ramp"
    elif any(w in lower for w in ("chill descent", "wind down", "cooldown", "ramp down")):
        flow_mode = "chill_descent"
    elif "shuffle" in lower:
        flow_mode = "shuffle"

    direction_prompt = None
    with_m = re.search(r"(?i)\b(?:with|more|towards?)\s+([A-Za-z0-9 ,&'\-]+)$", text)
    if with_m:
        direction_prompt = with_m.group(1).strip()
    elif any(w in lower for w in ("faster", "higher energy", "more upbeat", "sadder", "chiller", "heavier")):
        direction_prompt = text

    return {
        "drop_indices": sorted(drop_indices),
        "remove_artists": remove_artists,
        "filter_condition": filter_condition,
        "delta_tracks": delta_tracks,
        "target_size": target_size,
        "flow_mode": flow_mode,
        "direction_prompt": direction_prompt,
    }


def parse_refinement_instruction(pipe, instruction, current_tracks):
    """Use Local LLM (when warm) + heuristic fallback to parse playlist refinement requests."""
    heuristic = parse_refinement_heuristic(instruction, current_tracks)
    if pipe is None:
        return heuristic

    track_summary = "\n".join(
        f"[{i+1}] {t.get('name', '')} - {t.get('artist', '')} (BPM:{t.get('bpm', '-')}, E:{t.get('energy', '-')})"
        for i, t in enumerate((current_tracks or [])[:60])
    )
    sys_msg = f"""You are a Playlist Editor AI. Convert the user's refinement instruction into JSON actions.
Current tracks (1-based numbering):
{track_summary}

Output ONLY valid JSON:
{{
  "drop_track_numbers": [1-based integers to drop/replace],
  "remove_artists": [],
  "filter_condition": null or "drop_slow"|"drop_fast"|"drop_low_energy"|"drop_high_energy"|"drop_explicit",
  "delta_tracks": 0,
  "target_size": null,
  "flow_mode": null or "mmr"|"energy_ramp"|"chill_descent"|"warmup_peak_cooldown"|"harmonic"|"shuffle",
  "direction_prompt": null or short mood/genre string for replacement tracks
}}"""
    try:
        if isinstance(pipe, LocalLLMClient):
            out = pipe.generate_json(
                sys_msg,
                f"Instruction: {instruction}",
                json_schema=_REFINEMENT_SCHEMA,
                max_new_tokens=260,
                temperature=0.1,
            )
        else:
            prompt = f"<s>[INST] {sys_msg}\n\nInstruction: {instruction}\n\nReturn ONLY raw JSON. [/INST]"
            gen_cfg = _make_generation_config(max_new_tokens=220, temperature=0.1)
            out = pipe.generate(prompt, gen_cfg) if gen_cfg is not None else pipe.generate(prompt)
        parsed = extract_json_object(out)
        if not isinstance(parsed, dict):
            return heuristic
        drops = set(heuristic["drop_indices"])
        for num in parsed.get("drop_track_numbers") or []:
            if isinstance(num, int) and not isinstance(num, bool):
                z = num - 1
                if 0 <= z < len(current_tracks or []):
                    drops.add(z)
        rem_art = list(dict.fromkeys(heuristic["remove_artists"] + [str(a).strip() for a in (parsed.get("remove_artists") or []) if str(a).strip()]))
        return {
            "drop_indices": sorted(drops),
            "remove_artists": rem_art,
            "filter_condition": parsed.get("filter_condition") or heuristic["filter_condition"],
            "delta_tracks": _clamp_int(parsed.get("delta_tracks"), -50, 50, heuristic["delta_tracks"]),
            "target_size": _clamp_int(parsed.get("target_size"), 1, 100, heuristic["target_size"]),
            "flow_mode": parsed.get("flow_mode") or heuristic["flow_mode"],
            "direction_prompt": parsed.get("direction_prompt") or heuristic["direction_prompt"],
        }
    except Exception:
        return heuristic


if __name__ == "__main__":
    print(json.dumps(parse_prompt_heuristic("Make a 50 song playlist with sad linkin park vibes. 10 songs from my library, 40 brand new."), indent=2))
