"""
Pluggable Local & Cloud LLM backends for Stage 3 Agentic Playlist Generation.

Supports:
  - Local: OpenVINO GenAI (XGrammar JSON Schema), Ollama, OpenAI-compatible local endpoints
  - Cloud (optional BYO API key): Google Gemini, OpenAI, Anthropic Claude, OpenRouter
"""

import json
import os
import re
import sys
import time
import urllib.parse

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import config
from utils import get_llm_api_key

ALL_LLM_BACKENDS = ("openvino", "ollama", "openai_local", "gemini", "openai", "anthropic", "openrouter")
CLOUD_LLM_BACKENDS = ("gemini", "openai", "anthropic", "openrouter")

ALLOWED_CLOUD_ENDPOINTS = {
    "gemini": "https://generativelanguage.googleapis.com",
    "openai": "https://api.openai.com",
    "anthropic": "https://api.anthropic.com",
    "openrouter": "https://openrouter.ai",
}

ALLOWED_CLOUD_HOSTS = {
    "gemini": "generativelanguage.googleapis.com",
    "openai": "api.openai.com",
    "anthropic": "api.anthropic.com",
    "openrouter": "openrouter.ai",
}


def _redact_secret(text, secret=None):
    msg = str(text or "")
    if secret and len(secret) >= 6:
        msg = msg.replace(secret, "[REDACTED]")
    return msg


def _assert_cloud_url(backend, url):
    parsed = urllib.parse.urlparse(url)
    expected_host = ALLOWED_CLOUD_HOSTS.get(backend)
    if parsed.scheme != "https" or parsed.hostname != expected_host:
        raise ValueError(f"Refusing non-allowlisted endpoint for cloud backend '{backend}'.")
    return url


def infer_capability_tier(backend, model_id_or_key=""):
    """Infer reasoning tier ('compact' | 'standard' | 'frontier') for bounded agent loops."""
    b = str(backend or "openvino").strip().lower()
    m = str(model_id_or_key or "").strip().lower()

    if b in CLOUD_LLM_BACKENDS:
        providers = getattr(config, "CLOUD_LLM_PROVIDERS", {}) or {}
        p_cfg = providers.get(b) or {}
        for item in p_cfg.get("models") or []:
            if str(item.get("id") or "").strip().lower() == m:
                tier = str(item.get("tier") or "standard").strip().lower()
                if tier in ("compact", "standard", "frontier"):
                    return tier
        if any(tok in m for tok in ("sonnet", "opus", "pro", "gpt-4.1", "gpt-5", "o3", "o4", "70b", "72b", "405b")):
            if "mini" not in m and "lite" not in m:
                return "frontier"
        if any(tok in m for tok in ("flash-lite", "nano")):
            return "compact"
        return "standard"

    if b == "openvino":
        for item in getattr(config, "LLM_MODEL_CATALOG", []) or []:
            if str(item.get("key") or "").strip().lower() == m:
                t = int(item.get("tier") or 2)
                return "compact" if t <= 1 else "standard"
        if any(tok in m for tok in ("1.7b", "1.5b", "2b", "3b")):
            return "compact"
        return "standard"

    # Ollama or OpenAI-compatible local server
    match = re.search(r"(\d+(?:\.\d+)?)\s*b\b", m)
    if match:
        try:
            params_b = float(match.group(1))
            if params_b <= 4.5:
                return "compact"
            if params_b >= 24.0:
                return "frontier"
            return "standard"
        except ValueError:
            pass
    if any(tok in m for tok in ("mini", "small", "tiny", "1.7b", "3b")):
        return "compact"
    return "standard"


def default_critic_budget(tier):
    """Default max critic iterations allowed for a given model capability tier."""
    return {"compact": 1, "standard": 2, "frontier": 3}.get(str(tier or "standard").lower(), 2)


def resolve_cloud_model(backend, cloud_model=None):
    """Resolve explicit or default cloud model ID for a cloud provider."""
    b = str(backend or "").strip().lower()
    explicit = str(cloud_model or "").strip()
    if explicit:
        return explicit
    providers = getattr(config, "CLOUD_LLM_PROVIDERS", {}) or {}
    return str((providers.get(b) or {}).get("default_model") or "").strip()


def _to_gemini_schema(schema):
    """Convert standard JSON Schema subset into Gemini responseSchema format."""
    if not isinstance(schema, dict):
        return None
    type_map = {
        "object": "OBJECT",
        "array": "ARRAY",
        "string": "STRING",
        "integer": "INTEGER",
        "number": "NUMBER",
        "boolean": "BOOLEAN",
    }
    out = {}
    raw_type = schema.get("type")
    if isinstance(raw_type, str) and raw_type.lower() in type_map:
        out["type"] = type_map[raw_type.lower()]
    if isinstance(schema.get("description"), str):
        out["description"] = schema["description"]
    if isinstance(schema.get("enum"), list):
        out["enum"] = list(schema["enum"])
    if isinstance(schema.get("properties"), dict):
        props = {}
        for k, v in schema["properties"].items():
            conv = _to_gemini_schema(v)
            if conv is not None:
                props[k] = conv
        out["properties"] = props
    if isinstance(schema.get("items"), dict):
        conv_items = _to_gemini_schema(schema["items"])
        if conv_items is not None:
            out["items"] = conv_items
    if isinstance(schema.get("required"), list):
        out["required"] = [str(r) for r in schema["required"]]
    return out


class UnifiedLLMClient:
    """Unified wrapper across OpenVINO, Ollama, local OpenAI APIs, and official cloud LLM APIs."""

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
        api_key=None,
        cloud_model=None,
        capability_tier=None,
    ):
        self.backend = str(backend or "openvino").strip().lower()
        self.model_key = model_key
        self.label = label
        self.device = device
        self.pipe = pipe
        self.base_url = str(base_url or getattr(config, "DEFAULT_OLLAMA_URL", "http://127.0.0.1:11434")).rstrip("/")
        self.ollama_model = str(ollama_model or getattr(config, "DEFAULT_OLLAMA_MODEL", "qwen3:8b"))
        self.is_qwen3 = bool(is_qwen3)
        self.cloud_model = resolve_cloud_model(self.backend, cloud_model) if self.backend in CLOUD_LLM_BACKENDS else str(
            cloud_model or ""
        )
        self.api_key = str(api_key or "").strip() if api_key is not None else ""
        if self.backend in CLOUD_LLM_BACKENDS and not self.api_key:
            self.api_key = get_llm_api_key(self.backend)
        resolved_id = self.cloud_model or (
            self.ollama_model if self.backend in ("ollama", "openai_local") else self.model_key
        )
        self.capability_tier = capability_tier or infer_capability_tier(self.backend, resolved_id)

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

    def _require_cloud_key(self):
        if not self.api_key:
            providers = getattr(config, "CLOUD_LLM_PROVIDERS", {}) or {}
            env_hint = (providers.get(self.backend) or {}).get("env_var") or "API key"
            raise RuntimeError(f"Missing API key for '{self.backend}'. Configure it in Preferences or {env_hint}.")

    def _generate_gemini_json(self, system_msg, user_msg, json_schema=None, max_new_tokens=380, temperature=0.1):
        import requests

        self._require_cloud_key()
        model_id = urllib.parse.quote(self.cloud_model or "gemini-2.5-flash", safe="-._")
        endpoint = _assert_cloud_url(
            "gemini",
            f"{ALLOWED_CLOUD_ENDPOINTS['gemini']}/v1beta/models/{model_id}:generateContent",
        )
        gen_cfg = {
            "temperature": float(temperature),
            "maxOutputTokens": int(max_new_tokens),
            "responseMimeType": "application/json",
        }
        g_schema = _to_gemini_schema(json_schema) if isinstance(json_schema, dict) else None
        if g_schema:
            gen_cfg["responseSchema"] = g_schema

        payload = {
            "systemInstruction": {"parts": [{"text": str(system_msg or "")}]},
            "contents": [{"role": "user", "parts": [{"text": str(user_msg or "")}]}],
            "generationConfig": gen_cfg,
        }
        headers = {
            "Content-Type": "application/json",
            "x-goog-api-key": self.api_key,
        }
        try:
            resp = requests.post(endpoint, headers=headers, json=payload, timeout=90)
            if resp.status_code == 400 and "responseSchema" in gen_cfg:
                gen_cfg.pop("responseSchema", None)
                resp = requests.post(endpoint, headers=headers, json=payload, timeout=90)
        except Exception as exc:
            raise RuntimeError(_redact_secret(exc, self.api_key)) from exc

        if resp.status_code != 200:
            raise RuntimeError(f"Gemini HTTP {resp.status_code}: {_redact_secret(resp.text[:200], self.api_key)}")
        data = resp.json()
        candidates = data.get("candidates") or []
        if not candidates:
            return ""
        parts = ((candidates[0].get("content") or {}).get("parts")) or []
        texts = [str(p.get("text") or "") for p in parts if isinstance(p, dict) and p.get("text")]
        return "\n".join(texts)

    def _generate_openai_compat_cloud_json(
        self, backend, system_msg, user_msg, max_new_tokens=380, temperature=0.1
    ):
        import requests

        self._require_cloud_key()
        base = ALLOWED_CLOUD_ENDPOINTS[backend]
        path = "/api/v1/chat/completions" if backend == "openrouter" else "/v1/chat/completions"
        endpoint = _assert_cloud_url(backend, f"{base}{path}")
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        if backend == "openrouter":
            headers["HTTP-Referer"] = "http://127.0.0.1:8765"
            headers["X-Title"] = "MusicTasteAnalyzer"

        payload = {
            "model": self.cloud_model,
            "messages": [
                {"role": "system", "content": str(system_msg or "")},
                {"role": "user", "content": str(user_msg or "")},
            ],
            "temperature": float(temperature),
            "max_tokens": int(max_new_tokens),
            "response_format": {"type": "json_object"},
        }
        try:
            resp = requests.post(endpoint, headers=headers, json=payload, timeout=90)
            if resp.status_code == 400 and "response_format" in payload:
                payload.pop("response_format", None)
                resp = requests.post(endpoint, headers=headers, json=payload, timeout=90)
        except Exception as exc:
            raise RuntimeError(_redact_secret(exc, self.api_key)) from exc

        if resp.status_code != 200:
            label = "OpenRouter" if backend == "openrouter" else "OpenAI"
            raise RuntimeError(f"{label} HTTP {resp.status_code}: {_redact_secret(resp.text[:200], self.api_key)}")
        data = resp.json()
        choices = data.get("choices") or []
        if not choices:
            return ""
        return str(((choices[0].get("message") or {}).get("content")) or "")

    def _generate_anthropic_json(self, system_msg, user_msg, json_schema=None, max_new_tokens=380, temperature=0.1):
        import requests

        self._require_cloud_key()
        endpoint = _assert_cloud_url("anthropic", f"{ALLOWED_CLOUD_ENDPOINTS['anthropic']}/v1/messages")
        headers = {
            "Content-Type": "application/json",
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
        }
        payload = {
            "model": self.cloud_model or "claude-3-5-haiku-latest",
            "max_tokens": int(max_new_tokens),
            "temperature": float(temperature),
            "system": str(system_msg or ""),
            "messages": [{"role": "user", "content": str(user_msg or "")}],
        }
        used_tool_schema = False
        if isinstance(json_schema, dict) and json_schema.get("properties"):
            payload["tools"] = [
                {
                    "name": "emit_structured_json",
                    "description": "Return the structured JSON result strictly matching the schema.",
                    "input_schema": json_schema,
                }
            ]
            payload["tool_choice"] = {"type": "tool", "name": "emit_structured_json"}
            used_tool_schema = True

        try:
            resp = requests.post(endpoint, headers=headers, json=payload, timeout=90)
            if resp.status_code == 400 and used_tool_schema:
                payload.pop("tools", None)
                payload.pop("tool_choice", None)
                used_tool_schema = False
                resp = requests.post(endpoint, headers=headers, json=payload, timeout=90)
        except Exception as exc:
            raise RuntimeError(_redact_secret(exc, self.api_key)) from exc

        if resp.status_code != 200:
            raise RuntimeError(f"Anthropic HTTP {resp.status_code}: {_redact_secret(resp.text[:200], self.api_key)}")
        data = resp.json()
        blocks = data.get("content") or []
        for blk in blocks:
            if isinstance(blk, dict) and blk.get("type") == "tool_use" and isinstance(blk.get("input"), dict):
                return json.dumps(blk["input"], ensure_ascii=False)
        texts = [str(blk.get("text") or "") for blk in blocks if isinstance(blk, dict) and blk.get("type") == "text"]
        return "\n".join(texts)

    def generate_json(self, system_msg, user_msg, json_schema=None, max_new_tokens=380, temperature=0.1):
        if self.backend == "ollama":
            return self._generate_ollama_json(
                system_msg, user_msg, json_schema=json_schema, max_new_tokens=max_new_tokens, temperature=temperature
            )
        if self.backend == "openai_local":
            return self._generate_openai_local_json(
                system_msg, user_msg, max_new_tokens=max_new_tokens, temperature=temperature
            )
        if self.backend == "gemini":
            return self._generate_gemini_json(
                system_msg, user_msg, json_schema=json_schema, max_new_tokens=max_new_tokens, temperature=temperature
            )
        if self.backend in ("openai", "openrouter"):
            return self._generate_openai_compat_cloud_json(
                self.backend, system_msg, user_msg, max_new_tokens=max_new_tokens, temperature=temperature
            )
        if self.backend == "anthropic":
            return self._generate_anthropic_json(
                system_msg, user_msg, json_schema=json_schema, max_new_tokens=max_new_tokens, temperature=temperature
            )
        return self._generate_openvino_json(
            system_msg, user_msg, json_schema=json_schema, max_new_tokens=max_new_tokens, temperature=temperature
        )


def build_cloud_llm_client(backend, cloud_model=None, api_key=None):
    """Build a UnifiedLLMClient configured for one of the official cloud providers."""
    b = str(backend or "").strip().lower()
    if b not in CLOUD_LLM_BACKENDS:
        raise ValueError(f"Unsupported cloud backend: {backend}")
    providers = getattr(config, "CLOUD_LLM_PROVIDERS", {}) or {}
    p_cfg = providers.get(b) or {}
    model_id = resolve_cloud_model(b, cloud_model)
    p_label = p_cfg.get("label") or b.title()
    return UnifiedLLMClient(
        backend=b,
        model_key=model_id,
        label=f"{p_label}: {model_id}",
        device="Cloud API",
        api_key=api_key,
        cloud_model=model_id,
        capability_tier=infer_capability_tier(b, model_id),
    )


def test_llm_connection(
    backend=None,
    cloud_model=None,
    api_key=None,
    ollama_url=None,
    ollama_model=None,
):
    """Run a tiny JSON round-trip ping against the requested LLM backend (without leaking secrets)."""
    b = str(backend or "openvino").strip().lower()
    t0 = time.time()
    ping_schema = {
        "type": "object",
        "properties": {"ok": {"type": "boolean"}},
        "required": ["ok"],
    }
    try:
        if b in CLOUD_LLM_BACKENDS:
            client = build_cloud_llm_client(b, cloud_model=cloud_model, api_key=api_key)
            raw = client.generate_json(
                "Return JSON only.",
                'Reply with {"ok": true}.',
                json_schema=ping_schema,
                max_new_tokens=40,
                temperature=0.0,
            )
            parsed = json.loads(raw) if raw.strip().startswith("{") else {}
            ok = bool(parsed.get("ok", True)) if raw.strip() else False
            return {
                "ok": ok,
                "backend": b,
                "model": client.cloud_model,
                "label": client.label,
                "capability_tier": client.capability_tier,
                "latency_ms": int(round((time.time() - t0) * 1000)),
            }

        if b in ("ollama", "openai_local"):
            from utils import _sanitize_llm_url

            clean_url = _sanitize_llm_url(ollama_url)
            model_name = str(ollama_model or getattr(config, "DEFAULT_OLLAMA_MODEL", "qwen3:8b")).strip()
            client = UnifiedLLMClient(
                backend=b,
                model_key=model_name,
                label=f"{b}: {model_name}",
                device="Local GPU",
                base_url=clean_url,
                ollama_model=model_name,
                is_qwen3="qwen3" in model_name.lower(),
            )
            raw = client.generate_json(
                "Return JSON only.",
                'Reply with {"ok": true}.',
                json_schema=ping_schema,
                max_new_tokens=40,
                temperature=0.0,
            )
            return {
                "ok": bool(raw.strip()),
                "backend": b,
                "model": model_name,
                "label": client.label,
                "capability_tier": client.capability_tier,
                "latency_ms": int(round((time.time() - t0) * 1000)),
            }

        return {
            "ok": True,
            "backend": b,
            "model": "openvino",
            "label": "OpenVINO Local",
            "capability_tier": infer_capability_tier("openvino", "qwen3-4b-ov"),
            "latency_ms": int(round((time.time() - t0) * 1000)),
        }
    except Exception as exc:
        return {
            "ok": False,
            "backend": b,
            "model": resolve_cloud_model(b, cloud_model) if b in CLOUD_LLM_BACKENDS else (ollama_model or ""),
            "error": _redact_secret(str(exc), api_key),
            "latency_ms": int(round((time.time() - t0) * 1000)),
        }
