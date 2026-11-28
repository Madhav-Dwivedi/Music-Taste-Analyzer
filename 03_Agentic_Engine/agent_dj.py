"""
Bounded Agentic DJ loop & stateful multi-turn DJ Chat session.

Features:
  - Capability-aware critic loop (0 fast, 1 compact, 2 standard, 3 frontier)
  - Hard iteration caps and explicit user cancellation support
  - Multi-turn stateful DJ Chat with track locking and targeted remote refills
"""

import datetime
import json
import os
import re
import sys

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from llm_parser import (
    extract_json_object,
    parse_refinement_instruction,
    sanitize_constraints,
)
from llm_providers import default_critic_budget
from utils import get_effective_preferences, get_track_fingerprint, normalize_avoid_list


class GenerationCancelledError(Exception):
    """Raised when an active playlist generation run is cancelled by the user."""


def check_cancelled(cancel_event):
    if cancel_event is not None and cancel_event.is_set():
        raise GenerationCancelledError("Generation cancelled by user.")


def resolve_critic_budget(pipe, fast=False, override_depth=None):
    """Resolve bounded critic pass budget from fast flag, override, preference, or model tier."""
    if fast or pipe is None:
        return 0

    raw_val = override_depth
    if raw_val is None or str(raw_val).strip().lower() in ("", "auto"):
        raw_val = get_effective_preferences().get("critic_depth", "auto")

    s_val = str(raw_val).strip().lower()
    if s_val in ("0", "1", "2", "3"):
        return int(s_val)

    tier = getattr(pipe, "capability_tier", "standard")
    return default_critic_budget(tier)


_AGENT_CRITIC_SCHEMA = {
    "type": "object",
    "properties": {
        "sufficient": {"type": "boolean"},
        "critique": {"type": "string"},
        "new_queries": {"type": "array", "items": {"type": "string"}},
        "seed_tracks": {
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
    },
    "required": ["sufficient", "critique", "new_queries", "seed_tracks"],
}


_CHAT_TURN_SCHEMA = {
    "type": "object",
    "properties": {
        "assistant_reply": {"type": "string"},
        "action": {"type": "string"},
        "drop_track_numbers": {"type": "array", "items": {"type": "integer"}},
        "remove_artists": {"type": "array", "items": {"type": "string"}},
        "filter_condition": {"type": "string"},
        "direction_prompt": {"type": "string"},
        "delta_tracks": {"type": "integer"},
        "target_size": {"type": "integer"},
        "flow_mode": {"type": "string"},
        "discover_queries": {"type": "array", "items": {"type": "string"}},
        "playlist_name": {"type": "string"},
    },
    "required": ["assistant_reply", "action"],
}


def summarize_candidate_pool_for_critic(
    local_candidates, discovered_tracks, constraints, target_local_count, target_size
):
    """Build deterministic diagnostics so the critic only fires when improvement is actionable."""
    target_new = max(0, int(target_size) - int(target_local_count))
    local_list = list(local_candidates or [])
    disc_list = list(discovered_tracks or [])

    min_bpm = constraints.get("min_bpm")
    max_bpm = constraints.get("max_bpm")
    bpm_violations = 0
    prompt_matches = []
    audio_verified = 0
    artists = set()

    for t in local_list + disc_list:
        pa = str(t.get("artist") or "").split(",")[0].strip().lower()
        if pa and pa != "unknown":
            artists.add(pa)
        bpm = t.get("bpm")
        if isinstance(bpm, (int, float)):
            if isinstance(min_bpm, (int, float)) and bpm < min_bpm:
                bpm_violations += 1
            if isinstance(max_bpm, (int, float)) and bpm > max_bpm:
                bpm_violations += 1
        if t.get("audio_verified"):
            audio_verified += 1
        pm = t.get("prompt_match")
        if isinstance(pm, (int, float)):
            prompt_matches.append(float(pm))

    mean_match = round(sum(prompt_matches) / len(prompt_matches), 3) if prompt_matches else None
    underfilled_new = target_new > 0 and len(disc_list) < max(3, int(target_new * 0.7))
    weak_match = mean_match is not None and mean_match < 0.14
    low_diversity = len(artists) < max(3, min(8, target_size // 3))

    sample_titles = [
        f"{t.get('name', 'Unknown')} - {t.get('artist', 'Unknown')} (Match:{t.get('prompt_match', '-')})"
        for t in (disc_list[:6] + local_list[:4])
    ]
    needs_recovery = bool(underfilled_new or weak_match or low_diversity or bpm_violations > max(3, target_size // 4))

    return {
        "local_count": len(local_list),
        "discovered_count": len(disc_list),
        "target_local_count": int(target_local_count),
        "target_new_count": target_new,
        "audio_verified_count": audio_verified,
        "mean_prompt_match": mean_match,
        "unique_artists": len(artists),
        "bpm_violations": bpm_violations,
        "underfilled_new": underfilled_new,
        "weak_match": weak_match,
        "low_diversity": low_diversity,
        "needs_recovery": needs_recovery,
        "sample_titles": sample_titles,
    }


def run_bounded_critic_step(
    pipe, user_prompt, pool_summary, past_queries=None, taste_context="", iteration=1
):
    """Ask the LLM critic whether the candidate pool satisfies the prompt or needs targeted queries."""
    fallback = {"sufficient": True, "critique": "", "new_queries": [], "seed_tracks": []}
    if pipe is None:
        return fallback

    past_low = {str(q).strip().lower() for q in (past_queries or []) if str(q).strip()}
    sys_msg = (
        "You are an autonomous Music DJ Critic reviewing a candidate track pool before final curation.\n"
        f'Prompt: "{user_prompt}"\n'
        f"Critic Iteration: {iteration}\n"
        f"Pool Summary: {json.dumps(pool_summary, ensure_ascii=False)}\n"
        f"Queries Already Tried: {list(past_queries or [])[-12:]}\n"
        + (f"Taste DNA:\n{taste_context}\n" if taste_context else "")
        + "\nIf the pool has enough varied, on-vibe candidates, set sufficient=true and leave new_queries empty.\n"
        "Otherwise set sufficient=false and propose 2-3 fresh, specific search_queries (do NOT repeat past queries) "
        "and optionally 1-3 real seed_tracks ({name, artist}).\n"
        "Return ONLY JSON matching the schema."
    )

    try:
        if hasattr(pipe, "generate_json"):
            raw = pipe.generate_json(
                sys_msg,
                f"Review pool for '{user_prompt}' (pass {iteration}).",
                json_schema=_AGENT_CRITIC_SCHEMA,
                max_new_tokens=260,
                temperature=0.2,
            )
        else:
            raw = pipe.generate(f"<s>[INST] {sys_msg}\nReturn JSON only. [/INST]")
        parsed = extract_json_object(raw)
        if not isinstance(parsed, dict) or "sufficient" not in parsed:
            return fallback

        sufficient = bool(parsed.get("sufficient"))
        critique = str(parsed.get("critique") or "").strip()[:240]
        clean_queries = []
        seen_q = set(past_low)
        for q in parsed.get("new_queries") or []:
            qs = str(q or "").strip()
            ql = qs.lower()
            if qs and ql not in seen_q:
                seen_q.add(ql)
                clean_queries.append(qs)
            if len(clean_queries) >= 3:
                break

        clean_seeds = []
        for s in parsed.get("seed_tracks") or []:
            if isinstance(s, dict):
                nm = str(s.get("name") or "").strip()
                ar = str(s.get("artist") or "").strip()
                if nm and ar:
                    clean_seeds.append({"name": nm, "artist": ar})
            if len(clean_seeds) >= 4:
                break

        if not clean_queries and not clean_seeds:
            sufficient = True

        return {
            "sufficient": sufficient,
            "critique": critique,
            "new_queries": clean_queries,
            "seed_tracks": clean_seeds,
        }
    except Exception:
        return fallback


class DJChatSession:
    """Stateful multi-turn DJ Chat manager with locked-track memory and persistence."""

    def __init__(self):
        self.history = []
        self.locked_ids = set()

    def reset(self):
        self.history = []
        self.locked_ids = set()

    def _track_keys(self, track):
        keys = set()
        if not isinstance(track, dict):
            return keys
        tid = str(track.get("id") or "").strip()
        if tid:
            keys.add(tid)
        fp = get_track_fingerprint(track.get("name", ""), track.get("artist", ""))
        if fp:
            keys.add(f"fp:{fp}")
        return keys

    def is_locked(self, track):
        return bool(self._track_keys(track) & self.locked_ids)

    def toggle_lock(self, track_id, locked=None, tracks=None):
        tid = str(track_id or "").strip()
        if not tid:
            raise ValueError("track_id is required.")

        matched_fp = None
        for tr in tracks or []:
            if str(tr.get("id") or "").strip() == tid:
                fp = get_track_fingerprint(tr.get("name", ""), tr.get("artist", ""))
                if fp:
                    matched_fp = f"fp:{fp}"
                break

        currently = tid in self.locked_ids or (matched_fp in self.locked_ids if matched_fp else False)
        target_state = (not currently) if locked is None else bool(locked)

        if target_state:
            self.locked_ids.add(tid)
            if matched_fp:
                self.locked_ids.add(matched_fp)
        else:
            self.locked_ids.discard(tid)
            if matched_fp:
                self.locked_ids.discard(matched_fp)

        for tr in tracks or []:
            if isinstance(tr, dict):
                tr["locked"] = self.is_locked(tr)
        return target_state

    def append_turn(self, user_msg, assistant_reply, plan=None):
        now = datetime.datetime.now(datetime.timezone.utc).isoformat().replace("+00:00", "Z")
        self.history.append({"role": "user", "content": str(user_msg), "ts": now})
        self.history.append(
            {
                "role": "assistant",
                "content": str(assistant_reply),
                "ts": now,
                "action": (plan or {}).get("action", "refine"),
            }
        )
        if len(self.history) > 24:
            self.history = self.history[-24:]

    def serialize(self):
        return {
            "chat_history": list(self.history),
            "locked_ids": sorted(k for k in self.locked_ids if not k.startswith("fp:")),
            "locked_keys": sorted(self.locked_ids),
        }

    def load_from_payload(self, payload):
        self.reset()
        if not isinstance(payload, dict):
            return
        hist = payload.get("chat_history")
        if isinstance(hist, list):
            self.history = [dict(m) for m in hist[-24:] if isinstance(m, dict) and m.get("content")]
        for k in payload.get("locked_keys") or payload.get("locked_ids") or []:
            ks = str(k or "").strip()
            if ks:
                self.locked_ids.add(ks)
        for tr in payload.get("tracks") or []:
            if isinstance(tr, dict) and tr.get("locked"):
                self.locked_ids.update(self._track_keys(tr))

    def plan_chat_turn(self, pipe, message, current_tracks, taste_context=""):
        """Interpret a conversational DJ turn into structured playlist edits + natural reply."""
        msg = str(message or "").strip()
        if not msg:
            raise ValueError("Chat message cannot be empty.")

        heuristic_plan = parse_refinement_instruction(None, msg, current_tracks or [])
        heuristic_plan["action"] = "refine"
        heuristic_plan["discover_queries"] = []

        # Heuristic remote discovery detection if user asks to swap/add specific vibe
        lower = msg.lower()
        wants_discover = any(
            kw in lower
            for kw in ("discover", "new songs", "fresh", "more like", "swap", "replace", "add ", "brighter", "darker", "faster", "slower")
        )
        if wants_discover and not heuristic_plan.get("direction_prompt"):
            heuristic_plan["direction_prompt"] = msg[:140]

        if pipe is None:
            reply = self._build_fallback_reply(msg, heuristic_plan)
            return reply, heuristic_plan

        track_lines = []
        for idx, tr in enumerate(current_tracks or [], start=1):
            lock_tag = " [LOCKED]" if self.is_locked(tr) else ""
            bpm = f" {tr['bpm']}BPM" if tr.get("bpm") is not None else ""
            track_lines.append(
                f"{idx}. {tr.get('name', 'Unknown')} - {tr.get('artist', 'Unknown')} ({tr.get('source', '')}{bpm}){lock_tag}"
            )

        hist_lines = [
            f"{m.get('role', 'user').upper()}: {m.get('content', '')}"
            for m in self.history[-6:]
        ]

        sys_msg = (
            "You are the Studio AI DJ collaborating with a listener on their active playlist.\n"
            "Rules:\n"
            "- NEVER drop, replace, or remove tracks marked [LOCKED].\n"
            "- Keep playlist size stable unless the user asks to add/remove tracks.\n"
            "- Populate discover_queries when fresh remote/library candidates matching a new vibe are needed.\n\n"
            + (f"Taste DNA:\n{taste_context}\n\n" if taste_context else "")
            + ("Recent Chat:\n" + "\n".join(hist_lines) + "\n\n" if hist_lines else "")
            + "Current Playlist (1-based numbers):\n"
            + "\n".join(track_lines[:45])
            + "\n\nOutput ONLY JSON matching _CHAT_TURN_SCHEMA."
        )

        try:
            if hasattr(pipe, "generate_json"):
                raw = pipe.generate_json(
                    sys_msg,
                    msg,
                    json_schema=_CHAT_TURN_SCHEMA,
                    max_new_tokens=360,
                    temperature=0.2,
                )
            else:
                raw = pipe.generate(f"<s>[INST] {sys_msg}\nUser: {msg}\nReturn JSON only. [/INST]")
            parsed = extract_json_object(raw)
            if not isinstance(parsed, dict):
                raise ValueError("Invalid DJ Chat JSON")

            plan = dict(heuristic_plan)
            total = len(current_tracks or [])
            if isinstance(parsed.get("drop_track_numbers"), list):
                drops = []
                for num in parsed["drop_track_numbers"]:
                    if isinstance(num, int) and not isinstance(num, bool) and 1 <= num <= total:
                        z = num - 1
                        if z not in drops:
                            drops.append(z)
                if drops:
                    plan["drop_indices"] = drops
            if isinstance(parsed.get("remove_artists"), list):
                plan["remove_artists"] = normalize_avoid_list(parsed["remove_artists"])
            if parsed.get("filter_condition") in (
                "drop_slow",
                "drop_fast",
                "drop_low_energy",
                "drop_high_energy",
                "drop_explicit",
            ):
                plan["filter_condition"] = parsed["filter_condition"]
            if isinstance(parsed.get("direction_prompt"), str) and parsed["direction_prompt"].strip():
                plan["direction_prompt"] = parsed["direction_prompt"].strip()[:160]
            if isinstance(parsed.get("delta_tracks"), int) and not isinstance(parsed["delta_tracks"], bool):
                plan["delta_tracks"] = max(-25, min(25, parsed["delta_tracks"]))
            if isinstance(parsed.get("target_size"), int) and not isinstance(parsed["target_size"], bool):
                plan["target_size"] = max(5, min(100, parsed["target_size"]))
            if parsed.get("flow_mode") in ("mmr", "energy_arc", "bpm_ramp", "harmonic", "chill_out"):
                plan["flow_mode"] = parsed["flow_mode"]
            if isinstance(parsed.get("discover_queries"), list):
                plan["discover_queries"] = [
                    str(q).strip()[:120] for q in parsed["discover_queries"] if str(q or "").strip()
                ][:3]
            if isinstance(parsed.get("playlist_name"), str) and parsed["playlist_name"].strip():
                plan["playlist_name"] = parsed["playlist_name"].strip()[:100]
            plan["action"] = str(parsed.get("action") or "refine").strip().lower()

            reply = str(parsed.get("assistant_reply") or "").strip()
            if not reply:
                reply = self._build_fallback_reply(msg, plan)
            return reply, plan
        except Exception:
            reply = self._build_fallback_reply(msg, heuristic_plan)
            return reply, heuristic_plan

    def _build_fallback_reply(self, msg, plan):
        bits = []
        if plan.get("drop_indices"):
            bits.append(f"updated {len(plan['drop_indices'])} slot(s)")
        if plan.get("remove_artists"):
            bits.append(f"filtered out {', '.join(plan['remove_artists'])}")
        if plan.get("direction_prompt"):
            bits.append(f"steered toward '{plan['direction_prompt']}'")
        if plan.get("flow_mode"):
            bits.append(f"re-sequenced with {plan['flow_mode']} flow")
        if plan.get("delta_tracks"):
            d = plan["delta_tracks"]
            bits.append(f"adjusted length by {'+' if d > 0 else ''}{d}")
        if not bits:
            return f"Updated the mix for: {msg[:90]}."
        return "Done — " + ", ".join(bits) + " while keeping locked tracks intact."
