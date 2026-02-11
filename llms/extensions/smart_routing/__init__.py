"""
Smart Routing extension.

Routes requests with model="auto" to a selected provider/model based on
prompt complexity and configured preferences.
"""

from __future__ import annotations

import json
import math
import os
import re
from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple

from aiohttp import web

from .config import (
    DEFAULT_AGENTIC_PREFERENCES,
    DEFAULT_OVERRIDES,
    DEFAULT_SCORING_CONFIG,
    DEFAULT_TIER_PREFERENCES,
    TIER_RANK,
    TIERS,
    default_config,
)
from .router import rank_candidates
from .scorer import classify

g_ctx = None
g_config = None
g_stats = {
    "total_routed": 0,
    "tiers": {"SIMPLE": 0, "MEDIUM": 0, "COMPLEX": 0, "REASONING": 0},
    "providers": {},
    "ambiguous": 0,
    "fallback_attempts": 0,
    "candidate_failures": 0,
}

_SELECTION_CONTEXT_KEY = "_smart_routing_selection"


def _config_path():
    return os.path.expanduser("~/.llms/smart_routing.json")


def _log(message):
    if g_ctx:
        g_ctx.log(message)


def _deep_merge(base, override):
    if not isinstance(base, dict):
        return deepcopy(override)
    merged = deepcopy(base)
    if not isinstance(override, dict):
        return merged

    for key, value in override.items():
        base_value = merged.get(key)
        if isinstance(base_value, dict) and isinstance(value, dict):
            merged[key] = _deep_merge(base_value, value)
        else:
            merged[key] = deepcopy(value)
    return merged


def _coerce_bool(value, default):
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in ("1", "true", "yes", "on"):
            return True
        if lowered in ("0", "false", "no", "off"):
            return False
    return default


def _coerce_int(value, default, minimum=0):
    try:
        coerced = int(value)
    except (TypeError, ValueError):
        return default
    return coerced if coerced >= minimum else default


def _coerce_float(value, default, minimum=None, maximum=None):
    try:
        coerced = float(value)
    except (TypeError, ValueError):
        return default
    if minimum is not None and coerced < minimum:
        return default
    if maximum is not None and coerced > maximum:
        return default
    return coerced


def _coerce_tier(value, default):
    if value is None:
        return default
    tier = str(value).upper()
    return tier if tier in TIER_RANK else default


def _sanitize_string_list(value, default):
    if not isinstance(value, list):
        return list(default)
    ret = []
    for item in value:
        if isinstance(item, str):
            stripped = item.strip()
            if stripped:
                ret.append(stripped)
    return ret if ret else list(default)


def _sanitize_capabilities(value, default):
    if not isinstance(value, dict):
        return deepcopy(default)
    return {str(k): bool(v) for k, v in value.items()}


def _normalize_scoring(scoring):
    merged = _deep_merge(DEFAULT_SCORING_CONFIG, scoring if isinstance(scoring, dict) else {})

    merged["tokenCountThresholds"] = {
        "simple": _coerce_int(
            merged.get("tokenCountThresholds", {}).get("simple"),
            DEFAULT_SCORING_CONFIG["tokenCountThresholds"]["simple"],
            minimum=1,
        ),
        "complex": _coerce_int(
            merged.get("tokenCountThresholds", {}).get("complex"),
            DEFAULT_SCORING_CONFIG["tokenCountThresholds"]["complex"],
            minimum=1,
        ),
    }
    if merged["tokenCountThresholds"]["complex"] < merged["tokenCountThresholds"]["simple"]:
        merged["tokenCountThresholds"]["complex"] = merged["tokenCountThresholds"]["simple"]

    keyword_lists = [
        "codeKeywords",
        "reasoningKeywords",
        "simpleKeywords",
        "technicalKeywords",
        "creativeKeywords",
        "imperativeVerbs",
        "constraintIndicators",
        "outputFormatKeywords",
        "referenceKeywords",
        "negationKeywords",
        "domainSpecificKeywords",
        "agenticTaskKeywords",
    ]
    for key in keyword_lists:
        merged[key] = _sanitize_string_list(merged.get(key), DEFAULT_SCORING_CONFIG[key])

    default_weights = DEFAULT_SCORING_CONFIG["dimensionWeights"]
    weights = merged.get("dimensionWeights", {})
    normalized_weights = {}
    for key, default_value in default_weights.items():
        normalized_weights[key] = _coerce_float(weights.get(key), default_value, minimum=-2.0, maximum=2.0)
    merged["dimensionWeights"] = normalized_weights

    default_boundaries = DEFAULT_SCORING_CONFIG["tierBoundaries"]
    boundaries = merged.get("tierBoundaries", {})
    simple_medium = _coerce_float(boundaries.get("simpleMedium"), default_boundaries["simpleMedium"])
    medium_complex = _coerce_float(boundaries.get("mediumComplex"), default_boundaries["mediumComplex"])
    complex_reasoning = _coerce_float(boundaries.get("complexReasoning"), default_boundaries["complexReasoning"])
    if medium_complex < simple_medium:
        medium_complex = simple_medium
    if complex_reasoning < medium_complex:
        complex_reasoning = medium_complex
    merged["tierBoundaries"] = {
        "simpleMedium": simple_medium,
        "mediumComplex": medium_complex,
        "complexReasoning": complex_reasoning,
    }

    merged["confidenceSteepness"] = _coerce_float(
        merged.get("confidenceSteepness"),
        DEFAULT_SCORING_CONFIG["confidenceSteepness"],
        minimum=0.1,
        maximum=100.0,
    )
    merged["confidenceThreshold"] = _coerce_float(
        merged.get("confidenceThreshold"),
        DEFAULT_SCORING_CONFIG["confidenceThreshold"],
        minimum=0.0,
        maximum=1.0,
    )
    return merged


def _normalize_preferences(prefs, defaults):
    source = prefs if isinstance(prefs, dict) else {}
    normalized = {}
    for tier in TIERS:
        default_entry = defaults[tier]
        raw_entry = source.get(tier, {})
        if not isinstance(raw_entry, dict):
            raw_entry = {}
        normalized[tier] = {
            "preferred_models": _sanitize_string_list(
                raw_entry.get("preferred_models"),
                default_entry.get("preferred_models", []),
            ),
            "capabilities": _sanitize_capabilities(
                raw_entry.get("capabilities"),
                default_entry.get("capabilities", {}),
            ),
        }
    return normalized


def _normalize_overrides(overrides):
    merged = _deep_merge(DEFAULT_OVERRIDES, overrides if isinstance(overrides, dict) else {})
    normalized = {
        "maxTokensForceComplex": _coerce_int(
            merged.get("maxTokensForceComplex"),
            DEFAULT_OVERRIDES["maxTokensForceComplex"],
            minimum=1,
        ),
        "structuredOutputMinTier": _coerce_tier(
            merged.get("structuredOutputMinTier"),
            DEFAULT_OVERRIDES["structuredOutputMinTier"],
        ),
        "ambiguousDefaultTier": _coerce_tier(
            merged.get("ambiguousDefaultTier"),
            DEFAULT_OVERRIDES["ambiguousDefaultTier"],
        ),
        "agenticMode": _coerce_bool(merged.get("agenticMode"), DEFAULT_OVERRIDES["agenticMode"]),
    }
    return normalized


def normalize_config(raw_config):
    merged = _deep_merge(default_config(), raw_config if isinstance(raw_config, dict) else {})
    return {
        "scoring": _normalize_scoring(merged.get("scoring", {})),
        "overrides": _normalize_overrides(merged.get("overrides", {})),
        "tierPreferences": _normalize_preferences(
            merged.get("tierPreferences", {}),
            DEFAULT_TIER_PREFERENCES,
        ),
        "agenticPreferences": _normalize_preferences(
            merged.get("agenticPreferences", {}),
            DEFAULT_AGENTIC_PREFERENCES,
        ),
    }


def _load_user_config():
    config_path = _config_path()
    if not os.path.exists(config_path):
        return {}
    try:
        with open(config_path, "r", encoding="utf-8") as handle:
            loaded = json.load(handle)
            if isinstance(loaded, dict):
                return loaded
            _log("smart_routing config ignored: expected a JSON object")
            return {}
    except Exception as ex:
        _log("smart_routing config load failed (%s), using defaults" % ex)
        return {}


def _save_user_config(config):
    config_path = _config_path()
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    tmp_path = "%s.tmp" % config_path
    with open(tmp_path, "w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(tmp_path, config_path)


def _chat_content_to_text(content):
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for part in content:
            if not isinstance(part, dict):
                continue
            if part.get("type") == "text" and isinstance(part.get("text"), str):
                parts.append(part["text"])
            elif isinstance(part.get("content"), str):
                parts.append(part["content"])
        return " ".join(parts)
    if isinstance(content, dict):
        text_value = content.get("text")
        if isinstance(text_value, str):
            return text_value
    return ""


def _last_user_prompt(messages):
    if not isinstance(messages, list):
        return ""
    for message in reversed(messages):
        if isinstance(message, dict) and message.get("role") == "user":
            return _chat_content_to_text(message.get("content"))
    return ""


def _system_prompt(messages):
    if not isinstance(messages, list):
        return None
    for message in messages:
        if isinstance(message, dict) and message.get("role") == "system":
            content = message.get("content")
            text = _chat_content_to_text(content)
            return text if text else None
    return None


class SmartRouterProvider:
    """Virtual provider that claims model='auto' and delegates to real providers."""

    sdk = "@llmspy/smart-routing"

    def __init__(self, ctx, config=None):
        self.ctx = ctx
        self.id = "smart_routing"
        self.name = "Smart Routing"
        self.api = ""
        self.api_key = None
        self.env = []
        self.map_models = {}
        self.modalities = {}
        self.models = {
            "auto": {
                "id": "auto",
                "name": "Auto (Smart Routing)",
                "family": "smart-routing",
                "tool_call": True,
                "reasoning": True,
                "modalities": {"input": ["text"], "output": ["text"]},
                "cost": {"input": 0, "output": 0},
                "limit": {"context": 1_000_000, "output": 100_000},
            }
        }

        self.config = normalize_config(config or {})
        self.scoring_config = self.config["scoring"]
        self.overrides = self.config["overrides"]
        self.tier_preferences = self.config["tierPreferences"]
        self.agentic_preferences = self.config["agenticPreferences"]

    def test(self, **kwargs):
        return True

    async def load(self):
        return None

    def validate(self, **kwargs):
        return None

    def provider_model(self, model):
        if model and str(model).lower() == "auto":
            return "auto"
        return None

    def model_info(self, model):
        if model and str(model).lower() == "auto":
            return self.models["auto"]
        return None

    def model_cost(self, model):
        info = self.model_info(model)
        return info.get("cost") if info else None

    def _get_pinned_selection(self, context, providers):
        if not isinstance(context, dict):
            return None
        pinned = context.get(_SELECTION_CONTEXT_KEY)
        if not isinstance(pinned, dict):
            return None

        provider_id = pinned.get("provider")
        model_id = pinned.get("model")
        if not provider_id or not model_id:
            return None
        provider = providers.get(provider_id)
        if not provider or provider_id == "smart_routing":
            return None

        resolved = provider.provider_model(model_id)
        if not resolved:
            return None
        info = provider.model_info(model_id) or provider.model_info(resolved)
        if not isinstance(info, dict):
            models = getattr(provider, "models", {}) or {}
            info = models.get(resolved)
            if not isinstance(info, dict):
                return None
        return provider_id, resolved, info

    def _pin_selection(self, context, provider_id, model_id, tier, confidence, agentic):
        if not isinstance(context, dict):
            return
        context[_SELECTION_CONTEXT_KEY] = {
            "provider": provider_id,
            "model": model_id,
            "tier": tier,
            "confidence": round(confidence, 6),
            "agentic": bool(agentic),
        }

    def _update_context_provider_info(self, context, provider, provider_id, model_id):
        if not isinstance(context, dict):
            return
        context["provider"] = provider_id
        model_info = provider.model_info(model_id) or provider.model_info(provider.provider_model(model_id) or model_id)
        if isinstance(model_info, dict):
            context["modelInfo"] = model_info
            context["modelCost"] = model_info.get("cost", {"input": 0, "output": 0})

    async def chat(self, chat, context=None):
        messages = chat.get("messages", [])
        prompt = _last_user_prompt(messages)
        system_prompt = _system_prompt(messages)

        full_text = "%s %s" % (system_prompt or "", prompt)
        estimated_tokens = max(1, int(math.ceil(len(full_text) / 4.0)))

        has_tools = bool(chat.get("tools"))
        result = classify(prompt, system_prompt, estimated_tokens, self.scoring_config)

        is_auto_agentic = result.agentic_score >= 0.75
        is_explicit_agentic = bool(self.overrides.get("agenticMode", False))
        use_agentic = bool(has_tools or is_auto_agentic or is_explicit_agentic)

        tier = result.tier
        confidence = float(result.confidence)
        reasoning_parts = ["score=%.2f" % result.score]
        if result.signals:
            reasoning_parts.append(", ".join(result.signals))

        if estimated_tokens > int(self.overrides.get("maxTokensForceComplex", 100_000)):
            tier = "COMPLEX"
            confidence = 0.95
            reasoning_parts.append("large context (%d tokens)" % estimated_tokens)

        if tier is None:
            tier = self.overrides.get("ambiguousDefaultTier", "MEDIUM")
            confidence = 0.5
            reasoning_parts.append("ambiguous -> %s" % tier)
            g_stats["ambiguous"] += 1

        if system_prompt and re.search(r"json|structured|schema", system_prompt, re.IGNORECASE):
            min_tier = self.overrides.get("structuredOutputMinTier", "MEDIUM")
            if TIER_RANK.get(tier, 0) < TIER_RANK.get(min_tier, 0):
                tier = min_tier
                reasoning_parts.append("upgraded to %s (structured output)" % min_tier)

        if has_tools:
            reasoning_parts.append("agentic")
        elif is_auto_agentic:
            reasoning_parts.append("auto-agentic")
        elif is_explicit_agentic:
            reasoning_parts.append("forced-agentic")

        providers = self.ctx.get_providers()
        pinned_selection = self._get_pinned_selection(context, providers)
        if pinned_selection:
            candidates = [pinned_selection]
            pinned = True
        else:
            candidates = rank_candidates(
                tier,
                use_agentic,
                providers,
                preferences=self.tier_preferences,
                agentic_preferences=self.agentic_preferences,
            )
            pinned = False

        if not candidates:
            raise Exception("Smart routing: no available provider candidates")

        first_exception = None
        for index, candidate in enumerate(candidates):
            provider_id, model_id, _model_info = candidate
            provider = providers.get(provider_id)
            if not provider or provider_id == "smart_routing":
                continue

            if index > 0:
                g_stats["fallback_attempts"] += 1
                reasoning_parts.append("fallback=%s/%s" % (provider_id, model_id))

            chat["model"] = model_id
            self._update_context_provider_info(context, provider, provider_id, model_id)

            try:
                response = await provider.chat(chat, context=context)
            except Exception as ex:
                g_stats["candidate_failures"] += 1
                if first_exception is None:
                    first_exception = ex
                self.ctx.log("Candidate failed %s/%s: %s" % (provider_id, model_id, ex))
                continue

            self._pin_selection(context, provider_id, model_id, tier, confidence, use_agentic)

            g_stats["total_routed"] += 1
            g_stats["tiers"][tier] = g_stats["tiers"].get(tier, 0) + 1
            g_stats["providers"][provider_id] = g_stats["providers"].get(provider_id, 0) + 1

            self.ctx.log(
                "Routed: tier=%s conf=%.2f -> %s/%s (%s)"
                % (tier, confidence, provider_id, model_id, " | ".join(reasoning_parts))
            )

            if isinstance(response, dict):
                response["routing"] = {
                    "tier": tier,
                    "confidence": round(confidence, 3),
                    "score": round(result.score, 4),
                    "provider": provider_id,
                    "model": model_id,
                    "agentic": use_agentic,
                    "signals": result.signals,
                    "pinned": pinned,
                    "attempts": index + 1,
                    "reasoning": " | ".join(reasoning_parts),
                }
            return response

        if first_exception:
            raise first_exception
        raise Exception("Smart routing: no candidate provider could fulfill this request")


async def get_config_handler(request):
    return web.json_response(g_config or default_config())


async def update_config_handler(request):
    global g_config
    try:
        updates = await request.json()
        if not isinstance(updates, dict):
            raise ValueError("Expected JSON object")

        g_config = normalize_config(_deep_merge(g_config or default_config(), updates))
        _save_user_config(g_config)

        providers = g_ctx.get_providers()
        providers["smart_routing"] = SmartRouterProvider(g_ctx, g_config)
        return web.json_response({"status": "ok", "config": g_config})
    except Exception as ex:
        return web.json_response({"error": str(ex)}, status=400)


async def get_stats_handler(request):
    return web.json_response(g_stats)


def install(ctx):
    global g_ctx
    g_ctx = ctx
    ctx.add_get("config", get_config_handler)
    ctx.add_post("config", update_config_handler)
    ctx.add_get("stats", get_stats_handler)


async def load(ctx):
    global g_config
    raw_config = _load_user_config()
    g_config = normalize_config(raw_config)

    providers = ctx.get_providers()
    providers["smart_routing"] = SmartRouterProvider(ctx, g_config)

    ctx.log("Smart routing enabled. Model 'auto' is now available.")


__install__ = install
__load__ = load
