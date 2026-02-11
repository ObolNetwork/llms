"""
Smart Routing Extension for llmspy

Routes requests with model="auto" to the best available provider/model
based on prompt complexity analysis. Uses a 14-dimension weighted scoring
system ported from ClawRouter.
"""

import json
import math
import os
import re
import time

from aiohttp import web

from .config import (
    DEFAULT_OVERRIDES,
    DEFAULT_SCORING_CONFIG,
    DEFAULT_TIER_PREFERENCES,
    DEFAULT_AGENTIC_PREFERENCES,
    TIER_RANK,
)
from .scorer import classify
from .router import select_provider

g_ctx = None
g_config = None
g_stats = {
    "total_routed": 0,
    "tiers": {"SIMPLE": 0, "MEDIUM": 0, "COMPLEX": 0, "REASONING": 0},
    "providers": {},
    "ambiguous": 0,
}


def _load_user_config():
    """Load user config from ~/.llms/smart_routing.json if it exists."""
    config_path = os.path.expanduser("~/.llms/smart_routing.json")
    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def _merge_config(base, override):
    """Deep merge override into base config."""
    merged = base.copy()
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _merge_config(merged[key], value)
        else:
            merged[key] = value
    return merged


class SmartRouterProvider:
    """Virtual provider that routes model='auto' to the best real provider."""

    sdk = "@llmspy/smart-routing"

    def __init__(self, ctx, config=None):
        self.ctx = ctx
        self.id = "smart_routing"
        self.name = "Smart Routing"
        self.api = ""
        self.api_key = None
        self.env = []
        self.map_models = {}
        self.models = {
            "auto": {
                "id": "auto",
                "name": "Auto (Smart Routing)",
                "family": "smart-routing",
                "tool_call": True,
                "reasoning": True,
                "modalities": {"input": ["text"], "output": ["text"]},
                "cost": {"input": 0, "output": 0},
                "limit": {"context": 1000000, "output": 100000},
            }
        }
        self.modalities = {}
        self.scoring_config = config.get("scoring", DEFAULT_SCORING_CONFIG) if config else DEFAULT_SCORING_CONFIG
        self.overrides = config.get("overrides", DEFAULT_OVERRIDES) if config else DEFAULT_OVERRIDES
        self.tier_preferences = config.get("tierPreferences", DEFAULT_TIER_PREFERENCES) if config else DEFAULT_TIER_PREFERENCES
        self.agentic_preferences = config.get("agenticPreferences", DEFAULT_AGENTIC_PREFERENCES) if config else DEFAULT_AGENTIC_PREFERENCES

    def test(self, **kwargs):
        return True

    async def load(self):
        pass

    def validate(self, **kwargs):
        return None

    def provider_model(self, model):
        if model and model.lower() == "auto":
            return "auto"
        return None

    def model_info(self, model):
        if model and model.lower() == "auto":
            return self.models["auto"]
        return None

    def model_cost(self, model):
        info = self.model_info(model)
        return info.get("cost") if info else None

    async def chat(self, chat, context=None):
        """Score the prompt, select best provider, and delegate."""
        messages = chat.get("messages", [])

        # Extract user prompt (last user message)
        prompt = ""
        for msg in reversed(messages):
            if msg.get("role") == "user":
                content = msg.get("content", "")
                if isinstance(content, str):
                    prompt = content
                elif isinstance(content, list):
                    prompt = " ".join(
                        part.get("text", "") for part in content
                        if isinstance(part, dict) and part.get("type") == "text"
                    )
                break

        # Extract system prompt
        system_prompt = None
        for msg in messages:
            if msg.get("role") == "system":
                content = msg.get("content", "")
                system_prompt = content if isinstance(content, str) else str(content)
                break

        # Estimate tokens
        full_text = f"{system_prompt or ''} {prompt}"
        estimated_tokens = math.ceil(len(full_text) / 4)

        # Check if agentic (tools present in request)
        has_tools = bool(chat.get("tools"))
        is_explicit_agentic = self.overrides.get("agenticMode", False)

        # Score the prompt
        result = classify(prompt, system_prompt, estimated_tokens, self.scoring_config)

        # Determine agentic mode
        is_auto_agentic = result.agentic_score >= 0.75
        use_agentic = (is_auto_agentic or has_tools or is_explicit_agentic)

        # Override: large context → force COMPLEX
        tier = result.tier
        confidence = result.confidence
        reasoning_parts = [f"score={result.score:.2f}"]
        if result.signals:
            reasoning_parts.append(", ".join(result.signals))

        if estimated_tokens > self.overrides.get("maxTokensForceComplex", 100_000):
            tier = "COMPLEX"
            confidence = 0.95
            reasoning_parts.append(f"large context ({estimated_tokens} tokens)")

        if tier is None:
            tier = self.overrides.get("ambiguousDefaultTier", "MEDIUM")
            confidence = 0.5
            reasoning_parts.append(f"ambiguous -> {tier}")
            g_stats["ambiguous"] += 1

        # Override: structured output → minimum tier
        if system_prompt and re.search(r"json|structured|schema", system_prompt, re.IGNORECASE):
            min_tier = self.overrides.get("structuredOutputMinTier", "MEDIUM")
            if TIER_RANK.get(tier, 0) < TIER_RANK.get(min_tier, 0):
                reasoning_parts.append(f"upgraded to {min_tier} (structured output)")
                tier = min_tier

        if use_agentic:
            reasoning_parts.append("agentic" if has_tools else "auto-agentic")

        # Select provider
        providers = self.ctx.get_providers()
        selection = select_provider(
            tier, use_agentic, providers,
            self.tier_preferences, self.agentic_preferences,
        )

        if not selection:
            raise Exception("Smart routing: no available provider found for any tier")

        provider_id, model_id, model_info = selection

        # Update stats
        g_stats["total_routed"] += 1
        g_stats["tiers"][tier] = g_stats["tiers"].get(tier, 0) + 1
        g_stats["providers"][provider_id] = g_stats["providers"].get(provider_id, 0) + 1

        # Rewrite chat model and delegate
        chat["model"] = model_id
        provider = providers[provider_id]

        # Update context with real provider info
        if context:
            context["provider"] = provider_id
            real_model_info = provider.model_info(model_id)
            if real_model_info:
                context["modelInfo"] = real_model_info
                context["modelCost"] = real_model_info.get("cost", {"input": 0, "output": 0})

        self.ctx.log(
            f"Routed: tier={tier} conf={confidence:.2f} -> {provider_id}/{model_id} "
            f"({' | '.join(reasoning_parts)})"
        )

        response = await provider.chat(chat, context=context)

        # Add routing metadata to response
        response["routing"] = {
            "tier": tier,
            "confidence": round(confidence, 3),
            "score": round(result.score, 4),
            "provider": provider_id,
            "model": model_id,
            "agentic": use_agentic,
            "signals": result.signals,
            "reasoning": " | ".join(reasoning_parts),
        }

        return response


# ─── API Handlers ───

async def get_config_handler(request):
    return web.json_response({
        "scoring": g_config.get("scoring", DEFAULT_SCORING_CONFIG) if g_config else DEFAULT_SCORING_CONFIG,
        "overrides": g_config.get("overrides", DEFAULT_OVERRIDES) if g_config else DEFAULT_OVERRIDES,
        "tierPreferences": g_config.get("tierPreferences", DEFAULT_TIER_PREFERENCES) if g_config else DEFAULT_TIER_PREFERENCES,
        "agenticPreferences": g_config.get("agenticPreferences", DEFAULT_AGENTIC_PREFERENCES) if g_config else DEFAULT_AGENTIC_PREFERENCES,
    })


async def update_config_handler(request):
    global g_config
    try:
        updates = await request.json()
        g_config = _merge_config(g_config or {}, updates)

        # Save to disk
        config_path = os.path.expanduser("~/.llms/smart_routing.json")
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(g_config, f, indent=2)

        # Rebuild provider with new config
        providers = g_ctx.get_providers()
        providers["smart_routing"] = SmartRouterProvider(g_ctx, g_config)

        return web.json_response({"status": "ok"})
    except Exception as e:
        return web.json_response({"error": str(e)}, status=400)


async def get_stats_handler(request):
    return web.json_response(g_stats)


# ─── Extension Lifecycle ───

def install(ctx):
    global g_ctx
    g_ctx = ctx

    # Register API endpoints
    ctx.add_get("config", get_config_handler)
    ctx.add_post("config", update_config_handler)
    ctx.add_get("stats", get_stats_handler)


async def load(ctx):
    global g_config

    # Load user config
    user_config = _load_user_config()
    g_config = _merge_config({}, user_config) if user_config else {}

    # Create and register the virtual provider
    provider = SmartRouterProvider(ctx, g_config if g_config else None)
    providers = ctx.get_providers()
    providers["smart_routing"] = provider

    ctx.log(f"Smart routing enabled. Model 'auto' is now available.")


__install__ = install
__load__ = load
