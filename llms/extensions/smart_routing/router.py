"""
Tier-to-Provider Mapping

Maps scoring tiers to llmspy's dynamically active providers.
Uses preferred model lists, capability filtering, and cost-based fallback.
"""

import re

from .config import (
    DEFAULT_AGENTIC_PREFERENCES,
    DEFAULT_TIER_PREFERENCES,
    TIER_COST_THRESHOLDS,
    TIER_RANK,
)


def select_provider(tier, agentic, providers, preferences=None, agentic_preferences=None):
    """
    Select the best provider and model for a given tier.

    Args:
        tier: "SIMPLE"|"MEDIUM"|"COMPLEX"|"REASONING"
        agentic: Whether to use agentic tier preferences
        providers: dict of {provider_id: provider_instance} from ctx.get_providers()
        preferences: Optional override for tier preferences
        agentic_preferences: Optional override for agentic preferences

    Returns:
        (provider_id, model_id, model_info) or None if no provider found
    """
    prefs = agentic_preferences or DEFAULT_AGENTIC_PREFERENCES if agentic else preferences or DEFAULT_TIER_PREFERENCES
    tier_pref = prefs.get(tier)
    if not tier_pref:
        return None

    required_caps = tier_pref.get("capabilities", {})

    # Try each preferred model in order
    for preferred_model in tier_pref["preferred_models"]:
        result = _find_provider_for_model(preferred_model, providers, required_caps)
        if result:
            return result

    # Fallback: find cheapest available model matching tier cost threshold
    return _fallback_by_cost(tier, providers, required_caps)


def _find_provider_for_model(model_name, providers, required_caps):
    """Find an active provider that supports the given model name."""
    for provider_id, provider in providers.items():
        # Skip the smart_routing provider itself
        if provider_id == "smart_routing":
            continue

        resolved = provider.provider_model(model_name)
        if not resolved:
            continue

        # Check capabilities
        info = provider.model_info(model_name)
        if info and _meets_capabilities(info, required_caps):
            return provider_id, resolved, info

    return None


def _meets_capabilities(model_info, required_caps):
    """Check if a model meets required capabilities."""
    for cap, required in required_caps.items():
        if required and not model_info.get(cap):
            return False
    return True


def _fallback_by_cost(tier, providers, required_caps):
    """Find the cheapest model within the tier's cost threshold."""
    max_cost = TIER_COST_THRESHOLDS.get(tier, 50.0)
    candidates = []

    for provider_id, provider in providers.items():
        if provider_id == "smart_routing":
            continue

        for model_id, model_info in provider.models.items():
            cost = model_info.get("cost", {})
            input_cost = cost.get("input", 0) if isinstance(cost, dict) else 0

            if input_cost <= max_cost and _meets_capabilities(model_info, required_caps):
                candidates.append((provider_id, model_id, model_info, input_cost))

    if not candidates:
        # Last resort: return any available model
        return _any_available_model(providers)

    # Sort by input cost ascending
    candidates.sort(key=lambda x: x[3])
    best = candidates[0]
    return best[0], best[1], best[2]


def _any_available_model(providers):
    """Return any available model as absolute last resort."""
    for provider_id, provider in providers.items():
        if provider_id == "smart_routing":
            continue
        for model_id, model_info in provider.models.items():
            return provider_id, model_id, model_info
    return None
