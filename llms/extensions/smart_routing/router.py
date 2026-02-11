"""
Tier-to-provider ranking and selection.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

from .config import (
    DEFAULT_AGENTIC_PREFERENCES,
    DEFAULT_TIER_PREFERENCES,
    TIER_COST_THRESHOLDS,
)

Candidate = Tuple[str, str, Dict[str, Any]]


def select_provider(
    tier: str,
    agentic: bool,
    providers: Dict[str, Any],
    preferences: Optional[Dict[str, Any]] = None,
    agentic_preferences: Optional[Dict[str, Any]] = None,
) -> Optional[Candidate]:
    ranked = rank_candidates(
        tier,
        agentic,
        providers,
        preferences=preferences,
        agentic_preferences=agentic_preferences,
    )
    return ranked[0] if ranked else None


def rank_candidates(
    tier: str,
    agentic: bool,
    providers: Dict[str, Any],
    preferences: Optional[Dict[str, Any]] = None,
    agentic_preferences: Optional[Dict[str, Any]] = None,
) -> List[Candidate]:
    prefs = _tier_preferences(agentic, preferences, agentic_preferences)
    tier_pref = prefs.get(tier) or {}
    required_caps = tier_pref.get("capabilities", {}) if isinstance(tier_pref, dict) else {}
    preferred_models = tier_pref.get("preferred_models", []) if isinstance(tier_pref, dict) else []

    candidates: List[Candidate] = []

    for preferred_model in preferred_models:
        candidates.extend(_find_provider_candidates_for_model(str(preferred_model), providers, required_caps))

    candidates.extend(_fallback_by_cost_candidates(tier, providers, required_caps))
    if not candidates:
        candidates.extend(_any_available_models(providers))

    return _dedupe_candidates(candidates)


def _tier_preferences(
    agentic: bool,
    preferences: Optional[Dict[str, Any]],
    agentic_preferences: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    if agentic:
        return agentic_preferences or DEFAULT_AGENTIC_PREFERENCES
    return preferences or DEFAULT_TIER_PREFERENCES


def _find_provider_candidates_for_model(
    model_name: str,
    providers: Dict[str, Any],
    required_caps: Dict[str, Any],
) -> List[Candidate]:
    ranked: List[Tuple[float, Candidate]] = []
    for provider_id, provider in providers.items():
        if provider_id == "smart_routing":
            continue

        resolved = provider.provider_model(model_name)
        if not resolved:
            continue

        info = _model_info(provider, model_name, resolved)
        if not info or not _meets_capabilities(info, required_caps):
            continue
        ranked.append((_input_cost(info), (provider_id, resolved, info)))

    ranked.sort(key=lambda item: item[0])
    return [candidate for _cost, candidate in ranked]


def _meets_capabilities(model_info: Dict[str, Any], required_caps: Dict[str, Any]) -> bool:
    if not isinstance(required_caps, dict):
        return True
    for capability, required in required_caps.items():
        if bool(required) and not bool(model_info.get(capability)):
            return False
    return True


def _fallback_by_cost_candidates(
    tier: str,
    providers: Dict[str, Any],
    required_caps: Dict[str, Any],
) -> List[Candidate]:
    max_cost = float(TIER_COST_THRESHOLDS.get(tier, 50.0))
    ranked: List[Tuple[float, Candidate]] = []

    for provider_id, provider in providers.items():
        if provider_id == "smart_routing":
            continue

        models = getattr(provider, "models", {}) or {}
        for model_id, model_info in models.items():
            if not isinstance(model_info, dict):
                continue
            if not _meets_capabilities(model_info, required_caps):
                continue
            input_cost = _input_cost(model_info)
            if input_cost <= max_cost:
                ranked.append((input_cost, (provider_id, str(model_id), model_info)))

    ranked.sort(key=lambda item: item[0])
    return [candidate for _cost, candidate in ranked]


def _any_available_models(providers: Dict[str, Any]) -> List[Candidate]:
    ret: List[Candidate] = []
    for provider_id, provider in providers.items():
        if provider_id == "smart_routing":
            continue
        models = getattr(provider, "models", {}) or {}
        for model_id, model_info in models.items():
            if isinstance(model_info, dict):
                ret.append((provider_id, str(model_id), model_info))
    return ret


def _input_cost(model_info: Dict[str, Any]) -> float:
    cost = model_info.get("cost", {})
    if not isinstance(cost, dict):
        return math.inf
    input_cost = cost.get("input")
    try:
        if input_cost is None:
            return math.inf
        return float(input_cost)
    except (TypeError, ValueError):
        return math.inf


def _model_info(provider: Any, requested_model: str, resolved_model: str) -> Optional[Dict[str, Any]]:
    info = provider.model_info(requested_model)
    if isinstance(info, dict):
        return info
    info = provider.model_info(resolved_model)
    if isinstance(info, dict):
        return info
    models = getattr(provider, "models", {}) or {}
    direct = models.get(resolved_model)
    if isinstance(direct, dict):
        return direct
    return None


def _dedupe_candidates(candidates: List[Candidate]) -> List[Candidate]:
    seen = set()
    deduped: List[Candidate] = []
    for candidate in candidates:
        key = (candidate[0], candidate[1])
        if key in seen:
            continue
        seen.add(key)
        deduped.append(candidate)
    return deduped
