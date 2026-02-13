"""
14-dimension weighted scoring classifier.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class ScoringResult:
    score: float
    tier: Optional[str]
    confidence: float
    signals: List[str] = field(default_factory=list)
    agentic_score: float = 0.0


DimensionScore = Tuple[str, float, Optional[str]]


def _score_token_count(estimated_tokens: int, thresholds: Dict[str, Any]) -> DimensionScore:
    simple_threshold = int(thresholds.get("simple", 50))
    complex_threshold = int(thresholds.get("complex", 500))
    if estimated_tokens < simple_threshold:
        return "tokenCount", -1.0, "short (%d tokens)" % estimated_tokens
    if estimated_tokens > complex_threshold:
        return "tokenCount", 1.0, "long (%d tokens)" % estimated_tokens
    return "tokenCount", 0.0, None


def _score_keyword_match(
    text: str,
    keywords: List[str],
    name: str,
    signal_label: str,
    thresholds: Dict[str, int],
    scores: Dict[str, float],
) -> DimensionScore:
    matches = [kw for kw in keywords if kw.lower() in text]
    if len(matches) >= int(thresholds.get("high", 2)):
        return name, float(scores.get("high", 0.0)), "%s (%s)" % (signal_label, ", ".join(matches[:3]))
    if len(matches) >= int(thresholds.get("low", 1)):
        return name, float(scores.get("low", 0.0)), "%s (%s)" % (signal_label, ", ".join(matches[:3]))
    return name, float(scores.get("none", 0.0)), None


def _score_multi_step(text: str) -> DimensionScore:
    patterns = [r"first.*then", r"step \d", r"\d\.\s"]
    for pattern in patterns:
        if re.search(pattern, text, re.IGNORECASE):
            return "multiStepPatterns", 0.5, "multi-step"
    return "multiStepPatterns", 0.0, None


def _score_question_complexity(prompt: str) -> DimensionScore:
    count = prompt.count("?")
    if count > 3:
        return "questionComplexity", 0.5, "%d questions" % count
    return "questionComplexity", 0.0, None


def _score_agentic_task(text: str, keywords: List[str]) -> Tuple[DimensionScore, float]:
    match_count = 0
    signals = []
    for keyword in keywords:
        if keyword.lower() in text:
            match_count += 1
            if len(signals) < 3:
                signals.append(keyword)

    if match_count >= 4:
        return ("agenticTask", 1.0, "agentic (%s)" % ", ".join(signals)), 1.0
    if match_count >= 3:
        return ("agenticTask", 0.6, "agentic (%s)" % ", ".join(signals)), 0.6
    if match_count >= 1:
        return ("agenticTask", 0.2, "agentic-light (%s)" % ", ".join(signals)), 0.2
    return ("agenticTask", 0.0, None), 0.0


def _calibrate_confidence(distance: float, steepness: float) -> float:
    return 1.0 / (1.0 + math.exp(-steepness * distance))


def classify(prompt: str, system_prompt: Optional[str], estimated_tokens: int, config: Dict[str, Any]) -> ScoringResult:
    text = ("%s %s" % (system_prompt or "", prompt or "")).lower()
    user_text = (prompt or "").lower()

    dimensions = [
        _score_token_count(estimated_tokens, config.get("tokenCountThresholds", {})),
        _score_keyword_match(
            text,
            config.get("codeKeywords", []),
            "codePresence",
            "code",
            {"low": 1, "high": 2},
            {"none": 0, "low": 0.5, "high": 1.0},
        ),
        _score_keyword_match(
            user_text,
            config.get("reasoningKeywords", []),
            "reasoningMarkers",
            "reasoning",
            {"low": 1, "high": 2},
            {"none": 0, "low": 0.7, "high": 1.0},
        ),
        _score_keyword_match(
            text,
            config.get("technicalKeywords", []),
            "technicalTerms",
            "technical",
            {"low": 2, "high": 4},
            {"none": 0, "low": 0.5, "high": 1.0},
        ),
        _score_keyword_match(
            text,
            config.get("creativeKeywords", []),
            "creativeMarkers",
            "creative",
            {"low": 1, "high": 2},
            {"none": 0, "low": 0.5, "high": 0.7},
        ),
        _score_keyword_match(
            text,
            config.get("simpleKeywords", []),
            "simpleIndicators",
            "simple",
            {"low": 1, "high": 2},
            {"none": 0, "low": -1.0, "high": -1.0},
        ),
        _score_multi_step(text),
        _score_question_complexity(prompt or ""),
        _score_keyword_match(
            text,
            config.get("imperativeVerbs", []),
            "imperativeVerbs",
            "imperative",
            {"low": 1, "high": 2},
            {"none": 0, "low": 0.3, "high": 0.5},
        ),
        _score_keyword_match(
            text,
            config.get("constraintIndicators", []),
            "constraintCount",
            "constraints",
            {"low": 1, "high": 3},
            {"none": 0, "low": 0.3, "high": 0.7},
        ),
        _score_keyword_match(
            text,
            config.get("outputFormatKeywords", []),
            "outputFormat",
            "format",
            {"low": 1, "high": 2},
            {"none": 0, "low": 0.4, "high": 0.7},
        ),
        _score_keyword_match(
            text,
            config.get("referenceKeywords", []),
            "referenceComplexity",
            "references",
            {"low": 1, "high": 2},
            {"none": 0, "low": 0.3, "high": 0.5},
        ),
        _score_keyword_match(
            text,
            config.get("negationKeywords", []),
            "negationComplexity",
            "negation",
            {"low": 2, "high": 3},
            {"none": 0, "low": 0.3, "high": 0.5},
        ),
        _score_keyword_match(
            text,
            config.get("domainSpecificKeywords", []),
            "domainSpecificity",
            "domain-specific",
            {"low": 1, "high": 2},
            {"none": 0, "low": 0.5, "high": 0.8},
        ),
    ]

    agentic_dim, agentic_score = _score_agentic_task(text, config.get("agenticTaskKeywords", []))
    dimensions.append(agentic_dim)

    signals = [signal for _, _, signal in dimensions if signal]

    weights = config.get("dimensionWeights", {})
    weighted_score = 0.0
    for name, score, _signal in dimensions:
        weighted_score += float(score) * float(weights.get(name, 0.0))

    reasoning_matches = [kw for kw in config.get("reasoningKeywords", []) if kw.lower() in user_text]
    steepness = float(config.get("confidenceSteepness", 12))

    if len(reasoning_matches) >= 2:
        confidence = _calibrate_confidence(max(weighted_score, 0.3), steepness)
        return ScoringResult(
            score=weighted_score,
            tier="REASONING",
            confidence=max(confidence, 0.85),
            signals=signals,
            agentic_score=agentic_score,
        )

    boundaries = config.get("tierBoundaries", {})
    simple_medium = float(boundaries.get("simpleMedium", 0.0))
    medium_complex = float(boundaries.get("mediumComplex", 0.18))
    complex_reasoning = float(boundaries.get("complexReasoning", 0.4))

    if weighted_score < simple_medium:
        tier = "SIMPLE"
        distance = simple_medium - weighted_score
    elif weighted_score < medium_complex:
        tier = "MEDIUM"
        distance = min(weighted_score - simple_medium, medium_complex - weighted_score)
    elif weighted_score < complex_reasoning:
        tier = "COMPLEX"
        distance = min(weighted_score - medium_complex, complex_reasoning - weighted_score)
    else:
        tier = "REASONING"
        distance = weighted_score - complex_reasoning

    confidence = _calibrate_confidence(distance, steepness)
    threshold = float(config.get("confidenceThreshold", 0.7))
    if confidence < threshold:
        return ScoringResult(
            score=weighted_score,
            tier=None,
            confidence=confidence,
            signals=signals,
            agentic_score=agentic_score,
        )

    return ScoringResult(
        score=weighted_score,
        tier=tier,
        confidence=confidence,
        signals=signals,
        agentic_score=agentic_score,
    )
