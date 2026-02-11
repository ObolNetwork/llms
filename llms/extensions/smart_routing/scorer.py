"""
14-Dimension Weighted Scoring Classifier

Ported from ClawRouter's router/rules.ts.
Scores a request across 14 weighted dimensions and maps the aggregate
score to a tier using configurable boundaries. Confidence is calibrated
via sigmoid.
"""

import math
import re
from dataclasses import dataclass, field


@dataclass
class ScoringResult:
    score: float
    tier: str | None  # "SIMPLE"|"MEDIUM"|"COMPLEX"|"REASONING"|None (ambiguous)
    confidence: float
    signals: list[str] = field(default_factory=list)
    agentic_score: float = 0.0


def _score_token_count(estimated_tokens, thresholds):
    if estimated_tokens < thresholds["simple"]:
        return "tokenCount", -1.0, f"short ({estimated_tokens} tokens)"
    if estimated_tokens > thresholds["complex"]:
        return "tokenCount", 1.0, f"long ({estimated_tokens} tokens)"
    return "tokenCount", 0.0, None


def _score_keyword_match(text, keywords, name, signal_label, thresholds, scores):
    matches = [kw for kw in keywords if kw.lower() in text]
    if len(matches) >= thresholds["high"]:
        return name, scores["high"], f"{signal_label} ({', '.join(matches[:3])})"
    if len(matches) >= thresholds["low"]:
        return name, scores["low"], f"{signal_label} ({', '.join(matches[:3])})"
    return name, scores["none"], None


def _score_multi_step(text):
    patterns = [r"first.*then", r"step \d", r"\d\.\s"]
    hits = [p for p in patterns if re.search(p, text, re.IGNORECASE)]
    if hits:
        return "multiStepPatterns", 0.5, "multi-step"
    return "multiStepPatterns", 0.0, None


def _score_question_complexity(prompt):
    count = prompt.count("?")
    if count > 3:
        return "questionComplexity", 0.5, f"{count} questions"
    return "questionComplexity", 0.0, None


def _score_agentic_task(text, keywords):
    match_count = 0
    signals = []
    for kw in keywords:
        if kw.lower() in text:
            match_count += 1
            if len(signals) < 3:
                signals.append(kw)

    if match_count >= 4:
        return ("agenticTask", 1.0, f"agentic ({', '.join(signals)})"), 1.0
    if match_count >= 3:
        return ("agenticTask", 0.6, f"agentic ({', '.join(signals)})"), 0.6
    if match_count >= 1:
        return ("agenticTask", 0.2, f"agentic-light ({', '.join(signals)})"), 0.2
    return ("agenticTask", 0.0, None), 0.0


def _calibrate_confidence(distance, steepness):
    return 1.0 / (1.0 + math.exp(-steepness * distance))


def classify(prompt, system_prompt, estimated_tokens, config):
    """
    Classify a prompt into a complexity tier.

    Args:
        prompt: The user's prompt text
        system_prompt: Optional system prompt (or None)
        estimated_tokens: Estimated token count for the full text
        config: Scoring config dict (DEFAULT_SCORING_CONFIG)

    Returns:
        ScoringResult with tier, confidence, signals, and agentic_score
    """
    text = f"{system_prompt or ''} {prompt}".lower()
    user_text = prompt.lower()

    # Score all 14 dimensions
    dimensions = [
        _score_token_count(estimated_tokens, config["tokenCountThresholds"]),
        _score_keyword_match(
            text, config["codeKeywords"],
            "codePresence", "code",
            {"low": 1, "high": 2}, {"none": 0, "low": 0.5, "high": 1.0},
        ),
        # Reasoning markers: user prompt only
        _score_keyword_match(
            user_text, config["reasoningKeywords"],
            "reasoningMarkers", "reasoning",
            {"low": 1, "high": 2}, {"none": 0, "low": 0.7, "high": 1.0},
        ),
        _score_keyword_match(
            text, config["technicalKeywords"],
            "technicalTerms", "technical",
            {"low": 2, "high": 4}, {"none": 0, "low": 0.5, "high": 1.0},
        ),
        _score_keyword_match(
            text, config["creativeKeywords"],
            "creativeMarkers", "creative",
            {"low": 1, "high": 2}, {"none": 0, "low": 0.5, "high": 0.7},
        ),
        _score_keyword_match(
            text, config["simpleKeywords"],
            "simpleIndicators", "simple",
            {"low": 1, "high": 2}, {"none": 0, "low": -1.0, "high": -1.0},
        ),
        _score_multi_step(text),
        _score_question_complexity(prompt),
        _score_keyword_match(
            text, config["imperativeVerbs"],
            "imperativeVerbs", "imperative",
            {"low": 1, "high": 2}, {"none": 0, "low": 0.3, "high": 0.5},
        ),
        _score_keyword_match(
            text, config["constraintIndicators"],
            "constraintCount", "constraints",
            {"low": 1, "high": 3}, {"none": 0, "low": 0.3, "high": 0.7},
        ),
        _score_keyword_match(
            text, config["outputFormatKeywords"],
            "outputFormat", "format",
            {"low": 1, "high": 2}, {"none": 0, "low": 0.4, "high": 0.7},
        ),
        _score_keyword_match(
            text, config["referenceKeywords"],
            "referenceComplexity", "references",
            {"low": 1, "high": 2}, {"none": 0, "low": 0.3, "high": 0.5},
        ),
        _score_keyword_match(
            text, config["negationKeywords"],
            "negationComplexity", "negation",
            {"low": 2, "high": 3}, {"none": 0, "low": 0.3, "high": 0.5},
        ),
        _score_keyword_match(
            text, config["domainSpecificKeywords"],
            "domainSpecificity", "domain-specific",
            {"low": 1, "high": 2}, {"none": 0, "low": 0.5, "high": 0.8},
        ),
    ]

    # Score agentic task indicators
    agentic_dim, agentic_score = _score_agentic_task(text, config["agenticTaskKeywords"])
    dimensions.append(agentic_dim)

    # Collect signals
    signals = [sig for _, _, sig in dimensions if sig is not None]

    # Compute weighted score
    weights = config["dimensionWeights"]
    weighted_score = 0.0
    for name, score, _ in dimensions:
        w = weights.get(name, 0)
        weighted_score += score * w

    # Reasoning override: 2+ reasoning markers in user prompt → force REASONING
    reasoning_matches = [kw for kw in config["reasoningKeywords"] if kw.lower() in user_text]
    if len(reasoning_matches) >= 2:
        confidence = _calibrate_confidence(max(weighted_score, 0.3), config["confidenceSteepness"])
        return ScoringResult(
            score=weighted_score,
            tier="REASONING",
            confidence=max(confidence, 0.85),
            signals=signals,
            agentic_score=agentic_score,
        )

    # Map weighted score to tier using boundaries
    boundaries = config["tierBoundaries"]
    simple_medium = boundaries["simpleMedium"]
    medium_complex = boundaries["mediumComplex"]
    complex_reasoning = boundaries["complexReasoning"]

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

    # Calibrate confidence via sigmoid
    confidence = _calibrate_confidence(distance, config["confidenceSteepness"])

    # Below threshold → ambiguous
    if confidence < config["confidenceThreshold"]:
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
