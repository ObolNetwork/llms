#!/usr/bin/env python3
"""
Unit tests for smart routing extension behavior.
"""

import asyncio
import os
import sys
import unittest

# Add parent directory to path to import llms module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from llms.extensions.smart_routing import SmartRouterProvider, normalize_config
from llms.extensions.smart_routing.config import DEFAULT_SCORING_CONFIG
from llms.extensions.smart_routing.scorer import classify


class DummyProvider:
    def __init__(self, provider_id, models, fail_calls=0):
        self.id = provider_id
        self.name = provider_id
        self.models = models
        self._fail_calls = fail_calls
        self.calls = 0

    def provider_model(self, model):
        model_lower = str(model).lower()
        for model_id, model_info in self.models.items():
            candidate_id = str(model_info.get("id") or model_id)
            if model_lower in (str(model_id).lower(), candidate_id.lower()):
                return candidate_id
            if "/" in candidate_id and candidate_id.split("/")[-1].lower() == model_lower:
                return candidate_id
        return None

    def model_info(self, model):
        resolved = self.provider_model(model)
        if not resolved:
            return None
        for model_id, model_info in self.models.items():
            candidate_id = str(model_info.get("id") or model_id)
            if candidate_id.lower() == resolved.lower() or str(model_id).lower() == resolved.lower():
                return model_info
        return None

    async def chat(self, chat, context=None):
        self.calls += 1
        if self._fail_calls > 0:
            self._fail_calls -= 1
            raise Exception("%s forced failure" % self.id)
        return {
            "choices": [{"message": {"role": "assistant", "content": "ok"}}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        }


class DummyCtx:
    def __init__(self, providers):
        self._providers = providers
        self.logs = []

    def get_providers(self):
        return self._providers

    def log(self, message):
        self.logs.append(message)


class TestSmartRoutingScorer(unittest.TestCase):
    def test_simple_prompt_classified_simple(self):
        result = classify("What is Python?", None, 4, DEFAULT_SCORING_CONFIG)
        self.assertEqual(result.tier, "SIMPLE")

    def test_reasoning_override(self):
        result = classify("Prove the theorem step by step", None, 10, DEFAULT_SCORING_CONFIG)
        self.assertEqual(result.tier, "REASONING")
        self.assertGreaterEqual(result.confidence, 0.85)


class TestSmartRoutingConfig(unittest.TestCase):
    def test_partial_scoring_override_preserves_defaults(self):
        config = normalize_config({"scoring": {"dimensionWeights": {"reasoningMarkers": 0.25}}})
        self.assertEqual(config["scoring"]["dimensionWeights"]["reasoningMarkers"], 0.25)
        self.assertIn("tokenCount", config["scoring"]["dimensionWeights"])
        self.assertIn("tokenCountThresholds", config["scoring"])
        self.assertIn("codeKeywords", config["scoring"])


class TestSmartRoutingProvider(unittest.TestCase):
    def test_provider_fallback_when_first_candidate_fails(self):
        async def run_test():
            provider_a = DummyProvider(
                "provider_a",
                {"gemini-2.5-flash": {"id": "gemini-2.5-flash", "cost": {"input": 0.1, "output": 0.3}}},
                fail_calls=1,
            )
            provider_b = DummyProvider(
                "provider_b",
                {"gemini-2.5-flash": {"id": "gemini-2.5-flash", "cost": {"input": 0.2, "output": 0.3}}},
            )
            ctx = DummyCtx({"provider_a": provider_a, "provider_b": provider_b})
            smart = SmartRouterProvider(ctx)
            chat = {"model": "auto", "messages": [{"role": "user", "content": "What is Python?"}]}
            context = {}

            response = await smart.chat(chat, context=context)
            self.assertEqual(response["routing"]["provider"], "provider_b")
            self.assertEqual(response["routing"]["attempts"], 2)
            self.assertEqual(context["_smart_routing_selection"]["provider"], "provider_b")

        asyncio.run(run_test())

    def test_provider_selection_is_pinned_within_context(self):
        async def run_test():
            provider_a = DummyProvider(
                "provider_a",
                {"flash": {"id": "flash", "cost": {"input": 0.1, "output": 0.2}}},
            )
            provider_b = DummyProvider(
                "provider_b",
                {"pro": {"id": "pro", "cost": {"input": 5.0, "output": 10.0}}},
            )
            ctx = DummyCtx({"provider_a": provider_a, "provider_b": provider_b})
            smart = SmartRouterProvider(
                ctx,
                {
                    "tierPreferences": {
                        "SIMPLE": {"preferred_models": ["flash"], "capabilities": {}},
                        "REASONING": {"preferred_models": ["pro"], "capabilities": {}},
                    }
                },
            )
            context = {}
            chat = {"model": "auto", "messages": [{"role": "user", "content": "What is Python?"}]}

            first = await smart.chat(chat, context=context)
            self.assertEqual(first["routing"]["provider"], "provider_a")
            self.assertIn("_smart_routing_selection", context)

            chat["messages"].append({"role": "user", "content": "Prove the theorem step by step"})
            second = await smart.chat(chat, context=context)
            self.assertEqual(second["routing"]["provider"], "provider_a")
            self.assertTrue(second["routing"]["pinned"])

        asyncio.run(run_test())


if __name__ == "__main__":
    unittest.main()
