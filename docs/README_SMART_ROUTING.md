# Smart Routing Extension

Smart Routing automatically routes `model="auto"` requests to the best available provider and model
based on prompt complexity analysis. It uses a 14-dimension weighted scoring system ported from
[ClawRouter](https://github.com/AIChatBotHub/ClawRouter) to classify prompts into complexity tiers
and select the most cost-effective capable model.

## How It Works

1. A request arrives with `model: "auto"`
2. The scorer analyzes the prompt across 14 dimensions (code presence, reasoning markers, technical terms, etc.)
3. A weighted score maps to a complexity tier: **SIMPLE**, **MEDIUM**, **COMPLEX**, or **REASONING**
4. The router selects the best available provider/model for that tier
5. The request is delegated to the selected provider transparently

The extension registers a virtual provider that claims `model="auto"`. No changes to llmspy core are required.

---

## Scoring Dimensions

| Dimension | Weight | What It Detects |
|-----------|--------|-----------------|
| `reasoningMarkers` | 0.18 | "prove", "theorem", "step by step", "chain of thought" |
| `codePresence` | 0.15 | `function`, `class`, `import`, `def`, code fences |
| `multiStepPatterns` | 0.12 | "first...then", "step 1", numbered lists |
| `technicalTerms` | 0.10 | "algorithm", "kubernetes", "architecture", "distributed" |
| `tokenCount` | 0.08 | Short (<50 tokens) vs long (>500 tokens) |
| `creativeMarkers` | 0.05 | "story", "poem", "brainstorm", "imagine" |
| `questionComplexity` | 0.05 | Multiple question marks (>3) |
| `constraintCount` | 0.04 | "at most", "within", "maximum", "budget" |
| `agenticTask` | 0.04 | "read file", "deploy", "fix", "debug", "step 1" |
| `imperativeVerbs` | 0.03 | "build", "create", "implement", "deploy" |
| `outputFormat` | 0.03 | "json", "yaml", "table", "csv", "markdown" |
| `simpleIndicators` | 0.02 | "what is", "define", "translate", "hello" (negative score) |
| `domainSpecificity` | 0.02 | "quantum", "fpga", "genomics", "homomorphic" |
| `referenceComplexity` | 0.02 | "above", "the docs", "the api", "attached" |
| `negationComplexity` | 0.01 | "don't", "avoid", "never", "without", "except" |

All keyword lists include multilingual support (English, Chinese, Japanese, Russian, German).

## Tier Boundaries

| Tier | Score Range | Typical Prompts |
|------|-------------|-----------------|
| **SIMPLE** | < 0.0 | "What is Python?", "Translate hello to French" |
| **MEDIUM** | 0.0 – 0.18 | "Write a REST API endpoint", "Compare Redis vs Memcached" |
| **COMPLEX** | 0.18 – 0.4 | "Design a distributed cache with consistency guarantees" |
| **REASONING** | > 0.4 | "Prove that sqrt(2) is irrational step by step" |

Special overrides:
- **2+ reasoning keywords** in user prompt → forced **REASONING** (min 0.85 confidence)
- **>100k estimated tokens** → forced **COMPLEX**
- **Structured output** detected in system prompt → minimum **MEDIUM**
- **Confidence < 0.7** → ambiguous, defaults to **MEDIUM**

---

## Default Model Preferences

### Standard Routing

| Tier | Preferred Models |
|------|------------------|
| SIMPLE | `gemini-2.5-flash`, `deepseek-chat`, `gpt-4o-mini` |
| MEDIUM | `deepseek-chat`, `gpt-4o-mini`, `gemini-2.5-flash` |
| COMPLEX | `gemini-2.5-pro`, `claude-sonnet-4`, `gpt-4o` |
| REASONING | `deepseek-reasoner`, `o3-mini`, `gemini-2.5-pro` |

### Agentic Routing (tool use detected)

| Tier | Preferred Models | Required |
|------|------------------|----------|
| SIMPLE | `claude-haiku-4.5`, `gpt-4o-mini`, `gemini-2.5-flash` | `tool_call` |
| MEDIUM | `claude-sonnet-4`, `gpt-4o`, `gemini-2.5-flash` | `tool_call` |
| COMPLEX | `claude-sonnet-4`, `claude-opus-4`, `gpt-4o` | `tool_call` |
| REASONING | `claude-sonnet-4`, `deepseek-reasoner`, `gemini-2.5-pro` | `tool_call` |

If no preferred model is available, the router falls back to the cheapest model within the tier's cost threshold.

---

## Configuration

User configuration is stored at `~/.llms/smart_routing.json`. All fields are optional — unspecified values use defaults.

```json
{
  "scoring": {
    "dimensionWeights": {
      "reasoningMarkers": 0.25,
      "codePresence": 0.15
    },
    "tierBoundaries": {
      "simpleMedium": -0.05,
      "mediumComplex": 0.2,
      "complexReasoning": 0.45
    }
  },
  "overrides": {
    "maxTokensForceComplex": 50000,
    "structuredOutputMinTier": "COMPLEX",
    "ambiguousDefaultTier": "MEDIUM",
    "agenticMode": false
  },
  "tierPreferences": {
    "SIMPLE": {
      "preferred_models": ["gpt-4o-mini", "gemini-2.5-flash"],
      "capabilities": {}
    }
  }
}
```

### Configuration via API

```bash
# Get current config
curl http://localhost:8000/ext/smart_routing/config

# Update config (deep-merged with current)
curl -X POST http://localhost:8000/ext/smart_routing/config \
  -H "Content-Type: application/json" \
  -d '{"overrides": {"agenticMode": true}}'
```

---

## API Endpoints

All endpoints are prefixed with `/ext/smart_routing`.

### `GET /config`
Returns the current routing configuration.

### `POST /config`
Update routing configuration. Accepts a JSON object that is deep-merged with the current config.
Changes are persisted to `~/.llms/smart_routing.json`.

### `GET /stats`
Returns routing statistics since server start.

```json
{
  "total_routed": 142,
  "tiers": {"SIMPLE": 68, "MEDIUM": 45, "COMPLEX": 22, "REASONING": 7},
  "providers": {"google": 71, "deepseek": 38, "anthropic": 33},
  "ambiguous": 3,
  "fallback_attempts": 2,
  "candidate_failures": 1
}
```

---

## Usage

```bash
# Start llmspy with smart routing enabled (extension auto-loads)
llms --serve 8000

# Send a request with model="auto"
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "auto",
    "messages": [{"role": "user", "content": "What is Python?"}]
  }'
```

The response includes routing metadata:

```json
{
  "choices": [...],
  "routing": {
    "tier": "SIMPLE",
    "confidence": 0.92,
    "score": -0.02,
    "provider": "google",
    "model": "gemini-2.5-flash",
    "agentic": false,
    "signals": ["simple (what is)", "short (4 tokens)"],
    "reasoning": "score=-0.02 | simple (what is), short (4 tokens)"
  }
}
```

---

## Extension Files

| File | Description |
|------|-------------|
| `llms/extensions/smart_routing/__init__.py` | Extension entry point, `SmartRouterProvider` class |
| `llms/extensions/smart_routing/scorer.py` | 14-dimension weighted scoring algorithm |
| `llms/extensions/smart_routing/config.py` | Default configuration (keywords, weights, boundaries) |
| `llms/extensions/smart_routing/router.py` | Tier-to-provider mapping and candidate ranking |
