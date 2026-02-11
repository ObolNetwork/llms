"""
Smart Routing Configuration

Default scoring config ported from ClawRouter's router/config.ts.
All routing parameters: keyword lists, dimension weights, tier boundaries,
confidence calibration, and tier-to-model preferences.
"""

from copy import deepcopy

TIERS = ("SIMPLE", "MEDIUM", "COMPLEX", "REASONING")

DEFAULT_SCORING_CONFIG = {
    "tokenCountThresholds": {"simple": 50, "complex": 500},

    # Multilingual keywords: EN + ZH + JP + RU + DE
    "codeKeywords": [
        "function", "class", "import", "def", "select", "async", "await",
        "const", "let", "var", "return", "```",
        "函数", "类", "导入", "定义", "查询", "异步", "等待", "常量", "变量", "返回",
        "関数", "クラス", "インポート", "非同期", "定数", "変数",
        "функция", "класс", "импорт", "определ", "запрос", "асинхронный",
        "ожидать", "константа", "переменная", "вернуть",
        "funktion", "klasse", "importieren", "definieren", "abfrage",
        "asynchron", "erwarten", "konstante", "variable", "zurückgeben",
    ],
    "reasoningKeywords": [
        "prove", "theorem", "derive", "step by step", "chain of thought",
        "formally", "mathematical", "proof", "logically",
        "证明", "定理", "推导", "逐步", "思维链", "形式化", "数学", "逻辑",
        "証明", "定理", "導出", "ステップバイステップ", "論理的",
        "доказать", "докажи", "доказательств", "теорема", "вывести",
        "шаг за шагом", "пошагово", "поэтапно", "цепочка рассуждений",
        "рассуждени", "формально", "математически", "логически",
        "beweisen", "beweis", "theorem", "ableiten", "schritt für schritt",
        "gedankenkette", "formal", "mathematisch", "logisch",
    ],
    "simpleKeywords": [
        "what is", "define", "translate", "hello", "yes or no", "capital of",
        "how old", "who is", "when was",
        "什么是", "定义", "翻译", "你好", "是否", "首都", "多大", "谁是", "何时",
        "とは", "定義", "翻訳", "こんにちは", "はいかいいえ", "首都", "誰",
        "что такое", "определение", "перевести", "переведи", "привет",
        "да или нет", "столица", "сколько лет", "кто такой", "когда", "объясни",
        "was ist", "definiere", "übersetze", "hallo", "ja oder nein",
        "hauptstadt", "wie alt", "wer ist", "wann", "erkläre",
    ],
    "technicalKeywords": [
        "algorithm", "optimize", "architecture", "distributed", "kubernetes",
        "microservice", "database", "infrastructure",
        "算法", "优化", "架构", "分布式", "微服务", "数据库", "基础设施",
        "アルゴリズム", "最適化", "アーキテクチャ", "分散", "マイクロサービス", "データベース",
        "алгоритм", "оптимизировать", "оптимизаци", "оптимизируй", "архитектура",
        "распределённый", "микросервис", "база данных", "инфраструктура",
        "algorithmus", "optimieren", "architektur", "verteilt", "kubernetes",
        "mikroservice", "datenbank", "infrastruktur",
    ],
    "creativeKeywords": [
        "story", "poem", "compose", "brainstorm", "creative", "imagine", "write a",
        "故事", "诗", "创作", "头脑风暴", "创意", "想象", "写一个",
        "物語", "詩", "作曲", "ブレインストーム", "創造的", "想像",
        "история", "рассказ", "стихотворение", "сочинить", "сочини",
        "мозговой штурм", "творческий", "представить", "придумай", "напиши",
        "geschichte", "gedicht", "komponieren", "brainstorming", "kreativ",
        "vorstellen", "schreibe", "erzählung",
    ],
    "imperativeVerbs": [
        "build", "create", "implement", "design", "develop", "construct",
        "generate", "deploy", "configure", "set up",
        "构建", "创建", "实现", "设计", "开发", "生成", "部署", "配置", "设置",
        "構築", "作成", "実装", "設計", "開発", "生成", "デプロイ", "設定",
        "построить", "построй", "создать", "создай", "реализовать", "реализуй",
        "спроектировать", "разработать", "разработай", "сконструировать",
        "сгенерировать", "сгенерируй", "развернуть", "разверни", "настроить", "настрой",
        "erstellen", "bauen", "implementieren", "entwerfen", "entwickeln",
        "konstruieren", "generieren", "bereitstellen", "konfigurieren", "einrichten",
    ],
    "constraintIndicators": [
        "under", "at most", "at least", "within", "no more than", "o(",
        "maximum", "minimum", "limit", "budget",
        "不超过", "至少", "最多", "在内", "最大", "最小", "限制", "预算",
        "以下", "最大", "最小", "制限", "予算",
        "не более", "не менее", "как минимум", "в пределах", "максимум",
        "минимум", "ограничение", "бюджет",
        "höchstens", "mindestens", "innerhalb", "nicht mehr als",
        "maximal", "minimal", "grenze", "budget",
    ],
    "outputFormatKeywords": [
        "json", "yaml", "xml", "table", "csv", "markdown", "schema",
        "format as", "structured",
        "表格", "格式化为", "结构化",
        "テーブル", "フォーマット", "構造化",
        "таблица", "форматировать как", "структурированный",
        "tabelle", "formatieren als", "strukturiert",
    ],
    "referenceKeywords": [
        "above", "below", "previous", "following", "the docs", "the api",
        "the code", "earlier", "attached",
        "上面", "下面", "之前", "接下来", "文档", "代码", "附件",
        "上記", "下記", "前の", "次の", "ドキュメント", "コード",
        "выше", "ниже", "предыдущий", "следующий", "документация", "код",
        "ранее", "вложение",
        "oben", "unten", "vorherige", "folgende", "dokumentation", "der code",
        "früher", "anhang",
    ],
    "negationKeywords": [
        "don't", "do not", "avoid", "never", "without", "except", "exclude",
        "no longer",
        "不要", "避免", "从不", "没有", "除了", "排除",
        "しないで", "避ける", "決して", "なしで", "除く",
        "не делай", "не надо", "нельзя", "избегать", "никогда", "без",
        "кроме", "исключить", "больше не",
        "nicht", "vermeide", "niemals", "ohne", "außer", "ausschließen", "nicht mehr",
    ],
    "domainSpecificKeywords": [
        "quantum", "fpga", "vlsi", "risc-v", "asic", "photonics", "genomics",
        "proteomics", "topological", "homomorphic", "zero-knowledge", "lattice-based",
        "量子", "光子学", "基因组学", "蛋白质组学", "拓扑", "同态", "零知识", "格密码",
        "量子", "フォトニクス", "ゲノミクス", "トポロジカル",
        "квантовый", "фотоника", "геномика", "протеомика", "топологический",
        "гомоморфный", "с нулевым разглашением", "на основе решёток",
        "quanten", "photonik", "genomik", "proteomik", "topologisch",
        "homomorph", "zero-knowledge", "gitterbasiert",
    ],
    "agenticTaskKeywords": [
        "read file", "read the file", "look at", "check the", "open the",
        "edit", "modify", "update the", "change the", "write to", "create file",
        "execute", "deploy", "install", "npm", "pip", "compile",
        "after that", "and also", "once done", "step 1", "step 2",
        "fix", "debug", "until it works", "keep trying", "iterate",
        "make sure", "verify", "confirm",
        "读取文件", "查看", "打开", "编辑", "修改", "更新", "创建",
        "执行", "部署", "安装", "第一步", "第二步", "修复", "调试",
        "直到", "确认", "验证",
    ],

    # Dimension weights (sum to ~1.0)
    "dimensionWeights": {
        "tokenCount": 0.08,
        "codePresence": 0.15,
        "reasoningMarkers": 0.18,
        "technicalTerms": 0.10,
        "creativeMarkers": 0.05,
        "simpleIndicators": 0.02,
        "multiStepPatterns": 0.12,
        "questionComplexity": 0.05,
        "imperativeVerbs": 0.03,
        "constraintCount": 0.04,
        "outputFormat": 0.03,
        "referenceComplexity": 0.02,
        "negationComplexity": 0.01,
        "domainSpecificity": 0.02,
        "agenticTask": 0.04,
    },

    # Tier boundaries on weighted score axis
    "tierBoundaries": {
        "simpleMedium": 0.0,
        "mediumComplex": 0.18,
        "complexReasoning": 0.4,
    },

    "confidenceSteepness": 12,
    "confidenceThreshold": 0.7,
}

# Tier-to-model preferences (model short names matched against llmspy providers)
DEFAULT_TIER_PREFERENCES = {
    "SIMPLE": {
        "preferred_models": [
            "gemini-2.5-flash", "deepseek-chat", "gpt-4o-mini",
        ],
        "capabilities": {},
    },
    "MEDIUM": {
        "preferred_models": [
            "deepseek-chat", "gpt-4o-mini", "gemini-2.5-flash",
        ],
        "capabilities": {},
    },
    "COMPLEX": {
        "preferred_models": [
            "gemini-2.5-pro", "claude-sonnet-4", "gpt-4o",
        ],
        "capabilities": {},
    },
    "REASONING": {
        "preferred_models": [
            "deepseek-reasoner", "o3-mini", "gemini-2.5-pro",
        ],
        "capabilities": {"reasoning": True},
    },
}

# Agentic tier preferences (when tool use or agentic patterns detected)
DEFAULT_AGENTIC_PREFERENCES = {
    "SIMPLE": {
        "preferred_models": [
            "claude-haiku-4.5", "gpt-4o-mini", "gemini-2.5-flash",
        ],
        "capabilities": {"tool_call": True},
    },
    "MEDIUM": {
        "preferred_models": [
            "claude-sonnet-4", "gpt-4o", "gemini-2.5-flash",
        ],
        "capabilities": {"tool_call": True},
    },
    "COMPLEX": {
        "preferred_models": [
            "claude-sonnet-4", "claude-opus-4", "gpt-4o",
        ],
        "capabilities": {"tool_call": True},
    },
    "REASONING": {
        "preferred_models": [
            "claude-sonnet-4", "deepseek-reasoner", "gemini-2.5-pro",
        ],
        "capabilities": {"tool_call": True},
    },
}

DEFAULT_OVERRIDES = {
    "maxTokensForceComplex": 100_000,
    "structuredOutputMinTier": "MEDIUM",
    "ambiguousDefaultTier": "MEDIUM",
    "agenticMode": False,
}

TIER_RANK = {"SIMPLE": 0, "MEDIUM": 1, "COMPLEX": 2, "REASONING": 3}

# Cost thresholds per tier (input cost per 1M tokens) for fallback selection
TIER_COST_THRESHOLDS = {
    "SIMPLE": 1.0,
    "MEDIUM": 5.0,
    "COMPLEX": 20.0,
    "REASONING": 50.0,
}

DEFAULT_CONFIG = {
    "scoring": DEFAULT_SCORING_CONFIG,
    "overrides": DEFAULT_OVERRIDES,
    "tierPreferences": DEFAULT_TIER_PREFERENCES,
    "agenticPreferences": DEFAULT_AGENTIC_PREFERENCES,
}


def default_config():
    return deepcopy(DEFAULT_CONFIG)
