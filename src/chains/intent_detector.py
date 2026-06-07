# ══════════════════════════════════════════════════════════
#  src/chains/intent_detector.py
#
#  WHAT THIS FILE DOES:
#  This is the "Intent Detection" box at the top of our
#  architecture diagram — the decision-maker that routes
#  each question to the right chain.
#
#  Question → Intent Detection → Financial / Textual / Mixed
#
#  HOW IT WORKS:
#  1. Keyword scoring: fast, free, no LLM needed
#  2. Pattern matching: catches "Why did X..." type questions
#  3. Entity extraction: finds companies, years, metrics mentioned
#
#  The result tells the chatbot engine WHICH chain to use
#  and WHAT data to fetch (which companies, years, metrics).
# ══════════════════════════════════════════════════════════

import re
from enum import Enum


class QueryType(str, Enum):
    """
    The three branches from our architecture diagram.
    Using str as base class means these print as "FINANCIAL" etc.
    """
    FINANCIAL = "FINANCIAL"   # → SQL Database branch
    TEXTUAL   = "TEXTUAL"     # → Vector DB branch
    MIXED     = "MIXED"       # → Both branches


# ══════════════════════════════════════════════════════════
#  KEYWORD LISTS
#  Words that signal which type of question this is
# ══════════════════════════════════════════════════════════

# Words that strongly suggest a NUMERICAL / FINANCIAL question
FINANCIAL_KEYWORDS = {
    # What to measure
    "revenue", "sales", "income", "profit", "loss", "earnings",
    "assets", "liabilities", "equity", "debt", "cash",
    "margin", "ratio", "return", "eps", "earnings per share",

    # Calculations
    "growth", "change", "increase", "decrease", "grew", "declined",
    "year-over-year", "yoy", "percent", "percentage",

    # Comparisons
    "compare", "comparison", "higher", "lower", "highest", "lowest",
    "most", "least", "largest", "smallest", "rank", "ranking",
    "better", "worse", "outperform",

    # Question starters
    "how much", "what was", "what is", "how many",
    "total", "net", "gross", "operating",

    # Currency / numbers
    "$", "billion", "million", "trillion",

    # Years (will also be caught by regex)
    "2019", "2020", "2021", "2022", "2023", "2024",
}

# Words that strongly suggest a TEXTUAL / QUALITATIVE question
TEXTUAL_KEYWORDS = {
    # Document sections
    "risk", "risks", "risk factors",
    "strategy", "strategic",
    "business model", "business overview",
    "management", "management discussion", "md&a",
    "competition", "competitive", "competitors",

    # Qualitative verbs
    "mention", "discuss", "describe", "explain", "say", "stated",
    "highlight", "note", "warn", "caution",

    # Business concepts
    "market", "industry", "sector", "segment",
    "product", "service", "platform", "technology",
    "employees", "workforce", "talent",
    "regulation", "regulatory", "compliance", "legal",
    "supply chain", "operations", "operational",
    "international", "geographic", "region",
    "future", "outlook", "guidance", "plan", "initiative",
    "opportunity", "challenge", "threat",
    "innovation", "research and development",
}

# Regex patterns that STRONGLY indicate a MIXED question
# (needs numbers AND explanation)
MIXED_PATTERNS = [
    r"\bwhy\s+(did|was|were|has|have)\b",          # "Why did revenue..."
    r"\bwhat\s+(caused|drove|led|contributed)\b",   # "What caused the growth?"
    r"\bhow\s+did\b.{0,30}(revenue|profit|income)", # "How did revenue change?"
    r"\bexplain\b.{0,20}(growth|increase|decline)", # "Explain the growth"
    r"\breason(s)?\s+for\b",                        # "Reasons for the decline"
    r"\bfactor(s)?\s+(behind|driving|affecting)\b", # "Factors driving growth"
    r"\bwhat\s+contributed\b",                      # "What contributed to..."
]


# ══════════════════════════════════════════════════════════
#  ENTITY EXTRACTION
# ══════════════════════════════════════════════════════════

# Map of variations → canonical name
COMPANY_ALIASES = {
    "tesla":     "Tesla",  "tsla":      "Tesla",
    "microsoft": "Microsoft", "msft":   "Microsoft",
    "apple":     "Apple",  "aapl":      "Apple",
}

METRIC_ALIASES = {
    "revenue":            "Revenue",
    "sales":              "Revenue",
    "net income":         "Net Income",
    "earnings":           "Net Income",
    "net profit":         "Net Income",
    "profit":             "Net Income",
    "gross profit":       "Gross Profit",
    "operating income":   "Operating Income",
    "operating profit":   "Operating Income",
    "total assets":       "Total Assets",
    "assets":             "Total Assets",
    "liabilities":        "Total Liabilities",
    "debt":               "Long-Term Debt",
    "long-term debt":     "Long-Term Debt",
    "cash":               "Cash and Equivalents",
    "r&d":                "R&D Expense",
    "research":           "R&D Expense",
    "eps":                "EPS Diluted",
    "earnings per share": "EPS Diluted",
    "gross margin":       "Gross Profit",  # will trigger margin calculation
    "operating margin":   "Operating Income",
}


def extract_companies(question_lower: str) -> list[str]:
    """Find all company names mentioned in the question."""
    found = []
    for alias, canonical in COMPANY_ALIASES.items():
        if alias in question_lower and canonical not in found:
            found.append(canonical)
    return found


def extract_years(question_lower: str) -> list[int]:
    """Find all 4-digit years (2019-2024) mentioned in the question."""
    year_matches = re.findall(r'\b(201[9]|202[0-4])\b', question_lower)
    return sorted(set(int(y) for y in year_matches))


def extract_metrics(question_lower: str) -> list[str]:
    """Find all financial metrics mentioned in the question."""
    found = []
    # Sort by length (longest first) so "net income" matches before "income"
    sorted_aliases = sorted(METRIC_ALIASES.items(), key=lambda x: -len(x[0]))
    for alias, canonical in sorted_aliases:
        if alias in question_lower and canonical not in found:
            found.append(canonical)
    return found


# ══════════════════════════════════════════════════════════
#  MAIN CLASSIFIER
# ══════════════════════════════════════════════════════════

def detect_intent(question: str) -> dict:
    """
    Analyze a question and determine:
      1. What TYPE of question it is (Financial / Textual / Mixed)
      2. Which companies are mentioned
      3. Which years are mentioned
      4. Which financial metrics are mentioned

    Returns a dict that the chatbot engine uses to route the question.

    Example:
        Input:  "Why did Tesla's revenue increase in 2023?"
        Output: {
            "type":      QueryType.MIXED,
            "companies": ["Tesla"],
            "years":     [2023],
            "metrics":   ["Revenue"],
            "confidence": 0.9,
        }
    """
    q = question.lower().strip()

    # ── Step 1: Extract entities ──────────────────────────────
    companies = extract_companies(q)
    years     = extract_years(q)
    metrics   = extract_metrics(q)


    # ── Legal questions → always TEXTUAL (no financial numbers needed) ───
    legal_kws = ["legal proceeding", "legal proceedings", "lawsuit",
                 "litigation", "court", "complaint", "injunction",
                 "antitrust", "settlement", "epic games", "masimo"]
    if any(kw in q for kw in legal_kws):
        return _make_result(QueryType.TEXTUAL, companies, years, metrics, 0.95)

    # ── Product/category breakdown → always TEXTUAL (in MD&A text) ───────────
    category_kws = [
        "by category", "net sales by", "product category",
        "iphone", "ipad", "mac sales", "wearables", "services revenue",
        "segment", "breakdown", "product mix", "by product",
    ]
    if any(kw in q for kw in category_kws):
        return _make_result(QueryType.TEXTUAL, companies, years, metrics, 0.95)

    # ── Step 2: Check MIXED patterns (highest priority) ───────
    for pattern in MIXED_PATTERNS:
        if re.search(pattern, q):
            return _make_result(QueryType.MIXED, companies, years, metrics, 0.9)

    # ── Step 3: Score keywords ────────────────────────────────
    fin_score  = sum(1 for kw in FINANCIAL_KEYWORDS if kw in q)
    text_score = sum(1 for kw in TEXTUAL_KEYWORDS   if kw in q)

    # Boost financial score for explicit number indicators
    if re.search(r'\$|\d+\s*(billion|million|%)', q):
        fin_score += 3

    # Boost financial score if a specific year is mentioned (with a metric)
    if years and metrics:
        fin_score += 2

    # ── Step 4: Determine type ────────────────────────────────
    if fin_score == 0 and text_score == 0:
        # No strong signal — default to MIXED to cover all bases
        query_type = QueryType.MIXED
        confidence = 0.5

    elif fin_score > 0 and text_score > 0:
        # Both types of keywords present → MIXED
        query_type = QueryType.MIXED
        confidence = 0.7

    elif fin_score > text_score:
        query_type = QueryType.FINANCIAL
        confidence = min(1.0, fin_score / 6)

    else:
        query_type = QueryType.TEXTUAL
        confidence = min(1.0, text_score / 5)

    return _make_result(query_type, companies, years, metrics, confidence)


def _make_result(query_type, companies, years, metrics, confidence) -> dict:
    """Helper to build the result dict with defaults applied."""
    # Apply defaults if nothing was found
    if not companies:
        companies = ["Tesla", "Microsoft", "Apple"]  # show all if unspecified

    if not years:
        years = [2023]  # default to most recent year

    if not metrics:
        metrics = ["Revenue"]  # default metric

    return {
        "type":       query_type,
        "companies":  companies,
        "years":      years,
        "metrics":    metrics,
        "confidence": confidence,
    }


# ── Quick test ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    test_questions = [
        "What was Tesla's revenue in 2023?",
        "Compare Apple and Microsoft revenue in 2023",
        "What risks does Apple mention in its 10-K?",
        "Why did Tesla's revenue increase in 2023?",
        "What is Microsoft's business strategy?",
        "Which company had the highest net income in 2022?",
    ]

    print("Intent Detection Test\n" + "="*50)
    for q in test_questions:
        result = detect_intent(q)
        print(f"\nQ: {q}")
        print(f"   Type:      {result['type']}")
        print(f"   Companies: {result['companies']}")
        print(f"   Years:     {result['years']}")
        print(f"   Metrics:   {result['metrics']}")
