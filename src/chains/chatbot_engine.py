# ══════════════════════════════════════════════════════════
#  src/chains/chatbot_engine.py
#
#  WHAT THIS FILE DOES:
#  This is the MASTER CONTROLLER that implements the full
#  architecture diagram from top to bottom:
#
#  User Question
#       ↓
#  Intent Detection  ← detect_intent()
#       ↓
#  ┌────────────────────┬──────────────┐
#  │                    │              │
#  Financial         Textual         Mixed
#  │                    │              │
#  SQL Calculations  Vector DB       Both
#  │                    │              │
#  └────────────────────┴──────────────┘
#       ↓
#  LLM Explanation  ← Ollama via LangChain
#       ↓
#  Streamed Response
# ══════════════════════════════════════════════════════════

from src.chains.intent_detector import detect_intent, QueryType
from src.chains.rag_chain import FinancialChain, TextualChain, MixedChain


# Create chain instances once (reused for all questions)
# This avoids re-initialising the LLM connection on every question
_financial_chain = None
_textual_chain   = None
_mixed_chain     = None


def _get_chains():
    """
    Lazy initialisation of chains.
    First call creates them; subsequent calls reuse them.
    This speeds up the app after the first question.
    """
    global _financial_chain, _textual_chain, _mixed_chain

    if _financial_chain is None:
        print("Initialising LangChain chains...")
        _financial_chain = FinancialChain()
        _textual_chain   = TextualChain()
        _mixed_chain     = MixedChain()
        print("Chains ready.")

    return _financial_chain, _textual_chain, _mixed_chain


def ask(question: str, stream: bool = True):
    """
    Process a user question through the full architecture pipeline.

    This is the ONE function the Streamlit UI calls.
    Everything else is handled internally.

    Args:
        question: The user's natural language question
        stream:   If True, yields text chunks as they generate
                  If False, returns the complete response as a string

    Yields (stream=True) or Returns (stream=False):
        Text chunks from the LLM

    Example:
        for chunk in ask("What was Tesla's revenue in 2023?"):
            print(chunk, end="", flush=True)
    """
    # ── Step 1: Intent Detection ──────────────────────────────
    # Analyze the question to determine type and extract entities
    intent    = detect_intent(question)
    qtype     = intent["type"]
    companies = intent["companies"]
    years     = intent["years"]
    metrics   = intent["metrics"]

    # ── Step 2: Route to the correct chain ───────────────────
    fin_chain, text_chain, mix_chain = _get_chains()

    if qtype == QueryType.FINANCIAL:
        # → SQL Database → Python Calculations → LLM
        generator = fin_chain.run(
            question  = question,
            companies = companies,
            metrics   = metrics,
            years     = years,
        )

    elif qtype == QueryType.TEXTUAL:
        # → Vector DB → LLM
        generator = text_chain.run(
            question  = question,
            companies = companies,
            years     = years,
        )

    else:  # MIXED
        # → Both SQL and Vector DB → LLM
        generator = mix_chain.run(
            question  = question,
            companies = companies,
            metrics   = metrics,
            years     = years,
        )

    # ── Step 3: Stream or collect the response ────────────────
    if stream:
        # Yield each text chunk as it arrives from Ollama
        return generator
    else:
        # Collect all chunks into one string
        return "".join(generator)


def get_intent_info(question: str) -> dict:
    """
    Public helper for the UI to show routing information.
    Returns the intent detection result for display purposes.
    """
    return detect_intent(question)
