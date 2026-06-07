# ══════════════════════════════════════════════════════════
#  src/chains/rag_chain.py
# ══════════════════════════════════════════════════════════

import os
from dotenv import load_dotenv

from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from src.chains.calculator import build_financial_context
from src.ingestion.vector_store import hybrid_search

load_dotenv()

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
LLM_MODEL       = os.getenv("LLM_MODEL", "llama3.2:3b")


def get_llm() -> ChatOllama:
    return ChatOllama(
        model       = LLM_MODEL,
        base_url    = OLLAMA_BASE_URL,
        temperature = 0.1,
        num_ctx     = 8192,
    )


FINANCIAL_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a precise financial analyst answering questions about SEC 10-K filings.

STRICT RULES:
1. Answer ONLY what was specifically asked — nothing more
2. Never add unsolicited information or background context
3. Format every number exactly as: Metric (Year): $Value
   Example: Revenue (2023): $96.8B
4. For year-over-year always write exactly:
   Metric grew from $Value (prior year) to $Value (current year), an increase/decrease of $Value (+/-%)
   Example: Revenue grew from $81.5B (2022) to $96.8B (2023), an increase of $15.3B (+18.8%)
5. Use plain text only — no asterisks, no markdown symbols, no bold
6. Use these exact section headers:

DIRECT ANSWER
[one or two sentences answering exactly what was asked]

KEY NUMBERS
[only numbers directly relevant to the question]

YEAR OVER YEAR
[only include if growth or change was asked about]"""),

    ("human", """Financial data (from SQL database):

{financial_context}

Question: {question}

Answer following the exact structure above. Answer ONLY what was asked.""")
])


TEXTUAL_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a precise financial analyst answering questions strictly based on SEC 10-K filing excerpts.

STRICT RAG RULES — MUST FOLLOW:
1. Use ONLY the information contained in the provided excerpts below
2. Do NOT use any external knowledge, assumptions, or information not present in the excerpts
3. If the excerpts do not contain enough information to answer the question, explicitly state:
   "The retrieved excerpts do not contain sufficient information to answer this question fully."
4. Every point in your answer must be directly traceable to one of the provided excerpts
5. Use plain text only — no asterisks, no markdown symbols, no bold
6. IMPORTANT: Use ALL relevant excerpts regardless of their year tag in the source.
   Legal proceedings and events described in any excerpt may be relevant to the question year.
   Do NOT skip an excerpt just because its source year differs from the question year.
6. IMPORTANT: Use ALL relevant excerpts regardless of their year tag in the source.
   Legal proceedings and events described in any excerpt may be relevant to the question year.
   Do NOT skip an excerpt just because its source year differs from the question year.

SPECIAL RULE FOR MULTI-YEAR OR MULTI-ITEM COMPARISONS:
If the question asks to compare data across two years OR lists multiple products/segments/items,
present the data as a plain text table using this exact format:

ITEM              | YEAR1 ($M)  | YEAR2 ($M)
------------------|-------------|-------------
Item Name         | $Value      | $Value
Item Name         | $Value      | $Value
TOTAL             | $Value      | $Value

Always state the unit (Millions or Billions) in the column header.
Example: YEAR2023 ($M) means values are in millions.

ANSWER STRUCTURE — use these exact headers with the divider lines:

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SUMMARY
[5 to 6 line concise overview based strictly on the retrieved excerpts.
Write like an analyst executive summary. Cover only what the excerpts contain.]

DATA TABLE
[Only include if question involves multiple items or years side by side.
Use the plain text table format shown above.]

DETAILED EXPLANATION
List every relevant point from the excerpts as a numbered list.
Use this exact format for each point:
1. [Point title]: [Explanation from the filing excerpt. Cite excerpt e.g. (Excerpt 1).]
2. [Point title]: [Explanation from the filing excerpt. Cite excerpt e.g. (Excerpt 2).]
Cover ALL relevant points found across ALL provided excerpts. Do not skip any.

SOURCES
[List every excerpt used, one per line, in this exact format:
Company 10-K (Year) | Part X | Item Y — Section Name]
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"""),

    ("human", """10-K filing excerpts:

{rag_context}

Question: {question}

Answer following the exact structure above using ONLY the provided excerpts.""")
])


MIXED_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a precise financial analyst answering questions about SEC 10-K filings.

STRICT RAG RULES — MUST FOLLOW:
1. For textual insights use ONLY the provided filing excerpts — no external knowledge
2. If excerpts do not contain enough information, state that explicitly
3. Format every number exactly as: Metric (Year): $Value
4. For year-over-year always write exactly:
   Metric grew/declined from $Value in prior year to $Value in current year, an increase/decrease of $Value (+/-%)
5. Use plain text only — no asterisks, no markdown symbols, no bold
6. Always list every source used at the end in this exact format:
   Company 10-K (Year) | Part X | Item Y — Section Name
7. CRITICAL: Use ALL relevant excerpts regardless of their source year tag.
   Legal proceedings, risks and events described in any excerpt are valid answers.
   NEVER say "the excerpt does not provide details" if ANY excerpt mentions the topic.
   Search ALL excerpts carefully before concluding information is missing.

SPECIAL RULE FOR MULTI-YEAR OR MULTI-ITEM COMPARISONS:
If the question asks to compare data across two years OR lists multiple products/segments/items,
present the data as a plain text table using this exact format:

ITEM              | YEAR1 ($M)  | YEAR2 ($M)
------------------|-------------|-------------
Item Name         | $Value      | $Value
Item Name         | $Value      | $Value
TOTAL             | $Value      | $Value

Always state the unit (Millions or Billions) in the column header.
Example: YEAR2023 ($M) means values are in millions.

ANSWER STRUCTURE:

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
DIRECT ANSWER
[one or two sentences answering exactly what was asked]

DATA TABLE
[Only include if question involves multiple items or years side by side.
Use the plain text table format shown above.]

KEY NUMBERS
[only numbers directly relevant to the question]
Metric (Year): $Value

KEY INSIGHTS FROM FILING
[insights strictly from the retrieved excerpts only]
- [insight from excerpt — cite excerpt number]
- [insight from excerpt — cite excerpt number]

SOURCES
[one source per line:
Company 10-K (Year) | Part X | Item Y — Section Name]
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"""),

    ("human", """Financial data (from SQL):
{financial_context}

10-K filing excerpts:
{rag_context}

Question: {question}

Answer following the exact structure above. Answer ONLY what was asked.""")
])


def build_rag_context_string(documents: list) -> str:
    """
    Format retrieved documents with clean source citations.
    Format: Company 10-K (Year) | Part X | Item Y — Section Name
    No passage numbers.
    """
    if not documents:
        return "No relevant filing excerpts found."

    lines = []
    for i, doc in enumerate(documents, 1):
        meta = doc.metadata

        company = meta.get("company",     "Unknown")
        year    = meta.get("year",        "?")
        part    = meta.get("part_number", "")
        item    = meta.get("item_number", "")
        section = meta.get("section",     "")

        # Build clean source — skip empty or Generic fields
        source_parts = [f"{company} 10-K ({year})"]
        if part and part != "General":
            source_parts.append(part)
        if item and item != "General":
            source_parts.append(f"{item} — {section}" if section and section != "General" else item)
        elif section and section != "General":
            source_parts.append(section)

        source = " | ".join(source_parts)

        preview = doc.page_content[:800]
        if len(doc.page_content) > 800:
            preview += "..."

        lines.append(f"[Excerpt {i}]")
        lines.append(f"Source: {source}")
        lines.append(preview)
        lines.append("")

    return "\n".join(lines)


class FinancialChain:
    def __init__(self):
        self.llm   = get_llm()
        self.chain = FINANCIAL_PROMPT | self.llm | StrOutputParser()

    def run(self, question, companies, metrics, years):
        financial_context = build_financial_context(
            companies, metrics, years, question
        )
        return self.chain.stream({
            "financial_context": financial_context,
            "question":          question,
        })


class TextualChain:
    def __init__(self):
        self.llm   = get_llm()
        self.chain = TEXTUAL_PROMPT | self.llm | StrOutputParser()

    def run(self, question, companies, years):
        all_docs = []
        if len(companies) == 1:
            all_docs = hybrid_search(
                question, k=8, company_filter=companies[0]
            )
        else:
            for company in companies:
                all_docs.extend(
                    hybrid_search(question, k=5, company_filter=company)
                )
        rag_context = build_rag_context_string(all_docs)
        return self.chain.stream({
            "rag_context": rag_context,
            "question":    question,
        })


class MixedChain:
    def __init__(self):
        self.llm   = get_llm()
        self.chain = MIXED_PROMPT | self.llm | StrOutputParser()

    def run(self, question, companies, metrics, years):
        financial_context = build_financial_context(
            companies, metrics, years, question
        )
        all_docs = []
        for company in companies:
            all_docs.extend(
                hybrid_search(question, k=4, company_filter=company)
            )
        rag_context = build_rag_context_string(all_docs)
        return self.chain.stream({
            "financial_context": financial_context,
            "rag_context":       rag_context,
            "question":          question,
        })
