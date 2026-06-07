# ══════════════════════════════════════════════════════════
#  src/ingestion/text_processor.py
# ══════════════════════════════════════════════════════════

import re
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

SECTION_MAP = [
    (r"item\s+1a[\.\s]+risk\s+factors",
        "Part I",  "Item 1A", "Risk Factors"),
    (r"item\s+1b[\.\s]+unresolved",
        "Part I",  "Item 1B", "Unresolved Staff Comments"),
    (r"item\s+1[\.\s]+business",
        "Part I",  "Item 1",  "Business Overview"),
    (r"item\s+2[\.\s]+properties",
        "Part I",  "Item 2",  "Properties"),
    (r"item\s+3[\.\s]+legal",
        "Part I",  "Item 3",  "Legal Proceedings"),
    (r"legal\s+proceedings",
        "Part I",  "Item 3",  "Legal Proceedings"),
    (r"epic\s+games|masimo|antitrust|litigation|lawsuit|injunction",
        "Part I",  "Item 3",  "Legal Proceedings"),
    (r"item\s+7a[\.\s]+quantitative",
        "Part II", "Item 7A", "Quantitative Risk"),
    (r"item\s+7[\.\s]+management",
        "Part II", "Item 7",  "MD&A"),
    (r"item\s+8[\.\s]+financial",
        "Part II", "Item 8",  "Financial Statements"),
    (r"item\s+9a[\.\s]+controls",
        "Part II", "Item 9A", "Controls and Procedures"),
    (r"item\s+9[\.\s]+changes",
        "Part II", "Item 9",  "Changes in Accountant"),
]


def detect_section(char_position: int, full_text_lower: str) -> tuple:
    window_start = max(0, char_position - 4000)
    window       = full_text_lower[window_start:char_position]
    best         = ("Part I", "Item 1", "General")
    best_pos     = -1

    for pattern, part, item, name in SECTION_MAP:
        for match in re.finditer(pattern, window, re.IGNORECASE):
            if match.start() > best_pos:
                best_pos = match.start()
                best     = (part, item, name)
    return best


def create_langchain_documents(
    text:          str,
    company:       str,
    year:          int,
    chunk_size:    int = 1000,
    chunk_overlap: int = 150,
) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size      = chunk_size,
        chunk_overlap   = chunk_overlap,
        length_function = len,
        separators      = ["\n\n", "\n", ". ", " ", ""],
    )

    raw_chunks  = splitter.split_text(text)
    text_lower  = text.lower()
    documents   = []
    current_pos = 0

    for chunk_idx, chunk_text in enumerate(raw_chunks):
        if len(chunk_text.strip()) < 80:
            continue

        found_pos = text.find(chunk_text[:50], current_pos)
        if found_pos >= 0:
            current_pos = found_pos

        part, item, section = detect_section(current_pos, text_lower)

        chunk_id = (
            f"{company}_{year}_"
            f"{item.replace(' ','')}_"
            f"{section.replace(' ','_')}_"
            f"{chunk_idx}"
        )

        source = f"{company} 10-K ({year}) | {part} | {item} — {section}"

        doc = Document(
            page_content = chunk_text,
            metadata = {
                "company":      company,
                "year":         year,
                "part_number":  part,
                "item_number":  item,
                "section":      section,
                "source":       source,
                "chunk_id":     chunk_id,
                "chunk_index":  chunk_idx,
            }
        )
        documents.append(doc)

    print(
        f"    {company} {year}: "
        f"{len(raw_chunks)} chunks → {len(documents)} documents"
    )
    return documents


def process_all_filings(filings: dict) -> list[Document]:
    all_documents = []

    for company, year_texts in filings.items():
        print(f"\n  Processing {company}:")
        for year, text in year_texts.items():
            if not text:
                continue
            docs = create_langchain_documents(text, company, year)
            all_documents.extend(docs)

    sections = {}
    for doc in all_documents:
        s = doc.metadata.get("section", "Unknown")
        sections[s] = sections.get(s, 0) + 1

    print(f"\n  Total documents: {len(all_documents)}")
    print("  By section:")
    for section, count in sorted(sections.items(), key=lambda x: -x[1]):
        print(f"    {section}: {count} chunks")

    return all_documents
