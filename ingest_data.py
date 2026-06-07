# ══════════════════════════════════════════════════════════
#  ingest_data.py  —  Run this ONCE before starting the app
#
#  This script runs the full data pipeline:
#    Step 1: Download XBRL financial data from SEC EDGAR
#    Step 2: Save to SQLite database
#    Step 3: Download 10-K filing text from SEC EDGAR
#    Step 4: Split text into chunks (LangChain)
#    Step 5: Generate embeddings (Ollama nomic-embed-text)
#    Step 6: Save embeddings to ChromaDB
#
#  Usage:
#    python ingest_data.py              # Full pipeline
#    python ingest_data.py --xbrl-only  # Only steps 1-2 (no Ollama needed)
#    python ingest_data.py --years 2023 # Only fetch year 2023
# ══════════════════════════════════════════════════════════

import sys
import os
import argparse
import requests

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule

console = Console()


def verify_ollama_ready():
    """Check Ollama is running and has both required models before embedding."""
    from dotenv import load_dotenv
    load_dotenv()

    base        = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    embed_model = os.getenv("EMBEDDING_MODEL", "nomic-embed-text")

    console.print("\n[dim]Checking Ollama...[/dim]")

    try:
        resp   = requests.get(f"{base}/api/tags", timeout=5)
        models = [m["name"] for m in resp.json().get("models", [])]
    except Exception:
        console.print(Panel(
            "[bold red]❌ Ollama is not running![/bold red]\n\n"
            "The embedding step requires Ollama.\n\n"
            "To fix:\n"
            "  1. Open a [bold]new Terminal window[/bold]\n"
            "  2. Run: [bold green]ollama serve[/bold green]\n"
            "  3. Then re-run this script\n\n"
            "Or skip RAG with: [bold]python ingest_data.py --xbrl-only[/bold]",
            style="red"
        ))
        sys.exit(1)

    if not any(embed_model.split(":")[0] in m for m in models):
        console.print(Panel(
            f"[bold yellow]⚠️  Embedding model '{embed_model}' not found![/bold yellow]\n\n"
            f"Pull it with: [bold green]ollama pull {embed_model}[/bold green]\n"
            f"Then re-run: [bold]python ingest_data.py[/bold]",
            style="yellow"
        ))
        sys.exit(1)

    console.print(f"  [green]✓ Ollama running, '{embed_model}' available[/green]")


def run_xbrl_pipeline():
    """Steps 1-2: Fetch XBRL data → SQLite"""
    from src.ingestion.sec_fetcher import fetch_all_xbrl
    from src.database.db_manager import init_database, save_metrics

    console.print(Rule("[bold blue]STEP 1 — Fetch XBRL Data from SEC EDGAR[/bold blue]"))
    console.print("[dim]  Free public API · No key needed · Rate-limited for politeness[/dim]\n")

    # Initialise DB tables
    init_database()

    # Download all XBRL data
    records = fetch_all_xbrl()

    console.print(Rule("[bold blue]STEP 2 — Save to SQLite Database[/bold blue]"))
    saved = save_metrics(records)

    console.print(f"\n  [bold green]✓ SQLite complete: {saved} records saved[/bold green]")


def run_text_pipeline(years):
    """Steps 3-6: Fetch 10-K text → chunk → embed → ChromaDB"""
    from src.ingestion.sec_fetcher import fetch_all_10k_texts
    from src.ingestion.text_processor import process_all_filings
    from src.ingestion.vector_store import add_documents, get_store_stats

    console.print(Rule(f"[bold blue]STEP 3 — Fetch 10-K Filing Text (years: {years})[/bold blue]"))
    console.print("[dim]  Each 10-K is ~500KB of HTML stripped to clean text[/dim]\n")

    filings = fetch_all_10k_texts(years=years)
    total_filings = sum(len(v) for v in filings.values())
    console.print(f"\n  [green]✓ Retrieved {total_filings} filings[/green]")

    console.print(Rule("[bold blue]STEP 4 — Split Text into Chunks (LangChain)[/bold blue]"))
    console.print("[dim]  RecursiveCharacterTextSplitter · 1000 chars · 150 overlap[/dim]\n")

    documents = process_all_filings(filings)
    console.print(f"\n  [green]✓ {len(documents)} LangChain Document chunks created[/green]")

    console.print(Rule("[bold blue]STEP 5+6 — Embed with Ollama + Store in ChromaDB[/bold blue]"))
    console.print("[dim]  nomic-embed-text (274MB) · 768-dim vectors · local, free[/dim]\n")

    added = add_documents(documents)

    stats = get_store_stats()
    console.print(
        f"\n  [bold green]✓ ChromaDB complete:[/bold green] "
        f"{stats['total_chunks']} chunks · "
        f"Companies: {', '.join(str(c) for c in stats['companies'])} · "
        f"Years: {', '.join(str(y) for y in stats['years'])}"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Financial AI Chatbot — Data Ingestion Pipeline"
    )
    parser.add_argument(
        "--xbrl-only",
        action="store_true",
        help="Only download XBRL structured data (no Ollama needed, very fast)"
    )
    parser.add_argument(
        "--text-only",
        action="store_true",
        help="Only process 10-K text for RAG (skip XBRL)"
    )
    parser.add_argument(
        "--years",
        nargs="+",
        type=int,
        default=[2022, 2023],
        help="Which years to fetch 10-K text for (default: 2022 2023)"
    )
    args = parser.parse_args()

    console.print(Panel(
        "[bold white]🏦 Financial AI Chatbot — Data Ingestion Pipeline[/bold white]\n\n"
        "Companies  : Tesla · Microsoft · Apple\n"
        "Data Source: [green]SEC EDGAR[/green] (free, public US government API)\n"
        "Embeddings : [green]Ollama nomic-embed-text[/green] (free, runs on your Mac)\n"
        "Databases  : [green]SQLite[/green] (structured) + [green]ChromaDB[/green] (vectors)\n"
        "Total Cost : [bold green]$0.00[/bold green]",
        style="bold blue",
        title="Financial AI Chatbot"
    ))

    if not args.text_only:
        run_xbrl_pipeline()

    if not args.xbrl_only:
        verify_ollama_ready()
        run_text_pipeline(args.years)

    console.print(Panel(
        "[bold green]🎉 Ingestion Complete![/bold green]\n\n"
        "Now start the chatbot:\n\n"
        "  [bold white]streamlit run app.py[/bold white]",
        style="green"
    ))


if __name__ == "__main__":
    main()
