# ══════════════════════════════════════════════════════════
#  src/ingestion/sec_fetcher.py
#
#  WHAT THIS FILE DOES:
#  Talks to the U.S. SEC EDGAR API to download:
#    1. XBRL data  → structured financial numbers (revenue, net income, etc.)
#    2. 10-K text  → unstructured filing text (risks, strategy, MD&A)
#
#  SEC EDGAR is a FREE public API — no key needed.
#  It just asks for a User-Agent header to know who's calling.
# ══════════════════════════════════════════════════════════

import os          # reads environment variables from .env
import time        # we use time.sleep() to be polite to SEC servers
import json        # SEC returns JSON data
import re          # regular expressions for cleaning HTML text
import requests    # makes HTTP requests (like a browser visiting a website)
from pathlib import Path    # handles file paths cleanly across Mac/Windows
from dotenv import load_dotenv   # reads your .env file

# Load the .env file so os.getenv() can read our settings
load_dotenv()

# ── Configuration ──────────────────────────────────────────────────────────────

# The SEC requires a descriptive User-Agent. Read from .env
USER_AGENT = os.getenv("SEC_USER_AGENT", "FinancialBot research@example.com")

# Every request to SEC needs this header — otherwise they'll block us
HEADERS = {
    "User-Agent": USER_AGENT,
    "Accept-Encoding": "gzip, deflate",
    "Host": "data.sec.gov",
}

# ── Company Identifiers ────────────────────────────────────────────────────────
# CIK = Central Index Key — the SEC's unique ID for every company
# You can find any company's CIK at: https://www.sec.gov/cgi-bin/browse-edgar
COMPANIES = {
    "Tesla":     "0001318605",
    "Microsoft": "0000789019",
    "Apple":     "0000320193",
}

# ── Which Financial Metrics to Extract ────────────────────────────────────────
# These are official XBRL tag names (us-gaap standard).
# Left side = XBRL tag in the SEC data
# Right side = friendly name we'll use in our database
XBRL_METRICS = {
    # Revenue can have different tags depending on the company
    "Revenues":                                           "Revenue",
    "RevenueFromContractWithCustomerExcludingAssessedTax": "Revenue",

    # Income statement items
    "NetIncomeLoss":                                      "Net Income",
    "GrossProfit":                                        "Gross Profit",
    "OperatingIncomeLoss":                                "Operating Income",
    "CostOfRevenue":                                      "Cost of Revenue",
    "ResearchAndDevelopmentExpense":                      "R&D Expense",
    "SellingGeneralAndAdministrativeExpense":              "SG&A Expense",

    # Balance sheet items
    "Assets":                                             "Total Assets",
    "Liabilities":                                        "Total Liabilities",
    "StockholdersEquity":                                 "Stockholders Equity",
    "CashAndCashEquivalentsAtCarryingValue":               "Cash and Equivalents",
    "LongTermDebt":                                       "Long-Term Debt",

    # Per-share data
    "EarningsPerShareBasic":                              "EPS Basic",
    "EarningsPerShareDiluted":                            "EPS Diluted",
}

# Where to save raw downloaded files (so we don't re-download if we restart)
RAW_DATA_DIR = Path("data/raw")
RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

BASE_URL = "https://data.sec.gov"


# ══════════════════════════════════════════════════════════
#  HELPER FUNCTIONS
# ══════════════════════════════════════════════════════════

def _get_json(url: str) -> dict:
    """
    Make a GET request to the SEC API and return JSON.

    We add a small delay (0.2 seconds) between requests.
    The SEC asks for no more than 10 requests per second.
    Being polite = not getting blocked.
    """
    time.sleep(0.2)  # be polite to SEC servers
    response = requests.get(url, headers=HEADERS, timeout=30)
    response.raise_for_status()  # raises an error if request failed
    return response.json()


def _get_text(url: str) -> str:
    """
    Make a GET request to www.sec.gov (different from data.sec.gov)
    and return the raw text content.
    """
    time.sleep(0.2)
    # Filing documents are on www.sec.gov, not data.sec.gov
    # So we need different headers
    headers = {
        "User-Agent": USER_AGENT,
        "Accept-Encoding": "gzip, deflate",
        "Host": "www.sec.gov",
    }
    response = requests.get(url, headers=headers, timeout=60)
    response.raise_for_status()
    return response.text


def _clean_html(text: str) -> str:
    """
    10-K filings are HTML files. We strip the HTML tags
    to get clean readable text for our RAG pipeline.

    Example:
        Input:  "<p>Revenue was <b>$100B</b></p>"
        Output: "Revenue was $100B"
    """
    # Remove all HTML tags (anything between < and >)
    text = re.sub(r'<[^>]+>', ' ', text)
    # Collapse multiple whitespace/newlines into single spaces
    text = re.sub(r'\s+', ' ', text).strip()
    return text


# ══════════════════════════════════════════════════════════
#  PART 1: XBRL STRUCTURED FINANCIAL DATA
# ══════════════════════════════════════════════════════════

def fetch_xbrl_facts(company_name: str, cik: str) -> list[dict]:
    """
    Download ALL financial facts for one company from SEC EDGAR.

    The SEC XBRL API returns a huge JSON with every metric
    the company has ever reported in a structured format.

    We filter it down to:
      - Only the metrics we care about (from XBRL_METRICS above)
      - Only annual 10-K filings (not quarterly 10-Q)
      - Only years 2019-2024

    Returns a list of dicts like:
      {"company": "Tesla", "year": 2023, "metric": "Revenue",
       "value": 96773000000.0, "unit": "USD"}
    """
    print(f"\n  Fetching XBRL data for {company_name}...")

    # The SEC's "company facts" endpoint — one URL per company
    url = f"{BASE_URL}/api/xbrl/companyfacts/CIK{cik}.json"
    data = _get_json(url)

    # Save raw data to disk (useful for debugging)
    cache_path = RAW_DATA_DIR / f"{company_name.lower()}_xbrl_raw.json"
    cache_path.write_text(json.dumps(data, indent=2))

    # The response structure is:
    # data["facts"]["us-gaap"]["Revenues"]["units"]["USD"] = [list of entries]
    us_gaap_facts = data.get("facts", {}).get("us-gaap", {})

    records = []

    # Loop through each XBRL concept we care about
    for xbrl_tag, friendly_name in XBRL_METRICS.items():
        # Skip if this company doesn't report this metric
        if xbrl_tag not in us_gaap_facts:
            continue

        concept_data = us_gaap_facts[xbrl_tag]

        # Metrics come in different units: USD, USD/shares, shares
        for unit_type, entries in concept_data.get("units", {}).items():
            for entry in entries:
                # Only keep annual 10-K filings (not quarterly 10-Q)
                if "10-K" not in entry.get("form", ""):
                    continue

                # Get the end date of the reporting period (e.g. "2023-12-31")
                end_date = entry.get("end", "")
                if not end_date:
                    continue

                # Extract just the year
                year = int(end_date[:4])

                # For Apple fiscal year (ends September):
                # A filing ending Sep 2023 should be labeled 2023 not 2022
                # end_date already captures this correctly via [:4]

                # Only keep 2019-2024
                if year < 2019 or year > 2024:
                    continue

                value = entry.get("val")
                if value is None:
                    continue

                records.append({
                    "company":  company_name,
                    "year":     year,
                    "metric":   friendly_name,
                    "value":    float(value),
                    "unit":     unit_type,
                    "end_date": end_date,  # keep for de-duplication
                })

    # ── De-duplication ──────────────────────────────────────
    # A company might file amendments (10-K/A) which creates duplicates.
    # For the same (company, year, metric), keep the entry with
    # the LATEST end_date — it's the most complete/corrected version.
    seen = {}
    for record in records:
        key = (record["company"], record["year"], record["metric"])
        if key not in seen:
            seen[key] = record
        elif record["end_date"] > seen[key]["end_date"]:
            seen[key] = record  # replace with the more recent filing

    final_records = list(seen.values())
    print(f"  ✓ {len(final_records)} metrics extracted for {company_name}")
    return final_records


def fetch_all_xbrl() -> list[dict]:
    """Fetch XBRL data for ALL three companies and combine into one list."""
    all_records = []
    for company_name, cik in COMPANIES.items():
        records = fetch_xbrl_facts(company_name, cik)
        all_records.extend(records)

    print(f"\n  Total records across all companies: {len(all_records)}")
    return all_records


# ══════════════════════════════════════════════════════════
#  PART 2: 10-K FILING TEXT (for RAG)
# ══════════════════════════════════════════════════════════

def fetch_10k_text(company_name: str, cik: str, target_year: int) -> str | None:
    """
    Find and download the actual 10-K filing document for a given year.

    Process:
    1. Get the list of all filings this company has ever submitted
    2. Find 10-K filings from the target year (or year+1 for fiscal year lag)
    3. Download the primary HTML document
    4. Strip HTML tags to get clean text

    Returns the cleaned text, or None if not found.
    """
    print(f"  Fetching 10-K text: {company_name} {target_year}...")

    # Step 1: Get the submissions index for this company
    submissions_url = f"{BASE_URL}/submissions/CIK{cik}.json"
    submissions = _get_json(submissions_url)

    # The response has parallel arrays — all the same length
    # Index 0 of each array = the same filing
    filings   = submissions.get("filings", {}).get("recent", {})
    forms      = filings.get("form", [])          # e.g. ["10-K", "10-Q", ...]
    dates      = filings.get("filingDate", [])    # e.g. ["2024-01-26", ...]
    accnums    = filings.get("accessionNumber", []) # e.g. ["0001318605-24-000012"]
    primary_docs = filings.get("primaryDocument", []) # e.g. ["tsla-20231231.htm"]

    # Step 2: Find 10-K filings that match our target year
    candidates = []
    for i, form_type in enumerate(forms):
        if form_type != "10-K":
            continue

        filing_year = int(dates[i][:4])
        # Fiscal year 2023 might be FILED in early 2024, so check both
        if filing_year not in (target_year, target_year + 1):
            continue

        candidates.append({
            "date":    dates[i],
            "raw_accnum": accnums[i],                          # "0001318605-24-000012"
            "accnum":  accnums[i].replace("-", ""),            # "000131860524000012"
            "doc":     primary_docs[i],
        })

    if not candidates:
        print(f"  ✗ No 10-K found for {company_name} {target_year}")
        return None

    # Use the most recent filing (first in list = most recent)
    best = sorted(candidates, key=lambda x: x["date"], reverse=True)[0]

    # Step 3: Build the download URL for the actual document
    # SEC stores filing documents at this path structure:
    # /Archives/edgar/data/{CIK}/{accession_number}/{document_name}
    cik_int = int(cik)  # remove leading zeros for the URL
    doc_url = (
        f"https://www.sec.gov/Archives/edgar/data/"
        f"{cik_int}/{best['accnum']}/{best['doc']}"
    )

    # Step 4: Download and clean
    try:
        raw_html = _get_text(doc_url)
        clean_text = _clean_html(raw_html)

        # Sanity check — if text is too short, something went wrong
        if len(clean_text) < 5_000:
            print(f"  ✗ Retrieved text too short ({len(clean_text)} chars)")
            return None

        # Cap at 500,000 chars (about 125,000 tokens) to keep things manageable
        clean_text = clean_text[:500_000]

        # Cache to disk so we don't re-download on re-runs
        cache_path = RAW_DATA_DIR / f"{company_name.lower()}_{target_year}_10k.txt"
        cache_path.write_text(clean_text)

        print(f"  ✓ {len(clean_text):,} characters retrieved")
        return clean_text

    except Exception as e:
        print(f"  ✗ Failed to download ({str(e)[:60]})")
        return None


def fetch_all_10k_texts(years: list[int] = None) -> dict[str, dict[int, str]]:
    """
    Fetch 10-K text for all three companies for each specified year.

    Returns a nested dict:
    {
        "Tesla":     {2022: "full text...", 2023: "full text..."},
        "Microsoft": {2022: "full text...", 2023: "full text..."},
        "Apple":     {2022: "full text...", 2023: "full text..."},
    }
    """
    if years is None:
        years = [2022, 2023]  # default: last 2 years

    results = {}
    for company_name, cik in COMPANIES.items():
        results[company_name] = {}
        for year in years:
            text = fetch_10k_text(company_name, cik, year)
            if text:
                results[company_name][year] = text
            time.sleep(0.5)  # extra pause between companies

    return results


# ── Quick test when you run this file directly ─────────────────────────────────
if __name__ == "__main__":
    print("Testing SEC fetcher...")
    print("\n=== XBRL Test (Tesla only) ===")
    records = fetch_xbrl_facts("Tesla", "0001318605")
    if records:
        # Show a few sample records
        print("\nSample records:")
        for r in records[:3]:
            print(f"  {r['company']} | {r['year']} | {r['metric']} | ${r['value']:,.0f}")
