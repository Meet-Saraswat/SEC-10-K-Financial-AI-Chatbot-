# ══════════════════════════════════════════════════════════
#  src/database/db_manager.py
#
#  WHAT THIS FILE DOES:
#  Manages our SQLite database — the "SQL Database" box
#  in the architecture diagram.
#
#  ALL calculations now happen INSIDE SQL using:
#    - LAG()   → previous year's value for YoY growth
#    - RANK()  → ranking companies by a metric
#    - ROUND() → formatting to 2 decimal places
#    - CASE    → conditional logic (positive/negative growth)
#    - CTEs    → readable multi-step queries
#
#  The architecture flow is now:
#    SQL Database → SQL Calculations → LLM Explanation
#  (Python only formats the final string for display)
# ══════════════════════════════════════════════════════════

import os
import sqlite3
from pathlib import Path
from contextlib import contextmanager
from dotenv import load_dotenv

load_dotenv()

DB_PATH = Path(os.getenv("DB_PATH", "data/db/financial_data.db"))
DB_PATH.parent.mkdir(parents=True, exist_ok=True)


# ══════════════════════════════════════════════════════════
#  SCHEMA
# ══════════════════════════════════════════════════════════

CREATE_TABLES_SQL = """
CREATE TABLE IF NOT EXISTS companies (
    id      INTEGER PRIMARY KEY AUTOINCREMENT,
    name    TEXT    NOT NULL UNIQUE,
    ticker  TEXT,
    cik     TEXT,
    sector  TEXT
);

CREATE TABLE IF NOT EXISTS financial_metrics (
    id      INTEGER PRIMARY KEY AUTOINCREMENT,
    company TEXT    NOT NULL,
    year    INTEGER NOT NULL,
    metric  TEXT    NOT NULL,
    value   REAL    NOT NULL,
    unit    TEXT    DEFAULT 'USD',
    form    TEXT    DEFAULT '10-K',
    UNIQUE(company, year, metric)
);

CREATE INDEX IF NOT EXISTS idx_company ON financial_metrics(company);
CREATE INDEX IF NOT EXISTS idx_year    ON financial_metrics(year);
CREATE INDEX IF NOT EXISTS idx_metric  ON financial_metrics(metric);
"""

COMPANY_SEEDS = [
    ("Tesla",     "TSLA", "0001318605", "Automotive / Energy"),
    ("Microsoft", "MSFT", "0000789019", "Technology"),
    ("Apple",     "AAPL", "0000320193", "Technology"),
]


# ══════════════════════════════════════════════════════════
#  CONNECTION HELPER
# ══════════════════════════════════════════════════════════

@contextmanager
def get_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    # Enable SQLite window functions (LAG, RANK, etc.)
    # These require SQLite 3.25+ — ships with Python 3.8+
    conn.execute("PRAGMA journal_mode=WAL")
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


# ══════════════════════════════════════════════════════════
#  SETUP
# ══════════════════════════════════════════════════════════

def init_database():
    with get_connection() as conn:
        conn.executescript(CREATE_TABLES_SQL)
        conn.executemany(
            "INSERT OR IGNORE INTO companies (name, ticker, cik, sector) VALUES (?,?,?,?)",
            COMPANY_SEEDS
        )
    print(f"  ✓ Database initialised at: {DB_PATH}")


# ══════════════════════════════════════════════════════════
#  WRITE
# ══════════════════════════════════════════════════════════

def save_metrics(records: list[dict]) -> int:
    rows = [
        (r["company"], r["year"], r["metric"], r["value"],
         r.get("unit", "USD"), r.get("form", "10-K"))
        for r in records
    ]
    with get_connection() as conn:
        conn.executemany(
            """INSERT OR REPLACE INTO financial_metrics
               (company, year, metric, value, unit, form)
               VALUES (?, ?, ?, ?, ?, ?)""",
            rows
        )
    print(f"  ✓ Saved {len(rows)} records to database")
    return len(rows)


# ══════════════════════════════════════════════════════════
#  SQL CALCULATIONS
#  Every function below runs a SQL query that returns
#  already-computed results — no Python math needed.
# ══════════════════════════════════════════════════════════

def sql_yoy_growth(company: str, metric: str, year: int) -> dict | None:
    """
    Year-over-Year growth — calculated entirely in SQL.

    SQL technique used: LAG() window function
    LAG(value, 1) OVER (ORDER BY year) gives the PREVIOUS row's value
    without any Python loops or second query.

    SQL performed:
        WITH ordered AS (
            SELECT year, value,
                   LAG(value, 1) OVER (ORDER BY year) AS prior_value
            FROM financial_metrics
            WHERE company = ? AND metric LIKE ?
        )
        SELECT
            year                           AS current_year,
            year - 1                       AS prior_year,
            value                          AS current_value,
            prior_value,
            value - prior_value            AS absolute_change,
            ROUND((value - prior_value)
                  / ABS(prior_value) * 100, 2)  AS pct_change
        FROM ordered
        WHERE year = ?

    Returns a dict with all growth fields, or None if data missing.
    """
    sql = """
        WITH ordered AS (
            -- Step 1: Get all years for this company+metric,
            --         and attach the prior year's value using LAG()
            SELECT
                year,
                value,
                unit,
                LAG(value, 1) OVER (
                    PARTITION BY company, metric
                    ORDER BY year
                ) AS prior_value
            FROM financial_metrics
            WHERE LOWER(company) = LOWER(?)
              AND LOWER(metric)  LIKE LOWER(?)
        )
        -- Step 2: Calculate growth for the requested year only
        SELECT
            year                                          AS current_year,
            year - 1                                      AS prior_year,
            value                                         AS current_value,
            prior_value,
            (value - prior_value)                         AS absolute_change,
            ROUND((value - prior_value)
                  / ABS(prior_value) * 100.0, 2)          AS pct_change,
            unit,
            -- Flag whether growth was positive or negative
            CASE WHEN value >= prior_value THEN 'positive'
                 ELSE 'negative' END                      AS direction
        FROM ordered
        WHERE year = ?
          AND prior_value IS NOT NULL   -- can't calc growth for first year
    """
    with get_connection() as conn:
        row = conn.execute(sql, [company, f"%{metric}%", year]).fetchone()

    return dict(row) if row else None


def sql_multi_year_trend(company: str, metric: str) -> list[dict]:
    """
    Multi-year trend with YoY growth for each year — all in SQL.

    SQL technique: LAG() + window functions to compute growth
    for every year in one query (no Python loop over years).

    Returns rows like:
        [{"year": 2019, "value": 24578000000, "yoy_pct": null},
         {"year": 2020, "value": 31536000000, "yoy_pct": 28.31},
         ...]
    """
    sql = """
        WITH base AS (
            SELECT
                year,
                value,
                unit,
                LAG(value, 1) OVER (
                    PARTITION BY company, metric
                    ORDER BY year
                ) AS prior_value
            FROM financial_metrics
            WHERE LOWER(company) = LOWER(?)
              AND LOWER(metric)  LIKE LOWER(?)
        )
        SELECT
            year,
            value,
            unit,
            prior_value,
            CASE
                WHEN prior_value IS NOT NULL AND prior_value != 0
                THEN ROUND((value - prior_value) / ABS(prior_value) * 100.0, 2)
                ELSE NULL
            END AS yoy_pct,
            CASE
                WHEN prior_value IS NULL THEN '—'
                WHEN value >= prior_value THEN
                    '+' || ROUND((value - prior_value) / ABS(prior_value) * 100.0, 1) || '%'
                ELSE
                    ROUND((value - prior_value) / ABS(prior_value) * 100.0, 1) || '%'
            END AS yoy_fmt
        FROM base
        ORDER BY year ASC
    """
    with get_connection() as conn:
        rows = conn.execute(sql, [company, f"%{metric}%"]).fetchall()
    return [dict(r) for r in rows]


def sql_company_comparison(companies: list[str], metric: str, year: int) -> list[dict]:
    """
    Rank multiple companies by a metric for a given year — in SQL.

    SQL technique: RANK() window function assigns 1st/2nd/3rd place
    ordered by value descending.

    Returns rows sorted rank 1 → N:
        [{"rank": 1, "company": "Apple",    "value": 383285000000},
         {"rank": 2, "company": "Microsoft","value": 211915000000},
         {"rank": 3, "company": "Tesla",    "value": 96773000000}]
    """
    placeholders = ",".join("?" * len(companies))

    sql = f"""
        SELECT
            RANK() OVER (ORDER BY value DESC) AS rank,
            company,
            year,
            metric,
            value,
            unit
        FROM financial_metrics
        WHERE LOWER(company) IN ({placeholders})
          AND LOWER(metric)  LIKE LOWER(?)
          AND year = ?
        ORDER BY rank
    """
    params = [c.lower() for c in companies] + [f"%{metric}%", year]

    with get_connection() as conn:
        rows = conn.execute(sql, params).fetchall()
    return [dict(r) for r in rows]


def sql_gross_margin(company: str, year: int) -> dict | None:
    """
    Gross Margin % = (Revenue - Cost of Revenue) / Revenue * 100
    Calculated entirely in SQL using a self-join.

    SQL technique: Join the table to itself — one alias for Revenue,
    one alias for Cost of Revenue — then compute the ratio.
    """
    sql = """
        SELECT
            rev.company,
            rev.year,
            rev.value                                       AS revenue,
            cogs.value                                      AS cost_of_revenue,
            (rev.value - cogs.value)                        AS gross_profit,
            ROUND(
                (rev.value - cogs.value) / rev.value * 100.0
            , 2)                                            AS gross_margin_pct
        FROM financial_metrics AS rev
        -- Self-join: match the same company+year but different metric
        JOIN financial_metrics AS cogs
          ON  rev.company = cogs.company
          AND rev.year    = cogs.year
          AND LOWER(cogs.metric) LIKE '%cost of revenue%'
        WHERE LOWER(rev.metric) LIKE '%revenue%'
          AND LOWER(rev.company) = LOWER(?)
          AND rev.year = ?
          -- Exclude "Cost of Revenue" from the revenue row
          AND LOWER(rev.metric) NOT LIKE '%cost%'
        LIMIT 1
    """
    with get_connection() as conn:
        row = conn.execute(sql, [company, year]).fetchone()
    return dict(row) if row else None


def sql_operating_margin(company: str, year: int) -> dict | None:
    """
    Operating Margin % = Operating Income / Revenue * 100
    Computed in SQL with a self-join, same pattern as gross margin.
    """
    sql = """
        SELECT
            rev.company,
            rev.year,
            rev.value                                       AS revenue,
            op.value                                        AS operating_income,
            ROUND(op.value / rev.value * 100.0, 2)          AS operating_margin_pct
        FROM financial_metrics AS rev
        JOIN financial_metrics AS op
          ON  rev.company = op.company
          AND rev.year    = op.year
          AND LOWER(op.metric) LIKE '%operating income%'
        WHERE LOWER(rev.metric) LIKE '%revenue%'
          AND LOWER(rev.company) = LOWER(?)
          AND rev.year = ?
          AND LOWER(rev.metric) NOT LIKE '%cost%'
        LIMIT 1
    """
    with get_connection() as conn:
        row = conn.execute(sql, [company, year]).fetchone()
    return dict(row) if row else None


def sql_company_full_summary(company: str, year: int) -> list[dict]:
    """
    All key metrics for one company in one year — single SQL query.

    SQL technique: WHERE metric IN (...) to fetch multiple metrics
    at once instead of one query per metric.
    """
    sql = """
        SELECT
            company,
            year,
            metric,
            value,
            unit
        FROM financial_metrics
        WHERE LOWER(company) = LOWER(?)
          AND year = ?
          AND metric IN (
              'Revenue', 'Net Income', 'Gross Profit',
              'Operating Income', 'Total Assets', 'Total Liabilities',
              'Stockholders Equity', 'Cash and Equivalents',
              'R&D Expense', 'Cost of Revenue'
          )
        ORDER BY metric
    """
    with get_connection() as conn:
        rows = conn.execute(sql, [company, year]).fetchall()
    return [dict(r) for r in rows]


def sql_fastest_growing(metric: str, year: int) -> list[dict]:
    """
    Which company had the fastest growth for a metric? — SQL only.

    SQL technique: LAG() to get prior year, RANK() to order by growth rate.
    Returns all three companies ranked by growth percentage.
    """
    sql = """
        WITH with_prior AS (
            SELECT
                company,
                year,
                value,
                unit,
                LAG(value, 1) OVER (
                    PARTITION BY company
                    ORDER BY year
                ) AS prior_value
            FROM financial_metrics
            WHERE LOWER(metric) LIKE LOWER(?)
        ),
        with_growth AS (
            SELECT
                company,
                year,
                value,
                prior_value,
                CASE
                    WHEN prior_value IS NOT NULL AND prior_value != 0
                    THEN ROUND((value - prior_value) / ABS(prior_value) * 100.0, 2)
                    ELSE NULL
                END AS growth_pct
            FROM with_prior
            WHERE year = ?
        )
        SELECT
            RANK() OVER (ORDER BY growth_pct DESC) AS rank,
            company,
            value,
            unit,
            prior_value,
            growth_pct
        FROM with_growth
        WHERE growth_pct IS NOT NULL
        ORDER BY rank
    """
    with get_connection() as conn:
        rows = conn.execute(sql, [f"%{metric}%", year]).fetchall()
    return [dict(r) for r in rows]


# ══════════════════════════════════════════════════════════
#  SIMPLE READ HELPERS (used by UI filters)
# ══════════════════════════════════════════════════════════

def get_metric(company: str, metric: str, year: int = None) -> list[dict]:
    sql = """SELECT company, year, metric, value, unit
             FROM financial_metrics
             WHERE LOWER(company) = LOWER(?) AND LOWER(metric) LIKE LOWER(?)"""
    params = [company, f"%{metric}%"]
    if year is not None:
        sql += " AND year = ?"
        params.append(year)
    sql += " ORDER BY year DESC"
    with get_connection() as conn:
        rows = conn.execute(sql, params).fetchall()
    return [dict(r) for r in rows]


def get_all_records() -> list[dict]:
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT * FROM financial_metrics ORDER BY company, metric, year"
        ).fetchall()
    return [dict(r) for r in rows]


def get_distinct_metrics() -> list[str]:
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT DISTINCT metric FROM financial_metrics ORDER BY metric"
        ).fetchall()
    return [r["metric"] for r in rows]


def get_distinct_years() -> list[int]:
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT DISTINCT year FROM financial_metrics ORDER BY year DESC"
        ).fetchall()
    return [r["year"] for r in rows]


# ── Quick test ─────────────────────────────────────────────
if __name__ == "__main__":
    init_database()
    print("\nYoY growth test (needs data loaded first):")
    result = sql_yoy_growth("Tesla", "Revenue", 2023)
    print(result)

    print("\nComparison test:")
    result = sql_company_comparison(["Tesla", "Microsoft", "Apple"], "Revenue", 2023)
    for r in result:
        print(f"  #{r['rank']} {r['company']}: {r['value']}")
