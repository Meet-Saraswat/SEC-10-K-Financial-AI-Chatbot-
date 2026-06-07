# ══════════════════════════════════════════════════════════
#  src/chains/calculator.py
#  Formats SQL results for the LLM — no Python math here.
#  Numbers displayed as: Metric (Year): $Value
#  YoY displayed as: grew from $X (2022) to $Y (2023)
# ══════════════════════════════════════════════════════════

from src.database import db_manager as db


def format_currency(value: float, unit: str = "USD") -> str:
    if value is None:
        return "N/A"
    if unit == "USD/shares":
        return f"${value:.2f}"
    sign    = "-" if value < 0 else ""
    abs_val = abs(value)
    if abs_val >= 1e12:
        return f"{sign}${abs_val / 1e12:.1f}T"
    elif abs_val >= 1e9:
        return f"{sign}${abs_val / 1e9:.1f}B"
    elif abs_val >= 1e6:
        return f"{sign}${abs_val / 1e6:.1f}M"
    else:
        return f"{sign}${abs_val:,.0f}"


def format_pct(value: float | None, include_sign: bool = True) -> str:
    if value is None:
        return "N/A"
    sign = "+" if (include_sign and value >= 0) else ""
    return f"{sign}{value:.1f}%"


def build_single_company_context(company: str, year: int) -> str:
    lines = [f"=== {company} Financial Summary — {year} ===\n"]

    # Key metrics — labeled as Metric (Year): $Value
    metrics = db.sql_company_full_summary(company, year)
    if metrics:
        lines.append("Key Metrics:")
        for row in metrics:
            unit = row.get("unit", "USD")
            lines.append(
                f"  {row['metric']} ({year}): "
                f"{format_currency(row['value'], unit)}"
            )

    # YoY Revenue — correct verb (grew vs declined) based on direction
    yoy = db.sql_yoy_growth(company, "Revenue", year)
    if yoy:
        unit      = yoy.get("unit", "USD")
        prior_val = format_currency(yoy["prior_value"],          unit)
        curr_val  = format_currency(yoy["current_value"],        unit)
        chng_val  = format_currency(abs(yoy["absolute_change"]), unit)
        pct_val   = format_pct(yoy["pct_change"])
        positive  = yoy["direction"] == "positive"
        verb      = "grew" if positive else "declined"
        direction = "an increase" if positive else "a decrease"

        lines.append(
            f"\nYear-over-Year Revenue:\n"
            f"  Revenue {verb} from {prior_val} in {yoy['prior_year']} "
            f"to {curr_val} in {yoy['current_year']}, "
            f"{direction} of {chng_val} ({pct_val})"
        )

    # Margins
    gm = db.sql_gross_margin(company, year)
    om = db.sql_operating_margin(company, year)
    if gm or om:
        lines.append(f"\nMargins ({year}):")
    if gm and gm.get("gross_margin_pct"):
        lines.append(
            f"  Gross Margin ({year}): {gm['gross_margin_pct']}%"
            f"  (Revenue: {format_currency(gm['revenue'])} - "
            f"COGS: {format_currency(gm['cost_of_revenue'])})"
        )
    if om and om.get("operating_margin_pct"):
        lines.append(
            f"  Operating Margin ({year}): {om['operating_margin_pct']}%"
        )

    return "\n".join(lines)


def build_comparison_context(companies: list[str], metric: str, year: int) -> str:
    lines = [f"=== {metric} Comparison — {year} ===\n"]
    lines.append(f"Ranked by {metric} (highest first):\n")

    ranked = db.sql_company_comparison(companies, metric, year)
    if not ranked:
        return f"No {metric} data found for {companies} in {year}."

    for row in ranked:
        unit = row.get("unit", "USD")
        lines.append(
            f"  #{row['rank']}  {row['company']:<12}  "
            f"{metric} ({year}): {format_currency(row['value'], unit)}"
        )

    # 3-year trend per company
    lines.append(f"\n3-Year Trend:")
    for company in companies:
        trend = db.sql_multi_year_trend(company, metric)
        if trend:
            recent = trend[-3:]
            parts  = [
                f"{t['year']}: {format_currency(t['value'], t.get('unit','USD'))} ({t['yoy_fmt']})"
                for t in recent
            ]
            lines.append(f"  {company}: " + "  →  ".join(parts))

    return "\n".join(lines)


def build_growth_race_context(metric: str, year: int) -> str:
    rows = db.sql_fastest_growing(metric, year)
    if not rows:
        return f"No growth data found for {metric} in {year}."

    lines = [f"=== Fastest {metric} Growth — {year} ===\n"]
    for row in rows:
        unit      = row.get("unit", "USD")
        prior_fmt = format_currency(row["prior_value"], unit)
        curr_fmt  = format_currency(row["value"],       unit)
        lines.append(
            f"  #{row['rank']}  {row['company']:<12}  "
            f"{metric} grew from {prior_fmt} ({year-1}) "
            f"to {curr_fmt} ({year}), "
            f"growth: {format_pct(row['growth_pct'])}"
        )
    return "\n".join(lines)


def build_financial_context(
    companies: list[str],
    metrics:   list[str],
    years:     list[int],
    question:  str = "",
) -> str:
    year   = years[0]   if years   else 2023
    metric = metrics[0] if metrics else "Revenue"
    q      = question.lower()

    if any(kw in q for kw in ["fastest", "most growth", "grew most", "highest growth"]):
        return build_growth_race_context(metric, year)

    if len(companies) == 1:
        return build_single_company_context(companies[0], year)

    return build_comparison_context(companies, metric, year)
