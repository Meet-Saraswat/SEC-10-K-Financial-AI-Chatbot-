import os
import sys
import re
import streamlit as st
import pandas as pd
import plotly.express as px
from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

st.set_page_config(
    page_title = "Financial AI Chatbot",
    page_icon  = "📊",
    layout     = "wide",
    initial_sidebar_state = "expanded",
)

st.markdown("""
<style>
    .main-title {
        font-size: 2rem; font-weight: 800;
        background: linear-gradient(135deg, #1565C0, #0288D1);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    }
    .free-tag {
        background: #E8F5E9; color: #2E7D32; border: 1px solid #A5D6A7;
        border-radius: 20px; padding: 0.2rem 0.7rem; font-size: 0.78rem;
        font-weight: 600;
    }
    /* Word wrap for long answers */
    .stText, .stChatMessage, pre, code {
        white-space: pre-wrap !important;
        word-wrap: break-word !important;
        overflow-wrap: break-word !important;
        max-width: 100% !important;
    }
    .source-chip {
        background: #EDE7F6; color: #4527A0;
        border-radius: 20px; padding: 0.2rem 0.6rem; font-size: 0.75rem;
        margin: 0.15rem; display: inline-block;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data(ttl=20)
def check_ollama():
    import requests
    base = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    try:
        resp = requests.get(f"{base}/api/tags", timeout=4)
        resp.raise_for_status()
        models = [m["name"] for m in resp.json().get("models", [])]
        llm_model   = os.getenv("LLM_MODEL", "llama3.2:3b")
        embed_model = os.getenv("EMBEDDING_MODEL", "nomic-embed-text")
        missing = []
        if not any(llm_model.split(":")[0] in m for m in models):
            missing.append(f"`ollama pull {llm_model}`")
        if not any(embed_model.split(":")[0] in m for m in models):
            missing.append(f"`ollama pull {embed_model}`")
        if missing:
            return False, "Missing models: " + ", ".join(missing)
        return True, f"Running · {len(models)} model(s) loaded"
    except Exception:
        return False, "Not running — open Terminal and run: `ollama serve`"

@st.cache_data(ttl=30)
def check_database():
    try:
        from src.database.db_manager import get_all_records
        records = get_all_records()
        return len(records) > 0, len(records)
    except Exception:
        return False, 0

@st.cache_data(ttl=60)
def check_vectorstore():
    try:
        from src.ingestion.vector_store import get_store_stats
        stats = get_store_stats()
        count = stats.get("total_chunks", 0)
        return count > 0, count
    except Exception:
        return False, 0

with st.sidebar:
    st.markdown("## ⚙️ System Status")
    ollama_ok, ollama_msg = check_ollama()
    db_ok, db_count       = check_database()
    vs_ok, vs_count       = check_vectorstore()

    if ollama_ok:
        st.success(f"✅ Ollama — {ollama_msg}")
    else:
        st.error(f"❌ Ollama — {ollama_msg}")

    if db_ok:
        st.success(f"✅ SQLite — {db_count:,} records")
    else:
        st.warning("⚠️ SQLite — empty")
        st.code("python ingest_data.py --xbrl-only")

    if vs_ok:
        st.success(f"✅ ChromaDB — {vs_count:,} chunks")
    else:
        st.info("ℹ️ ChromaDB — empty (RAG disabled)")

    if st.button("🔄 Refresh Status", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

    st.divider()
    st.markdown("## 🏗️ Architecture")
    st.markdown("""
```
User Question
      ↓
Intent Detection
      ↓
┌────┬────┬────┐
│Fin │Text│Mix │
│SQL │Vec │Both│
│Calc│DB  │    │
└────┴────┴────┘
      ↓
LLM Explanation
(Ollama — local)
```
    """)

    st.divider()
    st.markdown("## 💰 Cost: **$0.00**")
    st.caption("SEC EDGAR · Ollama · ChromaDB · SQLite — all free")
    st.divider()

    st.markdown("## 💡 Example Questions")
    EXAMPLES = {
        "📊 Financial": [
            "What was Tesla's revenue in 2023?",
            "Compare Apple and Microsoft revenue in 2023",
            "Which company had the highest net income in 2022?",
            "What is Apple's gross margin in 2023?",
        ],
        "📄 Textual (needs RAG)": [
            "What risks does Tesla mention in its 10-K?",
            "Describe Apple's competitive landscape",
            "What does Microsoft say about its cloud strategy?",
        ],
        "🔀 Mixed": [
            "Why did Tesla's revenue increase in 2023?",
            "What drove Apple's profit growth?",
            "Explain Microsoft's revenue growth in 2023",
        ],
    }

    for category, questions in EXAMPLES.items():
        with st.expander(category):
            for q in questions:
                if st.button(q, use_container_width=True, key=f"ex_{q[:25]}"):
                    st.session_state["prefill_question"] = q

st.markdown('<p class="main-title">📊 Financial AI Chatbot</p>', unsafe_allow_html=True)
st.markdown(
    "SEC 10-K Analysis · Tesla · Microsoft · Apple &nbsp;"
    '<span class="free-tag">100% Free · Local AI</span>',
    unsafe_allow_html=True
)

tab_chat, tab_dashboard, tab_data = st.tabs(["💬 Chat", "📈 Dashboard", "🗄️ Data Explorer"])

# ── CHAT TAB ──────────────────────────────────────────────
with tab_chat:
    if "messages" not in st.session_state:
        st.session_state.messages = []

    from src.chains.intent_detector import detect_intent, QueryType
    from src.chains.chatbot_engine import ask

    routing_descriptions = {
        QueryType.FINANCIAL: "🗄️ SQLite → 🔢 SQL Calculations → 🤖 Ollama",
        QueryType.TEXTUAL:   "🔍 ChromaDB RAG → 🤖 Ollama",
        QueryType.MIXED:     "🗄️ SQLite + 🔍 ChromaDB → 🔢 SQL Calcs → 🤖 Ollama",
    }

    # ── INPUT AT THE TOP ──────────────────────────────────
    prefill = st.session_state.pop("prefill_question", "")

    # Form with clear_on_submit=True — most reliable way to clear input
    with st.form(key="chat_form", clear_on_submit=True):
        col_input, col_btn = st.columns([5, 1])
        with col_input:
            typed = st.text_input(
                label            = "question",
                value            = prefill if prefill else "",
                placeholder      = "Ask about financials, risks, growth, comparisons...",
                label_visibility = "collapsed",
            )
        with col_btn:
            send = st.form_submit_button("Send ➤", use_container_width=True, type="primary")

    question = typed

    # Show status warnings just below input if needed
    if not ollama_ok:
        st.error("❌ Ollama not running — open Terminal and run: `ollama serve`")
    if not db_ok:
        st.warning("⚠️ No data — run: `python ingest_data.py --xbrl-only`")

    if st.session_state.messages:
        if st.button("🗑️ Clear conversation", type="secondary"):
            st.session_state.messages = []
            st.rerun()

    st.divider()

    # ── PROCESS NEW QUESTION ──────────────────────────────
    if (send or prefill) and question.strip() and ollama_ok and db_ok:
        intent = detect_intent(question)
        qtype  = intent["type"]

        full_response = ""
        try:
            if qtype == QueryType.TEXTUAL and not vs_ok:
                full_response = (
                    "⚠️ This question needs RAG (10-K text search), "
                    "but ChromaDB is empty.\n\n"
                    "Run `python ingest_data.py` to enable text queries."
                )
            else:
                for chunk in ask(question, stream=True):
                    full_response += chunk
        except Exception as e:
            full_response = f"❌ Error: {str(e)}"

        # Clean up malformed markdown from LLM
        # Fixes "15.0B * *, while..." → "**15.0B**, while..."
        full_response = full_response.replace("*", "").strip()

        st.session_state.messages.append({
            "role":    "assistant",
            "content": full_response,
            "question": question,
            "routing": intent,
        })
        st.rerun()

    # ── DISPLAY MESSAGES — NEWEST FIRST ──────────────────
    if not st.session_state.messages:
        st.markdown("""
👋 **Welcome to the Financial AI Chatbot!**

Ask a question above to get started. I can answer questions about
**Tesla, Microsoft, and Apple** using real SEC 10-K filings.

- **Financial** → SQLite + SQL Calculations → Ollama
- **Textual** → ChromaDB RAG → Ollama  
- **Mixed** → Both sources combined → Ollama
        """)
    else:
        for message in reversed(st.session_state.messages):
            # ── Question bubble ───────────────────────────
            with st.chat_message("user"):
                st.markdown(f"**{message['question']}**")

            # ── Answer bubble ─────────────────────────────
            with st.chat_message("assistant"):
                r     = message.get("routing", {})
                qtype = r.get("type", "")
                st.caption(
                    f"**Route:** {routing_descriptions.get(qtype, str(qtype))} "
                    f"· **Companies:** {', '.join(r.get('companies', []))} "
                    f"· **Year:** {r.get('years', [2023])[0]}"
                )
                import html, re

                def convert_table_to_html(text):
                    """
                    Detect plain text pipe tables and convert to HTML tables.
                    Handles variable column widths automatically.
                    """
                    lines = text.split("\n")
                    result = []
                    i = 0
                    while i < len(lines):
                        line = lines[i]
                        # Detect table: line contains | and next line is dashes
                        if "|" in line and i + 1 < len(lines) and re.match(r"[-|\s]+$", lines[i+1]):
                            # Collect all table rows
                            table_lines = [line]
                            i += 1  # skip separator line
                            i += 1
                            while i < len(lines) and "|" in lines[i]:
                                table_lines.append(lines[i])
                                i += 1

                            # Build HTML table
                            html_table = [
                                '<table style="border-collapse:collapse; width:100%; margin:8px 0;">',
                            ]
                            for row_idx, row in enumerate(table_lines):
                                cells = [c.strip() for c in row.split("|") if c.strip()]
                                tag = "th" if row_idx == 0 else "td"
                                style_th = "background:#2a2a2a; color:#fff; padding:8px 12px; text-align:left; border:1px solid #444; white-space:nowrap;"
                                style_td = "padding:8px 12px; text-align:left; border:1px solid #444; white-space:nowrap; word-wrap:normal;"
                                html_table.append("<tr>")
                                # Skip rows that are split item names (only 1 cell, no data columns)
                                if tag == "td" and len(cells) == 1:
                                    # This is a continuation of previous item name — skip it
                                    # Remove the last </tr> and append to previous row instead
                                    if html_table and html_table[-1] == "</tr>":
                                        html_table.pop()  # remove </tr>
                                        # Find and update the first cell of previous row
                                        for _j in range(len(html_table)-1, -1, -1):
                                            if html_table[_j].startswith("<td"):
                                                prev = html_table[_j]
                                                # Append the continuation text before closing tag
                                                html_table[_j] = prev.replace("</td>", f" {cells[0]}</td>")
                                                break
                                        html_table.append("</tr>")
                                    continue
                                for cell in cells:
                                    cell_escaped = cell.replace("$", "&#36;")
                                    style = style_th if tag == "th" else style_td
                                    html_table.append(f"<{tag} style=\"{style}\">{cell_escaped}</{tag}>")
                                html_table.append("</tr>")
                            html_table.append("</table>")
                            result.append("\n".join(html_table))
                        else:
                            # Regular text line — escape and add
                            escaped = html.escape(line).replace("$", "&#36;")
                            result.append(escaped)
                            i += 1
                    return "\n".join(result)

                import re as _re
                _resp = message["content"]
                _yrs = sorted(set(_re.findall(r"\b(20[12][0-9])\b", message.get("question",""))), reverse=True)
                for _i, _yr in enumerate(_yrs[:4], 1):
                    _resp = _resp.replace(f"YEAR{_i} ($M)", f"{_yr} ($M)")
                    _resp = _resp.replace(f"YEAR{_i}($M)", f"{_yr}($M)")
                    _resp = _resp.replace(f"YEAR{_i}", _yr)
                display_text = convert_table_to_html(_resp)
                st.markdown(
                    f'<div style="white-space: pre-wrap; word-wrap: break-word; font-family: inherit; font-size: 0.95rem; line-height: 1.6;">{display_text}</div>',
                    unsafe_allow_html=True
                )
            st.divider()

# ── DASHBOARD TAB ─────────────────────────────────────────
with tab_dashboard:
    if not db_ok:
        st.info("📊 Run `python ingest_data.py --xbrl-only` to load data.")
        st.stop()

    from src.database.db_manager import (
        get_metric, sql_gross_margin, sql_company_comparison
    )
    from src.chains.calculator import format_currency

    ALL_COMPANIES = ["Tesla", "Microsoft", "Apple"]
    COLORS        = {"Tesla": "#CC0000", "Microsoft": "#00A4EF", "Apple": "#555555"}

    # ── Read last question intent from chat session ────────
    last_intent = None
    if st.session_state.get("messages"):
        last_intent = st.session_state.messages[-1].get("routing")

    # Defaults if no question asked yet
    active_companies = last_intent["companies"] if last_intent else ALL_COMPANIES
    active_year      = last_intent["years"][0]  if last_intent else 2023
    active_metric    = last_intent["metrics"][0] if last_intent else "Revenue"
    prior_year       = active_year - 1

    # ── Dashboard header showing what it is based on ──────
    if last_intent:
        st.info(
            f"📊 Showing charts based on your last question — "
            f"**{', '.join(active_companies)}** · "
            f"**{active_metric}** · "
            f"**{active_year}**"
        )
    else:
        st.info("📊 Ask a question in the Chat tab to see dynamic charts here.")

    # ── Chart 1: Metric trend for active companies ─────────
    st.subheader(f"📈 {active_metric} Trend — {', '.join(active_companies)}")
    trend_data = []
    for company in active_companies:
        for r in get_metric(company, active_metric):
            trend_data.append({
                "Company":            company,
                "Year":               r["year"],
                f"{active_metric} ($B)": r["value"] / 1e9,
            })
    if trend_data:
        df_trend = pd.DataFrame(trend_data)
        fig1 = px.line(
            df_trend, x="Year", y=f"{active_metric} ($B)",
            color="Company", color_discrete_map=COLORS,
            markers=True, title=f"{active_metric} Over Time",
        )
        fig1.update_layout(hovermode="x unified", height=400)
        st.plotly_chart(fig1, use_container_width=True)

    # ── Chart 2: Key metrics cards for active year ─────────
    st.subheader(f"🏆 Key Metrics — {active_year}")
    cols = st.columns(len(active_companies))
    icons = {"Tesla": "🚗", "Microsoft": "💻", "Apple": "🍎"}
    for i, company in enumerate(active_companies):
        with cols[i]:
            st.markdown(f"### {icons.get(company, '🏢')} {company}")
            for metric in ["Revenue", "Net Income", "Gross Profit"]:
                rows = get_metric(company, metric, active_year)
                if rows:
                    val        = rows[0]["value"]
                    unit       = rows[0].get("unit", "USD")
                    rows_prior = get_metric(company, metric, prior_year)
                    delta      = None
                    if rows_prior:
                        pct   = (val - rows_prior[0]["value"]) / abs(rows_prior[0]["value"]) * 100
                        delta = f"{pct:+.1f}% vs {prior_year}"
                    st.metric(metric, format_currency(val, unit), delta=delta)

    # ── Chart 3: Active metric comparison bar chart ────────
    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader(f"💰 {active_metric} Comparison — {active_year}")
        bar_data = []
        for company in active_companies:
            rows = get_metric(company, active_metric, active_year)
            if rows:
                bar_data.append({
                    "Company":            company,
                    f"{active_metric} ($B)": rows[0]["value"] / 1e9,
                })
        if bar_data:
            df_bar = pd.DataFrame(bar_data).sort_values(
                f"{active_metric} ($B)", ascending=False
            )
            fig2 = px.bar(
                df_bar, x="Company", y=f"{active_metric} ($B)",
                color="Company", color_discrete_map=COLORS, text_auto=".1f",
            )
            fig2.update_traces(
                texttemplate="$%{y:.1f}B", textposition="outside"
            )
            st.plotly_chart(fig2, use_container_width=True)

    # ── Chart 4: Gross margin for active companies ─────────
    with col_right:
        years_range = [active_year - 2, active_year - 1, active_year]
        st.subheader(f"📐 Gross Margin % — {years_range[0]}–{active_year}")
        margin_data = []
        for company in active_companies:
            for year in years_range:
                gm = sql_gross_margin(company, year)
                if gm and gm.get("gross_margin_pct"):
                    margin_data.append({
                        "Company":         company,
                        "Year":            year,
                        "Gross Margin (%)": gm["gross_margin_pct"],
                    })
        if margin_data:
            df_gm = pd.DataFrame(margin_data)
            fig3  = px.bar(
                df_gm, x="Year", y="Gross Margin (%)", color="Company",
                barmode="group", color_discrete_map=COLORS, text_auto=True,
            )
            st.plotly_chart(fig3, use_container_width=True)

    # ── Chart 5: R&D trend for active companies ────────────
    st.subheader(f"🔬 R&D Expense Trend — {', '.join(active_companies)}")
    rd_data = []
    for company in active_companies:
        for r in get_metric(company, "R&D Expense"):
            rd_data.append({
                "Company":  company,
                "Year":     r["year"],
                "R&D ($B)": r["value"] / 1e9,
            })
    if rd_data:
        df_rd = pd.DataFrame(rd_data)
        fig4  = px.bar(
            df_rd, x="Year", y="R&D ($B)", color="Company",
            barmode="group", color_discrete_map=COLORS,
        )
        st.plotly_chart(fig4, use_container_width=True)

# ── DATA EXPLORER TAB ─────────────────────────────────────
with tab_data:
    if not db_ok:
        st.info("Run `python ingest_data.py --xbrl-only` to load data.")
        st.stop()

    from src.database.db_manager import get_all_records, get_distinct_metrics
    from src.chains.calculator import format_currency

    st.subheader("🗄️ Raw Financial Data Explorer")

    col_a, col_b, col_c = st.columns(3)
    with col_a:
        filter_companies = st.multiselect(
            "Company", ["Tesla", "Microsoft", "Apple"],
            default=["Tesla", "Microsoft", "Apple"]
        )
    with col_b:
        all_metrics    = get_distinct_metrics()
        filter_metrics = st.multiselect(
            "Metric", all_metrics,
            default=["Revenue", "Net Income"] if "Revenue" in all_metrics else []
        )
    with col_c:
        filter_years = st.multiselect(
            "Year", [2019, 2020, 2021, 2022, 2023],
            default=[2021, 2022, 2023]
        )

    all_records = get_all_records()
    df = pd.DataFrame(all_records)

    if not df.empty:
        df_filtered = df[
            df["company"].isin(filter_companies) &
            df["year"].isin(filter_years)
        ]
        if filter_metrics:
            df_filtered = df_filtered[df_filtered["metric"].isin(filter_metrics)]

        df_filtered = df_filtered.copy()
        df_filtered["year"] = df_filtered["year"].astype(str)
        df_filtered["Formatted Value"] = df_filtered.apply(
            lambda row: format_currency(row["value"], row.get("unit", "USD")), axis=1
        )
        st.dataframe(
            df_filtered[["company", "year", "metric", "Formatted Value", "unit"]].rename(
                columns={"company": "Company", "year": "Year",
                         "metric": "Metric", "unit": "Unit"}
            ).sort_values(["Company", "Metric", "Year"]),
            use_container_width=True,
            height=500,
        )
        st.caption(f"Showing {len(df_filtered):,} of {len(df):,} total records · Source: SEC EDGAR XBRL")

    if vs_ok:
        st.subheader("🔍 Vector Store (RAG)")
        from src.ingestion.vector_store import get_store_stats
        stats = get_store_stats()
        c1, c2, c3 = st.columns(3)
        c1.metric("Total Chunks", f"{stats['total_chunks']:,}")
        c2.metric("Companies",    ", ".join(str(c) for c in stats["companies"]))
        c3.metric("Sections",     len(stats.get("sections", [])))
    else:
        st.info("🔍 Vector store empty — run `python ingest_data.py` for RAG support")
