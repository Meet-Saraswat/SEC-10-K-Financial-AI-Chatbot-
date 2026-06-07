# 💼 SEC 10-K Financial AI Chatbot

> An AI-powered research assistant that reads real SEC financial filings so you don't have to.

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://python.org)
[![Streamlit](https://img.shields.io/badge/UI-Streamlit-red)](https://streamlit.io)
[![Ollama](https://img.shields.io/badge/LLM-Ollama%20%28Local%29-green)](https://ollama.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

---

## 🧭 What Is This Project?

Every year, major companies like **Tesla**, **Apple**, and **Microsoft** are required by US law to file a document called a **10-K report** with the SEC (Securities and Exchange Commission). These reports are long — often 100–200 pages — and contain critical information about a company's finances, risks, legal issues, and strategy.

Most people never read them because they're dense and technical. **This project changes that.**

This chatbot lets you ask plain-English questions about these filings — and get instant, accurate, source-cited answers. It's like having a financial analyst available 24/7, for free, on your laptop.

**Example questions you can ask:**
- *"What was Apple's revenue in 2023 compared to 2022?"*
- *"What legal issues is Tesla facing?"*
- *"How has Microsoft's cloud business grown over the past 3 years?"*
- *"What are the biggest risks Apple mentioned in their latest filing?"*

---

## ✨ Key Features

| Feature | What It Means |
|---|---|
| 🔍 **Smart Question Routing** | The system automatically detects whether your question needs financial numbers, text from documents, or both — and picks the right tool |
| 📊 **SQL-Powered Calculations** | Year-over-year growth, percentage changes, and comparisons are calculated directly using SQL — fast and accurate |
| 🧠 **RAG (AI Document Search)** | For qualitative questions (risks, legal issues, strategy), the AI searches through the actual SEC filing text and generates an answer |
| 🏠 **100% Local & Free** | Everything runs on your laptop. No API keys, no subscription fees, no data sent to the cloud |
| 📎 **Source Citations** | Every answer tells you which year, which company, and which part of the filing it came from |
| 🏢 **3 Companies Covered** | Tesla (TSLA), Apple (AAPL), Microsoft (MSFT) — data from 2019 to 2024 |
| 💬 **Conversational Chat UI** | A clean chat interface built with Streamlit — type your question and get an answer in seconds |

---

## 🏗️ How It Works (Non-Technical Overview)

Think of this system as having three parts working together:

```
You type a question
       ↓
┌─────────────────────────────────┐
│     Smart Router (Intent AI)    │  ← Figures out what kind of question it is
└─────────────────────────────────┘
       ↓                    ↓
┌──────────────┐    ┌──────────────────┐
│  SQL Database│    │  Document Search │
│  (Numbers)   │    │  (Text/RAG)      │
│              │    │                  │
│ Revenue,     │    │ Risk factors,    │
│ Net Income,  │    │ Legal issues,    │
│ Growth %...  │    │ Strategy notes.. │
└──────────────┘    └──────────────────┘
       ↓                    ↓
┌─────────────────────────────────┐
│     Local AI (Llama 3.2 via     │  ← Writes a clear, human answer
│     Ollama — runs on your Mac)  │
└─────────────────────────────────┘
       ↓
   Your Answer (with sources cited)
```

**In simple terms:**
1. Your question is analyzed to understand what type of answer is needed
2. Numbers are pulled from a database with exact SQL calculations
3. Text answers are found by searching through actual 10-K filing content
4. The local AI model combines everything into a clear, readable answer

---

## 🗂️ Project Structure

```
financial-chatbot/
│
├── app.py                        # Main application — launches the chatbot UI
├── ingest_data.py                # Downloads and processes SEC filings
├── requirements.txt              # Python packages needed
├── .env                          # Your configuration (SEC email, etc.)
│
├── src/
│   ├── chains/
│   │   ├── intent_detector.py    # Classifies: is this a numbers or text question?
│   │   ├── chatbot_engine.py     # Core engine connecting all components
│   │   ├── rag_chain.py          # Searches document text and generates answers
│   │   └── calculator.py        # Runs SQL queries for financial calculations
│   │
│   ├── database/
│   │   └── db_manager.py        # Manages SQLite database (financial figures)
│   │
│   ├── ingestion/
│   │   ├── sec_fetcher.py        # Downloads 10-K filings from SEC EDGAR
│   │   └── text_processor.py    # Cleans and chunks filing text for search
│   │
│   └── rag/
│       └── vector_store.py      # Manages ChromaDB (AI document search index)
│
└── data/
    ├── db/                       # SQLite database files
    ├── vectors/                  # ChromaDB vector store
    └── raw/                      # Downloaded SEC filing data
```

---

## 🚀 Getting Started

### What You'll Need Before Starting

- A **Mac** (these instructions are for macOS)
- **Python 3.10 or newer** — [download here](https://python.org/downloads)
- **VS Code** (recommended code editor) — [download here](https://code.visualstudio.com)
- **Ollama** (runs the AI model locally) — [download here](https://ollama.com/download)
- An internet connection (only needed during initial data download)

---

### Step 1 — Install Ollama and Download AI Models

Ollama is a free app that lets you run AI models privately on your Mac.

1. Download and install Ollama from [ollama.com/download](https://ollama.com/download)
2. Open **Terminal** (press `Cmd + Space`, type "Terminal", press Enter)
3. Run these two commands to download the required AI models:

```bash
ollama pull llama3.2:3b
ollama pull nomic-embed-text
```

> ⏱️ This may take 5–10 minutes depending on your internet speed. The models are downloaded once and stored locally.

---

### Step 2 — Download This Project

```bash
git clone https://github.com/Meet-Saraswat/SEC-10-K-Financial-AI-Chatbot-.git
cd SEC-10-K-Financial-AI-Chatbot-
```

Or download the ZIP from GitHub and extract it.

---

### Step 3 — Set Up the Python Environment

Open the project folder in VS Code, then open its terminal (`Ctrl + ~`) and run:

```bash
# Create a virtual environment (isolated space for this project's packages)
python3 -m venv venv

# Activate it
source venv/bin/activate

# Install all required packages
pip install -r requirements.txt
```

> ✅ You'll know it worked when you see `(venv)` at the start of your terminal line.

---

### Step 4 — Configure Your Settings

Open the `.env` file in VS Code and update this one line:

```
SEC_USER_AGENT=Your Full Name your@email.com
```

This is required by the SEC to identify who is downloading the filings. Use your real name and email.

---

### Step 5 — Download SEC Filing Data

```bash
python ingest_data.py --xbrl-only
```

> ⏱️ This takes about 2 minutes. It downloads financial data for Tesla, Apple, and Microsoft (2019–2024) from the official SEC EDGAR database.

To also enable text search (for questions about risks, legal issues, etc.):
```bash
python ingest_data.py
```
> ⏱️ Full ingestion takes 15–20 minutes. Only needs to be done once.

---

### Step 6 — Launch the Chatbot

Open a **new Terminal tab** and start Ollama:
```bash
ollama serve
```

Back in your VS Code terminal (with venv active):
```bash
streamlit run app.py
```

Your browser will automatically open to `http://localhost:8501` and the chatbot will be ready.

---

## 💡 Example Questions to Try

**Financial Numbers:**
- *"What was Tesla's revenue in 2023?"*
- *"Compare Apple and Microsoft net income from 2021 to 2023"*
- *"Which company had the highest gross margin in 2022?"*
- *"How much did Microsoft's revenue grow year over year from 2020 to 2024?"*

**Text / Qualitative:**
- *"What legal proceedings is Tesla involved in?"*
- *"What does Apple say about competition risks?"*
- *"What are Microsoft's main risk factors?"*
- *"How does Tesla describe its manufacturing challenges?"*

**Mixed:**
- *"How has Apple's revenue grown, and what factors do they credit for this growth?"*

---

## 🛡️ Privacy & Cost

| Topic | Detail |
|---|---|
| **Cost** | Completely free — no API keys, no subscriptions |
| **Privacy** | Everything runs locally on your machine; no data is sent to any server |
| **Internet** | Only needed during initial data download (Step 5) |
| **Data Source** | All data is from [SEC EDGAR](https://www.sec.gov/edgar) — publicly available government database |

---

## 🔧 Technology Stack

| Component | Technology | Purpose |
|---|---|---|
| **User Interface** | Streamlit | Chat window and financial dashboard |
| **AI Language Model** | Llama 3.2 (3B) via Ollama | Generates natural language answers |
| **Embeddings Model** | nomic-embed-text via Ollama | Converts text into searchable vectors |
| **Financial Database** | SQLite | Stores structured financial metrics |
| **Document Search** | ChromaDB | Vector database for semantic text search |
| **Hybrid Search** | BM25 + ChromaDB (RRF fusion) | Combines keyword and semantic search |
| **Data Source** | SEC EDGAR XBRL API | Official US government financial data |
| **Calculations** | SQL (LAG, RANK, ROUND) | Year-over-year growth, comparisons |
| **Orchestration** | LangChain | Chains together retrieval and generation |

---

## 📋 Companies & Data Coverage

| Company | Ticker | Years Covered |
|---|---|---|
| Apple Inc. | AAPL | 2019 – 2024 |
| Microsoft Corporation | MSFT | 2019 – 2024 |
| Tesla Inc. | TSLA | 2019 – 2024 |

**Financial metrics included:** Revenue, Net Income, Gross Profit, Operating Income, Gross Margin %, R&D Expenses, Total Assets, Total Liabilities, Shareholders' Equity, EPS (Basic & Diluted), Operating Cash Flow

**Qualitative sections included:** Business Overview, Risk Factors, Legal Proceedings, Management Discussion & Analysis (MD&A), and more

---

## ❓ Troubleshooting

**"Ollama not running" error**
→ Open a new Terminal window and run `ollama serve`, then try again.

**"No data / ChromaDB empty" warning**
→ Run `python ingest_data.py --xbrl-only` with your venv active.

**"ModuleNotFoundError"**
→ Make sure `(venv)` is showing in your terminal, then run `pip install -r requirements.txt`.

**App not opening in browser**
→ Manually go to `http://localhost:8501` in your browser.

---

## 📁 Data Sources

All financial data is sourced directly from:
- **SEC EDGAR XBRL API** — `https://data.sec.gov/api/xbrl/`
- **SEC EDGAR Full-Text Filings** — `https://www.sec.gov/cgi-bin/browse-edgar`

This is 100% public, legal data published by the US Securities and Exchange Commission.

---

## 📄 License

This project is licensed under the MIT License — free to use, modify, and share.

---

## 🙋 Author

**Meet Saraswat**
[github.com/Meet-Saraswat](https://github.com/Meet-Saraswat)

---

*Built as a personal project to make SEC financial research accessible to everyone — from seasoned analysts to curious students.*
