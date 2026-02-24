# 🤖 AI Financial Analyst

An autonomous AI-powered financial analysis platform built with Python and Streamlit. The app uses a custom-built multi-agent ReAct architecture to automate financial tasks including stock analysis, financial modeling, budget variance analysis, news sentiment, and PDF report generation.

---

## 🏗️ Architecture

The app is built on a **custom ReAct (Reasoning + Acting) agent loop** implemented from scratch — not using LangChain's prebuilt agents. On top of ReAct, it implements:

- **Plan-and-Execute** — the orchestrator plans all steps before acting
- **Multi-agent orchestration** — a master orchestrator delegates to 3 specialist agents
- **Reflection loop** — agents self-critique and improve answers scoring below 7/10
- **Confidence scoring** — every response is rated 1-10 for reliability
- **Conversation memory** — context is retained across queries in the same session

### Specialist Agents

| Agent | Responsibility | Tools |
|-------|---------------|-------|
| **StockAgent** | Stock data, charts, news, comparison | `get_stock_data`, `clean_and_profile`, `create_chart`, `search_news`, `compare_stocks` |
| **ModelingAgent** | Financial models, budgets, data mining | `financial_model`, `budget_variance`, `data_mining` |
| **ReportAgent** | PDF report generation | `generate_report` |

---

## ✨ Features

- 📈 **Stock Analysis** — download 1 year of price data, statistical profiling, 50-day MA chart
- 📰 **News Sentiment** — real-time news via DuckDuckGo, bullish/bearish/neutral scoring
- ⚖️ **Stock Comparison** — side-by-side comparison with ROI, Sharpe ratio, max drawdown
- 💰 **Financial Modeling** — 3-year P&L what-if projections exported to Excel
- 📊 **Budget Variance** — upload budget vs actuals CSV, variance analysis with charts
- 🔍 **Data Mining** — profile and clean any uploaded CSV/Excel dataset
- 📄 **PDF Reports** — downloadable reports with charts and key statistics
- 🧠 **Memory** — remembers previous queries within a session

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| Frontend | Streamlit |
| LLM | Llama 4 via Groq API (free) |
| Agent Framework | Custom ReAct loop (built from scratch) |
| Data | yfinance, DuckDuckGo Search |
| Visualization | Matplotlib |
| Export | openpyxl (Excel), fpdf2 (PDF) |

---

## 🚀 Getting Started

### 1. Clone the repo
```bash
git clone https://github.com/dikshafaujdar003/ai-financial-analyst.git
cd ai-financial-analyst
```

### 2. Create a virtual environment
```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
source .venv/bin/activate     # Mac/Linux
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Set up your API key
Create a `.env` file in the root folder:
```
GROQ_API_KEY=your_groq_api_key_here
```
Get a free Groq API key at [console.groq.com](https://console.groq.com)

### 5. Run the app
```bash
streamlit run app.py
```

---

## 📁 Project Structure

```
ai-financial-analyst/
├── app.py              # Streamlit UI
├── my_agent.py         # Orchestrator ReAct agent
├── sub_agents.py       # Specialist agents (Stock, Modeling, Report)
├── tools.py            # All 8 financial tools
├── agent.py            # Original LangGraph agent (backup)
├── requirements.txt
├── .env                # API keys (not committed)
└── .gitignore
```

---

## 💡 Example Queries

```
Analyze AAPL stock, get recent news sentiment, and create a chart
Compare NVDA and MSFT and tell me which was the better investment this year
Build a financial model with revenue=2000000, cost_ratio=0.60, growth_rate=0.20, price_increase=0.05, tax_rate=0.21
Analyze TSLA stock and generate a PDF report
Run budget variance analysis on budget.csv
```

---

## 🎯 Key Technical Decisions

**Why build ReAct from scratch?**
LangChain's `create_react_agent` was removed in v1.2+. Building from scratch demonstrates understanding of the Thought → Action → Observation loop rather than just using a black-box framework.

**Why Groq?**
Free tier with fast inference on Llama 4, no credit card required. Easily swappable with OpenAI or Gemini by changing 2 lines in `my_agent.py`.

**Why multi-agent?**
Each specialist has a focused system prompt and limited tool set, which reduces hallucination compared to a single agent with all tools.


The app is built on a **custom ReAct (Reasoning + Acting) agent loop** implemented from scratch — not using LangChain's prebuilt agents. On top of ReAct, it implements:

- **Plan-and-Execute** — the orchestrator plans all steps before acting
- **Multi-agent orchestration** — a master orchestrator delegates to 3 specialist agents
- **Reflection loop** — agents self-critique and improve answers scoring below 7/10
- **Confidence scoring** — every response is rated 1-10 for reliability
- **Conversation memory** — context is retained across queries in the same session

### Specialist Agents
| Agent | Responsibility | Tools |
|-------|---------------|-------|
| **StockAgent** | Stock data, charts, news, comparison | `get_stock_data`, `clean_and_profile`, `create_chart`, `search_news`, `compare_stocks` |
| **ModelingAgent** | Financial models, budgets, data mining | `financial_model`, `budget_variance`, `data_mining` |
| **ReportAgent** | PDF report generation | `generate_report` |

## ✨ Features

- 📈 **Stock Analysis** — download 1 year of price data, statistical profiling, 50-day MA chart
- 📰 **News Sentiment** — real-time news via DuckDuckGo, bullish/bearish/neutral scoring
- ⚖️ **Stock Comparison** — side-by-side comparison with ROI, Sharpe ratio, max drawdown
- 💰 **Financial Modeling** — 3-year P&L what-if projections exported to Excel
- 📊 **Budget Variance** — upload budget vs actuals CSV, variance analysis with charts
- 🔍 **Data Mining** — profile and clean any uploaded CSV/Excel dataset
- 📄 **PDF Reports** — downloadable reports with charts and key statistics
- 🧠 **Memory** — remembers previous queries within a session

## 🛠️ Tech Stack

- **Frontend** — Streamlit
- **LLM** — Llama 4 via Groq API (free)
- **Agent Framework** — Custom ReAct loop (built from scratch)
- **Data** — yfinance, DuckDuckGo Search
- **Visualization** — Matplotlib
- **Export** — openpyxl (Excel), fpdf2 (PDF)

## 🚀 Getting Started

### 1. Clone the repo
```bash
git clone https://github.com/YOUR_USERNAME/ai-financial-analyst.git
cd ai-financial-analyst
```

### 2. Create a virtual environment
```bash
python -m venv .venv
.venv\Scripts\activate  # Windows
source .venv/bin/activate  # Mac/Linux
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Set up your API key
Create a `.env` file in the root folder:
```
GROQ_API_KEY=your_groq_api_key_here
```
Get a free Groq API key at [console.groq.com](https://console.groq.com)

### 5. Run the app
```bash
streamlit run app.py
```

## 📁 Project Structure

```
ai-financial-analyst/
├── app.py              # Streamlit UI
├── my_agent.py         # Orchestrator ReAct agent
├── sub_agents.py       # Specialist agents (Stock, Modeling, Report)
├── tools.py            # All 8 financial tools
├── agent.py            # Original LangGraph agent (backup)
├── requirements.txt
├── .env                # API keys (not committed)
└── .gitignore
```

## 💡 Example Queries

```
Analyze AAPL stock, get recent news sentiment, and create a chart
Compare NVDA and MSFT and tell me which was the better investment this year
Build a financial model with revenue=2000000, cost_ratio=0.60, growth_rate=0.20, price_increase=0.05, tax_rate=0.21
Analyze TSLA stock and generate a PDF report
Run budget variance analysis on budget.csv
```

## 🎯 Key Technical Decisions

**Why build ReAct from scratch?**
LangChain's `create_react_agent` was removed in v1.2+, migrated to LangGraph. Building from scratch demonstrates understanding of the underlying Thought → Action → Observation loop rather than just using a black-box framework.

**Why Groq?**
Free tier with fast inference on Llama 4, no credit card required. Easily swappable with OpenAI or Gemini by changing 2 lines in `my_agent.py`.

**Why multi-agent?**
Each specialist has a focused system prompt and limited tool set, which reduces hallucination compared to a single agent with all tools.