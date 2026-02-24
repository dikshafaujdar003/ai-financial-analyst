import os
import re
from dotenv import load_dotenv
from groq import Groq
from tools import (
    get_stock_data, clean_and_profile, create_chart,
    search_news, compare_stocks,
    financial_model, budget_variance, data_mining,
    generate_report,
)

# Explicitly load .env from project folder
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), ".env"))


# ── Base specialist agent ──────────────────────────────────────────────────────
class SpecialistAgent:
    """
    A lightweight single-purpose ReAct agent.
    Each specialist only knows about its own tools.
    """
    def __init__(self, name: str, tools: dict, system_prompt: str, max_steps: int = 10):
        self.name          = name
        self.tools         = tools
        self.system_prompt = system_prompt
        self.max_steps     = max_steps
        self.client        = Groq()
        self.model         = "meta-llama/llama-4-scout-17b-16e-instruct"

    def _call_llm(self, messages: list) -> str:
        r = self.client.chat.completions.create(
            model=self.model, messages=messages,
            temperature=0, max_tokens=1024,
        )
        return r.choices[0].message.content.strip()

    def _parse(self, text: str):
        if "Final Answer:" in text:
            return "FINAL", text.split("Final Answer:")[-1].strip()
        am = re.search(r"Action:\s*(\w+)", text)
        im = re.search(r"Input:\s*(.+?)(?:\n|$)", text, re.DOTALL)
        if am and im:
            return am.group(1).strip(), im.group(1).strip()
        return None, None

    def _run_tool(self, name: str, inp: str) -> str:
        if name not in self.tools:
            return f"Error: '{name}' not available. My tools: {list(self.tools)}"
        try:
            return self.tools[name].invoke(inp)
        except Exception as e:
            return f"Tool error: {e}"

    def run(self, task: str, step_callback=None) -> str:
        """Execute a task using this specialist's tools."""
        if step_callback:
            step_callback("specialist", f"**{self.name}** received task: {task}")

        # Force immediate tool calling by prepending a directive
        first_message = (
            f"{task}\n\n"
            f"IMPORTANT: Call a tool RIGHT NOW. Do not explain, do not ask questions. "
            f"Your first response must be a tool call using Action/Input format."
        )

        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user",   "content": first_message},
        ]
        observations = {}

        for step in range(self.max_steps):
            out = self._call_llm(messages)
            tool_name, tool_input = self._parse(out)

            # Block premature Final Answer if no tools have been called yet
            if tool_name == "FINAL" and not observations:
                messages.append({"role": "assistant", "content": out})
                messages.append({"role": "user", "content":
                    "You have NOT called any tools yet. You MUST call a tool first. "
                    "Use Action/Input format right now."})
                continue

            if tool_name == "FINAL":
                obs_text = "\n\n".join(f"[{t}]:\n{o}" for t, o in observations.items())
                grounded = self._call_llm([
                    {"role": "system", "content":
                        "You are a financial analyst. Answer using ONLY the data below. "
                        "Do not add any external knowledge, assumptions, or general facts."},
                    {"role": "user", "content":
                        f"Real data from tools:\n{obs_text}\n\nTask: {task}\n\n"
                        f"Write a clear answer using only these exact numbers."},
                ])
                return grounded

            if tool_name:
                if step_callback:
                    step_callback("action", f"[{self.name}] → `{tool_name}({tool_input})`")
                obs = self._run_tool(tool_name, tool_input)
                observations[tool_name] = obs
                if step_callback:
                    step_callback("observation", f"[{self.name}] {obs[:400]}")
                messages.append({"role": "assistant", "content": out})
                messages.append({"role": "user", "content":
                    f"Observation: {obs}\n\n"
                    f"Good. Now either call the next tool or give your Final Answer."})
            else:
                # LLM gave no tool call — force it
                messages.append({"role": "assistant", "content": out})
                messages.append({"role": "user", "content":
                    f"You did not call a tool. You MUST respond with:\n"
                    f"Thought: <reason>\nAction: <tool_name>\nInput: <input>\n\n"
                    f"Available tools: {list(self.tools.keys())}"})

        # Fallback — return raw observations if max steps hit
        if observations:
            obs_text = "\n\n".join(f"[{t}]:\n{o}" for t, o in observations.items())
            # Still try to produce a grounded answer from whatever we got
            return self._call_llm([
                {"role": "system", "content":
                    "Summarize the following tool results clearly. Use only this data."},
                {"role": "user", "content": f"Data:\n{obs_text}\n\nTask: {task}"},
            ])
        return f"Error: {self.name} could not complete the task. Please try again."


# ── 1. Stock Agent ─────────────────────────────────────────────────────────────
StockAgent = SpecialistAgent(
    name="Stock Agent",
    tools={
        "get_stock_data":    get_stock_data,
        "clean_and_profile": clean_and_profile,
        "create_chart":      create_chart,
        "search_news":       search_news,
        "compare_stocks":    compare_stocks,
    },
    system_prompt="""You are a stock market analyst specialist.
Your job: download stock data, profile it, create charts, fetch news sentiment, compare stocks.

Available tools and when to use them:
- get_stock_data(ticker) → download price data
- clean_and_profile(ticker) → get statistics
- create_chart(ticker) → generate chart PNG
- search_news(ticker) → get REAL news headlines and sentiment. Use THIS for any news/sentiment task. NEVER use data_mining for news.
- compare_stocks(ticker1,ticker2) → compare two stocks side by side

To call a tool:
Thought: <reasoning>
Action: <tool_name>
Input: <input>

When done:
Thought: I have all the data needed.
Final Answer: <summary using only real data from observations>

NEVER invent numbers. NEVER use data_mining for news — always use search_news.
Only use data from tool observations.""",
)

# ── 2. Modeling Agent ──────────────────────────────────────────────────────────
ModelingAgent = SpecialistAgent(
    name="Modeling Agent",
    tools={
        "financial_model": financial_model,
        "budget_variance": budget_variance,
        "data_mining":     data_mining,
    },
    system_prompt="""You are a financial modeling specialist.
Your job: build financial models, analyze budget variance, mine datasets.

Available tools:
- financial_model(assumptions) → build 3-year P&L model. Pass as: 'revenue=X, cost_ratio=X, price_increase=X, growth_rate=X, tax_rate=X'
- budget_variance(filename) → analyze budget vs actuals from uploaded file
- data_mining(filename) → profile and clean any uploaded dataset

To call a tool:
Thought: <reasoning>
Action: <tool_name>
Input: <input>

When done:
Thought: I have all the data needed.
Final Answer: <summary using only real data from observations>

NEVER invent numbers. Only use data from tool observations.""",
)

# ── 3. Report Agent ────────────────────────────────────────────────────────────
ReportAgent = SpecialistAgent(
    name="Report Agent",
    tools={
        "generate_report": generate_report,
    },
    system_prompt="""You are a financial report generation specialist.
Your ONLY job is to generate PDF reports for stocks using the generate_report tool.

Available tools:
- generate_report(ticker) → generate a PDF report for a stock ticker

To call a tool:
Thought: I will generate the PDF report.
Action: generate_report
Input: <ticker symbol only, e.g. AAPL>

When done:
Thought: I have all the data needed.
Final Answer: <confirm what was generated>

Always call generate_report immediately. Do not ask questions.""",
)