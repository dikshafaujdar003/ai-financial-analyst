import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langgraph.prebuilt import create_react_agent
from tools import (
    get_stock_data, clean_and_profile, create_chart,
    financial_model, budget_variance, data_mining, generate_report
)

load_dotenv()

llm = ChatGroq(model="meta-llama/llama-4-scout-17b-16e-instruct", temperature=0)

tools = [
    get_stock_data,
    clean_and_profile,
    create_chart,
    financial_model,
    budget_variance,
    data_mining,
    generate_report,
]

system_prompt = """You are an expert AI financial analyst. When given a task, immediately execute it 
using the available tools without asking for confirmation. Always call tools directly and complete 
the full analysis in one go. Be concise in your final summary."""

analyst_agent = create_react_agent(llm, tools, prompt=system_prompt)