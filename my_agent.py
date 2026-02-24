import re
import os
from dotenv import load_dotenv
from groq import Groq
from sub_agents import StockAgent, ModelingAgent, ReportAgent

# Explicitly load .env from project folder
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), ".env"))

# ── Prompts ────────────────────────────────────────────────────────────────────
ORCHESTRATOR_SYSTEM = """You are a master AI financial analyst that orchestrates specialist agents.

You have 3 specialist agents you can delegate to:
1. StockAgent     — stock data, charts, news sentiment, stock comparison
2. ModelingAgent  — financial models, budget variance, data mining
3. ReportAgent    — PDF report generation ONLY

ROUTING RULES (follow strictly):
- Any task involving stock prices, charts, news, sentiment → StockAgent
- Any task involving financial models, budgets, data files → ModelingAgent
- Any task involving "PDF", "report", "generate report" → ReportAgent
- Always delegate PDF generation as a SEPARATE final step to ReportAgent

To delegate to a specialist, respond EXACTLY like this:
Thought: <your reasoning>
Delegate: <StockAgent | ModelingAgent | ReportAgent>
Task: <specific task for that agent>

When ALL tasks are done and you have all results, respond EXACTLY like this:
Thought: I have all results needed.
Final Answer: <your synthesized response>

Rules:
- Always plan before delegating
- Delegate each part to the right specialist — NEVER skip ReportAgent when PDF is requested
- Never do analysis yourself — always delegate
- Synthesize all specialist results into one coherent final answer
"""

PLANNING_PROMPT = """You are a financial analysis planner.
Given the user's request, produce a step-by-step plan listing:
- Which specialist agent handles each step (StockAgent, ModelingAgent, ReportAgent)
- What specific task each agent will perform

Format as a numbered list. Be specific. Do not execute — just plan.
"""

REFLECTION_PROMPT = """You are a critical financial analyst reviewing an AI response.

Original question: {question}
AI response: {answer}

Evaluate fairly:
1. Completeness — did it answer the main question?
2. Accuracy — are numbers from real data, not assumptions?
3. Note: if the response mentions files were saved (Excel, PDF, PNG), that counts as complete for those outputs

Score generously — if the main question was answered with real data, score 7+.
Only score below 7 if key numbers are clearly missing or hallucinated.

Respond in this EXACT format:
Score: <1-10>/10
Verdict: <PASS or NEEDS_IMPROVEMENT>
Feedback: <one sentence>
"""


# ── Orchestrator Agent ─────────────────────────────────────────────────────────
class OrchestratorAgent:
    name = "Orchestrator + Sub-agents (Phase 3)"

    def __init__(self, max_steps: int = 8, max_retries: int = 2):
        self.client      = Groq()
        self.model       = "meta-llama/llama-4-scout-17b-16e-instruct"
        self.max_steps   = max_steps
        self.max_retries = max_retries
        self.memory: list[dict] = []
        self.agents = {
            "StockAgent":    StockAgent,
            "ModelingAgent": ModelingAgent,
            "ReportAgent":   ReportAgent,
        }

    def _call_llm(self, messages: list, temperature: float = 0) -> str:
        r = self.client.chat.completions.create(
            model=self.model, messages=messages,
            temperature=temperature, max_tokens=1024,
        )
        return r.choices[0].message.content.strip()

    def _parse(self, text: str):
        """Parse orchestrator output — either Delegate or Final Answer."""
        if "Final Answer:" in text:
            return "FINAL", text.split("Final Answer:")[-1].strip()
        dm = re.search(r"Delegate:\s*(StockAgent|ModelingAgent|ReportAgent)", text)
        tm = re.search(r"Task:\s*(.+?)(?:\n|$)", text, re.DOTALL)
        if dm and tm:
            return dm.group(1).strip(), tm.group(1).strip()
        return None, None

    # ── Planning ───────────────────────────────────────────────────────────────
    def _plan(self, query: str, step_callback=None) -> str:
        plan = self._call_llm([
            {"role": "system", "content": PLANNING_PROMPT},
            {"role": "user",   "content": query},
        ])
        if step_callback:
            step_callback("plan", plan)
        return plan

    # ── Reflection ─────────────────────────────────────────────────────────────
    def _reflect(self, question: str, answer: str, step_callback=None) -> dict:
        out = self._call_llm([
            {"role": "system", "content": REFLECTION_PROMPT.format(
                question=question, answer=answer)},
            {"role": "user",   "content": "Evaluate the response."},
        ])
        sm = re.search(r"Score:\s*(\d+)/10",                 out)
        vm = re.search(r"Verdict:\s*(PASS|NEEDS_IMPROVEMENT)", out)
        fm = re.search(r"Feedback:\s*(.+)",                   out)
        score    = int(sm.group(1))    if sm else 5
        verdict  = vm.group(1)         if vm else "PASS"
        feedback = fm.group(1).strip() if fm else ""
        if step_callback:
            step_callback("reflection", f"Score: {score}/10 | {verdict} | {feedback}")
        return {"score": score, "verdict": verdict, "feedback": feedback}

    # ── Main invoke ────────────────────────────────────────────────────────────
    def invoke(self, user_query: str, step_callback=None) -> str:

        # 1. Plan
        plan = self._plan(user_query, step_callback)

        # 2. Build messages with memory
        messages = [{"role": "system", "content": ORCHESTRATOR_SYSTEM}]
        if self.memory:
            mem_text = "Previous context:\n" + "\n".join(
                f"{'User' if m['role']=='user' else 'Assistant'}: {m['content'][:300]}"
                for m in self.memory[-6:]
            )
            messages += [
                {"role": "user",      "content": mem_text},
                {"role": "assistant", "content": "Understood, I have the context."},
            ]
        messages.append({"role": "user", "content":
            f"Plan:\n{plan}\n\nNow execute it. User request: {user_query}"})

        session   = list(messages)
        results   = {}   # specialist_name → result
        final_ans = None

        # 3. Orchestration loop
        for step in range(self.max_steps):
            llm_out = self._call_llm(session)

            agent_name, task = self._parse(llm_out)

            # Show orchestrator's thought
            tm = re.search(r"Thought:\s*(.+?)(?:\nDelegate:|$)", llm_out, re.DOTALL)
            if step_callback and tm:
                step_callback("thought", f"Orchestrator: {tm.group(1).strip()}")

            if agent_name == "FINAL":
                final_ans = task
                if step_callback:
                    step_callback("final", "Orchestrator producing final answer...")
                break

            if agent_name and agent_name in self.agents:
                if step_callback:
                    step_callback("delegate", f"Delegating to **{agent_name}**: {task}")

                specialist = self.agents[agent_name]

                # Retry once if specialist fails
                result = specialist.run(task, step_callback=step_callback)
                if "could not complete" in result.lower() or "max steps" in result.lower():
                    if step_callback:
                        step_callback("thought", f"Retrying {agent_name}...")
                    result = specialist.run(task, step_callback=step_callback)

                results[agent_name] = result

                session.append({"role": "assistant", "content": llm_out})
                session.append({"role": "user",
                    "content": f"{agent_name} completed. Result:\n{result}"})
            else:
                session.append({"role": "assistant", "content": llm_out})
                session.append({"role": "user",
                    "content": "Please delegate to a specialist or provide Final Answer."})

        # 4. Synthesize all specialist results into final answer
        if results:
            results_text = "\n\n".join(
                f"=== {agent} Result ===\n{res}" for agent, res in results.items()
            )
            final_ans = self._call_llm([
                {"role": "system",    "content":
                    "You are a financial analyst. Synthesize the specialist reports below "
                    "into one clear, comprehensive answer. Use ONLY data from the reports. "
                    "No assumptions or external knowledge."},
                {"role": "user",      "content":
                    f"User question: {user_query}\n\nSpecialist Reports:\n{results_text}\n\n"
                    f"Write a clear, complete answer using only these results."},
            ], temperature=0)

        if not final_ans:
            final_ans = "Could not complete the analysis."

        # 5. Reflect and improve if needed
        for _ in range(2):
            reflection = self._reflect(user_query, final_ans, step_callback)
            if reflection["verdict"] == "PASS":
                final_ans += f"\n\n✅ **Confidence: {reflection['score']}/10**"
                break
            if step_callback:
                step_callback("thought",
                    f"Score {reflection['score']}/10 — improving: {reflection['feedback']}")
            final_ans = self._call_llm([
                {"role": "system",    "content":
                    "Improve this financial analysis response. "
                    "Use only data already present — no new assumptions."},
                {"role": "user",      "content":
                    f"Original question: {user_query}\n"
                    f"Current answer: {final_ans}\n"
                    f"Improvement needed: {reflection['feedback']}\n"
                    f"Rewrite using only data already in the answer."},
            ], temperature=0.1)
        else:
            final_ans += f"\n\n⚠️ **Confidence: {reflection['score']}/10** — {reflection['feedback']}"

        # 6. Save to memory
        self.memory.append({"role": "user",      "content": user_query})
        self.memory.append({"role": "assistant", "content": final_ans})
        if len(self.memory) > 20:
            self.memory = self.memory[-20:]

        return final_ans

    def clear_memory(self):
        self.memory = []


# ── Singleton ──────────────────────────────────────────────────────────────────
analyst_agent = OrchestratorAgent()