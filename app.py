import streamlit as st
import os
from PIL import Image

st.set_page_config(page_title="AI Financial Analyst", layout="wide")

st.markdown("""
<style>
    .main { background-color: #0f1117; }
    .stTextInput>div>div>input { background-color: #1e2130; color: white; }
    .chat-msg { padding: 10px 14px; border-radius: 8px; margin-bottom: 8px; }
    .user-msg  { background-color: #1e3a5f; color: white; }
    .agent-msg { background-color: #1e2130; color: #e0e0e0; }
</style>
""", unsafe_allow_html=True)

st.title("🤖 AI Financial Analyst")

# ── Session state ─────────────────────────────────────────────────────────────
if "my_agent"           not in st.session_state: st.session_state.my_agent           = None
if "chat_history"       not in st.session_state: st.session_state.chat_history       = []
if "prefill"            not in st.session_state: st.session_state.prefill            = ""
if "uploaded_filename"  not in st.session_state: st.session_state.uploaded_filename  = None

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Setup")
    if st.button("🚀 Initialize Agent", use_container_width=True):
        try:
            import my_agent
            st.session_state.my_agent = my_agent.analyst_agent
            st.success("Agent is online!")
            st.info(f"🧠 {my_agent.analyst_agent.name}")
        except Exception as e:
            st.error(f"Startup error: {e}")

    st.divider()
    st.header("📂 Upload Files")
    st.caption("For Budget Analysis & Data Mining")
    uploaded = st.file_uploader("Upload CSV or Excel", type=["csv", "xlsx", "xls"])
    if uploaded:
        with open(uploaded.name, "wb") as f:
            f.write(uploaded.getbuffer())
        st.session_state.uploaded_filename = uploaded.name
        st.success(f"Saved: {uploaded.name}")

    st.divider()
    st.header("📖 Quick Actions")
    ticker  = st.text_input("Stock Ticker",  value="TSLA", placeholder="e.g. AAPL").upper().strip()
    ticker2 = st.text_input("Compare With",  value="NVDA", placeholder="e.g. MSFT").upper().strip()

    ufile = st.session_state.get("uploaded_filename")
    st.caption(f"Uploaded file: `{ufile}`" if ufile else "Uploaded file: _(none)_")

    if st.button("📈 Stock Analysis",   use_container_width=True):
        st.session_state.prefill = f"Analyze {ticker} stock, get news sentiment, and create a chart"
    if st.button("⚖️ Compare Stocks",   use_container_width=True):
        st.session_state.prefill = f"Compare {ticker} and {ticker2} and tell me which performed better"
    if st.button("💰 Financial Model",  use_container_width=True):
        st.session_state.prefill = "Build a financial model with revenue=2000000, cost_ratio=0.55, price_increase=0.05, growth_rate=0.12, tax_rate=0.21"
    if st.button("📄 PDF Report",       use_container_width=True):
        st.session_state.prefill = f"Generate a PDF report for {ticker}"
    if st.button("📰 News Sentiment",   use_container_width=True):
        st.session_state.prefill = f"Search for recent news about {ticker} and give me a sentiment summary"

    if ufile:
        if st.button("📊 Budget Variance", use_container_width=True):
            st.session_state.prefill = f"Run budget variance analysis on {ufile}"
        if st.button("🔍 Data Mining",     use_container_width=True):
            st.session_state.prefill = f"Mine the data in {ufile}"
    else:
        st.button("📊 Budget Variance (upload file first)", disabled=True, use_container_width=True)
        st.button("🔍 Data Mining (upload file first)",     disabled=True, use_container_width=True)

    st.divider()
    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.chat_history = []
        if st.session_state.my_agent:
            st.session_state.my_agent.clear_memory()
        st.rerun()

# ── Chat history ──────────────────────────────────────────────────────────────
if st.session_state.chat_history:
    st.subheader("💬 Conversation")
    for msg in st.session_state.chat_history:
        cls  = "user-msg" if msg["role"] == "user" else "agent-msg"
        icon = "🧑" if msg["role"] == "user" else "🤖"
        st.markdown(f'<div class="chat-msg {cls}">{icon} {msg["content"]}</div>',
                    unsafe_allow_html=True)

# ── Main input & run ──────────────────────────────────────────────────────────
if st.session_state.my_agent:
    query = st.text_input(
        "What should I analyze?",
        value=st.session_state.prefill,
        placeholder="e.g. Analyze NVDA stock and generate a full report"
    )
    st.session_state.prefill = ""

    if query:
        st.session_state.chat_history.append({"role": "user", "content": query})

        # ── Snapshot files BEFORE agent runs ─────────────────────────────────
        before_files = set(
            f for f in os.listdir(".")
            if f.endswith((".png", ".pdf", ".xlsx"))
        )
        # Also record modification times so we catch overwritten files
        before_mtimes = {
            f: os.path.getmtime(f) for f in before_files
        }

        with st.status("🔍 Agent is working...", expanded=True) as status:
            def step_callback(step_type, content):
                if step_type == "plan":
                    st.write(f"📋 **Plan:**\n{content}")
                elif step_type == "thought":
                    st.write(f"💭 **Thought:** {content}")
                elif step_type == "delegate":
                    st.write(f"📨 **Delegating:** {content}")
                elif step_type == "specialist":
                    st.write(f"🤖 **Specialist:** {content}")
                elif step_type == "action":
                    st.write(f"⚙️ **Action:** {content}")
                elif step_type == "observation":
                    st.write(f"👁️ **Observation:** {content[:300]}")
                elif step_type == "reflection":
                    st.write(f"🪞 **Self-Review:** {content}")
                elif step_type == "final":
                    st.write("✅ **Done!**")

            try:
                response = st.session_state.my_agent.invoke(query, step_callback=step_callback)
                status.update(label="✅ Analysis complete!", state="complete")
            except Exception as e:
                response = f"❌ Error: {e}"
                status.update(label="❌ Error occurred", state="error")

        # ── Snapshot files AFTER agent runs ──────────────────────────────────
        after_files = set(f for f in os.listdir(".") if f.endswith((".png", ".pdf", ".xlsx")))
        # New files OR files that were overwritten (mtime changed)
        new_files = {
            f for f in after_files
            if f not in before_files
            or os.path.getmtime(f) > before_mtimes.get(f, 0)
        }

        st.session_state.chat_history.append({"role": "agent", "content": response})
        st.markdown(f"**🤖 Response:** {response}")

        # ── Show only charts generated by THIS query ──────────────────────────
        new_charts = [f for f in new_files if f.endswith(".png")]
        if new_charts:
            st.subheader("📊 Generated Charts")
            cols = st.columns(min(len(new_charts), 2))
            for i, chart in enumerate(sorted(new_charts)):
                with cols[i % 2]:
                    st.image(Image.open(chart), caption=chart, use_container_width=True)

        # ── Download buttons for THIS query's files only ──────────────────────
        new_pdfs   = [f for f in new_files if f.endswith(".pdf")]
        new_excels = [f for f in new_files if f.endswith(".xlsx")]

        if new_pdfs or new_excels:
            st.subheader("📥 Downloads")
            dl_cols = st.columns(max(len(new_pdfs) + len(new_excels), 1))
            col_idx = 0

            for pdf in new_pdfs:
                with open(pdf, "rb") as f:
                    dl_cols[col_idx].download_button(
                        f"📄 {pdf}", f, file_name=pdf, mime="application/pdf"
                    )
                col_idx += 1

            for xlsx in new_excels:
                with open(xlsx, "rb") as f:
                    dl_cols[col_idx].download_button(
                        f"📥 {xlsx}", f, file_name=xlsx,
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )
                col_idx += 1

else:
    st.info("👈 Click **Initialize Agent** in the sidebar to get started.")