import os
os.environ["STREAMLIT_SERVER_FILE_WATCHER_TYPE"] = "none"
os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import streamlit as st
import re, json, time, glob, hashlib
from typing import List, Dict, Tuple

# ---------- Cache the model once ----------
@st.cache_resource(show_spinner=False)
def get_llm():
    try:
        from transformers import pipeline
        return pipeline("text2text-generation", model="google/flan-t5-base", device=-1)  # CPU
    except Exception:
        return None

# ---------- Helpers ----------
def sanitize_output(text: str) -> str:
    """Remove markdown headings and trim to avoid huge H1 outputs."""
    text = re.sub(r'^\s*#{1,6}\s*', '', text, flags=re.MULTILINE)
    return text.strip()

def infer_topic(q: str) -> str:
    ql = q.lower()
    if any(k in ql for k in ["apr", "apy", "annual percentage rate", "credit card interest"]):
        return "APR"
    if "emergency" in ql:
        return "EMERGENCY"
    if "budget" in ql or "50/30/20" in ql or "50 30 20" in ql:
        return "BUDGET"
    if "snowball" in ql or "avalanche" in ql or "debt" in ql:
        return "DEBT"
    if "savings goal" in ql or "save per month" in ql:
        return "GOAL"
    return "GENERAL"

def fallback_answer(topic: str, role: str, income: float, fixed: float, ctx_items, user_q: str) -> str:
    if topic == "APR":
        example_apr = 0.24
        daily_rate = example_apr / 365
        example_balance = 1000
        days = 30
        interest = example_balance * daily_rate * days
        return (
            "APR (Annual Percentage Rate) is the yearly borrowing rate without compounding. "
            "APY includes compounding (used for savings). For revolving credit like credit cards, "
            "interest is computed daily using the daily periodic rate.\n\n"
            "**Formula**\n"
            "```\n"
            "daily_rate = APR / 365\n"
            "cycle_interest = average_daily_balance × daily_rate × days\n"
            "```\n\n"
            "**Example**\n"
            f"APR 24% ⇒ daily_rate = {daily_rate:.5f}\n\n"
            "```\n"
            f"interest = {example_balance} × {daily_rate:.5f} × {days}\n"
            f"≈ ${interest:.2f}\n"
            "```\n\n"
            "Tip: Pay the full statement balance by the due date to avoid interest; if you carry a balance, "
            "larger payments and a lower APR reduce total cost."
        )

    if topic == "EMERGENCY":
        low = fixed * 3
        high = fixed * 6
        return (
            "**Emergency Fund** is cash reserved for unexpected expenses. "
            "A common target is **3–6 months of essential expenses** (rent, utilities, groceries, transport).\n\n"
            "**Formula**\n"
            "```\n"
            "emergency_fund = monthly_fixed_expenses × months\n"
            "```\n\n"
            "**Your Numbers**\n"
            "```\n"
            f"Monthly fixed expenses: ${fixed:,.0f}\n"
            f"Target range (3–6 mo): ${low:,.0f} – ${high:,.0f}\n"
            "```\n\n"
            "Tip: Automate a small transfer each payday into a separate high-yield savings account."
        )

    if topic == "BUDGET":
        needs = income * 0.50
        wants = income * 0.30
        savings = income * 0.20
        return (
            "A simple starting point is the **50/30/20 guideline**: "
            "~50% for needs, 30% for wants, and 20% for savings or debt repayment. "
            "Adjust as needed based on your goals and obligations.\n\n"
            "**Formula**\n"
            "```\n"
            "needs   = income × 0.50\n"
            "wants   = income × 0.30\n"
            "savings = income × 0.20\n"
            "```\n\n"
            "**Your Numbers**\n"
            f"- Income: ${income:,.0f} / month\n"
            f"- Needs (50%): ${needs:,.0f}\n"
            f"- Wants (30%): ${wants:,.0f}\n"
            f"- Savings/Debt (20%): ${savings:,.0f}\n\n"
            "Tip: Track spending for a month to see if your actual ratios line up. "
            "Even small adjustments can free up money for savings or debt payoff."
        )

    if topic == "DEBT":
        return (
            "There are two popular strategies for debt repayment: **Avalanche** and **Snowball**. "
            "Both require making minimum payments on all debts, then using extra money toward one focus debt.\n\n"

            "**Method 1: Avalanche (Mathematically Optimal)**\n"
            "- List debts by **highest APR first**.\n"
            "- Pay minimums on all debts.\n"
            "- Direct extra payments to the **highest APR debt**.\n"
            "- Saves the most money on interest.\n\n"

            "**Method 2: Snowball (Motivation Boost)**\n"
            "- List debts by **smallest balance first**.\n"
            "- Pay minimums on all debts.\n"
            "- Direct extra payments to the **smallest balance debt**.\n"
            "- Builds momentum through quick wins.\n\n"

            "**Comparison**\n"
            "- Avalanche: lowest total interest, may take longer to feel progress.\n"
            "- Snowball: faster motivation, but costs more in interest.\n\n"

            "Tip: If your priority is saving money, choose Avalanche. "
            "If you need motivation from early wins, choose Snowball. "
            "The best method is the one you can stick with consistently."
        )

    # GENERAL / GOAL fallback
    return (
        "I focus on personal finance education: budgeting, savings goals, debt, and definitions like APR/APY. "
        "Ask about any of these and I’ll include formulas and step-by-step examples.\n\n"
        "Tip: Start with an emergency fund and avoid carrying high-interest balances."
    )

# ---------- Minimal KB bootstrap (auto-creates /kb on first run) ----------
HERE = os.path.dirname(__file__)
KB_DIR = os.path.join(HERE, "kb")
os.makedirs(KB_DIR, exist_ok=True)

DEFAULT_KB = {
"budgeting_503020.md": """# 50/30/20 Budget Rule
Allocate ~50/%/ of after-tax income to needs, 30% to wants, 20% to savings/debt.
Adjust based on fixed obligations and goals. Example: $3,000 income → $1,500 needs, $900 wants, $600 savings.
""",
"emergency_fund.md": """# Emergency Fund Basics
Target 3–6 months of essential expenses (rent, utilities, groceries, transport).
Formula: Emergency fund = monthly fixed expenses × months (3–6).
Start with a 1-month micro-goal if cash is tight; automate transfers.
""",
"apr_vs_apy.md": """# APR vs APY
APR is the annual rate without compounding; APY includes compounding effects.
Credit cards quote APR; total cost depends on balance, compounding, and fees.
""",
"compound_interest.md": """# Compound Interest
Future value with monthly compounding: FV = P × (1 + r/12)^(12t).
Small differences in rate and time can meaningfully change outcomes.
""",
"debt_snowball.md": """# Debt Snowball
Pay smallest balance first (while making minimums on others) to build momentum.
Motivating, but may cost more interest than avalanche.
""",
"debt_avalanche.md": """# Debt Avalanche
Focus extra payments on highest APR first to minimize total interest paid.
Often faster in interest terms than snowball when budgets are similar.
""",
"diversification_basics.md": """# Diversification Basics (Educational)
Diversification spreads risk across assets so one holding’s loss is less likely to sink the portfolio.
This app does not give security/product recommendations.
""",
"roth_vs_traditional.md": """# Roth vs Traditional (Educational)
Roth: after-tax contributions, qualified withdrawals tax-free. Traditional: pre-tax contributions, taxed at withdrawal.
Choice depends on current vs expected future tax rates and eligibility.
"""
}

def _ensure_kb():
    for fname, content in DEFAULT_KB.items():
        path = os.path.join(KB_DIR, fname)
        if not os.path.exists(path):
            with open(path, "w", encoding="utf-8") as f:
                f.write(content)

_ensure_kb()

# ---------- Load mini FAQ (glossary) ----------
FAQ_PATH = os.path.join(HERE, "faq_data.json")
if not os.path.exists(FAQ_PATH):
    with open(FAQ_PATH, "w") as f:
        json.dump({
            "apr": "APR (Annual Percentage Rate) is the yearly cost of borrowing without compounding. APY includes compounding.",
            "emergency fund": "An emergency fund typically covers 3–6 months of essential expenses.",
            "50/30/20": "A simple budgeting guideline: ~50% needs, 30% wants, 20% Savings/debt."
        }, f, indent=2)
with open(FAQ_PATH, "r") as f:
    FAQ_DATA: Dict[str, str] = json.load(f)

# ---------- Simple RAG using TF-IDF over /kb ----------
@st.cache_resource(show_spinner=False)
def build_tfidf_index(kb_dir: str):
    from sklearn.feature_extraction.text import TfidfVectorizer
    paths = sorted(glob.glob(os.path.join(kb_dir, "*.md")))
    docs, titles = [], []
    for p in paths:
        with open(p, "r", encoding="utf-8") as f:
            text = f.read()
        title = (text.splitlines()[0].lstrip("# ").strip() or os.path.basename(p))
        docs.append(text)
        titles.append(title)
    vect = TfidfVectorizer(stop_words="english", ngram_range=(1,2), min_df=1)
    X = vect.fit_transform(docs)
    fp = hashlib.md5(("".join(docs)).encode("utf-8")).hexdigest()
    return {"vect": vect, "X": X, "docs": docs, "titles": titles, "paths": paths, "fingerprint": fp}

rag = build_tfidf_index(KB_DIR)

def rag_retrieve(query: str, k: int = 4) -> List[Tuple[str,str,str]]:
    """Returns list of (title, snippet, path)"""
    if not query.strip():
        return []
    vect, X = rag["vect"], rag["X"]
    q = vect.transform([query])
    import numpy as np
    sims = (q @ X.T).toarray()[0]
    idxs = np.argsort(-sims)[:k]
    results = []
    for i in idxs:
        doc = rag["docs"][i]
        title = rag["titles"][i]
        path = rag["paths"][i]
        lines = [ln.strip() for ln in doc.splitlines() if ln.strip()]
        snippet = " ".join(lines[:3])[:500]
        results.append((title, snippet, path))
    return results

# ---------- Domain guardrails ----------
DISCLAIMER = ("Educational demo only — not financial, tax, or legal advice. "
              "No investment/security recommendations are provided.")
OUT_OF_SCOPE_PATTERNS = [
    r"\b(what|which)\s+(stock|etf|fund|bond)\s+should\s+i\s+(buy|sell)\b",
    r"\ballocate\s+\d+%",
    r"\bwhich\s+credit\s+card\s+is\s+best\b",
    r"\btax\s+loophole\b",
    r"\b(weather|news|movie|restaurant|politics)\b",
]
def out_of_scope(user_text: str) -> bool:
    t = user_text.lower()
    return any(re.search(p, t) for p in OUT_OF_SCOPE_PATTERNS)

# ---------- Calculators ----------
def budget_503020(income: float):
    return {"needs": income * 0.50, "wants": income * 0.30, "savings": income * 0.20}

def monthly_savings_for_goal(target: float, months: int, apy: float = 0.0) -> float:
    if months <= 0:
        return float("inf")
    r = apy / 12.0
    if r == 0:
        return target / months
    return target * r / ((1 + r) ** months - 1)  # FV annuity (ordinary)

def simulate_debt_payoff(debts, extra_payment: float, method: str = "avalanche"):
    import copy
    ds = copy.deepcopy(debts)
    months = 0
    total_interest = 0.0
    def sort_key(d):
        return (-d["apr"], d["balance"]) if method == "avalanche" else (d["balance"], -d["apr"])
    if sum(d["min_payment"] for d in ds) + extra_payment <= 0:
        return 0, 0.0
    while True:
        if all(d["balance"] <= 0.005 for d in ds):
            break
        months += 1
        ds.sort(key=sort_key)
        for d in ds:
            if d["balance"] <= 0:
                continue
            monthly_rate = d["apr"] / 12.0
            interest = d["balance"] * monthly_rate
            total_interest += interest
            d["balance"] += interest
        payment_pool = sum(d["min_payment"] for d in ds) + extra_payment
        for d in ds:
            if d["balance"] <= 0:
                continue
            pay = min(d["min_payment"], d["balance"])
            d["balance"] -= pay
            payment_pool -= pay
        for d in ds:
            if d["balance"] > 0 and payment_pool > 0:
                pay = min(payment_pool, d["balance"])
                d["balance"] -= pay
                payment_pool -= pay
                break
        if months > 600:
            break
    return months, total_interest

# ---------- UI ----------
st.set_page_config(page_title="Personalized Financial Advisor", page_icon="💬", layout="centered")
st.title("💬 Personalized Financial Advisor")
st.caption(DISCLAIMER)

with st.sidebar:
    st.header("Your Profile")
    role = st.selectbox("Role", ["Student", "Early Career", "Parent"], index=1)
    income = st.number_input("Monthly take-home income ($)", min_value=0.0, value=4000.0, step=100.0)
    fixed = st.number_input("Monthly fixed expenses ($)", min_value=0.0, value=2200.0, step=50.0)
    wants_guess = st.number_input("Monthly variable/wants ($)", min_value=0.0, value=800.0, step=50.0)
    st.markdown("---")
    llm = get_llm()
    LLM_READY = llm is not None
    st.write("**LLM status:** " + ("✅ flan-t5-base ready" if LLM_READY else "⚠️ Fallback mode (no model found)"))

tab1, tab2, tab3, tab4 = st.tabs(["Ask the Advisor", "Budget", "Debt", "Savings Goal"])

# ---------- FLAN-friendly prompt builder with RAG context ----------
def build_prompt(user_q: str, role: str, income: float, fixed: float, ctx_items: List[Tuple[str,str,str]]):
    ctx_lines = [f"[{title}] {snippet}" for (title, snippet, _path) in ctx_items] or ["(no retrieved notes)"]
    ctx_block = "\n".join(ctx_lines)
    tone = {
        "Student": "As a student, focus on building strong habits with small, consistent steps.",
        "Early Career": "In early career, prioritize stability and automate savings where possible.",
        "Parent": "As a parent, emphasize resilience: emergency fund and manageable debt first."
    }[role]
    instruction = (
        "You are a financial education assistant. Only answer budgeting, debt, savings, and basic definitions "
        "(APR, APY, compound interest). Do not give product picks or tax/legal strategies. "
        "Use the context and the user's numbers when helpful. Write 4–6 sentences in plain paragraphs. "
        "Do NOT start with a title or heading. If math is involved, show the formula and computed numbers. "
        "End with one line starting with 'Tip:'."
    )
    prompt = f"""Instruction:
{instruction}

User profile:
- Role: {role}
- Income: ${income:.2f} / mo
- Fixed expenses: ${fixed:.2f} / mo
Tone hint: {tone}

Context (retrieved notes):
{ctx_block}

Question:
{user_q}

Answer:
"""
    return prompt

# ---------- Tab 1: Chat (domain-specific + RAG + robust fallbacks) ----------
with tab1:
    st.subheader("Ask the Advisor")
    st.session_state.setdefault("chat_answer", "")
    st.session_state.setdefault("chat_prompt", "")
    st.session_state.setdefault("chat_clicked", False)
    st.session_state.setdefault("chat_citations", [])

    user_q = st.text_input(
        "Type a finance question (e.g., 'How big should my emergency fund be?' or 'APR vs APY?')",
        key="chat_q",
        value=st.session_state.get("chat_q", "")
    )

    cols = st.columns([1,1,1])
    with cols[0]:
        clicked = st.button("Get Answer", type="primary")
    with cols[1]:
        rebuild = st.button("Rebuild KB index")
    with cols[2]:
        show_raw = st.checkbox("Show raw answer (debug)", value=False)

    if rebuild:
        _ensure_kb()
        build_tfidf_index.clear()  # type: ignore
        rag = build_tfidf_index(KB_DIR)
        st.success("KB index rebuilt.")

    if clicked:
        st.session_state.chat_clicked = True
        if not user_q.strip():
            st.info("Please enter a finance question.")
            st.session_state.chat_answer = ""
            st.session_state.chat_prompt = ""
            st.session_state.chat_citations = []
        elif out_of_scope(user_q):
            st.warning("I’m focused on personal finance topics (budgeting, savings, debt, definitions). "
                       "Please rephrase your question within that scope.")
            st.session_state.chat_answer = ""
            st.session_state.chat_prompt = "(out-of-scope)"
            st.session_state.chat_citations = []
        else:
            ctx_items = rag_retrieve(user_q, k=4)
            prompt = build_prompt(user_q, role, income, fixed, ctx_items)
            with st.spinner("Generating answer..."):
                try:
                    topic = infer_topic(user_q)
                    text = ""
                    if LLM_READY:
                        out = llm(prompt, max_new_tokens=220)
                        text = sanitize_output((out[0].get("generated_text", "") or ""))

                    bad_echo = "answer:" in (text.lower()) or len(text) < 80
                    if not text or bad_echo:
                        text = fallback_answer(topic, role, income, fixed, ctx_items, user_q)

                    st.session_state.chat_answer = text.strip()
                    st.session_state.chat_prompt = prompt
                    st.session_state.chat_citations = [(t, p) for (t, _s, p) in ctx_items]

                except Exception as e:
                    st.session_state.chat_answer = ""
                    st.session_state.chat_prompt = prompt
                    st.session_state.chat_citations = []
                    st.error("Generation failed.")
                    if show_raw:
                        st.exception(e)

    if st.session_state.chat_clicked:
        ans = (st.session_state.chat_answer or "").strip()
        if ans:
            st.markdown("### Answer")
            st.markdown(ans)

            if st.session_state.chat_citations:
                st.markdown("**Sources (local KB):**")
                for (title, path) in st.session_state.chat_citations:
                    rel = os.path.relpath(path, HERE)
                    st.markdown(f"- {title}  \n  <small><code>{rel}</code></small>", unsafe_allow_html=True)

            if show_raw:
                with st.expander("Raw answer (debug)"):
                    st.text_area("Text", ans, height=180)
        else:
            st.info("No answer generated.")

    with st.expander("Show Prompt (for demo transparency)"):
        st.code((st.session_state.chat_prompt or "(no prompt yet)").strip())

# ---------- Tab 2: Budget ----------
with tab2:
    st.subheader("Budget Helper (50/30/20 baseline)")
    plan = budget_503020(income)

    # Pretty strings
    needs_s  = f"${plan['needs']:.0f}"
    wants_s  = f"${plan['wants']:.0f}"
    save_s   = f"${plan['savings']:.0f}"

    # Nice, readable metrics
    c1, c2, c3 = st.columns(3)
    c1.metric("Needs (50%)", needs_s)
    c2.metric("Wants (30%)", wants_s)
    c3.metric("Savings/Debt (20%)", save_s)

    free_cash = income - fixed - wants_guess
    st.write(f"**Your current free cash estimate:** ${free_cash:.0f} / mo")

    if free_cash < plan["savings"]:
        st.info("You're below the 20% Savings guideline. Consider trimming wants or raising income to hit goals.")
    else:
        st.success("You meet or exceed the 20% Savings guideline. Consider directing surplus to emergency fund or debt.")

# ---------- Tab 3: Debt ----------
with tab3:
    st.subheader("Debt Payoff Simulator")
    st.caption("Enter up to two debts (for demo). APR as decimal (e.g., 0.24 for 24%).")

    c1, c2 = st.columns(2)
    with c1:
        b1 = st.number_input("Debt 1 balance ($)", min_value=0.0, value=2000.0, step=50.0)
        a1 = st.number_input("Debt 1 APR", min_value=0.0, value=0.24, step=0.01, format="%.2f")
        m1 = st.number_input("Debt 1 min payment ($)", min_value=0.0, value=50.0, step=5.0)
    with c2:
        b2 = st.number_input("Debt 2 balance ($)", min_value=0.0, value=1000.0, step=50.0)
        a2 = st.number_input("Debt 2 APR", min_value=0.0, value=0.18, step=0.01, format="%.2f")
        m2 = st.number_input("Debt 2 min payment ($)", min_value=0.0, value=35.0, step=5.0)

    extra = st.number_input("Extra budget for debt beyond minimums ($/mo)", min_value=0.0, value=100.0, step=10.0)

    if st.button("Simulate"):
        debts = []
        if b1 > 0: debts.append({"balance": b1, "apr": a1, "min_payment": m1})
        if b2 > 0: debts.append({"balance": b2, "apr": a2, "min_payment": m2})
        if not debts:
            st.info("Enter at least one debt.")
        else:
            m_av, int_av = simulate_debt_payoff(debts, extra, method="avalanche")
            m_sn, int_sn = simulate_debt_payoff(debts, extra, method="snowball")
            st.write(f"**Avalanche:** ~{m_av} months, interest ≈ ${int_av:,.0f}")
            st.write(f"**Snowball:** ~{m_sn} months, interest ≈ ${int_sn:,.0f}")
            if int_av < int_sn:
                st.success("Avalanche typically minimizes total interest paid.")
            else:
                st.info("Snowball can feel more motivating despite potentially higher interest.")

# ---------- Tab 4: Savings ----------
with tab4:
    st.subheader("Savings Goal Planner")
    target = st.number_input("Target amount ($)", min_value=0.0, value=3000.0, step=100.0)
    months = st.number_input("Months to reach goal", min_value=1, value=6, step=1)
    apy = st.number_input("Savings APY (e.g., 0.04 for 4%)", min_value=0.0, value=0.0, step=0.01, format="%.2f")
    if st.button("Calculate monthly saving"):
        p = monthly_savings_for_goal(target, months, apy)
        st.write(f"**You need to save ≈ ${p:,.0f} per month**")
        if apy > 0:
            st.caption("Note: Interest helps slightly over short horizons; the monthly need is driven mostly by the target and months.")
    st.markdown("---")
    st.caption("This chatbot is for educational purposes only. No financial, tax, or legal advice provided.")
