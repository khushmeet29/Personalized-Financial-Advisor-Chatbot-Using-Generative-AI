
# Personalized Financial Advisor

A **one-file Streamlit app** that demonstrates a personalized financial advisor chatbot using an **open-source LLM** (google/flan-t5-base), a tiny **keyword-based context retriever**, and **three calculators** (budget, debt, savings).

## Quickstart

```bash
# 1) Create a virtual env (recommended)
python -m venv .venv && source .venv/bin/activate  # (Windows: .venv\Scripts\activate)

# 2) Install dependencies
pip install -r requirements.txt

# 3) Run
streamlit run app.py
```

> Note: The first run will download the flan-t5-base model (~1GB). If the model cannot be loaded, the app falls back to a rule-based response for the chat tab.

## Features
- **Chat tab**: Context-aware answers with flan-t5-base + mini KB snippets; refusal policy for out-of-scope asks.
- **Budget**: 50/30/20 baseline with quick feedback.
- **Debt**: Snowball vs avalanche simulator (months + total interest).
- **Savings**: Monthly saving needed to reach a goal (with optional APY).
- **Personalization**: Role-based tone (Student / Early Career / Parent).
- **Safety**: Clear disclaimer; blocks product picks and tax/legal advice.

## Files
- `app.py` — Streamlit app
- `faq_data.json` — tiny knowledge base
- `requirements.txt` — dependencies

## Disclaimer
**Educational purposes only. Not financial, tax, or legal advice.**
