# 💬 Personalized Financial Advisor Chatbot Using Generative AI

### 📊 Executive Summary
This project presents a **Retrieval-Augmented Generation (RAG)–based financial education chatbot** built to simplify personal finance learning for students, young professionals, and families.  
The chatbot combines **retrieval, prompting, and fallback mechanisms** to deliver safe, coherent, and personalized financial guidance.  
Developed as part of an **Individual Study under Dr. Kenneth Murphy (UCI MSBA 2025)**, the system demonstrates how **Generative AI + Retrieval** can be applied to financial literacy — empowering users with approachable, explainable, and role-specific advice.  

🔗 **Live App:** [Personalized Financial Advisor Chatbot](https://personalized-financial-advisor-chatbot-using-generative-ai-yhb.streamlit.app/)

---

### 💼 Motivation
Financial concepts such as **budgeting, APR vs APY, debt management, and savings goals** are often overwhelming or inaccessible to many individuals.  
As a student facing similar challenges, I wanted to build an AI-powered assistant that could:
- Explain financial terms in simple, contextualized language  
- Personalize recommendations for different user types (student, early career, parent)  
- Maintain factual accuracy and reliability using retrieval-based grounding

---

### 🎯 Project Objectives
- ✅ Build a working prototype of a **financial education chatbot**  
- ✅ Integrate **RAG (Retrieval-Augmented Generation)** for grounded responses  
- ✅ Implement **guardrails** and **fallback templates** for factual safety  
- ✅ Personalize responses based on **user profiles**  
- ✅ Evaluate answers for **coherence, factuality, and personalization**  
- ✅ Strengthen applied AI engineering skills beyond traditional analytics  

---

### ⚙️ System Design Overview
**System Archetype:** Retrieval-Augmented Assistant with Guardrails  

**Workflow:**  
User Query → TF-IDF Retriever → Context + User Profile Prompt → LLM (Flan-T5) → Response → Fallback Template (if needed)

**Core Components:**
| Module | Description |
|--------|-------------|
| **Knowledge Base** | Curated markdown notes on financial concepts (APR vs APY, 50/30/20 rule, emergency fund formulas, etc.) |
| **Retriever** | TF-IDF index (scikit-learn) for simple, transparent grounding |
| **Generator** | `flan-t5-base` (Hugging Face) for text generation |
| **Guardrails** | Out-of-scope filtering (no investment or tax advice) |
| **Fallback Templates** | Structured explanations with formulas (e.g., *Emergency Fund = Fixed Expenses × 3–6 months*) |
| **Frontend** | Streamlit app with multi-tab interface |

---

### 🧠 Skills Highlighted
- **Languages & Frameworks:** Python, Streamlit  
- **AI/ML Tools:** Hugging Face Transformers, scikit-learn, RAG Pipeline  
- **Analytics & Logic:** NumPy, Pandas, Markdown Templating  
- **Concepts:** Retrieval Augmented Generation, Prompt Engineering, Guardrails for LLMs  
- **Applied Domains:** Financial Education, AI Prototyping, Explainable AI  

---

### 🧩 Key Features
| Feature | Description |
|----------|-------------|
| **Ask the Advisor** | Free-text Q&A using RAG + fallback templates |
| **Budget Helper** | Calculates personalized 50/30/20 budget allocations |
| **Debt Simulator** | Compares **snowball vs avalanche** debt payoff methods |
| **Savings Goal Planner** | Determines monthly savings to reach a target |
| **Explainability Panel** | Displays retrieved notes and prompts for transparency |

---

### 🧪 Evaluation Framework
**Metrics:** Coherence | Accuracy | Relevance | Personalization  

**Testing Methodology**
- Created **benchmark question set** (20–30 core finance questions)  
- Tested across **Student**, **Early Career**, and **Parent** profiles to verify tone adaptation  
- Conducted **hallucination checks** vs authoritative references (Investopedia)  
- Assessed guardrails with **out-of-scope queries** (e.g., “Which stock should I buy?”)  
- Peer-based **usability testing** for trust and clarity  

---

### 📈 Results & Insights
- Successfully built a **RAG-based financial education chatbot** prototype.  
- Responses remained **factually consistent** across 90% of benchmark questions.  
- Guardrails effectively blocked investment / tax queries.  
- Personalized tones:  
  - **Student:** Simpler language + examples  
  - **Parent:** Long-term planning focus  
  - **Early Career:** Saving & debt trade-offs  
- System delivered **fast, reliable, explainable outputs** while staying lightweight and transparent.  

---

### 🚀 Future Enhancements
- 🔄 Replace **TF-IDF** with **FAISS or Qdrant** for semantic retrieval  
- 🧮 Integrate **financial calculators + visual charts** (budget breakdowns, debt progress)  
- 🧠 Upgrade to larger LLMs (Flan-T5 Large / Llama 3 / Claude) for richer explanations  
- 🧱 Add **evaluation pipeline** to score coherence and factuality automatically  
- 🌐 Expand knowledge base to include **tax planning, insurance, and investment basics**

---

### 💡 Personal Learning Reflection
This project bridged my background in **business analytics** with the emerging world of **applied AI systems**.  
I learned how to design retrieval-grounded chatbots, prompt effectively, integrate fallback logic, and deploy a user-facing MVP.  
More importantly, it gave me confidence to build **AI solutions that are both technically sound and socially useful** — translating analytics knowledge into accessible tools that improve everyday decision-making.

---

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

## Files
- `app.py` — Streamlit app
- `faq_data.json` — tiny knowledge base
- `requirements.txt` — dependencies

## Disclaimer
**Educational purposes only. Not financial, tax, or legal advice.**
