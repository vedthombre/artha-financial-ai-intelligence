# Artha AI | Autonomous Financial Intelligence Agent 📈

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![AI](https://img.shields.io/badge/GenAI-Llama3.3-orange)
![Framework](https://img.shields.io/badge/Framework-LangGraph-green)

**Artha AI** is a **Self-Correction Retrieval Augmented Generation (RAG)** system designed for high-stakes financial analysis. Unlike standard RAG chatbots, Artha features a **Hybrid Search Architecture** that autonomously detects missing data in static documents (10-K Reports) and triggers live web agents to fetch real-time market data (Stock Prices, News) to fill the gaps.

---

## 🧠 Key Features (The "Why It Matters")

### 1. Hybrid Intelligence (PDF + Web) 🌐
Most RAG bots fail when asked: *"What was the Revenue in 2023 and the Stock Price today?"*
* **Artha's Solution:** It retrieves the Revenue from the internal Vector DB (Chroma) and autonomously triggers a Tavily Web Search agent to fetch the live stock price, merging both into a single cohesive answer.

### 2. Self-Healing & Hallucination Check 🛡️
* **The Problem:** AI often hallucinates when it doesn't know the answer.
* **Artha's Solution:** A dedicated **Validator Node** checks the final answer. If the AI says "I don't know" or "Context missing," the graph autonomously re-routes the query to the Web Search node to find the answer, rather than giving up.

### 3. Structured Data Validation (Pydantic) ✅
* Utilizes **Strict Schema Validation** for grading document relevance.
* Optimized context windowing to prevent "Lost in the Middle" phenomenon during grading.

---

## 🛠️ Tech Stack

* **Orchestration:** LangGraph (Stateful Multi-Actor Applications)
* **LLM:** Llama-3.3-70B-Versatile (via Groq)
* **Vector DB:** ChromaDB (Local Persistence)
* **Search Tool:** Tavily Search API
* **Frontend:** Streamlit (Custom CSS & White Theme)
* **Embeddings:** HuggingFace (`all-MiniLM-L6-v2`)

---

## 🚀 How to Run Locally

### 1. Clone the Repository
```bash
git clone [https://github.com/YOUR_USERNAME/artha-financial-ai-intelligence.git](https://github.com/YOUR_USERNAME/artha-financial-ai-intelligence.git)
cd artha-financial-ai-intelligence 
```

2. Install Dependencies
```bash 
pip install -r requirements.txt 
```

3. Set Up API Keys
```bash 
Create a .env file in the root directory:
GROQ_API_KEY=gsk_...
TAVILY_API_KEY=tvly-...   
```

4. Run the Agent
```bash
streamlit run app.py
```

🧩 Architecture Flow
Retrieve: Fetch chunks from PDF (10-K).

Grade: AI judges if chunks are relevant.

Decide: * If data is missing → Trigger Web Search.

If data is present → Generate Answer.

Validate: Check final answer for hallucinations.

Heal: If validation fails, retry with broader search parameters.