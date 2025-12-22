# 📈 Artha: AI Financial Analyst (Self-Healing RAG)

Artha is not just a chatbot—it is an **autonomous financial agent** designed to analyze 10-K reports, compare companies, and visualize data. 

Unlike traditional RAG systems that fail when data is missing, Artha features a **"Self-Healing" architecture**. It detects missing information in documents and autonomously triggers a live web search to fill the gaps, ensuring 100% answer completion.

## 🚀 Key Features (The "Wow" Factor)

### 1. 🧠 Query Decomposition Engine
* **Problem:** Standard RAG fails at "Compare Apple and Microsoft" because it retrieves a mix of irrelevant chunks.
* **Solution:** Artha breaks complex questions into sub-queries (`"Apple Revenue 2024"`, `"Microsoft Revenue 2024"`), retrieves them separately, and synthesizes the answer.

### 2. ❤️‍🩹 Self-Healing Retrieval
* **Problem:** If a 10-K PDF doesn't have the specific number (e.g., "2025 Forecast"), most bots hallucinate or say "I don't know."
* **Solution:** Artha's **Validator Node** detects "Data Missing" signals and automatically reroutes the workflow to a **Live Web Search (Tavily API)** to fetch real-time data.

### 3. 📊 Generative UI (Dynamic Charts)
* **Problem:** Financial analysts hate walls of text.
* **Solution:** Artha intelligently detects when a comparison is being made and **autonomously writes code** to render an interactive Bar Chart inside the chat interface.

---

## 🛠️ Tech Stack
* **Framework:** LangChain, LangGraph (Stateful Multi-Actor Orchestration)
* **LLM:** Llama-3.3-70b (via Groq)
* **Vector Database:** ChromaDB
* **Frontend:** Streamlit
* **Search:** Tavily AI (for self-healing fallback)

---

## ⚡ How to Run Locally

1. **Clone the repository**
   ```bash
   git clone [https://github.com/YOUR_USERNAME/artha-financial-ai.git](https://github.com/YOUR_USERNAME/artha-financial-ai.git)
   cd artha-financial-ai

2. **Install Dependencies**
```bash 
pip install -r requirements.txt
```

3. **Set Up API Keys**
```bash 
Create a .env file in the root directory:
GROQ_API_KEY=gsk_...
TAVILY_API_KEY=tvly-...   
```

4. **Run the Agent**
```bash
streamlit run app.py
```

🧩 **Architecture Flow**
User Query → Decomposer (Breaks into sub-questions).

Retriever → Fetches chunks from PDF.

Grader → Checks if data is sufficient.

Generator → Drafts an answer.

Hallucination Check → If data is missing → Trigger Web Search → Regenerate Answer.

UI Renderer → Displays Text + Dynamic Charts.
