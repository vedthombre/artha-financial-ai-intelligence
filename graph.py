from dotenv import load_dotenv

# --- IMPORTS ---
from langchain_groq import ChatGroq
from langchain_tavily import TavilySearch
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.documents import Document
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import END, StateGraph
from typing import List, TypedDict



load_dotenv()

# --- 1. STATE DEFINITION ---
class GraphState(TypedDict):
    question: str
    sub_queries: List[str]
    generation: str
    web_search: str
    web_search_results: str  # <--- NEW: Stores web data separately from internal docs
    documents: List[Document]
    file_filter: str
    loop_count: int
    document_only: bool

# --- 2. SETUP ---
print("--- INITIALIZING AGENT (Phase 1: Decomposition) ---")
llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0)
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2", model_kwargs={"device": "cpu"})

vectorstore = Chroma(
    persist_directory="./chroma_db", 
    embedding_function=embedding_model,
    collection_name="rag-chroma"
)

web_search_tool = TavilySearch(max_results=3)

# --- 3. NODES ---

def decompose_query(state):
    """
    Breaks down complex questions into sub-queries.
    """
    print("---DECOMPOSE QUERY---")
    question = state["question"]
    if not isinstance(question, str) or not question.strip():
        raise ValueError("question must be a non-empty string")
    
    parser = JsonOutputParser()
    prompt = ChatPromptTemplate.from_template(
        """You are a Query Decomposition Engine.
        Analyze the user's question to break it down into atomic search steps.
        
        ### RULES:
        1. **Complex Comparisons:** If asking "Apple vs Microsoft", generate 2 separate queries.
        2. **Ratios & Math:** If asking for a metric that requires calculation (e.g., "R&D Intensity"), you MUST generate queries for BOTH the Numerator and Denominator.
           * Example: "Calculate R&D Intensity" -> ["Microsoft R&D Expenses 2023", "Microsoft Total Revenue 2023"]
        3. **Single Entity:** If simple, return just the original question.

        ### CRITICAL FINANCIAL RULE (THE FIX):
        4. **Ambiguous Dates:** If a user mentions a year (e.g., "NVIDIA 2024") in a financial context:
           - You MUST append "Fiscal Year" and "10-K Annual Report" to the query.
           - This ensures we get the audited numbers, not random news estimates.
           - Example: "NVIDIA 2024 Revenue" -> "NVIDIA Fiscal Year 2024 Total Revenue 10-K Annual Report"
           
        Return JSON format: {{ "queries": ["query 1", "query 2", "query 3"] }}
        
        Question: {question}
        """
    )
    
    chain = prompt | llm | parser
    
    try:
        result = chain.invoke({"question": question})
        sub_queries = result.get("queries", [question])
        print(f"   -> Generated Sub-Queries: {sub_queries}")
    except (ValueError, TypeError, KeyError) as exc:
        print(f"   -> Query decomposition failed; using original question: {exc}")
        sub_queries = [question]
        
    return {"sub_queries": sub_queries, "question": question}

def retrieve(state):
    """
    Retrieves documents for EACH sub-query.
    """
    print("---RETRIEVE (MULTI-STEP)---")
    sub_queries = state["sub_queries"]
    file_filter = state.get("file_filter", "All Documents")
    loop_count = state.get("loop_count", 0)
    
    all_documents = []
    # A broad multi-metric query can rank intervening discussion chunks ahead of
    # the financial statement. Fetch enough candidates for deduplication to
    # preserve the statement even when a source has been ingested repeatedly.
    k_per_query = 10
    
    for query in sub_queries:
        print(f"   -> Searching for: '{query}'")
        if file_filter and file_filter != "All Documents":
            docs = vectorstore.similarity_search(query, k=k_per_query, filter={"source": file_filter})
        else:
            docs = vectorstore.similarity_search(query, k=k_per_query)
        all_documents.extend(docs)
        
    # Deduplicate
    unique_docs = []
    seen_content = set()
    for doc in all_documents:
        if doc.page_content not in seen_content:
            unique_docs.append(doc)
            seen_content.add(doc.page_content)
            
    print(f"   -> Total Unique Docs Retrieved: {len(unique_docs)}")
    return {"documents": unique_docs, "loop_count": loop_count}

def grade_documents(state):
    print("---CHECK RELEVANCE---")
    question = state["question"]
    documents = state["documents"]
    document_only = is_document_only_request(question)
    
    # If we have NO docs, definitely search web
    if not documents:
        print(
            "   -> No docs found. "
            + ("Web search disabled by request." if document_only else "Web search needed.")
        )
        return {"documents": [], "web_search": "No" if document_only else "Yes", "document_only": document_only}

    # Quick check for live data requests
    web_search = "No"
    if "live" in question.lower() or "stock price" in question.lower() or "today" in question.lower():
         web_search = "Yes"
             
    if document_only:
        web_search = "No"
    return {
        "documents": documents,
        "question": question,
        "web_search": web_search,
        "document_only": document_only,
    }


def is_document_only_request(question: str) -> bool:
    """Return whether the user explicitly forbids web search."""
    normalized = " ".join(question.lower().split())
    return any(
        phrase in normalized
        for phrase in (
            "use only the uploaded document",
            "use only the provided document",
            "use only uploaded documents",
            "do not use web search",
            "don't use web search",
            "without web search",
        )
    )



def transform_query(state):
    print("---TRANSFORM QUERY---")
    question = state["question"]
    if not isinstance(question, str) or not question.strip():
        raise ValueError("question must be a non-empty string")
    
    # 1. STRICT SYSTEM PROMPT
    system_prompt = """You are a Search Query Optimizer.
    TASK: Convert the user's question into a single, specific web search query.
    RULES:
    1. Output ONLY the query string. 
    2. Do NOT add explanations, bullet points, or quotes.
    3. Do NOT say "Here is the query".
    """
    
    # 2. CALL THE LLM (Using the 'llm' variable you already created)
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=question)
    ]
    
    # Use LangChain invoke instead of raw client
    response = llm.invoke(messages)
    if not isinstance(response.content, str) or not response.content.strip():
        raise ValueError("query optimizer returned an empty response")
    better_question = response.content.strip()

    # 3. THE SAFETY NET
    triggers = ["revenue", "income", "profit", "margin", "research", "r&d", "expense"]
    
    if any(keyword in question.lower() for keyword in triggers):
        better_question = f"{better_question} 10-K Annual Report"
        
    return {"question": better_question} 

def web_search_node(state):
    print("---WEB SEARCH---")
    question = state["question"]
    loop_count = state.get("loop_count", 0)
    
    try:
        docs = web_search_tool.invoke({"query": question})
        web_content = ""
        if isinstance(docs, list):
            for d in docs:
                content = d.get('content', '') if isinstance(d, dict) else str(d)
                web_content += f"\n[Source: Live Web] {content}"
        else:
            web_content = str(docs)
    except Exception as e:
        web_content = f"Web search failed: {e}"
    
    # FIX: Don't append to 'documents'. Save to 'web_search_results' to avoid context overflow.
    return {"web_search_results": web_content, "web_search": "Done", "loop_count": loop_count + 1}

def generate(state):
    """
    Final Generation Node with 'Context Rescue' Logic.
    Fixes the issue where 11 internal docs drown out the web search results.
    """
    print("---GENERATE---")
    
    question = state["question"]
    internal_docs = state.get("documents", [])
    web_results = state.get("web_search_results", "") # Get the separate web data
    loop_count = state.get("loop_count", 0)
    
    # 1. PRUNE INTERNAL DOCS
    # If we have too many docs (like 11), the LLM gets confused. Keep only the top 6.
    if len(internal_docs) > 6:
        # print(f"DEBUG: Pruning internal docs from {len(internal_docs)} to 6 to save context.")
        pruned_docs = internal_docs[:6]
    else:
        pruned_docs = internal_docs

    # Format internal docs
    internal_context = "\n\n".join([f"Source: {doc.metadata.get('source', 'Internal')}\nContent: {doc.page_content}" for doc in pruned_docs])

    # 2. CONSTRUCT THE CONTEXT
    if web_results:
        print("   -> Merging Web Results into Context (Prioritized).")
        final_context = f"""
        WARNING: PARTIAL DATA DETECTED IN INTERNAL DATABASE.
        
        === SOURCE A: INTERNAL DOCUMENTS (May be incomplete) ===
        {internal_context}
        
        === SOURCE B: LIVE WEB SEARCH RESULTS (Use this to fill gaps) ===
        {web_results}
        
        INSTRUCTION: You must prioritize 'SOURCE B' for any data missing from 'SOURCE A'. 
        If 'SOURCE B' has the answer, ignore 'SOURCE A' failures.
        """
    else:
        final_context = f"=== INTERNAL DOCUMENTS ===\n{internal_context}"

    # 3. GENERATE
    system_prompt = """You are a Senior Financial Analyst "Arth".
    
    ### INSTRUCTIONS:
    1. **Deep Comparison:** If comparing, explicitly contrast numbers (e.g., "Apple: $X vs Microsoft: $Y").
    2. **Use the Context:** Answer strictly based on the provided text.
    3. **Prioritize Web Data:** If Internal Docs are missing data, rely on the Web Search Results.
    4. **NO DOLLAR SIGNS:** Do NOT use the "$" symbol. Always use "USD" or "dollars".
    5. **Missing Data:** If a number is missing, check if [Live Web Search] data is available. If not, state "Data missing."
    6. **Formatting:** Use proper spacing. Do NOT mash words together. Use Markdown tables.
    7. **NO CHARTS FOR TEXT:** Do NOT generate a JSON block for qualitative lists (e.g., Risks, Strengths, Factors). ONLY generate charts for direct numerical comparisons (Revenue, Profit, Stock Price).
    8.CRITICAL DATA PARSING RULE:
      - Financial tables in 10-K filings often list numbers "in millions" or "in thousands".
      - If you see a number like "27,195" for a major company like Microsoft, it implies MILLIONS.
      - YOU MUST CONVERT IT: "27,195" -> "$27.195 Billion".
      - NEVER output raw table numbers without converting to Billions (B) or Millions (M).
    
    ### INSTRUCTIONS:
    You are answering questions based on the supplied internal documents and, when allowed,
    optional live web search results.
    
    ### RESPONSE STRUCTURE:
    - Clearly state the relevant findings from the supplied context.
    - Distinguish internal document data from live web data when both are present.
    - If a web search was used, briefly state why and summarize its findings.
    - For comparisons, show the entities side by side and normalize their units.
    
    ### CRITICAL RULES:
    - Do NOT use "Source A" or "Source B".
    - Do NOT say "Assuming." Use "The web search results indicate."
    

    ### CRITICAL DATE HANDLING:
    - When a fiscal year end date is available in the context, include it next to the year.
    

    ### VISUALIZATION REQUEST:
    If the answer involves comparing numbers (e.g., Revenue A vs B) or trends (e.g., 2022-2024),
    you MUST append a JSON block at the very end.

    ### CONTEXT RESCUE LOGIC:
    If the answer seems incomplete, state which requested information is absent from the
    supplied context. Do not invent values or imply that a web search was performed.

    Format exactly like this:
    ```json
    {{
        "bar_chart": {{
            "labels": ["Entity1", "Entity2"],
            "datasets": [
                {{ "label": "Metric", "data": [100, 200] }}
            ]
        }}
    }}
    ```
    """

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"Question: {question}\n\nContext Data:\n{final_context}")
    ]
    
    response = llm.invoke(messages)
    return {"generation": response.content, "loop_count": loop_count}

def check_hallucination(state):
    print("---CHECK VALIDITY---")
    generation = state["generation"]
    loop_count = state.get("loop_count", 0)
    if state.get("document_only", False):
        return "end"
    
    gen_lower = generation.lower()
    
    failure_phrases = [
        "context does not provide",
        "not explicitly provide",
        "information is not available",                 
        "does not contain",
        "not mentioned", 
        "not provided", 
        "data missing",
        "is missing",
        "cannot directly compare",
        "unable to find"
        # --- NEW PHRASES TO CATCH SIMULATION ---
        "assuming",          # <--- Kills "Assuming the web search..."
        "example data",      # <--- Kills "Note: This is an example"
        "hypothetical", 
        "actual data may vary"
    ]
    
    # If failure phrase found AND we haven't looped too many times
    if any(p in gen_lower for p in failure_phrases):
        if loop_count < 2: # Allow up to 2 rescues
            print("   -> 🚨 Answer Incomplete. ACTIVATING WEB SEARCH RESCUE.")
            return "web_search_node"
        else:
            return "end"
            
    return "end"

def decide_to_generate(state):
    """
    Determines whether to go to Web Search or Generate based on grading.
    """
    web_search = state.get("web_search", "No")
    
    if web_search == "Yes":
        return "transform_query"
    else:
        return "generate"

# --- 4. GRAPH BUILD ---
workflow = StateGraph(GraphState)

# Add Nodes
workflow.add_node("decompose_query", decompose_query)
workflow.add_node("retrieve", retrieve)
workflow.add_node("grade_documents", grade_documents)
workflow.add_node("generate", generate)
workflow.add_node("transform_query", transform_query)
workflow.add_node("web_search_node", web_search_node)

# Set Entry Point
workflow.set_entry_point("decompose_query")

# Edges
workflow.add_edge("decompose_query", "retrieve")
workflow.add_edge("retrieve", "grade_documents")

workflow.add_conditional_edges(
    "grade_documents",
    decide_to_generate,
    {"transform_query": "transform_query", "generate": "generate"},
)

workflow.add_edge("transform_query", "web_search_node")
workflow.add_edge("web_search_node", "generate")

workflow.add_conditional_edges(
    "generate",
    check_hallucination,
    {"web_search_node": "web_search_node", "end": END}
)

app = workflow.compile()