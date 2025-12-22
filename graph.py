import os
from dotenv import load_dotenv

# --- IMPORTS ---
from langchain_groq import ChatGroq
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.documents import Document
from langgraph.graph import END, StateGraph
from typing import List, TypedDict
from pydantic import BaseModel, Field

load_dotenv()

# --- 1. STATE DEFINITION (Added sub_queries) ---
class GraphState(TypedDict):
    question: str
    sub_queries: List[str]  # <--- NEW: Holds broken-down questions
    generation: str
    web_search: str
    documents: List[Document]
    file_filter: str
    loop_count: int

# --- 2. SETUP ---
print("--- INITIALIZING AGENT (Phase 1: Decomposition) ---")
llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

vectorstore = Chroma(
    persist_directory="./chroma_db", 
    embedding_function=embedding_model,
    collection_name="rag-chroma"
)

web_search_tool = TavilySearchResults(k=3)

# --- 3. NODES ---

def decompose_query(state):
    """
    Breaks down complex questions into sub-queries.
    Example: "Compare Apple and Microsoft Revenue" -> ["Apple Revenue 2023", "Microsoft Revenue 2023"]
    """
    print("---DECOMPOSE QUERY---")
    question = state["question"]
    
    # Simple JSON parser for robustness
    parser = JsonOutputParser()
    
    prompt = ChatPromptTemplate.from_template(
        """You are a Query Decomposition Engine.
        Analyze the user's question.
        
        1. If it involves **comparison** (e.g., "Apple vs Microsoft"), break it into 2 distinct sub-queries.
        2. If it is a **single entity** question (e.g., "What is Apple's revenue?"), return just the original question.
        3. Optimize queries for vector search (remove fluff words).
        
        Return JSON format: {{ "queries": ["query 1", "query 2"] }}
        
        Question: {question}
        """
    )
    
    chain = prompt | llm | parser
    
    try:
        result = chain.invoke({"question": question})
        sub_queries = result.get("queries", [question])
        print(f"   -> Generated Sub-Queries: {sub_queries}")
    except:
        # Fallback if JSON fails
        sub_queries = [question]
        
    return {"sub_queries": sub_queries, "question": question}

def retrieve(state):
    """
    Retrieves documents for EACH sub-query and combines them.
    This ensures we get data for BOTH entities in a comparison.
    """
    print("---RETRIEVE (MULTI-STEP)---")
    sub_queries = state["sub_queries"]
    file_filter = state.get("file_filter", "All Documents")
    loop_count = state.get("loop_count", 0)
    
    all_documents = []
    
    # We use a smaller K because we are running multiple searches
    # Total chunks = (Number of queries) * k_per_query
    # 2 queries * 3 chunks = 6 chunks total (Safe for Rate Limits)
    k_per_query = 3 
    
    for query in sub_queries:
        print(f"   -> Searching for: '{query}'")
        if file_filter and file_filter != "All Documents":
            docs = vectorstore.similarity_search(query, k=k_per_query, filter={"source": file_filter})
        else:
            docs = vectorstore.similarity_search(query, k=k_per_query)
        all_documents.extend(docs)
        
    # Deduplicate documents (in case sub-queries overlap)
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
    
    # We can skip strict Pydantic grading for now to save API calls (Rate Limit Optimization)
    # Or keep a lightweight check. Let's do a quick check.
    
    # Simplified Grader for speed
    filtered_docs = []
    web_search = "No"
    
    # If we have NO docs, definitely search web
    if not documents:
        print("   -> No docs found. Web Search needed.")
        return {"documents": [], "web_search": "Yes"}

    # Optimization: Just assume top docs are relevant if we did Decomposition
    # This saves N LLM calls for grading.
    # But if user asked for "Live Stock", we must check.
    
    if "live" in question.lower() or "stock price" in question.lower() or "today" in question.lower():
        # Check if any doc is from Web
        has_web = any("Live Web" in d.metadata.get("source", "") for d in documents)
        if not has_web:
             web_search = "Yes"
             
    return {"documents": documents, "question": question, "web_search": web_search}

def transform_query(state):
    print("---TRANSFORM QUERY---")
    question = state["question"]
    # Re-write for Google
    msg = [("human", f"Refine this query for a Google search to find missing financial data: {question}")]
    better_question = llm.invoke(msg).content
    return {"question": better_question}

def web_search_node(state):
    print("---WEB SEARCH---")
    question = state["question"]
    documents = state.get("documents", [])
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
    
    web_doc = Document(page_content=web_content, metadata={"source": "Live Web Search"})
    
    if documents is None:
        documents = [web_doc]
    else:
        documents.append(web_doc)
        
    return {"documents": documents, "web_search": "Done", "loop_count": loop_count + 1}

def generate(state):
    print("---GENERATE---")
    question = state["question"]
    documents = state["documents"]
    loop_count = state.get("loop_count", 0)
    
    # Truncate context
    full_context = "\n\n".join([f"Source: [{d.metadata.get('source', 'Unknown')}]\nContent: {d.page_content}" for d in documents])
    truncated_context = full_context[:12000] 
    
    prompt = ChatPromptTemplate.from_template(
        """You are a Senior Financial Analyst "Arth".
        
        ### INSTRUCTIONS:
        1. **Deep Comparison:** If comparing, explicitly contrast numbers (e.g., "Apple: $X vs Microsoft: $Y").
        2. **Use the Context:** Answer strictly based on the provided text.
        3. **Missing Data:** If a number is missing, check if [Live Web Search] data is available. If not, state "Data missing."
        4. **Formatting:** Use proper spacing. Do NOT mash words together. Use Markdown tables.
        5. **NO DOLLAR SIGNS:** Do NOT use the "$" symbol. Always use "USD" or "dollars" (e.g. "100 million USD"). The "$" symbol crashes the text formatting.
    
        
        ### VISUALIZATION REQUEST:
        If the answer involves comparing numbers (e.g., Revenue A vs B) or trends (e.g., 2022-2024), you MUST append a JSON block at the very end.
        
        Format exactly like this:
        ```json
        {{
            "bar_chart": {{
                "labels": ["Apple", "Microsoft"],
                "datasets": [
                    {{ "label": "Revenue (Billions)", "data": [383, 211] }}
                ]
            }}
        }}
        ```
        
        ### CONTEXT:
        {context}
        
        ### QUESTION:
        {question}
        
        Answer:"""
    )
    
    chain = prompt | llm | StrOutputParser()
    generation = chain.invoke({"context": truncated_context, "question": question})
    
    return {"generation": generation, "loop_count": loop_count}

def check_hallucination(state):
    print("---CHECK VALIDITY---")
    generation = state["generation"]
    web_search = state.get("web_search", "No")
    loop_count = state.get("loop_count", 0)
    
    # --- AGGRESSIVE FAILURE DETECTION ---
    # If ANY of these are found, we force a Web Search
    failure_phrases = [
        "context does not provide", 
        "information is not available", 
        "context missing", 
        "data missing",           # <--- Catch this!
        "not mentioned",          # <--- Catch this!
        "not provided",           # <--- Catch this!
        "cannot directly compare" # <--- Catch "I can't compare"
    ]
    
    # Check if we failed to answer fully
    if any(p in generation.lower() for p in failure_phrases):
        if loop_count < 1:
            print("   -> 🚨 Answer Incomplete. ACTIVATING WEB SEARCH RESCUE.")
            return "web_search_node"
        else:
            return "end"
            
    return "end"

def decide_to_generate(state):
    if state["web_search"] == "Yes":
        return "transform_query"
    else:
        return "generate"

# --- 4. GRAPH BUILD ---
workflow = StateGraph(GraphState)

# Add Nodes
workflow.add_node("decompose_query", decompose_query) # <--- NEW NODE
workflow.add_node("retrieve", retrieve)
workflow.add_node("grade_documents", grade_documents)
workflow.add_node("generate", generate)
workflow.add_node("transform_query", transform_query)
workflow.add_node("web_search_node", web_search_node)

# Set Entry Point (Start with Decomposition)
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