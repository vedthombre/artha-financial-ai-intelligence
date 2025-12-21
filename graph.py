import os
from dotenv import load_dotenv

# --- IMPORTS ---
from langchain_groq import ChatGroq
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from langgraph.graph import END, StateGraph
from typing import List, TypedDict
from pydantic import BaseModel, Field

load_dotenv()

# --- 1. STATE DEFINITION (Fixed) ---
class GraphState(TypedDict):
    question: str
    generation: str
    web_search: str         # "Yes" or "No"
    documents: List[Document]
    file_filter: str
    loop_count: int         # Prevents infinite loops

# --- 2. SETUP ---
print("--- INITIALIZING AGENT ---")
# Using the NEW reliable model
llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

vectorstore = Chroma(
    persist_directory="./chroma_db", 
    embedding_function=embedding_model,
    collection_name="rag-chroma"
)

web_search_tool = TavilySearchResults(k=3)

# --- 3. NODES ---

def retrieve(state):
    print("---RETRIEVE---")
    question = state["question"]
    file_filter = state.get("file_filter", "All Documents")
    loop_count = state.get("loop_count", 0)
    
    # Use larger K to ensure we catch financial tables
    k_depth = 8
    
    if file_filter and file_filter != "All Documents":
        documents = vectorstore.similarity_search(
            question, k=k_depth, filter={"source": file_filter}
        )
    else:
        documents = vectorstore.similarity_search(question, k=k_depth)
        
    return {"documents": documents, "question": question, "file_filter": file_filter, "loop_count": loop_count}

def grade_documents(state):
    print("---CHECK RELEVANCE---")
    question = state["question"]
    documents = state["documents"]
    
    # Structured Output Wrapper
    class Grade(BaseModel):
        binary_score: str = Field(description="Relevance score 'yes' or 'no'")

    structured_llm_grader = llm.with_structured_output(Grade)
    
    # --- CORRECTED PROMPT FOR DOCUMENT GRADING ---
    system = """You are a grader assessing relevance of a retrieved document to a user question.
    
    1. If the document contains keywords or semantic meaning related to the question, grade it as "yes".
    2. **EXCEPTION:** If the user specifically asks for "LIVE" data (e.g., "live stock price", "today's news") and the document is an old PDF, grade it as "no" so the system searches the web.
    
    Give a binary score 'yes' or 'no'."""
    
    grade_prompt = ChatPromptTemplate.from_messages([
        ("system", system),
        ("human", "Retrieved document: \n\n {document} \n\n User question: {question}"),
    ])
    
    retrieval_grader = grade_prompt | structured_llm_grader
    
    filtered_docs = []
    web_search = "No"
    
    for d in documents:
        score = retrieval_grader.invoke({"question": question, "document": d.page_content})
        if score.binary_score.lower() == "yes":
            filtered_docs.append(d)
    
    # If all docs are irrelevant (or it's a Live Data request), force Web Search
    if len(filtered_docs) == 0:
        print("   -> GRADE: Irrelevant/Missing Live Data. Web Search Needed.")
        web_search = "Yes"
    else:
        print("   -> GRADE: Relevant. Proceeding.")
        
    return {"documents": filtered_docs, "question": question, "web_search": web_search}

def transform_query(state):
    """Optimizes query for Google Search"""
    print("---TRANSFORM QUERY---")
    question = state["question"]
    
    # Simple re-writer
    msg = [("human", f"Refine this query for a Google search to find missing financial data: {question}")]
    better_question = llm.invoke(msg).content
    
    return {"question": better_question}

def web_search_node(state):
    """Hybrid Search: Appends Web Data to PDF Data"""
    print("---WEB SEARCH---")
    question = state["question"]
    documents = state.get("documents", [])
    loop_count = state.get("loop_count", 0)
    
    # 1. Fetch from Tavily
    try:
        docs = web_search_tool.invoke({"query": question})
        web_content = ""
        
        # Parse safely
        if isinstance(docs, list):
            for d in docs:
                content = d.get('content', '') if isinstance(d, dict) else str(d)
                web_content += f"\n[Source: Live Web] {content}"
        else:
            web_content = str(docs)
            
    except Exception as e:
        web_content = f"Web search failed: {e}"
    
    # 2. Create Web Document
    web_doc = Document(page_content=web_content, metadata={"source": "Live Web Search"})
    
    # 3. Append to existing documents (Hybrid approach)
    if documents is None:
        documents = [web_doc]
    else:
        documents.append(web_doc)
    
    # 4. Increment loop count
    return {
        "documents": documents, 
        "question": question, 
        "web_search": "Done",
        "loop_count": loop_count + 1 
    }

def generate(state):
    print("---GENERATE---")
    question = state["question"]
    documents = state["documents"]
    loop_count = state.get("loop_count", 0)
    
    # Format Context
    context_text = "\n\n".join([f"{d.page_content} \n[Source: {d.metadata.get('source', 'Unknown')}]" for d in documents])
    
    # Prompt
    prompt = ChatPromptTemplate.from_template(
        """You are a Senior Financial Analyst.
        
        ### INSTRUCTIONS:
        1. Use BOTH the PDF context and Live Web Search results to answer.
        2. If the user asks for "Sales and Stock Price", combine the data from both sources.
        3. Citation is MANDATORY: [Page 22] or [Live Web Search].
        
        ### CONTEXT:
        {context}
        
        ### QUESTION:
        {question}
        
        Answer:"""
    )
    
    chain = prompt | llm | StrOutputParser()
    generation = chain.invoke({"context": context_text, "question": question})
    
    return {"generation": generation, "loop_count": loop_count}

# --- 4. VALIDATOR (The Safety Net) ---
def check_hallucination(state):
    """Checks if the answer is 'I don't know' and triggers a retry."""
    print("---CHECKING ANSWER VALIDITY---")
    generation = state["generation"]
    web_search = state.get("web_search", "No")
    loop_count = state.get("loop_count", 0)
    
    # Expanded list of failure phrases
    failure_phrases = [
        "context does not provide", "not provided in the context", 
        "cannot accurately calculate", "information is not available",
        "context missing", "publicly available information",
        "i do not have access", "undisclosed"
    ]
    
    # If failed AND we haven't searched yet
    if any(p in generation.lower() for p in failure_phrases):
        if loop_count < 1: # Only retry once
            print("   -> ANSWER FAILED. ACTIVATING WEB SEARCH RESCUE.")
            return "web_search_node"
        else:
            print("   -> ANSWER FAILED (Retry limit reached).")
            return "end"
            
    print("   -> ANSWER VALID.")
    return "end"

def decide_to_generate(state):
    if state["web_search"] == "Yes":
        return "transform_query"
    else:
        return "generate"

# --- 5. GRAPH BUILD ---
workflow = StateGraph(GraphState)

workflow.add_node("retrieve", retrieve)
workflow.add_node("grade_documents", grade_documents)
workflow.add_node("generate", generate)
workflow.add_node("transform_query", transform_query)
workflow.add_node("web_search_node", web_search_node)

workflow.set_entry_point("retrieve")
workflow.add_edge("retrieve", "grade_documents")
workflow.add_conditional_edges(
    "grade_documents",
    decide_to_generate,
    {"transform_query": "transform_query", "generate": "generate"},
)

workflow.add_edge("transform_query", "web_search_node")
workflow.add_edge("web_search_node", "generate")

# Connect Generate -> Validator
workflow.add_conditional_edges(
    "generate",
    check_hallucination,
    {
        "web_search_node": "web_search_node",
        "end": END
    }
)

app = workflow.compile()