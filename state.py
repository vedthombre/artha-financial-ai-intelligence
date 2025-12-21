from typing import List, TypedDict

class GraphState(TypedDict):
    """
    Represents the state of our graph.
    
    Attributes:
        question: The user's question
        generation: The LLM's final answer
        web_search: (str) "Yes" or "No" - decision to search web
        documents: List of retrieved documents (facts)
    """
    question: str
    generation: str
    web_search: str
    documents: List[str]
    