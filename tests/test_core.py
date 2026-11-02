import pytest
from langchain_core.documents import Document

import graph
import ingest


class FakeVectorStore:
    def __init__(self, documents):
        self.documents = documents
        self.calls = []

    def similarity_search(self, query, k, filter=None):
        self.calls.append((query, k, filter))
        return self.documents


class FakeWebSearch:
    def invoke(self, payload):
        return [{"content": f"result for {payload['query']}"}]


class FakeLLM:
    def __init__(self, content):
        self.content = content
        self.messages = []

    def invoke(self, messages):
        self.messages.append(messages)
        return type("Response", (), {"content": self.content})()


def test_retrieve_deduplicates_documents_and_applies_file_filter(monkeypatch):
    duplicate = Document(page_content="same", metadata={"source": "report.pdf"})
    other = Document(page_content="other", metadata={"source": "report.pdf"})
    store = FakeVectorStore([duplicate, duplicate, other])
    monkeypatch.setattr(graph, "vectorstore", store)

    result = graph.retrieve(
        {
            "sub_queries": ["revenue", "profit"],
            "file_filter": "report.pdf",
            "loop_count": 0,
        }
    )

    assert [doc.page_content for doc in result["documents"]] == ["same", "other"]
    assert store.calls == [
        ("revenue", 10, {"source": "report.pdf"}),
        ("profit", 10, {"source": "report.pdf"}),
    ]


def test_retrieve_keeps_statement_ranked_after_duplicate_candidates(monkeypatch):
    candidates = [
        Document(page_content=f"discussion {index}", metadata={"source": "report.pdf"})
        for index in range(8)
    ]
    candidates.append(
        Document(
            page_content="Income Statements: net income, income before taxes, and provision for income taxes.",
            metadata={"source": "report.pdf", "page_label": "56"},
        )
    )
    store = FakeVectorStore(candidates)
    monkeypatch.setattr(graph, "vectorstore", store)

    result = graph.retrieve(
        {
            "sub_queries": ["net income income before taxes provision for income taxes"],
            "file_filter": "report.pdf",
            "loop_count": 0,
        }
    )

    assert any("Income Statements" in doc.page_content for doc in result["documents"])


@pytest.mark.parametrize(
    "query, expected_text",
    [
        ("Microsoft FY2024 revenue", "Total revenue"),
        ("Microsoft FY2024 operating income", "Operating income"),
    ],
)
def test_retrieve_preserves_existing_financial_metric_queries(
    monkeypatch, query, expected_text
):
    store = FakeVectorStore(
        [
            Document(
                page_content=f"Income statement: Total revenue and Operating income for {query}.",
                metadata={"source": "report.pdf"},
            )
        ]
    )
    monkeypatch.setattr(graph, "vectorstore", store)

    result = graph.retrieve(
        {"sub_queries": [query], "file_filter": "report.pdf", "loop_count": 0}
    )

    assert expected_text in result["documents"][0].page_content


def test_grade_documents_routes_empty_context_to_web_search():
    result = graph.grade_documents({"question": "revenue", "documents": []})
    assert result["documents"] == []
    assert result["web_search"] == "Yes"
    assert result["document_only"] is False


def test_document_only_request_never_routes_to_web_search():
    result = graph.grade_documents(
        {
            "question": "Use only the uploaded document; do not use web search.",
            "documents": [],
        }
    )
    assert result["web_search"] == "No"
    assert result["document_only"] is True
    assert graph.check_hallucination(
        {"generation": "Data missing.", "loop_count": 0, "document_only": True}
    ) == "end"


@pytest.mark.parametrize(
    "question",
    ["What is the live stock price?", "What happened today?"],
)
def test_grade_documents_routes_live_questions_to_web_search(question):
    document = Document(page_content="context")
    result = graph.grade_documents({"question": question, "documents": [document]})
    assert result["web_search"] == "Yes"


def test_web_search_keeps_results_separate_and_increments_loop(monkeypatch):
    monkeypatch.setattr(graph, "web_search_tool", FakeWebSearch())

    result = graph.web_search_node(
        {"question": "Apple revenue", "documents": ["internal"], "loop_count": 1}
    )

    assert result["web_search_results"] == "\n[Source: Live Web] result for Apple revenue"
    assert result["loop_count"] == 2
    assert result["web_search"] == "Done"


def test_hallucination_rescue_stops_after_two_web_searches():
    state = {"generation": "Data missing.", "loop_count": 2}
    assert graph.check_hallucination(state) == "end"
    assert graph.check_hallucination({**state, "loop_count": 1}) == "web_search_node"


def test_generation_prompt_is_not_company_specific(monkeypatch):
    fake_llm = FakeLLM("A general answer.")
    monkeypatch.setattr(graph, "llm", fake_llm)

    result = graph.generate(
        {
            "question": "What is Apple revenue?",
            "documents": [Document(page_content="Apple revenue was 100 million.")],
            "web_search_results": "",
            "loop_count": 0,
        }
    )

    prompt_text = fake_llm.messages[0][0].content
    assert "Microsoft data found" not in prompt_text
    assert "NVIDIA 2024 data" not in prompt_text
    assert result["generation"] == "A general answer."


def test_generation_receives_all_requested_financial_context(monkeypatch):
    fake_llm = FakeLLM("The requested values are in the uploaded statement.")
    monkeypatch.setattr(graph, "llm", fake_llm)
    statement = Document(
        page_content=(
            "Income Statements (In millions): Net income for FY2024 and FY2023; "
            "income before taxes; provision for income taxes."
        ),
        metadata={"source": "microsoft_data.pdf", "page_label": "56"},
    )

    result = graph.generate(
        {
            "question": "Provide FY2024 and FY2023 net income and tax figures.",
            "documents": [statement],
            "web_search_results": "",
            "loop_count": 0,
        }
    )

    prompt_text = fake_llm.messages[0][1].content
    assert "Net income for FY2024 and FY2023" in prompt_text
    assert "microsoft_data.pdf" in prompt_text
    assert result["generation"] == "The requested values are in the uploaded statement."


@pytest.mark.parametrize("state", [{"question": ""}, {"question": None}])
def test_decompose_query_rejects_empty_questions(state):
    with pytest.raises(ValueError, match="non-empty"):
        graph.decompose_query(state)


def test_ingest_rejects_pdf_with_no_pages(monkeypatch, tmp_path):
    class EmptyLoader:
        def __init__(self, path):
            self.path = path

        def load(self):
            return []

    monkeypatch.setattr(ingest, "PyPDFLoader", EmptyLoader)
    pdf_path = tmp_path / "empty.pdf"
    pdf_path.write_bytes(b"%PDF-1.4")

    with pytest.raises(ValueError, match="No readable pages"):
        ingest.ingest_data(str(pdf_path), "empty.pdf")


def test_ingest_skips_an_identical_existing_document(monkeypatch, tmp_path):
    class ExistingStore:
        def __init__(self, *args, **kwargs):
            self.deleted = False

        def get(self, **kwargs):
            return {"ids": ["existing"], "documents": ["page text"]}

        def delete(self, ids):
            self.deleted = True

    class Loader:
        def __init__(self, path):
            self.path = path

        def load(self):
            return [Document(page_content="page text")]

    monkeypatch.setattr(ingest, "PyPDFLoader", Loader)
    monkeypatch.setattr(ingest, "Chroma", ExistingStore)
    monkeypatch.setattr(ingest, "HuggingFaceEmbeddings", lambda **kwargs: object())
    pdf_path = tmp_path / "existing.pdf"
    pdf_path.write_bytes(b"%PDF-1.4")

    ingest.ingest_data(str(pdf_path), "existing.pdf")
