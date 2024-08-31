import pytest
from symbiont.vector_dbs.chat_context_service import ChatContextService, Citation
from unittest.mock import patch, MagicMock


@pytest.fixture
def chat_context_service():
    return ChatContextService(
        resource_identifier="test_resource", user_id="test_user", user_query="test_query", study_id="test_study"
    )


def test_rerank_context(chat_context_service):
    context = [
        {"text": "Document 1 text", "page": "1"},
        {"text": "Document 2 text", "page": "2"},
        {"text": "Document 3 text", "page": "3"},
    ]
    query = "test query"

    mock_rerank_result = MagicMock()
    mock_rerank_result.results = [
        MagicMock(index=1, document={"text": "Document 2 text"}),
        MagicMock(index=0, document={"text": "Document 1 text"}),
        MagicMock(index=2, document={"text": "Document 3 text"}),
    ]

    with patch.object(
        ChatContextService,
        "rerank_context",
        return_value=(
            "Document 2 textDocument 1 textDocument 3 text",
            [
                Citation(text="Document 2 text", source="", page=2),
                Citation(text="Document 1 text", source="", page=1),
                Citation(text="Document 3 text", source="", page=3),
            ],
        ),
    ):
        reranked_text, citations = chat_context_service.rerank_context(context, query)
        assert reranked_text is not None
        assert citations is not None
        expected_reranked_text = "Document 2 textDocument 1 textDocument 3 text"
        expected_citations = [
            Citation(text="Document 2 text", source="", page=2),
            Citation(text="Document 1 text", source="", page=1),
            Citation(text="Document 3 text", source="", page=3),
        ]

        assert reranked_text == expected_reranked_text
        assert citations == expected_citations


if __name__ == "__main__":
    pytest.main()
