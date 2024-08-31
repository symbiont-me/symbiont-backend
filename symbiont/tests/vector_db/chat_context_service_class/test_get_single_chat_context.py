import pytest
from unittest.mock import MagicMock, patch
from symbiont.vector_dbs.chat_context_service import ChatContextService, VectorSearchResult, Citation


@pytest.fixture
def chat_context_service():
    service = ChatContextService(resource_identifier="test_resource", user_id="test_user", study_id="test_study")
    return service


@patch("symbiont.vector_dbs.chat_context_service.get_vec_refs_from_db")
def test_get_single_chat_context(mock_get_vec_refs_from_db, chat_context_service):
    # Mock the vector_store_repo attribute directly on the service instance
    chat_context_service.vector_store_repo = MagicMock()
    chat_context_service.vector_store_repo.search_vectors.return_value = [
        VectorSearchResult(id="vec1", score=0.9),
        VectorSearchResult(id="vec2", score=0.8),
    ]

    # Mock the get_vec_refs_from_db function
    mock_get_vec_refs_from_db.return_value = [
        MagicMock(model_dump=lambda: {"id": "vec1", "text": "sample text 1", "source": "source1", "page": "1"}),
        MagicMock(model_dump=lambda: {"id": "vec2", "text": "sample text 2", "source": "source2", "page": "2"}),
    ]

    # Mock the rerank_context method
    chat_context_service.rerank_context = MagicMock(
        return_value=(
            "reranked text",
            [
                Citation(text="sample text 1", source="source1", page=1),
                Citation(text="sample text 2", source="source2", page=2),
            ],
        )
    )

    query = "test query"
    result = chat_context_service.get_single_chat_context(query)

    # Validate the input and output
    assert result == (
        "reranked text",
        [
            Citation(text="sample text 1", source="source1", page=1),
            Citation(text="sample text 2", source="source2", page=2),
        ],
    )
    chat_context_service.vector_store_repo.search_vectors.assert_called_once_with(
        namespace="test_resource", query=query, limit=10
    )
    mock_get_vec_refs_from_db.assert_called_once_with("test_study", "test_resource", ["vec1", "vec2"])
    chat_context_service.rerank_context.assert_called_once()


@patch("symbiont.vector_dbs.chat_context_service.get_vec_refs_from_db")
def test_get_single_chat_context_exception(mock_get_vec_refs_from_db, chat_context_service):
    # Mock the vector_store_repo attribute directly on the service instance to raise an exception
    chat_context_service.vector_store_repo = MagicMock()
    chat_context_service.vector_store_repo.search_vectors.side_effect = Exception("Test exception")

    query = "test query"
    result = chat_context_service.get_single_chat_context(query)

    # Validate that the method returns None when an exception occurs
    assert result is None
    chat_context_service.vector_store_repo.search_vectors.assert_called_once_with(
        namespace="test_resource", query=query, limit=10
    )
