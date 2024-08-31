import pytest
from unittest.mock import MagicMock, patch
from symbiont.vector_dbs.chat_context_service import ChatContextService, VectorSearchResult, Citation


@pytest.fixture
def chat_context_service():
    service = ChatContextService(resource_identifier="test_resource", user_id="test_user", study_id="test_study")
    return service


@patch("symbiont.vector_dbs.chat_context_service.get_vec_refs_from_db")
@patch("symbiont.vector_dbs.chat_context_service.studies_collection.find_one")
def test_get_combined_chat_context(mock_find_one, mock_get_vec_refs_from_db, chat_context_service):
    # Mock the vector_store_repo attribute directly on the service instance
    chat_context_service.vector_store_repo = MagicMock()
    chat_context_service.vector_store_repo.search_vectors.side_effect = [
        [VectorSearchResult(id="vec1", score=0.9)],
        [VectorSearchResult(id="vec2", score=0.8)],
    ]

    # Mock the find_one function to return a study with resources
    mock_find_one.return_value = {
        "_id": "test_study",
        "resources": [{"identifier": "resource1"}, {"identifier": "resource2"}],
    }

    # Mock the get_vec_refs_from_db function
    mock_get_vec_refs_from_db.return_value = [
        {"id": "vec1", "text": "sample text 1", "source": "source1", "page": "1"},
        {"id": "vec2", "text": "sample text 2", "source": "source2", "page": "2"},
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
    result = chat_context_service.get_combined_chat_context(query)

    # Validate the input and output
    assert result == (
        "reranked text",
        [
            Citation(text="sample text 1", source="source1", page=1),
            Citation(text="sample text 2", source="source2", page=2),
        ],
    )
    chat_context_service.vector_store_repo.search_vectors.assert_any_call(namespace="resource1", query=query, limit=10)
    chat_context_service.vector_store_repo.search_vectors.assert_any_call(namespace="resource2", query=query, limit=10)
    mock_get_vec_refs_from_db.assert_any_call("test_study", "resource1", ["vec1", "vec2"])
    mock_get_vec_refs_from_db.assert_any_call("test_study", "resource2", ["vec1", "vec2"])
    chat_context_service.rerank_context.assert_called_once()


@patch("symbiont.vector_dbs.chat_context_service.studies_collection.find_one")
def test_get_combined_chat_context_exception(mock_find_one, chat_context_service):
    # Mock the find_one function to raise an exception
    mock_find_one.side_effect = Exception("Test exception")

    query = "test query"
    result = chat_context_service.get_combined_chat_context(query)

    # Validate that the method returns None when an exception occurs
    assert result is None
    mock_find_one.assert_called_once_with({"_id": "test_study"})
