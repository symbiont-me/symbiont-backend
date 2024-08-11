import pytest
from unittest.mock import MagicMock, patch
from symbiont.vector_dbs.vector_service import (
    QdrantRepository,
    VectorStoreContext,
    create_vec_refs_in_db,
    get_vec_refs_from_db,
    VectorMetadata,
    DocumentPage,
    VectorSearchResult,
    vector_store_settings,
)
from qdrant_client import QdrantClient


# TODO use in memory qdrant client for testing
# Mock settings
vector_store_settings.vector_store = "qdrant"
vector_store_settings.vector_store_url = "http://localhost"
vector_store_settings.vector_store_port = "6333"
vector_store_settings.vector_store_token = "test_token"
vector_store_settings.vector_store_dimension = "1536"
vector_store_settings.vector_store_distance = "dot"


# Mock QdrantClient using in-memory storage
@pytest.fixture
def mock_qdrant_client():
    with patch("symbiont.vector_dbs.vector_service.QdrantClient") as MockClient:
        instance = MockClient.return_value
        instance.client = QdrantClient(":memory:")  # Use in-memory client
        yield MockClient


# Test QdrantRepository
def test_qdrant_repository_init(mock_qdrant_client):
    repo = QdrantRepository()
    assert repo.client is not None
    assert repo.dimension == vector_store_settings.vector_store_dimension
    assert repo.distance == vector_store_settings.vector_store_distance


def test_create_collection(mock_qdrant_client):
    repo = QdrantRepository()
    repo.client.collection_exists.return_value = False
    repo.create_collection("test_collection", 1536, "dot")
    repo.client.create_collection.assert_called_once()


def test_upsert_vectors(mock_qdrant_client):
    repo = QdrantRepository()
    docs = [DocumentPage(page_content="test", metadata={"page": "1"})]
    repo.client.collection_exists.return_value = False
    repo.client.upsert.return_value = None
    ids = repo.upsert_vectors("test_namespace", docs)
    assert len(ids) == len(docs)


def test_search_vectors(mock_qdrant_client):
    repo = QdrantRepository()
    repo.client.search.return_value = [MagicMock(id="1", score=0.9)]
    results = repo.search_vectors("test_namespace", "test_query", 10)
    assert len(results) == 1
    assert results[0].id == "1"


def test_delete_vectors(mock_qdrant_client):
    repo = QdrantRepository()
    repo.delete_vectors("test_namespace")
    repo.client.delete_collection.assert_called_once()


# Test VectorStoreContext
def test_vector_store_context():
    context = VectorStoreContext()
    assert context.vector_store_repo is not None


# Test create_vec_refs_in_db
def test_create_vec_refs_in_db():
    with patch("symbiont.vector_dbs.vector_service.studies_collection") as mock_collection:
        docs = [DocumentPage(page_content="test", metadata={"page": "1"})]
        ids = ["1"]
        create_vec_refs_in_db(ids, "file_id", docs, "user_id", "study_id")
        mock_collection.update_one.assert_called_once()


# Test get_vec_refs_from_db
def test_get_vec_refs_from_db():
    with patch("symbiont.vector_dbs.vector_service.studies_collection") as mock_collection:
        mock_collection.find_one.return_value = {
            "vectors": {"file_id": {"1": {"source": "file_id", "page": "1", "text": "test"}}}
        }
        results = get_vec_refs_from_db("study_id", "file_id", ["1"])
        assert len(results) == 1
        assert results[0]["text"] == "test"
