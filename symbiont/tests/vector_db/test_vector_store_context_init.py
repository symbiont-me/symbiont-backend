import pytest
from symbiont.vector_dbs.vector_service import VectorStoreContext, vector_store_settings, vector_store_repos


# Mocking the vector_store_settings and vector_store_repos for testing
class MockVectorStoreRepo:
    def __init__(self):
        pass


@pytest.fixture(autouse=True)
def mock_vector_store_settings(monkeypatch):
    monkeypatch.setattr(vector_store_settings, "vector_store", "mock_store")
    monkeypatch.setattr(vector_store_settings, "vector_store_url", "http://mockurl")
    monkeypatch.setattr(vector_store_settings, "vector_store_port", "1234")
    monkeypatch.setattr(vector_store_settings, "vector_store_dimension", "128")
    monkeypatch.setattr(vector_store_settings, "vector_store_distance", "cosine")
    monkeypatch.setattr(vector_store_settings, "vector_store_token", "mock_token")


@pytest.fixture(autouse=True)
def mock_vector_store_repos(monkeypatch):
    monkeypatch.setitem(vector_store_repos, "mock_store", MockVectorStoreRepo)


def test_vector_store_context_initialization():
    context = VectorStoreContext()
    assert context.vector_store == "mock_store"
    assert isinstance(context.vector_store_repo, MockVectorStoreRepo)


def test_vector_store_context_invalid_store(monkeypatch):
    with pytest.raises(ValueError, match="Vector store not supported"):
        monkeypatch.setattr(vector_store_settings, "vector_store", "invalid_store")
        VectorStoreContext()


def test_vector_store_context_no_store_name(monkeypatch):
    with pytest.raises(ValueError, match="Set the Vector Store name"):
        monkeypatch.setattr(vector_store_settings, "vector_store", None)
        VectorStoreContext()
