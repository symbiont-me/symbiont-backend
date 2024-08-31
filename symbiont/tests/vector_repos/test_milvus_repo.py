import pytest
from pymilvus import MilvusClient
from symbiont.models import DocumentPage
import numpy as np

from symbiont.vector_dbs.vector_store_repos.milvus_repo import (
    init_milvus,
    create_collection,
    upsert_vectors,
    dummy_data,
    vectors,
    data,
)


@pytest.fixture
def mock_client():
    client = MilvusClient("milvus_demo.db")
    return client


def test_init_milvus(mock_client):
    init_milvus()
    assert mock_client


def test_upsert_vectors(mock_client):
    mock_client.create_collection(collection_name="test", dimension=384)
    # Use the in-memory client to insert data
    res = mock_client.insert(collection_name="test", data=data)
    assert res is not None
    assert res["ids"] is not None
    assert len(res["ids"]) is len(data)


def test_search_vectors(mock_client):
    mock_client.create_collection(collection_name="test", dimension=384)
    # Use the in-memory client to insert data
    res = mock_client.insert(collection_name="test", data=data)
    assert res is not None
    assert res["ids"] is not None
    assert len(res["ids"]) is len(data)

    res = mock_client.search(
        collection_name="test",
        data=[vectors[0]],
        filter="subject == 'history'",
        limit=2,
        output_fields=["text", "subject"],
    )

    assert res
    print(res)


def test_delete_vectors(mock_client):
    mock_client.create_collection(collection_name="test", dimension=384)
    # Use the in-memory client to insert data
    res = mock_client.insert(collection_name="test", data=data)
    assert res is not None
    assert res["ids"] is not None
    assert len(res["ids"]) is len(data)

    mock_client.drop_collection(collection_name="test")
    assert mock_client.has_collection(collection_name="test") is False
