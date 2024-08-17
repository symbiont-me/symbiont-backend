import pytest
import mongomock
from symbiont.vector_dbs.chat_context_service import get_vec_refs_from_db


# Mock the studies_collection
@pytest.fixture
def mock_studies_collection(monkeypatch):
    mock_db = mongomock.MongoClient().db
    studies_collection = mock_db.studies_collection

    # Insert a mock study document
    studies_collection.insert_one(
        {
            "_id": "test_study_id",
            "vectors": {
                "test_file_identifier": {
                    "vec_id_1": {"source": "source1", "page": "1", "text": "text1"},
                    "vec_id_2": {"source": "source2", "page": "2", "text": "text2"},
                }
            },
        }
    )

    monkeypatch.setattr("symbiont.vector_dbs.chat_context_service.studies_collection", studies_collection)
    return studies_collection


def test_get_vec_refs_from_db(mock_studies_collection):
    study_id = "test_study_id"
    file_identifier = "test_file_identifier"
    ids = ["vec_id_1", "vec_id_2"]

    results = get_vec_refs_from_db(study_id, file_identifier, ids)

    assert len(results) == 2
    assert results[0] == {"source": "source1", "page": "1", "text": "text1"}
    assert results[1] == {"source": "source2", "page": "2", "text": "text2"}


def test_get_vec_refs_from_db_study_not_found(mock_studies_collection):
    study_id = "non_existent_study_id"
    file_identifier = "test_file_identifier"
    ids = ["vec_id_1", "vec_id_2"]

    with pytest.raises(ValueError, match="Study not found"):
        get_vec_refs_from_db(study_id, file_identifier, ids)


def test_get_vec_refs_from_db_no_vectors(mock_studies_collection):
    study_id = "test_study_id"
    file_identifier = "non_existent_file_identifier"
    ids = ["vec_id_1", "vec_id_2"]

    results = get_vec_refs_from_db(study_id, file_identifier, ids)

    assert len(results) == 2
    assert results[0] == {}
    assert results[1] == {}
