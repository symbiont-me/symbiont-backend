import pytest
import mongomock
from symbiont.vector_dbs.chat_context_service import create_vec_refs_in_db, DocumentPage
from unittest.mock import patch


# Mock the studies_collection
@mongomock.patch(servers=(("localhost", 27017),))
def test_create_vec_refs_in_db():
    # Arrange
    mock_studies_collection = mongomock.MongoClient().db.collection
    with patch("symbiont.vector_dbs.context_service.studies_collection", mock_studies_collection):
        ids = ["id1", "id2"]
        file_identifier = "file1"
        docs = [
            DocumentPage(page_content="content1", metadata={"source": "source1", "page": "1"}),
            DocumentPage(page_content="content2", metadata={"source": "source2", "page": "2"}),
        ]
        user_id = "user1"
        study_id = "study1"

        # Act
        create_vec_refs_in_db(ids, file_identifier, docs, user_id, study_id)

        # Assert
        study = mock_studies_collection.find_one({"_id": study_id})
        assert study is not None
        assert "vectors" in study
        assert file_identifier in study["vectors"]
        assert len(study["vectors"][file_identifier]) == 2
        assert study["vectors"][file_identifier]["id1"]["text"] == "content1"
        assert study["vectors"][file_identifier]["id2"]["text"] == "content2"


if __name__ == "__main__":
    pytest.main()
