import pytest
from unittest.mock import MagicMock, patch
from symbiont.vector_dbs.chat_context_service import ChatContextService, DocumentPage


@pytest.fixture
def mock_service():
    with patch("symbiont.vector_dbs.vector_service.create_vec_refs_in_db") as mock_create_vec_refs_in_db, patch(
        "symbiont.vector_dbs.vector_service.VectorStoreContext.__init__", lambda x: None
    ):
        service = ChatContextService()
        service.resource_doc = MagicMock(page_content="Test content")
        service.resource_identifier = "test_identifier"
        service.user_id = "test_user"
        service.study_id = "test_study"
        service.vector_store_repo = MagicMock()
        service.vector_store_repo.upsert_vectors.return_value = ["id1", "id2"]
        yield service, mock_create_vec_refs_in_db


def test_add_web_resource(mock_service):
    service, mock_create_vec_refs_in_db = mock_service

    # Mock the text_splitter
    with patch(
        "symbiont.vector_dbs.vector_service.text_splitter.create_documents",
        return_value=[DocumentPage(page_content="Parsed content", metadata={})],
    ):
        # Call the method
        service.add_web_resource()

        # Assertions
        service.vector_store_repo.upsert_vectors.assert_called_once_with(
            "test_identifier",
            [
                DocumentPage(
                    page_content="Parsed content",
                    metadata={"text": "Parsed content", "source": "test_identifier", "page": 0},
                )
            ],
        )
        mock_create_vec_refs_in_db.assert_called_once_with(
            ["id1", "id2"],
            "test_identifier",
            [
                DocumentPage(
                    page_content="Parsed content",
                    metadata={"text": "Parsed content", "source": "test_identifier", "page": 0},
                )
            ],
            "test_user",
            "test_study",
        )
