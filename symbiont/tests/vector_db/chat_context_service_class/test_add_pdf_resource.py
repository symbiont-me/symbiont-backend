import pytest
from unittest.mock import MagicMock, patch
from symbiont.vector_dbs.chat_context_service import ChatContextService, DocumentPage


@pytest.fixture
def mock_service():
    with patch("symbiont.vector_dbs.chat_context_service.create_vec_refs_in_db") as mock_create_vec_refs_in_db, patch(
        "symbiont.vector_dbs.chat_context_service.VectorStoreContext.__init__", lambda x: None
    ):
        service = ChatContextService()
        service.resource_doc = [DocumentPage(page_content="Test content", metadata={"page": "1"})]
        service.resource_identifier = "test_identifier"
        service.user_id = "test_user"
        service.study_id = "test_study"
        service.vector_store_repo = MagicMock()
        service.vector_store_repo.upsert_vectors.return_value = ["id1", "id2"]
        service._ChatContextService__parse_pdf_doc = MagicMock(
            return_value=[DocumentPage(page_content="Parsed content", metadata={"page": "1"})]
        )
        yield service, mock_create_vec_refs_in_db


def test_add_pdf_resource(mock_service):
    service, mock_create_vec_refs_in_db = mock_service

    # Call the method
    service.add_pdf_resource()

    # Assertions
    service.vector_store_repo.upsert_vectors.assert_called_once_with(
        "test_identifier", [DocumentPage(page_content="Parsed content", metadata={"page": "1"})]
    )
    mock_create_vec_refs_in_db.assert_called_once_with(
        ["id1", "id2"],
        "test_identifier",
        [DocumentPage(page_content="Parsed content", metadata={"page": "1"})],
        "test_user",
        "test_study",
    )
