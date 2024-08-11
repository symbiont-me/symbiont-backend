import pytest
from unittest.mock import patch, MagicMock
from symbiont.vector_dbs.vector_service import ChatContextService
from symbiont.models import DocumentPage


@pytest.fixture
def mock_document_page():
    return DocumentPage(
        page_content="This is a sample PDF content.\nIt has multiple lines.\n", metadata={"page": "1"}, type="pdf"
    )


@patch("symbiont.vector_dbs.vector_service.text_splitter.create_documents")
def test_parse_pdf_doc(mock_create_documents, mock_document_page):
    service = ChatContextService(
        resource_doc="sample_doc",
        resource_identifier="sample_identifier",
        resource_type="pdf",
        user_id="user_123",
        user_query="sample query",
        study_id="study_123",
    )

    # Mock the output of text_splitter.create_documents
    mock_split_text = MagicMock()
    mock_split_text.page_content = "This is a sample PDF content."
    mock_create_documents.return_value = [mock_split_text]

    docs = service._ChatContextService__parse_pdf_doc(mock_document_page)

    assert len(docs) == 1
    assert docs[0].page_content == "This is a sample PDF content."
    assert docs[0].metadata["text"] == "This is a sample PDF content."
    assert docs[0].metadata["source"] == "sample_identifier"
    assert docs[0].metadata["page"] == "1"
    assert docs[0].type == "pdf"

    # Test with content that exceeds 10000 bytes
    long_content = "a" * 10001
    mock_document_page.page_content = long_content
    mock_create_documents.return_value = [mock_split_text]

    docs = service._ChatContextService__parse_pdf_doc(mock_document_page)

    assert len(docs) == 1
    assert len(docs[0].page_content) <= 10000
    assert docs[0].metadata["text"] == "This is a sample PDF content."
    assert docs[0].metadata["source"] == "sample_identifier"
    assert docs[0].metadata["page"] == "1"
    assert docs[0].type == "pdf"
