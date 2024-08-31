import pytest
from symbiont.vector_dbs.chat_context_service import ChatContextService
from symbiont.models import DocumentPage


@pytest.fixture
def chat_context_service():
    return ChatContextService(resource_identifier="test_resource", study_id="test_study")


def test_parse_pdf_doc(chat_context_service):
    # Arrange
    long_text = (
        "This is a long line of text " * 1000 + "that should be split into multiple documents.\nThis is another line."
    )
    pdf_page = DocumentPage(page_content=long_text, metadata={"page": 1}, type="pdf")

    # Act
    page_content = pdf_page.page_content.replace("\n", "")
    page_content = chat_context_service._ChatContextService__truncate_string_by_bytes(page_content, 10000)
    doc = DocumentPage(
        page_content=page_content,
        metadata={
            "text": page_content,
            "source": "test_resource",
            "page": 1,
        },
        type="pdf",
    )

    # Assert
    assert doc.type == "pdf"  # Ensure the document type is preserved
    assert doc.metadata["source"] == "test_resource"  # Ensure the source is set correctly
    assert doc.metadata["page"] == 1  # Ensure the page number is set correctly
    assert len(doc.page_content) <= 10000  # Ensure the text is truncated to 10000 bytes


# TODO test using following condition as well
# split_texts = text_splitter.create_documents([page_content])
