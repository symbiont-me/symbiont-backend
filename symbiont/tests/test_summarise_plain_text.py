import pytest
from unittest.mock import patch, MagicMock
from symbiont.utils.llm_utils import summarise_plain_text_resource


@patch("symbiont.utils.llm_utils.ChatGoogleGenerativeAI")
@patch("symbiont.utils.llm_utils.load_summarize_chain")
@patch("symbiont.utils.llm_utils.NLTKTextSplitter")
def test_summarise_plain_text_resource(
    mock_nltk_text_splitter, mock_load_summarize_chain, mock_chat_google_generative_ai
):
    # Mock the NLTKTextSplitter
    mock_splitter_instance = MagicMock()
    mock_splitter_instance.create_documents.return_value = ["doc1", "doc2"]
    mock_nltk_text_splitter.return_value = mock_splitter_instance

    # Mock the summarize chain
    mock_summary_chain_instance = MagicMock()
    mock_summary_chain_instance.run.return_value = "This is a summary."
    mock_load_summarize_chain.return_value = mock_summary_chain_instance

    # Mock the ChatGoogleGenerativeAI
    mock_google_llm_instance = MagicMock()
    mock_chat_google_generative_ai.return_value = mock_google_llm_instance

    # Call the function
    document_text = "This is a test document."
    summary = summarise_plain_text_resource(document_text)

    # Assertions
    mock_nltk_text_splitter.assert_called_once()
    mock_splitter_instance.create_documents.assert_called_once_with([document_text])
    mock_load_summarize_chain.assert_called_once_with(
        llm=mock_google_llm_instance, chain_type="map_reduce", verbose=False
    )
    mock_summary_chain_instance.run.assert_called_once_with(["doc1", "doc2"])
    assert summary == "This is a summary."


def test_summarise_plain_text_resource_exception():
    with patch("symbiont.utils.llm_utils.ChatGoogleGenerativeAI", side_effect=Exception("Test exception")):
        document_text = "This is a test document."
        summary = summarise_plain_text_resource(document_text)
        assert summary == "An error occurred: Summary could not be generated."
