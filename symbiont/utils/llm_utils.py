# TODO this file should be in the llms folder
from langchain.chains.summarize import load_summarize_chain
from langchain.text_splitter import NLTKTextSplitter
from ..models import LLMModel
from langchain_google_genai import ChatGoogleGenerativeAI
from .. import logger
from ..llms import google_api_key

if google_api_key is None:
    logger.error("GOOGLE_GEMINI_API_KEY is not set in the environment variables. Summaries will not work.")


# TODO fix summariser
# sort of a dumb tokenizer but works
def truncate_prompt(prompt: str, query_size=500, model_context_window=4096) -> str:
    max_prompt_tokens = model_context_window - query_size
    tokens = prompt.split()
    if len(tokens) > max_prompt_tokens:
        tokens = tokens[:max_prompt_tokens]
        print(len(tokens))
    return " ".join(tokens)


# NOTE used for webpages and plain text
def summarise_plain_text_resource(document_text):
    prompt = document_text
    summary = ""
    try:
        google_llm = ChatGoogleGenerativeAI(
            model=LLMModel.GEMINI_1_PRO_LATEST,
            google_api_key=google_api_key,
            temperature=0,
            convert_system_message_to_human=True,
        )
        nltk_text_splitter = NLTKTextSplitter()
        docs = nltk_text_splitter.create_documents([prompt])
        summary_chain = load_summarize_chain(
            llm=google_llm,
            chain_type="map_reduce",
            verbose=False,  # Set verbose=True if you want to see the prompts being used
        )
        summary = summary_chain.run(docs)
        return summary
    except Exception as e:
        logger.error(f"An error occurred: {e}")
        summary = "An error occurred: Summary could not be generated."
    return summary
