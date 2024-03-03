from langchain import OpenAI
from langchain.chains.summarize import load_summarize_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter, NLTKTextSplitter
from langchain_openai import OpenAI
from ..models import LLMModel


llm = OpenAI(temperature=0, name=LLMModel.GPT_3_5_TURBO_16K)


# NOTE used for webpages and plain text
def summarise_plain_text_resource(document_text):

    nltk_text_splitter = NLTKTextSplitter()

    docs = nltk_text_splitter.create_documents([document_text])

    summary_chain = load_summarize_chain(
        llm=llm,
        chain_type="map_reduce",
        verbose=True,  # Set verbose=True if you want to see the prompts being used
    )
    summaries = summary_chain.run(docs)
    return summaries
