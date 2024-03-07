from langchain import OpenAI
from langchain.chains.summarize import load_summarize_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter, NLTKTextSplitter
from langchain_openai import OpenAI
from ..models import LLMModel


llm = OpenAI(temperature=0, name=LLMModel.GPT_3_5_TURBO_16K)


# TODO fix This model's maximum context length is 4097 tokens, however you requested 11438 tokens
# NOTE used for webpages and plain text
def summarise_plain_text_resource(document_text):
    summary = ""
    try:
        nltk_text_splitter = NLTKTextSplitter()

        docs = nltk_text_splitter.create_documents([document_text])

        summary_chain = load_summarize_chain(
            llm=llm,
            chain_type="map_reduce",
            verbose=False,  # Set verbose=True if you want to see the prompts being used
        )
        summary = summary_chain.run(docs)
        return summary
    except Exception as e:
        # Handle the exception here
        print(f"An error occurred: {e}")
        summary = f"An error occurred: Summary could not be generated."
    return summary
