from langchain import OpenAI
from langchain.chains.summarize import load_summarize_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter, NLTKTextSplitter
from langchain_openai import OpenAI
from ..models import LLMModel


llm = OpenAI(temperature=0, name=LLMModel.GPT_3_5_TURBO_16K)

# sort of a dumb tokenizer but works


def truncate_prompt(prompt: str, query_size=350, model_context_window=4096) -> str:
    max_prompt_tokens = model_context_window - query_size
    tokens = prompt.split()  # simple whitespace tokenization
    if len(tokens) > max_prompt_tokens:
        tokens = tokens[:max_prompt_tokens]
    return " ".join(tokens)


# NOTE used for webpages and plain text
def summarise_plain_text_resource(document_text):
    prompt = truncate_prompt(document_text)
    summary = ""
    try:
        nltk_text_splitter = NLTKTextSplitter()

        docs = nltk_text_splitter.create_documents([prompt])

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
