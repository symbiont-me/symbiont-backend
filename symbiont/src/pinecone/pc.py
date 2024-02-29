from hashlib import md5
from typing import List
from symbiont.src.models import PineconeRecord, PdfPage
from symbiont.src.fb.storage import download_from_firebase_storage, delete_local_file
from pinecone import Pinecone
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import NLTKTextSplitter
from hashlib import md5
import time
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAI, OpenAIEmbeddings
from symbiont.src.models import EmbeddingModels
import os
from dotenv import load_dotenv


# ~~~~~~~~~~~~~~~~~~~~
#       PINECONE
# ~~~~~~~~~~~~~~~~~~~~

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
pinecone_api_key = os.getenv("PINECONE_API_KEY")
pinecone_index = os.getenv("PINECONE_INDEX")
pinecone_endpoint = os.getenv("PINECONE_API_ENDPOINT")

# Initialize the OpenAIEmbeddings object, will use the same object for embedding tasks
embed = OpenAIEmbeddings(model=EmbeddingModels.TEXT_EMBEDDING_3_SMALL, dimensions=1536)


async def embed_document(doc: PdfPage) -> PineconeRecord:

    vec = await embed.aembed_query(doc.page_content)
    hash = md5(doc.page_content.encode("utf-8")).hexdigest()
    return PineconeRecord(id=hash, values=vec, metadata=doc.metadata)


async def handle_pdf_resource(file_path: str):
    if file_path is not None and file_path.endswith(".pdf"):
        loader = PyPDFLoader(file_path)
        pages = [PdfPage(**page.dict()) for page in loader.load_and_split()]

        docs = []
        for page in pages:
            prepared_pages = await prepare_pdf_for_pinecone(page)
            docs.extend(prepared_pages)
        vecs = [await embed_document(doc) for doc in docs]
        return vecs
    return []


async def prepare_resource_for_pinecone(file_identifier: str, download_url: str):
    file_path = await download_from_firebase_storage(file_identifier, download_url)
    # handle pdf only for now
    if file_path is not None and file_path.endswith(".pdf"):
        vecs = await handle_pdf_resource(file_path)
        await upload_vecs_to_pinecone(vecs, file_identifier)
        await delete_local_file(file_path)


async def upload_vecs_to_pinecone(vecs: List[PineconeRecord], file_identifier: str):
    # TODO don't initialize Pinecone client here
    client = Pinecone(api_key=pinecone_api_key, endpoint=pinecone_endpoint)
    index = client.Index("symbiont-me")
    formatted_vecs = [(vec.id, vec.values, vec.metadata) for vec in vecs]
    if index is None:
        raise Exception("Pinecone index not found")
    index.upsert(vectors=formatted_vecs, namespace=file_identifier)
    print("Uploaded to Pinecone")


def truncate_string_by_bytes(string, num_bytes):
    encoded_string = string.encode("utf-8")
    truncated_string = encoded_string[:num_bytes]
    return truncated_string.decode("utf-8", "ignore")


async def prepare_pdf_for_pinecone(pdf_page: PdfPage) -> List[PdfPage]:
    page_content = pdf_page.page_content.replace("\n", "")
    page_content = truncate_string_by_bytes(page_content, 10000)
    # TODO use NLTK Splitter with db reference and don't store text in pinecone
    # Pincecone is for embeddings only, it is expensive to store text in pinecone
    # text_splitter = NLTKTextSplitter()
    text_splitter = RecursiveCharacterTextSplitter(
        # Set a really small chunk size, just to show.
        chunk_size=1000,
        chunk_overlap=20,
        length_function=len,
        is_separator_regex=False,
    )

    split_texts = text_splitter.create_documents([page_content])
    docs = [
        PdfPage(
            page_content=split_text.page_content,
            metadata={
                "text": split_text.page_content,
                "source": pdf_page.metadata["source"],
                "page": pdf_page.metadata["page"],
            },
            type=pdf_page.type,
        )
        for split_text in split_texts
    ]
    return docs


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#       Vector Search
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


def get_query_embedding(query: str) -> List[float]:
    vec = embed.embed_query(query)
    return vec


def search_pinecone_index(query: str, file_identifier: str):
    # TODO don't initialize Pinecone client here
    client = Pinecone(api_key=pinecone_api_key, endpoint=pinecone_endpoint)
    index = client.Index("symbiont-me")

    if index is None:
        raise Exception("Pinecone index not found")

    query_embedding = get_query_embedding(query)
    query_matches = index.query(
        vector=query_embedding,
        top_k=2,
        namespace=file_identifier,
        include_metadata=True,
    )
    return query_matches


def get_chat_context(query: str, file_identifier: str):
    result = search_pinecone_index(query, file_identifier)
    context = ""
    for match in result.matches:
        context += match.metadata["text"] + " "
    # TODO return an object with matches for detailed footnoting
    return context
