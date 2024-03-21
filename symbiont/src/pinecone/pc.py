from hashlib import md5
from typing import List
from symbiont.src.models import PineconeRecord, DocumentPage
from symbiont.src.fb.storage import download_from_firebase_storage, delete_local_file
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import NLTKTextSplitter, RecursiveCharacterTextSplitter
from hashlib import md5

from langchain_openai import OpenAIEmbeddings
from symbiont.src.models import EmbeddingModels
import os
from dotenv import load_dotenv
from langchain_core.documents import Document
from typing import List, Union
from langchain_community.embeddings import CohereEmbeddings
from . import pc_index
from ..models import EmbeddingModels, StudyResource
from firebase_admin import firestore
import nltk
from pydantic import BaseModel
from ..models import Study
from typing import Union
from .. import logger

nltk.download("punkt")

load_dotenv()

# TODO import all these from __init__.py
cohere_api_key = os.getenv("COHERE_API_KEY")
api_key = os.getenv("OPENAI_API_KEY")
pinecone_api_key = os.getenv("PINECONE_API_KEY")
pinecone_index = os.getenv("PINECONE_INDEX")
pinecone_endpoint = os.getenv("PINECONE_API_ENDPOINT")


class VectorInDB(BaseModel):
    source: str
    page: int
    text: str


# TODO INITIALIZE THE EMBEDDINGS MODEL WITH USER'S API KEY FROM THE DB
class PineconeService:
    def __init__(
        self, user_uid=None, user_query=None, resource_identifier=None, study_id=None
    ):

        self.user_uid = user_uid
        self.user_query = user_query
        self.resource = resource_identifier
        self.study_id = study_id
        self.db = firestore.client()
        self.embed = OpenAIEmbeddings(
            model=EmbeddingModels.TEXT_EMBEDDING_3_SMALL, dimensions=1536
        )

    def get_vectors_from_db(self):
        vec_ref = self.db.collection("users").document(self.user_uid)
        vec_data = vec_ref.get().to_dict()
        if "vectors" not in vec_data:
            return None
        return vec_data["vectors"]

    def create_vec_ref_in_db(self, md5_hash, metadata):
        db = firestore.client()
        vec_ref = db.collection("users").document(self.user_uid)
        # Retrieve the current data to avoid overwriting
        current_data = vec_ref.get().to_dict()
        # Initialize 'vectors' as a mapping if it doesn't exist
        if "vectors" not in current_data:
            current_data["vectors"] = {}
        source = metadata["source"]  # @dev this is the same as the file_identifier
        if source not in current_data["vectors"]:
            current_data["vectors"][metadata["source"]] = {}
        current_data["vectors"][source][md5_hash] = VectorInDB(**metadata).dict()
        # Update the document with the new mapping of vectors
        vec_ref.set(current_data)
        return vec_ref

    # make this generic it should take various types of resources
    async def embed_document(
        self, doc: Union[DocumentPage, Document]
    ) -> PineconeRecord:
        vec = await self.embed.aembed_query(doc.page_content)
        hash = md5(doc.page_content.encode("utf-8")).hexdigest()
        self.create_vec_ref_in_db(hash, doc.metadata)
        return PineconeRecord(id=hash, values=vec, metadata=doc.metadata)

    def delete_vectors_from_pinecone(self, namespace):
        try:
            if pc_index:
                pc_index.delete(delete_all=True, namespace=namespace)
                print(f"Deleted namespace {namespace}")
        except Exception as e:
            print(f"Error deleting namespace {namespace}: {str(e)}")

    async def handle_pdf_resource(self, file_path: str):
        if file_path is not None and file_path.endswith(".pdf"):
            loader = PyPDFLoader(file_path)
            pages = [DocumentPage(**page.dict()) for page in loader.load_and_split()]

            docs = []
            for page in pages:
                prepared_pages = await self.prepare_pdf_for_pinecone(page)
                docs.extend(prepared_pages)
            vecs = [await self.embed_document(doc) for doc in docs]
            return vecs
        return []

    async def prepare_resource_for_pinecone(
        self, file_identifier: str, download_url: str
    ):
        file_path = await download_from_firebase_storage(file_identifier, download_url)
        # handle pdf only for now
        if file_path is not None and file_path.endswith(".pdf"):
            vecs = await self.handle_pdf_resource(file_path)

            await self.upload_vecs_to_pinecone(vecs, file_identifier)
            await delete_local_file(file_path)

    # TODO rename this function as it is used for more than just webpages
    async def upload_webpage_to_pinecone(self, resource, content):
        text_splitter = NLTKTextSplitter()
        # text_splitter = RecursiveCharacterTextSplitter(
        #     # Set a really small chunk size, just to show.
        #     chunk_size=1500,
        #     chunk_overlap=20,
        #     length_function=len,
        #     is_separator_regex=False,
        # )
        split_texts = text_splitter.create_documents([content])
        docs = [
            DocumentPage(
                page_content=split_text.page_content,
                metadata={
                    "text": split_text.page_content,
                    "source": resource.identifier,
                    "page": 0,  # there are no pages in a webpage or plain text
                },
                type=resource.category,
            )
            for split_text in split_texts
        ]

        vecs = [await self.embed_document(doc) for doc in docs]
        await self.upload_vecs_to_pinecone(vecs, resource.identifier)

    def get_chat_context(self, top_k=10):
        if self.resource is None or self.user_query is None:
            raise ValueError(
                "Resource and user query must be provided to get chat context"
            )
        context = ""
        pc_results = self.search_pinecone_index(self.user_query, self.resource, top_k)

        vec_metadata = []
        for match in pc_results.matches:
            vec_data = self.get_vectors_from_db()
            if vec_data is None:
                return ""
            resource_vecs = vec_data[self.resource]
            vec_metadata.append(resource_vecs[match.id])
            context += resource_vecs[match.id]["text"]
        print("CONTEXT", context)
        return context

    async def upload_yt_resource_to_pinecone(self, resource, content):
        # NOTE this is causing problems, it seems to be cutting off the text
        # content = truncate_string_by_bytes(content, 10000)
        text_splitter = NLTKTextSplitter()
        split_texts = text_splitter.create_documents([content])
        # NOTE I should be able to use the Document from langchain_core.documents everywhere
        docs = [
            DocumentPage(
                page_content=split_text.page_content,
                metadata={
                    "text": split_text.page_content,
                    "source": resource.identifier,
                    "page": 0,  # there are no pages in a youtube video
                },
                type=resource.category,
            )
            for split_text in split_texts
        ]
        vecs = [await self.embed_document(doc) for doc in docs]
        await self.upload_vecs_to_pinecone(vecs, resource.identifier)

    async def upload_vecs_to_pinecone(
        self, vecs: List[PineconeRecord], file_identifier: str
    ):
        # TODO don't initialize Pinecone client here
        metadata = (
            {}
        )  # TODO metadata is stored in the db. It should not be stored in Pinecone. Should have a better way to do this
        formatted_vecs = [(vec.id, vec.values, metadata) for vec in vecs]
        pc_index.upsert(vectors=formatted_vecs, namespace=file_identifier)
        print("Uploaded to Pinecone")

    def truncate_string_by_bytes(self, string, num_bytes):
        encoded_string = string.encode("utf-8")
        truncated_string = encoded_string[:num_bytes]
        return truncated_string.decode("utf-8", "ignore")

    async def prepare_pdf_for_pinecone(
        self, pdf_page: DocumentPage
    ) -> List[DocumentPage]:
        page_content = pdf_page.page_content.replace("\n", "")
        page_content = self.truncate_string_by_bytes(page_content, 10000)
        text_splitter = NLTKTextSplitter()
        split_texts = text_splitter.create_documents([page_content])
        docs = [
            DocumentPage(
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

    def get_query_embedding(self, query: str) -> List[float]:
        vec = self.embed.embed_query(query)
        return vec

    def search_pinecone_index(self, query: str, file_identifier: str, top_k=10):

        query_embedding = self.get_query_embedding(query)
        query_matches = pc_index.query(
            vector=query_embedding,
            top_k=top_k,
            namespace=file_identifier,
            include_metadata=True,
        )

        return query_matches
