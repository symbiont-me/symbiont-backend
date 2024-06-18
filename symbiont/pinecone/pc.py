from hashlib import md5
from typing import List, Union, Tuple
from ..models import PineconeRecord, DocumentPage, Citation, Vectors
from ..fb.storage import download_from_firebase_storage, delete_local_file
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import NLTKTextSplitter, RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from ..models import EmbeddingModels, CohereTextModels
import os
from dotenv import load_dotenv
from langchain_core.documents import Document
from . import pc_index
from ..models import StudyResource
from firebase_admin import firestore
import nltk
from pydantic import BaseModel
from .. import logger
import cohere
import time
from fastapi import HTTPException
import datetime
from langchain_voyageai import VoyageAIEmbeddings
from ..mongodb import studies_collection

nltk.download("punkt")

load_dotenv()


# TODO import all these from __init__.py
cohere_api_key = os.getenv("CO_API_KEY")
api_key = os.getenv("OPENAI_API_KEY_FOR_EMBEDDINGS")
voyage_api_key = os.getenv("VOYAGE_API_KEY")

co = cohere.Client(api_key=cohere_api_key or "")


embeddings_model = None

if os.getenv("FASTAPI_ENV") == "development":
    embeddings_model = VoyageAIEmbeddings(voyage_api_key=voyage_api_key, model=EmbeddingModels.VOYAGEAI_2_LARGE)
    logger.info("Using Free Embeddings Model: VoyageAI")
else:
    embeddings_model = OpenAIEmbeddings(
        model=EmbeddingModels.OPENAI_TEXT_EMBEDDING_3_SMALL,
        dimensions=1536,
        api_key=api_key,
    )
    logger.info("Using OpenAI Embeddings")


class VectorInDB(BaseModel):
    source: str
    page: int
    text: str


class PineconeService:
    """
    resource_identifier is the unique identifier for the resource in the database
    """

    def __init__(
        self,
        study_id: str,
        resource_identifier: str,
        user_uid=None,
        user_query="",
        threshold=0.1,
    ):
        self.user_uid = user_uid
        self.user_query = user_query
        self.study_id = study_id
        self.resource_identifier = resource_identifier
        self.threshold = threshold
        self.db = firestore.client()
        self.embed = embeddings_model
        # TODO use model based on user's settings and api key provided
        # TODO fix missing param error
        # NOTE: has a rate limit per seconds (I think) so it throws an error if you try to use it too much
        self.nltk_text_splitter = NLTKTextSplitter()
        self.recursive_text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=14000,
            chunk_overlap=20,
            length_function=len,
            is_separator_regex=False,
        )

        self.text_splitter = self.nltk_text_splitter
        self.db_vec_refs = {}

    def get_vectors_from_db(self) -> Vectors:
        logger.info("Getting vectors from Mongo")
        study = studies_collection.find_one({"_id": self.study_id})
        return study["vectors"]

    async def create_vec_ref_in_db(self):
        if self.db_vec_refs is None:
            raise ValueError("No vectors to save in the database")
        try:
            logger.info("Updating vectors in Mongo")
            studies_collection.update_one(
                {"_id": self.study_id},
                {"$set": {f"vectors.{self.resource_identifier}": self.db_vec_refs}},
            )
        except Exception as e:
            logger.error(f"Error updating vectors in Firestore: {str(e)}")
            raise HTTPException(status_code=500, detail="Error updating vectors in Firestore")

    # make this generic it should take various types of resources
    async def embed_document(self, doc: Union[DocumentPage, Document]) -> PineconeRecord:
        vec = await self.embed.aembed_query(doc.page_content)
        hash = md5(doc.page_content.encode("utf-8")).hexdigest()
        self.db_vec_refs[hash] = VectorInDB(**doc.metadata).dict()
        return PineconeRecord(id=hash, values=vec, metadata=doc.metadata)

    async def upload_vecs_to_pinecone(self, vecs: List[PineconeRecord]):
        try:
            metadata = {}  # NOTE metadata is stored in the db. It should not be stored in Pinecone
            formatted_vecs = [(vec.id, vec.values, metadata) for vec in vecs]
            if pc_index is None:
                logger.error("Pinecone index is not initialized")
                raise ValueError("Pinecone index is not initialized")
            pc_index.upsert(vectors=formatted_vecs, namespace=self.resource_identifier)
        except Exception as e:
            logger.error(f"Error upserting vectors to Pinecone: {str(e)}")
            raise HTTPException(status_code=500, detail="Error upserting vectors to Pinecone")

    def delete_vectors_from_pinecone(self, namespace):
        try:
            if pc_index:
                pc_index.delete(delete_all=True, namespace=namespace)
                logger.info(f"Deleted namespace from Pinecone {namespace}")
        except Exception as e:
            print(f"Error deleting namespace {namespace}: {str(e)}")

    async def add_file_resource_to_pinecone(self, pages):
        s = time.time()
        docs = []
        for page in pages:
            prepared_pages = await self.prepare_pdf_for_pinecone(page)
            docs.extend(prepared_pages)
        vecs = [await self.embed_document(doc) for doc in docs]
        # handle pdf only for now
        logger.info(f"Created {len(vecs)} vectors for {self.resource_identifier}")
        await self.upload_vecs_to_pinecone(vecs)
        await self.create_vec_ref_in_db()
        elapsed = time.time() - s
        logger.info("Took (%s) s to upload the file", elapsed)

    # TODO rename this function as it is used for more than just webpages

    async def upload_webpage_to_pinecone(self, resource, content):
        split_texts = self.text_splitter.create_documents([content])
        docs = [
            # TODO fix duplicate page_content
            DocumentPage(
                page_content=split_text.page_content,
                metadata={
                    "text": split_text.page_content,
                    "source": self.resource_identifier,
                    "page": 0,  # there are no pages in a webpage or plain text
                },
                type=resource.category,
            )
            for split_text in split_texts
        ]
        s = time.time()
        vecs = [await self.embed_document(doc) for doc in docs]
        logger.info(f"Created {len(vecs)} vectors for {self.resource_identifier}")
        elapsed = time.time() - s
        logger.info("Vectorisation took (%s) s", elapsed)
        await self.upload_vecs_to_pinecone(vecs)
        # await self.create_vec_ref_in_db()

    async def get_relevant_vectors(self, top_k=25):
        if self.resource_identifier is None or self.user_query is None:
            raise ValueError("Resource and user query must be provided to get chat context")
        logger.debug("search_pinecone_index")
        pinecone_start_time = time.time()
        pc_results = self.search_pinecone_index(self.resource_identifier, top_k)
        logger.info(f"Found {len(pc_results.matches)} matches from resource {self.resource_identifier}")
        pinecone_elapsed_time = time.time() - pinecone_start_time
        logger.info(
            f"Found {len(pc_results.matches)} matches in {str(datetime.timedelta(seconds=pinecone_elapsed_time))}"
        )

        filtered_matches = [match for match in pc_results.matches if match["score"] > self.threshold]
        logger.info(f"Found {len(filtered_matches)} matches after filtering")
        if not filtered_matches:
            return filtered_matches
        logger.debug(f"matches:\t{[match['score'] for match in filtered_matches]}")

        vec_metadata_start_time = time.time()
        vec_metadata = []
        logger.debug("Getting vectors from db")
        vec_data = self.get_vectors_from_db()

        logger.debug(f"vec_data: {vec_data}")
        if vec_data is None:
            logger.error("No vectors found in the database")
            return vec_data
        for match in pc_results.matches:
            # TODO fix: is there a reason for setting the resource_vecs in the loop
            resource_vecs = vec_data[self.resource_identifier]
            vec_metadata.append(resource_vecs[match.id])

        vec_metadata_elapsed_time = time.time() - vec_metadata_start_time
        logger.debug(f"Retrieved vec data in {str(datetime.timedelta(seconds=vec_metadata_elapsed_time))}")
        return vec_metadata

    def rerank_context(self, context) -> Union[Tuple[str, List[Citation]], None]:
        # fixes: cohere.error.CohereAPIError: invalid request: list of documents must not be empty
        if not context:
            return None
        logger.debug("Reranking")
        rerank_start_time = time.time()
        reranked_context = co.rerank(
            query=self.user_query,
            documents=context,
            top_n=3,
            model=CohereTextModels.COHERE_RERANK_V2,
        )
        reranked_indices = [r.index for r in reranked_context.results]
        citations = [context[i] for i in reranked_indices]
        reranked_text = ""
        for text in reranked_context.results:
            reranked_text += text.document.get("text", "")
        rerank_elapsed_time = time.time() - rerank_start_time
        logger.info(f"Context Reranked in {str(datetime.timedelta(seconds=rerank_elapsed_time))}")
        logger.info(f"relevance scores: {[r.relevance_score for r in reranked_context]}")
        return (reranked_text, citations)

    async def get_single_chat_context(self) -> Union[Tuple[str, List[Citation]], None]:
        context = await self.get_relevant_vectors()
        if context is None:
            return None
        reranked_context = self.rerank_context(context)
        return reranked_context

    async def get_combined_chat_context(
        self,
    ) -> Union[Tuple[str, List[Citation]], None]:
        s = time.time()
        resources = studies_collection.find_one({"_id": self.study_id})["resources"]

        if resources is None:
            raise HTTPException(status_code=404, detail="No Resources Found")
        # get the identifier for each resource

        all_resource_identifiers = [resource.get("identifier") for resource in resources]

        logger.debug(f"List of Resources: {all_resource_identifiers}")

        # get the context for each resource
        combined_vecs = []
        for identifier in all_resource_identifiers:
            # NOTE need to set the global resource identifier because get_relevant_vectors uses it
            self.resource_identifier = identifier
            vecs = await self.get_relevant_vectors(top_k=10)
            combined_vecs.extend(vecs)
        logger.info(f"Combined Context: {len(combined_vecs)}")
        elapsed = time.time() - s
        logger.info("Took (%s) s to get combined context", elapsed)
        reranked_context = self.rerank_context(combined_vecs)
        return reranked_context

    async def upload_yt_resource_to_pinecone(self, resource: StudyResource, content: str):
        # NOTE this is causing problems, it seems to be cutting off the text
        # content = truncate_string_by_bytes(content, 10000)
        split_texts = self.text_splitter.create_documents([content])
        # NOTE should be able to use the Document from langchain_core.documents everywhere
        docs = [
            DocumentPage(
                page_content=split_text.page_content,
                metadata={
                    "text": split_text.page_content,
                    "source": self.resource_identifier,
                    "page": 0,  # there are no pages in a youtube video
                },
                type=resource.category,
            )
            for split_text in split_texts
        ]
        vecs = [await self.embed_document(doc) for doc in docs]
        logger.info(f"Created {len(vecs)} vectors for {self.resource_identifier}")
        await self.upload_vecs_to_pinecone(vecs)
        await self.create_vec_ref_in_db()

    def truncate_string_by_bytes(self, string, num_bytes):
        encoded_string = string.encode("utf-8")
        truncated_string = encoded_string[:num_bytes]
        return truncated_string.decode("utf-8", "ignore")

    async def prepare_pdf_for_pinecone(self, pdf_page: DocumentPage) -> List[DocumentPage]:
        page_content = pdf_page.page_content.replace("\n", "")
        page_content = self.truncate_string_by_bytes(page_content, 10000)
        split_texts = self.text_splitter.create_documents([page_content])
        docs = [
            DocumentPage(
                page_content=split_text.page_content,
                metadata={
                    "text": split_text.page_content,
                    "source": self.resource_identifier,
                    "page": pdf_page.metadata["page"],
                },
                type=pdf_page.type,
            )
            for split_text in split_texts
        ]

        return docs

    def get_query_embedding(self) -> List[float]:
        try:
            vec = self.embed.embed_query(self.user_query)
            return vec
        except Exception as e:
            logger.error(f"Error embedding query: {str(e)}")
            raise HTTPException(status_code=500, detail="Error embedding query")

    def search_pinecone_index(self, file_identifier: str, top_k=25):
        try:
            query_embedding = self.get_query_embedding()
            if pc_index is None:
                raise ValueError("Pinecone index is not initialized")
            query_matches = pc_index.query(
                vector=query_embedding,
                top_k=top_k,
                namespace=file_identifier,
                include_metadata=True,
            )

            return query_matches
        except Exception as e:
            logger.error(f"Error querying Pinecone index: {str(e)}")
            raise HTTPException(status_code=500, detail="Error querying Pinecone index")
