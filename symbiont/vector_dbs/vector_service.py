"""
TODO: Needs to be updated to make methods async where appropriate
TODO: The whole file needs to be refactored.
TODO: Param and Return types need to be specified
TODO: Clear documentation needs to be added, with regards to purpose of the classes and the file
NOTE: The main interfaces should be maintained, because it allows for easy addition of new vector databases
"""

from qdrant_client import QdrantClient

from qdrant_client.models import Distance, VectorParams
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from langchain_community.document_loaders import TextLoader
from typing import List
from qdrant_client.http.models import Batch, PointStruct
import uuid
from langchain_openai import OpenAIEmbeddings
import numpy as np
from langchain_community.embeddings.huggingface import HuggingFaceBgeEmbeddings
import os
from abc import ABC, abstractmethod
from typing import Dict, Optional
from pydantic import BaseModel
from langchain.text_splitter import NLTKTextSplitter, RecursiveCharacterTextSplitter

from langchain_voyageai import VoyageAIEmbeddings

from ..models import EmbeddingModels, CohereTextModels, DocumentPage, Citation
from .. import logger
from ..mongodb import studies_collection
import cohere
from typing import Union, Tuple

# 1. Set up a Milvus client
# from env read name of the vector store
# from env read name of the embeddings model


cohere_api_key = os.getenv("CO_API_KEY")

co = cohere.Client(api_key=cohere_api_key or "")


class Metadata(BaseModel):
    source: str
    page: Optional[str]


class Document(BaseModel):
    page_content: str
    metadata: Metadata


class PineconeRecord(BaseModel):
    id: str
    values: List[float]
    metadata: Metadata


class PineconeResult(BaseModel):
    id: str
    score: float
    values: List


class PineconeResults(BaseModel):
    matches: List[PineconeResult]


class VectorSearchResult(BaseModel):
    id: str
    score: float


class VectorRef(BaseModel):
    source: str
    page: str
    text: str


def create_ids(n):
    return [str(uuid.uuid4()) for _ in range(n)]


class BaseVectorRepository(ABC):
    # @abstractmethod
    # def create_collection(self, collection_name: str, vector_size: int, distance: str):
    #     pass

    # TODO explain what the docs are supposed to be
    @abstractmethod
    def upsert_vectors(self, namespace: str, docs) -> List:  # docs can be either a list or None
        pass

    @abstractmethod
    # TODO return type should be List[VectorSearchResult]
    # TODO modify the method in other classes

    def search_vectors(self, namespace: str, query: str, limit: int) -> List[VectorSearchResult]:
        pass

    # removes the vectors associated with the resource
    @abstractmethod
    def delete_vectors(self, namespace: str):
        pass

    # TODO remove this and the one below
    def create_embeddings(self):
        pass

    # TODO add openai and other models
    def init_embeddings_model(self):
        model_name = "BAAI/bge-base-en"
        model_kwargs = {"device": "cpu"}
        encode_kwargs = {"normalize_embeddings": True}
        return HuggingFaceBgeEmbeddings(
            model_name=model_name,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs,
        )


# TODO try except for the env variables and remove the default values
class VectorStoreSettings:
    def __init__(self):
        self.vector_store = os.getenv("VECTOR_STORE")
        self.vector_store_url = os.getenv("VECTOR_STORE_URL")
        self.vector_store_port = os.getenv("VECTOR_STORE_PORT")
        self.vector_store_dimension = os.getenv("VECTOR_STORE_DIMENSION")
        self.vector_store_distance = os.getenv("VECTOR_STORE_DISTANCE")
        self.vector_store_token = os.getenv("VECTOR_STORE_TOKEN")  # should be optional


nltk_text_splitter = NLTKTextSplitter()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=0)

voyage_api_key = os.getenv("VOYAGE_API_KEY")

embeddings_model = VoyageAIEmbeddings(voyage_api_key=voyage_api_key, model=EmbeddingModels.VOYAGEAI_2_LARGE)


# class EmbeddingsService:
#     def init_embeddings_model(self, model_name: str):
#         if model_name == "bge-base-en":
#             model_name = "BAAI/bge-base-en"
#             model_kwargs = {"device": "cpu"}
#             encode_kwargs = {"normalize_embeddings": True}
#             return HuggingFaceBgeEmbeddings(
#                 model_name=model_name,
#                 model_kwargs=model_kwargs,
#                 encode_kwargs=encode_kwargs,
#             )
#
#     def __init__(self, model_name: str):
#         self.model = self.init_embeddings_model(model_name)
#
#     def create_docs_embeddings(self, docs):
#         docs_for_embedding = [doc.page_content for doc in docs]
#         if self.model is not None:
#             embeddings = [self.model.embed_query(text) for text in docs_for_embedding]
#             return embeddings
#         else:
#             raise ValueError("Embeddings model not initialized")
#
#
# embeddings_service = EmbeddingsService("bge-base-en")
vector_store_settings = VectorStoreSettings()


# TODO
# studies_repo = ...


class QdrantRepository(BaseVectorRepository):
    def __init__(self):
        self.dimension = vector_store_settings.vector_store_dimension
        self.distance = vector_store_settings.vector_store_distance
        self.client = QdrantClient(
            url=f"{vector_store_settings.vector_store_url}:{vector_store_settings.vector_store_port}",
            port=6333,
            api_key=vector_store_settings.vector_store_token,
        )

        print("Connected to Qdrant")

    def create_collection(self, collection_name: str, vector_size: int, distance: str):
        distance = Distance.DOT if distance.lower() == "dot" else Distance.COSINE

        is_collection = self.client.collection_exists(collection_name=collection_name)
        if is_collection is False:
            self.client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=vector_size, distance=Distance.DOT),
            )

    def __create_points_for_upsert(self, docs: List, embeddings: List):
        payload_list = [{"page_content": ""} for _ in docs]
        points = Batch(
            ids=[str(uuid.uuid4()) for _ in range(len(docs))],
            vectors=embeddings,
            payloads=payload_list,
        )
        return points

    def embed_single_document(self, doc):
        return embeddings_model.embed_query(doc.page_content)

    def upsert_vectors(self, namespace: str, docs):
        vectors = []
        for doc in docs:
            vec = self.embed_single_document(doc)
            vectors.append(vec)
        print(len(vectors))
        # vectors = embeddings_service.create_docs_embeddings(docs)
        #
        if self.client.collection_exists(collection_name=namespace) is False:
            self.create_collection(
                collection_name=namespace,
                vector_size=1536,  # TODO use vector_store_settings.vector_store_dimension
                distance=Distance.DOT,
            )
        #
        points = self.__create_points_for_upsert(docs, vectors)
        self.client.upsert(collection_name=namespace, points=points)
        # TODO check if there were no errors
        return points.ids

    # TODO create a ScoredPoint type
    def __transform_search_results(self, search_results) -> List[VectorSearchResult]:
        return [VectorSearchResult(id=result.id, score=result.score) for result in search_results]

    def search_vectors(self, namespace: str, query, limit: int) -> List:
        if embeddings_model.model is None:
            raise ValueError("Embeddings model not initialized")
        vectorised_query = embeddings_model.embed_query(query)
        results = self.client.search(collection_name=namespace, query_vector=vectorised_query, limit=limit)
        transformed_results = self.__transform_search_results(results)
        logger.info(f"Found {len(transformed_results)} results")
        return transformed_results

    def delete_vectors(self, namespace: str):
        self.client.delete_collection(collection_name=namespace)
        logger.info(f"Deleted: Vectors for {namespace}")


vector_store_repos = {
    "qdrant": QdrantRepository,
}


class VectorStoreContext:
    def __init__(self):
        self.vector_store = vector_store_settings.vector_store
        if self.vector_store not in vector_store_repos:
            raise ValueError("Vector store not supported")
        if vector_store_settings.vector_store is None:
            raise ValueError("Set the Vector Store name")
        # NOTE this instiates the vector store repo using the object
        self.vector_store_repo = vector_store_repos[vector_store_settings.vector_store]()


def create_vec_refs_in_db(ids, file_identifier, docs, user_id, study_id):
    # TODO check user access
    if len(ids) != len(docs):
        raise ValueError("The lengths of 'ids' and 'docs' must be the same")

    vec_data = {}
    for id, doc in zip(ids, docs):
        vec_data[id] = {
            "source": file_identifier,
            "page": doc.metadata.get("page"),
            "text": doc.page_content,
        }

    logger.debug(f"Creating Vectors in DB: {vec_data}")
    studies_collection.update_one(
        {"_id": study_id},
        {"$set": {f"vectors.{file_identifier}": vec_data}},
    )


# TODO this should be part of a db repo or service
def get_vec_refs_from_db(study_id, file_identifier, ids) -> List:
    logger.info(f"Fetching Vectors from {study_id}")

    logger.info("Fetching Vectors from DB")
    results = []
    study = studies_collection.find_one({"_id": study_id})
    if study is None:
        raise ValueError("Study not found")
    study_vectors = study.get("vectors", {})
    file_vectors = study_vectors.get(file_identifier, {})

    for id in ids:
        vec = file_vectors.get(id, {})
        logger.debug(f"Found vector: {vec}")
        results.append(vec)
    logger.debug(f"Found {len(results)} vectors")
    # TODO rename: this is the vec data from the db
    return results


# 1. langchain parses a resource and creates a document with a sinlge element called Document
# 2. we have to split the document into smaller chunks
# 3. we have to create embeddings for each chunk
# 4. we have to use the standard return object
class ChatContextService(VectorStoreContext):
    def __init__(
        self,
        resource_doc=None,
        resource_identifier="",  # TODO make this a required param for initialisation
        resource_type=None,
        user_id: str = "",
        user_query: str = "",
        study_id: str = "",  # TODO this should be required
    ):
        super().__init__()
        self.user_id = user_id
        self.resource_identifier = resource_identifier
        self.user_query = user_query
        self.resource_doc = resource_doc
        self.resource_type = resource_type
        self.study_id = study_id

    def add_pdf_resource(self):
        pass

    def add_web_resource(self):
        content = getattr(self.resource_doc, "page_content", None)
        if content is None:
            raise ValueError("There is no resource content to be added")
        split_texts = text_splitter.create_documents([content])
        docs = [
            DocumentPage(
                page_content=split_text.page_content,
                metadata={
                    "text": split_text.page_content,
                    "source": self.resource_identifier,
                    "page": 0,
                },
            )
            for split_text in split_texts
        ]
        ids = self.vector_store_repo.upsert_vectors(self.resource_identifier, docs)
        create_vec_refs_in_db(ids, self.resource_identifier, docs, self.user_id, self.study_id)

    def add_yt_resource(self):
        content = getattr(self.resource_doc, "page_content", None)
        if content is None:
            raise ValueError("There is no resource content to be added")
        split_texts = text_splitter.create_documents([content])
        docs = [
            DocumentPage(
                page_content=split_text.page_content,
                metadata={
                    "text": split_text.page_content,
                    "source": self.resource_identifier,
                    "page": 0,
                },
            )
            for split_text in split_texts
        ]
        ids = self.vector_store_repo.upsert_vectors(self.resource_identifier, docs)
        logger.debug(ids)
        create_vec_refs_in_db(ids, self.resource_identifier, docs, self.user_id, self.study_id)

    # TODO move this some place appropriate
    resource_adders = {
        "pdf": add_pdf_resource,
        "webpage": add_web_resource,
        "youtube": add_yt_resource,
    }

    # TODO this should single Document
    def add_resource(self):
        if self.resource_type is None:
            raise ValueError("Resource document not provided")
        if self.resource_type not in self.resource_adders:
            raise ValueError("Resource type not supported")
        self.resource_adders[self.resource_type](self)

    # TODO Remove the context from db from here
    def delete_context(self):
        self.vector_store_repo.delete_vectors(self.resource_identifier)

    # TODO add the type for context
    def rerank_context(self, context, query) -> Union[Tuple[str, List[Citation]], None]:
        # fixes: cohere.error.CohereAPIError: invalid request: list of documents must not be empty
        if not context:
            return None
        logger.debug("Reranking")
        reranked_context = co.rerank(
            query=query,
            documents=context,
            top_n=3,
            model=CohereTextModels.COHERE_RERANK_V2,
        )
        reranked_indices = [r.index for r in reranked_context.results]
        citations = [context[i] for i in reranked_indices]
        reranked_text = ""
        for text in reranked_context.results:
            reranked_text += text.document.get("text", "")
        return (reranked_text, citations)

    # TODO document this
    def get_single_chat_context(self, query: str) -> Union[Tuple[str, List[Citation]], None]:
        results = self.vector_store_repo.search_vectors(namespace=self.resource_identifier, query=query, limit=10)
        ids = [result.id for result in results]
        logger.debug(f"Found {len(ids)} results")
        vectors_metadata_from_db = get_vec_refs_from_db(self.study_id, self.resource_identifier, ids)
        logger.debug(f"Found {len(vectors_metadata_from_db)} vectors from db")
        logger.debug("Reranking")
        reranked_context = self.rerank_context(vectors_metadata_from_db, query)
        return reranked_context


# TODO remove as not being used
def create_docs(path: str):
    loader = TextLoader(path)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    docs = text_splitter.split_documents(documents)
    return docs
