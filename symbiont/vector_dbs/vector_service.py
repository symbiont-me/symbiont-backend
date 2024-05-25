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

from ..models import EmbeddingModels, CohereTextModels, DocumentPage
from .. import logger

# 1. Set up a Milvus client
# from env read name of the vector store
# from env read name of the embeddings model


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

voyage_api_key = os.getenv("VOYAGE_API_KEY")


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
        # return points.ids
        return []

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


mock_db = {}


def create_vec_refs_in_db(ids, file_identifier, docs, user_id):
    if len(ids) != len(docs):
        raise ValueError("The lengths of 'ids' and 'docs' must be the same")

    vec_data = {file_identifier: {}}
    for id, doc in zip(ids, docs):
        vec_data[file_identifier][id] = VectorRef(source=file_identifier, page="", text=doc.page_content)

    mock_db.update(vec_data)
    return mock_db


def get_vec_refs_from_db(file_identifier, ids):
    print(f"found {len(ids)}")
    results = [mock_db[file_identifier][id] for id in ids]

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
    ):
        super().__init__()
        self.user_id = user_id
        self.resource_identifier = resource_identifier
        self.user_query = user_query
        self.resource_doc = resource_doc
        self.resource_type = resource_type

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
        self.vector_store_repo.upsert_vectors(self.resource_identifier, docs)

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
        self.vector_store_repo.upsert_vectors(self.resource_identifier, docs)

    # TODO move this some place appropriate
    resource_adders = {
        "pdf": add_pdf_resource,
        "webpage": add_web_resource,
        "yt": add_yt_resource,
    }

    # TODO this should single Document
    def add_resource(self):
        if self.resource_type is None:
            raise ValueError("Resource document not provided")
        if self.resource_type not in self.resource_adders:
            raise ValueError("Resource type not supported")
        self.resource_adders[self.resource_type](self)

    def delete_context(self):
        self.vector_store_repo.delete_vectors(self.resource_identifier)

        # create_vec_refs_in_db(ids, file_identifier, docs, self.user_id)

    def get_chat_context(self, query: str):
        results = self.vector_store_repo.search_vectors(namespace=self.resource_identifier, query=query, limit=10)
        ids = [result.id for result in results]
        logger.debug(f"Found {len(ids)} results")
        logger.debug(ids)
        # TODO use ids to retrieve the data from the db
        # get_vec_refs_from_db("chat_context", ids)
        # chat_context = self.rerank_results(results)
        return results


def create_docs(path: str):
    loader = TextLoader(path)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    docs = text_splitter.split_documents(documents)
    return docs
